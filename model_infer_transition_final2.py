#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轉換推論：候選規則的洩漏偵測驗證，與正式驗收

用「已訓練完成的模型」讀轉換特徵 CSV（正常 → 洩漏，含 onset 標記），
量干擾推論提出的候選規則抓不抓得到洩漏、多久抓到。不重新訓練模型、不重做特徵工程。

在完整流程中的位置：
  1. 干擾推論  mode="select"            ：干擾 dev 上提出 16 個候選
  2. 轉換推論  run_transition_postprocess_candidates  ← 本程式（只用轉換 dev）
       16 個候選的偵測率、太早報、延遲，和干擾誤報放同一張表
       → transition_postprocess_candidate_comparison.csv，你看這張表選定一條
  3. 干擾推論  mode="evaluate"          ：干擾 holdout 上量選定的那條，寫出 postprocess_config.json
                                           （validation_status = pending_transition_holdout）
  4. 轉換推論  run_transition_feature_inference        ← 本程式（只用轉換 holdout）
       只量正式版那一條，成績附加到 postprocess_config.json（validation_status → final_verified）
     接著 save_final_runtime_feature_schema → runtime_feature_schema.json

輸入：
  - metadata.json（模型資料夾裡）
  - transition_rule_features_eval.csv（切分程式輸出，有 split 欄：dev / holdout）
  - postprocess_config_candidate_top*.json（第 2 步）／postprocess_config.json（第 4 步）
  - feature_config_micro.json（可選，只用來核對特徵清單和模型一致）

評估定義：
  - 報警時間 = 窗結束時間（這一窗收完才算得出結果）
  - onset_tolerance_sec（預設 1 秒）：onset 前這麼多秒內的報警不算誤報（手動標記誤差）
  - 沒有 onset 的 record：只要報警就算誤報
  - label 為空的窗（灰色地帶：標記前後、轉閥暫態）不列入 window 指標

重要原則：
  1. 模型特徵、門檻以模型存檔為準；規則以 postprocess config 為準，缺參數就報錯，不補預設值
  2. 告警一律由 alarm_rule_utils.apply_alarm_method_by_record 計算（和干擾推論、runtime 同一套）
  3. 選規則只用 dev；holdout 只量已定案的那一條，不排名、不換規則
  4. runtime_feature_schema.json 只有在干擾、轉換 holdout 都評估過（final_verified）才能產生
"""

from pathlib import Path
import sys
import numpy as np
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
import matplotlib.pyplot as plt

# 中文字型：依序找系統裡有的字型（Mac：PingFang / Heiti；Windows：微軟正黑體；Linux：Noto），都沒有才退回預設
plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS",
                                   "Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False    # 換成中文字型後，負號才不會變成方塊


# 設定專案路徑
CURRENT_DIR = Path(__file__).resolve().parent    # 這支 .py 所在的資料夾
PROJECT_ROOT = CURRENT_DIR                       # 目前直接等於這個資料夾

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 匯入你自己寫的模型管理器
from model_manager_two_sensor_final import ModelManager

# 共用告警規則函式
from alarm_rule_utils_v2 import (
    apply_alarm_method_by_record,
    rule_kwargs_from_cfg,     # 規則參數檢查（和干擾推論、runtime 同一套，只維護一份）
    _cfg_value,               # 取設定值：None、NaN 都當作「沒寫」
)

# 和干擾推論共用：載入模型、拆推論設定、檢查欄位、正式 runtime schema
# （只維護一份，避免兩支程式的檢查規則不一致）
from model_infer_disturbance_final3 import (
    resolve_model_location_from_metadata,
    get_xgb_model,
    extract_inference_config,
    check_feature_alignment,
    save_final_runtime_feature_schema,
    split_dev_holdout_for_rules,      # 依 split 欄分 dev / holdout（沒有 split 欄、session 跨兩邊都會報錯）
    estimate_window_timing,           # 從資料估計窗長、步長
    _json_safe,                       # numpy / NaN → JSON 可寫的型別（NaN 存成 null）
)


# ============================================================
# 讀取 feature config / postprocess config
# ============================================================
def load_feature_config(feature_config_json):
    '''
    feature_config_micro.json（Step2 輸出）。這裡只用來做一致性檢查：
    selected_feature_names 必須和模型 metadata 的特徵完全一樣。
    模型門檻以模型存檔為準；告警規則以 postprocess config 為準，都不從這裡讀。
    有給這份檔案卻沒有 selected_feature_names（例如傳錯檔），直接報錯，不會悄悄跳過核對。
    '''
    payload = json.loads(Path(feature_config_json).read_text(encoding="utf-8"))
    if not payload.get("selected_feature_names"):
        raise ValueError(f"{feature_config_json} 沒有 selected_feature_names，"
                         "請確認傳的是 Step2 輸出的 feature_config（不是 band_config 等其他設定檔）")
    return payload


# 讀候選檔或正式版的 postprocess config，至少要有 baseline_rule_settings，沒有就報錯
def load_postprocess_config(postprocess_config_path):
    payload = json.loads(Path(postprocess_config_path).read_text(encoding="utf-8"))
    if not payload.get("baseline_rule_settings"):
        raise ValueError(f"{postprocess_config_path} 缺少 baseline_rule_settings")
    return payload


# ============================================================
# 把候選 / 正式設定檔裡的規則，轉成告警引擎吃的 cfg
# ============================================================
# 告警引擎會用到的鍵。
# 設定檔（baseline_rule_settings）裡其他的鍵——effective_fa、veto_rate、事件命中率等——只是成績紀錄，不傳給引擎
ALARM_RULE_KEYS = (
    "name", "method",
    "alarm_threshold", "smooth_window", "consecutive", "n_hits", "m_window",   # 各方法的參數
    "high_threshold", "low_threshold",                                         # 遲滯（hysteresis）的雙門檻
    "min_leak_duration_sec",                                                   # 警報要持續幾秒才報（沒寫 = 不限制）
    "use_veto", "veto_column", "veto_threshold", "veto_rules",                 # veto 設定
)


def rule_cfg_to_alarm_cfg(rule_cfg):
    '''
    baseline_rule_settings（干擾推論寫的）→ alarm_rule_utils.apply_alarm_method_by_record 的 cfg。

    這裡只做兩件事：
      1. 挑出引擎用得到的鍵，並去掉空值（None、NaN 都當作「沒寫」，JSON / CSV 讀回來的空值常常是 NaN）
      2. 呼叫 alarm_rule_utils.rule_kwargs_from_cfg 檢查參數
    參數規則只維護在 alarm_rule_utils 一個地方（干擾推論、轉換推論、runtime 共用）：
    支援哪些 method、每種 method 需要哪些參數、數值範圍，都以那裡為準；
    缺參數、two_stage、門檻不在 0～1、n_hits > m_window… 在這裡就會報錯，不補任何預設值。
    '''
    # 1. 挑鍵、去空值（_cfg_value 會把 None 和 NaN 都回傳 None）
    cfg = {k: rule_cfg[k] for k in ALARM_RULE_KEYS
           if k in rule_cfg and _cfg_value(rule_cfg, k) is not None}
    cfg["name"] = cfg.get("name") or "unnamed_rule"       # 只是讓錯誤訊息有名字可以顯示，不影響規則
    cfg["use_veto"] = bool(cfg.get("use_veto", False))    # 沒寫 use_veto = 不啟用 veto
    cfg["veto_rules"] = cfg.get("veto_rules") or []       # 統一成 list，後面不用再判斷 None

    # 2. 規則參數檢查：交給引擎（不合格直接報錯，錯誤訊息帶規則名稱）
    rule_kwargs_from_cfg(cfg)

    # 3. veto 設定要完整：要嘛有 veto_rules，要嘛有 veto_column + veto_threshold（引擎要等到真的套用規則、建 veto 遮罩時才會發現，這裡提早檢查，錯誤訊息比較好懂）
    if cfg["use_veto"] and not cfg["veto_rules"] and not (cfg.get("veto_column") and "veto_threshold" in cfg):
        raise ValueError(f"規則 {cfg['name']} 啟用 veto，但沒有 veto_rules，也沒有完整的 veto_column + veto_threshold")
    return cfg


def required_rule_columns(alarm_cfg):
    '''這條規則需要資料裡有哪些欄位（veto 用）'''
    cols = []
    if alarm_cfg.get("use_veto"):
        if alarm_cfg.get("veto_column"):
            cols.append(alarm_cfg["veto_column"])
        cols += [r["col"] for r in alarm_cfg.get("veto_rules", []) if r.get("col")]
    return sorted(set(cols))


# ============================================================
# 載入模型、推論 pred_proba（候選比較與正式評估共用）
# ============================================================
def load_model_and_predict(feature_csv, metadata_json, feature_config_json=None):
    '''
    1. 依 metadata.json 載入「指定的那顆」模型
    2. 有給 feature_config_json 時，檢查它的 selected_feature_names 和模型特徵完全一致（不一致就停）
    3. 讀特徵 CSV、檢查欄位（缺欄、NaN/inf 都會停），推論 pred_proba / pred_label
    回傳 dict：df_pred, model_id, model_folder, metadata, feature_names, window_threshold, file_threshold
    '''
    feature_csv, metadata_json = Path(feature_csv), Path(metadata_json)
    models_dir, model_id, model_folder = resolve_model_location_from_metadata(metadata_json)
    manager = ModelManager(models_dir=str(models_dir))
    model_id, pkg = get_xgb_model(manager, model_id=model_id, metadata_json=metadata_json)
    model, scaler, feature_names, window_threshold, file_threshold, metadata, model_type = extract_inference_config(pkg)

    if feature_config_json is not None:
        cfg_names = load_feature_config(feature_config_json)["selected_feature_names"]   # 沒有這個鍵在 load_feature_config 就會報錯
        if list(cfg_names) != list(feature_names):
            only_model = sorted(set(feature_names) - set(cfg_names))[:10]
            only_cfg = sorted(set(cfg_names) - set(feature_names))[:10]
            raise ValueError(
                "feature_config 與模型 metadata 的特徵不一致（請確認是同一次 Step2 的輸出）\n"
                f"  metadata 特徵數 = {len(feature_names)}，feature_config 特徵數 = {len(cfg_names)}\n"
                f"  只在模型裡：{only_model}\n"
                f"  只在 feature_config 裡：{only_cfg}\n"
                f"  兩邊都是空的，代表名單相同、只差在順序"
            )

    df = pd.read_csv(feature_csv)
    check_feature_alignment(df, feature_names)
    X = df[feature_names].copy()
    if scaler is not None:
        X = scaler.transform(X)
    df_pred = df.copy()
    df_pred["pred_proba"] = model.predict_proba(X)[:, 1]
    df_pred["pred_label"] = (df_pred["pred_proba"] >= window_threshold).astype(int)

    print(f"使用模型: {model_id}（window_threshold = {window_threshold}）")
    return {
        "df_pred": df_pred, "model_id": model_id, "model_folder": model_folder, "metadata": metadata,
        "feature_names": list(feature_names), "window_threshold": float(window_threshold),
        "file_threshold": file_threshold,
    }


#                                    ********************************  window-level 評估  ********************************
# ============================================================
# window-level 評估:評估每一個 window 預測得好不好
# ============================================================
def evaluate_window_level(
        df_pred,
        # pred_col="pred_label"：模型原始門檻的判定（還沒套規則）
        # pred_col="detected_rule"：套了正式規則之後的判定
        pred_col="pred_label",
        result_name="Window-level",
    ):
    """
    評估指定判定欄位的 Window-level 表現。

    pred_col="pred_label"：
        模型原始門檻判定。

    pred_col="detected_rule"：
        套用正式後處理規則後的判定。
    """
    if pred_col not in df_pred.columns:
        raise ValueError(f"找不到評估欄位：{pred_col}")
        
    if "label" not in df_pred.columns:
        raise ValueError(f"{result_name}：資料沒有 label 欄，無法做 window-level 評估")

    # 灰色地帶（標記前後、轉閥暫態）的 label 是空值：不給標籤，也不拿來評估
    label_num = pd.to_numeric(df_pred["label"], errors="coerce")

    # label 只能是 0、1 或空值；其他值（例如 0.5、2）不能用 astype(int) 硬轉，0.5 會被悄悄截成 0
    bad = label_num.notna() & ~label_num.isin([0, 1])
    if bad.any():
        raise ValueError(f"{result_name}：label 只能是 0、1 或空值（灰色地帶），"
                         f"出現了 {sorted(label_num[bad].unique().tolist())[:5]}")

    labeled = df_pred[label_num.notna()]
    if labeled.empty:
        raise ValueError(f"{result_name}：所有窗都是灰色地帶（label 全是空值），沒有可評估的窗")
    
    n_gray = len(df_pred) - len(labeled)
    if n_gray:
        print(f"{result_name}：{n_gray} 窗沒有標籤（灰色地帶），不列入評估")
    y_true = label_num[label_num.notna()].astype(int).values     # 取出真實標籤 label
    y_pred = labeled[pred_col].astype(int).values               # 取出模型預測標籤

    # 算各種指標：accuracy, precision, recall, f1, confusion matrix
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n" + "=" * 80)
    print(f"{result_name} 評估結果")
    print("=" * 80)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["正常", "洩漏"],
        zero_division=0,
    ))

    return {
        "prediction_column": pred_col,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ============================================================
# 模型本身分不分得開「穩定正常」和「穩定洩漏」（不套任何告警規則、不看門檻）
# ============================================================
def evaluate_steady_separability(
        df_pred,                         # 有 pred_proba、t_start（最好有 t_ref）、onset_gt_sec、label 的資料
        margins=(0.5, 1.0, 2.0, 3.0),    # 排除 onset 前後多少秒的過渡區
        window_sec=0.2,                  # 窗長（沒有 t_ref 時用來算窗中心）
        verbose=False):                  # True 才印出每個 margin 的詳細報告
    '''
    診斷用：偵測率不好的時候，先看是「模型分不開」還是「規則太嚴」。
      - ROC-AUC 高、偵測率低 → 模型分得開，是告警規則（門檻、持續時間、veto）擋掉了
      - ROC-AUC 低            → 模型本身分不開，調規則沒用，要回頭看特徵或訓練資料
    ROC-AUC / PR-AUC 不需要門檻，看的是機率排序：洩漏窗的機率普遍比正常窗高，就接近 1。

    每個 margin 把窗分成兩類（離 onset 太近的過渡區排除）：
      0 = 穩定正常：onset 前至少 margin 秒；沒有 onset 的 record（整條正常）全部都算
      1 = 穩定洩漏：onset 後至少 margin 秒
    label 為空的灰色地帶（轉閥暫態等）一律排除，和 window-level 評估一致。
    '''
    df = df_pred.copy()
    # 窗中心：有 t_ref 就用（特徵工程定義的窗中心），沒有才用 t_start + 窗長 / 2
    df["t_center"] = df["t_ref"] if "t_ref" in df.columns else df["t_start"] + window_sec / 2.0
    onset = pd.to_numeric(df["onset_gt_sec"], errors="coerce") if "onset_gt_sec" in df.columns \
        else pd.Series(np.nan, index=df.index)
    has_onset = onset.notna()

    # 灰色地帶排除
    if "label" in df.columns:
        keep = pd.to_numeric(df["label"], errors="coerce").notna()
        df, onset, has_onset = df[keep], onset[keep], has_onset[keep]

    rows = []
    for margin in margins:
        normal_mask = (~has_onset) | (df["t_center"] <= onset - margin)    # 沒有 onset 的整條都是正常
        leak_mask = has_onset & (df["t_center"] >= onset + margin)
        y_true = np.r_[np.zeros(int(normal_mask.sum()), dtype=int), np.ones(int(leak_mask.sum()), dtype=int)]
        y_prob = np.r_[df.loc[normal_mask, "pred_proba"].to_numpy(float), df.loc[leak_mask, "pred_proba"].to_numpy(float)]

        both = len(np.unique(y_true)) == 2
        row = {
            "margin_sec": margin,
            "n_normal_windows": int(normal_mask.sum()),
            "n_leak_windows": int(leak_mask.sum()),
            "roc_auc": roc_auc_score(y_true, y_prob) if both else np.nan,
            "pr_auc": average_precision_score(y_true, y_prob) if both else np.nan,
            "normal_proba_median": float(np.median(y_prob[y_true == 0])) if (y_true == 0).any() else np.nan,
            "leak_proba_median": float(np.median(y_prob[y_true == 1])) if (y_true == 1).any() else np.nan,
            "normal_proba_p95": float(np.quantile(y_prob[y_true == 0], 0.95)) if (y_true == 0).any() else np.nan,
            "leak_proba_p05": float(np.quantile(y_prob[y_true == 1], 0.05)) if (y_true == 1).any() else np.nan,
        }
        rows.append(row)
        if verbose:
            print(f"\nSteady 驗證 margin = {margin:.1f}s：正常 {row['n_normal_windows']} 窗、洩漏 {row['n_leak_windows']} 窗，"
                  f"ROC-AUC = {row['roc_auc']:.4f}、PR-AUC = {row['pr_auc']:.4f}")

    out = pd.DataFrame(rows)
    print("\n模型本身的可分離度（不套規則；排除 onset 前後 margin 秒）")
    print(out[["margin_sec", "n_normal_windows", "n_leak_windows", "roc_auc",
               "normal_proba_p95", "leak_proba_p05"]].to_string(index=False))
    return out



#                                 ********************************  onset 評估  ********************************
'''
模型什麼時候「第一次真正偵測到洩漏開始」？
  告警規則（平滑、連續命中、最短持續時間、veto）一律由 alarm_rule_utils.apply_alarm_method_by_record 計算，
  這裡只負責「拿算好的警報序列，和真實 onset 比較」，不再自己算一次警報。
'''
# ============================================================
# Onset-level 評估：以每條 record 為單位，評估 onset 偵測得如何
# ============================================================
def run_onset_evaluation(
    df_rule,                             # 已套用告警規則的資料（apply_alarm_method_by_record 的輸出）
    window_sec=0.2,                      # 窗長
    onset_tolerance_sec=1.0,             # onset 容許範圍：手動標記本身有誤差，onset 前這麼多秒內的報警不算誤報
    detected_col="detected_rule",        # 警報欄位
    smoothed_col="smoothed_proba_rule",  # 平滑後分數欄位（只用來輸出，不影響判斷）
):
    '''
    每條 record 輸出一列。報警時間用「窗結束時間」（有 t_ref 時 = 2 × t_ref − t_start，否則 t_start + window_sec），
    因為要等這一窗收完才算得出結果，這才是現場真正能發出警報的時間。
    
    有 onset 的 record（洩漏）：
      - false_alarm      ：在 onset − onset_tolerance_sec 之前就報警（太早，不是標記誤差能解釋的）
      - valid_detection  ：在 onset − onset_tolerance_sec 之後，警報「響起」（0 → 1）
                           之前就在響、一路延續過來的警報不算：那是誤報沒停，不是偵測到洩漏
      - latency_sec      ：第一次有效響起 − onset（落在容許範圍內會是負的，最小 −onset_tolerance_sec）
    沒有 onset 的 record（整條都是正常）：
      - false_alarm      ：只要報警就算
      - valid_detection / latency_sec：不適用，填 NaN

    false_alarm、valid_detection 存成 1.0 / 0.0 / NaN，直接取平均就是比例，NaN 會自動略過。
    '''
    if detected_col not in df_rule.columns:
        raise ValueError(f"找不到警報欄位 {detected_col}：請先用 apply_alarm_method_by_record 套用規則")

    rows = []
    for rid, sub in df_rule.groupby("record_id"):
        sub = sub.sort_values("t_start")
        detected = pd.to_numeric(sub[detected_col], errors="coerce").fillna(0).astype(int).to_numpy()
        # 窗結束時間：有 t_ref（窗中心）就用 2 × t_ref − t_start，和干擾推論同一個定義
        if "t_ref" in sub.columns:
            t_alarm = (2.0 * sub["t_ref"] - sub["t_start"]).to_numpy(dtype=float)
        else:
            t_alarm = sub["t_start"].to_numpy(dtype=float) + window_sec
        alarm_idx = np.flatnonzero(detected == 1)
        first_alarm = float(t_alarm[alarm_idx[0]]) if len(alarm_idx) else np.nan
        # 警報「響起」的窗：這一窗是 1、前一窗是 0（第一窗就是 1 也算）
        rise = (detected == 1) & (np.r_[0, detected[:-1]] == 0)
        
        onset_vals = pd.to_numeric(sub.get("onset_gt_sec"), errors="coerce").dropna() if "onset_gt_sec" in sub.columns else pd.Series(dtype=float)
        has_onset = not onset_vals.empty
        if onset_vals.nunique() > 1:
            raise ValueError(f"record {rid} 的 onset_gt_sec 不只一個值：{sorted(onset_vals.unique().tolist())[:5]}"
                             "（同一條 record 只能有一個 onset，請檢查特徵工程或標記）")

        row = {
            "record_id": rid,
            "has_onset": has_onset,
            "first_alarm_sec": first_alarm,
            "any_alarm": float(len(alarm_idx) > 0),
            "max_smoothed_proba": float(pd.to_numeric(sub[smoothed_col], errors="coerce").max())
                                  if smoothed_col in sub.columns else np.nan,
        }

        if not has_onset:
            # 純正常 record：任何報警都是誤報
            row.update({
                "onset_gt_sec": np.nan,
                "false_alarm": float(len(alarm_idx) > 0),
                "valid_detection": np.nan,
                "valid_detected_onset_sec": np.nan,
                "latency_sec": np.nan,
                "first_alarm_lead_sec": np.nan,
                "pre_onset_detected_windows": int(detected.sum()),
                "pre_onset_total_windows": int(len(detected)),
                "pre_onset_detected_ratio": float(detected.mean()) if len(detected) else np.nan,
                "post_onset_1s_detected_windows": np.nan,
                "post_onset_1s_total_windows": np.nan,
                "post_onset_1s_detected_ratio": np.nan,
            })
            rows.append(row)
            continue

        onset = float(onset_vals.iloc[0])
        early_limit = onset - onset_tolerance_sec               # 比這更早的報警才算誤報

        pre_mask = t_alarm < early_limit
        valid_idx = np.flatnonzero(rise & (t_alarm >= early_limit))     # 容許範圍開始後「響起」才算偵測到
        valid_onset = float(t_alarm[valid_idx[0]]) if len(valid_idx) else np.nan

        post_mask = (t_alarm >= onset) & (t_alarm <= onset + 1.0)
        pre_total, post_total = int(pre_mask.sum()), int(post_mask.sum())

        row.update({
            "onset_gt_sec": onset,
            "false_alarm": float(bool((detected[pre_mask] == 1).any())),
            "valid_detection": float(len(valid_idx) > 0),
            "valid_detected_onset_sec": valid_onset,
            
            # 容許範圍開始時警報已經在響（誤報一路延續過來）：這條 record 的偵測不能算數
            "alarm_ongoing_at_tolerance_start": float(bool(detected[pre_mask][-1] == 1)) if pre_mask.any() else 0.0,
                        
            "latency_sec": valid_onset - onset if len(valid_idx) else np.nan,
            "first_alarm_lead_sec": first_alarm - onset if len(alarm_idx) else np.nan,
            "pre_onset_detected_windows": int(detected[pre_mask].sum()),
            "pre_onset_total_windows": pre_total,
            "pre_onset_detected_ratio": float(detected[pre_mask].mean()) if pre_total else np.nan,
            "post_onset_1s_detected_windows": int(detected[post_mask].sum()),
            "post_onset_1s_total_windows": post_total,
            "post_onset_1s_detected_ratio": float(detected[post_mask].mean()) if post_total else np.nan,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 整理 onset_result 大表
# ============================================================
def export_onset_summary_tables(
        df_pred,             # window-level 預測結果表
        onset_result_df,     # 每個 record_id 的 onset 偵測結果（run_onset_evaluation 的輸出）
        output_dir):         # 輸出資料夾
    '''
    把每條 record 的 onset 結果，併上洩漏量、距離、壓力，輸出幾張摘要表：
      onset_results_with_meta.csv          ：每條 record 一列（明細）
      onset_summary_by_volume.csv          ：依洩漏量（有多種壓力時也依壓力分）
      onset_summary_by_volume_distance.csv ：依洩漏量 × 距離（同上）
      onset_failed_cases.csv               ：有問題的 record（漏抓、太早報、純正常卻報警），附原因

    誤報分兩種，分開算（定義不同，不能混在同一欄平均）：
      early_false_alarm_rate        ：洩漏 record 在 onset − 容許秒數之前就報警
      normal_record_false_alarm_rate：純正常 record（沒有 onset）只要報警就算
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 每個 record 取一筆 metadata，併回 onset-level 結果
    meta_cols = ["record_id", "leak_quantity", "volume_ml", "distance_cm", "pressure_bar"]
    keep_meta_cols = [c for c in meta_cols if c in df_pred.columns]
    record_meta = (
        df_pred.sort_values(["record_id", "t_start"])
        .groupby("record_id", as_index=False)[keep_meta_cols]
        .first()
    )
    onset_with_meta = onset_result_df.merge(record_meta, on="record_id", how="left")

    # 洩漏量欄位：volume_ml 優先，沒有就用 leak_quantity
    if "volume_ml" in onset_with_meta.columns:
        volume_col = "volume_ml"
    elif "leak_quantity" in onset_with_meta.columns:
        volume_col = "leak_quantity"
    else:
        raise ValueError("onset summary 找不到 volume_ml 或 leak_quantity 欄位")

    # 有多種壓力時（例如舊的低壓資料 + 新的高壓資料）一定要分開看，混在一起的偵測率沒有意義
    pressure_keys = (["pressure_bar"] if "pressure_bar" in onset_with_meta.columns
                     and onset_with_meta["pressure_bar"].nunique(dropna=True) > 1 else [])

    # 兩種誤報分開成兩欄，方便 groupby 直接平均
    has_onset = onset_with_meta["has_onset"].astype(bool)
    onset_with_meta["early_false_alarm"] = onset_with_meta["false_alarm"].where(has_onset)
    onset_with_meta["normal_record_false_alarm"] = onset_with_meta["false_alarm"].where(~has_onset)

    def _summarize(keys):
        return (
            onset_with_meta.groupby(keys, dropna=False)
            .agg(
                n_records=("record_id", "count"),
                n_leak_records=("has_onset", "sum"),
                valid_detection_rate=("valid_detection", "mean"),             # 只算洩漏 record（純正常是 NaN，自動略過）
                early_false_alarm_rate=("early_false_alarm", "mean"),
                normal_record_false_alarm_rate=("normal_record_false_alarm", "mean"),
                mean_latency_sec=("latency_sec", "mean"),
                median_latency_sec=("latency_sec", "median"),
                max_latency_sec=("latency_sec", "max"),                     # 最慢的那一條（看最壞情況）
            )
            .reset_index()
            .sort_values(keys)
        )

    # 1) 依洩漏量（＋壓力）
    volume_summary = _summarize(pressure_keys + [volume_col])
    # 2) 依洩漏量 × 距離（＋壓力）
    volume_distance_summary = (_summarize(pressure_keys + [volume_col, "distance_cm"])
                               if "distance_cm" in onset_with_meta.columns else pd.DataFrame())

    # 3) 有問題的 record：漏抓、太早報、純正常卻報警，都列出來並寫原因
    missed = has_onset & (onset_with_meta["valid_detection"] == 0)
    early = has_onset & (onset_with_meta["false_alarm"] == 1)
    normal_fa = (~has_onset) & (onset_with_meta["false_alarm"] == 1)
    reason = np.select(
        [missed & early, missed, early, normal_fa],
        ["漏抓＋太早報", "漏抓", "太早報", "純正常卻報警"],
        default="",
    )
    failed_cases = onset_with_meta.assign(fail_reason=reason).loc[
        reason != "",
        [c for c in [
            "record_id", "fail_reason", volume_col, "distance_cm", "pressure_bar",
            "onset_gt_sec", "first_alarm_sec", "valid_detected_onset_sec", "latency_sec",
            "false_alarm", "alarm_ongoing_at_tolerance_start",
            "pre_onset_detected_ratio", "post_onset_1s_detected_ratio", "max_smoothed_proba",
        ] if c in onset_with_meta.columns or c == "fail_reason"]
    ].copy()

    # 輸出
    onset_with_meta_csv = output_dir / "onset_results_with_meta.csv"
    volume_summary_csv = output_dir / "onset_summary_by_volume.csv"
    volume_distance_csv = output_dir / "onset_summary_by_volume_distance.csv"
    failed_cases_csv = output_dir / "onset_failed_cases.csv"

    onset_with_meta.to_csv(onset_with_meta_csv, index=False, encoding="utf-8-sig")
    volume_summary.to_csv(volume_summary_csv, index=False, encoding="utf-8-sig")
    if not volume_distance_summary.empty:
        volume_distance_summary.to_csv(volume_distance_csv, index=False, encoding="utf-8-sig")
    failed_cases.to_csv(failed_cases_csv, index=False, encoding="utf-8-sig")

    print(f"已輸出 onset 明細+meta: {onset_with_meta_csv}")
    print(f"已輸出 volume 摘要: {volume_summary_csv}")
    if pressure_keys:
        print(f"⚠️ 資料裡有 {onset_with_meta['pressure_bar'].nunique()} 種壓力："
              f"{sorted(onset_with_meta['pressure_bar'].dropna().unique().tolist())}，摘要已依壓力分開；"
              "如果預期只有一種壓力，請檢查是不是混到了別批資料")
    if not volume_distance_summary.empty:
        print(f"已輸出 volume x distance 摘要: {volume_distance_csv}")
    print(f"已輸出 failed cases（{len(failed_cases)} 條）: {failed_cases_csv}")

    return onset_with_meta, volume_summary, volume_distance_summary, failed_cases


# ============================================================
# 畫混淆矩陣
# ============================================================
def plot_confusion_matrix(cm,              # 混淆矩陣，2×2 的 numpy 陣列（列 = 真實、欄 = 預測）
                          class_names,     # 類別名稱清單，例如 ["正常", "洩漏"]
                          title,           # 圖標題
                          save_path):      # 存檔路徑
    '''
    每一格寫「窗數」和「佔該真實類別的比例」：
      例如真實洩漏 300 窗、其中 60 窗被判成正常 → 那一格寫「60 (20.0%)」，一看就知道漏抓兩成。
    正常窗通常比洩漏窗多很多，只看窗數很難判斷嚴不嚴重，所以比例是以「每一列（真實類別）」為分母。
    '''
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    # 用 fig / ax 物件畫圖，不會影響其他地方的 plt 狀態
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)                      # 色階條：顏色深淺對應的窗數
    ax.set_title(title)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted（預測）")
    ax.set_ylabel("True（真實）")

    # 每一格：窗數 + 比例；深色格子用白字，淺色格子用黑字，才看得清楚
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 已儲存: {save_path}")


# ============================================================
# 解決「json.dump 不能讀取 numpy/pandas 型別」的問題
# ============================================================
# 和干擾推論共用同一個轉換函式（只維護一份）。
# 舊版自己寫的 _to_builtin 不處理 NaN，postprocess_config.json 會寫出不合法的 NaN
_to_builtin = _json_safe


# ============================================================
# 輸出推論結果：run_info、window 預測、window 指標、onset 結果、可分離度、混淆矩陣
# ============================================================
def save_transition_inference_outputs(
    output_dir,
    df_pred,                    # 套用規則後的逐窗結果（有 detected_rule）
    window_metrics,             # 套用規則後的 window-level 指標（evaluate_window_level(..., pred_col="detected_rule")）
    raw_window_metrics=None,    # 模型原始門檻的 window-level 指標（pred_col="pred_label"），有給就一起存、一起畫
    onset_result_df=None,
    steady_summary_df=None,
    model_id=None,
    model_folder=None,
    feature_csv=None,
    settings=None,
    metadata=None,
):
    '''
    輸出到 output_dir：
      run_info.json                     ：這次用了哪顆模型、哪條規則、哪些參數
      window_predictions.csv            ：逐窗 pred_proba、detected_rule 等
      window_metrics.json               ：套用規則後的 window-level 指標
      window_metrics_raw.json           ：模型原始門檻的 window-level 指標（有給 raw_window_metrics 才有）
      onset_results.csv                 ：每條 record 的 onset 偵測結果
      steady_separability_summary.csv   ：模型本身的可分離度（不套規則）
      plots/window_confusion_matrix_rule.png / _raw.png：兩張混淆矩陣，並排比就看得出規則改變了什麼
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = metadata or {}
    settings = settings or {}

    # 1. run info：這次推論到底用了什麼模型、什麼規則、什麼設定
    run_info = {
        # 模型（來自 metadata.json）
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "primary_eval_name": metadata.get("performance", {}).get("primary_eval_name"),

        # 路徑與資料
        "model_folder": str(model_folder) if model_folder is not None else None,
        "feature_csv": str(feature_csv) if feature_csv is not None else None,
        "data_split": settings.get("data_split"),                # 這次評估用的是哪一份（holdout）
        "n_rows": int(len(df_pred)),
        "n_records": int(df_pred["record_id"].nunique()) if "record_id" in df_pred.columns else None,
        "n_features_used": int(len(settings.get("feature_names", []))),
        "feature_names": list(settings.get("feature_names", [])),

        # 門檻與評估參數
        "window_threshold": settings.get("window_threshold"),
        "file_threshold": settings.get("file_threshold"),
        "window_sec": settings.get("window_sec"),
        "onset_tolerance_sec": settings.get("onset_tolerance_sec"),

        # 告警規則（完整內容見 settings_raw）
        "rule_name": settings.get("name"),
        "alarm_method": settings.get("method"),
        "min_leak_duration_sec": settings.get("min_leak_duration_sec"),
        "use_veto": settings.get("use_veto"),

        "settings_raw": settings,
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(run_info), f, indent=2, ensure_ascii=False)

    # 2. 逐窗結果
    df_pred.to_csv(output_dir / "window_predictions.csv", index=False, encoding="utf-8-sig")

    # 3. window-level 指標：套規則後、模型原始各一份
    with open(output_dir / "window_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(window_metrics), f, indent=2, ensure_ascii=False)
    if raw_window_metrics is not None:
        with open(output_dir / "window_metrics_raw.json", "w", encoding="utf-8") as f:
            json.dump(_to_builtin(raw_window_metrics), f, indent=2, ensure_ascii=False)

    # 4. onset 結果、可分離度
    if onset_result_df is not None:
        onset_result_df.to_csv(output_dir / "onset_results.csv", index=False, encoding="utf-8-sig")
    if steady_summary_df is not None:
        steady_summary_df.to_csv(output_dir / "steady_separability_summary.csv", index=False, encoding="utf-8-sig")

    # 5. 混淆矩陣：套規則後一定畫；有給模型原始的也畫一張，標題寫清楚是哪一種
    rule_name = settings.get("name", "")
    for metrics, tag, title in [
        (window_metrics, "rule", f"套用規則後（{rule_name}）"),
        (raw_window_metrics, "raw", f"模型原始門檻（window_threshold = {settings.get('window_threshold')}）"),
    ]:
        cm = None if metrics is None else metrics.get("confusion_matrix")
        if cm is not None and np.asarray(cm).size > 0:
            plot_confusion_matrix(
                np.asarray(cm),
                class_names=["正常", "洩漏"],
                title=f"Transition holdout window-level\n{title}",
                save_path=plots_dir / f"window_confusion_matrix_{tag}.png",
            )

    print(f"✅ 推論結果已輸出到: {output_dir}")


#                                      ********************************  後處理規則比較（選規則：只用 dev）  ********************************
# ============================================================
# 讀取所有候選 postprocess config（干擾推論 mode="select" 的輸出）
# ============================================================
def load_postprocess_candidates(postprocess_dir, expected_model_id=None, pattern="postprocess_config_candidate_top*.json"):
    '''
    讀干擾推論 mode="select" 寫出的候選檔。兩種候選直接報錯，不拿來比：
      - source 不是 disturbance_dev_rule_selection：舊流程（在 holdout 上選規則）留下的檔案，拿來比等於偷看 holdout
      - model_id 和這次載入的模型不同：候選是為別顆模型選的，誤報成績對這顆模型不成立
    '''
    paths = sorted(Path(postprocess_dir).glob(pattern))
    if not paths:
        raise ValueError(f"找不到候選 postprocess config: {Path(postprocess_dir) / pattern}"
                         "（請先跑干擾推論 mode='select'）")
    out = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rule_cfg = payload.get("baseline_rule_settings", {})
        if not rule_cfg:
            raise ValueError(f"{path.name} 缺少 baseline_rule_settings")
        if payload.get("source") != "disturbance_dev_rule_selection":
            raise ValueError(f"{path.name} 的 source = {payload.get('source')!r}，不是干擾 dev 選出的候選"
                             "（可能是舊流程留下的檔案）：請重跑干擾推論 mode='select'")
        if expected_model_id is not None and payload.get("model_id") != expected_model_id:
            raise ValueError(f"{path.name} 是給模型 {payload.get('model_id')} 的候選，這次載入的是 {expected_model_id}"
                             "：請用這顆模型重跑干擾推論 mode='select'")
        out.append({"path": path, "payload": payload,
                    "rule_name": payload.get("baseline_rule_name", rule_cfg.get("name", path.stem)),
                    "rule_cfg": rule_cfg})
    return out


# ============================================================
# 一條規則在轉換資料上的成績（window + onset）
# ============================================================
def score_rule_on_transition(df_pred, alarm_cfg, window_threshold, window_sec, onset_tolerance_sec):
    '''套用規則，回傳 (df_rule, onset_df, 成績 dict)；window 指標只算有標籤的窗（灰色地帶排除）'''
    df_rule = apply_alarm_method_by_record(df_pred, alarm_cfg, window_threshold)
    onset_df = run_onset_evaluation(df_rule, window_sec=window_sec, onset_tolerance_sec=onset_tolerance_sec)

    label = pd.to_numeric(df_rule["label"], errors="coerce") if "label" in df_rule.columns else pd.Series(dtype=float)
    # label 只能是 0、1 或空值（灰色地帶）；0.5 之類的值用 astype(int) 會被悄悄截成 0
    bad = label.notna() & ~label.isin([0, 1])
    if bad.any():
        raise ValueError(f"label 只能是 0、1 或空值，出現了 {sorted(label[bad].unique().tolist())[:5]}")
    mask = label.notna()
    y_true = label[mask].astype(int).values
    y_pred = df_rule.loc[mask, "detected_rule"].astype(int).values

    leak = onset_df[onset_df["has_onset"]]
    normal = onset_df[~onset_df["has_onset"]]
    score = {
        "n_leak_records": int(len(leak)),
        "n_normal_records": int(len(normal)),
        "valid_detection_rate": leak["valid_detection"].mean() if len(leak) else np.nan,     # 洩漏 record 有抓到的比例
        "early_false_alarm_rate": leak["false_alarm"].mean() if len(leak) else np.nan,       # 洩漏 record 在 onset 前就亂報的比例
        "normal_record_false_alarm_rate": normal["false_alarm"].mean() if len(normal) else np.nan,  # 純正常 record 報警的比例
        "median_latency_sec": leak["latency_sec"].median() if len(leak) else np.nan,
        "p90_latency_sec": leak["latency_sec"].quantile(0.9) if leak["latency_sec"].notna().any() else np.nan,
        "window_precision": precision_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
        "window_recall": recall_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
        "window_f1": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
        "n_gray_windows": int((~mask).sum()),
        # 洩漏開始時警報已經在響（誤報一路延續過來）的 record 數：這些 record 的偵測不算數
        "n_alarm_ongoing_at_onset": int(leak["alarm_ongoing_at_tolerance_start"].sum())
                                    if "alarm_ongoing_at_tolerance_start" in leak.columns else 0,
    }

    return df_rule, onset_df, score


# ============================================================
# 選規則：16 個候選在轉換 dev 上的偵測成績 + 干擾 dev 的誤報成績，放同一張表
# ============================================================
def run_transition_postprocess_candidates(
    feature_csv,                   # transition_rule_features_eval.csv（有 split 欄；只用 dev）
    metadata_json,
    candidate_dir,                 # 干擾推論輸出候選檔的資料夾（模型資料夾）
    feature_config_json=None,
    output_dir=None,
    run_name="transition_select",
    onset_tolerance_sec=1.0,       # onset 前這麼多秒內的報警不算誤報（手動標記誤差）
):
    '''
    只用轉換的 dev 份，不碰 holdout。
    輸出 transition_postprocess_candidate_comparison.csv：每個候選一列，左邊是干擾成績、右邊是轉換成績，
    你看這張表選定一條規則，再分別跑干擾推論與轉換推論的 evaluate。
    '''
    pred = load_model_and_predict(feature_csv, metadata_json, feature_config_json)
    df_dev, df_hold, _ = split_dev_holdout_for_rules(pred["df_pred"])
    window_sec, _ = estimate_window_timing(df_dev)

    output_dir = Path(output_dir) if output_dir else (
        pred["model_folder"] / "inference_runs" / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_postprocess_candidates(candidate_dir, expected_model_id=pred["model_id"])
    rows, onset_all = [], []
    for item in candidates:
        rule_cfg = item["rule_cfg"]
        alarm_cfg = rule_cfg_to_alarm_cfg(rule_cfg)
        missing = [c for c in required_rule_columns(alarm_cfg) if c not in df_dev.columns]
        if missing:
            raise ValueError(f"規則 {item['rule_name']} 需要的 veto 欄位不在轉換資料裡：{missing}")

        _, onset_df, score = score_rule_on_transition(
            df_dev, alarm_cfg, pred["window_threshold"], window_sec, onset_tolerance_sec)
        onset_all.append(onset_df.assign(rule_name=item["rule_name"]))

        rows.append({
            "rule_name": item["rule_name"],
            "base_rule": rule_cfg.get("base_rule"),
            "min_leak_duration_sec": rule_cfg.get("min_leak_duration_sec"),
            "min_alarm_latency_sec": rule_cfg.get("min_alarm_latency_sec"),
            # 干擾 dev 的成績（來自候選檔）
            "dist_effective_fa": rule_cfg.get("effective_fa"),
            "dist_effective_fa_source": rule_cfg.get("effective_fa_source"),
            "dist_meets_fa_limit": rule_cfg.get("meets_fa_limit"),
            "dist_clean_normal_fa": rule_cfg.get("clean_normal_false_alarm_rate"),
            "dist_event_hit_rate": rule_cfg.get("disturbance_event_hit_rate"),
            "dist_alarm_onsets": rule_cfg.get("disturbance_negative_alarm_onsets"),
            # 轉換 dev 的成績
            **{f"trans_{k}": v for k, v in score.items()},
        })

    # 排序只是方便看（不會自動選規則），順序照目標的優先順序：
    #   1. 干擾誤報達標 → 2. 轉換資料上不亂報（太早報、純正常 record 報警）→ 3. 抓得到 → 4. 抓得快
    summary = (pd.DataFrame(rows)
               .sort_values(["dist_meets_fa_limit", "trans_early_false_alarm_rate",
                             "trans_normal_record_false_alarm_rate", "trans_valid_detection_rate",
                             "trans_median_latency_sec"],
                            ascending=[False, True, True, False, True], na_position="last")
               .reset_index(drop=True))
    comparison_csv = output_dir / "transition_postprocess_candidate_comparison.csv"
    summary.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    pd.concat(onset_all, ignore_index=True).to_csv(
        output_dir / "transition_candidate_onset_results.csv", index=False, encoding="utf-8-sig")

    # 這張比較表是怎麼來的：之後寫 selection_reason 時，transition_comparison_csv 就填這次的 comparison_csv
    run_info = {
        "model_id": pred["model_id"],
        "feature_csv": str(feature_csv),
        "data_split": "dev",
        "n_dev_records": int(df_dev["record_id"].nunique()),
        "n_dev_leak_records": int(summary["trans_n_leak_records"].iloc[0]) if len(summary) else 0,
        "onset_tolerance_sec": onset_tolerance_sec,
        "candidate_files": [item["path"].name for item in candidates],
        "comparison_csv": str(comparison_csv),
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(run_info), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"候選規則比較（轉換 dev {run_info['n_dev_records']} 筆 record，其中洩漏 {run_info['n_dev_leak_records']} 筆；"
          f"onset 容許 {onset_tolerance_sec} 秒）")
    print("=" * 80)
    show_cols = ["rule_name", "dist_effective_fa", "dist_event_hit_rate", "dist_meets_fa_limit",
                 "trans_early_false_alarm_rate", "trans_normal_record_false_alarm_rate",
                 "trans_valid_detection_rate", "trans_median_latency_sec", "trans_n_alarm_ongoing_at_onset"]
    print(summary[[c for c in show_cols if c in summary.columns]].to_string(index=False))
    print(f"\n已輸出：{comparison_csv}")
    print("選定規則後，干擾推論 evaluate 的 transition_comparison_csv 請填上面這個路徑")
    
    return summary


#                                      ********************************  正式評估（只用 holdout）  ********************************
# ============================================================
# 驗收標準檢查：成績有沒有達到事先訂好的需求
# ============================================================
ACCEPTANCE_KEYS = {
    # 鍵                                    : (成績欄位, 比較方式)
    "min_valid_detection_rate":            ("valid_detection_rate", ">="),             # 洩漏 record 至少抓到幾成
    "max_early_false_alarm_rate":          ("early_false_alarm_rate", "<="),           # 洩漏 record 太早報的比例上限
    "max_normal_record_false_alarm_rate":  ("normal_record_false_alarm_rate", "<="),   # 純正常 record 報警的比例上限
    "max_median_latency_sec":              ("median_latency_sec", "<="),               # 延遲中位數上限（秒）
    "max_p90_latency_sec":                 ("p90_latency_sec", "<="),                  # 延遲 P90 上限（秒）
}


def check_acceptance(score, acceptance, disturbance_holdout):
    '''
    回傳 (是否通過, 沒通過的項目清單)。
    acceptance 裡寫 None 的項目不檢查；成績是 NaN（例如 holdout 沒有純正常 record）也算沒通過，
    因為「量不到」不等於「達標」。
    干擾 holdout 的 meets_fa_limit 也一併檢查：兩邊都達標才算驗收通過。
    '''
    unknown = set(acceptance) - set(ACCEPTANCE_KEYS)
    if unknown:
        raise ValueError(f"acceptance 有不認得的項目：{sorted(unknown)}（只支援 {sorted(ACCEPTANCE_KEYS)}）")
    failed = []
    for key, limit in acceptance.items():
        if limit is None:
            continue
        col, op = ACCEPTANCE_KEYS[key]
        v = score.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            failed.append(f"{col} 量不到（NaN），無法確認 {op} {limit}")
        elif (op == ">=" and not v >= limit) or (op == "<=" and not v <= limit):
            failed.append(f"{col} = {v:.4g}，未達 {op} {limit}")
    if disturbance_holdout.get("meets_fa_limit") is False:
        failed.append(f"干擾 holdout effective_fa = {disturbance_holdout.get('effective_fa')}，"
                      f"超過上限 {disturbance_holdout.get('max_disturbance_fa')}")
    return len(failed) == 0, failed


# ============================================================
# 評估已定案的規則：讀 postprocess_config.json，只在轉換 holdout 上量一次
# ============================================================
def run_transition_feature_inference(
    feature_csv,                   # transition_rule_features_eval.csv（有 split 欄；只用 holdout）
    metadata_json,
    postprocess_config_json,       # 干擾推論 mode="evaluate" 寫出的鎖定規則
    acceptance,                    # 驗收標準（需求，必須在看結果之前訂好）：見 ACCEPTANCE_KEYS
    feature_config_json=None,
    run_name="transition_evaluate",
    onset_tolerance_sec=1.0,
    export_onset_summary=True,
    allow_reevaluate=False,        # 已經評估過還要再跑一次才設 True（舊成績會留在 transition_holdout_history）
):
    '''
    不排名、不換規則：只量鎖定的那一條，然後和驗收標準比：
      - 通過（轉換 holdout 達標，且干擾 holdout 也達標）→ validation_status = final_verified，才能產生正式 runtime schema
      - 沒通過 → validation_status = rejected_transition_holdout，正式 schema 會被擋下，需要回頭重新選規則或改善模型
    「量完」不等於「通過」：沒有驗收標準的話，再差的成績也會被標成可部署。

    holdout 只量一次：已經有轉換 holdout 成績時預設會報錯，避免改了 onset_tolerance_sec 之類的設定重跑到通過為止。
    成績另存一份，也附加進 postprocess_config.json 的 transition_holdout_evaluation。
    '''
    if not isinstance(acceptance, dict) or not any(v is not None for v in acceptance.values()):
        raise ValueError("必須給 acceptance（驗收標準），至少一項不是 None：例如 "
                         "{'min_valid_detection_rate': 0.95, 'max_early_false_alarm_rate': 0.05}")

    postprocess_config_json = Path(postprocess_config_json)
    payload = load_postprocess_config(postprocess_config_json)
    allowed_status = ("pending_transition_holdout", "final_verified", "rejected_transition_holdout")
    if (payload.get("source") != "disturbance_dev_rule_selection" or "holdout_evaluation" not in payload
            or payload.get("validation_status") not in allowed_status):
        raise ValueError("postprocess_config.json 不是干擾推論 mode='evaluate' 寫出的鎖定規則"
                         "（source 應為 disturbance_dev_rule_selection、要有 holdout_evaluation、"
                         f"validation_status 目前是 {payload.get('validation_status')!r}）")
    if "transition_holdout_evaluation" in payload and not allow_reevaluate:
        raise ValueError("這條規則已經在轉換 holdout 上評估過了（holdout 只量一次）；"
                         "確定要重跑請設 allow_reevaluate=True，舊成績會保留在 transition_holdout_history")
    alarm_cfg = rule_cfg_to_alarm_cfg(payload["baseline_rule_settings"])

    pred = load_model_and_predict(feature_csv, metadata_json, feature_config_json)
    if payload.get("model_id") not in (None, pred["model_id"]):
        raise ValueError(f"postprocess_config 是給模型 {payload.get('model_id')} 的，這次載入的是 {pred['model_id']}")
    df_dev, df_hold, _ = split_dev_holdout_for_rules(pred["df_pred"])
    window_sec, _ = estimate_window_timing(df_hold)

    missing = [c for c in required_rule_columns(alarm_cfg) if c not in df_hold.columns]
    if missing:
        raise ValueError(f"正式規則需要的 veto 欄位不在轉換資料裡：{missing}")

    print(f"\n鎖定規則：{alarm_cfg['name']}（只在轉換 holdout 上評估）")
    raw_window_metrics = evaluate_window_level(df_hold, pred_col="pred_label", result_name="模型原始 Window-level（holdout）")
    steady_df = evaluate_steady_separability(df_hold, window_sec=window_sec)     # 模型本身分不分得開（不套規則）
    df_rule, onset_df, score = score_rule_on_transition(
        df_hold, alarm_cfg, pred["window_threshold"], window_sec, onset_tolerance_sec)
    window_metrics = evaluate_window_level(df_rule, pred_col="detected_rule", result_name="正式後處理 Window-level（holdout）")

    output_dir = pred["model_folder"] / "inference_runs" / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    settings = {"feature_names": pred["feature_names"], "window_threshold": pred["window_threshold"],
                "file_threshold": pred["file_threshold"], "window_sec": window_sec,
                "onset_tolerance_sec": onset_tolerance_sec, "data_split": "holdout", **alarm_cfg}
    save_transition_inference_outputs(
        output_dir=output_dir, df_pred=df_rule, window_metrics=window_metrics,
        raw_window_metrics=raw_window_metrics,
        onset_result_df=onset_df, steady_summary_df=steady_df,
        model_id=pred["model_id"], model_folder=pred["model_folder"],
        feature_csv=feature_csv, settings=settings, metadata=pred["metadata"],
    )
    if export_onset_summary:
        export_onset_summary_tables(df_pred=df_rule, onset_result_df=onset_df, output_dir=output_dir)

    # 驗收：轉換 holdout 成績 vs 驗收標準，再加上干擾 holdout 有沒有達標
    passed, failed = check_acceptance(score, acceptance, payload["holdout_evaluation"])

    holdout_eval = _to_builtin({
        "rule_name": alarm_cfg["name"],
        "evaluated_at": datetime.now().isoformat(),
        "n_windows": int(len(df_hold)),
        "onset_tolerance_sec": onset_tolerance_sec,
        **score,
        "raw_window_recall": raw_window_metrics["recall"],
        "raw_window_precision": raw_window_metrics["precision"],
        # 模型本身的可分離度（不套規則）：每個 margin 的 ROC-AUC
        "steady_roc_auc_by_margin": {f"{r.margin_sec:g}s": r.roc_auc for r in steady_df.itertuples()},
        "acceptance": acceptance,
        "acceptance_passed": passed,
        "acceptance_failed_items": failed,
    })
    with open(output_dir / "transition_holdout_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(holdout_eval, f, indent=2, ensure_ascii=False)

    # 重跑時，舊成績留在 history，不直接蓋掉
    if "transition_holdout_evaluation" in payload:
        payload.setdefault("transition_holdout_history", []).append(payload["transition_holdout_evaluation"])
    payload["transition_holdout_evaluation"] = holdout_eval
    payload["validation_status"] = "final_verified" if passed else "rejected_transition_holdout"
    postprocess_config_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(holdout_eval, indent=2, ensure_ascii=False))
    if passed:
        print(f"✅ 驗收通過：validation_status = final_verified，可以產生正式 runtime schema（{postprocess_config_json}）")
    else:
        print("❌ 驗收沒通過：validation_status = rejected_transition_holdout，正式 runtime schema 會被擋下")
        for item in failed:
            print("   -", item)

    return {"df_rule": df_rule, "onset_result_df": onset_df, "window_metrics": window_metrics,
            "holdout_evaluation": holdout_eval, "acceptance_passed": passed,
            "output_dir": output_dir, "model_id": pred["model_id"]}




if __name__ == "__main__":
    MODEL_FOLDER = Path(
        "/Users/paul/Desktop/Final/test_result/"
        "saved_models_transition_disturbance_final/xgb/"
        "xgb_20260923_164944"
    )
    MODEL_METADATA_JSON = MODEL_FOLDER / "metadata.json"

    # 轉換切分程式輸出的 dev + holdout（有 split 欄）
    FEATURE_CSV = Path(
        "/Users/paul/Desktop/Final/transition_feature_engineering/"
        "transition_30ml_test/transition_rule_features_eval.csv"
    )
    FEATURE_CONFIG_JSON = Path("/Users/paul/Desktop/Final/test_result/feature_config_micro.json")

    # ============================================================
    # 第 2 步（選規則）：干擾推論 mode="select" 之後執行
    #   16 個候選在轉換 dev 上的成績 + 干擾 dev 的誤報成績 → 看表選定一條
    # ============================================================
    run_transition_postprocess_candidates(
        feature_csv=FEATURE_CSV,
        metadata_json=MODEL_METADATA_JSON,
        candidate_dir=MODEL_FOLDER,
        feature_config_json=FEATURE_CONFIG_JSON,
    )

    # ============================================================
    # 第 4 步（最終驗收）：選定規則、跑完干擾推論 mode="evaluate"（寫出鎖定規則）之後才執行
    #   只在轉換 holdout 上量一次，和 ACCEPTANCE 比；通過才產生正式 runtime schema
    # ============================================================
    '''
    POSTPROCESS_CONFIG_JSON = MODEL_FOLDER / "postprocess_config.json"
    # 驗收標準 = 需求，必須在看 holdout 結果之前訂好，跑完之後不能再改
    ACCEPTANCE = {
        "min_valid_detection_rate": 0.95,            # 洩漏至少抓到 95%
        "max_early_false_alarm_rate": 0.05,          # 洩漏前就亂報的比例最多 5%
        "max_normal_record_false_alarm_rate": 0.0,   # 純正常 record 不能報警（holdout 沒有純正常 record 時改成 None）
        "max_median_latency_sec": 2.0,               # 延遲中位數最多 2 秒
        "max_p90_latency_sec": None,                 # 不檢查
    }
    result = run_transition_feature_inference(
        feature_csv=FEATURE_CSV,
        metadata_json=MODEL_METADATA_JSON,
        postprocess_config_json=POSTPROCESS_CONFIG_JSON,
        acceptance=ACCEPTANCE,
        feature_config_json=FEATURE_CONFIG_JSON,
    )
    if not result["acceptance_passed"]:
        raise SystemExit("驗收沒通過，不產生正式 runtime schema")
    RUNTIME_SCHEMA_JSON = save_final_runtime_feature_schema(
        model_folder=MODEL_FOLDER,
        metadata_json=MODEL_METADATA_JSON,
        postprocess_config_json=POSTPROCESS_CONFIG_JSON,
    )
    print("\n正式部署檔案已完成：")
    print("postprocess   :", POSTPROCESS_CONFIG_JSON)
    print("runtime schema:", RUNTIME_SCHEMA_JSON)
    '''