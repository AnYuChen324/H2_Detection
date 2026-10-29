#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
干擾推論與抗干擾規則比較

用「已訓練完成的模型」讀干擾特徵 CSV，量模型和告警規則在干擾資料上的誤報。
不重新訓練模型，不重新找頻帶 / 特徵；rel_ / z_ / offband 欄位由干擾特徵工程產生，這裡只檢查存在。

在完整流程中的位置：
  1. 干擾推論  mode="select"   ← 本程式：dev 上比較 272 條規則，每種基礎規則挑一個代表 → 16 份候選
  2. 轉換推論  選規則（dev）    ：16 個候選的洩漏偵測率，和干擾誤報放同一張表 → 你選定一條
  3. 干擾推論  mode="evaluate" ← 本程式：holdout 上只量選定的那一條，寫出正式 postprocess_config.json
  4. 轉換推論  正式評估（holdout）→ 產生正式 runtime_feature_schema.json

輸入：
  - metadata.json（模型資料夾裡；model_id 由資料夾名稱決定，載入後核對特徵清單）
  - disturbance_rule_features_eval.csv（切分程式輸出，有 split 欄：dev / holdout；沒有 split 欄會報錯）

主要輸出（模型資料夾）：
  - postprocess_config_candidate_top{名次}_{規則名}.json：16 份候選（select、evaluate 都會重寫）
  - postprocess_config.json：只有 evaluate 且 write_postprocess_final=True 才寫，附 holdout_evaluation
  - runtime_feature_schema_candidate.json
主要輸出（inference_runs/<run_name>_<時間>/）：
  - raw_predictions、disturbance_alarm_method_comparison.csv、run_info.json
  - {dev|holdout}_{規則名}_*：record / phase / disturbance_type / disturbance_id 摘要、false_alarm_cases

重要原則：
  1. 模型特徵、window_threshold 以模型存檔為準；找不到、來源不一致、不在 0～1 都會報錯，不補預設值
  2. 選規則只用 dev；holdout 只在規則定案後量一次（select 模式連 holdout 的數字都不顯示）
  3. 干擾誤報歸屬：干擾中＋干擾結束後 DISTURBANCE_TAIL_SEC 秒內的報警都算干擾的
     （干擾觸發的警報常在敲完、餘振時才響；特徵工程的 eval_class 是逐窗事實，歸屬是評估時決定）
  4. 規則只有一條，現場不知道干擾類型；依類型 / 事件拆開的表只用來診斷
  5. 舊資料（FINITE 錄法，每 0.5 秒有空檔）片段太短時，誤報率改用橋接值（fa_testable / bridged_*）
'''

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json
import re
import copy

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_manager_two_sensor_final import ModelManager
from alarm_rule_utils import apply_alarm_method_by_record, split_continuous_segments


#                                                ************************************************************
#                                           ****************************  共用工具函式  ********************************
# ============================================================
# 把 NaN 轉成可存 JSON 的小工具
# ============================================================
def _json_safe(obj):                 # 輸入一個 obj，可以是任何型別——字典、列表、tuple、NumPy 純量,甚至是一個普通的字串或數字
    
    # 處理 dict：遞迴進每個 value
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    # 處理 list / tuple：遞迴進每個元素（tuple 改成 list，JSON 沒有 tuple）
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # 處理 NumPy 布林（np.bool_ 不是 Python bool，json 不認得）
    if isinstance(obj, np.bool_):
        return bool(obj)
    # 處理 NumPy 整數型別
    if isinstance(obj, np.integer):
        return int(obj)
    # 處理浮點數（Python float 和 NumPy 浮點都走這裡）
    # 必須在這裡就檢查 NaN：否則 np.float64('nan') 轉成 float 後直接回傳，JSON 會寫出不合法的 NaN
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    # 處理 NumPy 陣列：轉成 list 後逐個元素處理
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    # 其他缺值（pd.NA、pd.NaT、None）
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


#                                                ************************************************************
#                                        ****************************  找出模型並核對資料  ********************************
# ============================================================
# 從模型 package 拆出推論需要的設定
# ============================================================
def extract_inference_config(pkg):
    '''
    從模型 package 裡拿出推論必需的東西，並確認沒有缺漏或版本衝突

    門檻一律以模型存檔為準，讀不到就直接報錯，不再自動用 0.5：
      - 0.5 不是模型訓練時決定的門檻，悄悄套用的話，後面所有誤報率都會失真
      - 讀不到通常代表模型存檔格式不對或存檔不完整，應該在這裡發現，而不是跑完才發現
    '''
    # 真正訓練好的模型本體
    model = pkg["model"]                              # 模型本體
    scaler = pkg.get("scaler", None)                  # 訓練時若有標準化才會有；XGB 通常沒有
    metadata = pkg.get("metadata", {}) or {}          # 模型描述資訊
    model_type = metadata.get("model_type", None)     # 從 metadata 裡拿模型類型
    
    # 模型訓練時真正吃進去的特徵清單（推論必須完全一樣）
    feature_names = metadata.get("features", {}).get("feature_names", None)
    # 檢查 feature names：如果沒有特徵清單，直接停止。
    if not feature_names:
        raise ValueError("metadata.features.feature_names 不存在")
        
    # 取得 thresholds 設定(先從 package 取得 threshold 集合)
    thresholds = pkg.get("thresholds", {}) or {}
    
    # 尋找指定門檻：依序找 package 頂層 → thresholds → 舊版 metadata.performance
    # 注意：pkg["threshold"] 是舊版留下的 file_threshold 別名（值 = file_threshold），刻意不讀，不能當 window 門檻
    # 有值的地方必須一致；不一致代表存檔裡有舊門檻沒更新，不能猜要用哪個
    def _find_threshold(key, perf_level):
        sources = {
            "package": pkg.get(key, None),
            "thresholds": thresholds.get(key, None),
            "metadata.performance": metadata.get("performance", {}).get(perf_level, {}).get("threshold", None),
        }
        found = {src: float(v) for src, v in sources.items() if v is not None}
        # 三個位置都找不到時，回傳 None
        if not found:
            return None
        # 模型存檔可能混入舊設定，或有某次更新沒有同步。若程式偷偷用 0.5，後面所有誤報率、告警率與比較結果都可能失真，直接停止
        if len(set(found.values())) > 1:
            raise ValueError(f"模型存檔裡 {key} 有不只一個值，請確認哪個才是正確的：{found}")
        return next(iter(found.values()))
    
    # window_threshold：把每個 window 的機率轉成 0/1（pred_label）的門檻；告警規則的門檻另外寫在規則設定裡
    window_threshold = _find_threshold("window_threshold", "window_level")shared_feature_utils
    if window_threshold is None:
        raise ValueError("模型存檔找不到 window_threshold（package / thresholds / metadata.performance 都沒有）")
        
    if not (0.0 < window_threshold < 1.0):
        raise ValueError(f"window_threshold = {window_threshold} 不在 0～1 之間，模型存檔可能有誤")

    # file_threshold 是檔案層級或 record 層級的判斷門檻, 把多個 window 的結果彙整成整支 TDMS 結果時，可能判斷：洩漏 window 比例 >= file_threshold → 整支檔案判定為 leak
    # file_threshold：這支程式沒有用到，只記錄下來；找不到就是 None，不補預設值
    file_threshold = _find_threshold("file_threshold", "file_level")

    return (
        model,
        scaler,
        feature_names,
        float(window_threshold),
        None if file_threshold is None else float(file_threshold),
        metadata,
        model_type,
    )


# ============================================================
# 從 metadata 路徑反推模型 root / model_id
# ============================================================
def resolve_model_location_from_metadata(metadata_json_path):
    '''
    從 metadata.json 的路徑反推模型位置（只看路徑，不讀內容）。
    路徑結構必須是：<models_dir>/<model_type>/<model_id>/metadata.json
      例：saved_models_.../xgb/xgb_20260923_164944/metadata.json
          → models_dir = saved_models_...，model_id = xgb_20260923_164944
    '''
    metadata_json_path = Path(metadata_json_path)     # 傳進來的路徑轉成 Path 物件
    # 檢查路徑的檔名
    if metadata_json_path.name != "metadata.json":
        raise ValueError("請傳入 metadata.json 路徑")
        
    # 路徑打錯時在這裡就停，不要等到 ModelManager 載入時才出現看不懂的錯誤
    if not metadata_json_path.is_file():
        raise FileNotFoundError(f"找不到 metadata.json：{metadata_json_path}")

    # 從路徑結構反推資訊，不是去讀 metadata 裡面的內容
    model_folder = metadata_json_path.parent          # .../xgb/<model_id>
    model_id = model_folder.name                      # <model_id>
    models_dir = model_folder.parent.parent           # 跳過 model_type 那一層（xgb）
    return models_dir, model_id, model_folder


# ============================================================
# 載入模型：只載入「指定的那顆」XGB
# ============================================================
def get_xgb_model(manager: ModelManager, model_id, metadata_json=None):     # model_id 必須給（不會自動挑最新的模型）；metadata_json 用來確認載入的是同一顆
    '''
    載入「指定的那顆」XGB 模型。
    model_id 必須明確給：不再自動抓「最新的模型」，否則規則可能是用 A 模型選的、卻套在 B 模型上，結果完全不會報錯。
    有給 metadata_json 時，會檢查載入的模型和這份 metadata.json 是同一顆（特徵清單一致）。
    '''
    if not model_id:
        raise ValueError("必須指定 model_id（請由 resolve_model_location_from_metadata 取得），不會自動挑最新的模型")
        
    pkg = manager.load_model(model_id)
    pkg_meta = pkg.get("metadata", {}) or {}
    
    # 這支函式只處理 XGB；metadata 有寫模型類型、而且不是 xgb 就停
    model_type = pkg_meta.get("model_type")
    if model_type is not None and str(model_type).lower() != "xgb":
        raise ValueError(f"模型 {model_id} 的 model_type = {model_type!r}，不是 xgb")
        
    # 載入的模型必須和你指定的 metadata.json 是同一顆
    if metadata_json is not None:
        disk_meta = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
        disk_names = disk_meta.get("features", {}).get("feature_names")
        pkg_names = pkg_meta.get("features", {}).get("feature_names")
        if disk_names is not None and pkg_names is not None and list(disk_names) != list(pkg_names):
            raise ValueError(
                f"載入的模型 {model_id} 和 {metadata_json} 的特徵清單不一致，"
                "可能載到別的資料夾裡同名的模型（請確認 models_dir）")

    return model_id, pkg


# ============================================================
# 檢查推論資料欄位是否和模型訓練時保存的特徵一致
# ============================================================
def check_feature_alignment(df, feature_names):
    """
    1. 欄位齊不齊：模型要的特徵，CSV 都要有 → 缺一個就停
    2. 值乾不乾淨：模型要的特徵不能有 NaN / inf → 有就停
       （XGB 遇到 NaN 不會報錯，會自己走預設分支，結果悄悄變了也不會發現）
    送進模型時一律用 df[feature_names]，所以欄位順序一定和訓練時一致，不需要另外檢查。
    """
    missing_cols = [c for c in feature_names if c not in df.columns]     # 在找「模型需要，但推論 CSV 裡沒有」的欄位
    
    print("\n" + "=" * 80)
    print("特徵對齊檢查")
    print("=" * 80)
    print("模型需要特徵數量:", len(feature_names))
    print("模型特徵名稱:", feature_names)
    print("缺少特徵:", missing_cols)
    
    if missing_cols:
        raise ValueError("推論資料缺少模型需要的特徵欄位:\n" + "\n".join(missing_cols))
    
    # 空資料在這裡就停（例如 split 篩選後一列都沒有），不要等到後面才出現看不懂的錯誤
    if len(df) == 0:
        raise ValueError("推論資料是空的（0 列），請確認特徵 CSV 或 split 篩選條件")
    
    raw = df[feature_names]
    X = raw.apply(pd.to_numeric, errors="coerce")
    
    # 先分出「原本有值、轉數字後才變 NaN」的情況：那是 CSV 裡混進文字，不是特徵工程算出 NaN
    text_counts = (X.isna() & raw.notna()).sum()
    text_counts = text_counts[text_counts > 0]
    if not text_counts.empty:
        raise ValueError(
            "模型特徵裡有轉不成數字的文字（請檢查 CSV 是否被手動編輯或寫入格式錯誤）:\n"
            + "\n".join(f"  {col}: {int(n)} 列" for col, n in text_counts.items())
        )

    # 檢查模型特徵裡有沒有 NaN / inf
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    bad_counts = (~np.isfinite(X)).sum()
    bad_counts = bad_counts[bad_counts > 0]
    if not bad_counts.empty:
        raise ValueError(
            "模型特徵含 NaN / inf（請回頭檢查特徵工程）:\n"
            + "\n".join(f"  {col}: {int(n)} 列" for col, n in bad_counts.items())
        )

    print("✅ 特徵齊全，且沒有 NaN / inf")
    print("=" * 80 + "\n")


#                                                ************************************************************
#                                          ****************************  誤報統計與報表  ********************************
## ============================================================
# 干擾誤報歸屬：干擾觸發的警報不一定在干擾窗裡響
# ============================================================
# 干擾事件結束後多少秒內的報警，仍算干擾造成的誤報。
#   要涵蓋：規則本身的延遲（consecutive、min_leak_duration_sec，候選裡最慢約 1.2 秒）＋ 管路餘振
#   必須事先決定，不能看誤報率結果再調（調短一點，干擾誤報就會被藏到乾淨段）
#   設長的代價只是乾淨段少幾秒資料，所以寧可設長
DISTURBANCE_TAIL_SEC = 2.0


def _window_end_times(sub):
    '''每一窗的結束時間：t_ref 是窗中心，所以結束時間 = 2 × t_ref − t_start'''
    if "t_ref" not in sub.columns:
        raise ValueError("資料沒有 t_ref 欄位，無法算窗結束時間（干擾誤報歸屬需要）")
    return (2.0 * sub["t_ref"] - sub["t_start"]).to_numpy(dtype=float)


def attribute_disturbance_windows(df, tail_sec=DISTURBANCE_TAIL_SEC):
    '''
    特徵工程的 eval_class 是逐窗的事實：和干擾事件重疊 = disturbance_negative，其他 = clean_normal。
    這裡在「評估」時另外決定：每一窗的報警要算給誰。

    新增三個欄位：
      fa_bucket                    : "during"（和干擾重疊）/ "tail"（干擾結束後 tail_sec 秒內）/ "clean"（其他）
      dist_event_id                : 這一窗屬於哪一次干擾事件（during、tail 才有；連續的干擾窗算同一次）
      attributed_disturbance_type  : 這次干擾事件的類型（tail 窗也標上，特徵工程裡 tail 窗是空的）
    tail 判斷用「窗結束時間」，因為報警要等這一窗收完才會發出。
    '''
    if "eval_class" not in df.columns:
        raise ValueError("資料沒有 eval_class 欄位，無法做干擾誤報歸屬")

    # 歸屬只跟資料（eval_class、時間）有關，跟規則無關：已經算過、而且 tail 秒數相同，就直接沿用
    # （compare_disturbance_alarm_methods 會先算一次，之後 272 條規則都不用重算）
    if "fa_tail_sec" in df.columns and np.allclose(df["fa_tail_sec"].to_numpy(dtype=float), tail_sec):
        return df
    
    out = df.copy()
    out["fa_tail_sec"] = float(tail_sec)
    out["fa_bucket"] = "clean"
    out["dist_event_id"] = None
    out["attributed_disturbance_type"] = ""

    for rid, sub in out.groupby("record_id"):
        sub = sub.sort_values("t_start")
        idx = sub.index.to_numpy()
        is_dist = (sub["eval_class"] == "disturbance_negative").to_numpy()
        if not is_dist.any():
            continue
        t_end = _window_end_times(sub)
        d_type = sub["disturbance_type"].fillna("").astype(str).to_numpy() if "disturbance_type" in sub.columns \
            else np.array([""] * len(sub))

        bucket = np.array(["clean"] * len(sub), dtype=object)
        event_id = np.array([None] * len(sub), dtype=object)
        ev_type = np.array([""] * len(sub), dtype=object)

        k = -1                 # 目前是第幾次干擾事件
        in_event = False
        event_end = -np.inf    # 目前這次干擾最後一個干擾窗的結束時間
        cur_type = ""
        for i in range(len(sub)):
            if is_dist[i]:
                if not in_event:                     # 新的一次干擾開始
                    k += 1
                    in_event = True
                    types = d_type[i:][is_dist[i:]]  # 這次事件之後的干擾窗類型，取第一個非空的
                    cur_type = next((x for x in types if x), "")
                bucket[i] = "during"
                event_end = t_end[i]
            else:
                in_event = False
                if k >= 0 and t_end[i] <= event_end + tail_sec + 1e-9:
                    bucket[i] = "tail"
                else:
                    continue
            event_id[i] = f"{rid}__ev{k}"
            ev_type[i] = cur_type

        out.loc[idx, "fa_bucket"] = bucket
        out.loc[idx, "dist_event_id"] = event_id
        out.loc[idx, "attributed_disturbance_type"] = ev_type

    return out


def _alarm_onset_mask(df_rule):
    '''每一窗是不是「警報響起」的那一窗（每條 record 依時間排序，detected 0→1；第一窗就是 1 也算）'''
    mask = pd.Series(False, index=df_rule.index)
    for _, sub in df_rule.sort_values(["record_id", "t_start"]).groupby("record_id"):
        d = sub["detected_rule"].to_numpy(dtype=int)
        prev = np.r_[0, d[:-1]]
        mask.loc[sub.index] = (d == 1) & (prev == 0)
    return mask


# ============================================================
# disturbance 指標摘要
# ============================================================
KNOWN_EVAL_CLASSES = {"clean_normal", "disturbance_negative"}


def _count_alarm_onsets(df_rule):
    '''警報「響起」的次數：每條 record 依時間排序，detected_rule 從 0 變 1 算一次（第一窗就是 1 也算）'''
    return int(_alarm_onset_mask(df_rule).sum()) if len(df_rule) else 0


def summarize_disturbance_rule(df_rule, tail_sec=DISTURBANCE_TAIL_SEC):
    '''
    把 alarm rule 的誤報表現整理成幾個摘要數字。
    干擾資料裡每一窗都不該報警，所以「有報警的窗」全部都是誤報；這裡決定的是「誤報算給誰」。

    干擾造成的誤報 = 干擾中（during）＋ 干擾結束後 tail_sec 秒內（tail）的報警
      - disturbance_negative_false_alarm_rate：during＋tail 窗裡有報警的比例（選規則用的就是這個）
      - clean_normal_false_alarm_rate        ：離干擾夠遠的乾淨窗裡有報警的比例
      - disturbance_tail_false_alarm_rate    ：只看 tail（觀察有多少干擾誤報是「敲完才響」的）
      - disturbance_event_hit_rate           ：幾次干擾事件至少被報了一次（「敲 10 次管子，誤報幾次」）
      - *_alarm_onsets                       ：警報響起的次數，依「響起那一窗」歸屬
    '''
    # 干擾資料不該有洩漏窗：有的話，「報警 = 誤報」這個前提就不成立
    if "label" in df_rule.columns and (pd.to_numeric(df_rule["label"], errors="coerce") == 1).any():
        raise ValueError("干擾資料裡有 label=1 的窗，誤報率的前提（每一窗都不該報警）不成立，請檢查資料")

    onset = _alarm_onset_mask(df_rule)
    row = {
        "overall_false_alarm_rate": df_rule["detected_rule"].mean(),   # 全部窗（不分歸屬）
        "overall_alarm_onsets": int(onset.sum()),
    }

    if "eval_class" in df_rule.columns:
        unknown = set(df_rule["eval_class"].dropna().unique()) - KNOWN_EVAL_CLASSES
        if unknown:
            print(f"⚠️ eval_class 有未知的類別 {sorted(unknown)}：視為乾淨段")

        att = attribute_disturbance_windows(df_rule, tail_sec)
        bucket = att["fa_bucket"]
        dist_mask = bucket.isin(["during", "tail"])
        tail_mask = bucket == "tail"
        clean_mask = bucket == "clean"
        det = att["detected_rule"]

        row["disturbance_tail_sec"] = float(tail_sec)
        row["disturbance_negative_false_alarm_rate"] = det[dist_mask].mean() if dist_mask.any() else np.nan
        row["disturbance_tail_false_alarm_rate"] = det[tail_mask].mean() if tail_mask.any() else np.nan
        row["clean_normal_false_alarm_rate"] = det[clean_mask].mean() if clean_mask.any() else np.nan
        row["disturbance_negative_alarm_onsets"] = int((onset & dist_mask).sum())
        row["clean_normal_alarm_onsets"] = int((onset & clean_mask).sum())

        # 干擾事件層級：每次干擾（含結束後 tail_sec 秒）有沒有報過警
        ev = att[dist_mask]
        if len(ev):
            hit = ev.groupby("dist_event_id")["detected_rule"].max()
            row["disturbance_n_events"] = int(len(hit))
            row["disturbance_event_hit_rate"] = float(hit.mean())

        # 各干擾類型（during＋tail）
        for d_type, sub in ev[ev["attributed_disturbance_type"] != ""].groupby("attributed_disturbance_type"):
            row[f"fa_{d_type}"] = sub["detected_rule"].mean()

    return row


# ============================================================
# record-level summary：把 window-level 的資料，整理成 record-level 摘要
# ============================================================
def _unique_labels(sub, col):
    '''這條 record 裡這個欄位出現過的所有值，依出現順序用 + 串起來（沒有就空字串）'''
    if col not in sub.columns:
        return ""
    vals = [str(v) for v in pd.unique(sub[col].dropna()) if str(v) != ""]
    return "+".join(vals)


def build_record_level_summary(df_rule):
    '''
    每條 record 一列。
    報警時間一律用「窗結束時間」= t_start + 窗長（和轉換推論 onset 評估同一個定義：這一窗收完才算得出結果）。
    '''
    # 建立一個空 list, 等每處理完一個 record_id，就把那個 record 的摘要字典塞進去
    rows = []

    # 依 record_id 分組
    for rid, sub in df_rule.groupby("record_id"):
        sub = sub.sort_values("t_start")               # 先把這個 record 裡的 windows 依開始時間 t_start 排序
        detected = sub["detected_rule"].to_numpy(dtype=int)

        # 窗結束時間：t_ref 是窗中心，所以結束時間 = t_start + 2 × (t_ref − t_start)
        if "t_ref" in sub.columns:
            t_end = (2.0 * sub["t_ref"] - sub["t_start"]).to_numpy(dtype=float)
        else:
            t_end = sub["t_start"].to_numpy(dtype=float)     # 沒有 t_ref 時只能用開始時間（會早一個窗長）
        alarm_idx = np.flatnonzero(detected == 1)

        rows.append({
            "record_id": rid,
            # 一條 record 裡可能同時有乾淨段和干擾段：全部列出來（例如 clean_normal+disturbance_negative），不只取第一個
            "disturbance_type": _unique_labels(sub, "disturbance_type"),
            "eval_class": _unique_labels(sub, "eval_class"),
            "n_windows": int(len(sub)),
            "max_pred_proba": sub["pred_proba"].max(),              # 模型曾經有多相信這筆資料像 leak
            "max_smoothed_proba": sub["smoothed_proba_rule"].max(), # 平滑後機率的最大值
            "any_alarm": int(len(alarm_idx) > 0),                   # 是否曾經至少報警一次
            "first_alarm_time": float(t_end[alarm_idx[0]]) if len(alarm_idx) else np.nan,   # 第一次報警（窗結束時間）
            "n_alarm_onsets": _count_alarm_onsets(sub),             # 警報響了幾次
            "n_alarm_windows": int(detected.sum()),                 # alarm window 的個數
            "alarm_ratio": float(detected.mean()) if len(detected) else np.nan,  # alarm 比例
        })

    return pd.DataFrame(rows)


# ============================================================
# before / during / tail / after disturbance
# ============================================================
def build_phase_summary(df_rule, tail_sec=DISTURBANCE_TAIL_SEC):
    '''
    把有干擾的 record 依干擾事件切成四段（和誤報歸屬同一套定義）：
      before_disturbance : 第一次干擾之前的乾淨窗
      during_disturbance : 和干擾重疊的窗
      tail_disturbance   : 干擾結束後 tail_sec 秒內（報警仍算干擾誤報）
      after_disturbance  : 干擾結束 tail_sec 秒之後的乾淨窗（報警算乾淨段誤報）
    看 tail 的 mean_pred_proba，就知道敲完後管路餘振多久模型機率才降下來，可以回頭檢查 tail_sec 夠不夠長。
    '''
    att = attribute_disturbance_windows(df_rule, tail_sec)
    rows = []

    for rid, sub in att.groupby("record_id"):
        sub = sub.sort_values("t_start").copy()
        bucket = sub["fa_bucket"].to_numpy()
        if not (bucket == "during").any():     # 這條 record 沒有干擾事件：跳過
            continue

        first_during = int(np.flatnonzero(bucket == "during")[0])
        pos = np.arange(len(sub))
        sub["phase"] = np.select(
            [bucket == "during", bucket == "tail", pos < first_during],
            ["during_disturbance", "tail_disturbance", "before_disturbance"],
            default="after_disturbance",
        )
        record_type = _unique_labels(sub, "attributed_disturbance_type")   # 整條 record 的干擾類型，四段都標

        for phase_name, part in sub.groupby("phase"):
            rows.append({
                "record_id": rid,
                "disturbance_type": record_type,
                "phase": phase_name,
                "n_windows": int(len(part)),
                "mean_pred_proba": part["pred_proba"].mean(),     # 模型原始 leak 機率的平均值
                "alarm_rate": part["detected_rule"].mean(),       # 被規則報警的比例
                "max_pred_proba": part["pred_proba"].max(),       # 模型曾經給出的最高 leak 機率
            })

    return pd.DataFrame(rows)

# ============================================================
# disturbance-type ranking 干擾類型層級的摘要
# ============================================================
def build_disturbance_type_summary(df_rule, tail_sec=DISTURBANCE_TAIL_SEC):
    '''
    依干擾類型彙整誤報成績（診斷用，不影響規則）：規則只有一條，現場也不知道干擾類型；
    這張表是用來看誤報是「每一類都有」（通用問題，靠持續時間 / 負樣本訓練處理）
    還是「集中在某一類」（找出它和洩漏在訊號上的物理差別），不是替每種干擾各做一套規則。
    和 summarize_disturbance_rule 用同一套歸屬：干擾中＋干擾結束後 tail_sec 秒內的報警，都算這次干擾的。

    每次干擾事件（連續的干擾窗）是一個單位：
      event_hit_rate         ：這種干擾有幾成的事件至少被報了一次（「敲 10 次，誤報幾次」）
      tail_alarm_share       ：誤報窗裡有幾成落在干擾結束之後（高 = 多半是餘振或規則延遲造成的）
      median_alarm_delay_sec ：被誤報的事件，從干擾開始到第一次報警（窗結束時間）隔了幾秒
    '''
    att = attribute_disturbance_windows(df_rule, tail_sec)
    att = att[att["fa_bucket"].isin(["during", "tail"]) & (att["attributed_disturbance_type"] != "")].copy()
    if att.empty:
        return pd.DataFrame()
    att["t_end"] = 2.0 * att["t_ref"] - att["t_start"]

    # 每次干擾事件一列：有沒有被報、從干擾開始到第一次報警幾秒
    ev_rows = []
    for ev_id, ev in att.sort_values("t_start").groupby("dist_event_id"):
        ev_start = ev.loc[ev["fa_bucket"] == "during", "t_start"].min()
        alarm = ev[ev["detected_rule"] == 1]
        ev_rows.append({
            "dist_event_id": ev_id,
            "disturbance_type": ev["attributed_disturbance_type"].iloc[0],
            "hit": int(len(alarm) > 0),
            "delay_sec": float(alarm["t_end"].iloc[0] - ev_start) if len(alarm) else np.nan,
        })
    ev_df = pd.DataFrame(ev_rows)

    rows = []
    for d_type, sub in att.groupby("attributed_disturbance_type"):
        evs = ev_df[ev_df["disturbance_type"] == d_type]
        n_alarm = int(sub["detected_rule"].sum())
        delays = evs["delay_sec"].dropna()
        rows.append({
            "disturbance_type": d_type,
            "n_events": int(len(evs)),                                    # 這種干擾發生幾次
            "n_records": int(sub["record_id"].nunique()),                 # 分布在幾個 session
            "n_windows": int(len(sub)),                                   # 干擾中＋tail 的窗數
            "raw_false_alarm_rate": sub["pred_label"].mean() if "pred_label" in sub.columns else np.nan,   # 模型原始門檻的誤報窗比例
            "rule_false_alarm_rate": sub["detected_rule"].mean(),         # 套規則後的誤報窗比例
            "event_hit_rate": float(evs["hit"].mean()) if len(evs) else np.nan,
            "tail_alarm_share": float(sub.loc[sub["fa_bucket"] == "tail", "detected_rule"].sum() / n_alarm) if n_alarm else np.nan,
            "median_alarm_delay_sec": float(delays.median()) if len(delays) else np.nan,
        })

    return (pd.DataFrame(rows)
            .sort_values(["event_hit_rate", "rule_false_alarm_rate"], ascending=False)
            .reset_index(drop=True))


# ============================================================
# 整理大摘要表
# ============================================================
def export_disturbance_summary_tables(df_rule, output_dir, rule_name="baseline_rule", tail_sec=DISTURBANCE_TAIL_SEC):
    '''
    輸出三張診斷用的表（不影響規則）。和 summarize_disturbance_rule 用同一套歸屬：
    干擾中（during）＋干擾結束後 tail_sec 秒內（tail）的報警算干擾的，其他算乾淨段的。
      {rule_name}_eval_class_summary.csv              ：依 during / tail / clean 三類
      {rule_name}_disturbance_type_strength_summary.csv：依干擾類型 × 強度（tail 窗也算進那次干擾的類型、強度）
      {rule_name}_false_alarm_cases.csv               ：每一個報警窗，附上它算給誰
    （依干擾類型的摘要改由 build_disturbance_type_summary 輸出，這裡不再重複一份）
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    att = attribute_disturbance_windows(df_rule, tail_sec)

    # tail 窗在特徵工程裡沒有強度：用同一次干擾事件的干擾窗補上
    if "disturbance_strength" in att.columns:
        during = att[att["fa_bucket"] == "during"]
        ev_strength = (during.assign(_s=during["disturbance_strength"].fillna("").astype(str))
                       .groupby("dist_event_id")["_s"]
                       .agg(lambda s: next((x for x in s if x), "")))
        att["attributed_disturbance_strength"] = att["dist_event_id"].map(ev_strength).fillna("")
    else:
        att["attributed_disturbance_strength"] = ""

    # 1) 依歸屬分類摘要（during / tail / clean）
    eval_class_summary = (
        att.groupby("fa_bucket")
        .agg(
            n_windows=("record_id", "count"),
            false_alarm_rate=("detected_rule", "mean"),
            mean_pred_proba=("pred_proba", "mean"),
            mean_smoothed_proba=("smoothed_proba_rule", "mean"),
        )
        .reindex(["during", "tail", "clean"])
        .dropna(how="all")
        .reset_index()
    )
    eval_class_summary.insert(1, "tail_sec", float(tail_sec))

    # 2) 依干擾類型 × 強度摘要（只看 during＋tail）
    dist = att[att["fa_bucket"].isin(["during", "tail"]) & (att["attributed_disturbance_type"] != "")]
    if not dist.empty:
        type_strength_summary = (
            dist.groupby(["attributed_disturbance_type", "attributed_disturbance_strength"])
            .agg(
                n_events=("dist_event_id", "nunique"),
                n_windows=("record_id", "count"),
                false_alarm_rate=("detected_rule", "mean"),
                mean_pred_proba=("pred_proba", "mean"),
                mean_smoothed_proba=("smoothed_proba_rule", "mean"),
            )
            .reset_index()
            .rename(columns={"attributed_disturbance_type": "disturbance_type",
                             "attributed_disturbance_strength": "disturbance_strength"})
            .sort_values(["disturbance_type", "disturbance_strength"])
        )
    else:
        type_strength_summary = pd.DataFrame()

    # 3) false alarm cases：每一個報警窗，附上它算給誰（during / tail / clean、哪一次干擾）
    keep_cols = [
        c for c in [
            "record_id", "t_start", "t_ref",
            "eval_class", "fa_bucket", "dist_event_id",
            "attributed_disturbance_type", "attributed_disturbance_strength",
            "pred_proba", "smoothed_proba_rule", "detected_rule"
        ]
        if c in att.columns
    ]
    false_alarm_cases = att.loc[att["detected_rule"] == 1, keep_cols].copy()

    # 輸出
    eval_class_summary.to_csv(output_dir / f"{rule_name}_eval_class_summary.csv",
                              index=False, encoding="utf-8-sig")
    if not type_strength_summary.empty:
        type_strength_summary.to_csv(output_dir / f"{rule_name}_disturbance_type_strength_summary.csv",
                                     index=False, encoding="utf-8-sig")
    false_alarm_cases.to_csv(output_dir / f"{rule_name}_false_alarm_cases.csv",
                             index=False, encoding="utf-8-sig")

    print(f"已輸出額外干擾摘要表: {rule_name}")


# ============================================================
# 比較哪一次干擾最難辨識（每次干擾事件一列）
# ============================================================
def build_disturbance_id_summary(df_rule, tail_sec=DISTURBANCE_TAIL_SEC):
    '''
    找出「哪一次干擾」最難分辨，而不只是哪一類型平均比較難。

    單位是 attribute_disturbance_windows 切出來的干擾事件（連續的干擾窗 = 一次干擾），
    不是 manifest 的 disturbance_id：那個是每支檔 / 每個 session 一個值，一個 session 敲好幾次也只有一個，
    還會把整段乾淨背景一起算進去。manifest 的 disturbance_id 仍保留在表上，方便對回原始紀錄。
    誤報範圍和 summarize_disturbance_rule 相同：干擾中＋干擾結束後 tail_sec 秒內。
    '''
    att = attribute_disturbance_windows(df_rule, tail_sec)
    att = att[att["fa_bucket"].isin(["during", "tail"])].copy()
    if att.empty:
        return pd.DataFrame()
    att["t_end"] = 2.0 * att["t_ref"] - att["t_start"]

    rows = []
    for ev_id, ev in att.sort_values("t_start").groupby("dist_event_id"):
        during = ev[ev["fa_bucket"] == "during"]
        alarm = ev[ev["detected_rule"] == 1]
        ev_start = float(during["t_start"].min())
        n_alarm = int(len(alarm))

        rows.append({
            "dist_event_id": ev_id,                                            # 這次干擾事件（record + 第幾次）
            "record_id": ev["record_id"].iloc[0],
            "disturbance_type": ev["attributed_disturbance_type"].iloc[0],
            "disturbance_strength": _unique_labels(during, "disturbance_strength"),
            "manifest_disturbance_id": _unique_labels(during, "disturbance_id"),   # manifest 上的 ID，對回原始紀錄用
            "event_start_sec": ev_start,                                       # 干擾開始（session 時間）
            "event_end_sec": float(during["t_end"].max()),                     # 最後一個干擾窗的結束時間
            "n_during_windows": int(len(during)),
            "n_tail_windows": int((ev["fa_bucket"] == "tail").sum()),
            "n_files": int(ev["filename_stem"].nunique()) if "filename_stem" in ev.columns else np.nan,
            "hit": int(n_alarm > 0),                                           # 這次干擾有沒有被誤報
            "raw_false_alarm_rate": ev["pred_label"].mean() if "pred_label" in ev.columns else np.nan,
            "rule_false_alarm_rate": ev["detected_rule"].mean(),
            "n_alarm_windows": n_alarm,
            "tail_alarm_share": float((alarm["fa_bucket"] == "tail").mean()) if n_alarm else np.nan,   # 誤報有幾成是敲完才響
            "first_alarm_delay_sec": float(alarm["t_end"].iloc[0] - ev_start) if n_alarm else np.nan, # 干擾開始到第一次報警
            "max_pred_proba_during": float(during["pred_proba"].max()),       # 干擾中模型最高機率
            "max_pred_proba_tail": float(ev.loc[ev["fa_bucket"] == "tail", "pred_proba"].max())
                                   if (ev["fa_bucket"] == "tail").any() else np.nan,   # 敲完後餘振時的最高機率
        })

    return (pd.DataFrame(rows)
            .sort_values(["hit", "rule_false_alarm_rate", "max_pred_proba_during"], ascending=False)
            .reset_index(drop=True))


#                                                ************************************************************
#                                          ****************************  抗干擾規則比較  ********************************
# ============================================================
# 從資料估計窗長與步長
# ============================================================
def estimate_window_timing(df):
    """
    回傳 (window_sec, hop_sec)
      - 步長 hop_sec：同一 record 內相鄰窗 t_start 差的中位數（0.05 秒）
      - 窗長 window_sec：t_ref 是窗中心，(t_ref − t_start) × 2 就是窗長（0.20 秒）
    """
    diffs = (
        df.sort_values(["record_id", "t_start"])
        .groupby("record_id")["t_start"]
        .diff()
        .dropna()
    )
    if diffs.empty:
        raise ValueError("每個 record 都只有 1 個窗，無法估計步長")
    hop_sec = float(diffs.median())
    window_sec = float(((df["t_ref"] - df["t_start"]) * 2.0).median())
    return window_sec, hop_sec


# ============================================================
# 每個 window 所在「連續片段」有多長
# ============================================================
def compute_segment_span(df):
    """
    用 alarm_rule_utils.split_continuous_segments 把每條 record 依真正的空檔切段，
    回傳和 df 同 index 的 Series：這個 window 所在那一段連續資料，總共涵蓋幾秒。

    例：舊版 Disturbance 一支 0.5 秒的檔案切出 7 個窗，t_start 0.00~0.30，
        片段長度 = 0.30 − 0.00 + 窗長 0.20 = 0.50 秒
        新版連續資料則是整段錄製時間（例如 60 秒）
    """
    window_sec, _ = estimate_window_timing(df)
    span = pd.Series(np.nan, index=df.index)
    for _, rec in df.groupby("record_id"):
        rec = rec.sort_values("t_start")
        for seg_index in split_continuous_segments(rec):
            t = df.loc[seg_index, "t_start"]
            span.loc[seg_index] = float(t.max() - t.min()) + window_sec
    return span


# ============================================================
# 一條規則「最快」要多少秒的連續資料才可能報警
# ============================================================
def rule_min_alarm_latency_sec(cfg, window_threshold, window_sec, hop_sec, n_windows=400):
    """
    做法：造一段「每個窗機率都是 1.0」的假資料，直接丟進正式的告警引擎，
    看第一次報警出現在哪一窗。這就是這條規則在最理想情況下的最短反應時間。

    回傳「從資料開始到第一次報警那一窗結束」的秒數；永遠不會報警就回傳 inf。
    （veto 只會讓報警變少、不會變快，所以這裡關掉 veto，不需要 veto 欄位。）

    用途：資料的連續片段比這個秒數短，這條規則在那段資料上「不可能」報警，
    誤報率 0 只是因為資料太短，不代表規則真的擋得住干擾。
    """
    cfg_no_veto = copy.deepcopy(cfg)
    cfg_no_veto["use_veto"] = False

    t = np.arange(n_windows) * hop_sec
    df_syn = pd.DataFrame({
        "record_id": "_synthetic",
        "t_start": t,
        "t_ref": t + window_sec / 2.0,
        "pred_proba": 1.0,
    })
    out = apply_alarm_method_by_record(df_syn, cfg_no_veto, window_threshold)
    hit = np.flatnonzero(out["detected_rule"].to_numpy() == 1)
    if len(hit) == 0:
        return np.inf
    return float(t[hit[0]] + window_sec)


# ============================================================
# 建立所有候選規則
# ============================================================
def build_candidate_rule_configs(df_pred):
    """
    候選規則 =「基礎規則」×「min_leak_duration_sec 秒數」
    每條展開後的規則都帶 base_rule 欄位，記錄它是從哪條基礎規則展開的，
    後面每種基礎規則才能各挑一個代表給轉換推論。
    """
    duration_candidates = [round(x, 2) for x in np.arange(0.20, 1.05, 0.05)]

    def expand_duration_grid(base_rules):
        expanded = []
        for base in base_rules:
            for d in duration_candidates:
                cfg = copy.deepcopy(base)                 # 深複製：veto_rules 等內層 list/dict 也各自獨立
                cfg["base_rule"] = base["name"]           # 從哪條基礎規則展開
                cfg["name"] = f'{base["name"]}_d{d:.2f}'  # 展開後的完整名稱
                cfg["min_leak_duration_sec"] = d
                expanded.append(cfg)
        return expanded

    # --- 基礎規則：由寬鬆到嚴格都要有，否則排名永遠只會選到最嚴格的 ---
    base_rules = [
        # 寬鬆組（門檻 0.70）
        {"name": "threshold_s3_t0.70", "method": "threshold",
         "alarm_threshold": 0.70, "smooth_window": 3},
        {"name": "consecutive_s3_t0.70_k2", "method": "consecutive",
         "alarm_threshold": 0.70, "smooth_window": 3, "consecutive": 2},
        {"name": "consecutive_s3_t0.70_k3", "method": "consecutive",
         "alarm_threshold": 0.70, "smooth_window": 3, "consecutive": 3},
        {"name": "n_of_m_s3_t0.70_3of5", "method": "n_of_m",
         "alarm_threshold": 0.70, "smooth_window": 3, "n_hits": 3, "m_window": 5},
        {"name": "hysteresis_s3_hi0.70_lo0.50", "method": "hysteresis",
         "smooth_window": 3, "high_threshold": 0.70, "low_threshold": 0.50},

        # 嚴格組（門檻 0.80 / 0.85）
        {"name": "consecutive_s3_t0.80_k3", "method": "consecutive",
         "alarm_threshold": 0.80, "smooth_window": 3, "consecutive": 3},
        {"name": "consecutive_s3_t0.85_k3", "method": "consecutive",
         "alarm_threshold": 0.85, "smooth_window": 3, "consecutive": 3},
        {"name": "consecutive_s3_t0.80_k4", "method": "consecutive",
         "alarm_threshold": 0.80, "smooth_window": 3, "consecutive": 4},
        {"name": "n_of_m_s3_t0.80_4of6", "method": "n_of_m",
         "alarm_threshold": 0.80, "smooth_window": 3, "n_hits": 4, "m_window": 6},
        {"name": "hysteresis_s3_hi0.80_lo0.60", "method": "hysteresis",
         "smooth_window": 3, "high_threshold": 0.80, "low_threshold": 0.60},
    ]
    # two_stage 不放進候選：它要等 Stage 1 那段警報「結束」才知道要不要用 Stage 2 重判，
    # 即時系統做不到（等於偷看未來）。alarm_rule_utils 也已不支援，設定檔寫 two_stage 會直接報錯。
    configs = expand_duration_grid(base_rules)

    # --- veto 規則：主規則都是 consecutive_s3_t0.80_k4，只差 veto 條件 ---
    # 只保留「已經相對固定 normal baseline 標準化」的欄位（z 值、比值），
    # 不再用 offband_E_mean 這種原始能量的絕對門檻（1e-4…），因為它的大小隨增益、距離、感測器而變，
    # 寫死的數字在新資料上沒有意義。
    # veto 會不會把真正的洩漏也擋掉，要到轉換推論用洩漏資料才看得出來，這裡只負責提出候選。
    veto_main = {"method": "consecutive", "alarm_threshold": 0.80,
                 "smooth_window": 3, "consecutive": 4, "use_veto": True}

    if {"z_offband_E_mean", "z_offband_E_max"}.issubset(df_pred.columns):
        zveto_levels = {"lo": (2.0, 2.5), "mid": (2.5, 3.0), "hi": (3.0, 4.0)}
        zveto_rules = [
            {**veto_main,
             "name": f"consecutive_s3_t0.80_k4_zveto_{level}",
             "veto_rules": [
                 {"col": "z_offband_E_mean", "op": "ge", "threshold": thr_mean},
                 {"col": "z_offband_E_max", "op": "ge", "threshold": thr_max},
             ]}
            for level, (thr_mean, thr_max) in zveto_levels.items()
        ]
        configs.extend(expand_duration_grid(zveto_rules))
    else:
        print("⚠️ 缺少 z_offband_E_mean / z_offband_E_max，跳過 zveto 規則")

    ratio_col = "offbandmax_to_leak_focusmax_E_ratio"
    if ratio_col in df_pred.columns:
        ratio_levels = {"lo": 1.5, "mid": 2.0, "hi": 3.0}
        ratio_rules = [
            {**veto_main,
             "name": f"consecutive_s3_t0.80_k4_maxratio_veto_{level}",
             "veto_rules": [{"col": ratio_col, "op": "ge", "threshold": thr}]}
            for level, thr in ratio_levels.items()
        ]
        configs.extend(expand_duration_grid(ratio_rules))
    else:
        print(f"⚠️ 缺少 {ratio_col}，跳過 ratio veto 規則")

    # 名稱不能重複（後面用名稱找規則）
    names = [c["name"] for c in configs]
    if len(names) != len(set(names)):
        raise ValueError("候選規則名稱有重複")
    return configs


# ============================================================
# 比較多組規則
# ============================================================
def compare_disturbance_alarm_methods(
    df_pred,                    # 已有 pred_proba 的資料（只放「選規則用」的那一份）
    window_threshold,           # 模型的 window 門檻（告警引擎的介面需要；規則門檻寫在各規則裡）
    output_dir=None,            # 有給就輸出總比較表
    min_testable_ratio=0.5,     # 干擾窗中，至少要有這個比例落在「夠長、規則有機會報警」的片段，誤報率才算數
):
    """
    把每條候選規則套到同一份 pred_proba 上，整理成一張比較表。

    除了誤報率，每條規則多記這幾欄：
      - min_alarm_latency_sec：這條規則最快要多少秒的連續資料才可能報警
      - fa_testable_ratio：干擾窗中，所在連續片段長度 ≥ 上面那個秒數的比例
        → 比例太低，代表這份資料太短，規則根本沒機會報警，誤報率 0 不可信
      - fa_testable：fa_testable_ratio ≥ min_testable_ratio
      - bridged_*：把空檔當成不存在再算一次的誤報率（保守參考值，不放進排序）

    排序：先看測不測得出來，再看干擾誤報、乾淨正常誤報，同分時反應快的優先。
    這張表只是「干擾這一面」的成績，正式選規則要到轉換推論，和偵測率一起看。

    回傳：
      - summary_df：每條規則一列
      - config_by_name：規則名稱 → 完整規則設定（主流程要重算某條規則時用）
    """
    configs = build_candidate_rule_configs(df_pred)
    window_sec, hop_sec = estimate_window_timing(df_pred)
    segment_span = compute_segment_span(df_pred)

    # 干擾誤報歸屬先算一次（跟規則無關），後面每條規則套完都直接沿用，不用重算
    if "eval_class" in df_pred.columns:
        df_pred = attribute_disturbance_windows(df_pred)

    # 評估「測不測得出來」只看算給干擾的窗（干擾中＋干擾後 tail 秒）；沒有 eval_class 就看全部
    if "fa_bucket" in df_pred.columns and df_pred["fa_bucket"].isin(["during", "tail"]).any():
        dist_mask = df_pred["fa_bucket"].isin(["during", "tail"])
    else:
        dist_mask = pd.Series(True, index=df_pred.index)
        
    dist_span = segment_span[dist_mask]

    print(f"窗長 {window_sec:.3f} 秒、步長 {hop_sec:.3f} 秒；"
          f"干擾資料連續片段長度：中位數 {dist_span.median():.2f} 秒、最長 {dist_span.max():.2f} 秒")
    print(f"候選規則數：{len(configs)}")

    summary_rows = []
    for cfg in configs:
        df_rule = apply_alarm_method_by_record(df_pred, cfg, window_threshold)

        latency = rule_min_alarm_latency_sec(cfg, window_threshold, window_sec, hop_sec)
        testable_ratio = float((dist_span >= latency - 1e-9).mean())

        row = {
            "name": cfg["name"],
            "base_rule": cfg["base_rule"],
            "method": cfg["method"],
            "window_threshold": window_threshold,
            "alarm_threshold": cfg.get("alarm_threshold", np.nan),
            "smooth_window": cfg.get("smooth_window", np.nan),
            "consecutive": cfg.get("consecutive", np.nan),
            "n_hits": cfg.get("n_hits", np.nan),
            "m_window": cfg.get("m_window", np.nan),
            "high_threshold": cfg.get("high_threshold", np.nan),
            "low_threshold": cfg.get("low_threshold", np.nan),
            "min_leak_duration_sec": cfg["min_leak_duration_sec"],

            "use_veto": cfg.get("use_veto", False),
            "veto_column": cfg.get("veto_column", None),
            "veto_threshold": cfg.get("veto_threshold", np.nan),
            "veto_rules": cfg.get("veto_rules", []),

            # 這條規則在這份資料上測不測得出來
            "min_alarm_latency_sec": latency,
            "fa_testable_ratio": testable_ratio,
            "fa_testable": testable_ratio >= min_testable_ratio,

            # veto 效果
            "veto_rate": df_rule["veto_flag"].mean(),                     # 被 veto 的窗比例
            "raw_alarm_rate": df_rule["detected_rule_raw"].mean(),        # 不套 veto 的報警比例
            "final_alarm_rate": df_rule["detected_rule"].mean(),          # 套 veto 後的報警比例
        }
        row["alarm_drop"] = row["raw_alarm_rate"] - row["final_alarm_rate"]   # veto 壓掉多少

        row.update(summarize_disturbance_rule(df_rule))                   # 各類誤報率（含 fa_<干擾類型>）

        # 參考值：把空檔當成不存在，計數和平滑延續過去（假設干擾在空檔期間也持續）
        # 特徵還是各檔各自算的，所以不會有接縫假訊號；通常比「切開」版偏高、偏保守
        # 只當參考，不放進排序；資料本身連續時，兩者會一樣
        df_bridged = apply_alarm_method_by_record(df_pred, cfg, window_threshold, max_gap_sec=np.inf)
        bridged = summarize_disturbance_rule(df_bridged)
        row["bridged_disturbance_negative_false_alarm_rate"] = bridged.get(
            "disturbance_negative_false_alarm_rate", np.nan)
        row["bridged_overall_false_alarm_rate"] = bridged["overall_false_alarm_rate"]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
     
    # 干擾誤報歸屬的前提：干擾觸發的警報最晚在干擾結束後 DISTURBANCE_TAIL_SEC 秒內出現。
    # 規則本身最快就要超過這個秒數才會報的話，它被干擾觸發的警報可能落在 tail 之外、被算成乾淨段誤報
    too_slow = summary_df.loc[summary_df["min_alarm_latency_sec"] > DISTURBANCE_TAIL_SEC, "name"].tolist()
    if too_slow:
        print(f"⚠️ {len(too_slow)} 條規則最快報警時間超過 DISTURBANCE_TAIL_SEC = {DISTURBANCE_TAIL_SEC} 秒，"
              f"干擾誤報可能被低估（請加長 DISTURBANCE_TAIL_SEC）：{too_slow[:5]}")
        
    # 排序：測得出來的在前 → 干擾誤報低 → 乾淨正常誤報低 → 反應快
    sort_cols, ascending = ["fa_testable"], [False]
    for col in ["disturbance_negative_false_alarm_rate", "clean_normal_false_alarm_rate", "overall_false_alarm_rate"]:
        if col in summary_df.columns:
            sort_cols.append(col)
            ascending.append(True)
    sort_cols.append("min_alarm_latency_sec")
    ascending.append(True)
    summary_df = summary_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    n_untestable = int((~summary_df["fa_testable"]).sum())
    if n_untestable:
        print(f"⚠️ {n_untestable} 條規則在這份干擾資料上測不出來（片段太短，規則來不及報警），"
              f"誤報率 0 不代表擋得住干擾，請參考 bridged_* 欄位")

    if output_dir is not None:
        summary_df.to_csv(Path(output_dir) / "disturbance_alarm_method_comparison.csv",
                          index=False, encoding="utf-8-sig")

    config_by_name = {cfg["name"]: cfg for cfg in configs}
    return summary_df, config_by_name


#                                                ************************************************************
#                                          ****************************  正式輸出／設定檔  ********************************
# ============================================================
# 儲存後處理較佳規則成 postprocess_config
# ============================================================  
def save_postprocess_configs(
    model_folder,                                  # JSON 要存放的資料夾
    model_id,                                      # 模型版本
    summary_df,                                    # 各種後處理規則的比較結果表
    selected_rule_name=None,                       # 人工指定要正式採用的規則名稱
    source="disturbance_dev_rule_selection",       # 記錄規則結果來自哪個流程：干擾規則在 dev 上選出來的（轉換推論會檢查這個值）
    write_final=False,                             # 是否真的寫出正式規則:預設 False,代表預設不會寫出正式版 postprocess_config.json
    max_disturbance_fa=0.01,                       # 干擾誤報率的上限，例如 1%
):
    '''
    把干擾規則比較表整理成候選設定檔，交給轉換推論挑選。

    挑選方式：每一種「基礎規則」（base_rule）挑一個代表
      1. 有效誤報率：測得出來的規則用一般誤報率；測不出來的（舊資料片段太短）改用橋接誤報率
      2. 有效誤報率 ≤ max_disturbance_fa 的版本裡，挑 min_leak_duration_sec 最短的（反應最快）
      3. 全部都超過上限：挑有效誤報率最低的，並標 meets_fa_limit=False

    輸出：
      - postprocess_config_candidate_top{名次}_{規則名}.json：每個代表一份
      - postprocess_config.json：只有 write_final=True 才寫
    寫之前會先刪掉舊的 top*.json，避免轉換推論讀到上一次留下的候選。
    '''
    
    # 建資料夾、檢查輸入
    model_folder = Path(model_folder)                   # 先把 model_folder 轉成 Path
    model_folder.mkdir(parents=True, exist_ok=True)     # 確保這個資料夾都存在,資料夾已存在也不會報錯
    
    # 檢查規則比較表是否可用：檢查 summary_df——如果是 None 或是空的 DataFrame（沒有任何規則可比較）,就直接擋下來
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df 為空，無法輸出 postprocess config")
    
    # 每條候選規則至少要提供的資訊：summary_df 必要欄位檢查
    required_cols = {"name",                     # 規則名稱
                     "base_rule",                # 規則家族，例如連續幾窗超門檻、滑動平均、N-of-M
                     "method", 
                     "window_threshold",     
                     "min_leak_duration_sec",    # 最短持續多久才會告警
                     "fa_testable",              # 目前資料是否足以直接量到誤報率
                     "min_alarm_latency_sec"}    # 最短告警延遲
    missing_cols = required_cols - set(summary_df.columns)
    if missing_cols:
        raise ValueError(f"summary_df 缺少必要欄位: {missing_cols}")
    
    # --- 有效誤報率：測得出來用一般值，測不出來用橋接值（保守） ---
    # 優先使用針對干擾負樣本的誤報率
    fa_col = "disturbance_negative_false_alarm_rate"
    bridged_col = "bridged_disturbance_negative_false_alarm_rate"
    
    # 若資料沒有區分 clean_normal 與 disturbance_negative，才退回使用整體誤報率
    if fa_col not in summary_df.columns:          # 沒有 eval_class 時退回整體誤報率
        fa_col, bridged_col = "overall_false_alarm_rate", "bridged_overall_false_alarm_rate"
    df = summary_df.copy()
    
    # fa_testable=True：資料夠長，直接使用實測誤報率
    # fa_testable=False：資料片段太短，無法直接量測長時間規則的誤報率，因此使用橋接估計值 bridged_*
    df["effective_fa"] = np.where(
        df["fa_testable"],      
        df[fa_col], 
        df[bridged_col])
    # 記錄有效誤報率是直接測得，還是橋接估計，讓後續結果可追溯。
    df["effective_fa_source"] = np.where(df["fa_testable"], "measured", "bridged")
    # 標記是否符合干擾誤報率上限
    df["meets_fa_limit"] = df["effective_fa"] <= max_disturbance_fa

    # 各干擾類型的誤報率欄位：fa_<類型>，依資料自動產生（排除 fa_testable 這兩個非誤報率欄位）
    fa_type_cols = [c for c in df.columns
                    if c.startswith("fa_") and c not in ("fa_testable", "fa_testable_ratio")]

    # --- 每種基礎規則挑一個代表(連續超門檻規則 / 滑動平均規則 / N-of-M 規則) ---
    reps = []
    for base, grp in df.groupby("base_rule", sort=False):
        # 先找出誤報率符合上限的版本
        ok = grp[grp["meets_fa_limit"]]
        # 若有通過誤報率上限的版本，挑：
        # 1. min_leak_duration_sec 最短：最快發出告警；
        # 2. 若持續時間相同，再挑 effective_fa 較低者。
        if not ok.empty:
            pick = ok.sort_values(["min_leak_duration_sec", "effective_fa"]).iloc[0]
        # 若同一家族所有版本都不符合誤報率上限，仍保留誤報率最低的版本作為候選，但會標記
        else:
            pick = grp.sort_values(["effective_fa", "min_leak_duration_sec"]).iloc[0]
        reps.append(pick)
    
    # 將各規則家族代表再排序，排名邏輯是(先安全，再低誤報，最後才比速度。)：
    # 1. 先排「符合誤報率上限」的規則；
    # 2. 再排有效誤報率低的；
    # 3. 最後排告警延遲短的。
    reps = (pd.DataFrame(reps)
            .sort_values(["meets_fa_limit", "effective_fa", "min_alarm_latency_sec"],
                         ascending=[False, True, True])
            .reset_index(drop=True))

    print("\n" + "=" * 80)
    print(f"每種規則的代表（干擾誤報上限 {max_disturbance_fa:.1%}）")
    print("=" * 80)
    print(reps[["name", "min_leak_duration_sec", "min_alarm_latency_sec",
                "effective_fa", "effective_fa_source", "meets_fa_limit"]].to_string(index=False))
    n_fail = int((~reps["meets_fa_limit"]).sum())
    if n_fail:
        print(f"⚠️ {n_fail} 種規則所有秒數都超過誤報上限，仍列為候選但標 meets_fa_limit=False")
    

    # --- 將單條規則轉成正式設定格式：把一列整理成正式設定檔需要的欄位 ---
    def build_baseline_rule_settings(selected_row):
        settings = {
            "name": selected_row.get("name"),
            "base_rule": selected_row.get("base_rule"),
            "method": selected_row.get("method"),
            "window_threshold": selected_row.get("window_threshold"),
            "alarm_threshold": selected_row.get("alarm_threshold"),
            "smooth_window": selected_row.get("smooth_window"),
            "consecutive": selected_row.get("consecutive"),
            "n_hits": selected_row.get("n_hits"),
            "m_window": selected_row.get("m_window"),
            "high_threshold": selected_row.get("high_threshold"),
            "low_threshold": selected_row.get("low_threshold"),
            "min_leak_duration_sec": selected_row.get("min_leak_duration_sec"),

            # 抗干擾 veto 規則的設定
            "use_veto": bool(selected_row.get("use_veto", False)),
            "veto_column": selected_row.get("veto_column"),
            "veto_threshold": selected_row.get("veto_threshold"),
            "veto_rules": selected_row.get("veto_rules") or [],

            # 干擾這一面的成績（轉換推論會和偵測率放在同一張表）
            "min_alarm_latency_sec": selected_row.get("min_alarm_latency_sec"),
            "fa_testable": selected_row.get("fa_testable"),
            "effective_fa": selected_row.get("effective_fa"),
            "effective_fa_source": selected_row.get("effective_fa_source"),
            "meets_fa_limit": selected_row.get("meets_fa_limit"),
            "max_disturbance_fa": max_disturbance_fa,
            "veto_rate": selected_row.get("veto_rate"),
            "raw_alarm_rate": selected_row.get("raw_alarm_rate"),
            "final_alarm_rate": selected_row.get("final_alarm_rate"),
            "alarm_drop": selected_row.get("alarm_drop"),
            "overall_false_alarm_rate": selected_row.get("overall_false_alarm_rate"),
            "clean_normal_false_alarm_rate": selected_row.get("clean_normal_false_alarm_rate"),
            "disturbance_negative_false_alarm_rate": selected_row.get("disturbance_negative_false_alarm_rate"),
            "bridged_disturbance_negative_false_alarm_rate": selected_row.get("bridged_disturbance_negative_false_alarm_rate"),
            
            
            # 干擾誤報歸屬（B）：干擾後 tail 秒的誤報、警報響起次數、干擾事件被誤報的比例
            "disturbance_tail_sec": selected_row.get("disturbance_tail_sec"),
            "disturbance_tail_false_alarm_rate": selected_row.get("disturbance_tail_false_alarm_rate"),
            "disturbance_negative_alarm_onsets": selected_row.get("disturbance_negative_alarm_onsets"),
            "clean_normal_alarm_onsets": selected_row.get("clean_normal_alarm_onsets"),
            "disturbance_n_events": selected_row.get("disturbance_n_events"),
            "disturbance_event_hit_rate": selected_row.get("disturbance_event_hit_rate"),
        }
        for c in fa_type_cols:                       # 每種干擾類型各自的誤報率
            settings[c] = selected_row.get(c)

        # 確認單一 veto 至少有：要看哪一欄, 用什麼門檻。 → 半套設定，擋下來
        has_single_veto = settings["veto_column"] is not None and settings["veto_threshold"] is not None
        # 若設定宣稱啟用 veto，卻沒有任何實際 veto 條件，就停止
        if settings["use_veto"] and not has_single_veto and not settings["veto_rules"]:
            raise ValueError(f"規則 {settings['name']} 啟用 veto，但沒有 veto_column/veto_threshold，也沒有 veto_rules")
        return settings

    # --- 刪除舊候選檔 ---
    # 在寫出新候選前，先刪掉上次留下的 top01、top02 等候選 JSON
    for old in model_folder.glob("postprocess_config_candidate_top*.json"):
        old.unlink()

    # --- 輸出每種規則家族的代表候選 ---
    candidate_payloads = []
    # 逐一走過已排序的規則代表
    for rank, (_, row) in enumerate(reps.iterrows(), start=1):
        selected_row = _json_safe(row.to_dict())                  # 將 NumPy／Pandas 的特殊數值轉成 JSON 可儲存的普通型別
        settings = build_baseline_rule_settings(selected_row)     # 取得實際要給 Runtime 使用的規則設定
        payload = {
            "model_id": model_id,                                # 對應模型版本
            "selected_at": datetime.now().isoformat(),           # 建立時間
            "source": source,                                    # 來源
            "candidate_rank": rank,                              # 候選排名
            "baseline_rule_name": settings["name"],              # 規則名稱
            "baseline_rule_settings": _json_safe(settings),      # 完整規則參數與驗證結果
        }
        safe_name = re.sub(r"[^\w\-.]", "_", str(settings["name"]))
        path = model_folder / f"postprocess_config_candidate_top{rank:02d}_{safe_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        candidate_payloads.append(payload)
    print(f"✅ 已輸出 {len(candidate_payloads)} 份候選 postprocess config 到 {model_folder}")

    # 舊版的 postprocess_config_candidate.json（「推薦第 1 名」）不再輸出：
    # 第 1 名只是干擾這一面的排名，正式規則要看完轉換偵測率才由你決定，留著容易被誤當成定案
    old_pointer = model_folder / "postprocess_config_candidate.json"
    if old_pointer.exists():
        old_pointer.unlink()

    # --- 只有明確要求時，才寫出正式版 ---
    final_payload = None
    if write_final:
        # 避免程式自行把排名第一名直接視為正式決策
        if selected_rule_name is None:
            raise ValueError("write_final=True 時，selected_rule_name 不能是 None")
        matched = df.loc[df["name"] == selected_rule_name]
        # 指定的規則名稱必須在比較表裡，而且只能對應唯一一列
        if len(matched) != 1:
            raise ValueError(f"selected_rule_name={selected_rule_name} 在 summary_df 裡有 {len(matched)} 筆（應為 1 筆）")
        
        if selected_rule_name not in set(reps["name"]):
            print(f"⚠️ {selected_rule_name} 不在這次輸出的 {len(reps)} 個候選裡："
                  "轉換推論的候選比較表沒有比過它的偵測率，請確認是刻意選的")
                
        settings = build_baseline_rule_settings(_json_safe(matched.iloc[0].to_dict()))
        final_payload = {
            "model_id": model_id,
            "selected_at": datetime.now().isoformat(),
            "source": source,
            "baseline_rule_name": selected_rule_name,
            "baseline_rule_settings": _json_safe(settings),
            # 驗證進度：干擾 holdout 量完是 pending，轉換 holdout 量完才改成 final_verified
            "validation_status": "pending_transition_holdout",
        }
        final_path = model_folder / "postprocess_config.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)
        print(f"✅ 已輸出鎖定規則 {selected_rule_name}，等待轉換 holdout 最終驗證（validation_status = pending_transition_holdout）: {final_path}")

    return candidate_payloads, final_payload


# ============================================================
# 生成候選 schema 函式
# ============================================================  
def save_runtime_feature_schema(
    model_folder,
    model_id,
    feature_names,             # 模型真正吃的特徵欄位清單
    df_columns,                # 這次實際跑出來的特徵表
    baseline_rule_name,        # 之前用 save_postprocess_configs 選出來的那條規則的資訊
    baseline_rule_settings,    # 之前用 save_postprocess_configs 選出來的那條規則的資訊
):
    # 轉成 Path, 並確保資料夾存在
    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)

    # 一定要有的識別欄位一定要有的識別欄位
    base_columns = [
        "filename",
        "record_id",
        "t_start",
        "t_ref",
    ]

    # 列出「規則版」（rule-based,不是模型)後處理邏輯可能會用到的欄位候選
    default_rule_feature_candidates = [
        "offband_E_mean",
        "offband_E_max",
        "leak_focus_E_mean",
        "leak_focus_E_max",
        "resonance_E_mean",
        "resonance_E_max",
        "offband_to_leak_focus_E_ratio",
        "offbandmax_to_leak_focusmax_E_ratio",
        "offband_to_resonance_E_ratio",
        "offbandmax_to_resonancemax_E_ratio",
        "z_offband_E_mean",
        "z_offband_E_max",
    ]

    # 收集 veto 規則額外用到的欄位
    veto_rule_cols = []
    # 從選定的那條規則設定裡,拿出 veto_rules（一份 veto 規則清單，每條規則應該是一個 dict 結構)
    for rule in baseline_rule_settings.get("veto_rules", []) if isinstance(baseline_rule_settings, dict) else []:
        col = rule.get("col")     # 迴圈裡對每一條 veto 規則,用 rule.get("col") 拿出它依據的欄位名稱,
        if col:     # 確認這個值不是空的才收進 veto_rule_cols
            veto_rule_cols.append(col)

    # 篩出真正存在的規則特徵欄位
    rule_feature_names = []
    # 把「候選清單」（預設候選 + veto 額外欄位）跟「這次特徵工程實際產出的欄位」df_columns 做交集
    for col in default_rule_feature_candidates + veto_rule_cols:
        if col in df_columns and col not in rule_feature_names:
            rule_feature_names.append(col)        # 只有 col in df_columns 的欄位才會真正被收進 rule_feature_names

    # 組出最終「所有必要欄位」清單
    all_required_columns = []
    # 把識別欄位、模型特徵欄位、規則特徵欄位三份清單接在一起,一樣用「邊迭代邊檢查是否已存在」的方式去重,保留原本的出現順序
    for col in base_columns + list(feature_names) + rule_feature_names:
        if col not in all_required_columns:
            all_required_columns.append(col)

    # 組成 payload 並寫檔
    payload = {
        "schema_name": "runtime_feature_schema_candidate",
        # 暗示這份 JSON 只是「候選版」,呼應 save_postprocess_configs 裡「候選 vs 正式」的兩階段設計
        "schema_status": "candidate",                          
        "model_id": model_id,
        "baseline_rule_name": baseline_rule_name,   # 只是產生這份候選時參考的規則，不是定案；正式版會依 postprocess_config.json 重算
        "base_columns": base_columns,
        "model_feature_names": list(feature_names),
        "rule_feature_names": rule_feature_names,
        "all_required_columns": all_required_columns,
        
        # 保存 disturbance 特徵表實際能提供的全部欄位。
        # transition 定案規則時會用它檢查正式規則所需欄位。
        "available_columns": list(df_columns),
    }

    # 寫檔與回傳
    save_path = model_folder / "runtime_feature_schema_candidate.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✅ 已輸出 runtime feature schema: {save_path}")
    return payload     # 這份 schema candidate 的完整內容,同時也被寫成 JSON 檔


# ============================================================
# 生成正式 schema 函式
# ============================================================  
def save_final_runtime_feature_schema(
    *,
    model_folder,
    metadata_json,
    postprocess_config_json,
):
    # 讀三份輸入,candidate 不存在就先擋下來
    model_folder = Path(model_folder)
    metadata_json = Path(metadata_json)
    postprocess_config_json = Path(postprocess_config_json)

    candidate_path = (model_folder / "runtime_feature_schema_candidate.json")

    if not candidate_path.exists():
        raise FileNotFoundError(f"找不到 candidate schema: {candidate_path}")

    candidate_schema = json.loads(candidate_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    postprocess = json.loads(postprocess_config_json.read_text(encoding="utf-8"))

    # 檢查三份資料是不是同一顆模型
    candidate_model_id = candidate_schema.get("model_id")
    metadata_model_id = metadata.get("model_id")
    postprocess_model_id = postprocess.get("model_id")

    model_ids = {
        candidate_model_id,
        metadata_model_id,
        postprocess_model_id,
    }
    # 把三個來源各自的 model_id 丟進一個 Python set。如果三個值都相同,set 自動去重之後只會剩一個元素,len(model_ids) == 1
    if len(model_ids) != 1:
        raise ValueError(f"candidate schema、metadata、postprocess 不是同一顆模型: {model_ids}")

    # 確認 metadata 裡有模型特徵清單，且跟候選版一致
    model_feature_names = (metadata.get("features", {}).get("feature_names", []))
    if not model_feature_names:     # 先確認 metadata 裡真的有 feature_names
        raise ValueError("metadata.json 缺少 features.feature_names")
    # 拿候選 schema 裡存的 model_feature_names 跟 metadata 裡的做逐項比對,不只是比數量,連順序都要完全一樣
    candidate_model_features = candidate_schema.get("model_feature_names",[],)
    if list(candidate_model_features) != list(model_feature_names):
        raise ValueError("candidate schema 與 metadata 的模型特徵不一致")

    # 正式 schema 只能接在完整流程的最後：規則在 dev 選、干擾 holdout 和轉換 holdout 都評估過
    # （少任何一步，runtime 就會拿到一條沒驗證過偵測率或誤報率的規則）
    missing_steps = []
    if postprocess.get("source") != "disturbance_dev_rule_selection":
        missing_steps.append(f"source 應為 disturbance_dev_rule_selection（目前 {postprocess.get('source')!r}）")
    if "holdout_evaluation" not in postprocess:
        missing_steps.append("沒有干擾 holdout 成績（請跑干擾推論 mode='evaluate', write_postprocess_final=True）")
    if "transition_holdout_evaluation" not in postprocess:
        missing_steps.append("沒有轉換 holdout 成績（請跑轉換推論 run_transition_feature_inference）")
    if postprocess.get("validation_status") != "final_verified":
        missing_steps.append(f"validation_status 應為 final_verified（目前 {postprocess.get('validation_status')!r}）")
    if missing_steps:
        raise ValueError("postprocess_config.json 還不是完整驗證過的正式版，不能產生正式 schema：\n  - "
                         + "\n  - ".join(missing_steps))

    # 取出正式規則,決定 rule_feature_names 要收哪些
    rule_cfg = postprocess.get("baseline_rule_settings",{},)
    if not rule_cfg:
        raise ValueError("postprocess_config.json 缺少 baseline_rule_settings")

    # 正式定案的規則真的有啟用 use_veto 時，才去收集 veto_column 以及每條 veto_rules 用到的欄位
    # 也就是說,正式版的 rule_feature_names 精準對應到「這條被選定的規則實際會用到的欄位」
    rule_feature_names = []
    # 過濾掉空值、去重,避免同一個欄位被列兩次。
    def add_rule_feature(name):
        if name and name not in rule_feature_names:
            rule_feature_names.append(name)

    # 只有正式規則啟用 veto 時，才把 veto 所需欄位列為 runtime 必要欄位。
    if bool(rule_cfg.get("use_veto", False)):
        add_rule_feature(rule_cfg.get("veto_column"))

        for veto_rule in rule_cfg.get("veto_rules",[]):
            add_rule_feature(veto_rule.get("col"))

    # 檢查正式規則需要的欄位，disturbance 階段是否真的產生過。
    # 先找候選 schema 裡的 "available_columns"（也就是候選版存的那份「disturbance 特徵表實際有哪些欄位」的完整快照)；如果候選版沒有這個 key（例如是舊版候選檔,還沒有這個欄位),才退回用 "all_required_columns" 頂替
    available_columns = set(candidate_schema.get("available_columns",candidate_schema.get("all_required_columns",[])))

    # 核心防呆：拿「這條正式規則真正需要的欄位」rule_feature_names,去跟「disturbance 特徵工程實際產出過的欄位」available_columns 比對
    missing_rule_features = [
        col for col in rule_feature_names if col not in available_columns
    ]
    if missing_rule_features:
        raise ValueError(f"正式規則需要的欄位未出現在 disturbance 特徵結果中: {missing_rule_features}")

    # 組出正式 all_required_columns，寫出正式 schema
    base_columns = candidate_schema.get(       # 優先從候選 schema 裡拿,如果候選版沒存（同樣是舊版候選檔相容性考量),才退回寫死的預設四欄
        "base_columns",
        ["filename", "record_id", "t_start","t_ref"],
    )

    all_required_columns = []

    for col in (
        list(base_columns) + list(model_feature_names) + list(rule_feature_names)):
        if col not in all_required_columns:
            all_required_columns.append(col)

    payload = {
        "schema_name": "runtime_feature_schema",
        "schema_status": "final",        # schema_status": "final" 對應候選版的 "candidate",標示這是正式版
        "model_id": metadata_model_id,
        "baseline_rule_name": postprocess.get("baseline_rule_name"),
        "base_columns": list(base_columns),
        "model_feature_names": list(model_feature_names),
        "rule_feature_names": list(rule_feature_names),
        "all_required_columns": (all_required_columns),
        
        # 溯源欄位
        "source_candidate_schema": (candidate_path.name),
        "source_postprocess_config": (postprocess_config_json.name),
    }

    # 輸出並回傳
    final_path = (model_folder / "runtime_feature_schema.json")
    final_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,),encoding="utf-8")

    print("✅ 已輸出正式 runtime schema:",final_path)
    return final_path


# ============================================================
# 把干擾資料分成「選規則用（dev）」和「最後評估用（holdout）」
# ============================================================
def split_dev_holdout_for_rules(df, split_col="split"):
    """
    選規則和評估規則不能用同一份資料：規則是照 dev 的表現挑出來的，在 dev 上的成績一定偏好，
    holdout 只能在規則定案後量一次，才是可信的成績。

    1. 資料有 split 欄（特徵工程就切好，建議做法）：
       dev / holdout 照用；train 列是模型訓練看過的，丟掉並警告
    2. 沒有 split 欄就報錯：dev / holdout 一律由切分程式決定，這裡不自己切。

    回傳 (df_dev, df_holdout, split_table)；split_table 記錄每個單位分到哪一邊，方便追查
    """
    if split_col in df.columns:
        # 把 split 欄位統一整理成小寫純文字
        split = df[split_col].astype(str).str.strip().str.lower()
        
        # 如果輸入 CSV 裡還混有 train 資料，程式會提示你，但不會把它拿來：選規則, 評估規則, 報告誤報率
        n_train = int((split == "train").sum())
        if n_train:
            print(f"⚠️ 有 {n_train} 列 split=train（模型訓練看過），不用來選規則或評估，已排除")
        # 資料品質檢查：確認 split 只能有三種值，train / dev / holdout
        unknown = sorted(set(split) - {"train", "dev", "holdout"})
        if unknown:
            raise ValueError(f"{split_col} 欄出現未知的值：{unknown}（只能是 train / dev / holdout）")
        
        # 資料隔離的最小單位:優先用 session_id，如果沒有 session_id，才退而求其次以 record_id 為單位。
        unit_col = "session_id" if "session_id" in df.columns else "record_id"
        
        # 每個 session 出現過幾種不同的 split(每個 session 只屬於一個 split)？一個 session 從頭到尾只能屬於 train / dev / holdout 其中一份
        unit_split_count = (pd.DataFrame({"unit": df[unit_col], "split": split})
                            .groupby("unit")["split"].nunique())
        # 抓資料洩漏：只要同一個 session 出現在兩種以上 split，程式就直接停止
        mixed_units = unit_split_count[unit_split_count > 1].index.tolist()
        if mixed_units:
            raise ValueError(f"同一個 {unit_col} 出現在多個 split，可能造成資料洩漏："
                             f"{mixed_units[:10]}（共 {len(mixed_units)} 個）")
        
        # 真正取出兩份資料:df_dev,df_hold
        df_dev = df[split == "dev"].copy()          # 用於比較所有規則、選候選
        df_hold = df[split == "holdout"].copy()     # 規則定案後，只評估最後選中的那一條
        
        # 產生一張「這次規則評估實際用了哪些 session」的紀錄表
        split_table = (df[split != "train"][[unit_col]].assign(rule_split=split[split != "train"])
                       .drop_duplicates().reset_index(drop=True))
        source = f"欄位 {split_col}"
    else:
        # 不再自己切：dev / holdout 一律由切分程式決定（split_disturbance_dev_holdout.py、split_transition_dev_holdout.py）
        # 自己切的話，餵錯檔案也會照樣跑完，不會發現
        raise ValueError(f"資料沒有 {split_col} 欄：請餵切分程式輸出的 *_rule_features_eval.csv，"
                         "不要直接餵特徵工程的輸出")

    print(f"dev / holdout 分法：{source}")
    print(f"  dev     ：{len(df_dev)} 窗，{df_dev[unit_col].nunique()} 個 {unit_col}")
    print(f"  holdout ：{len(df_hold)} 窗，{df_hold[unit_col].nunique()} 個 {unit_col}")
    
    # 如果資料切分錯誤，導致 Dev 或 Holdout 沒有任何資料，程式不會硬跑出一個看似正常、其實無意義的結果。
    if df_dev.empty or df_hold.empty:
        raise ValueError("dev 或 holdout 是空的，無法分開選規則和評估")
    return df_dev, df_hold, split_table


# ============================================================
# 只評估一條規則（holdout 用，不排名）
# ============================================================
def evaluate_single_rule(df_pred,                     # 已經完成 XGBoost 推論的干擾資料，每個 window 都已有 pred_proba、pred_label
                         cfg,                         # 已選定的告警規則設定，例如連續幾個 window 超標才報警、是否啟用 veto 等
                         window_threshold,            # 模型把機率轉為初步 leak / non-leak 判定的門檻。
                         max_disturbance_fa=None,     # 可接受的干擾誤報上限，例如 0.01 代表 1%。
                         min_testable_ratio=0.5):     # 至少有 50% 的干擾片段夠長，能真正測到這條規則的告警延遲，才把一般誤報率視為可信。
    """
    在 holdout 上只量「已經定案的那一條規則」，不跑其他規則，也不排名，
    避免看了 holdout 的成績又回頭換規則。

    成績的定義和 compare_disturbance_alarm_methods（dev）完全相同，兩邊才能直接比：
      - 可測性只看算給干擾的窗（干擾中＋干擾後 tail 秒）
      - effective_fa：測得出來用一般誤報率，測不出來用橋接誤報率
      - 有給 max_disturbance_fa 時，另外記 meets_fa_limit（holdout 上有沒有達標）
    """
    # 重新標記每個 window 的「誤報歸屬」
    # during：干擾進行中
    # tail：干擾結束後的尾端時間
    # clean：真正沒有干擾的時間
    if "eval_class" in df_pred.columns:
        df_pred = attribute_disturbance_windows(df_pred)

    # 已定案的告警規則套到每個 record_id 上, 套用後的 df_rule 通常會多出：raw_alarm / veto_flag / final_alarm
    df_rule = apply_alarm_method_by_record(df_pred, cfg, window_threshold)
    # 從資料本身估計：每個 window 長度, 每次滑窗往前移多少秒。
    window_sec, hop_sec = estimate_window_timing(df_pred)
    # 計算每一段連續訊號實際有多長
    span = compute_segment_span(df_pred)
    
    # 只保留「可算給干擾的時間段」:during：干擾當下 / tail：干擾停止後的餘振期間
    if "fa_bucket" in df_pred.columns and df_pred["fa_bucket"].isin(["during", "tail"]).any():
        span = span[df_pred["fa_bucket"].isin(["during", "tail"])]
    # 算出這條規則理論上的「最短告警延遲」。
    latency = rule_min_alarm_latency_sec(cfg, window_threshold, window_sec, hop_sec)
    # 計算有多少比例的干擾片段「長度足夠測試此規則」
    testable_ratio = float((span >= latency - 1e-9).mean())

    # 先建立這條規則的基本結果摘要
    metrics = {
        "rule_name": cfg["name"],                               # 正在驗收哪一條規則
        "n_windows": int(len(df_pred)),                         # Holdout 有多少個 window
        "min_alarm_latency_sec": latency,                       # 這條規則理論上最少多久才能報警
        "fa_testable_ratio": testable_ratio,                    # 有多少干擾片段真的夠長、能測這條規則
        "fa_testable": testable_ratio >= min_testable_ratio,    # 可測比例是否至少達 50%
        "veto_rate": float(df_rule["veto_flag"].mean()),        # 有多少 window 被抗干擾 veto 擋掉
    }
    # 把實際套用規則後的誤報統計加進去
    metrics.update(summarize_disturbance_rule(df_rule))
    
    # 「橋接版」評估:正常狀況下，如果檔案之間有太長空檔，告警規則會重置，因為真實系統不能假設中間的訊號持續存在。
    bridged = summarize_disturbance_rule(
        apply_alarm_method_by_record(df_pred, cfg, window_threshold, max_gap_sec=np.inf))
    # 把橋接版結果另外存起來，刻意與一般版分開
    metrics["bridged_disturbance_negative_false_alarm_rate"] = bridged.get("disturbance_negative_false_alarm_rate", np.nan)
    metrics["bridged_overall_false_alarm_rate"] = bridged["overall_false_alarm_rate"]

    # 優先使用「干擾負樣本誤報率」:和 dev 選規則時同一個定義的有效誤報率
    if "disturbance_negative_false_alarm_rate" in metrics:
        fa, fa_bridged = metrics["disturbance_negative_false_alarm_rate"], metrics["bridged_disturbance_negative_false_alarm_rate"]
    else:
        fa, fa_bridged = metrics["overall_false_alarm_rate"], metrics["bridged_overall_false_alarm_rate"]
    
    # 若資料夠長、可正常測試規則 → effective_fa 使用實測誤報率
    # 若資料太短、無法真正測試這條規則 → effective_fa 改用橋接估計誤報率
    metrics["effective_fa"] = fa if metrics["fa_testable"] else fa_bridged
    # 記錄 effective_fa 的來源
    metrics["effective_fa_source"] = "measured" if metrics["fa_testable"] else "bridged"
    
    # Holdout 的驗收結果
    if max_disturbance_fa is not None:
        metrics["max_disturbance_fa"] = float(max_disturbance_fa)
        metrics["meets_fa_limit"] = bool(metrics["effective_fa"] <= max_disturbance_fa)

    return df_rule, _json_safe(metrics)


# ============================================================
# 主流程：模型推論 + alarm 規則評估（select 選規則 / evaluate 評估定案規則）
# ============================================================
def run_disturbance_feature_inference(
    feature_csv,                    # 已經做完特徵工程的 disturbance 資料
    metadata_json,
    output_dir=None,                # 沒給就自動建一個帶時間戳記的資料夾
    run_name="disturbance",
    mode="select",                  # "select" = 選規則；"evaluate" = 評估已定案的規則
    selected_rule_name=None,        # evaluate 模式必填：已定案的規則名稱
    write_postprocess_final=False,  # evaluate 模式才可設 True：寫出正式 postprocess_config.json
    max_disturbance_fa=0.01,        # 干擾誤報率上限（每種規則挑代表時用）
    selection_reason=None,          # 寫正式版時必填：為什麼選這條（看了轉換比較表的哪些數字）
    transition_comparison_csv=None, # 選規則時看的轉換比較表路徑（transition_postprocess_candidate_comparison.csv），記錄用
):
    '''
    兩種模式：

    mode="select"（選規則）
      1. 載入模型、推論 pred_proba
      2. 資料分成 dev / holdout，只用 dev
      3. 在 dev 上比較 272 條規則，每種基礎規則挑一個代表，輸出 16 份候選給轉換推論
      4. 不寫正式版，不看 holdout

    mode="evaluate"（評估定案規則）
      1～2 同上
      3. 在 dev 上重算候選（設定檔內容來自 dev）
      4. 在 holdout 上只量 selected_rule_name 這一條，不排名、不換規則
      5. write_postprocess_final=True 時寫出正式版，並附上 holdout 成績
    '''
    # 參數先檢查：錯了就在載入模型、寫任何檔案之前停下來
    if mode not in ("select", "evaluate"):
        raise ValueError(f"mode 只能是 'select' 或 'evaluate'，收到 {mode!r}")
    if mode == "select" and write_postprocess_final:
        raise ValueError("select 模式不能寫正式版：請先選定規則，再用 mode='evaluate' 寫出")
    if mode == "evaluate" and not selected_rule_name:
        raise ValueError("evaluate 模式必須指定 selected_rule_name")
    if write_postprocess_final and not (selection_reason and str(selection_reason).strip()):
        raise ValueError("寫正式版時必須填 selection_reason：記下為什麼選這條規則，之後才查得到")
    
    # ---  Step 1-3：載入模型  ---
    # 路徑初始化
    feature_csv = Path(feature_csv)
    metadata_json = Path(metadata_json)
    models_dir, model_id, model_folder = resolve_model_location_from_metadata(metadata_json)
    # 建立模型管理器
    manager = ModelManager(models_dir=str(models_dir))
    # 依 metadata.json 所在資料夾的 model_id 載入「指定的那顆」模型（不是自動抓最新）
    model_id, pkg = get_xgb_model(manager, model_id=model_id, metadata_json=metadata_json)
    # 拆出推論設定
    model, scaler, feature_names, window_threshold, file_threshold, metadata, model_type = extract_inference_config(pkg)

    print("=" * 80)
    print("使用模型")
    print("=" * 80)
    print("model_id         =", model_id)
    print("window_threshold =", window_threshold)
    print("file_threshold   =", file_threshold)
    print("model_folder     =", model_folder)

    # ---  Step 4 前置：讀特徵資料、對齊檢查  ---
    # 讀特徵資料
    df = pd.read_csv(feature_csv) 
    # 先確認模型特徵對齊
    check_feature_alignment(df, feature_names)

    # ========================================================
    # 檢查固定 baseline 與 veto 規則欄位
    # 這些欄位必須由 disturbance 特徵工程預先產生，
    # 推論階段不可重新建立 baseline。
    # ========================================================
    required_rule_feature_cols = [
        "z_offband_E_mean",
        "z_offband_E_max",
        "offbandmax_to_leak_focusmax_E_ratio",
    ]
        
    missing_rule_feature_cols = [
        column
        for column in required_rule_feature_cols
        if column not in df.columns
    ]
    
    if missing_rule_feature_cols:
        raise ValueError(
            "disturbance rule feature CSV "
            "缺少固定 baseline／veto 欄位："
            f"{missing_rule_feature_cols}"
        )
    
    print("✅ disturbance 固定 baseline／veto 欄位檢查通過：", len(required_rule_feature_cols),)

    # ---  Step 4-5：抽特徵、標準化、模型推論  ---
    X = df[feature_names].copy()         # 抽出模型真正要吃的特徵（不含 offband/veto 專用的診斷欄位)
    if scaler is not None:
        X = scaler.transform(X)
    
    # 做原始模型推論
    pred_proba = model.predict_proba(X)[:, 1]                  # 每個 window 是 leak 的機率
    pred_label = (pred_proba >= window_threshold).astype(int)  # 用模型原始 window_threshold 轉成 0/1 預測

    df_pred = df.copy()
    df_pred["pred_proba"] = pred_proba
    df_pred["pred_label"] = pred_label

    # 輸出資料夾、raw 預測結果、分群誤報率
    if output_dir is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_folder / "inference_runs" / f"{run_name}_{run_ts}"
    else:
        output_dir = Path(output_dir)
    # 沒給 output_dir 就在模型資料夾底下建一個帶時間戳記的子資料夾,確保每次重跑不會互相覆蓋
    output_dir.mkdir(parents=True, exist_ok=True)

    # 輸出 raw prediction CSV
    raw_output_csv = output_dir / f"{feature_csv.stem}_raw_predictions_{model_id}.csv"
    df_pred.to_csv(raw_output_csv, index=False, encoding="utf-8-sig")
    print(f"\n已輸出 raw 預測結果: {raw_output_csv}")
    
    # ---  Step 6：分成 dev / holdout  ---
    df_dev, df_hold, split_table = split_dev_holdout_for_rules(df_pred)
    split_table.to_csv(output_dir / "rule_split_assignment.csv", index=False, encoding="utf-8-sig")

    # 模型原始門檻的誤報率（還沒套規則）：select 只看 dev，evaluate 才看 holdout
    # （選規則時不偷看 holdout，連模型原始成績也不看）
    raw_df, raw_tag = (df_dev, "dev") if mode == "select" else (df_hold, "holdout")
    print(f"\n模型原始門檻 window-level 誤報率（{raw_tag}）=", raw_df["pred_label"].mean())
    if "eval_class" in raw_df.columns:
        print(f"\n依 eval_class 的模型原始誤報率（{raw_tag}）")
        print(raw_df.groupby("eval_class")["pred_label"].agg(["count", "mean"])
              .rename(columns={"mean": "false_alarm_rate"}))
    if "disturbance_type" in raw_df.columns:
        sub_df = raw_df[raw_df["disturbance_type"].fillna("") != ""]
        if not sub_df.empty:
            print(f"\n各干擾類型（干擾中的窗）的模型原始誤報率（{raw_tag}）")
            print(sub_df.groupby("disturbance_type")["pred_label"].agg(["count", "mean"])
                  .rename(columns={"mean": "false_alarm_rate"}))

    # ---  Step 7：在 dev 上比較所有規則（兩種模式都做；設定檔內容一律來自 dev）  ---
    summary_df, config_by_name = compare_disturbance_alarm_methods(
        df_pred=df_dev,
        window_threshold=window_threshold,
        output_dir=output_dir,
    )

    print("\n候選規則排名（dev）:")
    print(summary_df[[
        "name", "min_leak_duration_sec", "min_alarm_latency_sec", "fa_testable",
        "disturbance_negative_false_alarm_rate", "bridged_disturbance_negative_false_alarm_rate",
        "clean_normal_false_alarm_rate", "veto_rate",
    ]].head(20).to_string(index=False))

    if mode == "evaluate" and selected_rule_name not in config_by_name:
        raise ValueError(f"selected_rule_name 不存在: {selected_rule_name}")

    # ---  Step 8：輸出候選（evaluate 模式且 write_postprocess_final=True 才寫正式版）  ---
    candidate_postprocesses, final_postprocess = save_postprocess_configs(
        model_folder=model_folder,
        model_id=model_id,
        summary_df=summary_df,
        selected_rule_name=selected_rule_name,
        source="disturbance_dev_rule_selection",
        write_final=(mode == "evaluate" and write_postprocess_final),
        max_disturbance_fa=max_disturbance_fa,
    )

    # 分析表用哪條規則、哪份資料：
    #   select   → dev 上的第 1 名代表（只是方便看，不是定案）
    #   evaluate → holdout 上的定案規則
    if mode == "select":
        rule_name = candidate_postprocesses[0]["baseline_rule_name"]
        analysis_df = df_dev
        analysis_tag = "dev"
    else:
        rule_name = selected_rule_name
        analysis_df = df_hold
        analysis_tag = "holdout"

    baseline_df, rule_metrics = evaluate_single_rule(analysis_df, config_by_name[rule_name], window_threshold,
                                                     max_disturbance_fa=max_disturbance_fa)
    rule_row = _json_safe(summary_df.loc[summary_df["name"] == rule_name].iloc[0].to_dict())
    print(f"\n分析表使用：{rule_name}（{analysis_tag}）")
    print(json.dumps(rule_metrics, indent=2, ensure_ascii=False))

    # evaluate：holdout 成績另存一份，正式版也附上
    if mode == "evaluate":
        with open(output_dir / "holdout_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(rule_metrics, f, indent=2, ensure_ascii=False)
        if final_postprocess is not None:
            final_postprocess["holdout_evaluation"] = rule_metrics
            # 人工選規則的紀錄：選了哪條、為什麼、看的是哪張比較表
            final_postprocess["selection"] = _json_safe({
                "selected_rule_name": selected_rule_name,
                "selection_source": "disturbance_dev + transition_dev",
                "selection_reason": str(selection_reason).strip(),
                "transition_comparison_csv": str(transition_comparison_csv) if transition_comparison_csv else None,
                "max_disturbance_fa": max_disturbance_fa,
                "disturbance_tail_sec": DISTURBANCE_TAIL_SEC,
            })
            with open(model_folder / "postprocess_config.json", "w", encoding="utf-8") as f:
                json.dump(final_postprocess, f, indent=2, ensure_ascii=False)
            print("✅ postprocess_config.json 已附上干擾 holdout 成績；跑完轉換推論 run_transition_feature_inference 後才是可部署的正式規則")

    # 候選 runtime schema（用這次分析的那條規則）
    runtime_feature_schema = save_runtime_feature_schema(
        model_folder=model_folder,
        model_id=model_id,
        feature_names=feature_names,
        df_columns=df_pred.columns.tolist(),
        baseline_rule_name=rule_name,
        baseline_rule_settings=rule_row,
    )

    # 分析表（檔名前面加 dev_ 或 holdout_，一看就知道是哪份資料）
    prefix = f"{analysis_tag}_{rule_name}"
    export_disturbance_summary_tables(df_rule=baseline_df, output_dir=output_dir, rule_name=prefix)
    for table, suffix in (
        (build_record_level_summary(baseline_df), "record_summary"),
        (build_phase_summary(baseline_df), "phase_summary"),
        (build_disturbance_type_summary(baseline_df), "disturbance_type_summary"),
        (build_disturbance_id_summary(baseline_df), "disturbance_id_summary"),
    ):
        table.to_csv(output_dir / f"{prefix}_{suffix}.csv", index=False, encoding="utf-8-sig")

    run_info = {
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "model_folder": str(model_folder),
        "run_name": run_name,
        "feature_csv": str(feature_csv),
        "mode": mode,

        "n_rows": int(len(df_pred)),
        "n_rows_dev": int(len(df_dev)),
        "n_rows_holdout": int(len(df_hold)),

        "analysis_rule_name": rule_name,
        "analysis_data": analysis_tag,
        "analysis_rule_metrics": rule_metrics,
        "selected_rule_name": selected_rule_name,
        "candidate_rule_names": [x["baseline_rule_name"] for x in candidate_postprocesses],
        "max_disturbance_fa": max_disturbance_fa,
        "disturbance_tail_sec": DISTURBANCE_TAIL_SEC,      # 干擾後幾秒內的報警算干擾誤報
        "write_postprocess_final": bool(mode == "evaluate" and write_postprocess_final),

        "runtime_feature_schema_candidate_path": str(model_folder / "runtime_feature_schema_candidate.json"),
        "runtime_rule_feature_names": runtime_feature_schema["rule_feature_names"],

        "n_features_used": int(len(feature_names)),
        "feature_names": list(feature_names),
        "window_threshold": float(window_threshold),
        "file_threshold": file_threshold,
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(run_info), f, indent=2, ensure_ascii=False)

    return {
        "df_pred": df_pred,
        "df_dev": df_dev,
        "df_holdout": df_hold,
        "summary_df": summary_df,
        "config_by_name": config_by_name,
        "candidate_postprocesses": candidate_postprocesses,
        "baseline_df": baseline_df,
        "rule_metrics": rule_metrics,
        "output_dir": output_dir,
        "model_id": model_id,
    }





if __name__ == "__main__":
    # 已經訓練好並存好的那顆模型的 metadata.json 路徑
    MODEL_METADATA_JSON = Path(
        "/Users/paul/Desktop/Final/test_result/"
        "saved_models_transition_disturbance_final/xgb/"
        "xgb_20260923_164944/metadata.json"
    )

    # 切分程式輸出的規則版特徵（dev + holdout 都在裡面，有 split 欄）
    FEATURE_CSV = Path(
        "/Users/paul/Desktop/Final/disturbance_feature_engineering/"
        "disturbance_rule_features_eval.csv"
    )
    
    # 第 1 步：選規則（只用 dev），輸出 16 份候選給轉換推論
    run_disturbance_feature_inference(
        feature_csv=FEATURE_CSV,
        metadata_json=MODEL_METADATA_JSON,
        run_name="disturbance_select",
        mode="select",
    )
    
    # 第 2 步：在轉換推論中和偵測率一起看、選定一條規則後，才執行這段
    #          （只在 holdout 上量這一條；write_postprocess_final=True 才寫正式版）
    '''
    run_disturbance_feature_inference(
        feature_csv=FEATURE_CSV,
        metadata_json=MODEL_METADATA_JSON,
        run_name="disturbance_evaluate",
        mode="evaluate",
        selected_rule_name="這裡填轉換推論選定的規則名稱",
        write_postprocess_final=True,
        selection_reason="這裡寫為什麼選這條（例如：符合干擾誤報上限，轉換 dev 偵測率最高、延遲可接受）",
        transition_comparison_csv="這裡填轉換推論輸出的 transition_postprocess_candidate_comparison.csv 路徑",
    )
    '''
    
