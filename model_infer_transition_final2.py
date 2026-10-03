#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transition 推論、候選規則驗證與正式部署設定定案流程。

這支程式不負責重新訓練模型，也不重新執行特徵工程。
它使用已訓練完成的模型與 transition 特徵 CSV，驗證 disturbance 階段找到的候選抗干擾規則，
確認規則套用到真實洩漏轉換資料後，仍能保有足夠的洩漏偵測能力。

這支程式在完整流程中的位置：
1. disturbance inference：
   - 評估不同 alarm / veto 規則
   - 輸出 postprocess_config_candidate_top*.json
   - 輸出 runtime_feature_schema_candidate.json

2. transition candidate validation：
   - 讀取所有候選 postprocess config
   - 將每條候選規則套用到 transition holdout
   - 比較偵測率、誤報率與偵測延遲
   - 輸出 transition_postprocess_candidate_comparison.csv

3. 正式規則定案：
   - 由使用者設定 SELECTED_RULE_NAME
   - 將選定候選規則寫成 postprocess_config.json

4. 正式 transition 驗證：
   - 使用正式 postprocess_config.json 執行推論
   - 評估 window-level 分類結果
   - 執行 onset evaluation
   - 輸出推論結果與評估摘要

5. 正式 runtime schema 定案：
   - 只有在 transition 正式驗證成功後才執行
   - 根據 metadata.json 的模型特徵
   - 結合正式 postprocess_config.json 的規則需求
   - 將 candidate schema 升級為
     runtime_feature_schema.json

主要輸入：
- model.pkl
- metadata.json
- thresholds.json
- transition_features_holdout.csv
- postprocess_config_candidate_top*.json
- runtime_feature_schema_candidate.json
- feature_config_micro.json

主要輸出：
- transition_postprocess_candidate_comparison.csv
- postprocess_config.json
- runtime_feature_schema.json
- window_predictions.csv
- window_metrics.json
- onset_results.csv
- run_info.json
- plots/window_confusion_matrix.png

重要原則：
1. 模型特徵名稱與順序以
   metadata.features.feature_names 為準。

2. disturbance 階段產生的 schema 只是 candidate，
   不能直接提供正式 runtime 使用。

3. runtime_feature_schema.json 只能在：
   - 正式規則已選定
   - transition 正式驗證成功
   之後產生。

4. 若 transition 特徵 CSV 缺少模型需要的欄位，
   應立即停止，不可繼續推論。

5. run_transition_feature_inference() 本身仍允許不傳入
   postprocess_config_json，供單純模型推論使用；
   但完整正式定案流程必須提供正式 postprocess config。

6. 若正式規則未啟用 veto，
   transition 資料不需要包含未使用的 offband 規則欄位。

7. runtime_feature_schema.json 是正式部署的欄位契約；
   runtime 不可載入 schema_status != "final" 的 schema。
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
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt


# 設定專案路徑
CURRENT_DIR = Path(__file__).resolve().parent    # 這支 .py 所在的資料夾
PROJECT_ROOT = CURRENT_DIR                       # 目前直接等於這個資料夾

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 匯入你自己寫的模型管理器
from model_manager_two_sensor_final import ModelManager

# 共用告警規則函式
from alarm_rule_utils import (
    find_positive_runs,
    apply_alarm_method_by_record,
)

# Runtime Schema 產生函式
from model_infer_disturbance_final2 import (
    save_final_runtime_feature_schema,
)


# ============================================================
# 從 metadata 路徑反推模型 root / model_id
# ============================================================
def resolve_model_location_from_metadata(metadata_json_path):
    metadata_json_path = Path(metadata_json_path)     # 傳進來的路徑轉成 Path 物件
    # 檢查路徑的檔名
    if metadata_json_path.name != "metadata.json":
        raise ValueError("請傳入 metadata.json 路徑")

    # 從路徑結構反推資訊，不是去讀 metadata 裡面的內容
    model_id = metadata_json_path.parent.name
    model_type_dir = metadata_json_path.parent.parent
    models_dir = model_type_dir.parent

    return models_dir, model_id, metadata_json_path.parent


#                                         ********************************  載入模型  ********************************
# ============================================================
# 載入模型：抓最新 XGB(假設已經有一個建好的 ModelManager 物件)
# ============================================================
def get_xgb_model(manager: ModelManager, model_id=None):     # 函式接受一個已經建好的 ModelManager 物件及一個可選的 model_id，預設 None
    '''假設已經有一個建好的 ModelManager 物件了,它只負責一件事：如果沒指定 model_id,就去這個 manager 裡面找出「最新的那顆 XGB 模型」並載入'''    
    if model_id is None:
        # 列出所有 model_type="xgb" 的模型,並依建立時間排序
        models = manager.list_models(model_type="xgb", sort_by="created_at")
        # 如果沒有任何 XGB 模型，就報錯
        if not models:
            raise ValueError("找不到 XGB 模型")
        # 因為排序是由新到舊，所以取第一筆就是最新的 model_id
        model_id = models[0]["model_id"]
    
    # 不管 model_id 是你自己傳進來的，還是剛剛自動找出來的「最新那個」，這裡統一呼叫 manager.load_model(model_id) 把模型真的載入記憶體
    pkg = manager.load_model(model_id)
    return model_id, pkg


def load_xgb_model_package(models_dir, model_id=None):
    '''
    輸入是一個資料夾路徑（字串）,不是 manager 物件。
    做的事情比較完整：先自己動手 manager = ModelManager(models_dir=str(models_dir)) 把 manager 建出來,然後才呼叫剛剛那個 get_xgb_model(manager) 去做「找最新模型」這件事
    '''
    # 根據輸入路徑建立一個 ModelManager 物件
    manager = ModelManager(models_dir=str(models_dir))

    # 載入最新模型
    if model_id is None:
        # 如果沒指定 model_id，就呼叫剛剛那個 get_xgb_model(manager)，讓它自動找最新的 XGB 模型並載入,拿到 model_id 跟 pkg。
        model_id, pkg = get_xgb_model(manager)
    else:
        # 如果有指定 model_id,就直接呼叫 manager.load_model(model_id) 載入你指定的那顆,model_id 保持原樣不用重新算。
        pkg = manager.load_model(model_id)

    return model_id, pkg, manager


# ============================================================
# 解析模型推論設定
# ============================================================
def extract_inference_config(pkg):
    '''從 pkg 裡抽出推論真正需要的資訊'''
    model = pkg["model"]                # 實際模型本體
    scaler = pkg.get("scaler", None)    # 如果有 scaler 就拿出來，沒有就 None
    metadata = pkg.get("metadata", {})  # 模型描述資訊
    model_type = metadata.get("model_type", None)    # 模型類型，例如 xgb

    # 從 metadata 裡取出訓練時使用的特徵欄位清單
    # 推論的時候，新進來的資料一定要按照同樣的欄位、同樣的順序餵給模型，不然 XGBoost 會直接對錯位置、算出無意義的結果，甚至報錯
    feature_names = metadata.get("features", {}).get("feature_names", None)
    # 如果抓不到特徵名稱，代表這模型無法正確對應推論欄位
    if not feature_names:
        raise ValueError("metadata.features.feature_names 不存在")
    
    thresholds = pkg.get("thresholds", {}) or {}
    # # 找 window_threshold , 而且用了「多層備援」的寫法
    # 先看 pkg 頂層有沒有直接放 window_threshold
    window_threshold = pkg.get("window_threshold", None)
    if window_threshold is None:    # 退而求其次看 pkg["thresholds"]
        window_threshold = thresholds.get("window_threshold", None)
    if window_threshold is None:    # 再沒有，就去 metadata 裡面的 performance.window_level.threshold 找
        window_threshold = metadata.get("performance", {}).get("window_level", {}).get("threshold", None)
    if window_threshold is None:    # 如果三種來源都找不到，直接用寫死的 0.5 保底
        window_threshold = 0.5

    # 如果三種來源都找不到，最後死馬當活馬醫，直接用寫死的 0.5 保底
    file_threshold = pkg.get("file_threshold", None)
    if file_threshold is None:
        file_threshold = thresholds.get("file_threshold", None)
    if file_threshold is None:
        file_threshold = metadata.get("performance", {}).get("file_level", {}).get("threshold", None)
    if file_threshold is None:
        file_threshold = 0.5

    return model, scaler, feature_names, float(window_threshold), float(file_threshold), metadata, model_type


# ============================================================
# 讀取 feature config json(接受一個路徑，讀出這個 JSON 檔案的文字內容)
# ============================================================
def load_feature_config(feature_config_json):
    '''
    feature config json 包含：特徵設定,視窗設定,threshold,onset / alarm 後處理規則...等等
    {
      "selected_feature_names": ["rel_xxx", "z_xxx", "cusum_xxx"],
      "window_sec": 0.2,
      "hop_sec": 0.05,
      "threshold": 0.5,
      "min_consecutive_windows": 3,
      "alarm_method": "consecutive"
    }
    注意：
    這裡如果有 threshold，可視為歷史設定或參考值；
    正式推論時的最終模型分類 threshold 仍以 metadata.json / thresholds.json 為準。
    '''
    return json.loads(Path(feature_config_json).read_text(encoding="utf-8"))


# ============================================================
# 正式 runtime 後處理規則:模型已經說疑似 leak 了，接下來要不要真的判成 leak)
# ============================================================
def load_postprocess_config(postprocess_config_path):
    postprocess_config_path = Path(postprocess_config_path)
    payload = json.loads(postprocess_config_path.read_text(encoding="utf-8"))

    rule_cfg = payload.get("baseline_rule_settings", {})
    if not rule_cfg:
        raise ValueError("postprocess_config.json 缺少 baseline_rule_settings")

    return payload


# ============================================================
# 整理 transition 推論設定
# ============================================================
def resolve_transition_inference_settings(pkg,                  # 模型包，通常是 ModelManager 載進來的東西
                                          feature_cfg=None):    # 額外的 feature_config.json 內容，可以不給；如果有給，就用它補充正式推論流程參數
    # 模型欄位仍以 model metadata 為主
    # feature_config 主要提供：1. 一致性檢查  2. threshold / onset 後處理參數
    
    # 先從模型包 pkg 裡，把最核心的推論資訊拆出來
    #   model：真正的模型         scaler：訓練時如果有標準化，就把 scaler 一起拿出來
    #   feature_names：模型訓練時實際使用的特徵欄位順序      window_threshold：窗級 threshold
    #   file_threshold：檔案級 threshold     metadata：模型描述資訊      model_type：模型類型，例如 xgb
    model, scaler, feature_names, window_threshold, file_threshold, metadata, model_type = extract_inference_config(pkg)

    # 如果呼叫這個函式的人沒有另外傳 feature_cfg，就自動改用模型 metadata 裡本來就存的那份 feature_config
    if feature_cfg is None:
        feature_cfg = metadata.get("signal_processing", {}).get("feature_config", None)

    # 只要有 feature_cfg，就做一次「一致性檢查」
    # 如果你有另外提供 feature_config.json，那就進來做補充和檢查；沒有的話，就完全照模型包原本的設定走
    if feature_cfg is not None:
        # 從 feature_config.json 裡抓 selected_feature_names, 如果有這份清單，就拿它去跟模型 metadata 裡的 feature_names 比較, 如果兩邊不一樣，就印警告
        cfg_feature_names = feature_cfg.get("selected_feature_names", [])
        if cfg_feature_names and list(cfg_feature_names) != list(feature_names):
            print("⚠️ feature_config 與模型 metadata 的特徵順序或內容不一致")
            print("metadata feature 數量 =", len(feature_names))
            print("feature_config feature 數量 =", len(cfg_feature_names))
            
        # feature_config 裡如果有 threshold，只當歷史參考，不參與正式推論
        cfg_threshold = feature_cfg.get("threshold", None)
        if cfg_threshold is not None and float(cfg_threshold) != float(window_threshold):
            print("⚠️ feature_config threshold 與模型 threshold 不一致")
            print("model threshold =", window_threshold)
            print("feature_config threshold =", cfg_threshold)

    # 把這次正式推論真正要用到的參數，整理成一個乾淨的字典 settings
    settings = {
        "feature_names": feature_names,                 # 特徵欄位清單：模型真正要吃的欄位, 真正推論欄位仍以模型 metadata 為主
        "window_threshold": float(window_threshold),    # 窗級判斷門檻
        "file_threshold": float(file_threshold),        # 檔案級判斷門檻

        # onset / alarm 後處理參數
        "window_sec": feature_cfg.get("window_sec", 0.2) if feature_cfg else 0.2,    # 每個時間窗多長
        "smooth_window": feature_cfg.get("smooth_window", 3) if feature_cfg else 3,    # 平滑時看幾個窗
        "consecutive": feature_cfg.get("min_consecutive_windows", 2) if feature_cfg else 2,  # 至少連續幾個命中窗才算警報

        # 如果之後你要擴充 n_of_m / hysteresis，也可以從 config 讀
        "alarm_method": feature_cfg.get("alarm_method", "consecutive") if feature_cfg else "consecutive",     # 連續命中判斷法:alarm_method 預設用 "consecutive"
        "n_hits": feature_cfg.get("n_hits", 3) if feature_cfg else 3,
        "m_window": feature_cfg.get("m_window", 5) if feature_cfg else 5,
        "high_threshold": feature_cfg.get("high_threshold", 0.8) if feature_cfg else 0.8,
        "low_threshold": feature_cfg.get("low_threshold", 0.4) if feature_cfg else 0.4,
    }

    return model, scaler, metadata, model_type, settings


# ============================================================
# 專門處理「數值到底算不算『沒提供』」這件事
# ============================================================
def _clean_optional_value(v, cast=None):
    # 如果 v 本身就是 None，就直接回傳 None，代表「沒有值可以用」。
    if v is None:
        return None
    # 萬一 v 是某種 pd.isna 處理不了、會丟例外的怪型別（例如某些巢狀結構），就直接跳過這個檢查，不讓整個程式崩潰，繼續往下走
    try:
        # pd.isna(v) 是 pandas 提供的「缺值檢測」函式，它認得的缺值不只 None，還包含 float('nan')、pd.NaT、pd.NA 等等。所以如果 v 是 NaN，這裡就會提早 return None，不會讓後面的轉型動作去處理一個「無意義的數字」
        if pd.isna(v):
            return None
    except Exception:
        pass
    return cast(v) if cast is not None else v     # 走到這裡，代表 v 是一個「有意義的值」（不是 None，也不是 NaN）


# ============================================================
# 把 disturbance holdout 已經選好的正式規則，套回 transition 驗證流程裡
# ============================================================
def apply_postprocess_rule_to_transition_settings(settings, postprocess_payload):
    # 從整份 postprocess_config.json（也就是 postprocess_payload）取出 baseline_rule_settings 這個子物件——也就是 disturbance holdout 規則搜索選出來的那組正式參數
    rule_cfg = postprocess_payload.get("baseline_rule_settings", {})
    # 如果這個子物件不存在或是空的，代表沒有規則可以套用，直接把原本傳進來的 settings 原封不動退回去，不做任何覆蓋
    if not rule_cfg:
        return settings

    # 防禦性複製：因為 dict 是可變物件
    settings = dict(settings)

    # 先用 .get("method") 安全地取值，丟進 _clean_optional_value 做「是不是缺值」的統一判斷。如果確實有值，才覆蓋 settings["alarm_method"]
    method = _clean_optional_value(rule_cfg.get("method"))      
    if method is not None:
        settings["alarm_method"] = method     # 這個欄位決定下游的告警邏輯要用哪一種演算法（"consecutive"/"n_of_m"/"hysteresis"）

    # 如果這個值有效，會先轉成 Python 的 float 再存進去
    window_threshold = _clean_optional_value(rule_cfg.get("window_threshold"), float)     # 實際值來自你 disturbance holdout 規則搜索選出來的
    if window_threshold is not None:
        settings["window_threshold"] = window_threshold

    # smooth_window 是對 window-level 機率分數做平滑（rolling mean）時看的視窗大小
    smooth_window = _clean_optional_value(rule_cfg.get("smooth_window"), int)
    if smooth_window is not None:
        settings["smooth_window"] = smooth_window

    # consecutive 是 method="consecutive" 這種告警邏輯專用的參數——要連續幾個 window 都超過 window_threshold 才真正觸發告警
    consecutive = _clean_optional_value(rule_cfg.get("consecutive"), int)
    if consecutive is not None:
        settings["consecutive"] = consecutive

    # n_hits/m_window 是給 method="n_of_m"（M 個 window 裡有 N 個超標即算命中）這種告警邏輯用的參數
    n_hits = _clean_optional_value(rule_cfg.get("n_hits"), int)
    if n_hits is not None:
        settings["n_hits"] = n_hits

    m_window = _clean_optional_value(rule_cfg.get("m_window"), int)
    if m_window is not None:
        settings["m_window"] = m_window

    # high_threshold/low_threshold 是給 method="hysteresis"（遲滯判斷：分數超過高閾值才「進入」告警狀態，掉到低閾值以下才「解除」）用的雙門檻參數
    high_threshold = _clean_optional_value(rule_cfg.get("high_threshold"), float)
    if high_threshold is not None:
        settings["high_threshold"] = high_threshold

    low_threshold = _clean_optional_value(rule_cfg.get("low_threshold"), float)
    if low_threshold is not None:
        settings["low_threshold"] = low_threshold
    
    # 補進 min_leak_duration_sec
    min_leak_duration_sec = _clean_optional_value(rule_cfg.get("min_leak_duration_sec"), float)
    if min_leak_duration_sec is not None:
        settings["min_leak_duration_sec"] = min_leak_duration_sec
        
    alarm_threshold = _clean_optional_value(rule_cfg.get("alarm_threshold"),float)
    
    if alarm_threshold is not None:
        settings["alarm_threshold"] = alarm_threshold
    
    settings["use_veto"] = bool(rule_cfg.get("use_veto", False))
    
    settings["veto_column"] = rule_cfg.get("veto_column")
    
    settings["veto_threshold"] = _clean_optional_value(rule_cfg.get("veto_threshold"),float)
    
    settings["veto_rules"] = (rule_cfg.get("veto_rules") or [])
    
    settings["stage1"] = rule_cfg.get("stage1")
    settings["stage2"] = rule_cfg.get("stage2")
    
    use_veto = settings["use_veto"]
    veto_column = settings["veto_column"]
    veto_threshold = settings["veto_threshold"]
    veto_rules = settings["veto_rules"]
    
    if not isinstance(veto_rules, list):
        raise ValueError(
            "postprocess_config.json 的 "
            "veto_rules 必須是 list"
        )
    
    if use_veto and not veto_column and not veto_rules:
        raise ValueError(
            "use_veto=True，但沒有提供 "
            "veto_column 或 veto_rules"
        )
    
    if (
        use_veto
        and veto_column
        and veto_threshold is None
    ):
        settings["veto_threshold"] = 0.5
    
    if settings["alarm_method"] == "two_stage":
        if (
            not isinstance(settings["stage1"], dict)
            or not isinstance(settings["stage2"], dict)
        ):
            raise ValueError(
                "method=two_stage 時必須提供 "
                "stage1 與 stage2"
            )

    return settings

#                                    ********************************  window-level 評估  ********************************
# ============================================================
# window-level 評估:評估每一個 window 預測得好不好
# ============================================================
def evaluate_window_level(
        df_pred,
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
        
    y_true = df_pred["label"].astype(int).values         # 取出真實標籤 label
    y_pred = df_pred[pred_col].astype(int).values        # 取出模型預測標籤

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
# 驗證檢查模型能不能把「穩定正常段」和「穩定洩漏段」分開，而且會刻意避開異常發生前後那段過渡區：
# ============================================================
def evaluate_steady_separability(
        df_pred,     # 模型預測結果表,通常至少要有 t_start, onset_gt_sec, pred_proba, pred_label
        margins=(0.5, 1.0, 2.0, 3.0),    # 要排除 onset 前後多少秒的過渡區
        window_sec=0.2):     # 每個 window 的長度
    '''
    把每個 window 依照 onset_gt_sec 分成兩類：
        異常發生前，而且離 onset 至少 margin 秒：當作 pre-steady normal
        異常發生後，而且離 onset 至少 margin 秒：當作 post-steady leak
    然後看模型在這兩類上的分類能力，例如 ROC-AUC、PR-AUC、混淆矩陣、分類報告。
    '''
    df = df_pred.copy()
    df["t_center"] = df["t_start"] + window_sec / 2.0    # 建立 t_center，也就是每個 window 的中心時間

    # 準備一個 list，後面把每個 margin 的評估結果收集起來，最後轉成 DataFrame
    # 測試不同的 margin，例如 0.5s、1.0s、2.0s、3.0s
    rows = []
    for margin in margins:
        # 把 onset 附近那段最模糊、最像過渡區的資料排除掉
        steady_df = df[
            (df["t_center"] <= df["onset_gt_sec"] - margin) |     # t_center <= onset - margin：onset 前夠遠的穩定正常窗
            (df["t_center"] >= df["onset_gt_sec"] + margin)       # t_center >= onset + margin：onset 後夠遠的穩定洩漏窗
        ].copy()

        # 根據時間相對於 onset 來定義類別,建立真實標籤 steady_phase：0 -> onset 前，代表穩定正常 / 1 -> onset 後，代表穩定洩漏
        steady_df["steady_phase"] = np.where(
            steady_df["t_center"] < steady_df["onset_gt_sec"], 0, 1
        )

        y_true = steady_df["steady_phase"].astype(int).values    # 真實類別
        y_prob = steady_df["pred_proba"].astype(float).values    # 模型預測機率
        y_pred = steady_df["pred_label"].astype(int).values      # 模型預測類別

        # 建立模型效能指標
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
        ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

        print("\n" + "=" * 80)
        print(f"Steady 驗證 margin = {margin:.1f}s")
        print("=" * 80)
        print("n =", len(steady_df))
        print("ROC-AUC =", auc)
        print("PR-AUC  =", ap)
        print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
        print(classification_report(
            y_true, y_pred,
            labels=[0, 1],
            target_names=["pre-steady normal", "post-steady leak"],
            zero_division=0,
        ))
        print("\nProbability by steady_phase:")
        print(steady_df.groupby("steady_phase")["pred_proba"].describe())

        rows.append({
            "margin_sec": margin,
            "n_windows": len(steady_df),
            "roc_auc": auc,
            "pr_auc": ap,
        })

    return pd.DataFrame(rows)



#                                 ********************************  onset / alarm decision  ********************************
'''
模型什麼時候「第一次真正偵測到洩漏開始」？
    它不是只看每個 window 有沒有分對，而是看一整條時間序列裡，警報是何時出現、是不是太早亂報、是不是有抓到真正 onset。

  第一層：先把一串 pred_proba 轉成警報序列
   - 不要只看單一窗的分數，而是看「最近幾窗的平均分數」，讓警報不要太抖
   - 這裡的意思是：
       如果平滑後分數 p >= threshold，就算一次命中
       要連續命中 consecutive 次，才把當前這個時間點標成 detected=1
       
  第二層：判斷這個警報和真實 onset 的關係
   - 對每個 record_id 分開算，因為每條記錄都有自己的時間序列

'''
# ============================================================
# onset detection:把連續 windows 轉成事件偵測
# ============================================================
def detect_leak_events(
    probs,                    # 每個 window 的 leak 機率分數
    method="consecutive",     # 要用哪種警報規則,可選："threshold", "consecutive", "n_of_m", "hysteresis"
    alarm_threshold=0.5,      # 一般門檻值，給 threshold、consecutive、n_of_m 用
    smooth_window=3,          # rolling mean 的視窗大小,例如 3 代表每個時間點看最近 3 個窗的平均分數
    consecutive=2,            # 連續幾次超過 threshold 才報警
    n_hits=3,                 # N-of-M 規則的參數,例如 3-of-5
    m_window=5,               # N-of-M 規則的參數,例如 3-of-5
    high_threshold=0.8,       # 給 hysteresis 用的雙門檻
    low_threshold=0.4,        # 給 hysteresis 用的雙門檻
):
    '''
    把模型對每個 window 輸出的 pred_proba，轉成比較穩定的「警報序列」。
    也就是說，它不是直接回答「這窗是不是 leak」，而是在回答：根據一整串分數，這個時間點要不要算成真的偵測到事件。
    '''
    # 先做平滑：把原本每窗的 probs 轉成 Series -> 對它做 rolling mean -> 視窗大小是 smooth_window
    s = pd.Series(probs).rolling(window=smooth_window, min_periods=1).mean().values     # min_periods=1 表示前幾個點即使不夠完整視窗，也照樣算平均
    
    # 初始化偵測結果,先建立一個和 s 同長度的全 0 陣列,後面如果某時間點符合警報條件，就把那個位置改成 1
    detected = np.zeros_like(s, dtype=int)

    # ==  1.threshold  ==
    # 只要平滑後分數 s >= threshold,這個時間點就報警,優點是簡單,缺點是容易受抖動影響。
    if method == "threshold":
        detected = (s >= alarm_threshold).astype(int)

    # ==  2.consecutive 連續  ==
    # 如果目前分數高於 threshold，就累積連續命中次數,如果某次低於 threshold，就歸零,只有連續命中次數達到 consecutive，才把當前點標成 1
    elif method == "consecutive":
        count = 0
        for i, p in enumerate(s):
            if p >= alarm_threshold:
                count += 1
            else:
                count = 0
            if count >= consecutive:
                detected[i] = 1
                
    # ==  3.n_of_m  ==
    # 維護一個長度最多為 m_window 的小緩衝區,每來一個點，就記錄它有沒有過 threshold,如果最近 m_window 個點中，至少有 n_hits 個過門檻，就報警
    # 3-of-5：最近 5 個窗中只要有 3 個高分,即使不是連續的，也算偵測到,這比 consecutive 更不嚴格，對有些訊號比較穩。
    elif method == "n_of_m":
        buf = deque(maxlen=m_window)
        for i, p in enumerate(s):
            buf.append(int(p >= alarm_threshold))
            if sum(buf) >= n_hits:
                detected[i] = 1
                
    # ==  4.hysteresis 雙閥值  == 
    # 平常狀態下，要超過 high_threshold 才能進入 alarm,一旦進入 alarm，不會因為稍微掉一點就立刻解除,只有低到 low_threshold 以下才解除 alarm
    elif method == "hysteresis":
        alarm_on = False
        for i, p in enumerate(s):
            if not alarm_on and p >= high_threshold:
                alarm_on = True
            elif alarm_on and p < low_threshold:
                alarm_on = False
            detected[i] = int(alarm_on)

    else:
        raise ValueError(f"未知 method: {method}")

    # 回傳兩個東西：s -> 平滑後分數, detected -> 最終警報序列
    return s, detected

# ============================================================
# 刪掉太短 alarm run
# ============================================================
def suppress_short_alarm_runs(sub, detected, min_leak_duration_sec):
        if min_leak_duration_sec is None or pd.isna(min_leak_duration_sec):
            return np.asarray(detected, dtype=int)
    
        detected = np.asarray(detected, dtype=int).copy()
        if detected.sum() == 0:
            return detected
    
        time_vals = sub["t_ref"].to_numpy(dtype=float)
    
        if {"t_ref", "t_start"}.issubset(sub.columns):
            window_sec = float(np.median((sub["t_ref"] - sub["t_start"]).to_numpy(dtype=float)) * 2.0)
        elif len(time_vals) >= 2:
            window_sec = float(np.median(np.diff(time_vals)))
        else:
            window_sec = 0.0
    
        runs = find_positive_runs(detected)
        for start, end in runs:
            duration_sec = (time_vals[end] - time_vals[start]) + window_sec
            if duration_sec < float(min_leak_duration_sec):
                detected[start:end + 1] = 0
    
        return detected.astype(int)


# ============================================================
# Onset-level 評估:以每條 record 為單位，評估 onset 偵測得如何
# ============================================================
def run_onset_evaluation(
    df_pred,
    method="consecutive",
    alarm_threshold=0.5,
    window_sec=0.2,
    smooth_window=3,
    consecutive=2,
    n_hits=3,
    m_window=5,
    high_threshold=0.8,
    low_threshold=0.4,
    min_leak_duration_sec=None,
    detected_col=None,
    smoothed_col=None,
):
    '''
    把前面每個 window 的預測結果，進一步整理成「每條 record 的事件偵測結果」。
    也就是說，它不是在看單一窗有沒有分對，而是在看：這條資料最後有沒有成功抓到洩漏開始點 onset，而且有沒有太早亂報。
    '''
    # 每一條 record_id 最後會產生一列 summary，先存在 rows
    rows = []

    # 對每個 record_id 分開算，因為每條記錄都有自己的時間序列
    # 因為 onset 是「每條完整時間序列」自己的事件，不是把所有 windows 混在一起算。也就是說：一條 record 有自己的一串 pred_proba, 一條 record 也有自己的一個 onset_gt_sec
    for rid, sub in df_pred.groupby("record_id"):
        sub = sub.sort_values("t_start").copy()    # 把同一條 record 的 windows 依時間排序
        probs = sub["pred_proba"].values           # 再取出它的機率序列

        # 對這條 record 的機率序列做警報偵測
        # smoothed：平滑後的機率, detected：每個時間點有沒有觸發警報
        if detected_col is not None:
            if detected_col not in sub.columns:
                raise ValueError(
                    f"找不到預先計算的警報欄位: {detected_col}"
                )
        
            detected = (pd.to_numeric(sub[detected_col],errors="coerce").fillna(0).astype(int).to_numpy())
        
            if (smoothed_col is not None and smoothed_col in sub.columns):
                smoothed = (pd.to_numeric(sub[smoothed_col],errors="coerce",).to_numpy())
            else:
                smoothed = probs
        
        else:
            # 保留舊功能：沒有預先計算結果時才自行計算
            smoothed, detected = detect_leak_events(
                probs,
                method=method,
                alarm_threshold=alarm_threshold,
                smooth_window=smooth_window,
                consecutive=consecutive,
                n_hits=n_hits,
                m_window=m_window,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
        
            detected = suppress_short_alarm_runs(sub, detected, min_leak_duration_sec,)
        
        # 存回結果
        sub["smoothed_proba"] = smoothed    # 看平滑後分數怎麼變
        sub["detected"] = detected          # 看哪幾個窗被判定為 alarm

        # onset_gt：真實的洩漏開始時間 , t_detect：每個 window 對應的時間點
        onset_gt = float(sub["onset_gt_sec"].iloc[0])
        t_detect = sub["t_start"].values + window_sec / 2.0    # 用 window 中心時間 當作這個窗的代表時間

        # 找出所有警報位置,不管發生在 onset 前還是後,全部先找出來
        any_idx = np.where(detected == 1)[0]
        # 找出 onset 之後的有效警報,只有發生在真實 onset 之後的警報，才算 valid detection
        valid_idx = np.where((detected == 1) & (t_detect >= onset_gt))[0] 
        
        # 第一個警報時間 first_alarm:第一個警報，不管它是早報還是正常報
        first_alarm = None if len(any_idx) == 0 else float(t_detect[any_idx[0]])
        # 第一個有效偵測時間 valid_onset:第一個發生在 onset_gt 之後的警報,這才是你真正想要的「有效偵測時間」
        valid_onset = None if len(valid_idx) == 0 else float(t_detect[valid_idx[0]])
        
        # 在算 onset 發生前的警報情況,量化：在真正異常發生前，你到底亂報了多少。
        pre_mask = t_detect < onset_gt     # 建一個布林遮罩，挑出所有 t_detect < onset_gt 的 windows,也就是「真實洩漏開始之前」的 windows
        pre_total = int(np.sum(pre_mask))  # onset 前總共有幾個 windows
        pre_detected = int(np.sum(detected[pre_mask] == 1))   # onset 前有幾個 windows 被規則判成 detected=1

        # onset 後 1 秒內的 windows,是量化：異常開始後，你有沒有快速而穩定地抓到。
        post_1s_mask = (t_detect >= onset_gt) & (t_detect <= onset_gt + 1.0)   # 挑出 onset_gt 到 onset_gt + 1 秒 之間的 windows
        post_1s_total = int(np.sum(post_1s_mask))                              # 這 1 秒內總共有幾個 windows
        post_1s_detected = int(np.sum(detected[post_1s_mask] == 1))            # 這 1 秒內有幾個 windows 被判成 alarm

        # 新增更細緻的 onset 指標
        first_alarm_lead_sec = None if first_alarm is None else (first_alarm - onset_gt)    # 如果 < 0：表示提早報警, 如果 > 0：表示晚於 onset 才報警, 越接近 0 越理想
        pre_onset_detected_ratio = None if pre_total == 0 else (pre_detected / pre_total)   # onset 前的 windows 裡，有多少比例被判成 alarm
        post_onset_1s_detected_ratio = None if post_1s_total == 0 else (post_1s_detected / post_1s_total)   # onset 發生後 1 秒內，有多少比例的 windows 被穩定抓到

        rows.append({
            "record_id": rid,                                                           # 哪一筆資料
            "onset_gt_sec": onset_gt,                                                   # 真實 onset 時間
            "first_alarm_sec": first_alarm,                                             # 第一個警報時間，不分早晚
            "valid_detected_onset_sec": valid_onset,                                    # 第一個有效警報時間，也就是 onset 後第一次偵測到
            "any_alarm": first_alarm is not None,                                       # 這條 record 有沒有出現過任何警報
            "false_alarm": first_alarm is not None and first_alarm < onset_gt,          # 第一個警報是不是發生在真實 onset 之前,如果是，就代表早報
            "valid_detection": valid_onset is not None,                                 # 有沒有在 onset 之後成功抓到警報
            "latency_sec": None if valid_onset is None else (valid_onset - onset_gt),   # 成功抓到後，比真實 onset 晚了幾秒,就是偵測延遲
            
            # 新增細緻 onset 指標
            "first_alarm_lead_sec": first_alarm_lead_sec,
            "pre_onset_detected_windows": pre_detected,
            "pre_onset_total_windows": pre_total,
            "pre_onset_detected_ratio": pre_onset_detected_ratio,
            "post_onset_1s_detected_windows": post_1s_detected,
            "post_onset_1s_total_windows": post_1s_total,
            "post_onset_1s_detected_ratio": post_onset_1s_detected_ratio,
        })

    return pd.DataFrame(rows)


# ============================================================
# 比較模型 B 最佳的三個方法
# ============================================================
def compare_alarm_methods(df_pred, window_threshold, output_dir=None, window_sec=0.2):
    '''
    把同一份模型預測結果 df_pred，套上多種不同的 alarm 規則
    同一個模型
    → 換不同 alarm 規則
    → 各自算 window-level / onset-level 表現
    → 整理成 summary_df
    '''
    # 三個比較方法
    configs = [
        {
            "name": "consecutive_s3_t0.85_k3",
            "method": "consecutive",
            "alarm_threshold": 0.85,
            "smooth_window": 3,
            "consecutive": 3,
        },
        {
            "name": "consecutive_s3_t0.80_k4",
            "method": "consecutive",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "consecutive": 4,
        },
        {
            "name": "consecutive_s3_t0.80_k3",
            "method": "consecutive",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "consecutive": 3,
        },
    ]

    summary_rows = []

    # 逐一跑每種比較方法
    for cfg in configs:
        # 先產生 rule-based window 預測
        df_rule = apply_alarm_method_by_record(df_pred, cfg, window_threshold=window_threshold)
        
        # 做 onset-level 評估
        # 在每個 record_id 上，用該規則去算：有沒有報警, 有沒有 false alarm, 有沒有 valid detection, latency 多久
        onset_df = run_onset_evaluation(
            df_rule,
            window_sec=window_sec,
            detected_col="detected_rule",
            smoothed_col="smoothed_proba_rule",
        )

        # 做 window-level 評估
        y_true = df_rule["label"].astype(int).values
        y_pred = df_rule["detected_rule"].astype(int).values
        
        # 整理成一列 summary,包含:window-level, onset-level
        summary_rows.append({
            "name": cfg["name"],
            "method": cfg["method"],
            # window-level
            "window_threshold": window_threshold,
            "alarm_threshold": cfg.get("alarm_threshold", np.nan),
            "smooth_window": cfg.get("smooth_window", np.nan),
            "consecutive": cfg.get("consecutive", np.nan),
            "n_hits": cfg.get("n_hits", np.nan),
            "m_window": cfg.get("m_window", np.nan),
            "high_threshold": cfg.get("high_threshold", np.nan),
            "low_threshold": cfg.get("low_threshold", np.nan),
            "window_precision": precision_score(y_true, y_pred, zero_division=0),
            "window_recall": recall_score(y_true, y_pred, zero_division=0),
            "window_f1": f1_score(y_true, y_pred, zero_division=0),
            
            # onset-level
            "any_alarm_rate": onset_df["any_alarm"].mean(),            # 平均有多少 record 至少報過一次警
            "false_alarm_rate": onset_df["false_alarm"].mean(),          # 平均有多少 record 第一個警報是早於真實 onset 的
            "valid_detection_rate": onset_df["valid_detection"].mean(),    # 平均有多少 record 最後真的有在 onset 後抓到異常
            "mean_latency_sec": onset_df.loc[onset_df["valid_detection"], "latency_sec"].mean(),    # 抓到的那些 record，平均晚多久才報警
            
            "mean_first_alarm_lead_sec": onset_df["first_alarm_lead_sec"].dropna().mean(),       # 平均提早或延後多少秒報第一個警
            "median_first_alarm_lead_sec": onset_df["first_alarm_lead_sec"].dropna().median(),   # 中位數版本，較不容易被極端值影響
            "mean_pre_onset_detected_ratio": onset_df["pre_onset_detected_ratio"].dropna().mean(),   # 平均 onset 前亂報比例
            "mean_post_onset_1s_detected_ratio": onset_df["post_onset_1s_detected_ratio"].dropna().mean(),   # 平均 onset 後 1 秒內的穩定抓取比例
    })
        
        # 把每種方法的詳細結果存檔
        if output_dir is not None:
            df_rule.to_csv(
                Path(output_dir) / f"{cfg['name']}_window_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
            onset_df.to_csv(
                Path(output_dir) / f"{cfg['name']}_onset_eval.csv",
                index=False,
                encoding="utf-8-sig",
            )
            
    # 做總表並排序,優先按照：window_precision, window_recall, window_f1 由高到低排序。意思就是目前比較偏向先挑「少誤報」的方法。
    summary_df = pd.DataFrame(summary_rows).sort_values(
        [
            "window_precision",
            "mean_pre_onset_detected_ratio",
            "mean_first_alarm_lead_sec",
            "mean_post_onset_1s_detected_ratio",
            "window_recall",
        ],
        ascending=[False, True, False, False, False],
    )

    if output_dir is not None:
        summary_df.to_csv(Path(output_dir) / "alarm_method_comparison.csv", index=False, encoding="utf-8-sig")

    return summary_df


# ============================================================
# 整理 onset_result 大表
# ============================================================
def export_onset_summary_tables(
        df_pred,             # window-level 預測結果表
        onset_result_df,     # 每個 record_id 的 onset 偵測結果
        output_dir):         # 輸出資料夾
    '''把 onset_result_df 這張很大的 onset 明細表，跟原始特徵表中的 record metadata 合併，然後再自動輸出幾張比較好看的摘要表。'''
    # 把輸出路徑轉成 Path，如果資料夾不存在就先建立
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 每個 record 取一筆 metadata，併回 onset-level 結果
    meta_cols = ["record_id", "leak_quantity", "distance_cm", "pressure_bar"]
    keep_meta_cols = [c for c in meta_cols if c in df_pred.columns]

    # 先把 df_pred 依 record_id, t_start 排序 -> 再以 record_id 分組 -> 每個 record_id 只留第一列
    record_meta = (
        df_pred.sort_values(["record_id", "t_start"])
        .groupby("record_id", as_index=False)[keep_meta_cols]
        .first()
    )

    # 把剛剛整理好的 record_meta 合併到 onset_result_df,就會增加leak_quantity, distance_cm, pressure_bar
    onset_with_meta = onset_result_df.merge(record_meta, on="record_id", how="left")

    # 欄位名稱統一
    # 如果目前沒有 volume_ml 這欄，但有 leak_quantity，就新增一欄 volume_ml，內容直接等於 leak_quantity
    if "volume_ml" in onset_with_meta.columns:
        volume_col = "volume_ml"
    elif "leak_quantity" in onset_with_meta.columns:
        volume_col = "leak_quantity"
    else:
        raise ValueError("onset summary 找不到 volume_ml 或 leak_quantity 欄位")

    # 1) volume summary 每個 volume 的整體表現 -> onset_summary_by_volume.csv
    #    會得到：10ml 有幾筆, 10ml 偵測率多少, 10ml 誤報率多少, 10ml 平均 latency 幾秒
    volume_summary = (
        onset_with_meta.groupby(volume_col, dropna=False)
        .agg(
            n_records=("record_id", "count"),
            valid_detection_rate=("valid_detection", "mean"),
            false_alarm_rate=("false_alarm", "mean"),
            mean_latency_sec=("latency_sec", "mean"),
            median_latency_sec=("latency_sec", "median"),
        )
        .reset_index()
        .sort_values(volume_col)
    )

    # 2) volume x distance summary volume × distance 的表現
    #    10ml 在 0cm 的偵測率, 10ml 在 90cm 的偵測率, 20ml 在不同距離下的 latency
    if "distance_cm" in onset_with_meta.columns:
        volume_distance_summary = (
            onset_with_meta.groupby([volume_col, "distance_cm"], dropna=False)
            .agg(
                n_records=("record_id", "count"),
                valid_detection_rate=("valid_detection", "mean"),
                false_alarm_rate=("false_alarm", "mean"),
                mean_latency_sec=("latency_sec", "mean"),
                median_latency_sec=("latency_sec", "median"),
            )
            .reset_index()
            .sort_values([volume_col, "distance_cm"])
        )
    else:
        volume_distance_summary = pd.DataFrame()

    # 3) failed cases 偵測失敗的資料
    failed_cases = onset_with_meta.loc[
        onset_with_meta["valid_detection"] == False,
        [c for c in [
            "record_id", volume_col, "distance_cm", "pressure_bar",
            "onset_gt_sec", "first_alarm_sec", "valid_detected_onset_sec",
            "latency_sec", "false_alarm", "pre_onset_detected_ratio",
            "post_onset_1s_detected_ratio"
        ] if c in onset_with_meta.columns]
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
    if not volume_distance_summary.empty:
        print(f"已輸出 volume x distance 摘要: {volume_distance_csv}")
    print(f"已輸出 failed cases: {failed_cases_csv}")

    return onset_with_meta, volume_summary, volume_distance_summary, failed_cases


# ============================================================
# 畫混淆矩陣
# ============================================================
def plot_confusion_matrix(cm,              # 混淆矩陣,一個2x2的numpy陣列
                          class_names,     # 類別名稱清單
                          title,           # 圖標題
                          save_path):      # 存檔路徑

    # 建立新畫布
    plt.figure(figsize=(5, 4), dpi=160)
    plt.imshow(cm, cmap="Blues")    # 用imshow把cm這個2x2矩陣畫成色塊圖
    plt.title(title) 
    plt.colorbar()                  # 圖旁邊加一條色階條,標示顏色深淺對應的數值範圍
    plt.xticks(range(len(class_names)), class_names)     # 設定x軸的刻度位置跟文字
    plt.yticks(range(len(class_names)), class_names)     # 設定y軸的刻度位置跟文字
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # 雙層迴圈,i跑過矩陣的每一列(row)、j跑過每一欄(column):(0,0)、(0,1)、(1,0)、(1,1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ 已儲存: {save_path}")


# ============================================================
# 解決「json.dump 不能讀取 numpy/pandas 型別」的問題
# ============================================================
def _to_builtin(obj):
    """把 numpy / pandas 型別轉成 json 可寫出的 Python 型別"""
    if isinstance(obj, dict):     # 如果是 dict，就對每個 value 遞迴處理
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):     # 如果是 list/tuple，就對每個元素遞迴處理（tuple 會被轉成 list，因為 JSON 本來就沒有 tuple 這個型別）
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):    # 如果是 np.ndarray，用 .tolist() 轉成原生 Python list 
        return obj.tolist()
    if isinstance(obj, (np.integer,)):    # 如果是 np.integer/np.floating 這種 numpy 純量，分別轉成 Python 的 int/float
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ============================================================

# ============================================================
def save_transition_inference_outputs(
    output_dir,
    df_pred,
    window_metrics,
    onset_result_df=None,
    alarm_compare_df=None,
    steady_summary_df=None,
    model_id=None,
    model_folder=None,
    feature_csv=None,
    settings=None,
    metadata=None,
):
    # 建立輸出資料夾
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = metadata or {}
    settings = settings or {}

    # 1. run info(組出一份「這次推論到底用了什麼模型、什麼設定」的摘要資訊)
    run_info = {
        # 來自 metadata.json
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "primary_eval_name": metadata.get("performance", {}).get("primary_eval_name"),
    
        # 路徑資訊
        "model_folder": str(model_folder) if model_folder is not None else None,
        "feature_csv": str(feature_csv) if feature_csv is not None else None,
    
        "n_rows": int(len(df_pred)),
        "n_features_used": int(len(settings.get("feature_names", []))),
        "feature_names": list(settings.get("feature_names", [])),
    
        # 這次推論實際用到的所有告警/門檻參數
        "window_threshold": settings.get("window_threshold"),
        "file_threshold": settings.get("file_threshold"),
        "window_sec": settings.get("window_sec"),
        "smooth_window": settings.get("smooth_window"),
        "consecutive": settings.get("consecutive"),
        "alarm_method": settings.get("alarm_method"),
        "n_hits": settings.get("n_hits"),
        "m_window": settings.get("m_window"),
        "high_threshold": settings.get("high_threshold"),
        "low_threshold": settings.get("low_threshold"),
    
        "settings_raw": settings,
    }
    
    # 把 run_info 這個 dict 丟進 _to_builtin 做遞迴轉型後再寫成 JSON
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(run_info), f, indent=2, ensure_ascii=False)

    # 2. window predictions(把逐 window 的推論結果（分數、預測標籤等）存成 CSV)
    df_pred.to_csv(output_dir / "window_predictions.csv", index=False, encoding="utf-8-sig")

    # 3. window metrics(把整包 window_metrics 經過 _to_builtin 轉型後寫成 JSON)
    with open(output_dir / "window_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(window_metrics), f, indent=2, ensure_ascii=False)

    # 4. onset / alarm / steady summary(選擇性輸出)
    if onset_result_df is not None:
        onset_result_df.to_csv(output_dir / "onset_results.csv", index=False, encoding="utf-8-sig")    # 洩漏起始點（onset）偵測結果

    if alarm_compare_df is not None:
        alarm_compare_df.to_csv(output_dir / "alarm_method_comparison.csv", index=False, encoding="utf-8-sig")    # 不同告警方法的比較表

    if steady_summary_df is not None:
        steady_summary_df.to_csv(output_dir / "steady_separability_summary.csv", index=False, encoding="utf-8-sig")     # 穩態可分離度的摘要表

    # 5. window confusion matrix plot
    cm_raw = window_metrics.get("confusion_matrix")
    if cm_raw is not None:
        cm = np.asarray(cm_raw)
        if cm.size > 0:
            plot_confusion_matrix(
                cm,
                class_names=["正常", "洩漏"],
                title="Transition Window-level Confusion Matrix",
                save_path=plots_dir / "window_confusion_matrix.png",
            )

    print(f"✅ 推論結果已輸出到: {output_dir}")


# ============================================================
# 檢查推論資料欄位是否和模型訓練時保存的特徵一致
# ============================================================
def check_feature_alignment(df, feature_names):
    """
    檢查推論資料欄位是否和模型訓練時保存的特徵一致
    """
    missing_cols = [c for c in feature_names if c not in df.columns]     # 在找「模型需要，但推論 CSV 裡沒有」的欄位
    matched_cols = [c for c in feature_names if c in df.columns]         # 在找「模型需要，而且推論 CSV 裡也真的有」的欄位

    # 定義哪些欄位是基本資訊，不算模型特徵
    meta_cols = {
        "filename", "record_id", "t_start", "t_ref", "label",
        "interval_name", "distance_cm", "leak_quantity",
        "pressure_bar", "output_tdms", "normal_end_sec",
        "leak_start_sec", "onset_gt_sec", "pred_proba", "pred_label"
    }

    # 在找「不是基本欄位，也不是模型要吃的特徵欄位」的其他欄位
    extra_feature_cols = [
        c for c in df.columns
        if c not in meta_cols and c not in feature_names
    ]

    print("\n" + "=" * 80)
    print("特徵對齊檢查")
    print("=" * 80)
    print("模型需要特徵數量:", len(feature_names))
    print("成功對齊特徵數量:", len(matched_cols))
    print("模型特徵名稱:", feature_names)
    print("缺少特徵:", missing_cols)
    print("額外但未使用的特徵欄位:", extra_feature_cols)

    # 如果模型需要的特徵有缺，就直接停掉，不要讓模型硬跑
    if missing_cols:
        raise ValueError("推論資料缺少模型需要的特徵欄位:\n" + "\n".join(missing_cols))

    # 確認真正送進模型的欄位順序
    used_feature_cols = df[feature_names].columns.tolist()
    print("實際送進模型的特徵順序:", used_feature_cols)
    print("特徵順序是否一致:", used_feature_cols == feature_names)
    print("=" * 80 + "\n")


# ============================================================
# 定案正式規則儲存成最終 postprocess config
# ============================================================
def save_selected_candidate_as_final(candidate_dir, selected_rule_name):
    candidate_dir = Path(candidate_dir)
    candidates = load_postprocess_candidates(candidate_dir)

    matched = None
    for item in candidates:
        if item["rule_name"] == selected_rule_name:
            matched = item
            break

    if matched is None:
        raise ValueError(f"找不到指定規則: {selected_rule_name}")

    final_path = candidate_dir / "postprocess_config.json"
    final_path.write_text(
        json.dumps(matched["payload"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("已儲存正式 postprocess_config.json")
    print("來源規則:", selected_rule_name)
    print("儲存位置:", final_path)

    return final_path



#                                      ********************************  後處理規則比較  ********************************
# ============================================================
# 讀取所有候選干擾後處理規則 postprocess config
# ============================================================
def load_postprocess_candidates(postprocess_dir, pattern="postprocess_config_candidate_top*.json"):
    # 去資料夾找所有符合 postprocess_config_candidate_top*.json 這個檔名樣式的檔案（例如 ..._top1.json、..._top2.json）
    postprocess_dir = Path(postprocess_dir)
    paths = sorted(postprocess_dir.glob(pattern))
    if not paths:
        raise ValueError(f"找不到候選 postprocess config: {postprocess_dir / pattern}")

    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # 對每個檔案，讀出 baseline_rule_settings
        rule_cfg = payload.get("baseline_rule_settings", {})
        if not rule_cfg:
            raise ValueError(f"{path.name} 缺少 baseline_rule_settings")
        # 包成一個 dic
        payloads.append({
            "path": path,          # 檔案路徑
            "payload": payload,    # 整份 JSON 內容
            "rule_name": payload.get("baseline_rule_name", rule_cfg.get("name", path.stem)),    # 規則名稱
            "rule_cfg": rule_cfg,     # 規則設定本體
        }) 
    return payloads

'''
def build_manual_transition_rule_candidates():
    """
    當 disturbance holdout 對所有規則都是 0 誤報時，
    top candidates 會太單一，所以額外加入幾條代表性規則給 transition 比較。
    """
    manual_rules = [
        {
            "name": "threshold_s3_t0.70_d0.40",
            "method": "threshold",
            "alarm_threshold": 0.70,
            "smooth_window": 3,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "consecutive_s3_t0.80_k3_d0.40",
            "method": "consecutive",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "consecutive": 3,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "consecutive_s3_t0.85_k3_d0.40",
            "method": "consecutive",
            "alarm_threshold": 0.85,
            "smooth_window": 3,
            "consecutive": 3,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "consecutive_s3_t0.80_k4_d0.40",
            "method": "consecutive",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "consecutive": 4,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "n_of_m_s3_t0.80_n4_m6_d0.40",
            "method": "n_of_m",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "n_hits": 4,
            "m_window": 6,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "hysteresis_s3_hi0.80_lo0.60_d0.40",
            "method": "hysteresis",
            "smooth_window": 3,
            "high_threshold": 0.80,
            "low_threshold": 0.60,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
    ]

    candidates = []
    for rule in manual_rules:
        payload = {
            "baseline_rule_name": rule["name"],
            "baseline_rule_settings": rule,
            "source": "manual_transition_representative_rules",
        }
        candidates.append({
            "path": None,
            "payload": payload,
            "rule_name": rule["name"],
            "rule_cfg": rule,
        })

    return candidates
'''


# ============================================================
# 批次版主流程：讀轉換資料 features csv 做推論
# ============================================================
def run_transition_postprocess_candidates(
    feature_csv,
    metadata_json,
    candidate_dir,
    output_dir=None,
    run_name="transition_postprocess_candidates",
):
    feature_csv = Path(feature_csv)
    metadata_json = Path(metadata_json)

    # 1.載入模型與資料
    #   從 metadata JSON 找出模型存放位置
    models_dir, model_id, model_folder = resolve_model_location_from_metadata(metadata_json)
    # 載入對應的 XGBoost 模型包
    model_id, pkg, manager = load_xgb_model_package(models_dir, model_id=model_id)
    model, scaler, metadata, model_type, settings = resolve_transition_inference_settings(pkg)
    # 讀入已經做好特徵工程的轉換資料 CSV
    df = pd.read_csv(feature_csv)

    # 2.算出每個 window 的預測機率與標籤，這段沿用你原本主流程：feature alignment、X、pred_proba、df_pred
    # 取出模型用到的特徵欄位、做 scaling
    feature_names = settings["feature_names"]
    X = df[feature_names].copy()
    if scaler is not None:
        X = scaler.transform(X)

    # 進模型算出洩漏機率
    pred_proba = model.predict_proba(X)[:, 1]
    # 再用 window_threshold（單一 window 層級的分類門檻，跟後處理規則裡的 alarm_threshold 是不同層次的東西）轉成 0/1 標籤
    pred_label = (pred_proba >= settings["window_threshold"]).astype(int)

    df_pred = df.copy()
    df_pred["pred_proba"] = pred_proba
    df_pred["pred_label"] = pred_label


    # 3.決定輸出資料夾
    if output_dir is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_folder / "inference_runs" / f"{run_name}_{run_ts}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_postprocess_candidates(candidate_dir)
    summary_rows = []

    # 4.對每組候選規則跑一次完整後處理 + 評估
    for item in candidates:
        rule_name = item["rule_name"]
        payload = item["payload"]
        
        # 把這組候選規則的參數套用到 base settings 上，產生這一輪要用的完整設定 rule_settings
        rule_settings = apply_postprocess_rule_to_transition_settings(settings, payload)
        cfg = {
            "name": rule_name,
            "method": rule_settings["alarm_method"],
        
            "alarm_threshold": rule_settings.get("alarm_threshold",rule_settings["window_threshold"],),
        
            "smooth_window": rule_settings["smooth_window"],
            "consecutive": rule_settings["consecutive"],
            "n_hits": rule_settings["n_hits"],
            "m_window": rule_settings["m_window"],
            "high_threshold": rule_settings["high_threshold"],
            "low_threshold": rule_settings["low_threshold"],
        
            "min_leak_duration_sec": rule_settings.get("min_leak_duration_sec"),
        
            "use_veto": rule_settings.get("use_veto",False,),
            "veto_column": rule_settings.get("veto_column"),
            "veto_threshold": rule_settings.get("veto_threshold"),
            "veto_rules": rule_settings.get("veto_rules",[],),
        
            "stage1": rule_settings.get("stage1"),
            "stage2": rule_settings.get("stage2"),
        }
        
        # 在 window 層級上套用這組規則的判斷邏輯，得到每個 window 的 detected_rule（這組規則判斷是否為警報）欄位，並且應該還保留了 label
        df_rule = apply_alarm_method_by_record(
            df_pred=df_pred,
            cfg=cfg,
            window_threshold=rule_settings["window_threshold"],
        )

        # 以每一次「事件」為單位的評估，算出這組規則對於「有沒有偵測到洩漏事件」「偵測延遲多久」「有沒有誤報」的表現
        onset_df = run_onset_evaluation(
            df_rule,
            window_sec=rule_settings["window_sec"],
            detected_col="detected_rule",
            smoothed_col="smoothed_proba_rule",
        )

        y_true = df_rule["label"].astype(int).values
        y_pred = df_rule["detected_rule"].astype(int).values

        summary_rows.append({
            "rule_name": rule_name,
            "min_leak_duration_sec": cfg.get("min_leak_duration_sec"),
            "window_precision": precision_score(y_true, y_pred, zero_division=0),
            "window_recall": recall_score(y_true, y_pred, zero_division=0),
            "window_f1": f1_score(y_true, y_pred, zero_division=0),
            "valid_detection_rate": onset_df["valid_detection"].mean() if "valid_detection" in onset_df.columns else np.nan,
            "false_alarm_rate": onset_df["false_alarm"].mean() if "false_alarm" in onset_df.columns else np.nan,
            "mean_latency_sec": onset_df["latency_sec"].dropna().mean() if "latency_sec" in onset_df.columns else np.nan,
        })

        df_rule.to_csv(output_dir / f"{rule_name}_window_predictions.csv", index=False, encoding="utf-8-sig")
        onset_df.to_csv(output_dir / f"{rule_name}_onset_results.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["valid_detection_rate", "false_alarm_rate", "mean_latency_sec"],
        ascending=[False, True, True],
    )
    summary_df.to_csv(output_dir / "transition_postprocess_candidate_comparison.csv", index=False, encoding="utf-8-sig")
    return summary_df





#                                      ********************************  主流程1:轉換資料  ********************************
# ============================================================
# 主流程：讀轉換資料 features csv 做推論
# ============================================================
'''
「把 transition 特徵表讀進來 -> 載模型 -> 做每個 window 的 leak 機率預測 -> 做 onset 判斷 -> 輸出結果 -> 再做幾個驗證」
'''
def run_transition_feature_inference(
    feature_csv,
    metadata_json,
    feature_config_json=None,
    postprocess_config_json=None,
    run_name="transition_holdout",
    do_onset_eval=True,
    do_alarm_compare=False,
    do_steady_summary=False,
    export_onset_summary=True,
):
    # 路徑統一轉成 Path 物件
    feature_csv = Path(feature_csv)
    metadata_json = Path(metadata_json)

    # 從 metadata.json 的路徑結構反推出 models_dir（模型總目錄）、model_id（這顆模型的資料夾名）、model_folder（這顆模型完整路徑）
    models_dir, model_id, model_folder = resolve_model_location_from_metadata(metadata_json)

    # 建立 ModelManager
    manager = ModelManager(models_dir=str(models_dir))
    model_id, pkg = get_xgb_model(manager, model_id=model_id)     # 用 get_xgb_model 把「你指定的這個 model_id」實際載入記憶體
    print(f"使用模型: {model_id}")
    print(f"模型資料夾: {model_folder}")


    feature_cfg = None
    if feature_config_json is not None:
        feature_cfg = load_feature_config(feature_config_json)
     
    # 讀後處理規則設定檔 postprocess_config.json，可選擇是否套用
    postprocess_payload = None
    if postprocess_config_json is not None:
        postprocess_payload = load_postprocess_config(postprocess_config_json)

    # 因為 feature_cfg=None，內部會退回用模型 metadata 自帶的 signal_processing.feature_config（如果有的話）來補齊 settings 的預設值
    model, scaler, metadata, model_type, settings = resolve_transition_inference_settings(
        pkg,
        feature_cfg=feature_cfg
    )
    
    # 用 disturbance holdout 選出來的正式規則（baseline_rule_settings）覆蓋掉 settings 裡對應的欄位，
    if postprocess_payload is not None:
        settings = apply_postprocess_rule_to_transition_settings(settings, postprocess_payload)
    
        print("套用正式 postprocess 規則做 transition 驗證")
        print("alarm_method     =", settings["alarm_method"])
        print("window_threshold =", settings["window_threshold"])
        print("smooth_window    =", settings["smooth_window"])
        print("consecutive      =", settings["consecutive"])
        print("n_hits           =", settings["n_hits"])
        print("m_window         =", settings["m_window"])
        print("high_threshold   =", settings["high_threshold"])
        print("low_threshold    =", settings["low_threshold"])

    # 讀入特徵 CSV
    df = pd.read_csv(feature_csv)
    
    # 做特徵對齊檢查
    check_feature_alignment(df, settings["feature_names"])

    # 如果模型訓練時有標準化，就先做 transform
    X = df[settings["feature_names"]].copy()
    if scaler is not None:
        X = scaler.transform(X)
    
    # 取的是「洩漏」這個類別（class 1）的機率
    pred_proba = model.predict_proba(X)[:, 1]
    pred_label = (pred_proba >= settings["window_threshold"]).astype(int)    # pred_label 用 settings["window_threshold"] 切門檻

    # 把預測結果併回原始資料表
    df_pred = df.copy()
    df_pred["pred_proba"] = pred_proba
    df_pred["pred_label"] = pred_label
    
    formal_cfg = {
        "method": settings["alarm_method"],
        "alarm_threshold": settings.get(
            "alarm_threshold",
            settings["window_threshold"],
        ),
        "smooth_window": settings["smooth_window"],
        "consecutive": settings["consecutive"],
        "n_hits": settings["n_hits"],
        "m_window": settings["m_window"],
        "high_threshold": settings["high_threshold"],
        "low_threshold": settings["low_threshold"],
        "min_leak_duration_sec": settings.get(
            "min_leak_duration_sec"
        ),
    
        "use_veto": settings.get("use_veto", False),
        "veto_column": settings.get("veto_column"),
        "veto_threshold": settings.get("veto_threshold"),
        "veto_rules": settings.get("veto_rules", []),
    
        "stage1": settings.get("stage1"),
        "stage2": settings.get("stage2"),
    }
    
    df_rule = apply_alarm_method_by_record(
        df_pred=df_pred,
        cfg=formal_cfg,
        window_threshold=settings["window_threshold"],
    )

    # 如果這份 CSV 有真實標籤 label，就實際算 window-level 的四項指標
    if "label" in df_rule.columns:
        # 額外印出模型原始結果，方便比較。
        raw_window_metrics = (
            evaluate_window_level(
                df_pred,
                pred_col="pred_label",
                result_name="模型原始 Window-level",
            )
        )
    
        # 正式輸出以套用後處理的 detected_rule 為準。
        window_metrics = (
            evaluate_window_level(
                df_rule,
                pred_col="detected_rule",
                result_name="正式後處理 Window-level",
            )
        )
    else:
        # 如果沒有 label（代表是沒有標註的現場資料），就給一組全 None/全零的假結果
        window_metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    # 只有在呼叫端要求做 onset 評估（do_onset_eval=True）而且資料裡真的有 onset 評估必須的四個欄位時，才會真的執行
    onset_result_df = None
    if do_onset_eval and {"record_id", "t_start", "onset_gt_sec", "pred_proba"}.issubset(df_pred.columns):
        onset_result_df = run_onset_evaluation(
            df_rule,
            window_sec=settings["window_sec"],
            detected_col="detected_rule",
            smoothed_col="smoothed_proba_rule",
        )
        
    # 先建立 output_dir
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = model_folder / "inference_runs" / f"{run_name}_{run_ts}"
    
    # 多種告警方法比較(可選)
    alarm_compare_df = None
    if do_alarm_compare:
        alarm_compare_df = compare_alarm_methods(
            df_pred,
            window_threshold=settings["window_threshold"],
            output_dir=output_dir,
            window_sec=settings["window_sec"],
        )

    # 穩態可分離度分析(可選)
    steady_summary_df = None
    if do_steady_summary:
        steady_summary_df = evaluate_steady_separability(
            df_pred,
            margins=(0.5, 1.0, 2.0, 3.0),
            window_sec=settings["window_sec"],
        )

    # 把這次跑出來的所有結果一次寫進 output_dir
    save_transition_inference_outputs(
        output_dir=output_dir,
        df_pred=df_rule,
        window_metrics=window_metrics,
        onset_result_df=onset_result_df,
        alarm_compare_df=alarm_compare_df,
        steady_summary_df=steady_summary_df,
        model_id=model_id,
        model_folder=model_folder,
        feature_csv=feature_csv,
        settings=settings,
        metadata=metadata,
    )

    # 如果有做 onset 評估而且呼叫端要求輸出摘要表，就再多產生一組依 volume/distance 分組的摘要表和失敗案例表
    if export_onset_summary and onset_result_df is not None:
        export_onset_summary_tables(
            df_pred=df_rule,
            onset_result_df=onset_result_df,
            output_dir=output_dir,
        )

    return {
        "df_pred": df_pred,        # df_pred：原始模型輸出
        "df_rule": df_rule,        # df_rule：正式規則結果
        "window_metrics": window_metrics,
        "onset_result_df": onset_result_df,
        "alarm_compare_df": alarm_compare_df,
        "steady_summary_df": steady_summary_df,
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

    # 已經做完 transition 特徵工程的 feature CSV 路徑
    FEATURE_CSV = Path(
        "/Users/paul/Desktop/Final/transition_feature_engineering/"
        "transition_30ml_test/transition_rule_features_holdout.csv"
    )
    
    # 特徵設定檔
    FEATURE_CONFIG_JSON = Path("/Users/paul/Desktop/Final/test_result/feature_config_micro.json")
    
    # 後處理規則路徑：目的是為了要拿干擾最佳規則回頭驗證轉換資料
    POSTPROCESS_DIR = Path(
        "/Users/paul/Desktop/Final/test_result/"
        "saved_models_transition_disturbance_final/xgb/"
        "xgb_20260923_164944"
    )
    
    manual_rules = [
        # 較寬鬆：觀察能否提升Transition事件召回率
        {
            "name": "threshold_s3_t0.70_d0.20",
            "method": "threshold",
            "alarm_threshold": 0.70,
            "smooth_window": 3,
            "min_leak_duration_sec": 0.20,
            "use_veto": False,
        },
        {
            "name": "consecutive_s3_t0.70_k2_d0.20",
            "method": "consecutive",
            "alarm_threshold": 0.70,
            "smooth_window": 3,
            "consecutive": 2,
            "min_leak_duration_sec": 0.20,
            "use_veto": False,
        },
        {
            "name": "consecutive_s3_t0.70_k3_d0.30",
            "method": "consecutive",
            "alarm_threshold": 0.70,
            "smooth_window": 3,
            "consecutive": 3,
            "min_leak_duration_sec": 0.30,
            "use_veto": False,
        },
        {
            "name": "n_of_m_s3_t0.70_n3_m5_d0.20",
            "method": "n_of_m",
            "alarm_threshold": 0.70,
            "smooth_window": 3,
            "n_hits": 3,
            "m_window": 5,
            "min_leak_duration_sec": 0.20,
            "use_veto": False,
        },
        {
            "name": "hysteresis_s3_hi0.70_lo0.50_d0.20",
            "method": "hysteresis",
            "smooth_window": 3,
            "high_threshold": 0.70,
            "low_threshold": 0.50,
            "min_leak_duration_sec": 0.20,
            "use_veto": False,
        },
    
        # 原本規則：保留作為較保守的比較基準
        {
            "name": "consecutive_s3_t0.80_k3_d0.40",
            "method": "consecutive",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "consecutive": 3,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "consecutive_s3_t0.85_k3_d0.40",
            "method": "consecutive",
            "alarm_threshold": 0.85,
            "smooth_window": 3,
            "consecutive": 3,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "consecutive_s3_t0.80_k4_d0.40",
            "method": "consecutive",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "consecutive": 4,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "n_of_m_s3_t0.80_n4_m6_d0.40",
            "method": "n_of_m",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "n_hits": 4,
            "m_window": 6,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
        {
            "name": "hysteresis_s3_hi0.80_lo0.60_d0.40",
            "method": "hysteresis",
            "smooth_window": 3,
            "high_threshold": 0.80,
            "low_threshold": 0.60,
            "min_leak_duration_sec": 0.40,
            "use_veto": False,
        },
    ]
        
    for i, rule in enumerate(manual_rules, start=101):
        payload = {
            "baseline_rule_name": rule["name"],
            "baseline_rule_settings": rule,
            "source": "manual_transition_representative_rule",
        }
    
        out_json = POSTPROCESS_DIR / f"postprocess_config_candidate_top{i}_{rule['name']}.json"
        out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
    
    # ============================================================
    # 1. 先跑候選規則比較
    # 會讀 postprocess_config_candidate_top*.json
    # 輸出 transition_postprocess_candidate_comparison.csv
    # ============================================================
    # 1. 比較所有候選規則在 transition 上的表現
    transition_comparison_df = (
        run_transition_postprocess_candidates(
            feature_csv=FEATURE_CSV,
            metadata_json=MODEL_METADATA_JSON,
            candidate_dir=POSTPROCESS_DIR,
            run_name="transition_postprocess_candidates",
        )
    )
    print("\nTransition 候選規則比較：")
    print(transition_comparison_df)
    
    
    # 2. 看完 comparison 表後，手動指定你要定案的規則名稱
    SELECTED_RULE_NAME = "consecutive_s3_t0.70_k3_d0.30"
    
    # 3. 將指定候選規則寫成 postprocess_config.json
    POSTPROCESS_CONFIG_JSON = (
        save_selected_candidate_as_final(
            candidate_dir=POSTPROCESS_DIR,
            selected_rule_name=SELECTED_RULE_NAME,
        )
    )
    
    # 補上模型 ID，讓 postprocess_config.json 和 metadata.json 對齊
    with open(MODEL_METADATA_JSON, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    with open(POSTPROCESS_CONFIG_JSON, "r", encoding="utf-8") as f:
        post_cfg = json.load(f)
    
    post_cfg["model_id"] = metadata.get("model_id")
    
    with open(POSTPROCESS_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(post_cfg, f, ensure_ascii=False, indent=2)
    
    # 4. 使用選定規則跑正式 transition 驗證
    # 如果這一步發生例外，下面的正式 schema 不會生成。
    transition_result = (
        run_transition_feature_inference(
             feature_csv=FEATURE_CSV,
             metadata_json=MODEL_METADATA_JSON,
             feature_config_json=FEATURE_CONFIG_JSON,
             postprocess_config_json=POSTPROCESS_CONFIG_JSON,
             run_name="transition_holdout",
             do_onset_eval=True,
             do_alarm_compare=False,
             do_steady_summary=False,
             export_onset_summary=True,
         )
    )
    
    # 5. Transition 正式驗證執行成功後，才把 candidate schema 升級為正式 schema。
    RUNTIME_SCHEMA_JSON = save_final_runtime_feature_schema(
        model_folder=POSTPROCESS_DIR,
        metadata_json=MODEL_METADATA_JSON,
        postprocess_config_json=POSTPROCESS_CONFIG_JSON,
    )
    
    print("\n正式部署檔案已完成：")
    print("postprocess:",POSTPROCESS_CONFIG_JSON,)
    print("runtime schema:", RUNTIME_SCHEMA_JSON,)
        
    '''
    輸出結構
    
    xgb_20260823_204020/
    ├── model.pkl
    ├── metadata.json
    ├── thresholds.json
    ├── inference_runs/
    ├── transition_holdout_20260822_153000/
        ├── run_info.json
        ├── window_predictions.csv
        ├── window_metrics.json
        ├── onset_results.csv
        └── plots/
            └── window_confusion_matrix.png
        
    '''

