#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xgb_classifier_two_sensor_micro_final.py

用途：
    用 Step2/Step3 選定的特徵（feature_config_micro.json 裡的 selected_feature_names），在合併好的 dev 資料上訓練一顆 XGBoost 二元分類器（正常 vs 洩漏），
    並分別在 transition holdout、disturbance holdout 兩份「從未參與訓練/調參/調閾值」的資料上做最終驗收評估，最後把模型、指標、圖表統一透過 ModelManager 存檔。

整體流程（對應 run_xgb_training_and_save）：
    1. 讀入三份特徵表：
        - dev_csv_path：合併過 transition_dev + disturbance_dev 的訓練資料（concat_transition_disturbance.py 的產物）
        - transition_holdout_csv_path / disturbance_holdout_csv_path：兩支切分腳本各自產生、完全沒碰過訓練流程的 holdout
       並用 feature_config_path 讀出 Step2 選定的正式特徵欄位清單，逐一檢查三份資料是不是都具備這些欄位（validate_feature_columns）。

    2. 呼叫 fit_xgb_final()，只用 dev 資料訓練模型，內部再細分三層：
        - sub-train：用 GroupShuffleSplit 依 record_id 切出 80%，搭配 RandomizedSearchCV + GroupKFold（一樣依 record_id 分組）
          找最佳 XGBoost 超參數，避免同一個 session 的 window 同時出現在 train/val 兩邊造成資料外洩。
        - val（切剩的 20%）：分別找出 window-level 最佳判斷閾值
          （_find_optimal_classifier_threshold）跟 file-level 最佳判斷閾值
          （先用 _aggregate_file_scores 把同一個 record_id 的 window 機率聚合成檔案分數，再用 _find_optimal_file_threshold 找閾值）。
        - full train：用整份 dev 資料、套用剛找到的最佳超參數，重新訓練出真正要拿去評估/存檔的 final_model。
       回傳的 model_bundle 裡包含 final_model、window_threshold、file_threshold、兩條閾值-分數曲線、以及 file_val_df（validation 上的檔案層級分數表）。

    3. 用同一顆 model_bundle，分別對兩份 holdout 呼叫 evaluate_trained_xgb()：
        - transition_holdout：do_file_level=False，只做 window-level 評估
         （transition holdout 裡有真正的正常/洩漏兩類，是主要看模型抓漏能力的指標）。
        - disturbance_holdout：do_file_level=True，同時做 window-level 跟 file-level 評估
         （disturbance holdout 全部是負樣本，主要用來看模型在機械干擾下的誤報率／抗干擾能力，不是看抓漏能力）。
         
       evaluate_trained_xgb 內部：先用 window_threshold 把機率轉成 window-level 預測、呼叫 _evaluate() 算 accuracy/precision/recall/f1/AUC/confusion_matrix；若 do_file_level=True，
       再用 _aggregate_file_scores + file_threshold 呼叫 _evaluate_file_level()算檔案層級混淆矩陣、_summarize_file_level() 整理成 file_accuracy/file_f1_score 等 dict。

    4. 組裝最終要存檔的 results 字典：
        - results["evaluations"]：把兩份 holdout 的完整結果各自保留一份（transition_holdout / disturbance_holdout），供以後想深入查任一份 holdout 細節時使用。
        - results["summary_metrics"]：給 ModelManager／之後查模型用的精簡摘要，主指標固定取 transition holdout 的 window-level 表現
          （primary_eval_name="transition_holdout"），額外保留 disturbance 的 window/file 表現當作抗干擾參考指標。
        - 為了跟舊版 ModelManager（讀取攤平的 results['f1_score'] 等 key）相容，另外把 transition 的 window 指標、disturbance 的 file 指標複製一份到最外層 results。

    5. 用 ModelManager.save_model() 把 final_model、上述 results 整包存檔，產生 model_id，模型檔、thresholds.json、metadata.json 會落在 ave_base_path / f"saved_models{save_suffix}" / xgb / {model_id}/。

    6. 畫圖並存檔到 save_base_path / f"xgb_plots{save_suffix}"：
        - window-level / file-level 閾值-分數曲線（各一張，兩份 holdout 共用同一組閾值，因為是從 dev 的 validation 找出來的）。
        - 對 transition、disturbance 兩份 holdout 各自畫 window-level confusion matrix；只有 disturbance（do_file_level=True）才會多畫file-level confusion matrix 跟 file-level score distribution。

名詞說明：
    - window-level：以單一個特徵 window（切片）為單位的預測/評估。
    - file-level：把同一個 record_id（同一次收集/同一個檔案）底下所有 window 的機率聚合成一個分數（目前用平均值 agg="mean"）之後才做的預測/評估，比 window-level 更貼近「這次收集到底有沒有洩漏」的實際判斷情境。
    - dev：訓練 + 調參 + 調閾值都會用到的資料（來自 transition_dev + disturbance_dev 合併）。
    - holdout：從頭到尾不參與訓練/調參/調閾值的資料，只在最後拿來做一次正式驗收，分 transition holdout（有正常+洩漏兩類，測抓漏能力）跟 disturbance holdout（全部負樣本，測抗干擾/誤報率）兩份。
"""
# ======================================
# 匯入模組跟設定 import 路徑
# ======================================
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# 保留導入 model_manager_two_sensor_final 路徑(實質用不到)
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from model_manager_two_sensor_final import ModelManager

from shared_feature_utils import (
    calculate_file_sha256,
    load_feature_extraction_config,
    load_fixed_normal_baseline,
)


# ======================================
# 從 JSON 設定檔裡讀出特徵欄位清單
# ======================================
def load_feature_config(feature_config_path):
    feature_config_path = Path(feature_config_path)

    cfg = json.loads(feature_config_path.read_text(encoding="utf-8"))

    feature_cols = cfg.get("selected_feature_names")

    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("feature_config_micro.json 缺少有效的 selected_feature_names")

    duplicate_features = sorted({
        feature_name
        for feature_name in feature_cols
        if feature_cols.count(feature_name) > 1
    })

    if duplicate_features:
        raise ValueError(f"selected_feature_names 出現重複特徵：{duplicate_features}")

    execution_plan = cfg.get("feature_execution_plan")

    if not isinstance(execution_plan, dict):
        raise ValueError("feature_config_micro.json 缺少 feature_execution_plan")

    planned_final_features = execution_plan.get("final_keep_features")

    if planned_final_features != feature_cols:
        raise ValueError("feature_execution_plan.final_keep_features 與 selected_feature_names 不一致")

    if cfg.get("baseline_mode") != "fixed_normal_baseline":
        raise ValueError("模型訓練只接受 baseline_mode=fixed_normal_baseline")

    return feature_cols, cfg


# ======================================
# 訓練契約檢查函式
# ======================================
def validate_training_feature_contract(
    feature_cfg,
    *,
    band_config_path,
    feature_extraction_config_path,
    baseline_json_path,
    require_final_baseline=True,
):
    band_config_path = Path(band_config_path)
    feature_extraction_config_path = Path(
        feature_extraction_config_path
    )
    baseline_json_path = Path(baseline_json_path)

    extraction_cfg = load_feature_extraction_config(
        feature_extraction_config_path
    )

    baseline_payload = load_fixed_normal_baseline(
        baseline_json_path,
        band_config_path=band_config_path,
        feature_extraction_config_path=(
            feature_extraction_config_path
        ),
        require_final=require_final_baseline,
    )

    current_band_hash = calculate_file_sha256(
        band_config_path
    )

    current_extraction_hash = calculate_file_sha256(
        feature_extraction_config_path
    )

    if (
        feature_cfg.get("band_config_sha256")
        != current_band_hash
    ):
        raise ValueError(
            "feature_config_micro.json 與目前 "
            "band_config_step2.json 不一致"
        )

    if (
        feature_cfg.get(
            "feature_extraction_config_sha256"
        )
        != current_extraction_hash
    ):
        raise ValueError(
            "feature_config_micro.json 與目前 "
            "feature_extraction_config.json 不一致"
        )

    if (
        feature_cfg.get("baseline_config_fingerprint")
        != baseline_payload.get("config_fingerprint")
    ):
        raise ValueError(
            "feature_config_micro.json 與目前 "
            "site_baseline.json 不一致"
        )
        
    # 檢查 feature config 記錄的 baseline 狀態
    if (feature_cfg.get("baseline_schema_status") != baseline_payload.get("schema_status")):
        raise ValueError("feature_config_micro.json 記錄的 baseline 狀態與目前 site_baseline.json 不一致；請使用正式 baseline 重新執行 Step 2")

    if (
        feature_cfg.get("baseline_base_feature_version")
        != extraction_cfg.get("base_feature_version")
    ):
        raise ValueError(
            "feature config 與 extraction config 的 "
            "base_feature_version 不一致"
        )

    if (
        baseline_payload.get("base_feature_version")
        != extraction_cfg.get("base_feature_version")
    ):
        raise ValueError(
            "site baseline 與 extraction config 的 "
            "base_feature_version 不一致"
        )

    return {
        "baseline_mode": "fixed_normal_baseline",
        "baseline_schema_status": baseline_payload.get(
            "schema_status"
        ),
        "baseline_config_fingerprint": (
            baseline_payload.get("config_fingerprint")
        ),
        "base_feature_version": extraction_cfg.get(
            "base_feature_version"
        ),
        "band_config_sha256": current_band_hash,
        "feature_extraction_config_sha256": (
            current_extraction_hash
        ),
        "band_config_path": str(band_config_path),
        "feature_extraction_config_path": str(
            feature_extraction_config_path
        ),
        "baseline_json_path": str(baseline_json_path),
    }


# ============================================================
# 訓練前檢查欄位是否對齊
# ============================================================
def validate_feature_columns(df,               # 要檢查的 DataFrame
                             feature_cols,     # 需要的特徵欄位清單
                             df_name):         # 表的名稱
    duplicate_columns = (
        df.columns[
            df.columns.duplicated()
        ]
        .unique()
        .tolist()
    )

    if duplicate_columns:
        raise ValueError(
            f"{df_name} 出現重複欄名："
            f"{duplicate_columns}"
        )

    missing_columns = [
        feature_name
        for feature_name in feature_cols
        if feature_name not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{df_name} 缺少模型特徵："
            f"{missing_columns[:20]}"
        )

    non_numeric_columns = [
        feature_name
        for feature_name in feature_cols
        if not pd.api.types.is_numeric_dtype(
            df[feature_name]
        )
    ]

    if non_numeric_columns:
        raise ValueError(
            f"{df_name} 出現非數值模型特徵："
            f"{non_numeric_columns[:20]}"
        )

    all_invalid_columns = []
    infinite_columns = []

    for feature_name in feature_cols:
        values = pd.to_numeric(
            df[feature_name],
            errors="coerce",
        )

        if np.isinf(values.to_numpy()).any():
            infinite_columns.append(feature_name)

        finite_values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if finite_values.notna().sum() == 0:
            all_invalid_columns.append(feature_name)

    if infinite_columns:
        raise ValueError(
            f"{df_name} 以下特徵包含 inf："
            f"{infinite_columns[:20]}"
        )

    if all_invalid_columns:
        raise ValueError(
            f"{df_name} 以下特徵完全沒有有效值："
            f"{all_invalid_columns[:20]}"
        )

    print(
        f"✅ {df_name}："
        f"{len(feature_cols)} 個模型特徵檢查通過"
    )


# ============================================================
# 取得每份資料的 record_id 集合
# ============================================================
def get_record_ids(df, df_name):
    if "record_id" not in df.columns:
        raise ValueError(
            f"{df_name} 缺少 record_id"
        )

    if df["record_id"].isna().any():
        raise ValueError(
            f"{df_name} 的 record_id 含有空值"
        )

    return set(
        df["record_id"]
        .astype(str)
        .tolist()
    )


# ============================================================
# 共用：閾值優化
# ============================================================
# 幫分類器找最佳分類閾值，不一定固定用 0.5
def _find_optimal_classifier_threshold(classifier, X_val, y_val, metric="f1", verbose=True):
    thresholds = np.arange(0.05, 0.95, 0.01)             # 建立候選 threshold 清單：從 0.05 到 0.94，每次加 0.01
    y_proba = classifier.predict_proba(X_val)[:, 1]      # 取驗證集每筆資料屬於正類 1 的機率
    results = []                                         # 建空 list，準備存每個 threshold 的評估結果

    # 遞迴所有 thresholds(機率大於等於 t 判成 1，否則判成 0)
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        # 把每個 threshold 對應的各種指標記錄下來
        results.append({
            "threshold": t,
            "precision": precision_score(y_val, y_pred, zero_division=0),    # zero_division=0：避免完全沒預測某類時報錯
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
            "accuracy": accuracy_score(y_val, y_pred),
        })

    # 將結果轉成df格式，找出指定指標最高的那一列
    df = pd.DataFrame(results)
    best_idx = df[metric].idxmax()
    best_t = float(df.loc[best_idx, "threshold"])
    best_s = float(df.loc[best_idx, metric])

    if verbose:     # 是否要印出資訊
        r = df.loc[best_idx]
        print(f"\n🔍 閾值優化 | 指標={metric.upper()} | 最佳閾值={best_t:.2f} | 分數={best_s:.3f}")
        print(f"  Precision={r['precision']:.3f}  Recall={r['recall']:.3f}"
              f"  F1={r['f1']:.3f}  Bal.Acc={r['balanced_accuracy']:.3f}")

    return best_t, df


# ============================================================
# 共用：window-level 評估
# ============================================================
# 對 window-level 預測做正式評估
def _evaluate(y_test, y_pred, y_pred_proba, name):
    # 計算分類指標
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 如果 y_test 真的有兩類，就算 ROC-AUC,如果只有一類或出錯，就回 None。
    try:
        auc_val = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else None
    except Exception:
        auc_val = None

    # 產生混淆矩陣
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n{'='*80}")
    print(f"📊 {name} 評估結果")
    print(f"{'='*80}")
    print(f"  準確率={acc:.1%}  精確率={prec:.1%}  召回率={rec:.1%}  F1={f1:.3f}")
    if auc_val is not None:
        print(f"  AUC={auc_val:.3f}")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["正常", "洩漏"], zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc_val,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


# ============================================================
# file-level 聚合
# ============================================================
# 把同一個 record_id 的多個 windows 聚合成一個檔案分數
def _aggregate_file_scores(df_eval, y_pred_proba, agg="mean"):
    tmp = df_eval[["record_id", "label"]].copy()      # 只保留需要的欄位
    tmp["proba"] = y_pred_proba                       # 把每個 window 的 leak probability 加進去

    rows = []
    # 依 record_id 分組,rid 是檔案 ID,g 是這個檔案底下所有 windows。
    for rid, g in tmp.groupby("record_id"):
        record_labels = set(
            pd.to_numeric(g["label"], errors="raise",).astype(int))
        
        if not record_labels.issubset({0, 1}):
            raise ValueError(f"record_id={rid} 出現非法 label：{sorted(record_labels)}")
        
        # 只要這個 record 曾出現 leak window，file-level 就視為正樣本
        true_label = int(g["label"].max())

        # 計算三種檔案層級分數
        mean_prob = float(g["proba"].mean())       # 所有 windows 的平均機率
        max_prob = float(g["proba"].max())         # 最高機率
        k = max(1, int(len(g) * 0.1))              # 先算這個檔案總window數的10%(int(len(g)*0.1)),但外面包一層max(1, ...),確保k至少是1
        topk_prob = float(g["proba"].sort_values().tail(k).mean())    # 最高 10% windows 的平均

        # 根據 agg 參數選出這次要用哪種分數做檔案判斷
        score = {
            "mean": mean_prob,
            "max": max_prob,
            "topk": topk_prob,
        }[agg]

        # 每個檔案的一列摘要存起來
        rows.append({
            "record_id": rid,
            "true_label": true_label,
            "mean_prob": mean_prob,         # 整個檔案所有window的機率全部平均(不管是transition還是disturbance,一個session裡通常是「正常→洩漏(或干擾)→正常」這種結構,真正的洩漏段落往往只佔整個檔案的一小部分時間,)
            "max_prob": max_prob,           # 整個檔案裡機率最高的那一個window
            "topk_prob": topk_prob,         # 這個檔案裡,leak機率最高的那前10%windows,它們的平均機率
            "score": score,
            "n_windows": len(g),
        })

    return pd.DataFrame(rows)


# ============================================================
# file-level threshold
# ============================================================
# 幫檔案層級找最佳 threshold
def _find_optimal_file_threshold(df_val, y_val_proba, metric="balanced_accuracy", agg="mean", verbose=True):
    thresholds = np.arange(0.05, 0.95, 0.01)     # 建立候選 threshold
    file_val = _aggregate_file_scores(df_val, y_val_proba, agg=agg)    # 把驗證集的 window 機率聚合成 file-level 分數

    results = []
    y_true = file_val["true_label"].values      # 取 file-level 真實標籤

    # 遍歷所有閥值 threshold
    for t in thresholds:
        y_pred = (file_val["score"] >= t).astype(int)
        # 記錄每個閥值的 precision、recall、f1、accuracy
        results.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
        })

    # 找出檔案層級最佳閥值
    df = pd.DataFrame(results)
    best_idx = df[metric].idxmax()
    best_t = float(df.loc[best_idx, "threshold"])
    best_s = float(df.loc[best_idx, metric])

    if verbose:
        r = df.loc[best_idx]
        print(f"\n🔍 File-level 閾值優化 | agg={agg} | 指標={metric.upper()}")
        print(f"  最佳閾值={best_t:.2f} | 分數={best_s:.3f}")
        print(f"  Precision={r['precision']:.3f}  Recall={r['recall']:.3f}"
              f"  F1={r['f1']:.3f}  Acc={r['accuracy']:.3f}")

    return best_t, df, file_val


# ============================================================
# file-level 評估
# ============================================================
# 用已經決定好的 threshold，正式評估檔案層級
def _evaluate_file_level(file_df, threshold, name="XGBoost File-level"):
    # 根據 threshold 把每個檔案分數轉成 0/1 預測標籤
    file_df = file_df.copy()
    file_df["pred_label"] = (file_df["score"] >= threshold).astype(int)
    
    # 計算檔案層級混淆矩陣
    cm = confusion_matrix(file_df["true_label"], file_df["pred_label"], labels=[0, 1])

    print(f"\n=== {name} ===")
    print(classification_report(
        file_df["true_label"],
        file_df["pred_label"],
        labels=[0, 1],
        target_names=["正常", "洩漏"],
        zero_division=0
    ))

    return file_df, cm


# ============================================================
# 把 file-level 指標整理成 dict
# ============================================================
def _summarize_file_level(file_df, threshold):
    # 根據 threshold 建一個 pred_label
    file_df = file_df.copy()
    file_df["pred_label"] = (file_df["score"] >= threshold).astype(int)

    # 取出真實與預測標籤
    y_true = file_df["true_label"].values
    y_pred = file_df["pred_label"].values

    return {
        "file_accuracy": accuracy_score(y_true, y_pred),
        "file_precision": precision_score(y_true, y_pred, zero_division=0),
        "file_recall": recall_score(y_true, y_pred, zero_division=0),
        "file_f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


# ============================================================
# XGBoost 最終訓練：純訓練，不碰 holdout 資料
# ============================================================
def fit_xgb_final(X_train,                    # 訓練特徵,注意這裡還帶著record_id欄位,還沒拆掉
                  y_train,                    # 訓練標籤
                  optimize_params=True,       # 要不要做超參數搜尋,預設要
                  optimize_threshold=True):   # 要不要做閾值優化,預設要
    print(f"\n{'='*80}")
    print("🎯 XGBoost - 最終訓練（調參 + 調閾值）")
    print(f"{'='*80}")

    # -- Step 1：檢查輸入格式 --
    #    確保資料格式正確,一定要有 record_id
    if "record_id" not in X_train.columns:
        raise ValueError("X_train 必須包含 record_id 欄位")

    # -- Step 2：拆出真正特徵欄位 --
    groups = X_train["record_id"]     # groups：之後切 validation 與 CV 用
    # 真正給模型吃的數值特徵
    X_train_features = X_train.drop(columns=["record_id"])     # 訓練特徵
    print(f"📊 訓練特徵數: {X_train_features.shape[1]}")

    # -- Step 3：從 train 裡再切一塊 validation -- 
    '''
    - 從原本的 dev/train 再切出一塊 validation
    - 這塊 validation 不是最終 holdout，而是用來：
         - 找最佳 threshold
         - 在需要時輔助調參流程
    '''
    # 如果選擇閥值優化
    if optimize_threshold:
        # 用 GroupShuffleSplit 避免同一檔案混到 train/val
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X_train_features, y_train, groups))

        X_train_sub = X_train_features.iloc[train_idx]
        y_train_sub = y_train[train_idx]
        X_val = X_train_features.iloc[val_idx]
        y_val = y_train[val_idx]
        groups_sub = groups.iloc[train_idx]
    
    else:      # 如果不做 threshold optimization，就直接用全部訓練資料
        X_train_sub = X_train_features
        y_train_sub = y_train
        X_val = y_val = None
        groups_sub = groups
       
    # 依y_train_sub裡正常/洩漏兩類的比例,算出每一筆資料訓練時該給多少權重, "balanced"會自動把少數類別的權重調高、多數類別的權重調低
    sample_weight_sub = compute_sample_weight(
        class_weight="balanced",
        y=y_train_sub
    )


    # 建立基礎 XGBoost 分類器
    base_model = XGBClassifier(
        objective="binary:logistic",      # 二元分類，輸出機率
        eval_metric="logloss",            # 訓練時的評估指標
        n_estimators=300,                 # 樹的數量
        learning_rate=0.05,               # 學習率
        max_depth=4,                      # 每棵樹的最大深度
        subsample=0.8,                    # 每棵樹訓練時隨機抽樣的資料比例
        colsample_bytree=0.8,             # 每棵樹訓練時隨機抽樣的特徵比例
        random_state=42,                  # 固定亂數種子
        n_jobs=-1,                        # 用滿所有CPU核心平行運算
    )

    # -- Step 4：在 train_sub 上找最佳模型參數 --
    '''
    如果要調整參數，就用 GroupKFold 做 CV
        - 用 RandomizedSearchCV + GroupKFold
        - 找到一組最好的 XGBoost 超參數
    '''
    best_params = {}
    if optimize_params:     # 如果選擇要超參數搜尋
        print("🔍 XGBoost RandomizedSearchCV ...")
        gkf = GroupKFold(n_splits=3)    # 3折交叉驗證
        # 設定超參數候選搜尋空間
        param_grid = {
            "n_estimators": [100, 200, 300, 500],        # (樹的數量)的候選值
            "learning_rate": [0.03, 0.05, 0.1],          # (學習率)的候選值
            "max_depth": [3, 4, 5, 6],                   # (樹的最大深度)的候選值
            "subsample": [0.7, 0.8, 0.9, 1.0],           # (每棵樹用的資料比例)的候選值
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],    # (每棵樹用的特徵比例)的候選值
        }

        # n_iter=10隨機抽10組參數組合去試, 每組都跑一次CV
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,            # 剛剛定義的搜尋空間
            n_iter=10,                                 # 隨機試 10 組參數
            cv=gkf.split(X_train_sub, y_train_sub, groups_sub),   # 傳入交叉驗證的切分方式  
            scoring="balanced_accuracy",    # 以 balanced accuracy 當選參數依據
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        
        # 在 sub-train 上搜尋,取出最佳模型與最佳參數
        search.fit(X_train_sub, y_train_sub, sample_weight=sample_weight_sub)
        model_for_threshold = search.best_estimator_     # 分數最好的那組參數所訓練出來的模型
        best_params = search.best_params_
        cv_balanced_accuracy_mean = float(search.best_score_)
        cv_balanced_accuracy_std = float(search.cv_results_["std_test_score"][search.best_index_])
        print(f"✅ 最佳參數: {best_params}  CV balanced_accuracy={search.best_score_:.4f}")
    
    else:    # 如果不調參，就直接訓練 base model 
        model_for_threshold = base_model
        model_for_threshold.fit(
            X_train_sub,
            y_train_sub,
            sample_weight=sample_weight_sub
        )
        
        cv_balanced_accuracy_mean = None
        cv_balanced_accuracy_std = None

    # -- Step 5：用 validation 找最佳 threshold --
    '''
    threshold optimization(用 validation 找最佳 window-level threshold)
    決定「幾分以上算 leak」,分兩層：
        - window_threshold
        - file_threshold
    '''
    # 如果設定要找最佳閥值
    if optimize_threshold:
        # 先呼叫 _find_optimal_classifier_threshold ,拿剛剛那顆 model_for_threshold 對 X_val 預測機率, ,掃過0.05~0.94這些候選閾值,找出讓balanced_accuracy最高的那個閾值
        # 存成 window_threshold (還有整條閾值-分數曲線 window_threshold_df)
        window_threshold, window_threshold_df = _find_optimal_classifier_threshold(
            model_for_threshold, X_val, y_val, metric="balanced_accuracy", verbose=True
        )
        
        # 取 validation 機率，給 file-level threshold 用
        y_val_proba = model_for_threshold.predict_proba(X_val)[:, 1]

        # 建一個只有 record_id 和 label 的 validation 表
        df_val_eval = X_train.iloc[val_idx][["record_id"]].copy()
        df_val_eval["label"] = np.asarray(y_val)
        
        # 用 validation 的 file-level 分數找最佳 file-level threshold
        file_threshold, file_threshold_df, file_val_df = _find_optimal_file_threshold(
            df_val_eval,     # record_id+label
            y_val_proba,     # 每個window的機率
            metric="balanced_accuracy",    # 評分指標
            agg="mean",   # 用每個檔案 windows 機率的平均值當檔案分數 
            verbose=True,
        )
    else:    # 如果不做 threshold optimization，就都用 0.5
        window_threshold = 0.5
        file_threshold = 0.5
        window_threshold_df = None
        file_threshold_df = None
        file_val_df = None

    # 有做超參數搜尋(optimize_params是True),且best_params不是空字典:用全量 train 重訓 final model(如果有找到最佳參數，就用這些參數重新建一顆 final model)
    if optimize_params and best_params:
        final_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **best_params,     # 帶入搜尋到的最佳超參數
        )
    else:     # 否則就沿用 base model
        final_model = base_model
        
    # 再呼叫一次compute_sample_weight,這次傳入完整的y_train(不是y_train_sub)
    sample_weight_full = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    # -- Step 6：用完整 train/dev 重訓最終模型 --
    '''
    前面已經知道：最佳超參數、最佳 threshold
    所以現在把整份 training/dev 全部拿來重新訓練，這才是最後真正的模型
    '''
    final_model.fit(X_train_features, y_train, sample_weight=sample_weight_full)
    
    return {
        "model": final_model,                            # 最終訓練好的模型
        "window_threshold": window_threshold,            # window層級的最佳閾值
        "file_threshold": file_threshold,                # 檔案層級的最佳閾值
        "window_threshold_curve": window_threshold_df,   # window層級閾值搜尋的完整曲線資料
        "file_threshold_curve": file_threshold_df,       # 檔案層級閾值搜尋的完整曲線資料
        "file_val_results": file_val_df,                 # validation上每個檔案聚合後的分數表
        "best_params": best_params,                      # 搜尋到的最佳超參數
        "scaler": None,                                  # 這版XGBoost流程沒有做特徵標準化
        "cv_balanced_accuracy_mean": cv_balanced_accuracy_mean,   
        "cv_balanced_accuracy_std": cv_balanced_accuracy_std,     
    }


# ============================================================
# 評估函式
# ============================================================
def evaluate_trained_xgb(model_bundle,           #  fit_xgb_final回傳的那個字典,有訓練好的模型跟兩個閾值
                         X_test,                 # 拿來評估的資料特徵,含record_id
                         y_test,                 # 對應的真實標籤
                         eval_name="Holdout",    # 這次評估的名稱
                         do_file_level=True):    # 要不要做file-level評估
    '''
    sub-train：拿來找模型參數
    val：拿來找 threshold
    full train：拿來練最終模型
    holdout：拿來做最後一次正式考試
    '''

    print(f"\n{'='*80}")
    print("🎯 XGBoost - 評估函式")
    print(f"{'='*80}")

    # -- Step 1：檢查輸入格式 --
    #    確保資料格式正確,一定要有 record_id
    if "record_id" not in X_test.columns:
        raise ValueError("X_test 必須包含 record_id 欄位")
    
    final_model = model_bundle["model"]     # 從 model_bundle 裡取出訓練好的最終模型,存成 final_model
    window_threshold = model_bundle["window_threshold"]    # 從 model_bundle 裡取出 window 層級的閾值
    file_threshold = model_bundle["file_threshold"]        # 從 model_bundle 裡取出 file 層級的閾值
    
    # 把record_id從X_test裡拿掉,得到純數值特徵矩陣
    X_test_features = X_test.drop(columns=["record_id"])

    # -- Step 2：用 holdout 做最後評估 --
    '''
    用從來沒參與過訓練、調參、調 threshold 的 holdout 做最終測試,分兩層：
       - window-level
       - file-level
    '''
    #    holdout prediction:對 holdout 預測每個 window 的機率
    y_pred_proba = final_model.predict_proba(X_test_features)[:, 1]

    # window-level:用剛剛挑出的 window threshold 轉成 window-level 預測標籤
    y_pred_window = (y_pred_proba >= window_threshold).astype(int)
    
    # 先做 window-level 正式評估
    results = _evaluate(
        y_test,
        y_pred_window,
        y_pred_proba,
        f"XGBoost ({eval_name}) Window-level"
    )

    # ----- file level(可選，轉換資料不用做file) -----
    # 準備 file-level 評估用資料
    if do_file_level:
        df_test_eval = X_test.copy()
        df_test_eval["label"] = y_test     # 在df_test_eval裡新增一欄label,值是y_test(真實標籤)
    
        # 把 holdout windows 聚合成檔案分數
        file_test_df = _aggregate_file_scores(
            df_test_eval,     # record_id+label
            y_pred_proba,     # 每個window的機率
            agg="mean"        # 用平均值當檔案分數
        )
    
        # 用 file threshold 做檔案層級正式評估
        file_test_df, file_cm = _evaluate_file_level(
            file_test_df,                # 剛剛聚合好的file_test_df
            threshold=file_threshold,    # 檔案層級閾值
            name=f"XGBoost ({eval_name}) File-level"
        )
    
        # 把 file-level 指標整理成 dict
        file_metrics = _summarize_file_level(file_test_df, file_threshold)
        # 把模型與各種 metadata 存進 results
        results["file_test_results"] = file_test_df   # 把聚合完、也有預測標籤的file_test_df,存進results
        results["file_confusion_matrix"] = file_cm    # 把檔案層級的混淆矩陣存進results
        results["aggregation"] = "mean"               # 聚合方式(寫死"mean")存進results
        results.update(file_metrics)
    else:
        # 如果do_file_level是False,走這個分支
        results["file_test_results"] = None
        results["file_confusion_matrix"] = None
        results["aggregation"] = None
        results["file_accuracy"] = None
        results["file_precision"] = None
        results["file_recall"] = None
        results["file_f1_score"] = None

    return results


# ============================================================
# label 防呆
# ============================================================
def validate_binary_labels(
    df,
    df_name,
    require_both_classes=False,
):
    if "label" not in df.columns:
        raise ValueError(f"{df_name} 缺少 label")

    if df["label"].isna().any():
        raise ValueError(f"{df_name} 的 label 含有空值")

    labels = set(
        pd.to_numeric(
            df["label"],
            errors="raise",
        ).astype(int)
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{df_name} 出現非法 label："
            f"{sorted(labels)}"
        )

    if require_both_classes and labels != {0, 1}:
        raise ValueError(
            f"{df_name} 必須同時包含 label 0 和 1，"
            f"目前為 {sorted(labels)}"
        )


#                                     **************************************  繪圖  *************************************
# ============================================================
# 閾值-分數曲線圖
# ============================================================
def plot_threshold_curve(df,               # 候選閾值對應各項分數的表
                         best_threshold,   # 找到的最佳閾值
                         metric,           # 畫哪個指標
                         title,            # 圖的標題
                         save_path):       # 存檔路徑
    # 檢查df是不是None,或者是空的DataFrame
    if df is None or df.empty:
        print(f"⚠️ {title} 沒有資料可畫")
        return

    # 建立一張新的畫布,寬8高5(英吋),解析度dpi=160
    plt.figure(figsize=(8, 5), dpi=160)
    # 畫一條折線,x 軸是 df["threshold"](候選閾值),y 軸是 df[metric](對應指標的分數)
    plt.plot(df["threshold"], df[metric], linewidth=2, label=metric)
    # 畫一條垂直虛線(紅色、虛線樣式), 標出「最佳閾值」
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"best={best_threshold:.2f}")
    plt.xlabel("Threshold")     # 設定x軸標籤文字
    plt.ylabel(metric)          # 設定y軸標籤文字,直接用metric這個字串
    plt.title(title)            # 設定整張圖的標題
    plt.grid(True, alpha=0.25)  # 打開格線,alpha=0.25設定格線的透明度
    plt.legend()                # 顯示圖例
    plt.tight_layout()          # 自動調整圖裡各元素的間距,避免標籤文字被裁切或重疊

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ 已儲存: {save_path}")


# ============================================================
# 畫檔案層級分數的分布直方圖
# ============================================================
def plot_file_score_distribution(file_df,       # 檔案層級聚合後的表
                                 threshold,     # 檔案層級閾值
                                 save_path,     # 存檔路徑
                                 score_col="score"):    # 要畫哪一欄當分數,預設是"score"
    # 檢查file_df是不是None或空表
    if file_df is None or file_df.empty:
        print("⚠️ file-level 分數表為空")
        return

    # 建立新畫布
    plt.figure(figsize=(8, 5), dpi=160)
    # 篩出 file_df 裡 true_label 等於 0(正常)的那些列,取出 score_col 那一欄
    normal_scores = file_df.loc[file_df["true_label"] == 0, score_col]
    # 篩出true_label等於1(洩漏)的那些列的分數,存成leak_scores
    leak_scores = file_df.loc[file_df["true_label"] == 1, score_col]
    
    # 畫normal_scores的直方圖,分成25個bin,透明度0.6,圖例標籤"Baseline"
    plt.hist(normal_scores, bins=25, alpha=0.6, label="Baseline", color="steelblue", density=True)
    # 畫leak_scores的直方圖,圖例標籤"Baseline+Leak",顏色番茄紅色
    plt.hist(leak_scores, bins=25, alpha=0.6, label="Baseline+Leak", color="tomato", density=True)
    # 在x座標threshold的位置畫一條垂直虛線(黑色),標出目前用的判斷閾值在整個分布圖上的位置
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")

    plt.xlabel("File-level score")
    plt.ylabel("Density")
    plt.title("File-level Score Distribution")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ 已儲存: {save_path}")


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
# 從資料裡解析出洩漏量
# ============================================================
def add_volume_column(df):    # 要處理的DataFrame
    df = df.copy()
    # 檢查df裡有沒有leak_quantity這一欄, 如果有,就從leak_quantity這欄解析出數字部分
    if "leak_quantity" in df.columns:
        # 把解析出來的數字存進新欄位volume_ml
        df["volume_ml"] = (
            df["leak_quantity"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")[0]
            .astype(float)
        )
        return df

    # 如果沒有leak_quantity這一欄, 就改成從record_id這個字串欄位裡解析
    df["volume_ml"] = (
        df["record_id"]
        .astype(str)
        .str.extract(r"_(\d+\.?\d*)ml_")[0]
        .astype(float)
    )
    return df
    

# ============================================================
# 整個訓練流程 + 存檔
# ============================================================
def run_xgb_training_and_save(
        dev_csv_path,                     # 合併好的dev資料路徑
        transition_holdout_csv_path,      # 轉換holdout路徑
        disturbance_holdout_csv_path,     # 干擾holdout路徑
        feature_config_path,              # 特徵設定檔路徑)
        band_config_path,
        feature_extraction_config_path,
        baseline_json_path,
        save_base_path,                   # 存模型/圖表的根目錄
        save_suffix="",                   # 存檔資料夾名稱要加的後綴,預設空字串
        description_text="",              # 存進模型描述用的文字,預設空字串
        require_final_baseline=True,
    ):
    # 指定特徵工程輸出的訓練與測試資料檔
    dev_csv_path = Path(dev_csv_path)
    transition_holdout_csv_path = Path(transition_holdout_csv_path)
    disturbance_holdout_csv_path = Path(disturbance_holdout_csv_path)
    feature_config_path = Path(feature_config_path)
    save_base_path = Path(save_base_path)
    band_config_path = Path(band_config_path)
    feature_extraction_config_path = Path(feature_extraction_config_path)
    baseline_json_path = Path(baseline_json_path)

    # 用迴圈掃過這四個路徑
    for p in [
        dev_csv_path,
        transition_holdout_csv_path,
        disturbance_holdout_csv_path,
        feature_config_path,
        band_config_path,
        feature_extraction_config_path,
        baseline_json_path,

    ]:
        if not p.exists():     # 檢查這個路徑對應的檔案存不存在
            raise FileNotFoundError(f"找不到檔案: {p}")

    # ========================================================
    # 讀 CSV 資料(dev資料, 轉換holdout資料, 干擾holdout資料)
    # ========================================================
    df_dev = pd.read_csv(dev_csv_path)
    df_transition_holdout = pd.read_csv(transition_holdout_csv_path)
    df_disturbance_holdout = pd.read_csv(disturbance_holdout_csv_path)
    
    
    # ========================================================
    # 檢查三份資料的 label 是否符合二元分類規格
    # ========================================================
    validate_binary_labels(df_dev, "dev", require_both_classes=True,)
    
    validate_binary_labels(df_transition_holdout, "transition_holdout", require_both_classes=True,)
    
    validate_binary_labels(df_disturbance_holdout, "disturbance_holdout", require_both_classes=False,)
        
    
    # ========================================================
    # 檢查 dev 與 holdout 是否有 record_id 重疊
    # ========================================================
    dev_record_ids = get_record_ids(df_dev,"dev",)
    transition_holdout_record_ids = get_record_ids(df_transition_holdout,"transition_holdout",)
    disturbance_holdout_record_ids = get_record_ids(df_disturbance_holdout,"disturbance_holdout",)
    
    overlap_checks = {"dev / transition_holdout": (dev_record_ids & transition_holdout_record_ids),
        "dev / disturbance_holdout": (dev_record_ids & disturbance_holdout_record_ids),}
    
    for split_pair, overlap_ids in overlap_checks.items():
        if overlap_ids:
            raise ValueError(f"{split_pair} 出現重複 record_id：{sorted(overlap_ids)[:20]}")
    print("✅ dev 與兩份 holdout 沒有 record_id 重疊")
    
    
    # ========================================================
    # 3. 補 volume_ml
    # ========================================================
    # 幫 df_dev, df_transition_holdout, df_disturbance_holdout 加上 volume_ml 欄位
    df_dev = add_volume_column(df_dev)
    df_transition_holdout = add_volume_column(df_transition_holdout)
    df_disturbance_holdout = add_volume_column(df_disturbance_holdout)
    
    
    # ========================================================
    # # 4. 讀取正式特徵設定
    # ========================================================
    # 讀出特徵欄位清單feature_cols跟整份設定字典feature_cfg
    feature_cols, feature_cfg = load_feature_config(feature_config_path)
    
    print("\nDev label counts:")
    print(df_dev["label"].value_counts())
    
    print("\nTransition Holdout label counts:")
    print(df_transition_holdout["label"].value_counts())
    print("\nTransition Holdout n_record_ids:")
    print(df_transition_holdout["record_id"].nunique())

    print("\nDisturbance Holdout label counts:")
    print(df_disturbance_holdout["label"].value_counts())
    print("\nDisturbance Holdout n_record_ids:")
    print(df_disturbance_holdout["record_id"].nunique())
    
    feature_contract = validate_training_feature_contract(
        feature_cfg,
        band_config_path=band_config_path,
        feature_extraction_config_path=(
            feature_extraction_config_path
        ),
        baseline_json_path=baseline_json_path,
        require_final_baseline=require_final_baseline,
    )
    
    print("\n=== 正式特徵契約 ===")
    print("baseline status:",feature_contract["baseline_schema_status"],)
    print("baseline fingerprint:",feature_contract["baseline_config_fingerprint"],)
    print("base feature version:",feature_contract["base_feature_version"],)

    # 檢查df_dev, df_transition_holdout, df_disturbance_holdout 裡有沒有齊全feature_cols裡列出的所有欄位
    validate_feature_columns(df_dev, feature_cols, "dev")
    validate_feature_columns(df_transition_holdout, feature_cols, "transition_holdout")
    validate_feature_columns(df_disturbance_holdout, feature_cols, "disturbance_holdout")

    # 從 df_dev 裡挑出 record_id 欄位加上所有 feature_cols 列出的特徵欄位
    X_train = df_dev[["record_id"] + feature_cols].copy()
    y_train = df_dev["label"].values    # 取出df_dev的label欄位,
    # 從轉換holdout挑出record_id+特徵欄位,存成X_transition
    X_transition = df_transition_holdout[["record_id"] + feature_cols].copy()
    y_transition = df_transition_holdout["label"].values
    # 從干擾holdout挑出對應欄位,存成X_disturbance
    X_disturbance = df_disturbance_holdout[["record_id"] + feature_cols].copy()
    y_disturbance = df_disturbance_holdout["label"].values

    # 訓練模型
    model_bundle = fit_xgb_final(
        X_train=X_train,
        y_train=y_train,
        optimize_params=True,
        optimize_threshold=True,
    )

    # 轉換這份不做file-level評估
    transition_results = evaluate_trained_xgb(
        model_bundle, X_transition, y_transition,
        eval_name="Transition Holdout",
        do_file_level=False,      # ← 轉換沒有做 file-level
    )
    
    # 干擾holdout的資料要做file-level評估
    disturbance_results = evaluate_trained_xgb(
        model_bundle, X_disturbance, y_disturbance,
        eval_name="Disturbance Holdout",
        do_file_level=True,       # ← 干擾才有做 file-level
    )

    results = dict(model_bundle)
    
    results["n_features"] = len(feature_cols)    # 特徵欄位的數量
    results["feature_names"] = feature_cols      # 整份特徵欄位清單
    results["feature_config"] = feature_cfg      # 整份特徵設定字典
    results["feature_config_path"] = str(feature_config_path)     # 把Path物件轉回字串存起來
    
    # window-level 跟 file-level 完整評估結果
    results["evaluations"] = {
        "transition_holdout": transition_results,
        "disturbance_holdout": disturbance_results,
    }
    
    # 精簡過的摘要指標：給 model manager / runtime / 之後查模型時看的摘要
    results["summary_metrics"] = {
        "primary_eval_name": "transition_holdout",
        
        # 主指標：transition holdout 的 window-level
        "window_accuracy": transition_results["accuracy"],
        "window_precision": transition_results["precision"],
        "window_recall": transition_results["recall"],
        "window_f1_score": transition_results["f1_score"],
        "window_roc_auc": transition_results["roc_auc"],
        
        # transition 不做 file-level，所以這裡保留 None
        "file_level_applicable": False,
        "file_accuracy": None,
        "file_precision": None,
        "file_recall": None,
        "file_f1_score": None,
        
        # 額外保留 disturbance 抗干擾表現
        "disturbance_window_accuracy": disturbance_results["accuracy"],
        "disturbance_window_precision": disturbance_results["precision"],
        "disturbance_window_recall": disturbance_results["recall"],
        "disturbance_window_f1_score": disturbance_results["f1_score"],
        "disturbance_window_roc_auc": disturbance_results["roc_auc"],
        
        "disturbance_file_level_applicable": True,
        "disturbance_file_accuracy": disturbance_results.get("file_accuracy"),
        "disturbance_file_precision": disturbance_results.get("file_precision"),
        "disturbance_file_recall": disturbance_results.get("file_recall"),
        "disturbance_file_f1_score": disturbance_results.get("file_f1_score"),
    }

    # 為了跟舊版 ModelManager 相容，top-level 也補一份，值取自 transition 的 window-level accuracy
    results["accuracy"] = transition_results["accuracy"]
    results["precision"] = transition_results["precision"]
    results["recall"] = transition_results["recall"]
    results["f1_score"] = transition_results["f1_score"]
    results["roc_auc"] = transition_results["roc_auc"]
    results["confusion_matrix"] = transition_results["confusion_matrix"]
    
    # file-level 改放 disturbance，因為 transition 這邊不做 file-level
    results["file_accuracy"] = disturbance_results.get("file_accuracy")
    results["file_precision"] = disturbance_results.get("file_precision")
    results["file_recall"] = disturbance_results.get("file_recall")
    results["file_f1_score"] = disturbance_results.get("file_f1_score")
    results["file_confusion_matrix"] = disturbance_results.get("file_confusion_matrix")

    # 新增signal_processing欄位,值是一個字典,記錄這次訓練用的資料前處理方式
    results["signal_processing"] = {
        "preprocess_type": "precomputed_feature_csv",      # 代表特徵是已經算好的csv
        "feature_config_path": str(feature_config_path),   # 特徵設定檔路徑
        "feature_config": feature_cfg,                     # 完整的特徵設定內容
        "band_config_path": str(band_config_path),
        "feature_extraction_config_path": str(feature_extraction_config_path),
        "baseline_json_path": str(baseline_json_path),
        "feature_contract": feature_contract,
    }

    # 建立一個 ModelManager 物件,傳入 models_dir 參數,值是 save_base_path 底下一個叫 saved_models{save_suffix} 的資料夾路徑
    manager = ModelManager(models_dir=str(save_base_path / f"saved_models{save_suffix}"))
    # 把模型檔案跟metadata實際寫進磁碟
    model_id = manager.save_model(
        model=results["model"],
        results=results,
        model_type="xgb",
        use_two_stage=False,
        model_name="XGB two-sensor micro final",
        description=description_text
    )

    print(f"✅ 已存入 ModelManager, model_id={model_id}")
    
    # --- 繪圖 ---
    plot_dir = save_base_path / f"xgb_plots{save_suffix}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 畫window-level的閾值-分數曲線圖
    plot_threshold_curve(
        results["window_threshold_curve"],
        best_threshold=results["window_threshold"],
        metric="balanced_accuracy",
        title="XGBoost Window-level Threshold Curve",
        save_path=plot_dir / "window_threshold_curve.png",
    )
    
    # 畫file-level的閾值-分數曲線圖
    plot_threshold_curve(
        results["file_threshold_curve"],
        best_threshold=results["file_threshold"],
        metric="balanced_accuracy",
        title="XGBoost File-level Threshold Curve",
        save_path=plot_dir / "file_threshold_curve.png",
    )
    
    # --- confusion matrix / file score distribution 兩份 holdout 各畫一次 ---
    for tag, holdout_results in [
        ("transition", transition_results),
        ("disturbance", disturbance_results),
    ]:
        # window-level confusion matrix：_evaluate() 一定會算，兩份都有
        plot_confusion_matrix(
            holdout_results["confusion_matrix"],
            class_names=["Baseline", "Leak"],
            title=f"XGBoost Window-level Confusion Matrix({tag})",
            save_path=plot_dir / f"{tag}_window_confusion_matrix.png",
        )
        
        # file-level：只有 do_file_level=True 的那份（目前是 disturbance）才有資料
        # 注意：threshold 是兩份 holdout 共用同一組，要從外層 results 抓，不是 holdout_results
        if holdout_results.get("file_test_results") is not None:
            plot_file_score_distribution(
                holdout_results["file_test_results"],
                threshold=results["file_threshold"],
                save_path=plot_dir / f"{tag}_file_score_distribution.png",
            )
        
        # 檢查這份holdout有沒有file-level混淆矩陣, 畫file-level混淆矩陣圖
        if holdout_results.get("file_confusion_matrix") is not None:
            plot_confusion_matrix(
                holdout_results["file_confusion_matrix"],
                class_names=["Baseline", "Leak"],
                title=f"XGBoost File-level Confusion Matrix ({tag})",
                save_path=plot_dir / f"{tag}_file_confusion_matrix.png",
            ) 
        
    return results, model_id


if __name__ == "__main__":
    # 合併好的 dev 資料
    DEV_CSV = Path("/Users/paul/Desktop/Final/test_result/transition_disturbance_dev.csv")
    # 轉換的 Holdout 資料
    TRANSITION_HOLDOUT_CSV = Path("/Users/paul/Desktop/Final/transition_feature_engineering/transition_30ml_test/transition_features_holdout.csv")
    # 干擾的 Holdout 資料
    DISTURBANCE_HOLDOUT_CSV = Path("/Users/paul/Desktop/Final/disturbance_feature_engineering/disturbance_features_holdout.csv")
    # 特徵 JSON 路徑(Step2/Step3選定特徵的設定檔路徑)
    FEATURE_CONFIG_JSON = Path("/Users/paul/Desktop/Final/test_result/feature_config_micro.json")
    
    BAND_CONFIG_JSON = Path("/Users/paul/Desktop/Final/test_result/band_config_step2.json")

    FEATURE_EXTRACTION_CONFIG_JSON = Path("/Users/paul/Desktop/Final/test_result/feature_extraction_config.json")
    
    BASELINE_JSON = Path("/Users/paul/Desktop/Final/test_result/site_baseline.json")
    
    # 儲存路徑
    SAVE_BASE_PATH = Path("/Users/paul/Desktop/Final/test_result")

    # 整個訓練+評估+存檔+畫圖的流程
    results, model_id = run_xgb_training_and_save(
        dev_csv_path=DEV_CSV,
        transition_holdout_csv_path=TRANSITION_HOLDOUT_CSV,
        disturbance_holdout_csv_path=DISTURBANCE_HOLDOUT_CSV,
        feature_config_path=FEATURE_CONFIG_JSON,
        band_config_path=BAND_CONFIG_JSON,feature_extraction_config_path=(FEATURE_EXTRACTION_CONFIG_JSON),
        baseline_json_path=BASELINE_JSON,
        save_base_path=SAVE_BASE_PATH,
        save_suffix="_transition_disturbance_final",
        description_text=(
            "Fixed normal baseline; "
            "transition + disturbance training; "
            "evaluated on transition and disturbance holdouts"
        ),
        require_final_baseline=False,
    )

    