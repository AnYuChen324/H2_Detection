#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
干擾推論與抗干擾規則比較流程總結

這支程式不是拿來重新訓練模型，也不是拿來重新搜尋 band / feature，
而是用「已訓練完成的模型」去讀取 disturbance 特徵工程輸出的 feature CSV，
評估模型在干擾資料上的誤報情況，並比較不同 alarm rule / veto 規則的表現。

這支程式在整個專案中的位置：
1. Step1：先找 resonance bands
2. Step2：用純正常 / 純洩漏資料找 leak-focused bands 與 selected features
3. Step3 / Step4：依 Step2 設定完成 transition / disturbance 特徵工程
4. 模型訓練：用 transition + disturbance training features 訓練並存模型
5. Transition 推論：看模型抓 leak / onset 的能力
6. Disturbance 推論（本程式）：看模型在干擾資料上的誤報情況，並比較最佳抗干擾規則

本程式的目的：
1. 讀取已訓練完成的模型：
   - model.pkl
   - metadata.json
   - thresholds.json
2. 讀入已完成特徵工程的 disturbance feature CSV
3. 檢查推論資料欄位是否和模型訓練時保存的 feature_names 一致
4. 對每個 window 輸出 pred_proba
5. 先用模型本身的 window_threshold 產生原始 pred_label
6. 把同一份 pred_proba 套用多組 alarm rule / veto 規則
7. 比較各種規則在干擾資料上的 false alarm 表現
8. 選定一條 baseline rule，輸出對應的分析摘要表
9. 額外輸出候選 postprocess 設定，供後續正式離線 / 即時推論使用

整體流程：
1. 根據 metadata.json 找到模型資料夾與 model_id
2. 載入模型 package，拆出：
   - model
   - scaler
   - feature_names
   - window_threshold
   - file_threshold
   - metadata
3. 讀取 disturbance inference feature CSV
4. 做 feature alignment 檢查：
   - 缺少欄位
   - 額外欄位
   - 特徵順序是否一致
5. 若資料中有 offband baseline 所需欄位，建立：
   - rel_offband_*
   - z_offband_*
   供後續 veto / broadband 規則判斷使用
6. 依模型保存的 feature_names 抽出 X，必要時做 scaler.transform
7. 模型輸出：
   - pred_proba
   - pred_label
8. 輸出 raw window prediction 結果
9. 執行 compare_disturbance_alarm_methods()：
   - threshold
   - consecutive
   - n_of_m
   - hysteresis
   - veto variants
   - two-stage 規則
10. 產生規則比較總表 summary_df
11. 根據 baseline_rule_name 選出最終分析用規則
12. 輸出 baseline rule 對應的：
   - record summary
   - phase summary
   - disturbance type summary
   - disturbance id summary
13. 產生 run_info.json
14. 額外輸出：
   - postprocess_config_candidate.json
   - 若確認採用，也可寫入正式 postprocess_config.json

重點輸出：
- raw window prediction csv
- disturbance_alarm_method_comparison.csv
- baseline rule 對應的各種 summary csv
- run_info.json
- postprocess_config_candidate.json
- postprocess_config.json（可選）

重要觀念：
1. 這支程式的重點不是重新找 band / feature，而是驗證模型在干擾資料上的抗誤報能力
2. 模型本身的 feature_names 與 threshold 以模型存檔內容為主
3. 干擾規則比較是模型後處理評估的一部分，不是模型重新訓練
4. 最佳抗干擾規則應先在這支程式中驗證，再提供給正式離線 / 即時推論流程直接使用
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json
import re

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_manager_two_sensor_final import ModelManager
from alarm_rule_utils import apply_alarm_method_by_record


#                                                ************************************************************
#                                           ****************************  共用工具函式  ********************************
# ============================================================
# 把 NaN 轉成可存 JSON 的小工具
# ============================================================
def _json_safe(obj):                 # 輸入一個 obj，可以是任何型別——字典、列表、tuple、NumPy 純量,甚至是一個普通的字串或數字
    # 處理 dict：遞迴進每個 value
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    # 處理 list：遞迴進每個元素
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    # 處理 tuple：也遞迴，但改成回傳 list
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    # 處理 NumPy 整數型別
    if isinstance(obj, np.integer):
        return int(obj)
    # 處理 NumPy 浮點數型別
    if isinstance(obj, np.floating):
        return float(obj)
    # 用 try/except 包住的 pd.isna 檢查
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


#                                                ************************************************************
#                                        ****************************  找出模型並核對資料  ********************************
# ============================================================
# 載入模型:從 ModelManager 裡找 XGB 模型並載入
# ============================================================
def get_latest_xgb_model(manager: ModelManager):
    # 跟 ModelManager 要所有 xgb 類型的模型，並依建立時間排序
    models = manager.list_models(model_type="xgb", sort_by="created_at")
    if not models:
        raise ValueError("找不到 XGB 模型")
    # 取排序後的第一個模型 ID。
    model_id = models[0]["model_id"]
    # 模型的完整 package 載進來。
    pkg = manager.load_model(model_id)
    return model_id, pkg      # 回傳模型 ID 和整個模型 package。


# ============================================================
# 載從模型 package 拆出推論需要的設定
# ============================================================
# 把推論需要的設定從模型包裡拆出來
def extract_inference_config(pkg):
    '''把 pkg 裡真正推論時需要用到的東西拆出來'''
    model = pkg["model"]                  # 取出真正的模型本體
    scaler = pkg.get("scaler", None)      # 有些模型訓練前會先做標準化，所以會一起存一個 scaler。如果這包模型沒有 scaler，就回傳 None(你這個 transition XGB 版本通常是沒有 scaler 的。)
    metadata = pkg.get("metadata", {})    # 把模型的描述資訊拿出來。通常會有：用了哪些特徵, 模型類型, 訓練表現, 頻帶設定
    model_type = metadata.get("model_type", None)     # 從 metadata 裡拿模型類型
    
    # 取得模型特徵名稱：從 metadata 裡取出「訓練時模型真正吃進去的特徵欄位名稱清單」, 推論時必須用完全相同的特徵欄位餵模型
    feature_names = metadata.get("features", {}).get("feature_names", None)
    # 檢查 feature names
    if not feature_names:
        raise ValueError("metadata.features.feature_names 不存在")
    # 取得 thresholds 設定(先從 package 取得 threshold 集合)
    thresholds = pkg.get("thresholds", {}) or {}

    # window_threshold 是將每個 window 的模型機率轉成初步 0/1 標籤的門檻
    # 第一順位：package 頂層 - 先找 package 頂層是否直接儲存 threshold
    window_threshold = pkg.get("window_threshold", None)    
    # 第二順位：thresholds - 如果頂層沒有，再找：pkg["thresholds"]["window_threshold"]
    if window_threshold is None:
        window_threshold = thresholds.get("window_threshold", None)
    # 第三順位：metadata performance - 如果前兩處都沒有，再嘗試讀取舊版 metadata 的模型評估結果
    if window_threshold is None:
        window_threshold = metadata.get("performance", {}).get("window_level", {}).get("threshold", None)
    # 最後保底:所有位置都沒有時，才使用預設值 0.5
    if window_threshold is None:
        window_threshold = 0.5

    # file_threshold 是檔案層級或 record 層級的判斷門檻, 把多個 window 的結果彙整成整支 TDMS 結果時，可能判斷：洩漏 window 比例 >= file_threshold → 整支檔案判定為 leak
    # 第一順位:先讀 package 頂層
    file_threshold = pkg.get("file_threshold", None)
    # 第二順位:再讀 thresholds 設定
    if file_threshold is None:
        file_threshold = thresholds.get("file_threshold", None)
    # 第三順位:再從舊版 metadata 的 file-level 評估結果取得
    if file_threshold is None:
        file_threshold = metadata.get("performance", {}).get("file_level", {}).get("threshold", None)
    # 最後保底:所有位置都沒有時，才使用預設值 0.5
    if file_threshold is None:
        file_threshold = 0.5

    return (
        model,
        scaler,
        feature_names,
        float(window_threshold),
        float(file_threshold),
        metadata,
        model_type,
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


#                                                ************************************************************
#                                            ****************************  特徵前處理  ********************************



#                                                ************************************************************
#                                          ****************************  誤報統計與報表  ********************************
# ============================================================
# disturbance 指標摘要
# ============================================================
def summarize_disturbance_rule(df_rule):
    '''
    把 alarm rule 的誤報表現整理成幾個摘要數字
    '''
    # 整體有多少比例的 windows 被警報規則打成 alarm
    row = {
        "overall_false_alarm_rate": df_rule["detected_rule"].mean(),
    }

    # 如果有 eval_class
    if "eval_class" in df_rule.columns:
        clean_mask = df_rule["eval_class"] == "clean_normal"          # 很乾淨的正常資料
        dist_mask = df_rule["eval_class"] == "disturbance_negative"   # 有干擾、但其實不該報警的資料

        # clean normal 的誤報率:在乾淨正常資料裡, 有多少比例被 rule 報成 alarm
        row["clean_normal_false_alarm_rate"] = (
            df_rule.loc[clean_mask, "detected_rule"].mean() if clean_mask.any() else np.nan
        )
        # disturbance negative 的誤報率:在有干擾、但其實不是 leak 的資料裡, 有多少比例被報警
        row["disturbance_negative_false_alarm_rate"] = (
            df_rule.loc[dist_mask, "detected_rule"].mean() if dist_mask.any() else np.nan
        )
    
    # 如果有 disturbance_type, 更細分不同干擾類型，例如：fa_knock, fa_valve_switch, fa_background_noise, 每一種都各算自己的誤報率。
    if "disturbance_type" in df_rule.columns:
        sub_df = df_rule[df_rule["disturbance_type"].fillna("") != ""].copy()
        for d_type, sub in sub_df.groupby("disturbance_type"):
            row[f"fa_{d_type}"] = sub["detected_rule"].mean()

    return row


# ============================================================
# record-level summary：把 window-level 的資料，整理成 record-level 摘要
# ============================================================
def build_record_level_summary(df_rule):
    # 建立一個空 list, 等每處理完一個 record_id，就把那個 record 的摘要字典塞進去
    rows = []

    # 依 record_id 分組
    for rid, sub in df_rule.groupby("record_id"):
        sub = sub.sort_values("t_start")     # 先把這個 record 裡的 windows 依開始時間 t_start 排序
        alarm_sub = sub[sub["detected_rule"] == 1]     # 從這個 record 的所有 windows 中，挑出那些被規則判定為 alarm 的 windows

        # 如果這個 record 有 alarm window：取第一個 alarm window 的 t_start,就是第一次報警時間
        first_alarm_time = alarm_sub["t_start"].iloc[0] if not alarm_sub.empty else np.nan

        rows.append({
            "record_id": rid,           # 記錄這一列是哪個 record_id
            "disturbance_type": sub["disturbance_type"].dropna().iloc[0] if "disturbance_type" in sub.columns and sub["disturbance_type"].notna().any() else "",      # 替這個 record 找一個代表性的干擾類型
            "eval_class": sub["eval_class"].dropna().iloc[0] if "eval_class" in sub.columns and sub["eval_class"].notna().any() else "",       # 描述這筆資料性質的標籤,clean_normal,disturbance_negative
            "max_pred_proba": sub["pred_proba"].max(),         # 這筆 record 裡，所有 windows 的 pred_proba 最大值,模型曾經有多相信這筆資料像 leak
            "max_smoothed_proba": sub["smoothed_proba_rule"].max(),     # 這筆 record 裡，所有 windows 的「平滑後機率」最大值
            "any_alarm": int((sub["detected_rule"] == 1).any()),        # 看這筆 record 是否曾經至少有一個 window 被報警
            "first_alarm_time": first_alarm_time,                       # 第一次報警時間
            "n_alarm_windows": int(sub["detected_rule"].sum()),         # alarm window 的個數
            "alarm_ratio": sub["detected_rule"].mean(),                 # alarm 比例,數值高，代表這筆 record 有很大一段時間都被判成 alarm。
        })

    return pd.DataFrame(rows)


# ============================================================
# before / during / after disturbance
# ============================================================
def build_phase_summary(df_rule):
    '''把每筆 record 依照干擾事件切成三段：干擾前/中/後'''
    # 檢查輸入表有沒有 overlap_ratio 欄位,如果沒有，就直接報錯。
    if "overlap_ratio" not in df_rule.columns:
        raise ValueError("df_rule 缺少 overlap_ratio，無法做 before/during/after 分析")

    # 準備空 list，等等每個 record + phase 都會塞一列摘要進去
    rows = []

    # 依 record_id 分組,每次迴圈就是處理一筆 record
    for rid, sub in df_rule.groupby("record_id"):
        sub = sub.sort_values("t_start").copy()    # 把這筆 record 依時間排序，並做一份 copy

        # 建立一個布林遮罩：True 代表這個 window 和干擾事件有重疊,False 代表沒有重疊
        during_mask = sub["overlap_ratio"].fillna(0) > 0
        if not during_mask.any():     # 如果這筆 record 沒有任何一個 window 和干擾重疊：就跳過不分析
            continue

        # 找出所有 during_mask == True 的位置索引
        during_idx = np.where(during_mask.values)[0]
        first_during = during_idx[0]    # 取出：第一個干擾窗的位置
        last_during = during_idx[-1]    # 取出：最後一個干擾窗的位置

        # 先新增一欄 phase，預設全部都標成 during_disturbance
        sub["phase"] = "during_disturbance"
        sub.iloc[:first_during, sub.columns.get_loc("phase")] = "before_disturbance"        # 把第一個干擾窗之前的所有 windows，改標成 before_disturbance
        sub.iloc[last_during + 1:, sub.columns.get_loc("phase")] = "after_disturbance"      # 把最後一個干擾窗之後的所有 windows，改標成 after_disturbance

        # 把這筆 record 依 phase 分組,每次處理一個 phase
        for phase_name, part in sub.groupby("phase"):
            rows.append({
                "record_id": rid,      # 這一列屬於哪個 record
                "disturbance_type": part["disturbance_type"].dropna().iloc[0] if "disturbance_type" in part.columns and part["disturbance_type"].notna().any() else "",     # 如果有 disturbance_type，就取第一個非空值當代表
                "phase": phase_name,    # 記錄干擾前/中/後
                "mean_pred_proba": part["pred_proba"].mean(),     # 這個 phase 裡，模型原始 leak 機率的平均值
                "alarm_rate": part["detected_rule"].mean(),       # 這個 phase 裡，被規則報警的比例
                "max_pred_proba": part["pred_proba"].max(),       # 這個 phase 裡，模型曾經給出的最高 leak 機率
            })

    return pd.DataFrame(rows)


# ============================================================
# disturbance-type ranking 干擾類型層級的摘要
# ============================================================
def build_disturbance_type_summary(df_rule):
    '''把很多 window-level 資料，依 disturbance_type 彙整，看看哪一種干擾最容易被誤判成 leak'''
    # 先檢查輸入資料裡有沒有 disturbance_type 欄位,如果沒有，這個函式沒辦法運作，直接報錯。
    if "disturbance_type" not in df_rule.columns:
        raise ValueError("df_rule 缺少 disturbance_type")
    # 準備一個空 list，等等每種干擾類型算完一列摘要就塞進去
    rows = []
    # 先把 disturbance_type 是空值或空字串的列排除掉
    sub_df = df_rule[df_rule["disturbance_type"].fillna("") != ""].copy()

    # 依 disturbance_type 分組。每次迴圈：d_type 是某一種干擾類型名稱,sub 是該類型底下所有 windows 的資料
    for d_type, sub in sub_df.groupby("disturbance_type"):
        # 先依 record_id 分組，再算每筆 record 的 alarm_ratio
        record_alarm_ratio = sub.groupby("record_id")["detected_rule"].mean()

        # 準備一個 list，等等收集這一種干擾裡，每筆 record 的第一次報警時間
        first_alarm_times = []
        # 在同一種干擾類型底下，再依 record_id 分組
        for _, rec in sub.groupby("record_id"):
            rec = rec.sort_values("t_start")      # 先把這筆 record 的 windows 按時間排序
            hit = rec[rec["detected_rule"] == 1]  # 把這筆 record 中，被規則判定為 alarm 的 windows 挑出來 
            first_alarm_times.append(hit["t_start"].iloc[0] if not hit.empty else np.nan)    # 如果這筆 record 有 alarm：取第一個 alarm window 的 t_start

        rows.append({
            "disturbance_type": d_type,                                # 記錄這一列對應的是哪種干擾類型。
            "n_records": sub["record_id"].nunique(),                   # 這種干擾一共出現在多少筆不同的 record 中
            "raw_false_alarm_rate": sub["pred_label"].mean(),          # 這種干擾下，原始模型直接輸出的誤報率。
            "rule_false_alarm_rate": sub["detected_rule"].mean(),      # 這種干擾下，經過後處理規則之後的誤報率(所有 windows 直接平均)
            "mean_first_alarm_time": np.nanmean(first_alarm_times),    # 把這種干擾裡每筆 record 的第一次報警時間取平均
            "mean_alarm_ratio": record_alarm_ratio.mean(),             # 把剛剛每筆 record 的 alarm ratio 再取平均(先每筆 record 算 alarm ratio，再對 record 平均)
        })

    return pd.DataFrame(rows).sort_values("rule_false_alarm_rate", ascending=False)


# ============================================================
# 整理大摘要表
# ============================================================
def export_disturbance_summary_tables(df_rule, output_dir, rule_name="baseline_rule"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 依 eval_class 摘要
    if "eval_class" in df_rule.columns:
        eval_class_summary = (
            df_rule.groupby("eval_class", dropna=False)
            .agg(
                n_windows=("record_id", "count"),
                false_alarm_rate=("detected_rule", "mean"),
                mean_pred_proba=("pred_proba", "mean"),
                mean_smoothed_proba=("smoothed_proba_rule", "mean"),
            )
            .reset_index()
            .sort_values("eval_class")
        )
    else:
        eval_class_summary = pd.DataFrame()

    # 2) 依 disturbance_type 摘要
    if "disturbance_type" in df_rule.columns:
        sub_type = df_rule[df_rule["disturbance_type"].fillna("") != ""].copy()
        if not sub_type.empty:
            type_summary = (
                sub_type.groupby("disturbance_type", dropna=False)
                .agg(
                    n_windows=("record_id", "count"),
                    false_alarm_rate=("detected_rule", "mean"),
                    mean_pred_proba=("pred_proba", "mean"),
                    mean_smoothed_proba=("smoothed_proba_rule", "mean"),
                )
                .reset_index()
                .sort_values("false_alarm_rate", ascending=False)
            )
        else:
            type_summary = pd.DataFrame()
    else:
        type_summary = pd.DataFrame()

    # 3) 依 disturbance_type × disturbance_strength 摘要
    if {"disturbance_type", "disturbance_strength"}.issubset(df_rule.columns):
        sub_ts = df_rule[df_rule["disturbance_type"].fillna("") != ""].copy()
        if not sub_ts.empty:
            type_strength_summary = (
                sub_ts.groupby(["disturbance_type", "disturbance_strength"], dropna=False)
                .agg(
                    n_windows=("record_id", "count"),
                    false_alarm_rate=("detected_rule", "mean"),
                    mean_pred_proba=("pred_proba", "mean"),
                    mean_smoothed_proba=("smoothed_proba_rule", "mean"),
                )
                .reset_index()
                .sort_values(["disturbance_type", "disturbance_strength"])
            )
        else:
            type_strength_summary = pd.DataFrame()
    else:
        type_strength_summary = pd.DataFrame()

    # 4) false alarm cases：只挑被 rule 判成 alarm 的干擾/正常案例
    keep_cols = [
        c for c in [
            "record_id", "t_start", "t_ref",
            "eval_class", "disturbance_type", "disturbance_strength",
            "pred_proba", "smoothed_proba_rule", "detected_rule"
        ]
        if c in df_rule.columns
    ]

    false_alarm_cases = df_rule.loc[df_rule["detected_rule"] == 1, keep_cols].copy()

    # 輸出
    if not eval_class_summary.empty:
        eval_class_summary.to_csv(
            output_dir / f"{rule_name}_eval_class_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not type_summary.empty:
        type_summary.to_csv(
            output_dir / f"{rule_name}_disturbance_type_overview.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not type_strength_summary.empty:
        type_strength_summary.to_csv(
            output_dir / f"{rule_name}_disturbance_type_strength_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    false_alarm_cases.to_csv(
        output_dir / f"{rule_name}_false_alarm_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"已輸出額外干擾摘要表: {rule_name}")


# ============================================================
# 比較哪個干擾不易辨識(全資料)
# ============================================================
def build_disturbance_id_summary(df_rule):
    '''
    用途是找出：
     - 到底是哪一個具體事件最難分辨
     - 而不只是知道哪一類型平均上比較難
    '''
    # 先列出這個函式一定需要哪些欄位
    required_cols = ["disturbance_id", "record_id", "pred_proba", "pred_label", "detected_rule"]
    # 檢查這些必要欄位有沒有缺
    missing = [c for c in required_cols if c not in df_rule.columns]
    if missing:
        raise ValueError(f"df_rule 缺少必要欄位: {missing}")

    rows = []      # 準備空 list，等等每個 disturbance_id 算完一列摘要就放進去。

    # 依 disturbance_id 分組,每次迴圈就是處理一個具體的干擾事件。
    for disturbance_id, sub in df_rule.groupby("disturbance_id"):
        sub = sub.sort_values("t_start")     # 把這個干擾事件底下的 windows 依時間排序

        # 把這個事件裡有被規則判成 alarm 的 windows 抓出來
        alarm_sub = sub[sub["detected_rule"] == 1]
        # 如果這個事件曾經報警：取第一個 alarm 的 t_start
        first_alarm_time = alarm_sub["t_start"].iloc[0] if not alarm_sub.empty else np.nan

        # 建立這個 disturbance_id 的摘要列
        rows.append({
            "disturbance_type": sub["disturbance_type"].dropna().iloc[0] if "disturbance_type" in sub.columns and sub["disturbance_type"].notna().any() else "",     # 如果有 disturbance_type 欄位而且至少有一個有效值：取第一個非空值當代表
            "disturbance_id": disturbance_id,     # 這個事件自己的 ID
            "record_id": sub["record_id"].dropna().iloc[0] if sub["record_id"].notna().any() else "",      # 取這個事件對應的 record_id
            "n_windows": len(sub),                # 這個事件一共有多少個 windows
            "n_files": sub["filename_stem"].nunique() if "filename_stem" in sub.columns else np.nan,       # 如果有 filename_stem 欄位，就算這個事件涉及幾個不同檔案
            "raw_false_alarm_rate": sub["pred_label"].mean(),       # 這個事件在原始模型輸出下的誤報率
            "rule_false_alarm_rate": sub["detected_rule"].mean(),   # 這個事件在後處理規則下的誤報率
            "alarm_ratio": sub["detected_rule"].mean(),             # 事件內 alarm 覆蓋比例
            "max_pred_proba": sub["pred_proba"].max(),              # 這個事件中，模型曾經給出的最高 leak 機率
            "mean_pred_proba": sub["pred_proba"].mean(),            # 這個事件所有 windows 的平均 leak 機率
            "first_alarm_time": first_alarm_time,                   # 第一次報警時間
            "n_alarm_windows": int(sub["detected_rule"].sum()),     # 總共有多少個 windows 被規則判成 alarm
        })

    return pd.DataFrame(rows).sort_values(
        ["rule_false_alarm_rate", "raw_false_alarm_rate", "max_pred_proba"],
        ascending=[False, False, False]
    )


#                                                ************************************************************
#                                          ****************************  抗干擾規則比較  ********************************
# ============================================================
# 比較多組規則
# ============================================================
def compare_disturbance_alarm_methods(
    df_pred,            # 模型已經先跑完推論, 已經有 pred_proba 的 DataFrame
    window_threshold,   # 模型本來的 window-level threshold, 在這裡主要是當 fallback 用, 如果某個 cfg 沒自己指定 alarm_threshold，就可以用它
    output_dir=None,    # 如果你有給資料夾, 每個 rule 套完的結果和總比較表都會存 CSV
):
    """
    一次比較多組 alarm rule，包含：
    「多種 alarm 判斷方法」×「多種 min_leak_duration_sec 秒數門檻」×「多種 veto 設定」全部組合出來
    1. 單階段規則
    2. 含 veto 的單階段規則
    3. two-stage 規則（短干擾先判，長事件再複核）

    回傳：
    - summary_df：每條規則的整體比較結果
    - rule_outputs：每條規則對應的完整 window-level 預測表
    """
    # ============================================================
    # 比較多組規則 + 多個 min_leak_duration_sec
    # ============================================================
    duration_candidates = [round(x, 2) for x in np.arange(0.20, 1.05, 0.05)]
    
    # 把每條基礎規則展開成多個秒數版本
    def expand_duration_grid(base_rules, duration_list):
        expanded = []
        # 跑每一條「基礎規則」
        for base in base_rules:
            # 跑每一個候選秒數
            for d in duration_list:
                cfg = base.copy()     # 本身只複製最外層的 dict
                cfg["name"] = f'{base["name"]}_d{d:.2f}'
                cfg["min_leak_duration_sec"] = d

                if "veto_rules" in base:
                    cfg["veto_rules"] = [rule.copy() for rule in base["veto_rules"]]
                if "stage1" in base:
                    cfg["stage1"] = base["stage1"].copy()
                if "stage2" in base:
                    cfg["stage2"] = base["stage2"].copy()

                expanded.append(cfg)
        return expanded

    base_rules = [
        {   # 先做 3 點平滑, 然後只要分數 >= 0.70 就報警
            "name": "threshold_s3_t0.70",
            "method": "threshold",
            "alarm_threshold": 0.70,
            "smooth_window": 3,
        },
        {   # 先做 3 點平滑, 門檻 0.80, 要連續 3 個 windows 都超過 0.80 才報警
            "name": "consecutive_s3_t0.80_k3",
            "method": "consecutive",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "consecutive": 3,
        },
        {   # 先做 3 點平滑, 門檻 0.80, 要連續 3 個 windows 都超過 0.85 才報警
            "name": "consecutive_s3_t0.85_k3",
            "method": "consecutive",
            "alarm_threshold": 0.85,
            "smooth_window": 3,
            "consecutive": 3,
        },
        {   # 先做 3 點平滑, 門檻 0.80, 要連續 4 個 windows 都超過 0.80 才報警
            "name": "consecutive_s3_t0.80_k4",
            "method": "consecutive",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "consecutive": 4,
        },
        {   # 最近 6 個 windows 裡, 只要有 4 個超過 0.80, 就算觸發
            "name": "n_of_m_s3_t0.80_4of6",
            "method": "n_of_m",
            "alarm_threshold": 0.80,
            "smooth_window": 3,
            "n_hits": 4,
            "m_window": 6,
        },
        {   # 開警報要衝到 0.80, 開了以後要掉到 0.60 以下才關掉
            "name": "hysteresis_s3_hi0.80_lo0.60",
            "method": "hysteresis",
            "smooth_window": 3,
            "high_threshold": 0.80,
            "low_threshold": 0.60,
        },
        {   # 兩階段
            "name": "two_stage_shortk3_longk4ratio",
            "method": "two_stage",
            "use_veto": True,
            "stage1": {
                "method": "consecutive",
                "alarm_threshold": 0.85,
                "smooth_window": 3,
                "consecutive": 3,
            },
            "stage2": {
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "trigger_run_length": 4,
            },
            "veto_rules": [
                {"col": "offbandmax_to_leak_focusmax_E_ratio", "op": "ge", "threshold": 1.5},
            ],
        },
    ]

    configs = expand_duration_grid(base_rules, duration_candidates)

    # 三組「看資料裡有沒有這些欄位才加」的規則組
    # 檢查「這幾個欄位是不是都存在於 df_pred 裡」,存在才把對應那組規則（各 3 種門檻鬆緊：lo/mid/hi）加進 configs
    has_raw_veto_cols = {"offband_E_mean", "offband_E_max"}.issubset(df_pred.columns)
    if has_raw_veto_cols:     
        raw_veto_rules = [
            {   # veto 版本,主規則 smooth_window = 3 + alarm_threshold = 0.80 + consecutive = 4
                "name": "consecutive_s3_t0.80_k4_with_veto_lo",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    # 只要 offband 稍微升一點就 veto,這很敏感，可能擋掉很多 alarm。
                    {"col": "offband_E_mean", "op": "ge", "threshold": 1e-4},
                    {"col": "offband_E_max", "op": "ge", "threshold": 1e-3},
                ],
            },
            {   
                "name": "consecutive_s3_t0.80_k4_with_veto_mid",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    {"col": "offband_E_mean", "op": "ge", "threshold": 1e-3},
                    {"col": "offband_E_max", "op": "ge", "threshold": 1e-2},
                ],
            },
            {
                "name": "consecutive_s3_t0.80_k4_with_veto_hi",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    # 要 offband 很明顯升高才 veto,最寬鬆
                    {"col": "offband_E_mean", "op": "ge", "threshold": 1e-2},
                    {"col": "offband_E_max", "op": "ge", "threshold": 1e-1},
                ],
            },
        ]
        configs.extend(expand_duration_grid(raw_veto_rules, duration_candidates))
    else:
        print("⚠️ 缺少 offband_E_mean / offband_E_max，跳過 raw veto 規則")

    has_zveto_cols = {"z_offband_E_mean", "z_offband_E_max"}.issubset(df_pred.columns)
    if has_zveto_cols:
        zveto_rules = [
            {
                "name": "consecutive_s3_t0.80_k4_zveto_lo",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    {"col": "z_offband_E_mean", "op": "ge", "threshold": 2.0},
                    {"col": "z_offband_E_max", "op": "ge", "threshold": 2.5},
                ],
            },
            {
                "name": "consecutive_s3_t0.80_k4_zveto_mid",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    {"col": "z_offband_E_mean", "op": "ge", "threshold": 2.5},
                    {"col": "z_offband_E_max", "op": "ge", "threshold": 3.0},
                ],
            },
            {
                "name": "consecutive_s3_t0.80_k4_zveto_hi",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    {"col": "z_offband_E_mean", "op": "ge", "threshold": 3.0},
                    {"col": "z_offband_E_max", "op": "ge", "threshold": 4.0},
                ],
            },
        ]
        configs.extend(expand_duration_grid(zveto_rules, duration_candidates))
    else:
        print("⚠️ 缺少 z_offband_E_mean / z_offband_E_max，跳過 zveto 規則")

    has_ratio_veto_col = "offbandmax_to_leak_focusmax_E_ratio" in df_pred.columns
    if has_ratio_veto_col:
        ratio_veto_rules = [
            {
                "name": "consecutive_s3_t0.80_k4_maxratio_veto_lo",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    {"col": "offbandmax_to_leak_focusmax_E_ratio", "op": "ge", "threshold": 1.5},
                ],
            },
            {
                "name": "consecutive_s3_t0.80_k4_maxratio_veto_mid",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    {"col": "offbandmax_to_leak_focusmax_E_ratio", "op": "ge", "threshold": 2.0},
                ],
            },
            {
                "name": "consecutive_s3_t0.80_k4_maxratio_veto_hi",
                "method": "consecutive",
                "alarm_threshold": 0.80,
                "smooth_window": 3,
                "consecutive": 4,
                "use_veto": True,
                "veto_rules": [
                    {"col": "offbandmax_to_leak_focusmax_E_ratio", "op": "ge", "threshold": 3.0},
                ],
            },
        ]
        configs.extend(expand_duration_grid(ratio_veto_rules, duration_candidates))
    else:
        print("⚠️ 缺少 offbandmax_to_leak_focusmax_E_ratio，跳過 ratio veto 規則")
    
    summary_rows = []     # 存每條 rule 的摘要結果
    rule_outputs = {}     # 存每條 rule 跑完後的完整 df_rule

    # 逐一跑每種規則
    for cfg in configs:
        # 套規則到每個 record
        df_rule = apply_alarm_method_by_record(df_pred, cfg, window_threshold)
     
        # ule_outputs 用 cfg["name"] 當 key,把每條規則的完整 window-level 結果都存起來
        rule_outputs[cfg["name"]] = df_rule.copy()
        
        # 先記錄這個規則本身的設定
        row = {
            "name": cfg["name"],
            "method": cfg["method"],
            "window_threshold": window_threshold,
            "alarm_threshold": cfg.get("alarm_threshold", np.nan),
            "smooth_window": cfg.get("smooth_window", np.nan),
            "consecutive": cfg.get("consecutive", np.nan),
            "n_hits": cfg.get("n_hits", np.nan),
            "m_window": cfg.get("m_window", np.nan),
            "high_threshold": cfg.get("high_threshold", np.nan),
            "low_threshold": cfg.get("low_threshold", np.nan),
        
            # 正式 runtime event postprocess 也要一起存
            "min_leak_duration_sec": cfg.get("min_leak_duration_sec", 0.4),
            "disturbance_ratio_col": cfg.get("disturbance_ratio_col", None),
            "disturbance_ratio_threshold": cfg.get("disturbance_ratio_threshold", np.nan),
        
            # veto 規則本身
            "veto_column": cfg.get("veto_column", None),
            "veto_threshold": cfg.get("veto_threshold", np.nan),
            "veto_rules": cfg.get("veto_rules", []),
            
            "stage1": cfg.get("stage1"),
            "stage2": cfg.get("stage2"),
        }
        
        # veto 相關指標
        row["use_veto"] = cfg.get("use_veto", False)         # 這條規則有沒有用 veto
        row["veto_rate"] = df_rule["veto_flag"].mean() if "veto_flag" in df_rule.columns else np.nan       # 有多少比例的 windows 被 veto
        row["raw_alarm_rate"] = df_rule["detected_rule_raw"].mean() if "detected_rule_raw" in df_rule.columns else np.nan       # 不考慮 veto 時，原本會報警的比例
        row["final_alarm_rate"] = df_rule["detected_rule"].mean()       # 加了 veto 之後，最後真的報警的比例
        row["alarm_drop"] = row["raw_alarm_rate"] - row["final_alarm_rate"]       # veto 幫你壓掉了多少 alarm
                
        # 把誤報摘要加進來
        row.update(summarize_disturbance_rule(df_rule))
        summary_rows.append(row)     # 把這一列加入總表

        # 可選：把每個 rule 的 window 預測表存起來
        if output_dir is not None:
            save_path = Path(output_dir) / f"{cfg['name']}_window_predictions.csv"
            df_rule.to_csv(save_path, index=False, encoding="utf-8-sig")

    # 把全部規則的摘要整理成 DataFrame, 排序規則是：
    # 最優先：在「有干擾但不是 leak」的情況下，不要亂報
    # 第二優先：在乾淨正常資料中，也不要亂報
    # 第三優先：整體 false alarm rate 也盡量低
    summary_df = pd.DataFrame(summary_rows).sort_values(
        [
            "disturbance_negative_false_alarm_rate",     # 先看 disturbance_negative_false_alarm_rate
            "clean_normal_false_alarm_rate",             # 再看 clean_normal_false_alarm_rate
            "overall_false_alarm_rate",                  # 再看 overall_false_alarm_rate
        ],
        ascending=[True, True, True],
    )

    # 可選輸出總比較表
    if output_dir is not None:
        summary_df.to_csv(
            Path(output_dir) / "disturbance_alarm_method_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return summary_df, rule_outputs


#                                                ************************************************************
#                                          ****************************  正式輸出／設定檔  ********************************
# ============================================================
# 儲存後處理較佳規則成 postprocess_config
# ============================================================  
def save_postprocess_configs(
    model_folder,
    model_id,
    summary_df,
    selected_rule_name=None,                       # 代表預設情況下你還沒定案要用哪個規則
    source="disturbance_holdout_rule_selection",   # 是干擾規則在 holdout 資料上選出來的結果
    write_final=False,                             # 預設 False,代表預設不會寫出正式版 postprocess_config.json
    top_k_candidates=5,                            # 決定要輸出前幾名候選當備查用
):
    '''
    把 summary_df（干擾規則篩選跑完之後、每一列代表一種候選規則設定的比較表）轉換成正式的 postprocess_config.json 及其候選版本,
    寫到硬碟上,供之後 model_infer_transition_final.py 裡的 load_postprocess_config() 讀取
    '''
    # 建資料夾、檢查輸入
    model_folder = Path(model_folder)    # 先把 model_folder 轉成 Path
    model_folder.mkdir(parents=True, exist_ok=True)     # 確保這個資料夾都存在,資料夾已存在也不會報錯
    
    # 接著檢查 summary_df——如果是 None 或是空的 DataFrame（沒有任何規則可比較）,就直接擋下來
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df 為空，無法輸出 postprocess config")
    
    # summary_df 必要欄位檢查
    required_cols = {"name", "method", "window_threshold", "alarm_threshold"}
    missing_cols = required_cols - set(summary_df.columns)
    if missing_cols:
        raise ValueError(f"summary_df 缺少必要欄位: {missing_cols}")

    # 內部輔助函式：把一列規則整理成統一欄位結構
    def build_baseline_rule_settings(selected_row):
        '''
        把 summary_df 裡一整列（可能有幾十個欄位,包含各種統計數字、中間過程欄位）
        挑出「正式 postprocess 流程真正需要的那一組固定欄位」,整理成一個結構統一的 dict。
        '''
        settings = {
            "name": selected_row.get("name"),
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
            "disturbance_ratio_col": selected_row.get("disturbance_ratio_col"),
            "disturbance_ratio_threshold": selected_row.get("disturbance_ratio_threshold"),

            "use_veto": selected_row.get("use_veto", False),       # use_veto 是「要不要啟用 veto 機制」的開關,如果這個欄位不存在,語意上應該解讀成「沒有啟用」（False）
            "veto_column": selected_row.get("veto_column"),
            "veto_threshold": selected_row.get("veto_threshold"),
            "veto_rules": selected_row.get("veto_rules", []),      # veto_rules 給 [] 也是同樣道理——下游程式如果拿這個值去做 for rule in veto_rules: 迴圈,[] 可以安全地跑「零次」

            "stage1": selected_row.get("stage1"),
            "stage2": selected_row.get("stage2"),
        
            "veto_rate": selected_row.get("veto_rate"),
            "raw_alarm_rate": selected_row.get("raw_alarm_rate"),
            "final_alarm_rate": selected_row.get("final_alarm_rate"),
            "alarm_drop": selected_row.get("alarm_drop"),
            "overall_false_alarm_rate": selected_row.get("overall_false_alarm_rate"),
            "clean_normal_false_alarm_rate": selected_row.get("clean_normal_false_alarm_rate"),
            "disturbance_negative_false_alarm_rate": selected_row.get("disturbance_negative_false_alarm_rate"),
            "fa_adjacent_vibration": selected_row.get("fa_adjacent_vibration"),
            "fa_cable_pull": selected_row.get("fa_cable_pull"),
            "fa_touch_pipe": selected_row.get("fa_touch_pipe"),
        }
        
        # use_veto 開啟了,卻沒有給依據欄位/門檻 —— 半套設定,擋下來
        has_single_veto = (
            settings["veto_column"] is not None and settings["veto_threshold"] is not None)
        
        has_veto_rules = bool(settings["veto_rules"])
        
        if (settings["use_veto"] and not has_single_veto and not has_veto_rules):
            raise ValueError(f"規則 {settings['name']} 啟用 veto，但沒有完整的 veto_column/veto_threshold，也沒有 veto_rules")
    
        # disturbance_ratio_col 跟 disturbance_ratio_threshold 只給了一半 —— 也是半套設定
        has_ratio_col = settings["disturbance_ratio_col"] is not None
        has_ratio_thresh = settings["disturbance_ratio_threshold"] is not None
        if has_ratio_col != has_ratio_thresh:
            raise ValueError(
                f"規則 {settings['name']} 的 disturbance_ratio_col/"
                f"disturbance_ratio_threshold 只設定了一半"
            )
    
        return settings
            

    # Step 1：輸出前 K 名候選規則，各自存成一個檔案（預設前 5 列）
    top_df = summary_df.head(top_k_candidates).copy()
    candidate_payloads = []

    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):     # top_df.iterrows() 會逐列走過 DataFrame,每次給你 (index, row)
        selected_row = _json_safe(row.to_dict())        # 把裡面所有 NumPy 型別、NaN 都轉成 JSON 能吃的原生型別
        
        if not selected_row.get("name"):
            raise ValueError(f"summary_df 第 {rank} 名候選規則缺少 name 欄位")
        baseline_rule_settings = build_baseline_rule_settings(selected_row)    # 呼叫剛剛定義的內部函式,從整列資料裡挑出正式要用的固定欄位子集

        # 組出這個候選規則最終要寫進 JSON 的完整內容
        payload = {
            "model_id": model_id,                                              # 這份設定是給哪個模型用的
            "selected_at": datetime.now().isoformat(),                         # 用 datetime.now().isoformat() 記下產生時間,方便之後追溯是哪次跑出來的
            "source": source,                                                  
            "candidate_rank": rank,                                            # 名次
            "baseline_rule_name": selected_row["name"],                         
            "baseline_rule_settings": _json_safe(baseline_rule_settings),      # 規則本身的設定
        }

        # 把 payload 寫成 JSON 檔
        safe_name = re.sub(r"[^\w\-.]", "_", str(selected_row["name"]))     # 把 safe_name 真的做一次檔名安全化

        candidate_path = model_folder / f"postprocess_config_candidate_top{rank}_{safe_name}.json"
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"✅ 已輸出候選 postprocess config: {candidate_path}")
        candidate_payloads.append(payload)

    # 保留一份舊格式的總候選，指向第一名
    recommended_row = _json_safe(summary_df.iloc[0].to_dict())     # 直接取排序後的第一列
    candidate_payload = {
        "model_id": model_id,
        "selected_at": datetime.now().isoformat(),
        "source": source,
        "recommended_rule_name": recommended_row["name"],
        "recommended_rule_settings": recommended_row,
    }

    candidate_path = model_folder / "postprocess_config_candidate.json"
    with open(candidate_path, "w", encoding="utf-8") as f:
        json.dump(candidate_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ 已輸出總候選 postprocess config: {candidate_path}")

    # Step 2：只有明確要求時，才寫出正式版
    final_payload = None     # 先把 final_payload 初始化成 None
    # 只有當 write_final=True 時才進入這個區塊,而且進來後立刻檢查 selected_rule_name 是不是 None
    if write_final:
        if selected_rule_name is None:
            raise ValueError("write_final=True 時，selected_rule_name 不能是 None")
            
        # 用布林遮罩在 summary_df 裡找出 name 欄位等於 selected_rule_name 的那一列
        matched = summary_df.loc[summary_df["name"] == selected_rule_name]
        if matched.empty:
            raise ValueError(f"找不到 selected_rule_name: {selected_rule_name}")
            
        if len(matched) > 1:
            raise ValueError(
                f"selected_rule_name={selected_rule_name} 在 summary_df 裡有 {len(matched)} 筆重複,"
                f"請確認 summary_df 的 name 欄位沒有重複值"
            )
        
        # 取 matched 的第一列
        selected_row = _json_safe(matched.iloc[0].to_dict())
        # 整理成固定欄位結構
        baseline_rule_settings = build_baseline_rule_settings(selected_row)

        # 組出正式版的 final_payload,結構跟 Step 1 的候選 payload幾乎一樣（只差沒有 candidate_rank),寫成固定檔名 postprocess_config.json
        # 這正是 model_infer_transition_final.py 的 load_postprocess_config() 預期讀取
        final_payload = {
            "model_id": model_id,
            "selected_at": datetime.now().isoformat(),
            "source": source,
            "baseline_rule_name": selected_rule_name,
            "baseline_rule_settings": _json_safe(baseline_rule_settings),
        }

        final_path = model_folder / "postprocess_config.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)

        print(f"✅ 已輸出正式 postprocess config: {final_path}")

    return (candidate_payloads,    # 前 K 名候選規則各自的 payload,list
            final_payload)         # 正式定案規則的 payload,可能是 None


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
        "baseline_rule_name": baseline_rule_name,
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

    # 取出正式規則,決定 rule_feature_names 要收哪些
    rule_cfg = postprocess.get("baseline_rule_settings",{},)
    if not rule_cfg:
        raise ValueError("postprocess_config.json 缺少 baseline_rule_settings")

    # 正式定案的規則真的有啟用 use_veto**時,才去收集 disturbance_ratio_col、veto_column、以及每條 veto_rules 用到的欄位」
    # 也就是說,正式版的 rule_feature_names 精準對應到「這條被選定的規則實際會用到的欄位」
    rule_feature_names = []
    # 過濾掉空值、去重,避免同一個欄位被列兩次。
    def add_rule_feature(name):
        if name and name not in rule_feature_names:
            rule_feature_names.append(name)

    # 只有正式規則啟用 veto 時，才把 veto 所需欄位列為 runtime 必要欄位。
    if bool(rule_cfg.get("use_veto", False)):
        add_rule_feature(rule_cfg.get("disturbance_ratio_col"))
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
# 主流程：模型推論 + 單階段/兩階段 alarm 規則評估
# ============================================================
def run_disturbance_feature_inference(
    feature_csv,       # 已經做完特徵工程的 disturbance 資料
    metadata_json,
    output_dir=None,   # 如果沒給，就自動建一個預設輸出資料夾
    run_name="disturbance_holdout",
    baseline_rule_name="consecutive_s3_t0.85_k3_d0.40",      # 使用規則
    write_postprocess_final=False,     # 設定是否輸出最佳規則
):
    '''
    1.讀入 disturbance_features_holdout.csv
    2.載入指定的 XGBoost 模型
    3.從模型 package 拿出：
        - 模型本體 model
        - scaler
        - 訓練時用的 feature_names
        - window_threshold
        - file_threshold
    4.對每個 window 算 pred_proba
    5.用 window_threshold 先做最原始的 pred_label
    6.再把同一份 pred_proba 套進多組 alarm 規則
    7.比較各規則的 false alarm rate
    8.選一條 baseline rule，另外輸出 record/type/id/phase 幾張分析表
    '''
    # ---  Step 1-3：載入模型  ---
    # 路徑初始化
    feature_csv = Path(feature_csv)
    metadata_json = Path(metadata_json)
    models_dir, model_id, model_folder = resolve_model_location_from_metadata(metadata_json)
    # 建立模型管理器
    manager = ModelManager(models_dir=str(models_dir))
    # 呼叫 get_latest_xgb_model() 去抓最新模型包
    model_id, pkg = get_xgb_model(manager, model_id=model_id)
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
        "offband_E_mean",
        "offband_E_max",
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

    # 輸出 raw window-level false alarm rate
    print("\nraw window-level false alarm rate =", df_pred["pred_label"].mean())

    # 如果有 eval_class，分群看誤報率
    if "eval_class" in df_pred.columns:
        print("\n依 eval_class 的 raw 誤報率")
        print(
            df_pred.groupby("eval_class")["pred_label"]
            .agg(["count", "mean"])      # 同時算「這個分類有幾筆」跟「pred_label 的平均值（也就是報警比例)」
            .rename(columns={"mean": "false_alarm_rate"})
        )

    # 如果有 disturbance_type，看各種干擾類型
    if "disturbance_type" in df_pred.columns:
        sub_df = df_pred[df_pred["disturbance_type"].fillna("") != ""].copy()      # 先把 disturbance_type 是空值或空字串的列濾掉,只看真的標記了干擾類型的那些列
        if not sub_df.empty:
            print("\n各干擾類型的 raw 誤報率")
            print(
                sub_df.groupby("disturbance_type")["pred_label"]
                .agg(["count", "mean"])
                .rename(columns={"mean": "false_alarm_rate"})
            )

    # ---  Step 6-7：套用多組規則比較  ---
    summary_df, rule_outputs = compare_disturbance_alarm_methods(
        df_pred=df_pred,
        window_threshold=window_threshold,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("後處理規則比較結果")
    print("=" * 80)
    print(summary_df)
    
    # 選擇要的規則
    print("\n候選規則排名（先人工看）:")
    print(
        summary_df[
            [
                "name",
                "min_leak_duration_sec",
                "disturbance_negative_false_alarm_rate",
                "clean_normal_false_alarm_rate",
                "overall_false_alarm_rate",
                "veto_rate",
                "alarm_drop",
                "final_alarm_rate",
            ]
        ]
    )
    
    # ---  Step 8：挑定案規則,輸出分析表跟正式設定檔  ---
    # 如果你沒指定 baseline_rule_name（傳 None),就用這次排序算出來的第一名
    BASELINE_RULE_NAME = summary_df.iloc[0]["name"] if baseline_rule_name is None else baseline_rule_name

    if BASELINE_RULE_NAME not in rule_outputs:
        raise ValueError(f"baseline_rule_name 不存在: {BASELINE_RULE_NAME}")
    
    print("目前用來輸出分析表的規則:", BASELINE_RULE_NAME)
    baseline_df = rule_outputs[BASELINE_RULE_NAME]
    
    baseline_rule_row = _json_safe(
        summary_df.loc[summary_df["name"] == BASELINE_RULE_NAME].iloc[0].to_dict()
    )
        
    # 輸出分析表
    export_disturbance_summary_tables(
        df_rule=baseline_df,                # 選定的那條規則,套用在每個 window 上的完整判斷結果表
        output_dir=output_dir,
        rule_name=BASELINE_RULE_NAME,       # 規則名稱
    )
    
    # 寫出候選/正式 postprocess config
    candidate_postprocesses, final_postprocess = save_postprocess_configs(
        model_folder=model_folder,
        model_id=model_id,
        summary_df=summary_df,      # 全部規則比較結果
        selected_rule_name=BASELINE_RULE_NAME,
        source="disturbance_holdout_rule_selection",
        write_final=write_postprocess_final,
        top_k_candidates=5,         # 函式內部會自己取前 5 名
    )
    
    # 寫出候選 runtime schema
    runtime_feature_schema = save_runtime_feature_schema(
        model_folder=model_folder,
        model_id=model_id,
        feature_names=feature_names,
        df_columns=df_pred.columns.tolist(),
        baseline_rule_name=BASELINE_RULE_NAME,
        baseline_rule_settings=baseline_rule_row,
    )
    
    # 四張聚合摘要表
    record_summary_df = build_record_level_summary(baseline_df)
    phase_summary_df = build_phase_summary(baseline_df)
    type_summary_df = build_disturbance_type_summary(baseline_df)
    
    id_summary_df = build_disturbance_id_summary(baseline_df)
    
    record_summary_df.to_csv(
        output_dir / f"{BASELINE_RULE_NAME}_record_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    
    phase_summary_df.to_csv(
        output_dir / f"{BASELINE_RULE_NAME}_phase_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    
    type_summary_df.to_csv(
        output_dir / f"{BASELINE_RULE_NAME}_disturbance_type_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    
    id_summary_df.to_csv(
        output_dir / f"{BASELINE_RULE_NAME}_disturbance_id_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    
    # 組成 run_info,分幾類欄位
    run_info = {
        # 這次跑的是哪個模型、哪份資料
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "model_folder": str(model_folder),
        "run_name": run_name,
        "feature_csv": str(feature_csv),
        
        # 這次實際採用的規則
        "baseline_rule_name": BASELINE_RULE_NAME,
        "baseline_rule_settings": baseline_rule_row,
        
        # 記錄 runtime schema 相關的資訊
        "recommended_rule_name": candidate_postprocesses[0]["baseline_rule_name"],          # top1
        "recommended_rule_settings": candidate_postprocesses[0]["baseline_rule_settings"],  # top1
        "top_candidate_rule_names": [x["baseline_rule_name"] for x in candidate_postprocesses],    # 記錄前 5 名 postprocesses 清單
        
        "runtime_feature_schema_candidate_path": str(model_folder / "runtime_feature_schema_candidate.json"),
        "runtime_rule_feature_names": runtime_feature_schema["rule_feature_names"],

        "write_postprocess_final": write_postprocess_final,
        
        # 這次資料規模跟模型門檻的統計快照
        "n_rows": int(len(df_pred)),
        "n_features_used": int(len(feature_names)),
        "feature_names": list(feature_names),
        "window_threshold": float(window_threshold),
        "file_threshold": float(file_threshold),
    }

    # 把整份 run_info 寫成 JSON
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    return {
        "df_pred": df_pred,
        "summary_df": summary_df,
        "rule_outputs": rule_outputs,
        "baseline_df": baseline_df,
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

    # 已經做完干擾特徵工程的 feature CSV 路徑(規則版-有offband)
    FEATURE_CSV = Path(
        "/Users/paul/Desktop/Final/disturbance_feature_engineering/"
        "disturbance_rule_features_holdout.csv"
    )
    
    # 比較組：指定哪一條規則當作最後分析與輸出的 baseline rule 名稱，「先看結果、先挑前段班」
    run_disturbance_feature_inference(
        feature_csv=FEATURE_CSV,
        metadata_json=MODEL_METADATA_JSON,
        output_dir=None,
        run_name="disturbance_holdout_short",
        baseline_rule_name=None,
    )
    '''
    # 最終定案版 postprocess_config.json：如果你這次已經確定這條 rule 就是正式部署要用的，再改成：consecutive_s3_t0.80_k3(測試資料暫時)
    # 是要儲存最佳規則：/Users/paul/Desktop/Final/test_result/saved_models_transition_disturbance_final/xgb/xgb_20260824_094834/postprocess_config.json
    run_disturbance_feature_inference(
        feature_csv=FEATURE_CSV,
        metadata_json=MODEL_METADATA_JSON,
        output_dir=None,
        run_name="disturbance_holdout",
        baseline_rule_name="consecutive_s3_t0.80_k3_d0.40",
        write_postprocess_final=True,
    )
    '''
    
