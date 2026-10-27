#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正式 Transition 特徵工程流程
============================

本程式讀取 Transition TDMS 與時間標註，
使用 Step2 定案的正式頻帶、特徵萃取設定、
固定正常 Baseline 與 Feature Execution Plan，
建立連續且與 Step2／Runtime 計算方式一致的
window-level 特徵。

本程式不執行：

- 頻帶搜尋
- 特徵選擇
- Baseline 建立
- 模型訓練
- 模型機率推論
- 告警規則定案


一、正式輸入
------------

1. Transition TDMS
2. transition_meta.csv
3. band_config_step2.json
4. feature_extraction_config.json
5. site_baseline.json
6. feature_config_micro.json


二、執行前契約驗證
------------------

1. Band config 必須為 schema_status=final。
2. Leak-focused bands 不可為空。
3. 正式流程要求 Offband 不可為空。
4. Feature config 與 Band config SHA-256 必須一致。
5. Feature config 與 Extraction config SHA-256 必須一致。
6. Feature config 與 Site Baseline 的 config fingerprint 必須一致。
7. Baseline 建立時的 Band/Extraction Hash 必須一致。
8. Feature Execution Plan 必須完整。
9. Transition TDMS 不得與建立 Baseline 的正常 TDMS 重複。
10. TDMS 與 metadata 必須一對一完整配對。
11. metadata 的時間欄位必須完整且順序合理。

任一契約不一致時立即停止，不使用 fallback。


三、Feature Execution Plan
--------------------------

baseline_features：- 固定 Baseline rel/z 的 Raw 輸入欄位。

cusum_input_features：- CUSUM 使用的 z_* 輸入欄位。

temporal_base_features：- Delta、Local、Rolling 使用的基礎欄位。

final_keep_features：- 最後交給模型的正式特徵，欄位順序不可改變。


四、正式計算流程
----------------

1. 載入並驗證所有設定檔。
2. 掃描 Transition TDMS。
3. 驗證 TDMS 與 metadata 配對。
4. 同步 temporal_band_names。
5. 使用正式頻帶及 Extraction config 切窗。
6. 建立 Leak-focused、Resonance 與 Offband Raw features。
7. 依整個 window 的時間範圍貼上標籤（t_ref 只用於排序與時間特徵）：
    - window 結束 <= normal_end_sec：label=0
    - window 開始 >= leak_start_sec：label=1
    - 其他（跨到灰區或跨邊界）：灰區
8. 建立 Offband、Leak-focus、Resonance summary。
9. 保留完整正常、灰區及洩漏序列。
10. 依 Execution Plan 及規則摘要欄位套用固定 Baseline，建立 rel/z。
11. 依 Execution Plan 建立 CUSUM。
12. 依 Execution Plan 建立 Delta、Local、Rolling。
13. 驗證所有最終模型特徵均存在且有有效數值。
14. 完成連續時間特徵後，才另外建立明確標籤版本。


五、連續時間原則
----------------

CUSUM 與 Temporal 必須依每個 record_id 分開，
並按照 t_ref 由早到晚計算。

灰區雖然不提供明確分類標籤，仍必須參與：

- CUSUM 狀態更新
- Window-to-window Delta
- Local Baseline
- Rolling 統計
- Onset 前後變化

所有時間特徵只能使用當下與過去資料，
不得使用未來 Window。


六、固定 Baseline 原則
----------------------

所有 Transition records 使用同一份預先建立的正常 Baseline：

    rel_x = x - baseline_mean

    z_x = (
        x - baseline_mean
    ) / baseline_std

normal_end_sec 只用於貼標與 Onset 評估，
不參與 Baseline mean/std 計算。


七、正式輸出
------------

1. *_full.csv

- 保留正常、灰區、洩漏的完整連續 Windows。
- 包含完整 Raw、Summary、rel/z、CUSUM 與 Temporal。
- 作為特徵稽核與後續規則欄位來源。

2. *_continuous_selected.csv

- 保留完整連續 Windows。
- 只包含 metadata 與模型 selected features。
- 可用於模型機率及基礎 Onset 分析。
- 若規則使用 Veto，需另外補入規則欄位。

3. 正式 output_csv

- 排除灰區。
- 只保留 label=0/1。
- 包含 metadata 與模型 selected features。
- 用於模型資料合併及分類效能評估。


八、專案流程位置
----------------

Step1：搜尋 Resonance bands。

Step2a：搜尋 Leak-focused candidate bands。

人工定案：加入 Offband 並產生正式 Band config。

Final Baseline：依正式頻帶重建 Site Baseline。

Step2b：選出正式模型特徵與 Execution Plan。

Step3（本程式）：建立 Transition 連續時間特徵。

後續：

Disturbance 特徵工程
→ 模型訓練
→ Disturbance 規則搜尋
→ Transition 規則與 Onset 驗證
→ 產生正式 Runtime Schema
"""

# ============================================================
# 載入套件
# ============================================================
from pathlib import Path
import sys
import gc
import pandas as pd
import numpy as np
import json
import warnings

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

# 把專案路徑加進 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_feature_utils import (
    load_or_compute_features_by_file,
    merge_band_groups,
    build_offband_summary_features,
    add_cusum_features,
    add_window_dynamic_features,
    add_local_rolling_features,
    audit_band_feature_coverage,
    audit_offband_summary_coverage,
    #audit_derived_feature_coverage,
    audit_selected_feature_alignment,
    audit_rule_feature_alignment,
    BandConfig,
    
    calculate_file_sha256,
    load_feature_extraction_config,
    load_fixed_normal_baseline,
    apply_fixed_normal_baseline,
    validate_feature_stage_requirements,
    configure_temporal_band_names,
    validate_baseline_source_isolation,
    load_band_config,
    SIGNAL_RMS_MIN,
)


# ============================================================
# Baseline 資料隔離設定
# ============================================================
# Transition 不應與建立固定 Baseline 的正常 TDMS 重複。
# True：發現相同 TDMS 路徑、record_id 或檔案內容時直接停止。
# False：只顯示警告，僅供舊資料流程測試。
STRICT_BASELINE_ISOLATION = True


# ============================================================
# 內容重複檢查（比檔案內容，不比檔名；改過檔名的複本也抓得到）
# ============================================================
def check_content_duplicates(file_specs, baseline_paths, strict=True):
    """
    1. Transition 檔案彼此內容相同 → 同一段訊號被算兩次
    2. Transition 跟建立 baseline 的正常檔內容相同 → baseline 資料洩漏
    """
    seen = {}             # 記錄每個檔案指紋第一次出現時對應的 record_id
    dup_pairs = []        # 記錄 Transition 內部內容重複的成對檔案
    baseline_hits = []    # 記錄與 baseline 來源檔內容相同的 Transition 檔

    # 逐一讀取 Transition 檔案，用 SHA-256 算內容指紋；同一份檔案即使改名、移動資料夾，指紋仍相同
    for spec in file_specs:
        h = calculate_file_sha256(spec["filepath"])
        # 如果這個指紋之前看過，代表現在的檔案與先前某個 Transition 檔內容完全一致，記下兩個 record_id；否則把它登記成首次出現
        if h in seen:
            dup_pairs.append((seen[h], spec["record_id"]))
        else:
            seen[h] = spec["record_id"]

    missing_baseline = 0
    # 逐一檢查 baseline 建立時實際使用過的來源檔。若檔案已被移動或刪除，無法做內容比對，就累計並跳過
    for bp in baseline_paths:
        if not Path(bp).exists():
            missing_baseline += 1
            continue
        # 若 baseline 檔的 SHA-256 出現在 Transition 的 seen 裡，代表兩者其實是同一份訊號內容，記錄這個衝突
        h = calculate_file_sha256(bp)
        if h in seen:
            baseline_hits.append((Path(bp).stem, seen[h]))

    if missing_baseline:
        print(f"⚠️ baseline 來源檔有 {missing_baseline} 支找不到，這些無法比對內容")
    # 資料完整性檢查通過:沒有任何內部重複，也沒有撞到 baseline
    if not dup_pairs and not baseline_hits:
        print(f"✅ 內容重複檢查通過：{len(seen)} 支 transition 彼此不重複，也沒有跟 baseline 檔內容相同")
        return

    msg = (f"Transition 內容重複 {len(dup_pairs)} 組：{dup_pairs[:5]}；"
           f"與 baseline 內容相同 {len(baseline_hits)} 組：{baseline_hits[:5]}")
    
    # strict=True：正式流程建議使用。發現重複或洩漏直接停止，避免產出不可信的結果。
    # strict=False：只顯示警告，適合先盤點舊資料，但不適合正式產生特徵或訓練。
    if strict:
        raise ValueError(msg)
    print("⚠️ " + msg)


# ============================================================
# 讀 feature 設定檔
# ============================================================
def load_feature_config(feature_config_json):
    # 把 feature_config.json 讀進來，回傳一個字典
    return json.loads(Path(feature_config_json).read_text(encoding="utf-8"))


#                                               ****************************************************************
#                                                     transition 專屬工具：貼標 / baseline / temporal delta
#                                               ****************************************************************
# ============================================================
# 根據 transition_meta 貼 window label
# ============================================================
def attach_window_labels_from_transition_meta(
    df_windows,               # 特徵表，每列一個 window
    df_meta,                  # transition 的時間標註資訊
    window_sec=0.2,           # window 長度
    drop_transition=False,    # 是否丟掉中間灰區
):
    '''
    把每個 window 的時間位置，對照 transition_meta 裡的事件時間，幫每一列貼上分類標籤：
        0：整個窗都在 normal_end_sec 之前
        1：整個窗都在 leak_start_sec 之後
        其他（跨到灰區或跨邊界）：灰區，label 留空
    t_ref 一律用窗中心，只用來排序與時間特徵，不用來貼標。
    '''
    
    # 複製一份，避免直接改原始 DataFrame
    df = df_windows.copy()
    meta_cols = [
        "record_id", "interval_name", "distance_cm", "leak_quantity", "pressure_bar",
        "output_tdms", "normal_end_sec", "leak_start_sec", "onset_gt_sec",
    ]
    meta = df_meta[meta_cols].copy()
    
    # 特徵表裡不應該已經有 meta 欄位，否則 merge 會變成 _x / _y
    clash = [c for c in meta_cols if c != "record_id" and c in df.columns]
    if clash:
        raise ValueError(f"特徵表已經有這些欄位，merge 會撞名：{clash}")

    # many_to_one：每個 record_id 在 meta 只能出現一次，重複就直接報錯
    # 用 record_id 把 window 特徵表和 transition_meta.csv 接起來
    df = df.merge(meta, on="record_id", how="left", validate="many_to_one")
    
    win_start = df["t_start"]
    win_end = df["t_start"] + window_sec
    df["t_ref"] = df["t_start"] + window_sec / 2.0    # 窗中心：排序與時間特徵用

    # 先把所有 label 設成空值
    df["label"] = np.nan
    
    # 正常區:整個窗在正常段
    df.loc[win_end <= df["normal_end_sec"], "label"] = 0
    # 洩漏區:整個窗在洩漏段
    df.loc[win_start >= df["leak_start_sec"], "label"] = 1   

    # 如果你想把中間灰區丟掉，就只保留有 label 的列
    if drop_transition:
        # 舊模式：立即排除灰區
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
    else:
        # 新模式：暫時保留灰區，使用可容納缺值的整數型態
        df["label"] = df["label"].astype("Int64")

    return df


# ============================================================
# 主流程
# ============================================================
def build_inference_feature_csv(
    tdms_dir,                          # transition TDMS 檔案路徑
    meta_csv,                          # 對應的 transition 標註表
    band_config_json,                  # Step2 輸出的 band 設定
    feature_extraction_config_path,
    baseline_json,
    feature_config_json,               # Step2 輸出的 feature 設定
    output_csv,                        # 最後特徵表要輸出去哪裡
    cache_dir,                         # 中間 band 特徵的快取資料夾
    split_name="transition_infer",     # 給 cache / 識別用的名字
    n_jobs=2,                          # 平行處理數量
    require_final_baseline=False,
):
    """
    使用已定案的頻帶、特徵設定與固定正常 baseline，
    將 transition TDMS 轉換為可供最終分類模型使用的特徵表。

    流程：
    1. 驗證 transition 標註時間與檔案資料完整性
    2. 載入正式 band / feature / baseline 設定
    3. 以設定檔指紋確認各元件相容，避免混用不同版本
    4. 依 Feature Execution Plan 抽取、標準化與衍生特徵
    5. 僅輸出 Step2 已定案的 selected features
    """
    
    # ============================================================
    # -----  1. 把所有路徑轉成 Path  -----
    # ============================================================
    tdms_dir = Path(tdms_dir)
    meta_csv = Path(meta_csv)
    output_csv = Path(output_csv)
    cache_dir = Path(cache_dir)
    band_config_json = Path(band_config_json)
    feature_config_json = Path(feature_config_json)
    feature_extraction_config_path = Path(feature_extraction_config_path)
    baseline_json = Path(baseline_json)


    # ============================================================
    # -----  2. 讀 meta、讀正式設定檔  -----
    # ============================================================
    df_meta = pd.read_csv(meta_csv)                           # 讀 transition 的標註表(有每個 record 的normal_end_sec/leak_start_sec這些時間標註)
    df_meta["record_id"] = df_meta["record_id"].astype(str).str.strip()    # 避免 meta 的 record_id 如果尾巴不小心多一個空白，例如 abc_001 ，就會跟檔名對不上
    
    # meta 基本檢查：欄位齊全、record_id 不重複、時間完整且順序合理
    # normal_end_sec：可明確視為正常狀態的最後時間點
    # leak_start_sec：可明確視為洩漏狀態的開始時間點
    # onset_gt_sec：人工標註的實際洩漏起始參考點；
    # 位於兩者之間的過渡區通常不作為乾淨 Normal / Leak 訓練標籤。
    need_cols = ["record_id", "normal_end_sec", "leak_start_sec", "onset_gt_sec"]
    miss_cols = [c for c in need_cols if c not in df_meta.columns]
    # 必要欄位是否有缺失
    if miss_cols:
        raise ValueError(f"transition_meta.csv 缺少欄位：{miss_cols}")
    # record_id 不可重複
    dup_ids = df_meta.loc[df_meta["record_id"].duplicated(keep=False), "record_id"].unique().tolist()
    if dup_ids:
        raise ValueError(f"transition_meta.csv 的 record_id 重複：{dup_ids[:10]}")
    # 時間欄位不可空
    time_cols = ["normal_end_sec", "leak_start_sec", "onset_gt_sec"]
    for c in time_cols:
        converted = pd.to_numeric(df_meta[c], errors="coerce")
        bad_text = df_meta.loc[converted.isna() & df_meta[c].notna(), "record_id"].tolist()
        if bad_text:
            raise ValueError(f"transition_meta.csv 的 {c} 有非數字內容：{bad_text[:10]}")
        df_meta[c] = converted
    missing_time = {c: int(df_meta[c].isna().sum()) for c in time_cols if df_meta[c].isna().any()}
    if missing_time:
        raise ValueError(f"transition_meta.csv 時間欄位有空值：{missing_time}")
    # normal_end ≤ leak_start
    bad_order = df_meta[(df_meta["normal_end_sec"] < 0) | (df_meta["normal_end_sec"] > df_meta["leak_start_sec"])]
    if not bad_order.empty:
        raise ValueError(f"normal_end_sec 必須 ≤ leak_start_sec：{bad_order['record_id'].tolist()[:10]}")
    # onset 在兩者之間（只警告）
    odd_onset = df_meta[(df_meta["onset_gt_sec"] < df_meta["normal_end_sec"]) | (df_meta["onset_gt_sec"] > df_meta["leak_start_sec"])]
    if not odd_onset.empty:
        print(f"⚠️ onset_gt_sec 不在 normal_end ~ leak_start 之間：{odd_onset['record_id'].tolist()[:10]}")
    
    # 讀 4 份正式設定
    # 用哪些頻帶，必須是 final，而且要有 offband
    band_config = load_band_config(band_config_json, expected_status="final", require_offband=True)   # 讀 Step2 的正式 band 設定
    # 窗長、hop、取樣率等切窗參數
    extraction_cfg = load_feature_extraction_config(feature_extraction_config_path)
    # Step2 選出的特徵和執行計畫
    feature_cfg = load_feature_config(feature_config_json)    # 讀 Step2 的正式 feature 設定:這是 feature_config.json 的流程設定，不是特徵資料表本身
    # 正常資料的 mean/std
    fixed_baseline = load_fixed_normal_baseline(
        baseline_json,
        band_config_path=band_config_json,
        feature_extraction_config_path=(
            feature_extraction_config_path
        ),
        require_final=require_final_baseline,
    )
    
    # 比對設定指紋：確認 band、特徵抽取設定、固定 baseline
    # 與 Step2 的 feature_config 相容，避免混用不同版本的設定檔。
    current_band_hash = calculate_file_sha256(band_config_json)
    current_extraction_hash = calculate_file_sha256(feature_extraction_config_path)
    
    if (feature_cfg.get("band_config_sha256") != current_band_hash):
        raise ValueError("feature_config_micro.json 與目前 band_config_step2.json 不一致")
    
    if (feature_cfg.get("feature_extraction_config_sha256") != current_extraction_hash):
        raise ValueError("feature_config_micro.json 與目前 feature_extraction_config 不一致")
    
    fp_cfg = feature_cfg.get("baseline_config_fingerprint")
    fp_base = fixed_baseline.get("config_fingerprint")
    if not fp_cfg or not fp_base or fp_cfg != fp_base:
        raise ValueError(f"feature_config_micro.json 與目前固定 baseline 不一致：{fp_cfg} vs {fp_base}")
    
    # 把要算 temporal 特徵的頻帶名單交給特徵抽取器：把 band config 的 temporal band 名稱,同步給 shared_feature_utils 的 raw feature extractor。
    configure_temporal_band_names(band_config.temporal_band_names)
    
    
    # ============================================================
    # -----  3. 把正式 selected features 拿出來  -----
    # ============================================================
    # 先拿到模型最後真正要保留的特徵名稱，正式流程會先在正式 bands 內建立完整特徵鏈，最後再只保留這些欄位
    # 從 feature config 裡取出 Step2 最終選定要給模型用的特徵名單,存成final_keep_features
    selected_feature_names = feature_cfg.get("selected_feature_names", [])
    
    if not selected_feature_names:
        raise ValueError(
            "feature_config_micro.json "
            "缺少 selected_feature_names"
        )
    
    final_keep_features = list(selected_feature_names)
    
    # Feature Execution Plan 定義特徵的執行依賴關係：
    # baseline_features → 計算 relative / z 特徵；
    # cusum_input_features → 由 z 特徵計算累積偏移；
    # temporal_base_features → 計算 delta / rolling 類時間特徵；
    # final_keep_features → 最終保留給分類模型的欄位與順序。
    #
    # 這可確保 inference 階段完全沿用 Step2 定案的特徵邏輯，不會臨時多算、少算或改變模型輸入欄位順序。
    execution_plan = feature_cfg.get("feature_execution_plan")
    
    if not isinstance(execution_plan, dict):
        raise ValueError("feature_config_micro.json 缺少 feature_execution_plan")
    
    required_plan_keys = [
        "baseline_features", "cusum_input_features",
        "temporal_base_features", "final_keep_features",
    ]
    
    missing_plan_keys = [key for key in required_plan_keys if key not in execution_plan]
    
    if missing_plan_keys:
        raise ValueError(f"feature_execution_plan 缺少欄位：{missing_plan_keys}")
    
    for key in required_plan_keys:
        if not isinstance(execution_plan[key], list):
            raise ValueError(f"feature_execution_plan.{key} 必須是 list")
    # 要套 baseline 算 rel/z 的原始欄位
    baseline_feature_names = list(dict.fromkeys(execution_plan["baseline_features"]))
    # 要算 CUSUM 的 z 欄位
    cusum_input_feature_names = list(dict.fromkeys(execution_plan["cusum_input_features"])) 
    # 要算 delta/rolling 的欄位
    temporal_base_feature_names = list(dict.fromkeys(execution_plan["temporal_base_features"])) 
    # 最後給模型的欄位（順序不能變）
    planned_final_features = list(dict.fromkeys(execution_plan["final_keep_features"]))
    
    if planned_final_features != final_keep_features:
        raise ValueError("feature_execution_plan.final_keep_features 與 selected_feature_names 不一致")
    
    print("固定 baseline 特徵數量:",len(baseline_feature_names),)    
    print("CUSUM 輸入特徵數量:",len(cusum_input_feature_names),)
    print("temporal base 特徵數量:",len(temporal_base_feature_names),)
    print("最終 selected features 數量:", len(final_keep_features),)

    
    # ============================================================
    # -----  4. 取正式參數：切窗參數來自 feature_extraction_config，CUSUM 參數來自 feature_config  -----
    # ============================================================
    window_sec = float(extraction_cfg["window_sec"])                           # 窗長 0.2 秒
    hop_sec = float(extraction_cfg["hop_sec"])                                 # 每次往前滑 0.05 秒
    fs = float(extraction_cfg["fs"])                                           # 取樣率 1 MHz
    nperseg = int(extraction_cfg["nperseg"])                                   # 頻譜計算的段長
    max_points = int(extraction_cfg["max_points"])                             # 一支檔最多讀幾點，避免記憶體爆掉
    base_feature_version = str(extraction_cfg["base_feature_version"])         # 特徵公式版本，快取用它判斷能不能重用
    include_offband_summary = bool(extraction_cfg["include_offband_summary"])  # 要不要建 offband／leak_focus／resonance 摘要
    
    if "cusum_k" not in feature_cfg:
        raise ValueError("feature_config_micro.json 缺少 cusum_k")
    if "cusum_decay" not in feature_cfg:
        raise ValueError("feature_config_micro.json 缺少 cusum_decay")
    
    cusum_k = float(feature_cfg["cusum_k"])                                    # CUSUM 的容忍值 0.5
    cusum_decay = float(feature_cfg["cusum_decay"])                            # 衰減 0.97
    
    # 檢查 CUSUM 參數範圍是否合理
    if not (cusum_k > 0):
        raise ValueError(f"cusum_k 必須 > 0，目前 {cusum_k}")
    if not (0 < cusum_decay <= 1):
        raise ValueError(f"cusum_decay 必須在 (0, 1]，目前 {cusum_decay}")
    print(f"CUSUM 參數：k = {cusum_k}，decay = {cusum_decay}")


    # ============================================================
    # -----  5. 從 meta 裡決定哪些 TDMS 要處理  -----
    # ============================================================
    valid_record_ids = set(df_meta["record_id"])         # 先從 meta_csv 抓出合法的 record_id
    all_tdms_files = sorted(tdms_dir.rglob("*.tdms"))    # 掃 tdms_dir 底下所有 .tdms
    
    # 遞迴掃描 tdms_dir 底下的所有 .tdms，包含子資料夾
    tdms_by_id = {}
    for p in all_tdms_files:
        tdms_by_id.setdefault(p.stem, []).append(p)
    
    # 防止不同資料夾有同名檔
    same_name = {k: [str(x) for x in v] for k, v in tdms_by_id.items() if len(v) > 1}
    # 找出是否有不同路徑，但檔名相同的 TDMS
    if same_name:
        raise ValueError(f"不同資料夾有同檔名的 TDMS：{list(same_name.items())[:5]}")
    
    # 雙向檢查 meta 與檔案是否完整配對
    meta_without_tdms = sorted(valid_record_ids - set(tdms_by_id))
    tdms_without_meta = sorted(set(tdms_by_id) - valid_record_ids)
    if meta_without_tdms:
        raise ValueError(f"meta 有、但找不到 TDMS：{meta_without_tdms[:10]}（共 {len(meta_without_tdms)} 筆）")
    if tdms_without_meta:
        print(f"⚠️ 有 TDMS 不在 meta 裡，會略過：{tdms_without_meta[:10]}（共 {len(tdms_without_meta)} 筆）")
    
    # 建立真正會處理的檔案清單
    file_specs = [{"filepath": str(tdms_by_id[rid][0]), "record_id": rid} for rid in sorted(valid_record_ids)]

    # 如果完全沒對到，就直接報錯
    if not file_specs:
        raise ValueError("找不到符合 meta 的 TDMS 檔")
        
    
    # ============================================================
    # Transition 與固定 Baseline 資料隔離檢查
    # ============================================================
    # file_specs 包含本次實際會進入 Transition 特徵工程的 TDMS 路徑與 record_id。
    # 確認這些資料沒有被拿來建立 site_baseline.json，避免 Transition 訓練／驗證資料參與正常參考統計。
    validate_baseline_source_isolation(
        baseline_payload=fixed_baseline,
        analysis_records=file_specs,
        context="Transition feature engineering",
        strict=STRICT_BASELINE_ISOLATION,
    )
    
    # 內容層級的重複與洩漏檢查:比「檔案內容」的 SHA-256，不只比路徑或檔名
    check_content_duplicates(
        file_specs,
        fixed_baseline.get("source_tdms_paths", []),
        strict=STRICT_BASELINE_ISOLATION,
    )

    
    # ============================================================
    # -----  6. 真正 band 特徵抽取  -----
    # ============================================================
    # 對每個 TDMS 切窗 -> 在 core_bands (洩漏頻帶+共振頻帶合併)跟 aux_bands(offband) 上算出各種基礎頻帶特徵, low_bands 留空代表不再用舊版低頻設計
    features = load_or_compute_features_by_file(
        cache_dir=cache_dir,
        file_specs=file_specs,
        core_bands=band_config.final_main_bands,      # 正式主 band：leak_focused + resonance
        aux_bands=band_config.offband_bands,          # 輔助 band：offband
        low_bands=[],                                 # 不再另外保留舊版 low band
        split_name=split_name,
        n_jobs=n_jobs,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        target_levels=None,
        feature_version=base_feature_version,
        max_points=max_points,
    )

    # 把結果轉成 DataFrame, 按record_id跟t_start排序
    df = pd.DataFrame(features).sort_values(["record_id", "t_start"]).reset_index(drop=True)
    
    # 檢查：確認每支檔都有算出窗
    got_ids = set(df["record_id"])
    lost_ids = sorted({s["record_id"] for s in file_specs} - got_ids)
    if lost_ids:        # 只要任何 TDMS 完全沒有輸出特徵窗，就停止
        raise ValueError(f"有 {len(lost_ids)} 支 TDMS 沒有算出任何窗：{lost_ids[:10]}")
        
    # 檢查 訊號品質：門檻和 baseline 程式相同；有壞窗的 record 整支排除，保持時間序列連續
    bad = pd.Series(False, index=df.index)     # 先建立一欄全為 False 的標記；每一列代表一個 window
    # 逐一檢查兩個通道, 只要任一通道有問題，就把該 window 標成壞窗
    for ch in ("ai0", "ai1"):
        rms = pd.to_numeric(df[f"{ch}_rms"], errors="coerce")
        bad |= ~np.isfinite(rms) | (rms < SIGNAL_RMS_MIN)
    # 找出「曾出現壞窗」的所有 record
    if bad.any():
        # 不是只刪壞窗，而是刪掉整支 record
        bad_ids = sorted(df.loc[bad, "record_id"].unique())
        print(f"⚠️ 訊號 QC：{len(bad_ids)} 支 record 有壞窗（共 {int(bad.sum())} 窗），整支排除：{bad_ids[:10]}")
        df = df[~df["record_id"].isin(bad_ids)].reset_index(drop=True)
        if df.empty:
            raise ValueError("訊號 QC 後沒有任何可用的 record")
    # 檢查頻帶特徵是否齊全
    audit_band_feature_coverage(df, band_config)
    
    # 清除記憶體
    del features
    gc.collect()

    
    # ============================================================
    # -----  7. 根據 meta 幫每個 window 貼 transition 標註  -----
    # ============================================================
    # 把每個 window 對回 transition meta, 決定這個 window 是正常還是洩漏, 並補上後面會需要的欄位(label, normal_end_sec, leak_start_sec, onset_gt_sec, distance_cm, pressure_bar)
    df = attach_window_labels_from_transition_meta(
        df_windows=df,                  # 前一步抽出的逐窗頻帶特徵
        df_meta=df_meta,                # 每支 TDMS 的 normal_end_sec、leak_start_sec、onset_gt_sec 等標註
        window_sec=window_sec,          # 每個視窗長度，用來判定「整個 window」落在哪個時間區段
        # 先保留 Normal 與 Leak 中間的灰區，供後續 CUSUM／temporal 特徵使用；
        # 最終模型訓練或效能評估時，才依標籤規則排除灰區。
        drop_transition=False,
    )
    
    # 確認每一窗都成功對到 meta
    required_transition_meta_cols = [
        "normal_end_sec", "leak_start_sec", "onset_gt_sec",
    ]
    
    # 逐欄檢查：該欄有沒有成功加入 DataFrame；該欄是否有空值。若有問題，記錄「缺哪個欄位／有幾個窗沒有值」
    missing_meta_values = {
        column: int(df[column].isna().sum())
        for column in required_transition_meta_cols
        if (
            column not in df.columns
            or df[column].isna().any()
        )
    }
    # 只要任一 window 沒正確對應 metadata，就停止
    if missing_meta_values:
        raise ValueError(f"Transition windows 無法完整對齊 metadata：{missing_meta_values}")
        
    # 可選擇建立 offband / leak-focused / resonance 摘要欄位(預設是)
    if include_offband_summary:
        df = build_offband_summary_features(df, band_config)
        # 檢查:確認預期的 offband 摘要欄位都已建立，避免 band config 有 offband，但後面特徵鏈實際缺欄
        audit_offband_summary_coverage(df)
        
        
    # ============================================================
    # Step 8：使用固定正常 baseline 建立 rel / z 特徵
    #
    # baseline_feature_names 由 Step 2 的 execution plan 提供。
    # dev、holdout、transition、disturbance 與 Runtime
    # 都使用同一份固定 baseline。
    # normal_end_sec 只負責 transition 標註，不參與 baseline 統計。
    # ============================================================
    # 規則需要的摘要欄位也套 baseline（和 disturbance 一致，告警規則才能在兩邊用同一組欄位）
    rule_baseline_feature_names = [
        c for c in [
            "offband_E_mean", "offband_E_max",
            "leak_focus_E_mean", "leak_focus_E_max",
            "resonance_E_mean", "resonance_E_max",
        ]
        if c in df.columns
    ]
    baseline_apply_feature_names = list(dict.fromkeys(baseline_feature_names + rule_baseline_feature_names))
    
    # Baseline 前動態檢查
    validate_feature_stage_requirements(df, baseline_apply_feature_names,
        stage_name=("Transition / fixed baseline 前置 raw"),)
    
    # 對模型與規則需要的欄位，套用固定正常 baseline 的統計量
    df = apply_fixed_normal_baseline(df=df, feature_cols=baseline_apply_feature_names, baseline_payload=fixed_baseline,)
    
    # rel_：與正常平均值相差多少；z_：偏離幾個正常標準差
    expected_baseline_derived_features = [
        derived_feature
        for raw_feature in baseline_apply_feature_names
        for derived_feature in (f"rel_{raw_feature}", f"z_{raw_feature}",)
    ]
    
    validate_feature_stage_requirements(df, expected_baseline_derived_features,
        stage_name=("Transition / fixed baseline rel-z 輸出"),
    )
    
    
    # ============================================================
    # Step 9. 建 CUSUM 特徵
    # # 把 z_* 欄位再建 cusum, CUSUM 的作用是：把持續偏移或累積偏移的趨勢放大出來
    # ============================================================
    # CUSUM 前置檢查：確認要輸入 CUSUM 的特徵都存在
    validate_feature_stage_requirements(df, cusum_input_feature_names,
        stage_name=("Transition / CUSUM 輸入"),)
    
    # 對每個指定欄位建立 CUSUM 特徵
    if cusum_input_feature_names:
        df = add_cusum_features(df=df, input_feature_cols=(cusum_input_feature_names), k=cusum_k, decay=cusum_decay,)
        
    # 替每個 CUSUM 輸入欄位建立預期的正向、負向輸出名稱
    expected_cusum_output_features = [
        output_feature
        for input_feature in cusum_input_feature_names
        for output_feature in (f"cusum_pos_{input_feature}", f"cusum_neg_{input_feature}",)]
    
    # 確認 CUSUM 真的成功建立所有預期欄位，避免某個特徵名稱拼錯、欄位遺失，卻等到模型輸入時才出錯
    validate_feature_stage_requirements(df, expected_cusum_output_features,
        stage_name=( "Transition / CUSUM 輸出"),)

    
    # ============================================================
    # Step 10. 建 temporal 類特徵
    # 對 execution plan 指定的基礎欄位，於每個 record 內依時間建立：
    # 1. 相鄰窗變化：delta / delta_prev_mean
    # 2. 局部滾動統計：local_rel / local_z / rolling_std / rolling_slope / rolling_max_gap
    # 所有 temporal 特徵均只在同一 record 內計算，不跨錄音檔串接，避免不同實驗條件彼此影響。
    # ============================================================
    # Temporal 前置檢查:確認 execution plan 指定要做 temporal 衍生的基礎欄位都已存在
    validate_feature_stage_requirements(
        df,
        temporal_base_feature_names,
        stage_name=("Transition / temporal 輸入"),
    )
    
    # 建 temporal 類特徵
    if temporal_base_feature_names:
        # 對每一支 TDMS 各自依時間順序建立「相鄰 window 的變化量」
        df = add_window_dynamic_features(
            df=df,
            feature_cols=temporal_base_feature_names,
            group_col="record_id",    # 代表每支錄音各自計算，不會讓 record_A 最後一窗與 record_B 第一窗被誤當成連續訊號
            time_col="t_ref",         # 以視窗的參考時間排序與計算
        )
        
        # 以每支 record 的局部時間窗，建立滾動統計特徵
        df = add_local_rolling_features(
            df=df,
            feature_cols=temporal_base_feature_names,
            group_col="record_id",
            time_col="t_ref",
        )

    # 最後 selected features 檢查:確認 Step2 已定案、模型真正需要的所有欄位都已成功生成
    validate_feature_stage_requirements(
        df,
        final_keep_features,
        stage_name=("Transition / final selected features"),
    )
        
    
    # ============================================================
    # Step 11.1｜保留完整連續序列，區分明確標籤與 Transition 灰區
    # - df_continuous：Normal／灰區／Leak 全保留，供時間序列與 onset 分析
    # - df_labeled：僅保留明確 0/1 標籤，供分類訓練與效能評估
    # ============================================================
    df_continuous = df.copy()
    gray_mask = df_continuous["label"].isna()
    
    print("完整連續 windows：",len(df_continuous),)
    print("transition 灰區 windows：", int(gray_mask.sum()),)
    
    # 實驗資料的分類訓練／效能評估版本：
    # 僅保留明確 Normal／Leak 標籤；Transition 灰區不作為監督式分類真值。
    df_labeled = df_continuous.loc[~gray_mask].copy()
    df_labeled["label"] = (df_labeled["label"].astype(int))
    
    print("明確標籤 windows：",len(df_labeled),)
    
    # ============================================================
    # Step 11.2｜輸出每支 record 的 Normal／灰區／Leak 視窗統計
    # 用於確認每筆 transition 錄音的標註切分是否完整。
    # ============================================================
    lab = df_continuous["label"].astype(float)
    seg = np.where(lab.isna(), "gray", np.where(lab == 0, "normal", "leak"))
    # 統計每支錄音各有幾個 normal／gray／leak window
    record_summary = (
        pd.crosstab(df_continuous["record_id"].values, seg)
        .reindex(columns=["normal", "gray", "leak"], fill_value=0)
        .rename_axis("record_id").reset_index()
    )
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 輸出
    record_summary_csv = output_csv.with_name(output_csv.stem + "_record_summary.csv")
    record_summary.to_csv(record_summary_csv, index=False, encoding="utf-8-sig")
    
    # 找出完全沒有正常段或完全沒有洩漏段的 record
    bad = record_summary[(record_summary["normal"] == 0) | (record_summary["leak"] == 0)]
    if not bad.empty:
        print(f"⚠️ 有 {len(bad)} 個 record 缺少正常段或洩漏段：{bad['record_id'].tolist()[:10]}")
    print(f"✅ 每個 record 窗數統計：{record_summary_csv}")
    
    # ============================================================
    # Step 11.3｜定義模型輸出欄位並驗證 Step2 特徵對齊
    # 輸出欄位 = 基本 metadata + Step2 最終選定的特徵。
    # 特徵名稱與順序必須完全一致，避免模型輸入錯位。
    # ============================================================
    base_output_cols = [
        "filename", "record_id", "t_start", "t_ref", "label",
        "interval_name", "distance_cm", "leak_quantity", "pressure_bar",
        "output_tdms", "normal_end_sec", "leak_start_sec", "onset_gt_sec",
    ]

    # 檢查
    audit_selected_feature_alignment(df, feature_cfg)
    
    
    # ============================================================
    # 輸出 1｜完整特徵稽核版：*_full.csv
    # 包含：所有 metadata、所有 raw／rel／z／CUSUM／temporal 特徵，
    #       並保留 Normal、Transition 灰區與 Leak。
    # 用途：除錯、特徵核對、onset 前後視覺化；不作為固定模型輸入。
    # ============================================================
    full_output_csv = output_csv.with_name(output_csv.stem + "_full.csv")      # 如果你想確認 offband 摘要欄位到底有沒有被建立,直接打開這份_full.csv
    # 這份 _full.csv 會包含灰區，可以檢查真正連續的：- raw , rel/z , CUSUM , temporal , onset 前後變化
    df_continuous.to_csv(full_output_csv,index=False, encoding="utf-8-sig",)
    print(f"✅ full continuous feature csv 已輸出: {full_output_csv}")

    # 檢查模型輸出前必要欄位：必要欄位缺少就停止；filename 這類非必要欄位缺少就只警告
    missing_base = [c for c in base_output_cols if c not in df_labeled.columns]
    core_base = ["record_id", "t_start", "t_ref", "label", "distance_cm", "normal_end_sec", "leak_start_sec"]
    if any(c in core_base for c in missing_base):
        raise ValueError(f"缺少必要的 metadata 欄位：{[c for c in missing_base if c in core_base]}")
    if missing_base:
        print(f"⚠️ 以下 metadata 欄位不存在，不會輸出：{missing_base}")

    # 建立最終精簡輸出欄位清單:真正要交給推論程式用的精簡版欄位清單(基本欄位+模型特徵)
    requested_output_cols = (base_output_cols + final_keep_features)
    output_cols = list(dict.fromkeys(c for c in requested_output_cols if c in df_labeled.columns))
    
    
    # ============================================================
    # 輸出 2｜連續模型特徵版：*_continuous_selected.csv
    # 包含：metadata + Step2 selected features，保留 Transition 灰區。
    # 用途：連續模型分數、onset 偵測與灰區特徵變化分析。
    # 注意：灰區沒有明確真值，不直接用於分類 F1／Recall 評估。
    # ============================================================
    continuous_selected_csv = output_csv.with_name( output_csv.stem + "_continuous_selected.csv")
    df_continuous_selected = df_continuous[ output_cols].copy()
    df_continuous_selected.to_csv( continuous_selected_csv, index=False, encoding="utf-8-sig",)
    
    print("✅ continuous selected feature csv 已輸出:", continuous_selected_csv, )
        
    # 輸出 inference feature csv
    df_out = df_labeled[output_cols].copy()
    
    # 驗證是否跟STEP2輸出的特徵一樣
    feature_cols_only = [c for c in df_out.columns if c not in base_output_cols]
    missing = [c for c in final_keep_features if c not in feature_cols_only]
    extra = [c for c in feature_cols_only if c not in final_keep_features]
    
    print("selected features 預期數量 =", len(final_keep_features))
    print("selected features 實際輸出數量 =", len(feature_cols_only))
    print("missing features =", missing[:20])
    print("extra features =", extra[:20])
    
    if feature_cols_only != final_keep_features:
        raise ValueError(f"輸出欄位和 Step2 selected_feature_names 不一致：missing={missing[:10]}, extra={extra[:10]}")
    else:
        print("✅ 輸出特徵欄位與 Step2 selected_feature_names 完全一致")
    
    # ============================================================
    # 輸出 3｜明確標籤模型版：output_csv
    # 包含：metadata + Step2 selected features，僅保留 Normal(label=0)
    #       與 Leak(label=1) windows。
    # 用途：模型訓練、交叉驗證與 holdout 效能評估。
    # ============================================================
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    
    # ============================================================
    # rule 版本：保留模型特徵 + offband / veto 規則欄位，給 transition inference 套抗干擾後處理規則用
    # ============================================================
    # 建立一個清單，裡面放「除了模型 selected features 以外，規則可能會用到的額外欄位」。
    extra_rule_cols = [
        # offband 的平均能量與最大能量。用來判斷是不是整體非洩漏頻帶也一起變高，這通常比較像干擾
        "offband_E_mean",
        "offband_E_max",
        
        # 洩漏主頻帶的平均/最大能量。可以跟 offband 比較，判斷能量是集中在洩漏頻帶，還是到處都高。
        "leak_focus_E_mean",
        "leak_focus_E_max",
        
        # 共振頻帶的平均/最大能量。用來看是不是共振頻帶也一起被拉高。
        "resonance_E_mean",
        "resonance_E_max",
    
        # offband 相對 leak-focused 的比例。如果 offband 比例很高，代表可能是寬頻干擾，不一定是真洩漏。
        "offband_to_leak_focus_E_ratio",
        "offbandmax_to_leak_focusmax_E_ratio",
        
        # offband 相對 resonance 的比例，也是用來輔助判斷干擾型態。
        "offband_to_resonance_E_ratio",
        "offbandmax_to_resonancemax_E_ratio",
    
        # offband 相對固定 baseline 的偏移量與 z-score。這些欄位可以看 offband 是否明顯高於正常 baseline。
        "rel_offband_E_mean",
        "z_offband_E_mean",
        "rel_offband_E_max",
        "z_offband_E_max",
    
        # leak-focused 頻帶相對 baseline 的偏移量與 z-score。可以看洩漏主頻帶是否明顯高於正常 baseline。
        "rel_leak_focus_E_mean",
        "z_leak_focus_E_mean",
        "rel_leak_focus_E_max",
        "z_leak_focus_E_max",
    
        # resonance 頻帶相對 baseline 的偏移量與 z-score。用來補充判斷共振頻帶是否異常。
        "rel_resonance_E_mean",
        "z_resonance_E_mean",
        "rel_resonance_E_max",
        "z_resonance_E_max",
    ]
    
    # 也保留每個 offband 原始頻帶特徵，方便未來規則直接指定單一 offband 欄位
    offband_band_names = {str(band["name"]) for band in band_config.offband_bands}
    extra_rule_cols += [
        c for c in df_continuous.columns
        if "offband" in c.lower() or any(f"{name}_" in c for name in offband_band_names)
    ]
    
    missing_rule_cols = [c for c in dict.fromkeys(extra_rule_cols) if c not in df_continuous.columns]
    if missing_rule_cols:
        print(f"ℹ️ 以下規則欄位這次沒有建立，不會輸出：{missing_rule_cols}")
        
    # 去除重複欄位，而且保留原本順序，過濾掉不存在的欄位
    extra_rule_cols = [ c for c in dict.fromkeys(extra_rule_cols) if c in df_continuous.columns]
    # 決定 rule CSV 最後要輸出的欄位(基本 metadata 欄位 + 模型 selected features + offband / veto 規則欄位)
    rule_output_cols = list(dict.fromkeys(output_cols + extra_rule_cols))
    # 從完整連續序列取出模型特徵與規則欄位；
    # 保留灰區，供 onset 推論與連續後處理規則使用。
    df_rule_out = df_continuous[rule_output_cols].copy()
    # 建立 rule CSV 的檔名
    rule_output_csv = output_csv.with_name(output_csv.stem.replace("_model", "") + "_rule.csv")
    # 把 rule 版資料輸出成 CSV
    df_rule_out.to_csv(rule_output_csv, index=False, encoding="utf-8-sig")
    
    print(f"✅ transition rule feature csv 已輸出: {rule_output_csv}")
    print("rule features 數量 =", len(extra_rule_cols))

    print(f"✅ inference feature csv 已輸出: {output_csv}")
    print("shape =", df_out.shape)
    return df_out


if __name__ == "__main__":
    # Step2(純正常/純洩漏資料的頻帶探索流程)跑完之後輸出的正式頻帶設定檔,裡面存的是leak_focused_bands(洩漏關鍵頻帶)、resonance_bands(共振頻帶)、offband_bands(參照頻帶)
    BAND_CONFIG_JSON = "/Users/paul/Desktop/Final/test_result/band_config_step2.json"
    
    # Step2 輸出的特徵層面的設定——selected_feature_names(最終選定要給模型用的特徵)
    FEATURE_CONFIG_JSON = "/Users/paul/Desktop/Final/test_result/feature_config_micro.json"
    
    # 上游特徵設定檔
    FEATURE_EXTRACTION_CONFIG = ("/Users/paul/Desktop/Final/test_result/feature_extraction_config.json")
    
    # 正常資料 baseline
    BASELINE_JSON = ("/Users/paul/Desktop/Final/test_result/site_baseline.json")
    
    
    # 執行 Transition window-level 特徵工程。
    # 本程式只建立供模型訓練、模型推論與後續告警規則驗證使用的特徵 CSV，本身不執行模型預測。
    build_inference_feature_csv(
        tdms_dir="/Users/paul/Desktop/Final/transition_test_0615_0616_0617_0706/merged_output/transition_merged_tdms",     # 真正要拿來做特徵工程的原始感測器訊號檔案
        meta_csv="/Users/paul/Desktop/Final/transition_test_0615_0616_0617_0706/merged_output/transition_meta.csv",        # 上面那批 TDMS 檔案配對的標註表
        band_config_json=BAND_CONFIG_JSON,             # Step2 定下來的正式設定頻段
        feature_extraction_config_path=(FEATURE_EXTRACTION_CONFIG),
        baseline_json=BASELINE_JSON,
        feature_config_json=FEATURE_CONFIG_JSON,       # Step2 定下來的正式設定特徵
        # 這支腳本真正要產出的東西——把上面那批 TDMS 訊號跑完整條特徵工程之後,最終要拿去給模型推論用的精簡特徵表(只含基本 metadata 欄位+ selected features)
        output_csv="/Users/paul/Desktop/Final/transition_feature_engineering/transition_30ml_test/inference_features_transition_model.csv",
        cache_dir="/Users/paul/Desktop/Final/transition_feature_engineering/transition_30ml_test/feature_cache",    # 中繼快取資料夾
        split_name="transition_30ml_test",
        n_jobs=6,
    )
    
    '''
    # 檢查特徵對齊
    build_inference_feature_csv(
        tdms_dir="/Users/paul/Desktop/Final/compare feature",   # 只放那支檔的資料夾
        meta_csv="/Users/paul/Desktop/Final/compare feature/compare_meta.csv",  # ← 用這個假 meta
        band_config_json=BAND_CONFIG_JSON,
        feature_extraction_config_path=FEATURE_EXTRACTION_CONFIG,
        baseline_json=BASELINE_JSON,
        feature_config_json=FEATURE_CONFIG_JSON,
        output_csv="/Users/paul/Desktop/Final/compare feature/offline_features.csv",
        cache_dir="/Users/paul/Desktop/Final/compare feature/cache",
        split_name="compare_test",
        n_jobs=2,
    )
    '''


