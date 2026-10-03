#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2：頻帶探索與正式特徵選擇流程
================================

本程式使用純正常與純洩漏 TDMS，提供兩條互斥流程：

A. Step2a：候選洩漏頻帶探索
B. Step2b：正式特徵工程與特徵選擇

本程式不建立固定正常 Baseline。
Discovery 與 Final Baseline 必須由 build_site_baseline_v2.py 預先建立。


一、資料隔離原則
----------------

資料分成三種用途：

1. Baseline calibration normal 只用來建立固定正常 Baseline。

2. Step2 dev：用於頻帶探索、特徵排名、模型重要性與特徵選擇。

3. Step2 holdout：不參與頻帶探索、特徵排名或特徵選擇，僅用於最後的 window-level 評估。

Baseline calibration TDMS 不得與 dev 或 holdout 重複。


二、支線 A：Step2a 頻帶探索
---------------------------

啟動條件：

- FORCE_BAND_DISCOVERY=True；或
- band_config_step2.json 不存在。

必要輸入：

- 純正常與純洩漏 TDMS
- band_config_step2_discovery.json
- discovery_baseline.json
- feature_extraction_config.json
- Step1 resonance band JSON

執行流程：

1. 在 record 層級切分 dev 與 holdout。
2. Step2a 僅使用 dev records。
3. 載入完整 Discovery sliding bands。
4. 驗證 Discovery Baseline 的設定指紋。
5. 對全部 Discovery bands 建立 window-level raw features。
6. 使用固定 Discovery Baseline 建立 rel 與 z。
7. 從 z 特徵建立 CUSUM。
8. 第一輪使用 Raw、rel、z、CUSUM：
   - 聚合成 record-level median
   - 建立 feature ranking
   - 依 feature-family 支持度彙整 band ranking
   - 初選 PRELIMINARY_BANDS_TO_KEEP 個頻帶
9. 第二輪只對初選頻帶的 E、RMS、EnvRMS
   建立 delta、local 與 rolling temporal features。
10. 合併初選頻帶的 Raw、rel、z、CUSUM 與 Temporal。
11. 再次聚合與排名。
12. 保留 TOP_BANDS_TO_KEEP 個候選洩漏頻帶。
13. 加入 Step1 resonance bands 資訊。
14. 輸出 candidate band config 後停止程式。

主要輸出：

- feature_importance_first_pass.csv
- band_importance_first_pass.csv
- leak_candidate_bands_first_pass.json
- step2a_record_features_full.csv
- step2a_record_feature_columns_summary.csv
- feature_importance_summary.csv
- band_importance_summary.csv
- leak_candidate_bands.json
- band_config_step2_candidate.json

Step2a 不使用 holdout、不覆寫正式 band config，也不執行正式模型特徵選擇。


三、人工頻帶定案
----------------

Step2a 完成後必須人工處理：

1. 檢查候選 leak-focused bands。
2. 加入正式 Offband。
3. 保留需要的 Resonance bands。
4. 確認 temporal_band_names。
5. 另存為 band_config_step2.json。
6. 將 band config schema_status 設為 final。
7. 使用正式頻帶重新建立 site_baseline.json。


四、支線 B：Step2b 正式特徵選擇
-------------------------------

啟動條件：

- FORCE_BAND_DISCOVERY=False
- band_config_step2.json 已存在且為 final

必要輸入：

- band_config_step2.json
- feature_extraction_config.json
- site_baseline.json
- 純正常與純洩漏 TDMS

執行流程：

1. 讀取 TDMS 並執行 Hard QC。
2. 建立 Diff channel。
3. 在 record 層級切分 dev 與 holdout。
4. 載入並驗證正式 Band config。
5. 合併 Leak-focused 與 Resonance bands。
6. 將 Offband 保留為規則與抗干擾特徵來源。
7. 載入並驗證固定正常 Baseline。
8. 對 dev 與 holdout 分別執行 Soft QC 報告。
9. 使用相同頻帶與訊號參數建立完整 Raw features。
10. 建立 Offband、Leak-focus、Resonance summary。
11. 對正式擴充欄位套用固定 Baseline，建立 rel/z。
12. 對 z 特徵建立 CUSUM。
13. 對 Leak-focused E/RMS/EnvRMS 建立 Temporal。
14. 將 dev window 特徵聚合成 record-level median。
15. 僅使用 dev 執行：
    - 洩漏區分能力排名
    - 跨距離穩定性排名
    - 距離敏感性評估
    - 模型重要性評分
    - 共線性移除
16. 排除：
    - Offband/Veto rule-only features
    - 人工禁用的高距離敏感特徵
17. 使用 Group CV 與 Fold-wise selection 做內部評估。
18. 使用 LODO 評估跨距離泛化。
19. 使用 holdout 做 window-level 最終評估。
20. 輸出完整特徵表、模型特徵表與特徵契約。


五、固定 Baseline 原則
----------------------

Dev 與 holdout 使用同一份預先建立的固定正常 Baseline。

禁止：

- 使用 dev normal 重新 fit Baseline
- 使用 holdout normal 重新 fit Baseline
- 使用每筆資料前幾個 window 建立 Baseline
- 缺值時使用 mean=0、std=1
- 使用當下資料作為 fallback

定義：

    rel_x = x - baseline_mean

    z_x = (x - baseline_mean) / baseline_std


六、模型特徵與規則特徵
----------------------

模型候選特徵可來自：

- Leak-focused bands
- Resonance bands
- Raw features
- rel/z
- CUSUM
- Temporal features

以下欄位只供抗干擾規則使用，不可自動選入模型：

- Offband 原始特徵
- Offband summary
- Offband-to-leak ratios
- Veto 判斷欄位


七、Feature execution plan
--------------------------

feature_config_micro.json 必須記錄：

- baseline_features
- cusum_input_features
- temporal_base_features
- final_keep_features

後續 Transition、Disturbance 與 Runtime
只計算這份 execution plan 所要求的特徵鏈。

feature_config_micro.json 內的 threshold 與連續窗參數
只是 Step2 階段的初始或 fallback 設定。

正式模型 threshold、Two-stage 與 Veto 規則由後續：

- metadata.json
- postprocess_config.json
- runtime_feature_schema.json

完成定案。


八、正式輸出
------------

- feature_config_micro.json
- selected_feature_names_micro.csv
- features_dev_full_micro_DC.csv
- features_holdout_full_micro_DC.csv
- features_dev_selected_micro_DC.csv
- features_holdout_selected_micro_DC.csv
- dev_record_ids_micro.csv
- holdout_record_ids_micro.csv
- CV、Fold-wise、LODO、Permutation 與 Holdout 結果
- 特徵排名、稽核與視覺化結果


九、本程式不負責
----------------

本程式不負責：

- 建立固定 Baseline
- 人工決定 Offband
- 正式模型訓練與儲存
- Disturbance 告警規則搜尋
- Transition 事件層級驗證
- 正式 Postprocess 設定定案
- Runtime Schema 生成
- Runtime 即時推論
"""

# ============================================================
# 載入套件
# ============================================================
from nptdms import TdmsFile
import numpy as np
import warnings 
import re
from pathlib import Path
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, sosfiltfilt, hilbert
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.base import clone
from nptdms import TdmsWriter, ChannelObject
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import spearmanr, kruskal, kurtosis
from scipy.signal import correlate
from scipy.signal import spectrogram
import pickle
import hashlib
import json
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.linalg import lstsq
from dataclasses import dataclass

# 導入特徵工程共用函式
from shared_feature_utils import (
    load_or_compute_features_by_file,
    merge_band_groups,
    build_offband_summary_features,
    add_cusum_features,
    add_window_dynamic_features,
    add_local_rolling_features,
    audit_band_feature_coverage,
    audit_offband_summary_coverage,
    audit_derived_feature_coverage,
    audit_selected_feature_alignment,
    audit_rule_feature_alignment,
    make_feature_cache_hash,
    list_feature_caches,
    make_records_signature,
    make_file_specs_signature,
    load_one_tdms_record,
    _process_one_file_worker,
    extract_features_from_files_parallel,
    load_or_compute_features,
    _extract_one_window,
    _extract_features_parallel,
    parse_filename,
    sliding_window_records,
    compute_psd,
    compute_band_energy,
    bandpass_filter,
    extract_basic_time_features,
    extract_shape_only_psd_features,
    extract_temporal_event_features,
    extract_band_features_one_channel,
    extract_window_features_core_bands,
    BandConfig,
    calculate_file_sha256,
    load_feature_extraction_config,
    load_fixed_normal_baseline,
    apply_fixed_normal_baseline,
    configure_temporal_band_names,
    generate_sliding_bands,
    validate_baseline_source_isolation,
    
    # 全域變數
    TEMPORAL_BAND_NAMES,
    NPERSEG,       # Welch PSD 每段長度 2048
    FREQ_MIN,      # 找 peak 或做頻譜分析時，只看 5 kHz 以上 (5000)
    FS,            # 預設採樣率 1MHz(1_000_000)
    REMOVE_DC
    )

# 中文字型設定
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS',   # macOS 常見可用中文字型
    'Noto Sans CJK TC',   # 中文 fallback
    'PingFang TC'         # Apple 系統字型
]

plt.rcParams['axes.unicode_minus'] = False

# ======================================
# 全域常數/頻段設定
# ======================================
# 正式版本先不沿用舊資料留下來的 low / aux band
# 目前正式 band 只相信：
# 1. Step2 重新收斂出的 leak-focused bands
# 2. Step1 找出的 resonance bands
USE_LEGACY_SUPPORT_BANDS = False     

# soft QC / 頻譜一致性檢查用的大頻段，用來粗略觀察 near / far / diff 在主要分析區間是否有異常
QC_BANDS = [{"name": "b_all_5_70k", "f_low": 5_000, "f_high": 70_000}]

# near / far 頻譜比較用的頻帶清單，目前主要用舊版 low/core/aux bands 做輔助視覺化與比較，不代表正式模型最終使用頻帶
SPECTRAL_COMPARE_BANDS = QC_BANDS

# 主要分析：0 / 30
TARGET_LEVELS = {0, 30}
TARGET_LEAK_LEVELS = {30}
ANALYSIS_LEVELS = sorted(TARGET_LEVELS)          # [0, 30]
ANALYSIS_LEAK_LEVELS = sorted(TARGET_LEAK_LEVELS)  # [30]

TOP_N_FEATURES_TO_PLOT = 20
TOP_K_CANDIDATES = 200

MIN_FINAL_FEATURES = 30     # 新增最少保留特徵數


# 移除高風險距離特徵
drop_distance_sensitive = [
    "ai0_crest",
    "ai0_std",
    "high_low_ratio",
    "centroid_ai1_global",
    "diff_crest",
    "ai1_rms",
]

# 先初始化為空集合，再根據本次資料實際收斂出的 selected_leak_bands 動態指定  TEMPORAL_BAND_NAMES = set()
# 設定 DC 前處理 REMOVE_DC = True

# 第一輪：98 個候選頻帶先保留 25 個
PRELIMINARY_BANDS_TO_KEEP = 25

# 第二輪：加入 temporal 後，最後保留 12 個
TOP_BANDS_TO_KEEP = 12

# 每個 band 至少需要幾種 feature family 支持
MIN_FEATURE_FAMILY_COUNT = 2

if PRELIMINARY_BANDS_TO_KEEP < TOP_BANDS_TO_KEEP:
    raise ValueError(
        "PRELIMINARY_BANDS_TO_KEEP "
        "不可小於 TOP_BANDS_TO_KEEP"
    )

# ============================================================
# Step 2 正式上游設定
# ============================================================
FINAL_OUTPUT_ROOT = Path("/Users/paul/Desktop/Final/test_result")

FEATURE_EXTRACTION_CONFIG_PATH = (FINAL_OUTPUT_ROOT / "feature_extraction_config.json")

# Step2a 探索模式專用
DISCOVERY_BAND_CONFIG_PATH = (FINAL_OUTPUT_ROOT /"band_config_step2_discovery.json")
DISCOVERY_BASELINE_PATH = (FINAL_OUTPUT_ROOT /"discovery_baseline.json")

# Step2b 正式模式專用
FIXED_BASELINE_PATH = (FINAL_OUTPUT_ROOT / "site_baseline.json")
BAND_CONFIG_PATH = (FINAL_OUTPUT_ROOT / "band_config_step2.json")

# False：如果正式 band_config 已存在，就直接使用
# True：即使正式 band_config 已存在，也強制重新搜尋頻帶
FORCE_BAND_DISCOVERY = False

# 實驗室 baseline 是 candidate，所以目前 False
# 現場正式部署時再改成 True
REQUIRE_FINAL_BASELINE = False


# ============================================================
# Baseline 資料隔離檢查
# ============================================================

# False：目前舊實驗室資料可能同時被用於建立 Baseline 與 Step2 分析，因此只顯示警告，不停止程式。
# True：正式流程使用。Baseline-only 正常資料不得與 Step2 dev 或 holdout 使用相同 TDMS／record_id。
# 未來改用獨立 Baseline-only 正常資料後，必須將此值改成 True。
STRICT_BASELINE_ISOLATION = False


# Step1 輸出的共振頻帶設定檔
# Step2 會讀入這份 resonance bands，與本次 leak-focused bands 一起組成 final_core_bands
RESONANCE_JSON_PATH = Path(
    "/Users/paul/Desktop/Final/resonance/pipe_resonance_bands/configs/pipe_resonance_bands_resonance_bands.json"
)


# ======================================
# Step2b
# ======================================
# Step2b 正式擴充層要展開哪些 band raw feature
# 先保守用 E / RMS / EnvRMS
STEP2_EXPAND_SUFFIXES = ("E", "RMS", "EnvRMS")

# 如果之後想再加強，可以改成：
# STEP2_EXPAND_SUFFIXES = ("E", "RMS", "EnvRMS", "Entropy", "Flatness")


# ******************************  資料讀取/前處理/windowing  ******************************
# ============================================================
# 1. 資料讀取模組
# ============================================================
#  讀取tdms檔案:每個檔案都有雙通道ai0(fai-normal),ai1(near-leak)
#  列出所有的group,groups會是一個list,裡面每個元素都是一個dmsGroup物件

def load_tdms_data(data_folder, file_pattern = '*.tdms', max_points_per_file = 1_000_000, fs_target = 1000000,stop_flag = None):    
    """
    從資料夾批次讀取 .tdms 感測器訊號檔 → 解析訊號與 metadata → 整理成可用於後續機器學習的結構pandas DataFrame
    
    Parameters
    ----------
    data_folder 資料夾路徑 : str or Path
    file_pattern 檔案格式，預設 '*.tdms' : str
    max_points_per_file 每個檔案最多讀取的點數 : int (tdms檔可能很大，避免記憶體爆掉)
    fs_target 目標採樣率（若檔案中無法取得） : float  --> 檔案中取得為1000000
    stop_flag 停止讀檔函數（用於 GUI） : callable or None
        
    Returns
    ----------
        records : list of dict
        每個 dict 包含: filename, signal, fs, meta
    """
    print("=" * 60)
    print("階段 1: 載入 TDMS 資料")
    print("=" * 60)
    
    data_folder = Path(data_folder)
    files = sorted(data_folder.rglob(file_pattern))  # 使用 **/*.tdms 遞迴抓取所有子資料夾的 tdms
    
    if not files:    # 如果沒有檔案，直接報錯
        raise ValueError(f"在{data_folder}中找不到{file_pattern}")
    print(f"📁 在 {data_folder} 找到 {len(files)} 個檔案")    

    # 先確認group 名稱,channel順序是否正確
    for file_path in files[:5]:   # 只看前 5 個
        print(f"\n=== {file_path.name} ===")
        tdms = TdmsFile.read(file_path)
    
        for g in tdms.groups():
            print(f"Group: {g.name}")
            for i, ch in enumerate(g.channels()):
                ni_name = ch.properties.get("NI_ChannelName", "N/A")
                print(f"  [{i}] ch.name={ch.name}, NI_ChannelName={ni_name}")
    '''
    === leak0_0ml_4bar_straight0cm_20260325_160942.tdms ===
    Group: AE_Signals
      [0] ch.name=ai0, NI_ChannelName=N/A
      [1] ch.name=ai1, NI_ChannelName=N/A
     可以得到：
     channels[0] -> ai0 -> far
     channels[1] -> ai1 -> near
    '''
    records = []    # 建立一個 list，每個檔案都會變成一個 record dict
    
    # 逐檔處理
    for idx, file_path in enumerate(files, 1):  # 索引值從1開始      
        # ===== 停止判斷(可選) =====
        if stop_flag is not None and stop_flag():
            print(f"\n⚠️ 用戶中斷，已處理 {idx-1}/{len(files)} 個檔案")
            break
        # 顯示檔案讀取進度
        print(f"\r處理 [{idx}/{len(files)}]: {file_path.name}", end='')
        
        try:
            # 讀取 TDMS
            tdms = TdmsFile.read(file_path)
            group = tdms.groups()[0]
            # 利用channel name找通道名稱
            channels    = {ch.name: ch for ch in group.channels()}
            ch_names    = list(channels.keys())
            # 優先找 aio/ai1,找不到就用順序 fallback
            if "ai0" in channels and "ai1" in channels:
                channel_ai0 = channels["ai0"]
                channel_ai1 = channels["ai1"]
            else:
                # fallback：用順序，並警告
                ch_list = group.channels()
                warnings.warn(
                    f"{file_path.name}: 找不到 ai0/ai1，"
                    f"改用順序（channels: {ch_names}）"
                )
                channel_ai0 = ch_list[0]      # channel ai0: far-normal
                channel_ai1 = ch_list[1]      # channel ai1: near-leak
                
            signal_ai0 = channel_ai0[:max_points_per_file]    # 讀取訊號資料，最大100萬個點，不足點會以檔案大小為主
            signal_ai1 = channel_ai1[:max_points_per_file]
            
            # 解析檔名(使用寫好之parse_filename函式)、file_path.name指指拿檔名本身
            meta = parse_filename(file_path.name)
            
            # 確認tdms資料中有哪些資料
            #print("Channel properties:")
            #for k, v in channel.properties.items():
                #print(f"  {k}:{v}")
            
            '''
            leak0_0ml_03bar_straight40cm_ss.tdmsChannel properties:
              NI_Scaling_Status:unscaled
              NI_Number_Of_Scales:2
              NI_Scale[1]_Scale_Type:Linear
              NI_Scale[1]_Linear_Slope:0.000323503
              NI_Scale[1]_Linear_Y_Intercept:-0.001425779
              NI_Scale[1]_Linear_Input_Source:0
              NI_ChannelName:cDAQ9181-210F82CMod1/ai2
              unit_string:Volts
              NI_UnitDescription:Volts
              wf_start_time:2025-08-04T06:36:22.238478
              wf_increment:9.9999999999997e-07    採樣率fs=1000000,因此 Nyquist frequency = 500 kHz
              wf_start_offset:0.0
              wf_samples:1
            '''
            # 取得採樣率,fs取樣率-每秒量幾次(若檔案中有說多久量一次wf_increment那就只用檔案的，如果沒有就以fs_target當備用)
            fs = None
            if hasattr(channel_ai0, 'properties'):   # channel 物件裡，有沒有一個叫 properties 的東西
                inc = channel_ai0.properties.get('wf_increment')    # 找出感測器本身設定的取樣間隔時間
                if inc:
                    fs = 1.0 / inc
            if fs is None:   # 若tdms沒有sampling rate,使用預設
                fs = fs_target
            # 將資料整理存入每一筆record 
            records.append({
                "filename": file_path.name,
                "filepath": str(file_path),    # 品質檢查需要完整路徑
                "record_id": str(file_path.relative_to(data_folder.parent)),
                "fs": fs,
                "meta": meta,
                "signal": {
                    "ai0": signal_ai0,    # far-normal
                    "ai1": signal_ai1,    # near-leak
                },
                "label": None  # 之後再填
            })

        except Exception as e:
            warnings.warn(f"{file_path.name} 讀取失敗: {e}")
            
    print(f"\n完成讀檔，共 {len(records)} 個檔案")
    if records:
        fs_list = [r["fs"] for r in records]
        print(f"📏 採樣率範圍: {min(fs_list):.2f} ~ {max(fs_list):.2f} Hz")
    else:
        print("⚠️ 沒有成功讀入任何檔案")
    return records


#                                                                 =====     訊號分離     =====
# 1.訊號分離(diff = near - far)
# 差分函式：要先對其長度
def compute_diff_signal(records):
    '''
    每個 record 都包含： filename | ai1 | ai0 | label | diff
    
    每個 TDMS 對應的 record 就包含三組訊號：
    near / ai1 – 近洩漏感測器
    far / ai0 – 遠離洩漏感測器
    diff = ch_near − ch_far – 差分訊號，用來減掉背景、強化洩漏訊號

    統一在這裡做 DC removal，後面所有 feature extraction
    都使用去 DC 後的 ai0 / ai1 / diff
    '''
    result = []
    for r in records:
        r = r.copy()
        r["signal"] = r["signal"].copy()  # 避免改到原本的 dict
        
        ai0 = np.asarray(r["signal"]["ai0"])
        ai1 = np.asarray(r["signal"]["ai1"])
        
        n = min(len(ai0), len(ai1))   # 預防兩的通道訊號長度不同，以短的為主，方便對齊
        ai0 = ai0[:n]
        ai1 = ai1[:n]
        
        if REMOVE_DC:
            ai0 = ai0 - np.mean(ai0)
            ai1 = ai1 - np.mean(ai1)
        
        diff = ai1 - ai0
        
        r["signal"]["ai0"] = ai0
        r["signal"]["ai1"] = ai1
        r["signal"]["diff"] = diff
        
        result.append(r)
    return result


# 2.三通道(ai0/ai1/diff)存成新的 tdms 新檔
def save_diff_to_tdms(records, save_dir):
    """
    把每個 record 的 ai0 / ai1 / diff 三個通道存成一個 tdms 檔案
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)        # 建立好輸出資料夾
    saved_paths = []

    for r in records:
        ai0  = r["signal"]["ai0"]
        ai1  = r["signal"]["ai1"]
        diff = r["signal"]["diff"]
        fs   = r["fs"]
        
        fname = r["filename"].replace(".tdms", "_with_diff.tdms")     # 新黨名是在原本黨名後面加_with_diff
        fpath = save_dir / fname

        # 寫進 TDMS 的 channel properties
        props = {
            "wf_increment":    1.0 / fs,
            "wf_start_offset": 0.0,
            "label":           r["label"],
            "source":          r["filename"],
        }

        # 建立一個 group 叫 signals,裡面放三個channel(ai0,ai1,diff),用float64儲存,寫進同一個TDMS檔中
        with TdmsWriter(fpath) as writer:
            ch_ai0 = ChannelObject("signals", "ai0", data=ai0.astype(np.float64), properties=props)
            ch_ai1 = ChannelObject("signals", "ai1", data=ai1.astype(np.float64), properties=props)
            ch_diff = ChannelObject("signals", "diff", data=diff.astype(np.float64), properties=props)
            writer.write_segment([ch_ai0, ch_ai1, ch_diff])
            
        saved_paths.append(fpath)
        print(f"[Saved] {fpath}")
    print(f"✅ 共存 {len(records)} 個 tdms 檔案（三通道）")
    return saved_paths



#                                        *****************************************************************************
#                                                                     QC / 資料一致性檢查
#                                        *****************************************************************************
#                                                               =====     檢查異常資料     =====
'''
整體流程
    這段模組做的事是：
        1.先定義怎麼分 group
        2.先做硬性品質檢查
        3.再2對每個 group 做組內離群檢查
        4.把多個異常訊號整合成 soft_score
        5.最後輸出：
            - keep
            - review
            - remove
    你現在的實際策略是：
        - hard QC 才真的刪
        - soft QC 先保留，只做觀察和報表
'''
# meta來分組:哪一些 record 算同一組
def make_group_name(meta, use_pressure=False):    # use_pressure 預設為 False,是因為目前資料都是 4 bar
    '''
    把 meta 裡面的：leak_quantity, distance, pressure 整理成一個 group 名稱。
    例如：normal_0cm , leak150_20cm
    '''
    leak_q = int(meta["leak_quantity"])
    distance_cm = int(round(meta["distance"] * 100))
    pressure_bar = int(round(meta["pressure"]))

    leak_tag = "normal" if leak_q == 0 else f"leak{leak_q}"

    if use_pressure:
        return f"{leak_tag}_{distance_cm}cm_{pressure_bar}bar"
    return f"{leak_tag}_{distance_cm}cm"


#  硬性排除：異常資料檢查，normal/leak都可用，檢查：NaN/全零/clipping/RMS 過低/RMS 過高
def check_signal_quality(records, rms_min=1e-6):
    '''
    硬性品質檢查：NaN / Inf / 全零 / RMS過低 
    不做 clipping 檢查，因為目前沒有硬體量程資訊
    '''

    abnormal = []
    for r in records:
        for ch_name in ["ai0", "ai1"]:
            x = np.asarray(r["signal"][ch_name], dtype=float)

            # 檢查是否有 NaN / Inf
            if np.isnan(x).any()  or np.isinf(x).any():      
                abnormal.append(r["record_id"])
                break
            # 檢查是否全0
            if np.allclose(x, 0):      
                abnormal.append(r["record_id"])
                break
            # 檢查是否 RMS 太低
            rms = np.sqrt(np.mean(x**2))
            if rms < rms_min:          
                abnormal.append(r["record_id"])
                break

    return sorted(set(abnormal))


# 計算以中位數為中心的穩健離散程度，值越大代表這組資料在組內本來就比較分散；越小代表很集中
def _mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))    # 表示每個數值離中位數多遠,再取絕對值,再根據這些數值取中位數


# 每個 group 裡面，各自找離群值
def add_group_robust_outlier_flags(
        df,                  # 整個 DataFrame
        group_col,           # 分組欄位，例如 condition
        value_col,           # 要檢查的欄位，例如 rms_ratio
        z_thresh=2.5,        # robust z-score 超過多少算離群
        min_group_size=10,    # 每組至少幾筆才做離群判斷
        direction="both",    # 抓雙邊異常(低/高)
    ):
    """
    對某個欄位做 group-wise robust z-score，產生組內離群 flag
    例如你拿 rms0 進來，它就會做：
    1.每個 group 算自己的 median
    2.每個 group 算自己的 MAD
    3.把每筆資料轉成 robust z-score
    4.如果 |z| > z_thresh，就標成 group outlier

    """
    # 準備各欄位名稱
    median_col = f"{value_col}_group_median"
    mad_col = f"{value_col}_group_mad"
    z_col = f"{value_col}_robust_z"
    flag_col = f"flag_{value_col}_group_outlier"
    side_col = f"{value_col}_outlier_side"

    # 計算「每一組的中位數」,同一個 group 裡所有資料都會拿到同一個 median
    df[median_col] = df.groupby(group_col)[value_col].transform("median")
    # 計算「每一組的mad」,每筆資料除了知道自己原本的值，也知道自己那組的離散程度
    df[mad_col] = df.groupby(group_col)[value_col].transform(_mad)

    # 把 MAD 轉成比較接近標準差尺度的量, 1.4826 是 robust statistics 常用的校正常數, + 1e-12 是避免分母變成 0。
    min_scale = 1e-6    # 避免MAD是0
    scale = np.maximum(1.4826 * df[mad_col], min_scale)
    # 計算 robust z-score, 這筆資料距離組內中位數有多遠，距離用「該組自己的變異尺度」來標準化
    df[z_col] = (df[value_col] - df[median_col]) / scale

    # 如果只想抓「偏低」的異常，那就把 robust z-score 小於 -2.5 的標成 True
    if direction == "low":
        raw_flag = df[z_col] < (-z_thresh)
    elif direction == "high":    # 抓高
        raw_flag = df[z_col] > z_thresh
    else:   # 如果不是只抓偏低，而是上下都要抓，就用絕對值判斷,這樣太高或太低都算異常
        raw_flag = np.abs(df[z_col]) > z_thresh

    # 確認組內資料量夠
    df[flag_col] = (df["group_size"] >= min_group_size) & raw_flag
    
    df[side_col] = "normal"     # 先全部預設為normal
    df.loc[df[flag_col] & (df[z_col] < -z_thresh), side_col] = "low"     # 已經被判定為outlier,且它的 robust z-score 小於 -z_thresh,就標記為low
    df.loc[df[flag_col] & (df[z_col] > z_thresh), side_col] = "high"     # 已經是 outlier,且 robust z-score 大於 +z_thresh,就標記為high
    
    return df


# 根據一筆資料的各種 flag，最後決定它是 remove、review、keep
def classify_soft_decision(row, review_score_thresh=2):
    """
    軟性規則 soft QC 分級：
    - hard fail -> remove
    - 至少兩個特徵家族出現 group outlier -> review
    - 其他 -> keep
    """
    if row["hard_fail"]:    # hard fail → remove
        return "remove"
    
    if row["soft_score"] >= review_score_thresh:    # soft score 達標 → review
        return "review"
    
    return "keep"


#  軟性異常評分：負責把所有 record 過濾異常檔案，產生完整報表
def check_leak_channel_consistency(
        records,                      # 要檢查的 leak records
        target_bands=None,            # 要檢查哪些頻帶；建議由主程式明確傳入，例如 final_main_bands + offband_bands_for_rule
        rms_ratio_min=0.8,            # 只作 warning
        peak_ratio_min=0.8,           # 只作 warning
        band_ratio_min=0.7,           # 只作 warning
        outlier_z_thresh=2.5,         # group-wise robust z-score 的離群門檻
        min_group_size=5,             # 每組至少幾筆才做離群判斷
        suspicious_score_thresh=2,    # 軟性異常累積幾分以上就列為 suspicious
        save_csv=None,                # 要不要把報表存成 CSV
        verbose=True                  # 要不要印摘要
    ):
    """
    洩漏資料的軟性一致性檢查：不直接刪除，只標記 suspicious

    核心邏輯：
    1. decision 由 group-wise outlier 決定
    2. near/far ratio 只做 warning
    3. hard QC 才直接 remove
    
    soft QC 目前僅用於探索性資料檢查與可視化觀察
    其結果不直接用於樣本排除或模型訓練資料篩選
    若未來將 soft QC 作為資料排除依據，則應改為在 dev/holdout split 後分開執行，以避免資料洩漏
    """
    # -- Step1: 決定頻帶；soft QC 不再偷用舊版 band，必須由主程式明確傳入 --
    if target_bands is None:
        raise ValueError("check_leak_channel_consistency 需要明確提供 target_bands")
        
    # 初始化
    rows = []     # 等等每筆資料會整理成一個 dict 放進來
    eps = 1e-12   # 避免除以 0 或 log(0)
    
    # -- Step2:逐筆計算每個檔案的基本指標 -- 
    for r in records:
        # 取出兩通道訊號
        ai0 = np.asarray(r["signal"]["ai0"] ,dtype=float)     # 遠端
        ai1 = np.asarray(r["signal"]["ai1"], dtype=float)     # 近端
        fs = float(r["fs"])                                   # 採樣率
        
        # RMS 可以理解成整體能量或平均振幅強度
        rms0 = np.sqrt(np.mean(ai0 ** 2))                     # far 能量
        rms1 = np.sqrt(np.mean(ai1 ** 2))                     # near 能量
        
        # 兩個通道的 peak，就是最大絕對振幅
        peak0 = np.max(np.abs(ai0))                           # far 最大振幅
        peak1 = np.max(np.abs(ai1))                           # near 最大振幅
        
        # 根據 meta 資料分組
        meta = r["meta"]
        group_name = make_group_name(meta, use_pressure=False)
        # 建一列基本資訊
        row = {
            "record_id": r["record_id"],                               # 唯一識別碼,其實也就是檔名
            "filename": r["filename"],                                 # 檔名
            "condition": group_name,                                   # 所屬condition, 例如 150_leak-near 0 cm
            "group": group_name,                                       # 組別, 150_leak-near 0 cm near
            "leak_quantity": meta["leak_quantity"],                    # 洩漏量
            "distance_cm": int(round(meta["distance"] * 100)),         # 洩漏距離
            "pressure": meta["pressure"],
            "rms0": rms0,                                              # ai0 的 RMS
            "rms1": rms1,                                              # ai1 的 RMS
            "rms_ratio": rms1 / (rms0 + eps),
            "peak0": peak0,                                            # ai0 的 Peak
            "peak1": peak1,                                            # ai1 的 Peak
            "peak_ratio": peak1 / (peak0 + eps),
            "hard_fail": False,                                        # 先預設 False, 已在前面計算了
            # 對稱差異特徵：只看差異大不大，不看誰大誰小
            "log_rms_gap": abs(np.log((rms1 + eps) / (rms0 + eps))),
            "log_peak_gap": abs(np.log((peak1 + eps) / (peak0 + eps))),
            # 保留方向，知道是 near 高還是 far 高
            "log_peak_ratio": np.log((peak1 + eps) / (peak0 + eps)),
        }

        # -- Step3:計算個頻帶特徵,對每個 target band，它會算：band_e0, band_e1, band_ratio = e1 / e0, band_log_gap = abs(log(e1/e0))
        for band in target_bands:
            e0 = compute_band_energy(ai0, fs, band)
            e1 = compute_band_energy(ai1, fs, band)

            row[f"{band['name']}_e0"] = e0         # ai0 在這段頻帶的能量
            row[f"{band['name']}_e1"] = e1         # ai1 在這段頻帶的能量
            row[f"{band['name']}_ratio"] = e1 / (e0 + eps)
            row[f"{band['name']}_log_gap"] = abs(np.log((e1 + eps) / (e0 + eps)))     # 對稱差異特徵，後面會拿來抓 group outlier

        rows.append(row)

    # -- Step4:整理成report_df -- 
    report_df = pd.DataFrame(rows)
    report_df["group_size"] = report_df.groupby("group")["filename"].transform("count")    # 新增 group_size，表示每個 group 有幾筆資料,後面判斷「小樣本組不要硬抓離群」的依據
    
    # 把所有 band ratio 欄位名稱整理成 list，方便後面重複使用
    band_ratio_cols = [f"{b['name']}_ratio" for b in target_bands]

    # -- Step5:決定哪些欄位要做 group-wise outlier:定義了哪些欄位會真的拿去做「組內離群判斷」 -- 
    feature_cols = [
        # 幅度類
        "rms0",
        "rms1",
        "peak0",
        "peak1",
        # 差異類
        "log_rms_gap",
        "log_peak_gap",
        "log_peak_ratio",
        # 頻帶能量類
    ] + [f"{b['name']}_e0" for b in target_bands] \
      + [f"{b['name']}_e1" for b in target_bands] \
      + [f"{b['name']}_log_gap" for b in target_bands]
    
    # -- Step6:建立特徵家族：避免特徵高相關欄位重複加分,是以有幾個特徵家族一次異常來計算 -- 
    feature_families = {
        "amplitude": ["rms0", "rms1", "peak0", "peak1"],
        "gap": ["log_rms_gap", "log_peak_gap", "log_peak_ratio"] + [f"{b['name']}_log_gap" for b in target_bands],
        "band_energy": [f"{b['name']}_e0" for b in target_bands] + [f"{b['name']}_e1" for b in target_bands],
    }

    # -- Step7:對每個特徵都做一次「在同 group 內抓離群」,算完後，每個欄位都會多出：xxx_robust_z, flag_xxx_group_outlier, xxx_outlier_side
    for col in feature_cols:
        report_df = add_group_robust_outlier_flags(
            report_df,
            group_col="group",     # 代表是在 condition + label 內比較
            value_col=col,
            z_thresh=outlier_z_thresh,
            min_group_size=min_group_size,
            direction="both",      # 代表高的和低的都抓
        )

    # -- Step 8：把欄位層級整合成家族層級 -- 
    # soft score:每個 group outlier flag 算 1 分,看總共中了幾個(只要這個家族中任一欄離群, 就算這個家族異常)
    for family_name, cols in feature_families.items():
        family_flag_cols = [f"flag_{col}_group_outlier" for col in cols]
        report_df[f"family_{family_name}_outlier"] = report_df[family_flag_cols].any(axis=1)

    # -- Step 9：根據家族數算 soft_score -- 
    #    soft_score = 0：沒有家族異常, soft_score = 1：只有一類異常, soft_score >= 2：至少兩類異常，比較值得 review
    family_flag_summary_cols = [f"family_{name}_outlier" for name in feature_families]
    report_df["soft_score"] = report_df[family_flag_summary_cols].sum(axis=1)
    report_df["is_suspicious"] = report_df["soft_score"] >= suspicious_score_thresh       # 如果中標的特徵數量大於等於門檻，就先列為可疑
    
    # -- Step 10：ratio warning 只保留，不直接影響 decision -- 
    # 把 ratio 保留下來當輔助警示:如果 rms_ratio 太低 或 peak_ratio 太低 或某個 band ratio 太低,那就標記 ratio_warning = True,但不參與決策
    report_df["ratio_warning"] = (
        (report_df["rms_ratio"] < rms_ratio_min) |
        (report_df["peak_ratio"] < peak_ratio_min)
    )
    for col in band_ratio_cols:
        report_df["ratio_warning"] = report_df["ratio_warning"] | (report_df[col] < band_ratio_min)
    
    # -- Step 11：整理真正的異常/warning 的原因 --
    #    (A)把 ratio warning 轉成文字(生成人類可讀原因)
    def summarize_ratio_warning(row):
        reasons = []

        if row["rms_ratio"] < rms_ratio_min:
            reasons.append("rms_ratio_low")
        if row["peak_ratio"] < peak_ratio_min:
            reasons.append("peak_ratio_low")
        for col in band_ratio_cols:
            if row[col] < band_ratio_min:
                reasons.append(f"{col}_low")

        return "|".join(reasons) if reasons else "ok"

    report_df["ratio_reason"] = report_df.apply(summarize_ratio_warning, axis=1)

    #    (B)把真正影響 decision 的原因整理成文字
    def summarize_reason(row):
        reasons = []
        for col in feature_cols:
            if row[f"flag_{col}_group_outlier"]:
                side = row[f"{col}_outlier_side"]
                reasons.append(f"{col}_group_outlier_{side}")
        return "|".join(reasons) if reasons else "ok"

    # 原因填回報屌
    report_df["reason"] = report_df.apply(summarize_reason, axis=1)
    
    # -- Step 12：產生最後 decision：hard_fail -> remove, group outlier 數量 >= suspicious_score_thresh -> review, 否則 -> keep
    report_df["decision"] = report_df.apply(
        lambda row: classify_soft_decision(
            row,  
            review_score_thresh=suspicious_score_thresh
        ),
        axis=1,
    )
    
    #    (C)家族結果資訊：總結到家族層級
    def summarize_family_reason(row):
        reasons = []
        for family_name in feature_families:
            if row[f"family_{family_name}_outlier"]:
                reasons.append(f"{family_name}_outlier")
        return "|".join(reasons) if reasons else "ok"    # 串成字串
    
    report_df["family_reason"] = report_df.apply(summarize_family_reason, axis=1)

    # -- Step 13：算 max_abs_z(這筆資料在所有 QC 特徵裡，最極端的 robust z-score 有多大)
    report_df["max_abs_z"] = report_df[
        [f"{c}_robust_z" for c in feature_cols if f"{c}_robust_z" in report_df.columns]
    ].abs().max(axis=1)
    
    # 把單筆 record 分類結果整理成清單
    suspicious_files = report_df.loc[report_df["is_suspicious"], "record_id"].tolist()      # 去 report_df 裡找出 is_suspicious == True 的那些列,只取出 record_id -> 哪些檔案被 soft QC 判成「可疑」
    review_files = report_df.loc[report_df["decision"] == "review", "record_id"].tolist()   # 找 decision == "review" 的 record,把 record_id 取出來 -> 最後決策被標成 review 的檔案清單
    keep_files = report_df.loc[report_df["decision"] == "keep", "record_id"].tolist()       # 最後被判定為正常保留的檔案清單
    remove_files = report_df.loc[report_df["decision"] == "remove", "record_id"].tolist()   # 最後被判定要移除的檔案清單
    
    # -- Step 14：輸出 group-level summary各類檔名清單
    #    看整個group整體中位數很偏, 補足 group-wise outlier 抓不到「整組一起偏掉」的問題
    group_summary_cols = feature_cols.copy()     # 每個 group 中位數摘要
    
    #    對每個 group，把它的物理條件欄位整理出來
    group_meta_summary = (
        report_df.groupby("group")[["leak_quantity", "distance_cm", "pressure"]].median().reset_index()
        )
    
    # 這個 group 是什麼條件 + 這個 group 的各種特徵中位數是多少
    group_feature_summary = (
        group_meta_summary.merge(
            report_df.groupby("group")[group_summary_cols].median().reset_index(), on="group", how="left")
    )

    if verbose:
        print("\n" + "=" * 60)
        print("leak channel consistency 軟性檢查")
        print("=" * 60)
        print(f"總檔案數: {len(report_df)}")
        print(f"suspicious 數: {len(suspicious_files)}")
        print(f"review 數: {len(review_files)}")
        print(f"keep 數: {len(keep_files)}")

        print("\n各 group suspicious 比例:")
        # 整體統計
        group_summary = (
            report_df.groupby("group")["is_suspicious"]
            .agg(["count", "sum", "mean"])
            .rename(columns={"count": "n_total", "sum": "n_suspicious", "mean": "suspicious_ratio"})
            .sort_values("suspicious_ratio", ascending=False)
        )
        print(group_summary.to_string())

        print("\ndecision 分布:")
        print(report_df["decision"].value_counts(dropna=False).to_string())

        print("\n前 20 筆 review 檔案:")
        show_cols = [
            "filename",
            "group",
            "group_size",
            "soft_score",
            "decision",
            "ratio_warning",
            "ratio_reason",
            "reason",
        ]
        print(report_df.loc[report_df["decision"] == "review", show_cols].head(20).to_string(index=False))

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(save_csv, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"\nQC 報表已儲存: {save_csv}")

    return report_df, suspicious_files, remove_files, review_files, keep_files, group_feature_summary





#                                        *****************************************************************************
#                                                                       共用特徵工程工具
#                                        *****************************************************************************
# ============================================================
# 合併時間特徵(A+B)
# ============================================================
def add_step2_temporal_features(
    df,
    raw_feature_cols,
    group_col="record_id",
    time_col="t_ref",
):
    """
    對 Step2 的 window-level 特徵表，加上正式要納入特徵篩選的 temporal features。
    """
    # 先加 window-to-window dynamic features:前後窗變化量
    df = add_window_dynamic_features(
        df=df,
        feature_cols=raw_feature_cols,
        group_col=group_col,
        time_col=time_col,
        prev_windows=(3, 5),
    )
    
    # 再加 local rolling features:局部統計與趨勢
    df = add_local_rolling_features(
        df=df,
        feature_cols=raw_feature_cols,
        group_col=group_col,
        time_col=time_col,
        windows=(3, 5),
    )

    return df


# ============================================================
#  從指定 bands 精準挑要擴充的 base features
# ============================================================
def select_step2_expand_base_features(
    df,      # 特徵表
    bands,   # 指定要看的頻帶
    suffixes=("E", "RMS", "EnvRMS"),    # 想挑的特徵類型，預設是 E, RMS, EnvRMS
):
    """
    從指定 bands 中，挑出 Step2b 要做正式擴充的 base features。
    這裡不是挑所有 raw feature，
    而是挑要往下建立 rel / z / cusum / temporal 的那一批。
    """
    # 建立空清單，存最後找到的欄位
    cols = []

    # 同時兼容幾種欄位命名：
    # 1. band_E
    # 2. band_ai1_E
    # 3. band_diff_E
    # 4. band_ai0_E
    channel_patterns = [
        "{name}_{suffix}",
        "{name}_ai1_{suffix}",
        "{name}_diff_{suffix}",
        "{name}_ai0_{suffix}",
    ]
    
    # 逐一處理你指定的每個 band
    for band in bands:
        name = band["name"]     # 取出 band 的名稱
        # 對每一種特徵 suffix 檢查一次
        for suffix in suffixes:
            # 對每一種欄位命名模板檢查一次
            for pattern in channel_patterns:
                # 把 band name 和 suffix 填進模板
                col = pattern.format(name=name, suffix=suffix)
                if col in df.columns:
                    # 如果組出來的欄位名稱真的存在於 df.columns，就加入 cols
                    cols.append(col)

    return sorted(dict.fromkeys(cols))


# ============================================================
# 判斷特徵是否只能提供後處理規則使用
# ============================================================
def is_rule_only_feature(
    feature_name,
    offband_bands,
):
    """
    判斷某個特徵是否屬於 Offband／Veto 規則專用欄位。

    Rule-only 特徵仍會被計算並保留在完整特徵表中，
    但不得參與：

    - 模型特徵排名
    - 模型重要性評估
    - 共線性篩選
    - final selected features
    - 模型訓練

    它們只供模型預測後的抗干擾規則使用。
    """
    feature_name = str(feature_name)

    # 攔截群組摘要與比值，例如：
    # offband_E_mean , offband_E_max
    # offband_to_leak_focus_E_ratio , offbandmax_to_leak_focusmax_E_ratio
    # rel_offband_E_mean , z_offband_E_mean
    if "offband" in feature_name.lower():
        return True

    # 攔截單一 Offband 的原始特徵:
    # 即使 Offband 的名稱本身沒有包含 "offband"，只要特徵名稱包含正式設定中的 Offband band name，仍應視為 Rule-only。
    offband_names = {str(band["name"]) for band in offband_bands}

    return any(f"{band_name}_" in feature_name for band_name in offband_names)


#                                        *****************************************************************************
#                                                                     Step2a：候選頻帶探索
#                                        *****************************************************************************
# ==========================
# 從特徵名稱反推出 band 名稱
# ==========================
def infer_band_name_from_feature(feature_name):
    '''
    目前 Step2a 主要是抓 scan_... 這類候選頻帶名稱
    例如：
       rel_scan_001_5_15k_E
       z_scan_012_60_70k_E
       cusum_pos_z_scan_020_100_110k_E
     都要能抓回 scan_001_5_15k 這種 band 名稱
    '''
    # 把輸入轉成字串
    feature_name = str(feature_name)
    # 用 regex 去找 band 名稱
    match = re.search(r"(scan_\d+_\d+_\d+k)", feature_name)
    # 如果有成功找到，就把抓到的 band 名稱回傳
    if match:
        return match.group(1)

    return None


# ==========================
# 判斷特徵屬於哪一種 family
# ==========================
def infer_feature_family(feature_name):
    '''因為同一段 band 可能會產生很多衍生特徵, 如果不分 family，同一段 band 可能因為衍生特徵太多而看起來很強'''
    # 如果特徵名是 cusum_pos_z_ 開頭，就把它歸類成 cusum_pos_z family
    if feature_name.startswith("cusum_pos_z_"):
        return "cusum_pos_z"
    # 如果特徵名是 cusum_neg_z_ 開頭，就把它歸類成 cusum_neg_z family
    if feature_name.startswith("cusum_neg_z_"):
        return "cusum_neg_z"
    # 如果特徵名是 z_ 開頭，就把它歸類成 z family
    if feature_name.startswith("z_"):
        return "z"
    # 如果特徵名是 rel_ 開頭，就把它歸類成 rel family
    if feature_name.startswith("rel_"):
        return "rel"
    # 如果特徵名是 delta_ 開頭，就把它歸類成 delta family
    if feature_name.startswith("delta_"):
        return "delta"
    # 如果特徵名是 local_ 開頭，就把它歸類成 local family
    if feature_name.startswith("local_"):
        return "local"
    # 如果特徵名是 rolling_ 開頭，就把它歸類成 rolling family
    if feature_name.startswith("rolling_"):
        return "rolling"
    # 如果特徵名是 _E 結尾，就把它歸類成 E family，原始 band 能量特徵
    if feature_name.endswith("_E"):
        return "E"
    # 如果以上都不符合，就歸類成 other
    return "other"


# ==========================
# 從特徵重要性反推 band，並降低同一 band 因衍生特徵過多而灌水的影響，每個 family 保留前 2 個代表特徵
# ==========================
def summarize_feature_importance_by_band(feature_importance_df, discovery_bands):
    rows = []

    # 建立 band 查表，後面可依 band_name 補回 f_low / f_high
    band_lookup = {b["name"]: b for b in discovery_bands}

    # 先把每個 feature 對應到 band 與 family
    for _, row in feature_importance_df.iterrows():
        feat = row["feature"]
        imp = row["importance"]

        band_name = infer_band_name_from_feature(feat)
        if band_name is None:
            continue

        family = infer_feature_family(feat)

        rows.append({
            "band_name": band_name,
            "feature": feat,
            "importance": imp,
            "feature_family": family,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 每個 band、每個 family 只保留前 2 個最強特徵
    # --------------------------------------------------------
    family_top2 = (
        df.sort_values("importance", ascending=False)
        .groupby(["band_name", "feature_family"], group_keys=False)
        .head(2)
        .reset_index(drop=True)
    )

    # 原始總特徵數：只是觀察用，幫你知道某段 band 原本有多少特徵上榜
    raw_feature_count_df = (
        df.groupby("band_name")
        .agg(raw_feature_count=("feature", "count"))
        .reset_index()
    )

    # --------------------------------------------------------
    # 先在 family 層級整理
    # --------------------------------------------------------
    family_summary = (
        family_top2.groupby(["band_name", "feature_family"])
        .agg(
            top2_feature_count=("feature", "count"),            # 這個 family 保留了幾個代表（1 或 2）
            mean_top2_importance=("importance", "mean"),        # 前 2 個代表的重要性平均
            max_importance_in_family=("importance", "max"),     # 這個 family 最強代表的重要性
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 往上整理成 band 層級
    # --------------------------------------------------------
    summary = (
        family_summary.groupby("band_name")
        .agg(
            representative_feature_count=("top2_feature_count", "sum"),     # 所有 family 保留下來的代表特徵總數
            feature_family_count=("feature_family", "nunique"),             # 這段 band 被幾種不同 family 支持
            mean_importance=("mean_top2_importance", "mean"),               # 各 family 的 top2 平均後，再對 family 取平均
            max_importance=("max_importance_in_family", "max"),             # 所有 family 中最強代表的重要性
        )
        .reset_index()
    )

    # 把原始特徵數量補回來，方便你觀察「原本聲量」有多大
    summary = summary.merge(raw_feature_count_df, on="band_name", how="left")

    # 補回 band 頻率範圍
    summary["f_low"] = summary["band_name"].map(lambda x: band_lookup[x]["f_low"])
    summary["f_high"] = summary["band_name"].map(lambda x: band_lookup[x]["f_high"])

    # --------------------------------------------------------
    # 綜合 band 分數
    # --------------------------------------------------------
    summary["band_score"] = (
        0.4 * summary["feature_family_count"]    # 看支持的證據類型有多廣
        + 0.4 * summary["mean_importance"]       # 看整體代表特徵平均有多強
        + 0.2 * summary["max_importance"]        # 看這段 band 最強峰值有多強
    )

    # 依 band_score 排序，分數高的 band 排前面
    summary = summary.sort_values(
        ["band_score", "feature_family_count", "mean_importance"],
        ascending=False,
    ).reset_index(drop=True)

    return summary


# ============================================================
# 將 ranking 結果整理成統一的 feature / importance 格式
# 第一輪與第二輪共用
# ============================================================
def build_feature_importance_table(ranking_df):    # ranking_df: rank_features_for_leak_levels 之類的排名函式算出來的結果
    """
    將 rank_features_for_leak_levels() 的結果，
    整理成兩欄：
        - feature
        - importance
    因為不同版本的 ranking 可能使用不同分數欄名，所以依優先順序自動判斷。
    """
    # 檢查傳進來的東西真的是 pd.DataFrame
    if not isinstance(ranking_df, pd.DataFrame):
        raise TypeError("ranking_df 必須是 pandas DataFrame")
    # 如果傳進來的排名表是空的，直接擋下來
    if ranking_df.empty:
        raise ValueError("ranking_df 是空的，無法建立特徵重要性表")
    # 檢查 "feature" 這欄一定要存在
    if "feature" not in ranking_df.columns:
        raise ValueError("ranking_df 缺少 feature 欄位")

    # 列出一份「可能代表重要性分數」的候選欄名清單，而且順序就是優先順序——final_score, 退而求其次找 score；再沒有就找 abs_cohen_d；最後才是 cohen_d
    importance_candidates = [
        "final_score", "score",
        "abs_cohen_d", "cohen_d",
    ]

    # 依照 importance_candidates 的順序,依序檢查每個候選欄名是否真的存在於 ranking_df.columns 裡
    importance_col = next(
        (
            column
            for column in importance_candidates
            if column in ranking_df.columns
        ), None,)
    
    # importance_col 是 None（代表四個候選欄名都不存在），丟出錯誤
    if importance_col is None:
        raise ValueError(f"找不到可用的 importance 欄位；目前 ranking 欄位：請確認 {ranking_df.columns.tolist()}")

    # 格式統一
    feature_importance_df = (
        ranking_df[["feature", importance_col]]    # 從原表裡只挑出這兩欄 "feature", importance_col ，組成一個新的、較小的 DataFrame
        .rename(columns={importance_col: "importance",}).copy())    # 把剛剛動態找到的那個欄位（不管它原本叫 final_score 還是 score 還是 cohen_d）統一改名成 "importance"
    
    # 把 feature 這欄強制轉成字串型別
    feature_importance_df["feature"] = (feature_importance_df["feature"].astype(str))

    # 把 importance 欄強制轉成數值型別
    feature_importance_df["importance"] = (
        pd.to_numeric(feature_importance_df["importance"], errors="coerce",))

    # 先清掉異常值（inf）、再清掉真正的缺值列、依重要性排序、最後整理出乾淨連續的索引
    feature_importance_df = (
        feature_importance_df.replace([np.inf, -np.inf],np.nan,).dropna(subset=["feature", "importance"])
        .sort_values("importance",ascending=False,).reset_index(drop=True))
    # 再檢查一次結果是不是空的
    if feature_importance_df.empty:
        raise ValueError("importance 清理後沒有可用特徵")

    return feature_importance_df


# ==========================
# 輸出候選頻帶 JSON 設定檔
# ==========================
def save_leak_candidate_band_json(output_json,        # 把檔案存到哪裡
                                  selected_bands,     # 這次 Step2a 最後保留的候選頻帶清單
                                  source_note="micro_leak_feature_importance"):    # 附註這份檔案是怎麼來的
    # 組一個 Python 字典，等等要寫成 JSON    
    payload = {
        "source": source_note,
        "candidate_bands": selected_bands,
    }
    # 把輸入路徑轉成 Path 物件, 如果輸出資料夾還不存在，就先建立
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    # 把剛剛的 payload 寫成 JSON 檔
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ==========================
# 讀取正式 band 設定（Step 2b 專用，只接受 schema_status=final）
# ==========================
def load_frozen_band_config(band_config_path):     
    # 凍結版 band 設定檔的路徑,轉成 Path 物件
    band_config_path = Path(band_config_path)
    
    # 存在性檢查(band config 這份檔案找不到)
    if not band_config_path.exists():
        raise FileNotFoundError(f"找不到正式 band config：{band_config_path}")

    # 讀檔＋解析 JSON，存進 payload
    payload = json.loads(band_config_path.read_text(encoding="utf-8"))
    
    # 檢查 schema_name。這一關確保不會誤把別種設定檔
    if payload.get("schema_name") != "band_config":
        raise ValueError("band_config_step2.json 的 schema_name 必須是 band_config")
    # 對 schema_version 的檢查是同樣的版本鎖定
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("目前正式流程只接受 band config schema_version=2")
    # 寫死只接受 final
    if payload.get("schema_status") != "final":
        raise ValueError(f"Step 2b 只能載入 schema_status=final 的 band config；目前檔案：{band_config_path}")
    
    # 把預期的合併策略字串存成一個變數
    expected_merge_policy = ("keep_primary_overlap_""drop_secondary_overlap_v1")
    
    # band_merge_policy:這份 band config 在合併頻段時，實際用了哪種重疊處理邏輯
    if (payload.get("band_merge_policy") != expected_merge_policy):
        raise ValueError(f"band_merge_policy 不一致；預期：{expected_merge_policy}")

    # 先列出必要欄位清單，用 list comprehension 一次抓出所有缺的欄位，缺了就把完整清單丟進 ValueError
    required_keys = [
        "leak_focused_bands", "resonance_bands",
        "offband_bands", "temporal_band_names",
    ]
    missing_keys = [key for key in required_keys if key not in payload]

    if missing_keys:
        raise ValueError(f"band_config_step2.json 缺少欄位：{missing_keys}")
    # 逐一檢查每個必要欄位是不是真的是 list
    for key in required_keys:
        if not isinstance(payload[key], list):
            raise ValueError(f"band_config_step2.json 的 {key} 必須是 list")
    # 檢查 leak_focused_bands 是否為空清單
    if not payload["leak_focused_bands"]:
        raise ValueError("正式 band config 的 leak_focused_bands 不可為空")
    # # 檢查 offband_bands 是否為空清單
    if not payload["offband_bands"]:
        raise ValueError("正式 band config 的 offband_bands 不可為空；請先人工加入 offband 再定案")

    # 用 JSON 讀出來的四份清單去建構一個 BandConfig 物件
    band_config = BandConfig(
        leak_focused_bands=list(payload["leak_focused_bands"]),
        resonance_bands=list(payload["resonance_bands"]),
        offband_bands=list(payload["offband_bands"]),
        temporal_band_names=list(payload["temporal_band_names"]),
    )

    return payload, band_config


# ==========================
# 讀取探索頻段，並三層驗證
# ==========================
def load_discovery_band_config(config_path):      # 接收一個參數 config_path——探索階段輸出的那份 schema_status="discovery"
    # 統一轉成 Path 物件
    config_path = Path(config_path)
    # 確認檔案存在，不存在就丟出明確的 FileNotFoundError
    if not config_path.exists():
        raise FileNotFoundError(f"找不到探索頻帶設定：{config_path}")
    # 讀檔＋解析 JSON，存進 payload
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    # 確認這份檔案的種類正確,不是誤讀到別種設定檔
    if payload.get("schema_name") != "band_config":
        raise ValueError("discovery band config 的 schema_name 必須是 band_config")

    # 只接受 "discovery":頻帶搜尋流程自己要重新讀回之前的探索結果
    if payload.get("schema_status") != "discovery":
        raise ValueError("Step2a 只能載入 schema_status=discovery 的 band config")

    # 檢查你剛剛在 save_discovery_band_config 裡看到的那個探索階段獨有欄位 baseline_scope，確保它的值就是寫入時固定填的 "discovery_scan_grid"
    if payload.get("baseline_scope") != "discovery_scan_grid":
        raise ValueError("discovery band config 的 baseline_scope 必須是 discovery_scan_grid")
    # 取出 leak_focused_bands 這個欄位，探索階段是把所有候選頻帶（不管篩不篩選）都塞進這個欄位
    discovery_bands = payload.get("leak_focused_bands", [],)

    # 檢查 discovery_bands 不是空的
    if not discovery_bands:
        raise ValueError("discovery band config 沒有候選頻帶")

    return payload, list(discovery_bands)


# ==========================
# 讀取共振頻帶 JSON
# ==========================
def load_resonance_band_json(resonance_json_path):
    # 先把路徑轉成 Path
    resonance_json_path = Path(resonance_json_path)

    # 檢查檔案在不在，如果找不到，就直接報錯
    if not resonance_json_path.exists():
        raise FileNotFoundError(f"找不到共振頻帶 JSON: {resonance_json_path}")

    # 把共振 JSON 檔讀進來，轉回 Python 字典
    payload = json.loads(resonance_json_path.read_text(encoding="utf-8"))
    resonance_bands = payload.get("resonance_bands", [])    # 從這份 JSON 裡面拿出 resonance_bands 這個欄位

    print(f"✅ 已讀取共振頻帶: {len(resonance_bands)} 段")
    return resonance_bands


# ==========================
# 判斷兩個 band 是否重疊(leak candidate band 有沒有跟 resonance band 重疊)
# ==========================
def bands_overlap(b1, b2):
    '''
    拿 Step2a 找到的 leak candidate bands，去比對它們有沒有跟 Step1 的 resonance bands 重疊
        - 哪些 leak-focused band 剛好落在共振區
        - 哪些是比較獨立、不靠共振的 band
    '''
    return max(b1["f_low"], b2["f_low"]) < min(b1["f_high"], b2["f_high"])
    

# ==========================
# 輸出完整 band config
# ==========================
def save_band_config_json(
    output_json,
    leak_focused_bands,
    resonance_bands,
    offband_bands,
    temporal_band_names,
    source_note="step2_band_discovery_with_resonance",
    schema_status="candidate",
):
    """
    輸出後續正式流程要共用的 band config。

    正式概念上只保留三類 band：
    1. leak_focused_bands：本次資料收斂出的洩漏主頻帶
    2. resonance_bands：Step1 找出的共振頻帶
    3. offband_bands：後續干擾判斷 / veto 用的輔助頻帶
    """
    # 先建立要寫出去的內容
    payload = {
        "schema_name": "band_config",
        "schema_version": 2,
        "schema_status": schema_status,
        "source": source_note,                                            # 這份 band config 是怎麼來的
        
        "band_merge_policy": ("keep_primary_overlap_"
                              "drop_secondary_overlap_v1"),
        
        "leak_focused_bands": leak_focused_bands,
        "resonance_bands": resonance_bands,                               # Step1 已經找好的共振頻帶
        "offband_bands": offband_bands,                                   # 保留給後面 disturbance 判斷用的 offband
        "temporal_band_names": sorted(list(temporal_band_names)),         # 哪些 band 需要再額外做 temporal / CUSUM 類特徵
    }
    # 把輸出路徑轉成 Path 物件, 如果資料夾不存在，就先建立
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    # 把整份 payload 寫成 JSON 檔
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ 已輸出完整 band config: {output_json}")


# ==========================
# 篩選洩漏頻段主流程
# ==========================
def run_band_discovery_stage(
    records_dev,      # 輸入 dev 資料，也就是這次拿來做 band discovery 的正常/洩漏紀錄
    output_dir,       # 所有輸出檔要存到哪裡
    *,                # 全部強制關鍵字傳入
    window_sec,
    hop_sec,
    fs,
    nperseg,
    max_points,
    base_feature_version,
    resonance_json_path=None,      # 可選，若有提供就讀入 Step1 找到的共振頻帶
    top_bands_to_keep=TOP_BANDS_TO_KEEP,     # 最後要保留幾段候選 band，預設用全域設定
):
    # 把輸出路徑轉成 Path 物件，如果資料夾不存在就建立
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 讀取共振頻帶(先預設 resonance_bands 是空 list，如果有給 resonance_json_path，就去讀 Step1 的共振頻帶 JSON)
    resonance_bands = []
    if resonance_json_path is not None:
        resonance_bands = load_resonance_band_json(resonance_json_path)

    # 讀進地毯式掃描出來的全部候選頻帶
    discovery_payload, discovery_bands = (load_discovery_band_config(DISCOVERY_BAND_CONFIG_PATH))
    print(f"Step2a 候選頻帶數量: {len(discovery_bands)}")

    # ============================================================
    # Step2a 改成 by-file 版本
    # 原因：純正常 / 純洩漏 records 是整檔 TDMS，沒有 t_start
    # 必須先切成 windows，後面的特徵函式才吃得到 t_start
    # ============================================================
    # 建立 file_specs
    file_specs_dev = [
        {
            "filepath": r["filepath"],
            "record_id": r["record_id"],
            "label": r["label"],
        }
        for r in records_dev
    ]

    if not file_specs_dev:
        raise ValueError("Step2a 沒有可用的 dev file_specs")
    
    # 同步 temporal band names + 讀探索用 baseline
    configure_temporal_band_names(discovery_payload.get("temporal_band_names", [],))
    
    # 幫 Step2a 頻帶搜尋，準備一份「用來校正特徵」的固定基準值:把原始特徵轉換成「相對於正常狀態偏移了多少
    discovery_baseline = load_fixed_normal_baseline(
        DISCOVERY_BASELINE_PATH,                        # 專屬探索階段的 baseline
        band_config_path=DISCOVERY_BAND_CONFIG_PATH,    # 探索用 baseline 的指紋比對
        feature_extraction_config_path=(FEATURE_EXTRACTION_CONFIG_PATH),
        require_final=False,                            # 因為探索階段本來就是還在摸索的過程,不該要求 baseline 一定要是 "final"
    )
    # Step2a 專屬的檢查：確認這份 baseline 的 baseline_scope 也是 "discovery_scan_grid"
    if (discovery_baseline.get("baseline_scope") != "discovery_scan_grid"):
        raise ValueError("Step2a 必須使用 baseline_scope=discovery_scan_grid 的 baseline")   
        
    # ============================================================
    # Step2a Baseline 資料隔離檢查
    # ============================================================
    # 確認建立 discovery_baseline.json 的正常 TDMS，沒有再次被放入 Step2a dev 做頻帶重要性排名。
    #
    # 目前舊實驗室資料若有重複：STRICT_BASELINE_ISOLATION=False 時只會警告。
    #
    # 未來使用獨立 Baseline-only 資料後：STRICT_BASELINE_ISOLATION=True 時會直接阻止重複資料。
    validate_baseline_source_isolation(
        baseline_payload=discovery_baseline, analysis_records=records_dev,
        context="Step2a dev", strict=STRICT_BASELINE_ISOLATION,
    )
    
    # 對所有候選 band 做 window-level 特徵工程(全部探索候選頻帶)
    features_dev = load_or_compute_features_by_file(
        cache_dir=output_dir / "feature_cache",      # 特徵先存 cache，下次可直接讀
        file_specs=file_specs_dev,           # 處理這次 dev 資料
        core_bands=discovery_bands,          # 使用 discovery JSON 定義的候選頻帶
        aux_bands=[],                        # 探索階段不考慮 offband
        low_bands=[],
        split_name="band_discovery_dev",     # cache 和輸出命名
        n_jobs=-2,                           # 平行運算
        window_sec=window_sec,               # 每個視窗 0.2 秒
        hop_sec=hop_sec,                     # 視窗滑動步長 0.05 秒
        fs=fs,                               # 取樣率
        nperseg=nperseg,                     # Welch PSD 的設定
        target_levels=TARGET_LEVELS,         # 這次要比較哪些 leak level
        feature_version=base_feature_version,# 標記這次特徵版本
        max_points=max_points,
    )
    # 算出來的 features 轉成 DataFram，重排索引
    df_dev = pd.DataFrame(features_dev).reset_index(drop=True)
    
    # 用 list comprehension 從原始 record 裡抽出四個 metadata 欄位（record_id、洩漏量、距離、壓力），組成一份 DataFrame, 依 record_id 去重
    record_meta_dev_df = pd.DataFrame([
        {
            "record_id": r["record_id"],
            "leak_quantity": r["meta"]["leak_quantity"],
            "distance_cm": int(round(r["meta"]["distance"] * 100)),
            "pressure": r["meta"]["pressure"],
        }
        for r in records_dev]).drop_duplicates(subset=["record_id"])
    
    # ============================================================
    # Step2a：套用探索 baseline 建立 rel/z
    # ============================================================
    # 涵蓋全部候選頻帶（不是篩過的）建立 rel/z
    discovery_base_feature_cols = (
        select_step2_expand_base_features(
            df=df_dev,
            bands=discovery_bands,       # 涵蓋全部候選頻帶（不是篩過的）
            suffixes=("E", "RMS", "EnvRMS"),
        )
    )
    
    if not discovery_base_feature_cols:
        raise ValueError("Step2a 找不到可套用 baseline 的候選特徵")
    
    # 用剛才讀進來的探索用 discovery_baseline，對這些候選特徵套用你已經很熟悉的 apply_fixed_normal_baseline，產生 rel_*/z_*
    df_dev = apply_fixed_normal_baseline(
        df=df_dev,
        feature_cols=discovery_base_feature_cols,
        baseline_payload=discovery_baseline,
    )
    
    # ============================================================
    # Step2a：由 z 特徵建立 CUSUM
    # ============================================================
    discovery_z_feature_cols = [f"z_{feature_name}" for feature_name in discovery_base_feature_cols]
    
    df_dev = add_cusum_features(
        df=df_dev,
        input_feature_cols=discovery_z_feature_cols,
        k=0.02,
        decay=0.97,
    )
    
    # ============================================================
    # Step2a 第一輪：全部 discovery bands 使用 raw + rel/z + CUSUM 初選
    # ============================================================
    # apply_fixed_normal_baseline 產生的 rel / z 欄位
    discovery_rel_feature_cols = [
        f"rel_{feature_name}"
        for feature_name in discovery_base_feature_cols
    ]
    
    discovery_z_feature_cols = [
        f"z_{feature_name}"
        for feature_name in discovery_base_feature_cols
    ]
    
    # add_cusum_features 產生的正向／負向 CUSUM 欄位
    discovery_cusum_feature_cols = [
        derived_name
        for feature_name in discovery_z_feature_cols
        for derived_name in (
            f"cusum_pos_{feature_name}",
            f"cusum_neg_{feature_name}",
        )
    ]
    
    # 第一輪明確限制只使用：raw + rel + z + CUSUM
    # 把四種欄位家族（原始值、rel、z、CUSUM）全部串接起來，湊出「第一輪要用的完整特徵清單」
    first_pass_feature_cols = list(dict.fromkeys(     # dict.fromkeys(...) 去重保序技巧
        discovery_base_feature_cols
        + discovery_rel_feature_cols
        + discovery_z_feature_cols
        + discovery_cusum_feature_cols))
    
    # 確認前面組出來的特徵清單真的都存在於 df_dev 裡
    missing_first_pass_features = [
        feature_name
        for feature_name in first_pass_feature_cols
        if feature_name not in df_dev.columns
    ]
    
    # 直接列出「全部」缺失欄位不同
    if missing_first_pass_features:
        raise ValueError(f"Step2a 第一輪缺少必要特徵：{missing_first_pass_features[:30]}")
    
    print("\n=== Step2a 第一輪 ===")
    print("全部 discovery bands:", len(discovery_bands))
    print("第一輪特徵數:", len(first_pass_feature_cols))
    
    # 將第一輪 window 特徵聚合成 record-level:聚合 record-level + 排名 + 換算成 band 重要性
    first_pass_record_df = (
        build_record_level_feature_table(
            window_df=df_dev,
            qc_record_df=None,
            feature_cols=first_pass_feature_cols,
            extra_meta_df=record_meta_dev_df,
            agg="median",
        )
    )
    
    # 第一輪特徵排名
    first_pass_ranking_df = (
        rank_features_for_leak_levels(first_pass_record_df, target_levels=tuple(ANALYSIS_LEVELS),))
    # 儲存 csv
    first_pass_ranking_df.to_csv(
        output_dir / "feature_importance_first_pass.csv", index=False, encoding="utf-8-sig")
    # 排名結果統一整理成 feature/importance 兩欄格式
    first_pass_feature_importance_df = (build_feature_importance_table(first_pass_ranking_df))
    
    # 第一輪從 feature importance 彙整成 band importance:把「每個特徵各自的重要性分數」彙整成「每個頻段的重要性分數」
    # 因為一個頻段（band）可能對應到好幾個特徵（E/RMS/EnvRMS 加上 rel/z/cusum 好幾種衍生版本），需要把這些屬於同一個頻段的特徵重要性合併起來,才能回答「這個頻段整體而言值不值得留下」
    first_pass_band_summary_df = (
        summarize_feature_importance_by_band(
            first_pass_feature_importance_df,
            discovery_bands=discovery_bands,
        )
    )
    
    # 排除掉那種「只有極少數幾個衍生特徵有分數、其他都缺值或被清理掉」的不可靠頻段——如果一個頻段底下大部分特徵家族都沒有有效分數，代表這個頻段的資料品質或訊號強度可能有問題
    first_pass_band_summary_df = (
        first_pass_band_summary_df[
            first_pass_band_summary_df[
                # 只保留 feature_family_count（這個頻段底下,實際有算出重要性分數的特徵家族數量）大於等於 MIN_FEATURE_FAMILY_COUNT 的頻段
                "feature_family_count"] >= MIN_FEATURE_FAMILY_COUNT].copy().reset_index(drop=True)
    )
    
    # 如果篩完一個頻段都不剩，報錯
    if first_pass_band_summary_df.empty:
        raise ValueError("Step2a 第一輪沒有留下候選頻帶，請檢查 baseline、特徵排名或放寬條件")
    # 存檔
    first_pass_band_summary_df.to_csv(
        output_dir / "band_importance_first_pass.csv", index=False, encoding="utf-8-sig",)
    
    # 第一輪挑出保留的頻帶：保留前 20～30 個 band
    preliminary_band_count = min(PRELIMINARY_BANDS_TO_KEEP, len(first_pass_band_summary_df),)
    
    # 依 first_pass_band_summary_df（前面已經依重要性排序過）取前 preliminary_band_count 名的 band_name，轉字串、轉 list
    preliminary_band_names = (
        first_pass_band_summary_df
        .head(preliminary_band_count)["band_name"]
        .astype(str).tolist()
    )
    
    # 用 dict comprehension 建一個「頻帶名稱 → 完整頻帶定義（含 f_low/f_high）」的查找表 discovery_band_lookup
    discovery_band_lookup = {str(band["name"]): band for band in discovery_bands}
    
    # 依照 preliminary_band_names 的順序，把對應的完整頻帶字典找出來，組成 preliminary_bands
    preliminary_bands = [
        discovery_band_lookup[band_name]
        for band_name in preliminary_band_names
        if band_name in discovery_band_lookup
    ]
    
    # 如果一個都湊不出來,報錯
    if not preliminary_bands:
        raise ValueError("Step2a 第一輪無法建立 preliminary_bands")
    
    # 印出保留下來的每個頻帶名稱跟頻率範圍
    print("第一輪保留 bands:", len(preliminary_bands))
    for band in preliminary_bands:
        print(
            " ",
            band["name"],
            band["f_low"],
            "~",
            band["f_high"],
            "Hz",
        )
    
    # 輸出第一輪初選頻帶，方便人工檢查
    save_leak_candidate_band_json(
        output_json=(output_dir / "leak_candidate_bands_first_pass.json"),
        selected_bands=preliminary_bands,
        source_note=("step2a_first_pass_raw_rel_z_cusum"),)
    
    
    # ============================================================
    # Step2a 第二輪：只對第一輪初選 bands 建立完整 temporal features
    # ============================================================
    preliminary_base_feature_cols = (
        select_step2_expand_base_features(
            df=df_dev,
            bands=preliminary_bands,          # 只針對第一輪篩出來的頻帶（數量已經大幅縮減）挑基礎特徵，準備進一步建立 temporal 特徵。
            suffixes=STEP2_EXPAND_SUFFIXES,
        )
    )
    
    if not preliminary_base_feature_cols:
        raise ValueError("Step2a 第二輪找不到可建立 temporal 特徵的 base features")
    
    # 呼叫前後欄位集合做差集:記錄 temporal 建立以前有哪些欄位，之後用欄位差集找出新產生的 temporal 特徵
    columns_before_temporal = set(df_dev.columns)     # 把當下所有欄位名稱存成一個 set
    
    df_dev = add_step2_temporal_features(
        df=df_dev,
        raw_feature_cols=preliminary_base_feature_cols,
        group_col="record_id",
        time_col="t_ref",
    )
    
    temporal_feature_cols = [
        column
        for column in df_dev.columns
        if column not in columns_before_temporal
    ]
    
    # 如果一個新欄位都沒多出來，報錯
    if not temporal_feature_cols:
        raise ValueError("Step2a 第二輪沒有產生任何 temporal 特徵")
    
    print("\n=== Step2a 第二輪 ===")
    print("初選 bands:", len(preliminary_bands))
    print("temporal base features:", len(preliminary_base_feature_cols),)
    print("新產生 temporal features:", len(temporal_feature_cols),)
    
    # 第二輪特徵清單:第一輪特徵中只保留屬於 preliminary bands 的欄位
    preliminary_band_name_set = {str(band["name"]) for band in preliminary_bands}    # 把初選頻帶的名稱收集成一個 set，方便後面高效率查詢。
    # 第一輪已經算過 raw/rel/z/CUSUM 了，第二輪不需要重新計算，只要從第一輪的結果裡把「屬於初選頻帶」的那部分挑出來繼續用就好
    preliminary_first_pass_feature_cols = [
        feature_name
        for feature_name in first_pass_feature_cols
        if (
            infer_band_name_from_feature(feature_name)
            in preliminary_band_name_set
        )
    ]
    
    # 第二輪特徵：初選 band 的 raw/rel/z/CUSUM + 完整 temporal
    second_pass_feature_cols = list(dict.fromkeys(preliminary_first_pass_feature_cols + temporal_feature_cols))
    
    # 驗證
    missing_second_pass_features = [
        feature_name
        for feature_name in second_pass_feature_cols
        if feature_name not in df_dev.columns
    ]
    if missing_second_pass_features:
        raise ValueError(f"Step2a 第二輪缺少必要特徵：{missing_second_pass_features[:30]}")
    
    # 第二輪聚合成 record-level:聚合、存檔、排名
    record_df = build_record_level_feature_table(
        window_df=df_dev,
        qc_record_df=None,
        feature_cols=second_pass_feature_cols,
        extra_meta_df=record_meta_dev_df,
        agg="median",
    )
    
    record_df.to_csv(output_dir / "step2a_record_features_full.csv", index=False, encoding="utf-8-sig",)
    
    # 輸出第二輪欄位清單
    pd.DataFrame({"feature_name": second_pass_feature_cols,}).to_csv(
        output_dir /"step2a_record_feature_columns_summary.csv",index=False,encoding="utf-8-sig",)
    print("Step2a 第二輪 record-level 特徵數:",len(second_pass_feature_cols),)
    
    # 第二輪正式特徵排名：存的檔名是 feature_importance_summary.csv，沒有 _first_pass 後綴
    ranking_df = rank_features_for_leak_levels(record_df, target_levels=tuple(ANALYSIS_LEVELS),)
    ranking_df.to_csv(output_dir / "feature_importance_summary.csv", index=False, encoding="utf-8-sig",)
    feature_importance_df = (build_feature_importance_table(ranking_df))
    
    # 第二輪只在 preliminary bands 初選頻帶內做最終 band 排名
    band_summary_df = (
        summarize_feature_importance_by_band(feature_importance_df, discovery_bands=preliminary_bands,))
    # feature_family_count 篩選跟空表檢查
    band_summary_df = (
        band_summary_df[band_summary_df["feature_family_count"] >= MIN_FEATURE_FAMILY_COUNT].copy().reset_index(drop=True))
    if band_summary_df.empty:
        raise ValueError("Step2a 第二輪沒有留下正式候選頻帶")
        
    # 檢查和共振 band 是否重疊
    if resonance_bands:
        band_lookup = {b["name"]: b for b in discovery_bands}     # 用 band 名稱快速查回它的頻率範圍
        overlap_flags = []    # 記錄這段 band 有沒有跟共振 band 重疊
        overlap_names = []    # 記錄它重疊到哪幾段 resonance band

        # 逐段掃描目前保留下來的候選 band
        for band_name in band_summary_df["band_name"]:
            cur_band = band_lookup[band_name]          # 先找到這段 band 的頻率範圍
            # 把所有跟它有重疊的 resonance band 找出來
            matched = [
                rb["name"] for rb in resonance_bands
                if bands_overlap(cur_band, rb)
            ]
            overlap_flags.append(len(matched) > 0)                        # 記錄有沒有重疊
            overlap_names.append("|".join(matched) if matched else "")    # 把重疊的共振 band 名稱串起來
        # 把剛剛的重疊資訊補回 band 摘要表
        band_summary_df["overlaps_resonance"] = overlap_flags
        band_summary_df["matched_resonance_bands"] = overlap_names
    # 存最終的 band 重要性摘要表 
    band_summary_csv = output_dir / "band_importance_summary.csv"
    band_summary_df.to_csv(band_summary_csv, index=False, encoding="utf-8-sig")

    # 選出最後保留的候選 band(取前 top_bands_to_keep 名的頻帶名稱)
    selected_band_names = band_summary_df.head(top_bands_to_keep)["band_name"].tolist()
    band_lookup = {b["name"]: b for b in discovery_bands}
    selected_bands = [band_lookup[name] for name in selected_band_names if name in band_lookup]

    # 輸出候選洩漏 band JSON(只存 Step2a 找到的候選洩漏頻帶) - 「Step2a 純 discovery 結果」
    save_leak_candidate_band_json(
        output_json=output_dir / "leak_candidate_bands.json",
        selected_bands=selected_bands,
        source_note="step2a_micro_leak_band_discovery",
    )
    # 輸出正式 band config JSON(後續正式流程讀的完整 band config)
    save_band_config_json(
        output_json=output_dir / "band_config_step2_candidate.json",    # 避免把手動輸入的 offband 清空
        leak_focused_bands=selected_bands,   # Step2a 找到的洩漏主頻帶
        resonance_bands=resonance_bands,     # Step1 找到的共振頻帶
        offband_bands=[],                    # 目前先留空，之後做干擾規則再補
        temporal_band_names={b["name"] for b in selected_bands},    # 目前先指定只有 leak-focused bands 要做 temporal / CUSUM 類衍生特徵
        source_note="step2_band_discovery_with_resonance",
        schema_status="candidate",
    )

    print(f"✅ Step2a 完成，保留 {len(selected_bands)} 個候選頻帶")
    return {
        "discovery_bands": discovery_bands,    # 全部滑動候選 band
        "selected_bands": selected_bands,      # 最後保留的 leak-focused bands
        "resonance_bands": resonance_bands,    # Step1 讀進來的共振 band
        "ranking_df": ranking_df,              # 特徵重要性表
        "band_summary_df": band_summary_df,    # band 層級摘要表
    }




#                                        *****************************************************************************
#                                                 Step2b：record-level ranking / feature selection / validation
#                                        *****************************************************************************
# ***  record-level 聚合與 ranking 區  ***
# =====================================
# 把 window-level 特徵聚合成 record-level
# =====================================
def build_record_level_feature_table(
    window_df,                    # 每一列是一個 window
    qc_record_df=None,            # 可選，record-level 的 QC 結果表
    record_id_col="record_id",
    label_col="label",
    agg="median",                 # 要用 median 還是 mean 聚合 window 特徵
    exclude_cols=None,
    feature_cols=None,
    extra_meta_df=None,
):
    """
    把 window-level 特徵聚合成 record-level。
    用途：
    1. 若 feature_cols=None：
       自動抓數值特徵，適合 PCA / QC / 全體分析
    2. 若 feature_cols 有指定：
       只聚合指定特徵，適合 selected features 分析
    3. 可選 extra_meta_df：
       額外接回 leak_quantity / distance_cm / pressure 等 metadata
    """
    df = window_df.copy()    # 複製資料

    # 如果沒有指定排除欄位，使用預測排除清單
    if exclude_cols is None:
        exclude_cols = {
            "filename",
            "record_id",
            "label",
            "condition",
            "group",
            "leak_quantity",
            "distance_cm",
            "pressure",
        
            # 時間與索引欄位
            "t_start",
            "t_end",
            "t_ref",
            "window_idx",
            "source_index",
            "fs",
        
            # QC 欄位
            "decision",
            "soft_score",
            "ratio_warning",
            "ratio_reason",
            "reason",
            "family_reason",
        }

    # 如果有指定 feature_cols，就只用指定特徵
    if feature_cols is not None:
        numeric_cols = [c for c in feature_cols if c in df.columns]
    else:
        # 找出真正要聚合的數值特徵欄位
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
        ]

    # 依照 record_id 把所有 windows 聚合成一列
    # 同一個原始檔案切出來的所有 windows, 都會被分到同一組 -> 用一列代表這整筆 record 的典型特徵水準
    if agg == "mean":
        feat_df = df.groupby(record_id_col)[numeric_cols].mean().reset_index()
    else:
        feat_df = df.groupby(record_id_col)[numeric_cols].median().reset_index()

    # 把 record-level 的 label 抓回來
    meta_cols = [record_id_col, label_col]
    keep_meta = [c for c in meta_cols if c in df.columns]
    meta_df = df[keep_meta].drop_duplicates(subset=[record_id_col])

    record_df = feat_df.merge(meta_df, on=record_id_col, how="left")

    if qc_record_df is not None:
        qc_cols = [
            c for c in [
                record_id_col, "filename", "condition", "group",
                "leak_quantity", "distance_cm", "pressure",
                "decision", "soft_score", "ratio_warning",
                "ratio_reason", "reason", "family_reason"
            ]
            if c in qc_record_df.columns
        ]
        qc_map = qc_record_df[qc_cols].drop_duplicates(subset=[record_id_col])
        record_df = record_df.merge(qc_map, on=record_id_col, how="left")
        
    # merge 額外 metadata
    if extra_meta_df is not None:
        meta_cols2 = [c for c in extra_meta_df.columns if c != record_id_col]
        extra_map = extra_meta_df[[record_id_col] + meta_cols2].drop_duplicates(subset=[record_id_col])
        record_df = record_df.merge(extra_map, on=record_id_col, how="left")

    return record_df


# =============================================
# 每一筆 record 的 near/far 頻譜差異指標整理成一張表
# =============================================
def diagnose_baseline_vs_30(record_df,     # 已經整理好的 record-level 特徵表
                            out_dir,       # 輸出資料夾
                            top_n=30):     # 最後要印出前幾名特徵，預設 30 個
    """
    專門檢查 Baseline vs 30 ml 是否真的可分。
    用 record-level feature，避免 window 數量把信心灌水。
    """
    # 把輸出路徑轉成 Path 物件，然後確保輸出資料夾存在
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 篩選資料 -> 只保留：leak_quantity = 0 , leak_quantity = 30
    df = record_df[record_df["leak_quantity"].isin([0, 30])].copy()

    # 哪些欄位不是特徵欄，不應該拿來做比較
    exclude_cols = {
        "filename", "record_id", "label",
        "leak_quantity", "distance_cm", "pressure",
        "condition", "group",
        "decision", "soft_score", "ratio_warning",
        "ratio_reason", "reason", "family_reason",
    }

    # 自動找出要分析的特徵欄(條件是：特徵是數值型，且不在 exclude_cols 裡)
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
    ]
    # 先準備一個空 list，等等每個特徵算出來的結果會一列一列放進去
    rows = []
    # 開始逐個特徵處理
    for feat in feature_cols:
        x0 = df.loc[df["leak_quantity"] == 0, feat].dropna().values     # x0：Baseline 組的值
        x1 = df.loc[df["leak_quantity"] == 30, feat].dropna().values    # x1：30 ml 組的值

        # 如果其中一組樣本太少，小於 3 筆，就跳過這個特徵
        if len(x0) < 3 or len(x1) < 3:
            continue

        # 算這個特徵在 30 ml 相對 Baseline 的 Cohen's d(Cohen's d 是效果量，代表兩組差多大)
        d = _cohen_d(x1, x0)

        # 計算 AUC
        y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]    # y：真實標籤 Baseline = 0 , 30 ml = 1
        v = np.r_[x0, x1]    # 把兩組特徵值接起來

        try:    # 有些特徵是：30 ml 越大越像 leak，有些特徵可能相反：30 ml 越小越像 leak
            auc_pos = roc_auc_score(y, v)     # 直接用原值
            auc_neg = roc_auc_score(y, -v)    # 反過來用負值
            auc_best = max(auc_pos, auc_neg)    # 再取其中比較大的，這樣就能不管方向，先看「這個特徵最多能分多好」
            # 同時 direction 也會記錄方向:"higher_in_30ml"：30 ml 較高比較有辨識力 , "lower_in_30ml"：30 ml 較低比較有辨識力
            direction = "higher_in_30ml" if auc_pos >= auc_neg else "lower_in_30ml"
        except Exception:
            auc_best = np.nan
            direction = "unknown"

        rows.append({
            "feature": feat,       # 特徵名
            "n_baseline": len(x0),     # Baseline 樣本數
            "n_30ml": len(x1),      # 30 ml 樣本數
            "median_baseline": np.median(x0),     # Baseline 中位數
            "median_30ml": np.median(x1),     # 30 ml 中位數
            "median_gap_30_minus_baseline": np.median(x1) - np.median(x0),     # 中位數差值
            "cohen_d_30_vs_baseline": d,     # 效果量
            "abs_cohen_d": abs(d) if pd.notna(d) else np.nan,    # 效果量絕對值
            "auc_best_direction": auc_best,     # 最佳方向下的 AUC
            "direction": direction,     # 這個特徵是 30 ml 較高還是較低
        })

    # 把前面收集的結果轉成 DataFrame，並排序(auc_best_direction 高的排前面, 若 AUC 相同，再看 abs_cohen_d)
    diag_df = (
        pd.DataFrame(rows)
        .sort_values(["auc_best_direction", "abs_cohen_d"], ascending=False)
        .reset_index(drop=True)
    )
    # 把診斷結果存成 CSV
    diag_df.to_csv(out_dir / "diagnose_baseline_vs_30_features.csv", index=False, encoding="utf-8-sig")

    print("\n=== Baseline vs 30 ml diagnostic ===")
    print(diag_df.head(top_n).to_string(index=False))

    return diag_df


# ============================================================
# 哪些特徵最能分 normal / 30 
# ============================================================
def rank_features_for_leak_levels(record_df,     # record-level 特徵表
                                  target_levels=tuple(ANALYSIS_LEVELS)):     # 預設用全域常數 ANALYSIS_LEVELS 轉成 tuple
    """
    多 leak level 的 record-level 特徵排名
    目的：
    1. 找 normal vs 微小洩漏（尤其 30ml）最敏感的特徵
    2. 看特徵是否會隨 leak quantity 單調變化
    3. 看相鄰 leak level 之間是否也有區分力
    """
    # 先複製資料並只保留目標洩漏量
    df = record_df.copy()
    df = df[df["leak_quantity"].isin(target_levels)].copy()

    # 定義排除欄位清單
    exclude_cols = {
        "filename",
        "record_id",
        "label",
        "condition",
        "group",
        "leak_quantity",
        "distance_cm",
        "pressure",
    
        # 時間與索引欄位
        "t_start",
        "t_end",
        "t_ref",
        "window_idx",
        "source_index",
        "fs",
    
        # QC 欄位
        "decision",
        "soft_score",
        "ratio_warning",
        "ratio_reason",
        "reason",
        "family_reason",
    }

    # 找出候選特徵欄位:只要是數值欄位，而且不在排除名單裡，就視為候選特徵
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
    ]

    # 準備分組資訊
    rows = []    # 儲存每個特徵的評分結果
    sorted_levels = sorted(target_levels)
    base_level = 0     # baseline/正常）當比較基準
    leak_levels = [lv for lv in sorted_levels if lv != base_level]    # 用 list comprehension 濾掉 base_level，留下所有「真正有洩漏」的等級

    # 主迴圈：逐一計算每個特徵的評分
    for feat in feature_cols:
        level_data = {}
        # 特徵依照 leak_quantity 分組
        for lv in sorted_levels:
            # level_data[0] 是所有 baseline 樣本在這個特徵上的數值,level_data[30] 是所有洩漏 30ml 樣本的數值
            vals = df.loc[df["leak_quantity"] == lv, feat].dropna().values
            level_data[lv] = vals

        row = {"feature": feat,}

        # 各組 median：對每個洩漏等級,算這個特徵的中位數,存成 median_0、median_30 這種欄名
        for lv in sorted_levels:
            row[f"median_{int(lv)}"] = (
                np.median(level_data[lv]) if len(level_data[lv]) > 0 else np.nan
            )

        # normal vs 各 leak level：對每個「有洩漏」的等級,呼叫 _cohen_d，算「這個洩漏等級 vs baseline」的區分力,存成 d_0_30、abs_d_0_30（絕對值版本）
        for lv in leak_levels:
            d_val = _cohen_d(level_data[lv], level_data[base_level])
            row[f"d_0_{int(lv)}"] = d_val
            row[f"abs_d_0_{int(lv)}"] = abs(d_val) if pd.notna(d_val) else 0.0

        # 相鄰洩漏等級之間的區分力
        for lv1, lv2 in zip(sorted_levels[:-1], sorted_levels[1:]):
            d_adj = _cohen_d(level_data[lv2], level_data[lv1])
            row[f"d_{int(lv1)}_{int(lv2)}"] = d_adj
            row[f"abs_d_{int(lv1)}_{int(lv2)}"] = abs(d_adj) if pd.notna(d_adj) else 0.0

        # 算 Spearman 整體趨勢相關:這個特徵是否會隨 leak_quantity 增加而有單調變化, rho > 0：洩漏量越大，特徵值越大
        try:
            rho, _ = spearmanr(df["leak_quantity"], df[feat], nan_policy="omit")
        except Exception:
            rho = np.nan        
        row["rho_leakqty"] = rho

        # 算 Kruskal-Wallis 檢定:各組整體分布是否有顯著差異(適合非 常態資料)，只對有效 groups 計算
        try:
            # 篩出樣本數至少 2 筆的組別」，如果有效組別數 >= 2（至少要有兩組才能比較）就呼叫
            valid_groups = [
                level_data[lv] for lv in sorted_levels
                if len(level_data[lv]) >= 2
            ]
            if len(valid_groups) >= 2:
                kw_stat, kw_p = kruskal(*valid_groups)
            else:
                kw_stat, kw_p = np.nan, np.nan

        except Exception:
            kw_stat, kw_p = np.nan, np.nan

        row["kw_stat"] = kw_stat
        row["kw_p"] = kw_p

        rows.append(row)

    # 組成排名表 + 空表檢查
    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        print("⚠️ 沒有足夠資料做 multilevel 特徵比較")
        return rank_df
    
    # kw_stat 正規化，避免量級太大
    # 先把 kw_stat 欄裡的正負無窮大也當成缺值處理,再取最大值 kw_max。如果 kw_max 是有效數字且大於 0，就把每個特徵的 kw_stat 除以最大值,做 min-max 風格的正規化
    kw_max = rank_df["kw_stat"].replace([np.inf, -np.inf], np.nan).max()
    if pd.notna(kw_max) and kw_max > 0:
        rank_df["kw_stat_norm"] = rank_df["kw_stat"] / kw_max
    else:
        rank_df["kw_stat_norm"] = 0.0

    # 自定義總分：優先 normal vs 30，再看相鄰 level 與單調性
    # abs_d_0_30 = abs_d_0_30 = 這個特徵在 Baseline(0 ml) 和 Baseline+30 ml 之間的區分力大小。
    if sorted_levels == [0, 30]:
        rank_df["adjacent_effect_mean"] = rank_df["abs_d_0_30"].fillna(0)
        rank_df["leak_effect_score"] = rank_df["abs_d_0_30"].fillna(0)
        
    # 分支給更完整（三級以上）情境用的加權公式
    else:
        rank_df["adjacent_effect_mean"] = (
            rank_df["abs_d_0_30"].fillna(0) +
            rank_df["abs_d_30_60"].fillna(0)
        ) / 2.0
    
        rank_df["leak_effect_score"] = (
            0.50 * rank_df["abs_d_0_30"].fillna(0) +
            0.30 * rank_df["abs_d_0_60"].fillna(0) +
            0.20 * rank_df["adjacent_effect_mean"].fillna(0)
        )
    
    # 把 leak_effect_score 另外複製一份存成通用欄名 "score"
    rank_df["score"] = rank_df["leak_effect_score"]

    # 排序與輸出:組出排序依據的欄位清單, 主要依 "score" 排序
    sort_cols = ["score"]
    if "abs_d_0_30" in rank_df.columns:
        sort_cols.append("abs_d_0_30")
    if "abs_d_0_60" in rank_df.columns:
        sort_cols.append("abs_d_0_60")

    # 總分排序(如果分數差不多，再優先看abs_d_0_30)
    rank_df = rank_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    # 印出來預覽的欄位清單，固定包含 feature、score、rho_leakqty、kw_stat
    show_cols = ["feature", "score", "rho_leakqty", "kw_stat"]
    for col in ["d_0_30"]:
        if col in rank_df.columns:
            show_cols.append(col)

    print("\n=== Baseline / Baseline+30 Top 特徵 ===")
    print(rank_df[show_cols].head(20).to_string(index=False))

    return rank_df


# ============================================================
# 固定距離下比較 normal vs leak，檢查特徵是否跨距離穩定
# ============================================================
def rank_features_within_each_distance(
        record_df,
        target_levels=tuple(ANALYSIS_LEVELS),
        min_samples=3,
        ):
    """
    固定距離下比較 normal vs leak，檢查特徵是否跨距離穩定
    
    1.固定一個距離，在這個距離下比較 0 ml vs 30 ml
    2.算每個比較的 Cohen’s d
    3.再把所有距離上的結果整合起來
    4.分數高的特徵，就是跨距離比較穩定的特徵
    """
    # 複製資料並只保留指定 leak levels
    df = record_df.copy()
    df = df[df["leak_quantity"].isin(target_levels)].copy()

    # 排除不拿來當特徵的欄位
    exclude_cols = {
        "label", "leak_quantity", "distance_cm", "pressure",
        "record_id", "filename", "condition", "group",
        "decision", "soft_score", "ratio_warning",
        "ratio_reason", "reason", "family_reason"
    }

    # 找出真正的特徵欄位
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
    ]

    # 找出所有距離與所有 leak levels
    rows = []
    distances = sorted(df["distance_cm"].dropna().unique())
    leak_levels = [lv for lv in target_levels if lv != 0]

    # 逐個特徵處理
    for feat in feature_cols:
        row = {"feature": feat}

        # 初始化特徵收集器
        all_abs_d = []       # 收集這個特徵在所有距離、所有 leak level 下的 |Cohen's d|
        sign_hits = []       # 收集這個特徵在不同 leak level 下，「方向一致性」有多高

        # 逐個 leak level 處理
        for lv in leak_levels:
            d_vals = []

            # 在每個距離下比較 baseline vs leak
            for dist in distances:
                sub = df[df["distance_cm"] == dist]
                x0 = sub.loc[sub["leak_quantity"] == 0, feat].dropna().values       # 這個距離下，baseline 的特徵值
                x1 = sub.loc[sub["leak_quantity"] == lv, feat].dropna().values      # 這個距離下，lv 洩漏量的特徵值

                # 樣本不夠就跳過
                if len(x0) < min_samples or len(x1) < min_samples:
                    continue

                # 算這個距離下的 Cohen’s d
                d = _cohen_d(x1, x0)      # 在固定距離下，這個特徵對 Baseline vs Baseline+lv 的區分力
                if pd.notna(d):
                    d_vals.append(d)
            # 如果這個 leak level 一個有效距離都沒有，就補 NaN
            if len(d_vals) == 0:
                row[f"mean_abs_d_0_{lv}"] = np.nan
                row[f"min_abs_d_0_{lv}"] = np.nan
                row[f"sign_consistency_0_{lv}"] = np.nan
                continue

            # 計算這個 leak level 的跨距離穩定性
            d_vals = np.asarray(d_vals, dtype=float)
            row[f"mean_abs_d_0_{lv}"] = np.mean(np.abs(d_vals))     # 在所有距離下，平均區分力有多強
            row[f"min_abs_d_0_{lv}"] = np.min(np.abs(d_vals))       # 最差的那個距離，區分力還剩多少

            # 計算方向一致性:這個特徵在不同距離下，方向有沒有一致
            pos_ratio = np.mean(d_vals > 0)
            neg_ratio = np.mean(d_vals < 0)
            row[f"sign_consistency_0_{lv}"] = max(pos_ratio, neg_ratio)
            
            # 儲存資訊
            all_abs_d.extend(np.abs(d_vals).tolist())      # 累積所有 leak levels、所有距離的絕對區分力
            sign_hits.append(max(pos_ratio, neg_ratio))    # 累積每個 leak level 的方向一致性

        # 算整個特徵的總結指標
        row["distance_mean_abs_d"] = np.mean(all_abs_d) if len(all_abs_d) > 0 else np.nan        # 整體平均區分力,看這個特徵平均有沒有用
        row["distance_min_abs_d"] = np.min(all_abs_d) if len(all_abs_d) > 0 else np.nan          # 最差情況下的區分力,看它有沒有某些距離特別弱
        row["distance_sign_consistency"] = np.mean(sign_hits) if len(sign_hits) > 0 else np.nan  # 平均方向一致性,看它跨距離時趨勢穩不穩

        rows.append(row)

    out = pd.DataFrame(rows)

    # 計算最終的跨距離穩定分數
    out["distance_stability_score"] = (
        0.45 * out["distance_mean_abs_d"].fillna(0) +         # 最看重平均區分力
        0.35 * out["distance_min_abs_d"].fillna(0) +          # 也很重視最差距離下還能不能撐住
        0.20 * out["distance_sign_consistency"].fillna(0)     # 方向一致性也重要，但比前兩者稍微低一點
    )

    # 依穩定度排序:分數越高，代表這個特徵越值得信任
    out = out.sort_values("distance_stability_score", ascending=False).reset_index(drop=True)

    print("\n=== Cross-distance stable features ===")
    print(out[["feature", "distance_stability_score", "distance_mean_abs_d",
             "distance_min_abs_d", "distance_sign_consistency"]]
        .head(20).to_string(index=False)
    )

    return out


# ============================================================
# 固定洩漏量下比較不同距離，檢查特徵是否距離敏感
# ============================================================
def rank_features_within_each_leak_level(
    record_df,
    target_levels=tuple(ANALYSIS_LEVELS),
    min_samples=3,
):
    """
    固定洩漏量下比較不同距離，檢查特徵是否距離敏感

    分數越高 = 越容易受 distance 影響
    分數越低 = 距離越穩定
    """
    # 複製資料並只保留要分析的 leak levels
    df = record_df.copy()
    df = df[df["leak_quantity"].isin(target_levels)].copy()

    # 排除不是 feature 的欄位
    exclude_cols = {
        "label", "leak_quantity", "distance_cm", "pressure",
        "record_id", "filename", "condition", "group",
        "decision", "soft_score", "ratio_warning",
        "ratio_reason", "reason", "family_reason"
    }

    # 找出所有數值特徵
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
    ]

    rows = []
    sorted_levels = sorted(target_levels)

    # 逐個特徵處理
    for feat in feature_cols:
        row = {"feature": feat}

        # 建立這個特徵的總收集器
        all_pair_abs_d = []     # 不同距離 pairwise 比較的 |Cohen's d|
        all_rho = []            # distance 和 feature 的 Spearman 單調相關強度
        all_kw = []             # 不同距離整體分布差異的 Kruskal-Wallis 統計量

        # 固定某一個 leak level
        for lv in sorted_levels:
            sub = df[df["leak_quantity"] == lv].copy()
            dist_levels = sorted(sub["distance_cm"].dropna().unique())

            pair_abs_d_vals = []

            # 同一 leak level 內，做不同距離的 pairwise 比較
            for i, d0 in enumerate(dist_levels[:-1]):
                for d1 in dist_levels[i + 1:]:
                    # 抽出這兩個距離下的特徵值
                    x0 = sub.loc[sub["distance_cm"] == d0, feat].dropna().values
                    x1 = sub.loc[sub["distance_cm"] == d1, feat].dropna().values
                    
                    # 樣本不足就跳過
                    if len(x0) < min_samples or len(x1) < min_samples:
                        continue
                    # 算不同距離之間的 Cohen’s d
                    d_val = _cohen_d(x1, x0)
                    if pd.notna(d_val):
                        pair_abs_d_vals.append(abs(d_val))

            # 對 leak level 做 pairwise 差異摘要
            if len(pair_abs_d_vals) > 0:
                row[f"mean_abs_d_dist_{int(lv)}"] = np.mean(pair_abs_d_vals)     # 在這個 leak level 下，不同距離之間平均差多少
                row[f"max_abs_d_dist_{int(lv)}"] = np.max(pair_abs_d_vals)       # 在這個 leak level 下，最極端那組距離差多少
                all_pair_abs_d.extend(pair_abs_d_vals)
            else:
                row[f"mean_abs_d_dist_{int(lv)}"] = np.nan
                row[f"max_abs_d_dist_{int(lv)}"] = np.nan

            # 同一 leak level 內，distance 與 feature 的單調關係(Spearman 相關係數)
            # 同一 leak level 下，距離越遠，這個特徵會不會有系統性地變大或變小？距離越遠，能量越低 / 距離越遠，某個比值越高
            try:
                if sub["distance_cm"].nunique() >= 2 and sub[feat].notna().sum() >= min_samples:
                    rho, _ = spearmanr(sub["distance_cm"], sub[feat], nan_policy="omit")
                else:
                    rho = np.nan
            except Exception:
                rho = np.nan
            # 記錄：每個 leak level 的 distance-feature 單調關係
            row[f"rho_distance_{int(lv)}"] = rho
            if pd.notna(rho):
                all_rho.append(abs(rho))

            # 同一 leak level 內，不同距離整體分布是否有差
            # pairwise d 是兩兩比較,Kruskal 是一次看所有距離組整體有沒有差
            try:
                groups = [
                    sub.loc[sub["distance_cm"] == d, feat].dropna().values
                    for d in dist_levels
                ]
                groups = [g for g in groups if len(g) >= min_samples]

                if len(groups) >= 2:
                    kw_stat, kw_p = kruskal(*groups)
                else:
                    kw_stat, kw_p = np.nan, np.nan
            except Exception:
                kw_stat, kw_p = np.nan, np.nan
            
            # 記錄 Kruskal 結果
            row[f"kw_stat_dist_{int(lv)}"] = kw_stat
            row[f"kw_p_dist_{int(lv)}"] = kw_p
            if pd.notna(kw_stat):
                all_kw.append(kw_stat)

        row["distance_pair_mean_abs_d"] = np.mean(all_pair_abs_d) if len(all_pair_abs_d) > 0 else np.nan       # 平均而言，不同距離 pairwise 差多少
        row["distance_pair_max_abs_d"] = np.max(all_pair_abs_d) if len(all_pair_abs_d) > 0 else np.nan         # 最極端的距離 pair 差多少
        row["distance_rho_mean"] = np.mean(all_rho) if len(all_rho) > 0 else np.nan                            # 平均單調距離趨勢有多強
        row["distance_kw_mean"] = np.mean(all_kw) if len(all_kw) > 0 else np.nan                               # 平均整體距離組差異有多大

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        print("⚠️ 沒有足夠資料做 fixed-leak / cross-distance 特徵比較")
        return out

    # 正規化 Kruskal 統計量
    kw_max = out["distance_kw_mean"].replace([np.inf, -np.inf], np.nan).max()
    if pd.notna(kw_max) and kw_max > 0:
        out["distance_kw_mean_norm"] = out["distance_kw_mean"] / kw_max
    else:
        out["distance_kw_mean_norm"] = 0.0

    # 算最終距離敏感度分數
    # 分數越高 = 越距離敏感
    out["distance_sensitivity_score"] = (
        0.45 * out["distance_pair_mean_abs_d"].fillna(0) +      # 最看重平均 pairwise 差異
        0.25 * out["distance_pair_max_abs_d"].fillna(0) +       # 也在意最極端的距離差異
        0.15 * out["distance_rho_mean"].fillna(0) +             # 看是否有距離單調趨勢
        0.15 * out["distance_kw_mean_norm"].fillna(0)           # 看整體距離組分布差異
    )

    out = out.sort_values("distance_sensitivity_score", ascending=False).reset_index(drop=True)

    print("\n=== Distance-sensitive features within each leak level ===")
    print(
        out[[
            "feature",
            "distance_sensitivity_score",
            "distance_pair_mean_abs_d",
            "distance_pair_max_abs_d",
            "distance_rho_mean",
            "distance_kw_mean",
        ]].head(20).to_string(index=False)
    )

    return out


# ============================================================
# 7. 去除高共線性特徵-相關係數
# ============================================================
def remove_collinear_features(
        X,                     # 每欄一個特徵
        cohen_dict,            # 每個特徵的區分力分數，通常是 Cohen's d
        threshold=0.9,         # 相關係數超過多少算「太像」
        method='pearson',      # 相關係數算法，預設 pearson
        verbose=True, 
        save_path=None):
    """
    迭代式去除高共線性特徵把彼此太像的特徵刪掉，只留下比較有代表性的那個，避免模型吃到很多重複資訊：
    每輪重新計算相關矩陣，只刪一個最該刪的特徵，避免 pairwise 一次 to_drop 產生矛盾。
    """
    
    # 輸出基本資訊：原始特徵數、相關閥值、使用哪種相關係數
    if verbose:
        print("\n" + "=" * 60)
        print("去除高共線性特徵（迭代式，保留 Cohen's d 高者）")
        print("=" * 60)
        print(f"原始特徵數: {X.shape[1]}")
        print(f"相關閾值: {threshold}")
        print(f"計算方法: {method}")

    # 只保留數值欄位
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_work = X[numeric_cols].copy()

    # 如果沒有數值特徵就直接結束
    if len(numeric_cols) == 0:
        print("⚠️ 沒有數值特徵")
        return X, [], []

    dropped_features = []
    corr_pairs_history = []

    # 開始迭代刪除：每次只刪一個最該刪的特徵，刪完再重算一次相關矩陣，直到沒有任何高相關配對為止
    while True:
        # 計算相關矩陣(只看上三角)
        corr = X_work.corr(method=method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        # 找出所有高相關配對：找出所有 相關係數 > threshold 的特徵對
        pairs = []
        for col in upper.columns:
            high_corr_indices = upper[col][upper[col] > threshold].index.tolist()
            for idx in high_corr_indices:
                pairs.append((col, idx, upper.loc[idx, col]))
        # 如果沒有任何高相關配對，就停止
        if len(pairs) == 0:
            break

        corr_pairs_history.extend(pairs)    #  記錄這一輪看到的高相關配對

        # 統計每個特徵在高相關衝突中出現幾次，並偏向刪除 Cohen's d 較低者
        penalty = {}
        for a, b, corr_val in pairs:
            da = cohen_dict.get(a, 0.0)
            db = cohen_dict.get(b, 0.0)

            # 優先刪 Cohen's d 較低者
            if da < db:
                drop = a
            elif db < da:
                drop = b
            else:
                # Cohen's d 相同時，刪掉平均相關更高者；再相同就刪字典序後者
                mean_corr_a = corr[a].drop(labels=[a], errors="ignore").mean()
                mean_corr_b = corr[b].drop(labels=[b], errors="ignore").mean()
                if mean_corr_a > mean_corr_b:
                    drop = a
                elif mean_corr_b > mean_corr_a:
                    drop = b
                else:    # 如果還是一樣，就刪字典序後者
                    drop = sorted([a, b])[1]

            # 累積每個特徵的 penalty( penalty：「這個特徵在這一輪裡，被判定應該刪掉的次數」)
            penalty[drop] = penalty.get(drop, 0) + 1

        # 刪除衝突最多、且 Cohen's d 較低的那個
        worst = sorted(
            penalty.keys(),
            key=lambda feat: (-penalty[feat], cohen_dict.get(feat, 0.0), feat)
        )[0]

        # 刪除特徵(一輪只刪一個)
        X_work = X_work.drop(columns=[worst])
        dropped_features.append(worst)
        
    # 存放刪除的特徵
    X_reduced = X.drop(columns=[c for c in dropped_features if c in X.columns])
    
    # 輸出刪除結果：共刪了幾個、刪了哪些、特徵數從多少遍多少
    if verbose:
        print(f"\n移除 {len(dropped_features)} 個特徵:")
        for feat in dropped_features[:20]:
            print(f"  - {feat}")
        if len(dropped_features) > 20:
            print(f"  ... 還有 {len(dropped_features) - 20} 個")
        print(f"\n✅ 特徵數: {X.shape[1]} → {X_reduced.shape[1]}")

    # 輸出報表
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 高相關特徵對清單
        corr_pairs_file = save_path / f'corr_pairs_{timestamp}.csv'
        pd.DataFrame(
            corr_pairs_history,
            columns=['feature1', 'feature2', 'correlation']
        ).drop_duplicates().to_csv(corr_pairs_file, index=False, encoding='utf-8-sig')

        # 被刪除的特徵列表
        dropped_file = save_path / f'corr_dropped_features_{timestamp}.txt'
        with open(dropped_file, 'w', encoding='utf-8') as f:
            f.write("移除的高共線性特徵列表\n")
            f.write("=" * 50 + "\n")
            for feat in dropped_features:
                f.write(f"{feat}\n")
        
        # 保留的特徵列表
        retained_file = save_path / f'corr_retained_features_{timestamp}.txt'
        with open(retained_file, 'w', encoding='utf-8') as f:
            f.write("保留特徵列表\n")
            f.write("=" * 50 + "\n")
            for feat in X_reduced.columns:
                f.write(f"{feat}\n")

    return X_reduced, dropped_features, corr_pairs_history


# ============================================================
# 利用 VIF 進一步去除多重共線性(本專案不適合使用VIF)
# ============================================================
def remove_high_vif_features(X_df, cohen_dict, vif_threshold=10.0, verbose=True):
    """
    用 VIF 迭代移除多重共線性特徵。
    每輪找出 VIF 最高且超過閾值的特徵，移除後重新計算，直到所有特徵 VIF 都在閾值內。
    當有多個特徵 VIF 超標時，優先移除 Cohen's d 最低的那個。
    
    Parameters
    ----------
    X_df        : DataFrame，只含數值特徵（不含 label/meta 欄位）
    cohen_dict  : dict，每個特徵的 Cohen's d 分數
    vif_threshold : float，VIF 超過此值視為多重共線性（常用 5 或 10）
    verbose     : bool
    
    Returns
    -------
    X_reduced   : DataFrame
    dropped     : list of str，被移除的特徵
    vif_history : list of DataFrame，每輪的 VIF 表（方便 debug）
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X_work = X_df.select_dtypes(include=[np.number]).copy()
    dropped = []
    vif_history = []
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"VIF 共線性篩選（threshold={vif_threshold}）")
        print(f"起始特徵數: {X_work.shape[1]}")
        print("=" * 60)
    
    round_idx = 0
    while True:
        round_idx += 1
        cols = X_work.columns.tolist()
        
        # 計算每個特徵的 VIF
        vif_vals = []
        for i, col in enumerate(cols):
            try:
                v = variance_inflation_factor(X_work.values, i)
            except Exception:
                v = float('inf')   # 數值問題時當成無限大
            vif_vals.append({"feature": col, "VIF": v,
                             "cohen_d": cohen_dict.get(col, 0)})
        
        vif_df = pd.DataFrame(vif_vals).sort_values("VIF", ascending=False)
        vif_history.append(vif_df.copy())
        
        if verbose:
            print(f"\n--- Round {round_idx} ---")
            print(vif_df.head(10).to_string(index=False))
        
        # 找出所有超標特徵
        over_threshold = vif_df[vif_df["VIF"] > vif_threshold]
        if len(over_threshold) == 0:
            if verbose:
                print(f"\n✅ 所有特徵 VIF ≤ {vif_threshold}，停止")
            break
        
        # 在超標特徵中，移除 Cohen's d 最低的那個
        worst = over_threshold.sort_values("cohen_d").iloc[0]["feature"]
        X_work = X_work.drop(columns=[worst])
        dropped.append(worst)
        
        if verbose:
            print(f"  → 移除: {worst}  "
                  f"(VIF={over_threshold[over_threshold.feature==worst].VIF.values[0]:.1f}, "
                  f"Cohen's d={cohen_dict.get(worst, 0):.3f})")
    
    if verbose:
        print(f"\n移除特徵數: {len(dropped)}")
        print(f"最終特徵數: {X_work.shape[1]}")
    
    return X_work, dropped, vif_history


# ============================================================
# 用 RandomForest 對候選特徵算模型重要性
# ============================================================
def compute_model_feature_importance(
    df_dev,               # 開發集(dev set)資料,裡面包含所有候選特徵的欄位以及 label 欄位
    candidate_features,   # 一份特徵名單——通常是統計排名前 K 名
    random_state=42,
):
    """
    用 RandomForest 對候選特徵做重要性評分(window-level)。
    跟 Step2 統計排名(Cohen's d / rho / KW)是完全不同角度：
    這裡是「訓練一個分類器,看它自己覺得哪些特徵重要」。
    """
    X = df_dev[candidate_features].values     # 從 df_dev 裡把候選特徵的數值取出
    y = df_dev["label"].values                # 把 label 欄位取出

    # 對 X 做標準化(每個特徵減平均值除標準差)-嚴格來說 RF 不需要做標準化
    # 這個 scaler 是只在這個函式內部暫時用一下,算完重要性就丟掉了,不會存下來、也不會影響最終喂進 XGB 的特徵值
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 模型訓練
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_scaled, y)

    # clf.feature_importances_ 是 RandomForest 訓練完自帶的屬性,對每個特徵給出一個重要性分數
    return dict(zip(candidate_features, clf.feature_importances_))


# ============================================================
# 合併統計分數與模型重要性
# ============================================================
def combine_statistical_and_model_scores(
    statistical_score_dict,    # Step2 原本的統計綜合分數
    model_importance_dict,     # 上面那個函式的輸出
    stat_weight=0.5,           # 上面那個函式的輸出
    model_weight=0.5,          # 上面那個函式的輸出
):
    """
    把統計分數(final_score)跟模型重要性(feature_importances_)
    各自做 min-max normalize,再加權合併成一個綜合分數。
    這個綜合分數之後拿去做共線性篩選,決定最終保留哪些特徵。
    """
    # 以 statistical_score_dict 的 key 順序為準,建立一份特徵名稱列表
    features = list(statistical_score_dict.keys())
    # 分別把兩份字典裡對應這些特徵的數值,依序取出組成兩個 numpy array:stat_vals(統計分數)、model_vals(模型重要性)
    stat_vals = np.array([statistical_score_dict.get(f, 0.0) for f in features])
    model_vals = np.array([model_importance_dict.get(f, 0.0) for f in features])

    # 一組數值線性縮放到 [0, 1] 區間, 因為統計分數 跟 RandomForest importance 這兩者的數值尺度完全不同,不能直接相加比較
    def _minmax(x):
        lo, hi = np.min(x), np.max(x)
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    # 分別對統計分數、模型重要性做 normalize,得到 stat_norm、model_norm(都是 [0,1] 區間)
    stat_norm = _minmax(stat_vals)
    model_norm = _minmax(model_vals)
    combined = stat_weight * stat_norm + model_weight * model_norm    # 權重做加權平均,得到 combined

    return (
        dict(zip(features, combined)),      # 合併後的綜合分數(combined_score_dict,之後拿去做共線性篩選、決定最終保留哪些特徵)
        dict(zip(features, stat_norm)),     # 標準化後的統計分數
        dict(zip(features, model_norm)),    # 標準化後的模型重要性
    )


# ============================================================
# 原本的特徵挑選流程有沒有資料洩漏，導致 CV 分數過度樂觀
# ============================================================
# 嚴格版內部驗證工具：檢查你原本的 CV 有沒有因為 feature selection 放在外面而過度樂觀
def run_group_cv_with_foldwise_feature_selection(
    df_dev,
    record_meta_dev_df,
    target_levels=tuple(ANALYSIS_LEVELS),
    top_k_candidates=50,
    corr_threshold=0.95,
    random_state=42,
    banned_features=None,
):
    """
    先在全資料上選特徵，再做 CV，會不會偷看到 validation fold 的資訊？
    流程是：
    1.用 StratifiedGroupKFold
        stratified：盡量維持 label 分布
        grouped：同一個 record 不會被切到 train/val 兩邊
    2.對每個 fold：
        切出 df_train / df_val
        找出 train fold 裡有哪些 record_id
    3.用 train fold 的 window 資料聚合成 record-level 資料
        build_record_level_feature_table(...)
    4.只在 train fold 上做 feature ranking
        rank_features_for_leak_levels(...)
    5.取前 top_k_candidates 個候選特徵
    6.只在 train fold 上去共線性
        remove_collinear_features(...)
    7.得到這一 fold 真正保留的特徵後：
        train 用這組特徵訓練
        val 也只用這組特徵評估
    8.算每 fold 的 F1
    9.最後再統計：
        各 fold F1 的平均/標準差
        哪些特徵在多個 fold 都被選到
    """
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score
    
    if banned_features is None:
        banned_features = []

    # groups_all：每個 window 對應到哪個原始 record
    groups_all = df_dev["record_id"].values
    # y_all：每個 window 的 label
    y_all = df_dev["label"].values

    # 最多切 5 folds，但不能超過 group 數量
    unique_groups = np.unique(groups_all)
    n_splits = min(5, len(unique_groups))

    # 建立 grouped + stratified 的 CV 切法
    gkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_scores = []    # 存每個 fold 的 F1
    fold_selected_features = []    # 存每個 fold 最後選到哪些特徵

    # 這個 dummy X 只是給 gkf.split() 用, 因為切 fold 真正靠的是 y_all 和 groups_all
    X_dummy = np.zeros((len(df_dev), 1))

    # 開始逐 fold 做訓練與驗證
    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X_dummy, y_all, groups_all), start=1
    ):
        # train fold 的 windows
        df_train = df_dev.iloc[train_idx].copy()
        # validation fold 的 windows
        df_val = df_dev.iloc[val_idx].copy()

        # train fold 裡有哪些 record
        train_record_ids = set(df_train["record_id"].unique())

        # 1. 先把 train fold 的 window-level 特徵聚合成 record-level
        record_meta_train_df = record_meta_dev_df[
            record_meta_dev_df["record_id"].isin(train_record_ids)
        ].copy()

        record_df_train = build_record_level_feature_table(
            window_df=df_train,
            qc_record_df=None,
            extra_meta_df=record_meta_train_df,
            agg="median",
        )
        
        # 如果 merge 後產生 label_x / label_y，就整理回單一 label
        if "label_x" in record_df_train.columns:
            record_df_train["label"] = record_df_train["label_x"]
            drop_tmp_cols = [c for c in ["label_x", "label_y"] if c in record_df_train.columns]
            record_df_train = record_df_train.drop(columns=drop_tmp_cols)

        # 只保留 0 / 30 leak level
        record_df_train_levels = record_df_train[
            record_df_train["leak_quantity"].isin(target_levels)
        ].copy()

        # 2. 只在 train fold 上做特徵 ranking
        rank_df_fold = rank_features_for_leak_levels(
            record_df_train_levels,
            target_levels=target_levels,
        )
        
        # 先取前 top_k_candidates 個候選特徵
        candidate_features = rank_df_fold["feature"].head(top_k_candidates).tolist()
        candidate_features = [
            f for f in candidate_features
            if f not in banned_features
        ]
        
        if len(candidate_features) == 0:
            print(f"Fold {fold_idx}: 沒有可用候選特徵，跳過")
            continue

        # 3. 只在 train fold 上做去共線性
        X_train_candidate_df = df_train[candidate_features].copy()
        rank_score_dict = dict(
            zip(rank_df_fold["feature"], rank_df_fold["score"])
        )

        X_train_reduced, _, _ = remove_collinear_features(
            X_train_candidate_df,
            rank_score_dict,
            threshold=corr_threshold,
            method="pearson",
            verbose=False,
            save_path=None
        )
        
        # 這個 fold 最後真的保留下來的特徵
        selected_features = [
            f for f in X_train_reduced.columns.tolist()
            if f not in banned_features
        ]
        
        if len(selected_features) == 0:
            print(f"Fold {fold_idx}: 沒有可用最終特徵，跳過")
            continue

        fold_selected_features.append({
            "fold": fold_idx,
            "n_features": len(selected_features),
            "features": selected_features,
        })

        # 4. train / validation 都用同一組 selected features
        X_train = df_train[selected_features].values
        X_val = df_val[selected_features].values
        y_train = df_train["label"].values
        y_val = df_val["label"].values

        # 建立模型 pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state))
        ])

        # 用 train fold 訓練
        pipeline.fit(X_train, y_train)
        # 在 validation fold 預測
        y_val_pred = pipeline.predict(X_val)

        # 計算這個 fold 的 F1
        fold_f1 = f1_score(y_val, y_val_pred)
        fold_scores.append(fold_f1)

        print(f"Fold {fold_idx}: F1 = {fold_f1:.4f}, selected_features = {len(selected_features)}")

    # 所有 fold 的 F1 分數整理成 array
    fold_scores = np.array(fold_scores)

    print("\n" + "=" * 60)
    print("Fold-wise Feature Selection Group CV")
    print("=" * 60)
    print(f"F1 mean = {fold_scores.mean():.4f}")
    print(f"F1 std  = {fold_scores.std():.4f}")
    
    # 統計每個特徵在幾個 fold 被選到
    feature_freq = {}
    for item in fold_selected_features:
        for feat in item["features"]:
            feature_freq[feat] = feature_freq.get(feat, 0) + 1
    
    # 整理成 DataFrame，方便輸出與排序
    feature_freq_df = (
        pd.DataFrame([
            {"feature": feat, "n_folds_selected": n}
            for feat, n in feature_freq.items()
        ])
        .sort_values(["n_folds_selected", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    
    print("\nTop stable features across folds:")
    print(feature_freq_df.head(20).to_string(index=False))

    return fold_scores, fold_selected_features, feature_freq_df


# ============================================================
# 模型完全沒看過某個距離，它還能不能抓到 leak
# ============================================================
def run_leave_one_distance_out_validation(
    df_dev,           # 開發資料集，至少要有 record_id、label、distance_cm
    feature_names,    # 要拿來訓練模型的特徵欄位清單
    base_pipeline,    # 要用的模型流程，例如 scaler + classifier
    threshold=0.5,    # 把機率轉成 0/1 預測時使用的門檻
):
    """
    Leave-one-distance-out validation
    每次留一個 distance_cm 當 test，其餘 distance 當 train
    例如現在驗 20 cm：
        train_df：df_dev 裡除了 20 cm 以外的全部資料
        test_df：df_dev 裡只有 20 cm 的資料
    """
    # 檢查 df_dev 必要欄位
    required_cols = {"record_id", "label", "distance_cm"}
    missing = required_cols - set(df_dev.columns)
    if missing:
        raise ValueError(f"df_dev 缺少欄位: {sorted(missing)}")

    # 確認 feature_names 真的存在 df_dev, 如果一個都不存在，就直接停止
    feature_names = [c for c in feature_names if c in df_dev.columns]
    if len(feature_names) == 0:
        raise ValueError("feature_names 沒有任何欄位存在於 df_dev")

    # 先找出有哪些距離
    distances = sorted(df_dev["distance_cm"].dropna().unique())
    
    # 建立三個容器
    summary_rows = []        # 存每個 held-out distance 的總結指標
    window_pred_rows = []    # 存每個 window 的預測結果
    file_pred_rows = []      # 存每個 record/file 聚合後的預測結果

    print("\n" + "=" * 60)
    print("Leave-One-Distance-Out Validation")
    print("=" * 60)
    print(f"distance_cm: {distances}")

    # 主迴圈：每次留一個距離當 test(train_df：除了 dist 以外的所有距離, test_df：只有 dist 這個距離)
    for dist in distances:
        train_df = df_dev[df_dev["distance_cm"] != dist].copy()
        test_df = df_dev[df_dev["distance_cm"] == dist].copy()

        # 跳過不合法的 split
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"\n[Skip] distance={dist}: 空 split")
            continue

        if train_df["label"].nunique() < 2:
            print(f"\n[Skip] distance={dist}: train 只有單一類別")
            continue

        if test_df["label"].nunique() < 2:
            print(f"\n[Warn] distance={dist}: test 只有單一類別，AUC 會是 NaN")

        # 切出 X/y(把特徵和標籤分開，準備丟進模型)
        X_train = train_df[feature_names].values
        y_train = train_df["label"].values
        X_test = test_df[feature_names].values
        y_test = test_df["label"].values

        # 訓練模型
        model = clone(base_pipeline)    # clone(base_pipeline) 的意思是：每次測一個距離時，都重新複製一個乾淨的模型，避免前一次訓練留下狀態。
        model.fit(X_train, y_train)

        # 做預測
        y_prob = model.predict_proba(X_test)[:, 1]     # y_prob：模型預測是正類的機率
        y_pred = (y_prob >= threshold).astype(int)     # y_pred：依照門檻 threshold 轉成 0/1 類別

        # 計算 window-level 指標:window-level 細粒度片段判斷能力
        w_acc = accuracy_score(y_test, y_pred)
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        w_auc = _safe_auc(y_test, y_prob)

        # 印出 window-level 結果
        print("\n" + "-" * 60)
        print(f"Held-out distance = {int(dist)} cm")
        print("-" * 60)
        print(
            f"Train windows={len(train_df)}, "
            f"Train records={train_df['record_id'].nunique()}, "
            f"Positive rate={train_df['label'].mean():.3f}"
        )
        print(
            f"Test  windows={len(test_df)}, "
            f"Test  records={test_df['record_id'].nunique()}, "
            f"Positive rate={test_df['label'].mean():.3f}"
        )
        print(
            f"Window-level: acc={w_acc:.3f}, "
            f"precision={w_prec:.3f}, recall={w_rec:.3f}, "
            f"f1={w_f1:.3f}, auc={w_auc:.3f}"
        )
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))

        # 存每個 window 的預測結果
        tmp_win = test_df[["record_id", "label", "distance_cm"]].copy()
        tmp_win["y_prob"] = y_prob
        tmp_win["y_pred"] = y_pred
        tmp_win["held_out_distance_cm"] = dist
        window_pred_rows.append(tmp_win)

        # record/file 層級評估:把 window-level 聚合成 file-level
        file_df = (
            tmp_win.groupby("record_id", as_index=False)
            .agg(
                label=("label", "first"),
                distance_cm=("distance_cm", "first"),
                y_prob=("y_prob", "mean"),
            )
            .copy()
        )
        file_df["y_pred"] = (file_df["y_prob"] >= threshold).astype(int)
        file_df["held_out_distance_cm"] = dist
        file_pred_rows.append(file_df)

        # 計算 file-level 指標:file-level 整份檔案判斷能力
        fy = file_df["label"].values
        fy_pred = file_df["y_pred"].values
        fy_prob = file_df["y_prob"].values

        f_acc = accuracy_score(fy, fy_pred)
        f_prec, f_rec, f_f1, _ = precision_recall_fscore_support(
            fy, fy_pred, average="binary", zero_division=0
        )
        f_auc = _safe_auc(fy, fy_prob)

        print("File-level")
        print(
            f"acc={f_acc:.3f}, precision={f_prec:.3f}, "
            f"recall={f_rec:.3f}, f1={f_f1:.3f}, auc={f_auc:.3f}"
        )
        print(classification_report(fy, fy_pred, digits=3, zero_division=0))

        # 把這個距離的總結存起來
        summary_rows.append({
            "distance_cm": dist,
            "n_train_windows": len(train_df),
            "n_test_windows": len(test_df),
            "n_train_records": train_df["record_id"].nunique(),
            "n_test_records": test_df["record_id"].nunique(),
            "train_positive_rate": train_df["label"].mean(),
            "test_positive_rate": test_df["label"].mean(),
            "window_accuracy": w_acc,
            "window_precision": w_prec,
            "window_recall": w_rec,
            "window_f1": w_f1,
            "window_auc": w_auc,
            "file_accuracy": f_acc,
            "file_precision": f_prec,
            "file_recall": f_rec,
            "file_f1": f_f1,
            "file_auc": f_auc,
        })

    # 整合所有距離的結果
    summary_df = pd.DataFrame(summary_rows)              # 每個 distance 的總表
    window_pred_df = (                                   # 所有 window 預測明細
        pd.concat(window_pred_rows, ignore_index=True)
        if len(window_pred_rows) > 0 else pd.DataFrame()
    )
    file_pred_df = (                                     # 所有 file 預測明細
        pd.concat(file_pred_rows, ignore_index=True)
        if len(file_pred_rows) > 0 else pd.DataFrame()
    )

    # 印出最後總結
    if not summary_df.empty:
        print("\n" + "=" * 60)
        print("LODO Summary")
        print("=" * 60)
        print(
            summary_df[
                [
                    "distance_cm",
                    "n_test_windows",
                    "n_test_records",
                    "window_f1",
                    "window_auc",
                    "file_f1",
                    "file_auc",
                ]
            ].to_string(index=False)
        )
        print(f"\nWindow F1 mean = {summary_df['window_f1'].mean():.4f}")
        print(f"File   F1 mean = {summary_df['file_f1'].mean():.4f}")

    return summary_df, window_pred_df, file_pred_df


# ============================================================
# CV 分數是不是只是運氣好，還是真的比亂猜強
# ============================================================
# 顯著性檢查工具：permutation test -> 打亂 dev label，再做 GroupKFold CV, 檢查你的 CV 分數是不是明顯高於隨機亂數
def run_permutation_test_group_cv(
    X,
    y,
    groups,
    pipeline,
    cv,
    scoring="f1",
    n_iter=100,
    random_state=42,
):
    """
    CV 分數高，是因為模型真的有學到訊號，還是資料結構剛好讓模型看起來很厲害？
    group-level permutation :
    每次 permutation：
    - 把 group 對應的 label 打亂
    - 再映回每個 window
    - 用打亂後的標籤重新做一次 cross_val_score
    重複 n_iter 次
    最後看這些亂數分數的平均、標準差、最大值
    """
    # 建立隨機數產生器
    rng = np.random.default_rng(random_state)
    perm_scores = []    # 存每次 permutation 的 CV 平均分數

    # 轉成 numpy array，避免後面索引不一致
    groups = np.asarray(groups)
    y = np.asarray(y)

    # 每個 group 只取一個 label,因為同一個 record 底下所有 windows 的 label 都應該相同
    group_df = pd.DataFrame({
        "group": groups,
        "y": y,
    }).drop_duplicates(subset=["group"])

    # 取出所有 group id 與對應 label
    group_ids = group_df["group"].to_numpy()
    group_labels = group_df["y"].to_numpy()

    # 重複做 n_iter 次 permutation
    for _ in range(n_iter):
        # 在 group 層級打亂 label
        perm_group_labels = rng.permutation(group_labels)
        # 建立 group -> permuted label 的對照表
        group_label_map = dict(zip(group_ids, perm_group_labels))

        # # 再把 permuted group label 映回每個 window
        y_perm = np.array([group_label_map[g] for g in groups])

        # 在這組亂數標籤下重新做 grouped CV
        scores_perm = cross_val_score(
            pipeline,
            X,
            y_perm,
            groups=groups,
            cv=cv,
            scoring=scoring
        )
        
        # 記錄這次 permutation 的平均分數
        perm_scores.append(scores_perm.mean())

    # 轉成 numpy array，方便後續算 mean/std/max
    perm_scores = np.array(perm_scores)

    print("\n" + "=" * 60)
    print("Permutation Test (Group-level CV)")
    print("=" * 60)
    print(f"n_iter = {n_iter}")
    print(f"perm mean = {perm_scores.mean():.4f}")
    print(f"perm std  = {perm_scores.std():.4f}")
    print(f"perm max  = {perm_scores.max():.4f}")

    return perm_scores


# ============================================================
# 模型在不同條件下有沒有表現一致
# ============================================================
# 依照某個欄位分組(distance_cm or leak_quantity)，輸出 holdout classification report
def evaluate_holdout_by_group(
    df_holdout,
    y_true,
    y_pred,
    group_col,
):
    '''
    檢查模型在 holdout set 裡，不同 distance、leak_quantity 下是不是表現一致
     - 按 distance_cm 分組
     - 按 leak_quantity 分組
    '''
    # 如果指定的分組欄位不存在，就直接提醒並結束
    if group_col not in df_holdout.columns:
        print(f"⚠️ df_holdout 沒有欄位: {group_col}")
        return

    # 建立評估用 DataFrame，只保留分組欄位
    eval_df = df_holdout[[group_col]].copy()
    # 把真實標籤與預測標籤補回來
    eval_df["y_true"] = y_true
    eval_df["y_pred"] = y_pred

    print("\n" + "=" * 60)
    print(f"Holdout evaluation by {group_col}")
    print("=" * 60)

    # 對每個 group 個別輸出 classification report
    for group_value, sub in eval_df.groupby(group_col):
        print(f"\n--- {group_col} = {group_value} ---")
        print(f"n = {len(sub)}")
        print(classification_report(sub["y_true"], sub["y_pred"], digits=3))


# ============================================================
# 輸出最終 feature config JSON
# 給後續 transition / disturbance / runtime inference 直接讀取
# ============================================================
def save_feature_config_json(
    output_json,    # 輸出路徑
    selected_feature_names,     # 核心的特徵名單
    window_sec,       # 每個窗多長
    hop_sec,          # 每次滑動多少
    fs,               # 取樣率
    nperseg,          # Welch PSD 的參數
    max_points,       # 每筆檔最多讀多少點
    target_levels,    # 這次比較了哪些漏量
    top_k_candidates, # 先取前幾名候選特徵
    corr_threshold,   # 去共線性時的門檻
    temporal_band_names,   # 基礎提取器需要啟用 temporal raw/base 特徵的頻帶名稱
    
    baseline_feature_names,
    cusum_input_feature_names,
    temporal_base_feature_names,
    
    # 正式上游特徵契約
    band_config_path,
    band_config_hash,
    feature_extraction_config_path,
    feature_extraction_config_hash,
    baseline_path,
    baseline_payload,
    
    # 正式推論會用到的設定
    source_note="step2_micro_leak_feature_selection",
    score_col="leak_score",
    pred_col="pred_label",
    threshold=0.5,            # 分數超過多少算 leak
    min_consecutive_windows=3,     # 至少連續幾個窗才算真的事件
    min_leak_duration_sec=0.4,     # 至少持續多久
    disturbance_ratio_col="offband_to_leak_focus_E_ratio",
    disturbance_ratio_threshold=None,
    
    # 固定正常 baseline
    baseline_mode="fixed_normal_baseline",
    include_offband_summary=True,
    
    # CUSUM 類特徵怎麼算
    cusum_k=0.02,   
    cusum_decay=0.97,
):
    '''最後選出來的特徵與正式推論會用到的參數，整理成一份 feature_config.json'''
    payload = {
        # 來源與特徵名單
        "source": source_note,                                     # 標記這份設定檔是哪個流程產生的
        "selected_feature_names": list(selected_feature_names),    # 最終保留的特徵名單
        
        "schema_name": "feature_config_micro",                     # 標記這份檔案的種類
        "schema_version": 2,                                       # 標記這份檔案的版本
        
        # band config 與特徵萃取設定的契約資訊
        "band_config_path": str(Path(band_config_path)),           # 記錄這次用的 band config 檔案路徑（轉成字串）
        "band_config_sha256": band_config_hash,                    # band config 檔案指紋
        
        "feature_extraction_config_path": str(Path(feature_extraction_config_path)),      # 記錄特徵萃取參數設定檔的路徑
        "feature_extraction_config_sha256": (feature_extraction_config_hash),             # 記錄特徵萃取參數設定檔的指紋
        
        # Baseline 相關資訊
        "baseline_mode": baseline_mode,                                                    # baseline 模式
        "baseline_path": str(Path(baseline_path)),                                         # 固定 baseline JSON 檔案的路徑
        "baseline_schema_name": baseline_payload.get("schema_name"),                       # 記錄這次用的 baseline 是不是正式定案版本
        "baseline_schema_status": baseline_payload.get("schema_status"),                   # 記錄這次用的 baseline 是不是正式定案版本
        "baseline_environment": baseline_payload.get("environment"),                       # 標記這份 baseline 是在哪個場域/測試環境
        "baseline_config_fingerprint": baseline_payload.get("config_fingerprint"),         # baseline 建立時,把更多相關設定打包壓縮出的一個綜合指紋
        "baseline_base_feature_version": baseline_payload.get("base_feature_version"),     # 記錄 baseline 建立時用的是哪個版本的特徵計算邏輯

        # 正式特徵工程要共用的基本參數
        "window_sec": window_sec,    
        "hop_sec": hop_sec,
        "fs": fs,
        "nperseg": nperseg,
        "max_points": max_points,

        # Step2 特徵篩選過程資訊
        "target_levels": sorted(list(target_levels)),                 # 這次比較用的洩漏等級（例如 {0, 30}）排序後轉成 list 存下來
        "top_k_candidates": top_k_candidates,                         # 記錄這次候選特徵池的大小
        "corr_threshold": corr_threshold,                             # 共線性門檻
        "temporal_band_names": sorted(list(temporal_band_names)),     # 記錄哪些頻段的名稱有被納入 temporal/CUSUM 類衍生特徵的計算範圍

        # runtime / inference 會用到的特徵工程與後處理設定
        "include_offband_summary": include_offband_summary,           # 記錄這次是否有計算 offband 摘要特徵
        "cusum_k": cusum_k,                                           # 記錄下來讓推論端能用完全一致的參數重新計算 CUSUM 特徵
        "cusum_decay": cusum_decay,                                   # 記錄下來讓推論端能用完全一致的參數重新計算 CUSUM 特徵

        "score_col": score_col,                              # 模型輸出分數
        "pred_col": pred_col,                                # 模型預測標籤
        "threshold": threshold,                              # 分類門檻
        "min_consecutive_windows": min_consecutive_windows,  # 至少要連續幾個視窗都判定為洩漏
        "min_leak_duration_sec": min_leak_duration_sec,      # 事件至少要持續多久（秒）
        "disturbance_ratio_col": disturbance_ratio_col,
        "disturbance_ratio_threshold": disturbance_ratio_threshold,
        
        # 特徵執行計畫（execution plan）:把整條特徵工程管線裡「每個階段各自用了哪些欄位」分層記錄下來
        "feature_execution_plan": {"baseline_features": list(baseline_feature_names),   # 套用固定 baseline 產生 rel_*/z_* 的那批原始特徵
        "cusum_input_features": list(cusum_input_feature_names),                        # 對應 z_feature_cols（餵給 add_cusum_features 的那批 z_* 欄位）
        "temporal_base_features": list(temporal_base_feature_names),                    # 餵給 add_step2_temporal_features 的那批基礎欄位
        "final_keep_features": list(selected_feature_names),
        },
    }

    # 輸出成 JSON
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ 已輸出 feature config: {output_json}")





#                                         ********************************************************************
#                                                                           視覺化 
#                                         ********************************************************************
# ==========================
# 繪圖的 helper function:自動設定較穩定的y軸範圍
# ==========================
def _set_robust_ylim(
        ax, 
        series_list, 
        lower_pct=2,      # 全部資料的第 2 百分位
        upper_pct=98,     # 全部資料的第 98 百分位
        pad_ratio=0.08):
    """
    根據多組資料的分位數，自動設定較穩健的 y 軸範圍，避免少數極端值把箱體壓扁。
    """
    # series_list 裡的每一組資料：轉成 Series -> 去掉 NaN -> 轉成 float -> 再去掉 inf / -inf -> 如果這組還有有效資料，就收進 vals
    vals = []
    for s in series_list:
        arr = np.asarray(pd.Series(s).dropna(), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals.append(arr)
    # 如果完全沒資料就直接結束
    if not vals:
        return

    # 把所有組合併
    all_vals = np.concatenate(vals)
    y_low = np.percentile(all_vals, lower_pct)     # 算低分位數
    y_high = np.percentile(all_vals, upper_pct)    # 算高分位數

    # 如果算出來不是有效數字就結束
    if not np.isfinite(y_low) or not np.isfinite(y_high):
        return

    # 如果上下界一樣，特別處理
    if y_low == y_high:
        pad = max(abs(y_low) * pad_ratio, 1e-6)
        ax.set_ylim(y_low - pad, y_high + pad)
        return

    # 一般情況下，計算跨度與 padding
    span = y_high - y_low                       # span 是主要數據區間的高度
    pad = max(span * pad_ratio, 1e-6)           # 再拿 span * pad_ratio 當作上下邊界空間
    ax.set_ylim(y_low - pad, y_high + pad)      # 設定 y 軸範圍

# ============================================================
# 雙通道 ai0, ai1 時頻圖比較(x 軸 = 時間, y 軸 = 頻率)
# ============================================================   
def plot_record_dual_channel_spectrogram(
        record,
        save_path=None,
        nperseg=8192,       # 做 spectrogram 時，每個小時間窗用多少點來做頻譜分析
        noverlap=6144,      # 相鄰兩個小時間窗重疊多少點
        fmin=40_000,
        fmax=300_000,       # 頻率範圍
        cmap="magma",       # 圖配色
        diff_cmap="coolwarm",
        diff_vlim=5,
    ):
    """
    單一 record 的雙通道時頻圖
    x 軸 = time
    y 軸 = frequency
    color = power(dB)
    """
    # 從 record 裡拿出 ai0/ai1 訊號，轉成 numpy array，型別是 float
    ai0 = np.asarray(record["signal"]["ai0"], dtype=float)
    ai1 = np.asarray(record["signal"]["ai1"], dtype=float)
    fs = float(record["fs"])     # 取出採樣率 fs，並轉成浮點數

    # 對 ai0/ai1 做 spectrogram, 回傳f0-頻率軸, t0-時間軸, S0-功率矩陣
    f0, t0, S0 = spectrogram(
        ai0, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="psd"
    )
    f1, t1, S1 = spectrogram(
        ai1, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="psd"
    )

    # 頻率遮罩,保留需要的頻段
    mask0 = (f0 >= fmin) & (f0 <= fmax)
    mask1 = (f1 >= fmin) & (f1 <= fmax)

    # 只抓需要的頻段
    f0 = f0[mask0]
    f1 = f1[mask1]
    S0 = S0[mask0, :]    # 把 ai0,ai1 的 spectrogram 功率矩陣在頻率方向一起裁掉
    S1 = S1[mask1, :]

    # 把 S0,S1,diff 從線性功率轉成 dB 尺度
    S0_db = 10 * np.log10(S0 + 1e-12)
    S1_db = 10 * np.log10(S1 + 1e-12)
    if S0_db.shape != S1_db.shape:
        raise ValueError("ai0 and ai1 spectrogram shapes do not match")
    diff_db = S1_db - S0_db

    # 建立圖畫
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=160, sharex=True)

    im0 = axes[0].pcolormesh(t0, f0, S0_db, shading="gouraud", cmap=cmap, vmin=-120, vmax=-100)
    axes[0].set_title("ai0 (far) Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")
    plt.colorbar(im0, ax=axes[0], label="Power (dB)")

    im1 = axes[1].pcolormesh(t1, f1, S1_db, shading="gouraud", cmap=cmap, vmin=-120, vmax=-100)
    axes[1].set_title("ai1 (near) Spectrogram")
    axes[1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im1, ax=axes[1], label="Power (dB)")
    
    im2 = axes[2].pcolormesh(t1, f1, diff_db, shading="gouraud", cmap=diff_cmap, vmin=-diff_vlim, vmax=diff_vlim)
    axes[2].set_title("ai1 - ai0 Diff Spectrogram")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    plt.colorbar(im2, ax=axes[2], label="Near - Far (dB)")


    fig.suptitle(
        f"{record['filename']}\n"
        f"leak={record['meta']['leak_quantity']} ml, "
        f"distance={record['meta']['distance']*100:.0f} cm, "
        f"pressure={record['meta']['pressure']} bar"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])


    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 已儲存: {save_path}")
    else:
        plt.show()


# ==========================
# 3.1繪圖 ai0 / ai1 / diff(rms,raw signal)
# ==========================
# 繪圖：畫 ai0 / ai1 / diff 三條訊號，同一個檔案在同一張圖裡
def plot_signal_overview(records, num_samples=5, save_dir="base_path/signal_plots"):
    """
    對每個檔案畫一張「多通道 + 能量分析」的總覽圖（overview plot）
    1. 原始訊號（ai0 vs ai1）
    2. rolling RMS（能量變化）
    3. diff 訊號（near - far）
    4. diff 的能量
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for r in records[:num_samples]:          # 逐筆資料處理，每個 record 畫一張圖
        ch0  = r["signal"]["ai0"]            # ai0 → far sensor
        ch1  = r["signal"]["ai1"]            # ai1 → near sensor
        diff = r["signal"]["diff"]           # diff → 空間差異（你 leak detection 很關鍵）
        fs   = r["fs"]
        t    = np.arange(len(ch0)) / fs      # 把 sample index → 秒，把 sample index → 秒

        # 整體 RMS（全域能量）:比較 near / far 強度
        rms0 = np.sqrt(np.mean(ch0 ** 2))
        rms1 = np.sqrt(np.mean(ch1 ** 2))
        channel_note = f"ai0 = far, ai1 = near, RMS ratio(ai1/ai0) = {rms1/(rms0+1e-12):.2f}"

        # 局部能量 Rolling RMS
        window = max(1, int(fs * 0.001))   # window = 1 ms，避免window = 0 → 直接壞掉
        # 局部能量 = 移動平均(x^2) → 再開根號
        rms0_roll = np.sqrt(np.convolve(ch0**2, np.ones(window)/window, mode='same'))
        rms1_roll = np.sqrt(np.convolve(ch1**2, np.ones(window)/window, mode='same'))
        
        # subplot 結構（4層）
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), dpi=150, sharex=True)

        # (1) Raw signal(觀察振幅)
        axes[0].plot(t, ch0, linewidth=0.5, color='steelblue', label=f"ai0 (RMS={rms0:.5f})")
        axes[0].plot(t, ch1, linewidth=0.5, color='orange',    label=f"ai1 (RMS={rms1:.5f})")
        axes[0].set_title("Raw Signal (ai0 vs ai1)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()

        # (2) Rolling RMS:能量誰比較強,leak 是否集中某段時間
        axes[1].plot(t, rms0_roll, linewidth=1, color='steelblue', label="ai0 rolling RMS")
        axes[1].plot(t, rms1_roll, linewidth=1, color='orange',    label="ai1 rolling RMS")
        axes[1].set_title("Rolling RMS（能量趨勢）")
        axes[1].set_ylabel("RMS")
        axes[1].legend()

        # (3) diff signal:空間差異,傳播效應
        axes[2].plot(t, diff, linewidth=0.5, color='green')
        axes[2].set_title("diff = ai1(near) - ai0(far)")
        axes[2].set_ylabel("Amplitude")

        # (4) diff 的能量:差異訊號是否有 structure
        rms_diff_roll = np.sqrt(np.convolve(diff**2, np.ones(window)/window, mode='same'))
        axes[3].plot(t, rms_diff_roll, linewidth=1, color='green')
        axes[3].set_title("diff Rolling RMS")
        axes[3].set_ylabel("RMS")
        axes[3].set_xlabel("Time [s]")

        # 設定標題：把 metadata + signal 整合在同一張圖
        fig.suptitle(
            f"{r['filename']}\n"
            f"label={r['label']}  distance={r['meta']['distance']}m  "
            f"pressure={r['meta']['pressure']}bar  leak={r['meta']['leak_quantity']}ml\n"
            f"{channel_note}",
            fontsize=11, fontweight='bold'
        )

        plt.tight_layout()

        # 儲存圖片(每個record一張圖)
        if save_dir:
            fname = r['filename'].replace(".tdms", "_overview.png")
            plt.savefig(os.path.join(save_dir, fname))
            plt.close(fig)
        else:
            plt.show()

    print("✅ 繪圖完成")


# ============================================================
# 圖 4：band energy summary（boxplot）
# ============================================================
def plot_band_energy_summary(records_leak, 
                             records_normal, 
                             bands, 
                             channel="ai1",
                             label_leak="Baseline+Leak",
                             label_normal="Baseline",
                             save_path=None):
    """
    比較 Leak / Normal 在指定 channel 的 band energy 分布
    channel 可選：
    - ai0
    - ai1
    - diff
    """
    import matplotlib.patches as mpatches
    
    # 建立 X 軸標籤
    band_labels = [f"Band{i}\n{b['f_low']//1000}k~{b['f_high']//1000}k"
                   for i, b in enumerate(bands)]
    
    leak_energies   = []   # leak_energies[i] = 第 i 個 band 下，所有 leak 檔案的 energy
    normal_energies = []   # normal_energies[i] = 第 i 個 band 下，所有 normal 檔案的 energy
    
    # 逐 band 計算 energy,固定看主訊號(ai1)是因為要比較near 通道在不同 band 下，Leak 和 Normal 的群體差異。
    for b in bands:
        # 對每一筆 leak 資料：取出 ai1 -> 做 bandpass filter，只保留這個頻帶 -> 算 mean(filtered**2)，也就是 band energy
        le = [np.mean(bandpass_filter(r["signal"][channel], r["fs"],
                                      b["f_low"], b["f_high"]) ** 2)
              for r in records_leak]
        ne = [np.mean(bandpass_filter(r["signal"][channel], r["fs"],
                                      b["f_low"], b["f_high"]) ** 2)
              for r in records_normal]
        leak_energies.append(le)
        normal_energies.append(ne)

    # 設定 X 軸位置
    n_bands = len(bands)
    x       = np.arange(n_bands)
    width   = 0.35
    # 建立畫布
    fig, ax = plt.subplots(figsize=(max(10, n_bands * 1.4), 5.5), dpi=150)

    # 繪製 boxplot
    ax.boxplot(
        leak_energies,
        positions=x - width / 2,
        widths=width * 0.8,
        patch_artist=True,
        boxprops=dict(facecolor="orange", alpha=0.6),
        medianprops=dict(color="darkorange", linewidth=2),
        whiskerprops=dict(color="orange"),
        capprops=dict(color="orange"),
        flierprops=dict(marker='o', color='orange', alpha=0.25)
    )

    ax.boxplot(
        normal_energies,
        positions=x + width / 2,
        widths=width * 0.8,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="darkblue", linewidth=2),
        whiskerprops=dict(color="steelblue"),
        capprops=dict(color="steelblue"),
        flierprops=dict(marker='o', color='steelblue', alpha=0.25)
    )

    ax.set_yscale('log')    # 設定 y 軸為log scale,避免band energy差異太大
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Band Energy (mean filtered²)")
    ax.set_title(f"Band Energy Summary: {label_leak} vs {label_normal} ({channel})")

    leak_patch   = mpatches.Patch(color="orange",    alpha=0.6, label=f"{label_leak} (n={len(records_leak)})")
    normal_patch = mpatches.Patch(color="steelblue", alpha=0.6, label=f"{label_normal} (n={len(records_normal)})")
    ax.legend(handles=[leak_patch, normal_patch])
    ax.grid(True, alpha=0.3, axis='y')    # 加上格線

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Saved] {save_path}")
    else:
        plt.show()


# ============================================================
# 圖 5：雙通道比值 summary
# ============================================================
def plot_dual_channel_ratio_summary(records_leak, records_normal, bands, save_path=None):
    """
    把 Leak 與 Normal 在「雙通道關係」上的幾個關鍵 ratio 特徵畫成 boxplot，比較兩組分布差異。
    
    三個比值指標的分布：
    1. RMS(ai1) / RMS(ai0)
    2. Band0_E_ai1 / Band0_E_ai0
    3. Band0_E_diff / Band0_E_ai1
    
    雙通道之間的相對關係，在 leak 和 normal 下有沒有明顯不同？
    ai1 相對 ai0 是不是更強？
    diff 相對 ai1 保留得怎麼樣？
    """
    import matplotlib.patches as mpatches

    # 對一整組 records ,逐筆算出三種ratio
    def compute_ratios(records, b0):
        rms_ratios, band_ratios, diff_ratios = [], [], []
        for r in records:   # 逐筆取出ai0,ai1,diff,採樣率
            ai0  = r["signal"]["ai0"]
            ai1  = r["signal"]["ai1"]
            diff = r["signal"]["diff"]
            fs   = r["fs"]
            
            # 近端 / 遠端整體強度比
            rms0 = np.sqrt(np.mean(ai0 ** 2))
            rms1 = np.sqrt(np.mean(ai1 ** 2))
            rms_ratios.append(rms1 / (rms0 + 1e-12))
            
            # 針對 Band0(取決於傳進來的band是哪個區間)：分別把 ai1、ai0、diff 做 bandpass,再算 band energy
            E_ai1 = np.mean(bandpass_filter(ai1,  fs, b0["f_low"], b0["f_high"]) ** 2)
            E_ai0 = np.mean(bandpass_filter(ai0,  fs, b0["f_low"], b0["f_high"]) ** 2)
            E_diff = np.mean(bandpass_filter(diff, fs, b0["f_low"], b0["f_high"]) ** 2)

            band_ratios.append(E_ai1 / (E_ai0  + 1e-12))    # 在 Band0 裡，near 比 far 強多少,更聚焦在「最重要的那個 band」
            diff_ratios.append(E_diff / (E_ai1 + 1e-12))    # 差分訊號在 Band0 中，保留了多少 near 的主能量

        return rms_ratios, band_ratios, diff_ratios

    b0 = bands[0]   # 假設 Band 0 是最重要的，先看他的雙公道關係

    leak_rms,   leak_band,   leak_diff   = compute_ratios(records_leak,   b0)
    normal_rms, normal_band, normal_diff = compute_ratios(records_normal, b0)
    
    # 把三個指標整理成 metrics
    metrics = [
        ("RMS(ai1) / RMS(ai0)",
         leak_rms,   normal_rms),
        (f"Band0 E(ai1) / E(ai0)\n({b0['f_low']//1000}k~{b0['f_high']//1000}kHz)",
         leak_band,  normal_band),
        (f"Band0 E(diff) / E(ai1)\n({b0['f_low']//1000}k~{b0['f_high']//1000}kHz)",
         leak_diff,  normal_diff),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)    # 建立畫布(一列三欄)
    # 畫 boxplot(畫兩個箱型圖，第一個是leak,第二個是Normal)
    for ax, (title, leak_vals, normal_vals) in zip(axes, metrics):
        bp = ax.boxplot(
            [leak_vals, normal_vals],
            patch_artist=True,
            labels=["Baseline+Leak", "Baseline"],
            boxprops=dict(alpha=0.7),
            medianprops=dict(linewidth=2)
        )
        # 手動改顏色
        bp["boxes"][0].set_facecolor("orange")
        bp["boxes"][1].set_facecolor("steelblue")
        bp["medians"][0].set_color("darkorange")
        bp["medians"][1].set_color("darkblue")

        ax.set_yscale('log')   # 設定log scale
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='ratio=1')   # axhline(1.0)：畫一條 ratio=1 的虛線,當作分界點，ratio > 1：分子比較大
        ax.legend(fontsize=7)
        
    # 整張圖的標題
    plt.suptitle("Dual Channel Ratio Summary: Baseline+Leak vs Baseline", fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Saved] {save_path}")
    else:
        plt.show()
        

# ============================================================  
# 放大重要頻段訊號圖
# ============================================================        
def plot_band_zoom_comparison(records_leak, records_normal, band,
                               num_samples=None,    # ← None = 全部
                               alpha=0.2,           # ← 可調透明度
                               save_path=None):
    """
    把 leak 和 normal 的同一個 band 疊在一張圖上做直接比較
     - 挑一個 band
     - 把很多筆 leak 疊一起
     - 把很多筆 normal 疊一起
     - 看這個 band 下，兩組的波形與能量輪廓到底差多少
     
    上圖
     - bandpass 後的 waveform 疊圖
     - 橘色：Leak
     - 藍色：Normal
    看的是：這個 band 的原始振動波形長怎樣
    
    下圖
     - envelope 疊圖
     - 橘色：Leak
     - 藍色：Normal
    看的是：這個 band 的能量輪廓差異
    """
    from scipy.signal import hilbert
    # 建立畫布:2個子圖(2列1欄)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=150, sharex=True)
    
    # num_samples=None 時畫全部
    n_leak   = len(records_leak)   if num_samples is None else num_samples
    n_normal = len(records_normal) if num_samples is None else num_samples
    
    # 上圖：raw filtered waveform 疊圖
    # 對每一筆Leak:取出 ai1 -> 做 bandpass filter，只保留指定頻帶 -> 建立時間軸 -> 畫出這個 band 的波形(橘色)
    for r in records_leak[:num_samples]:
        sig      = r["signal"]["ai1"]
        fs       = r["fs"]
        filtered = bandpass_filter(sig, fs, band["f_low"], band["f_high"])
        t        = np.arange(len(filtered)) / fs
        axes[0].plot(t[::10], filtered[::10],
                     color="orange", alpha=0.5, linewidth=0.8)
    # normal:藍色
    for r in records_normal[:num_samples]:
        sig      = r["signal"]["ai1"]
        fs       = r["fs"]
        filtered = bandpass_filter(sig, fs, band["f_low"], band["f_high"])
        t        = np.arange(len(filtered)) / fs
        axes[0].plot(t[::10], filtered[::10],
                     color="steelblue", alpha=0.5, linewidth=0.8)

    axes[0].set_title(
        f"Bandpass {band['f_low']//1000}k~{band['f_high']//1000}kHz — "
        f"Baseline+Leak(橘) vs Baseline(藍) waveform"
    )
    axes[0].set_ylabel("Amplitude")

    # 下圖：envelope 疊圖（更清楚看能量差）
    # 對每一筆Leak:取出 ai1 -> 做 bandpass filter，只保留指定頻帶 -> 用 Hilbert transform 算 envelope -> 算 RMS -> 把 envelope 畫出來(橘色)
    for r in records_leak[:num_samples]:
        sig      = r["signal"]["ai1"]
        fs       = r["fs"]
        filtered = bandpass_filter(sig, fs, band["f_low"], band["f_high"])
        env      = np.abs(hilbert(filtered))   # 取出這個 band 的能量外框
        rms      = np.sqrt(np.mean(filtered**2))
        t        = np.arange(len(env)) / fs
        axes[1].plot(t[::10], env[::10],
                     color="orange", alpha=0.6, linewidth=1.0,
                     label=f"leak RMS={rms:.2e}")
    # normal:藍色
    for r in records_normal[:num_samples]:
        sig      = r["signal"]["ai1"]
        fs       = r["fs"]
        filtered = bandpass_filter(sig, fs, band["f_low"], band["f_high"])
        env      = np.abs(hilbert(filtered))
        rms      = np.sqrt(np.mean(filtered**2))
        t        = np.arange(len(env)) / fs
        axes[1].plot(t[::10], env[::10],
                     color="steelblue", alpha=0.6, linewidth=1.0,
                     label=f"normal E={rms:.2e}")

    axes[1].set_title("Envelope 比較（能量差異）")
    axes[1].set_ylabel("Envelope Amplitude")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(fontsize=7, ncol=2)

    plt.suptitle(
        f"Band Zoom: {band['f_low']//1000}k~{band['f_high']//1000}kHz",
        fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Saved] {save_path}")
    else:
        plt.show()   

   
# ============================================================
# 同一個類別內，比較兩個通道 ai0(far) vs ai1(near)的 PSD
# ============================================================
def plot_within_label_dual_channel_psd(
    records,                    # 同一類別的資料列表，每筆 record 應包含 signal 與 fs
    label_name="Baseline",      # 圖上顯示的類別名稱
    save_path=None,             # 若有指定路徑則存圖，否則直接顯示
    nperseg=NPERSEG,            # PSD 計算時每段長度
    smooth_sigma=1.0,           # 平滑用的高斯 sigma，越大越平滑
    fmin=0,                     # 顯示的最低頻率
    fmax=None,                  # 顯示的最高頻率，若不指定則用 Nyquist 頻率
    mark_low=5_000,             # 主觀察頻帶下限
    mark_high=70_000,           # 主觀察頻帶上限
    aux_low=420_000,            # 輔助觀察頻帶下限
    aux_high=470_000,           # 輔助觀察頻帶上限
):
    from scipy.ndimage import gaussian_filter1d    # 匯入一維高斯平滑函式，用來讓 PSD 曲線更平順

    # 如果沒有指定最高頻率，就用取樣頻率一半（Nyquist frequency）
    if fmax is None:
        fmax = FS / 2

    # ------------------------------------------------------------
    # 內部函式：計算某個 channel 在多筆 records 上的平均 PSD
    # ------------------------------------------------------------
    def avg_psd(records, channel):
        psd_list = []     # 用來收集每筆資料算出的 PSD
        f_ref = None      # 用來保存對應的頻率軸

        # 逐筆處理資料
        for r in records:
            # 取出指定通道的訊號，並轉成 float 的 numpy array
            x = np.asarray(r["signal"][channel], dtype=float)
            
            # 取出這筆資料的取樣頻率
            fs = float(r["fs"])
            
            # 計算該筆訊號的 PSD, smooth_sigma=0 表示先不要在 compute_psd 裡做平滑
            f, Pxx = compute_psd(
                x, fs,
                nperseg=nperseg,
                normalize=False,
                smooth_sigma=0
            )

            # # 建立頻率範圍遮罩，只保留 fmin ~ fmax 之間的頻率
            mask = (f >= fmin) & (f <= fmax)
            f = f[mask]      # 套用遮罩，保留目標頻段
            Pxx = Pxx[mask]

            psd_list.append(Pxx)     # 把這筆資料的 PSD 加入列表
            f_ref = f                # 記住頻率軸，後面畫圖要用

        if len(psd_list) == 0:       # 如果沒有任何資料，回傳空值
            return None, None

        # 把多筆 PSD 做幾何平均
        mean_psd = np.exp(np.mean(np.log(np.array(psd_list) + 1e-12), axis=0))

        # 如果有指定平滑參數，就再做一次高斯平滑
        if smooth_sigma and smooth_sigma > 0:
            mean_psd = gaussian_filter1d(mean_psd, sigma=smooth_sigma)

        # 回傳頻率軸與平均 PSD
        return f_ref, mean_psd
    
    # 分別計算 ai0 與 ai1 的平均 PSD
    f0, P0 = avg_psd(records, "ai0")
    f1, P1 = avg_psd(records, "ai1")

    # 如果任一通道沒有足夠資料，就印出警告並結束
    if f0 is None or f1 is None:
        print(f"⚠️ {label_name} 沒有足夠資料可畫 ai0 vs ai1 PSD")
        return

    # 建立圖表，設定尺寸與解析度
    plt.figure(figsize=(12, 6), dpi=160)
    # 畫 ai0 的 PSD 曲線，使用半對數座標（y 軸為 log）
    plt.semilogy(f0, P0, color="steelblue", linewidth=2, label=f"{label_name} ai0 (far)")
    # 畫 ai1 的 PSD 曲線
    plt.semilogy(f1, P1, color="tomato", linewidth=2, label=f"{label_name} ai1 (near)")
    # 用淡金色區塊標出核心觀察頻段 5k ~ 70k
    plt.axvspan(mark_low, mark_high, color="gold", alpha=0.08, label="Core 5–70k")
    # 畫出核心觀察頻段的左右邊界線
    plt.axvline(mark_low, color="goldenrod", linestyle="--", linewidth=1.0)
    plt.axvline(mark_high, color="goldenrod", linestyle="--", linewidth=1.0)
    # 用淡紅色區塊標出輔助觀察頻段 420k ~ 470k
    plt.axvspan(aux_low, aux_high, color="crimson", alpha=0.08, label="Aux 420–470k")

    plt.xlim(fmin, fmax)                      # 設定 x 軸範圍
    plt.xlabel("Frequency (Hz)")              # 設定 x 軸標籤
    plt.ylabel("PSD")                         # 設定 y 軸標籤
    plt.title(f"{label_name}: ai0 vs ai1")    # 設定圖標題
    plt.legend()                              # 顯示圖例
    plt.tight_layout()                        # 自動調整版面，避免標籤被截掉

    # 如果有指定儲存路徑
    if save_path:
        save_path = Path(save_path)    # 轉成 Path 物件
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ 已儲存: {save_path}")
    else:   # 若沒指定儲存路徑，就直接顯示圖
        plt.show()


# ============================================================
# 同一通道跨類別 PSD：Leak0 vs Leak1, 固定某個 channel（例如 ai1），比較 Leak0 與 Leak1 的平均 PSD
# ============================================================
def plot_cross_label_same_channel_psd(
    records_normal,          # 正常類別資料，例如 Leak0
    records_leak,            # 異常類別資料，例如 Leak1
    channel="ai1",           # 要比較的通道，預設 ai1
    save_path=None,          # 若有指定路徑則存圖，否則直接顯示
    nperseg=NPERSEG,         # PSD 計算時每段長度
    smooth_sigma=1.0,        # 平滑用的高斯 sigma
    fmin=0,                  # 顯示的最低頻率
    fmax=None,               # 顯示的最高頻率
    mark_low=5_000,          # 主觀察頻帶下限
    mark_high=70_000,        # 主觀察頻帶上限
    aux_low=420_000,         # 輔助觀察頻帶下限
    aux_high=470_000,        # 輔助觀察頻帶上限
):
    from scipy.ndimage import gaussian_filter1d

    # 如果沒有指定最高頻率，就自動設成 FS/2
    if fmax is None:
        fmax = FS / 2

    # ------------------------------------------------------------
    # 內部函式：把一整組 records 算成一條平均 PSD 曲線
    # ------------------------------------------------------------
    def avg_psd(records):
        psd_list = []      # 收集每一筆資料算出來的 PSD
        f_ref = None       # 記錄頻率軸 f

        # 逐筆處理這一組資料
        for r in records:
            x = np.asarray(r["signal"][channel], dtype=float)     # 從每一筆資料 r 中，抓出指定通道的訊號
            fs = float(r["fs"])
            f, Pxx = compute_psd(     # 計算 PSD, f：頻率座標, Pxx：每個頻率對應的功率譜密度
                x, fs,
                nperseg=nperseg,
                normalize=False,
                smooth_sigma=0
            )

            # 做頻率範圍篩選:只保留 fmin ~ fmax 之間的頻率
            mask = (f >= fmin) & (f <= fmax)
            f = f[mask]
            Pxx = Pxx[mask]

            psd_list.append(Pxx)
            f_ref = f
        # 如果這組資料根本沒有內容，就直接回傳空值，避免後面出錯
        if len(psd_list) == 0:
            return None, None
        # 計算多筆 PSD 的幾何平均：先取 log -> 對每個頻率位置做平均 -> 再用 exp 轉回原本尺度
        mean_psd = np.exp(np.mean(np.log(np.array(psd_list) + 1e-12), axis=0))

        # 如果有設定平滑參數，就把平均 PSD 再做一次高斯平滑
        if smooth_sigma and smooth_sigma > 0:
            mean_psd = gaussian_filter1d(mean_psd, sigma=smooth_sigma)

        return f_ref, mean_psd

    # 對兩個類別分別算平均 PSD
    fn, Pn = avg_psd(records_normal)
    fl, Pl = avg_psd(records_leak)

    # 如果任一組沒有資料，就印警告並停止
    if fn is None or fl is None:
        print(f"⚠️ {channel} 沒有足夠資料可畫 Leak0 vs Leak1 PSD")
        return

    # 繪圖：畫兩條 PSD 曲線 -> 藍色：Leak0 / 紅色：Leak1
    plt.figure(figsize=(12, 6), dpi=160)
    plt.semilogy(fn, Pn, color="steelblue", linewidth=2, label=f"Leak0 {channel}")
    plt.semilogy(fl, Pl, color="tomato", linewidth=2, label=f"Leak1 {channel}")

    plt.axvspan(mark_low, mark_high, color="gold", alpha=0.08, label="Core 5–70k")        # 畫一塊淡金色背景，表示你特別關注 5k ~ 70k Hz。
    plt.axvline(mark_low, color="goldenrod", linestyle="--", linewidth=1.0)
    plt.axvline(mark_high, color="goldenrod", linestyle="--", linewidth=1.0)
    plt.axvspan(aux_low, aux_high, color="crimson", alpha=0.08, label="Aux 420–470k")     # 畫第二個重點區間 420k ~ 470k Hz

    # 設定圖表資訊
    plt.xlim(fmin, fmax)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"{channel}: Leak0 vs Leak1")
    plt.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ 已儲存: {save_path}")
    else:
        plt.show()


# ============================================================
# 固定同一個 channel，比較兩組條件的平均 PSD, 例如：normal(ai1) vs ai1
# ============================================================
def plot_cross_condition_same_channel_psd(
    records_a,               # 要比較的資料
    records_b,               # 要比較的資料
    channel="ai1",           # 指定比較哪個通道,可用 "ai0"、"ai1"、"diff"
    label_a="Baseline",        # 圖例和標題顯示名稱
    label_b="Baseline+60",
    save_path=None,          # 如果有給路徑就存圖，沒有就直接顯示
    nperseg=NPERSEG,         # Welch PSD 的分段長度
    smooth_sigma=1.0,        # Gaussian 平滑強度
    fmin=0,                  # 顯示的頻率範圍
    fmax=None,
    mark_low=5_000,          # 核心頻段標示區間
    mark_high=70_000,        # 核心頻段標示區間
    aux_low=420_000,         # 輔助頻段標示區間
    aux_high=470_000,        # 輔助頻段標示區間
):
    from scipy.ndimage import gaussian_filter1d

    # 設定最高頻率(沒有手動指定 fmax，就用 Nyquist 頻率，也就是採樣率的一半)
    if fmax is None:
        fmax = FS / 2

    # ------------------------------------------------------------
    # 內部函式：把一組 records 的指定通道，全部算 PSD，再合成一條代表性的平均 PSD 曲線
    # ------------------------------------------------------------
    def avg_psd(records):
        psd_list = []     # 收每一筆資料的 PSD
        f_ref = None      # 記錄頻率軸，最後畫圖要用

        # 逐筆算 PSD
        for r in records:
            x = np.asarray(r["signal"][channel], dtype=float)     # 指定通道的訊號 x
            fs = float(r["fs"])                                   # 採樣率 fs

            # 呼叫 compute_psd(),計算出f：頻率軸, Pxx：對應的 PSD
            f, Pxx = compute_psd(
                x, fs,
                nperseg=nperseg,
                normalize=False,     # 保留原始 PSD 強度，不做最大值正規化
                smooth_sigma=0       # 先不要在每一筆資料層級做平滑
            )

            # 限制顯示頻率範圍(只保留關心的頻率區間)
            mask = (f >= fmin) & (f <= fmax)
            f = f[mask]
            Pxx = Pxx[mask]

            # 儲存每筆 PSD,之後做平均
            psd_list.append(Pxx)
            f_ref = f

        if len(psd_list) == 0:
            return None, None

        mean_psd = np.exp(np.mean(np.log(np.array(psd_list) + 1e-12), axis=0))

        if smooth_sigma and smooth_sigma > 0:
            mean_psd = gaussian_filter1d(mean_psd, sigma=smooth_sigma)

        return f_ref, mean_psd

    fa, Pa = avg_psd(records_a)
    fb, Pb = avg_psd(records_b)

    if fa is None or fb is None:
        print(f"⚠️ {label_a} vs {label_b} ({channel}) 沒有足夠資料可畫 PSD")
        return

    plt.figure(figsize=(12, 6), dpi=160)
    plt.semilogy(fa, Pa, color="steelblue", linewidth=2, label=f"{label_a} {channel}")
    plt.semilogy(fb, Pb, color="tomato", linewidth=2, label=f"{label_b} {channel}")

    plt.axvspan(mark_low, mark_high, color="gold", alpha=0.08, label="Core 5–70k")
    plt.axvline(mark_low, color="goldenrod", linestyle="--", linewidth=1.0)
    plt.axvline(mark_high, color="goldenrod", linestyle="--", linewidth=1.0)
    plt.axvspan(aux_low, aux_high, color="crimson", alpha=0.08, label="Aux 420–470k")

    plt.xlim(fmin, fmax)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"{label_a} vs {label_b} ({channel})")
    plt.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ 已儲存: {save_path}")
    else:
        plt.show()


# ============================================================
# 從整體角度看 normal 與 leak 兩類資料中，far/near 的平均 PSD 長什麼樣
# ============================================================
def plot_label_overview_psd(
    normal_records,    # normal 類別的 records
    leak_records,      # leak 類別的 records
    out_path=None,     
    smooth_sigma=1.0,  # 是否做平滑
    nperseg=NPERSEG,   # PSD 計算參數
):
    """
    畫一張只分 normal / leak 的總覽 PSD 圖：
    - 左圖：normal 的 ai0(far) vs ai1(near)
    - 右圖：leak 的 ai0(far) vs ai1(near)
    """
    from scipy.ndimage import gaussian_filter1d

    eps = 1e-12    # 避免後面幾何平均時 log(0) 出問題

    def mean_psd(records, ch_name):
        '''給一批 records 和一個 channel 名稱，算出這批資料在該 channel 的平均 PSD'''
        psd_list = []    # 收集每筆 record 的 PSD
        f_ref = None     # 記住頻率軸

        # 對每筆資料算 PSD
        for r in records:
            x = np.asarray(r["signal"][ch_name], dtype=float)    # 某筆 record 的指定 channel 訊號
            fs = float(r["fs"])
            f, Pxx = compute_psd(x, fs, nperseg=nperseg, normalize=False, smooth_sigma=0)    # 算出頻率與功率譜

            # 只保留 FREQ_MIN 以上
            mask = f >= FREQ_MIN
            f = f[mask]
            Pxx = Pxx[mask]

            # 收集 PSD
            psd_list.append(Pxx)
            f_ref = f
            
        # 如果沒資料就回傳空
        if len(psd_list) == 0:
            return None, None

        # 計算平均 PSD
        mean_psd = np.exp(np.mean(np.log(np.array(psd_list) + eps), axis=0))
        # 做平滑(如果有指定 smooth_sigma，就把平均後的 PSD 曲線平滑一下)
        if smooth_sigma and smooth_sigma > 0:
            mean_psd = gaussian_filter1d(mean_psd, sigma=smooth_sigma)

        return f_ref, mean_psd

    # 分別算四條曲線
    f_n0, P_n0 = mean_psd(normal_records, "ai0")    # normal 的 far
    f_n1, P_n1 = mean_psd(normal_records, "ai1")    # normal 的 near
    f_l0, P_l0 = mean_psd(leak_records, "ai0")      # leak 的 far
    f_l1, P_l1 = mean_psd(leak_records, "ai1")      # leak 的 near

    # 建立兩張子圖(sharey=True, 左右圖共用同一個 y 軸尺度)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160, sharey=True)

    # 左圖畫 normal(在 normal 類別中的平均 PSD)
    if f_n0 is not None and f_n1 is not None:
        axes[0].semilogy(f_n0, P_n0, color="steelblue", label="ai0 = far", linewidth=1.5)
        axes[0].semilogy(f_n1, P_n1, color="tomato", label="ai1 = near", linewidth=1.5)
        axes[0].set_title("Baseline - Mean PSD")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("PSD")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

    # 右圖畫 leak(在 leak 類別中的平均 PSD)
    if f_l0 is not None and f_l1 is not None:
        axes[1].semilogy(f_l0, P_l0, color="steelblue", label="ai0 = far", linewidth=1.5)
        axes[1].semilogy(f_l1, P_l1, color="tomato", label="ai1 = near", linewidth=1.5)
        axes[1].set_title("Baseline+Leak - Mean PSD")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"✅ normal/leak 總覽 PSD 圖已儲存: {out_path}")
    else:
        plt.show()


# ============================================================
# 兩組條件的箱型圖比較
# ============================================================
def plot_top_features_pairwise_boxplot(
        record_df,                # 每列是一個 record，每欄是一個特徵
        rank_df,                  # 特徵排名表，至少要有 feature 欄，最好還有像 d_0_50 這種效果量欄位
        level_a=0,                # 比較哪兩組，像 0 和 50
        level_b=60,       
        label_a="Baseline",       # 圖上顯示名稱
        label_b="Baseline+60",
        top_n=12,                 # 只畫前幾名特徵
        save_path=None,
        showfliers=True,         # 箱型圖要不要顯示離群值點
        robust_ylim=True,        # 要不要用穩健方式限制 y 軸範圍，避免極端值把圖撐壞 
        lower_pct=2,
        upper_pct=98,
    ):
    """
    畫兩組條件的 top feature boxplot
    例如：
    - Baseline vs Baseline+30
    - Baseline vs Baseline+50
    """
    # 從整張 record-level 表中,只挑出 level_a 和 level_b 這兩組
    df = record_df.copy()
    df = df[df["leak_quantity"].isin([level_a, level_b])].copy()

    # 從 ranking 裡拿前幾名特徵
    top_features = rank_df.head(top_n)["feature"].tolist()
    if len(top_features) == 0:    # 如果沒有特徵就結束
        print("⚠️ 沒有可畫的特徵")
        return

    # 建立畫布及設定
    n_cols = 4
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.2), dpi=160)
    axes = np.atleast_1d(axes).flatten()

    # 決定要顯示哪個 Cohen’s d 欄位
    d_col = f"d_{int(level_a)}_{int(level_b)}"
    if d_col not in rank_df.columns:
        d_col = f"d_{int(level_b)}_{int(level_a)}"
    if d_col not in rank_df.columns:
        d_col = None

    # 把 feature -> d值 做成字典
    d_map = {}
    if d_col is not None and d_col in rank_df.columns:
        d_map = dict(zip(rank_df["feature"], rank_df[d_col]))

    # 逐個特徵畫圖
    for idx, feat in enumerate(top_features):
        ax = axes[idx]

        # 準備這兩組特徵值
        data_to_plot = [
            df.loc[df["leak_quantity"] == level_a, feat].dropna(),
            df.loc[df["leak_quantity"] == level_b, feat].dropna(),
        ]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=[label_a, label_b],
            patch_artist=True,
            showfliers=showfliers
        )
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][1].set_facecolor("khaki" if level_b == 50 else "lightcoral")
        
        # 如果有開 robust_ylim，就限制 y 軸
        if robust_ylim:
            _set_robust_ylim(
                ax,
                data_to_plot,
                lower_pct=lower_pct,
                upper_pct=upper_pct,
                pad_ratio=0.08
            )

        d_val = d_map.get(feat, np.nan)    # 取出這個特徵的 Cohen’s d
        if pd.notna(d_val):
            ax.set_title(f"{feat}\n(Cohen's d = {d_val:.2f})", fontsize=9)
        else:
            ax.set_title(feat, fontsize=9)

        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(len(top_features), len(axes)):
        axes[idx].axis("off")

    suffix = " (robust view)" if robust_ylim and not showfliers else ""
    plt.suptitle(
        f"{label_a} vs {label_b} - Record-level Feature Comparison",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 已儲存: {save_path}")
    else:
        plt.show()


# ============================================================
# 同一個特徵從 Baseline -> +30 ，整體分布趨勢長什麼樣
# ============================================================
def plot_top_features_multilevel_boxplot(
        record_df,                               # record-level 特徵表
        rank_df,                                 # 特徵排名表，決定哪幾個特徵要畫
        target_levels=tuple(ANALYSIS_LEVELS),       # 一次要放進圖裡的 leak levels，預設 0, 30
        top_n=20,                                # 只畫前幾名特徵
        save_path=None,  
        showfliers=False,                        # 箱型圖是否顯示離群值
        robust_ylim=True,                        # 是否使用穩健 y 軸範圍
        lower_pct=2,
        upper_pct=98):
    """
    特徵有沒有隨 leak level 呈現趨勢
    哪些特徵在多個 leak level 間有層次感
    微小洩漏 +30 是否已開始偏移
    """
    
    # 只保留要的 leak levels:target_levels=tuple(ANALYSIS_LEVELS)
    df = record_df.copy()
    df = df[df["leak_quantity"].isin(target_levels)].copy()

    # 從排名表取前幾個特徵
    top_features = rank_df.head(top_n)["feature"].tolist()
    # 如果沒有特徵可畫就退出
    if len(top_features) == 0:
        print("⚠️ 沒有可畫的特徵")
        return

    # 整理 leak levels 順序與顯示名稱
    sorted_levels = sorted(target_levels)
    level_labels = [f"Baseline+{int(lv)}" if lv != 0 else "Baseline" for lv in sorted_levels]

    # 設定畫圖排版，每列放 4 張小圖，自動算需要幾列
    n_cols = 4
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    # 建立畫布
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.4), dpi=160)
    axes = np.atleast_1d(axes).flatten()

    # 設定各 leak level 的顏色
    color_map = {
        0: "lightblue",
        30: "khaki",
    }

    # 逐個特徵畫圖
    for idx, feat in enumerate(top_features):
        ax = axes[idx]

        # 準備這個特徵在各 leak level 的資料
        data_to_plot = [
            df.loc[df["leak_quantity"] == lv, feat].dropna()
            for lv in sorted_levels
        ]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=level_labels,
            patch_artist=True,
            showfliers=showfliers
        )

        for box, lv in zip(bp["boxes"], sorted_levels):
            box.set_facecolor(color_map.get(lv, "lightgray"))
            
        # 如果有開 robust_ylim，就調整 y 軸
        if robust_ylim:
            _set_robust_ylim(
                ax,
                data_to_plot,
                lower_pct=lower_pct,
                upper_pct=upper_pct,
                pad_ratio=0.08
            )

        ax.set_title(feat, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=25)

    # 多餘空子圖關掉
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis("off")

    # 準備總標題的 suffix
    suffix = " (robust view)" if robust_ylim and not showfliers else ""
    plt.suptitle(
        f"Baseline vs Baseline+30 - Record-level Feature Comparison{suffix}",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 已儲存: {save_path}")
    else:
        plt.show()
        
       
# ============================================================
# 趨勢折線圖，同一個特徵，隨 leak quantity 變化的折線圖，並把不同距離分成不同顏色
# ============================================================
def plot_top_features_multilevel_lines(
    record_df,                           # record-level 特徵表
    rank_df,                             # 特徵排名表，用來決定畫哪些特徵
    target_levels=tuple(ANALYSIS_LEVELS),   # 要看的 leak levels，例如 0, 30
    top_n=20,                            # 只畫前幾名特徵
    out_dir=None,
):
    """
    1.挑出排名前 top_n 的特徵
    2.對每個特徵，依 distance_cm × leak_quantity 分組
    3.算每組的 record-level 中位數
    4.每個距離畫成一條線
    5.看不同距離下，特徵是否隨 leak quantity 有一致趨勢
    """
    # 先只保留指定 leak levels
    df = record_df.copy()
    df = df[df["leak_quantity"].isin(target_levels)].copy()

    # 取前幾名特徵
    features_to_plot = rank_df.head(top_n)["feature"].tolist()

    # 如果有指定輸出資料夾，就先建立
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # 逐個特徵畫圖
    for feat in features_to_plot:
        if feat not in df.columns:
            continue
        
        # 依距離與 leak level 分組，取中位數
        g = (
            df.groupby(["distance_cm", "leak_quantity"])[feat]
            .median()
            .reset_index()
            .sort_values(["distance_cm", "leak_quantity"])
        )
        # 如果這個特徵沒有資料就跳過
        if g.empty:
            continue

        plt.figure(figsize=(8, 5), dpi=160)

        # 每個距離各畫一條線
        for dist, sub in g.groupby("distance_cm"):
            plt.plot(
                sub["leak_quantity"],
                sub[feat],
                marker="o",
                linewidth=1.8,
                label=f"{int(dist)} cm"
            )

        plt.title(f"{feat} vs Leak Quantity")
        plt.xlabel("Leak Quantity (ml)")
        plt.ylabel(f"{feat} (record-level median)")
        plt.xticks(sorted(target_levels))
        plt.legend(title="Distance")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        if out_dir is not None:
            plt.savefig(out_dir / f"{feat}_multilevel.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    if out_dir is not None:
        print(f"✅ 多層 leak 趨勢圖已存到: {out_dir}")


# ============================================================
# 畫前幾個重要特徵的折線圖：x=leak_quantity, color=distance_cm
# ============================================================
def plot_selected_feature_lines_by_leak_and_distance(
        record_df,
        features_to_plot,
        out_dir=None,
        label_filter=1,
        line_alpha=0.65,     # 線透明度
        marker_size=7,       # 點大小
        marker_edge_color="white",   # 點外框顏色
        marker_edge_width=0.8,       # 點外框寬度
    ):
    """
    每張圖一個 feature
    x 軸 = leak_quantity
    不同顏色 = distance_cm
    y 軸 = record-level median feature
    """
    df = record_df.copy()

    # 如果有指定 label_filter，就先只看某個類別
    if label_filter is not None:
        df = df[df["label"] == label_filter].copy()

    # 如果指定輸出資料夾，就先建立它
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # 開始逐個 feature 畫圖
    for feat in features_to_plot:
        if feat not in df.columns:    # 如果這個 feature 不存在，直接跳過
            continue
        # 按 distance_cm 和 leak_quantity 分組 -> 對該 feature 算中位數 -> 轉回一般表格 -> 排序
        g = (
            df.groupby(["distance_cm", "leak_quantity"])[feat]
            .median()
            .reset_index()
            .sort_values(["distance_cm", "leak_quantity"])
        )

        if g.empty:
            continue
        
        # 建立一張圖
        plt.figure(figsize=(8, 5), dpi=160)
        
        colors = plt.cm.tab10.colors         # 比較清楚的配色
        n_dist = g["distance_cm"].nunique()
        offsets = np.linspace(-3, 3, n_dist) if n_dist > 1 else np.array([0.0])     # 每條線一點點水平位移
        
        # 針對每個 distance_cm, 各自畫一條線, x 軸是 leak_quantity, y 軸是 feature 中位數
        for i, (dist, sub) in enumerate(g.groupby("distance_cm")):
            x = sub["leak_quantity"].to_numpy(dtype=float) + offsets[i]
            
            plt.plot(
                x,
                sub[feat],
                marker="o",
                markersize=marker_size,
                linewidth=1.8,
                alpha=line_alpha,
                color=colors[i % len(colors)],
                markerfacecolor=colors[i % len(colors)],
                markeredgecolor=marker_edge_color,
                markeredgewidth=marker_edge_width,
                label=f"{int(dist)} cm"
            )
            
        xticks = sorted(g["leak_quantity"].unique())
        plt.xticks(
            xticks,
            ["Baseline" if int(v) == 0 else f"Baseline+{int(v)}" for v in xticks]
        )

        plt.title(f"{feat} vs Leak Level Relative to Baseline (dev)")
        plt.xlabel("Leak Level Relative to Baseline (ml)")
        plt.ylabel(f"{feat} (record-level median)")
        plt.legend(title="Distance")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        if out_dir is not None:
            save_path = out_dir / f"{feat}_vs_leak_quantity.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    if out_dir is not None:
        print(f"✅ 折線圖已存到: {out_dir}")
        

# ============================================================
# 同一個 feature，在不同距離下，normal vs 該 leak level 的差異有沒有穩定
# ============================================================
def plot_top_features_distance_effect_lines(
    record_df,                           # record-level 特徵表
    rank_df,                             # 特徵排名表，用來決定畫哪些特徵
    target_levels=tuple(ANALYSIS_LEVELS),   # 要分析的 leak levels，預設 0, 30
    top_n=20,                            # 只畫排名前幾名特徵
    out_dir=None,                        # 輸出資料夾
):
    '''
    同一個特徵，在不同距離下，Baseline vs Baseline+leak 的區分力有沒有穩定。
    1.取前 top_n 個特徵
    2.對每個特徵、每個 leak level、每個距離
    3.計算 Baseline vs Baseline+leak 的 Cohen’s d
    4.以距離為 x 軸、Cohen’s d 為 y 軸畫線
    5.看這個特徵的區分力會不會因距離改變而失真
    '''
    # 先只保留指定 leak levels
    df = record_df.copy()
    df = df[df["leak_quantity"].isin(target_levels)].copy()

    # 取前幾名特徵
    top_features = rank_df.head(top_n)["feature"].tolist()
    leak_levels = [lv for lv in target_levels if lv != 0]       # 把 baseline 以外的 leak levels 抽出來
    distances = sorted(df["distance_cm"].dropna().unique())     # 取得所有距離

    # 如果有指定輸出資料夾就先建立
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    # 逐個特徵處理
    for feat in top_features:
        rows = []
        
        # 對每個 leak level、每個距離算 Cohen’s d
        for lv in leak_levels:
            for dist in distances:
                sub = df[df["distance_cm"] == dist]

                # 抽出 baseline 與 leak 的特徵值
                x0 = sub.loc[sub["leak_quantity"] == 0, feat].dropna().values     # 在固定距離 dist 下：x0：Baseline 的這個特徵值
                x1 = sub.loc[sub["leak_quantity"] == lv, feat].dropna().values    # 在固定距離 dist 下：x1：Baseline+lv 的這個特徵值

                # 樣本太少就跳過
                if len(x0) < 2 or len(x1) < 2:
                    continue
                
                # 計算 Cohen’s d:在固定距離下，這個特徵對 Baseline vs Baseline+lv 的分離強度
                d = _cohen_d(x1, x0)
                
                # 儲存結果
                rows.append({
                    "feature": feat,
                    "distance_cm": dist,
                    "leak_quantity": lv,
                    "cohen_d": d,
                    "abs_cohen_d": abs(d),
                })

        # 組合特徵表
        plot_df = pd.DataFrame(rows)
        if plot_df.empty:      # 如果沒有任何有效資料，就跳過這個特徵
            continue

        # 建立畫布
        plt.figure(figsize=(8, 5), dpi=160)

        # 每個 leak level 畫一條線
        for lv, sub in plot_df.groupby("leak_quantity"):
            sub = sub.sort_values("distance_cm")
            plt.plot(
                sub["distance_cm"],
                sub["cohen_d"],
                marker="o",
                linewidth=1.8,
                label=f"Baseline vs Baseline+{int(lv)}"
            )
        # 畫一條 y=0 基準線
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"{feat} - fixed distance normal vs leak effect")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Cohen's d")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        if out_dir is not None:
            plt.savefig(out_dir / f"{feat}_distance_effect_lines.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    if out_dir is not None:
        print(f"✅ 固定距離 effect line plots 已存到: {out_dir}")
              

# ============================================================
# record-level PCA:每個 record 壓成 2 維後畫散點圖
# ============================================================
def plot_record_level_pca(
    record_df,
    record_id_col="record_id",
    decision_col="decision",
    out_path=None,
    annotate_top_n=10,
    show_review_labels=True,
):
    """
    1.每個 record 是一個點
    2.用很多數值特徵做 PCA
    3.降到 2 維後畫在平面上
    4.依照 decision 上色，並特別把 review 點框起來
    """
    df = record_df.copy()

    # 選出所有數值欄位,排除幾個不想拿來做 PCA 的欄位["label", "leak_quantity", "distance_cm", "pressure", "soft_score"]
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in {"label", "leak_quantity", "distance_cm", "pressure", "soft_score"}
    ]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()    # 把 inf 轉成 nan,再把有缺值的列刪掉
    df = df.loc[X.index].reset_index(drop=True)                         # 確保 df 和特徵矩陣 X 保持同樣列數
    X = X.reset_index(drop=True)

    X_scaled = StandardScaler().fit_transform(X.values)                 # 標準化，避免某些特徵尺度太大主導 PCA
    pcs = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    plot_df = df.copy()
    plot_df["pc1"] = pcs[:, 0]
    plot_df["pc2"] = pcs[:, 1]
    plot_df["pc_radius"] = np.sqrt(plot_df["pc1"]**2 + plot_df["pc2"]**2)    # 計算每個點離原點的距離,用來找「最極端」的 review 點
    
    # 依照 decision_col 畫三類點
    color_map = {
        "keep": "steelblue",
        "review": "tomato",
        "remove": "gray",
    }

    plt.figure(figsize=(11, 8), dpi=160)

    values = [v for v in ["keep", "review", "remove"] if v in plot_df[decision_col].dropna().unique()]
    for v in values:
        sub = plot_df[plot_df[decision_col] == v]
        plt.scatter(
            sub["pc1"], sub["pc2"],
            s=52,
            alpha=0.78,
            color=color_map.get(v, "gray"),
            label=v,
        )
    # review 用黑框標
    review_sub = plot_df[plot_df[decision_col] == "review"]
    if len(review_sub) > 0:
        plt.scatter(
            review_sub["pc1"], review_sub["pc2"],
            s=95, facecolors="none", edgecolors="black",
            linewidths=1.0, label="review-outline",
            )

        # 只標最極端的幾個
        if show_review_labels:
            review_top = review_sub.sort_values("pc_radius", ascending=False).head(annotate_top_n)
            for _, row in review_top.iterrows():
                plt.annotate(
                    str(row[record_id_col])[-18:],
                    (row["pc1"], row["pc2"]),
                    fontsize=7, alpha=0.85
                )

    plt.title("Record-level PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return plot_df


# ============================================================
#  每個 group 各輸出兩種圖
# ============================================================
def plot_groupwise_qc_views(
        report_df,           # 完整資料表
        feature_cols=None,   # 想看的特徵欄位
        out_dir=None,        # 輸出資料夾
        max_groups=None,     # 最多畫幾個 group
    ):
    """
    每個 group 各輸出兩種圖：
    1. robust z-score heatmap：這個 group 裡每筆 record 在「標準化後」是否異常
    2. 原始特徵 boxplot + scatter points：在「原始特徵值」上實際落在哪裡
    """
    df = report_df.copy()

    # 沒有指定 feature_cols，就用預設
    if feature_cols is None:
        feature_cols = ["log_rms_gap", "log_peak_gap", "log_peak_ratio", "b_all_5_70k_log_gap",]

    # 保留真的存在於資料表的欄位
    valid_feature_cols = [c for c in feature_cols if c in df.columns]
    z_cols = [f"{c}_robust_z" for c in valid_feature_cols if f"{c}_robust_z" in df.columns]     # 找對應的 robust z-score 欄位

    # 建立輸出資料夾,如果有指定 out_dir，就先建立資料夾
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
    # 決定 group 的繪圖順序
    group_order = (
        df.groupby("group")["soft_score"]    # 先依 group 分組
        .sum()                               # 對每組的 soft_score 加總
        .sort_values(ascending=False)        # 依照總分由大到小排序
        .index
        .tolist()
    )
    # 如果有 max_groups，只取前幾個 group
    if max_groups is not None:
        group_order = group_order[:max_groups]
        
    # 顏色設定
    color_map = {"keep": "steelblue", "review": "tomato", "remove": "gray"}

    # 逐 group 處理
    for group_name in group_order:
        gdf = df[df["group"] == group_name].copy().reset_index(drop=True)    # 先把該 group 的資料取出, 如果這組沒資料就跳過
        if len(gdf) == 0:
            continue
        
        # 如果有 decision 欄位，就建立排序用欄位 _decision_rank
        if "decision" in gdf.columns:
            decision_rank = {"review": 0, "remove": 1, "keep": 2}
            gdf["_decision_rank"] = gdf["decision"].map(decision_rank).fillna(9)
        else:
            gdf["_decision_rank"] = 9
        # 計算 max_abs_z,如果 gdf 還沒有 max_abs_z 欄位，就自己算 -> 這個值可以視為「這筆資料最異常的程度」
        if "max_abs_z" not in gdf.columns:
            gdf["max_abs_z"] = gdf[z_cols].abs().max(axis=1) if len(z_cols) > 0 else 0.0

        # 排序 group 內 record,排序規則是：先看 decision(review 最前面) -> 再看 soft_score(分數高的在前) -> 再看 max_abs_z(越異常的在前)
        gdf = gdf.sort_values(
            by=["_decision_rank", "soft_score", "max_abs_z"],
            ascending=[True, False, False]
        ).reset_index(drop=True)
        # 整理檔名
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(group_name))

        # -- 圖1:heatmap
        #    只有在 z_cols 不為空時才畫
        if len(z_cols) > 0:
            heat_values = gdf[z_cols].values     # 取出 heatmap 數值

            fig, ax = plt.subplots(figsize=(8, max(6, 0.32 * len(gdf))), dpi=160)
            im = ax.imshow(heat_values, aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)    # 畫矩陣熱圖

            ax.set_title(f"{group_name} - robust z-score heatmap")   # X軸：把欄名顯示成原始特徵名，不顯示 _robust_z 後綴
            ax.set_xticks(np.arange(len(z_cols)))
            ax.set_xticklabels([c.replace("_robust_z", "") for c in z_cols], rotation=45, ha="right")

            # y軸：record_id 最後 18 個字 + decision
            y_labels = [
                f"{str(row['record_id'])[-18:]} | {row['decision']}"
                for _, row in gdf.iterrows()
            ]
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=8)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("robust z-score")

            plt.tight_layout()
            if out_dir is not None:
                plt.savefig(out_dir / f"{stem}_heatmap.png", bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

        # -- 圖2:boxplot + points
        n = len(valid_feature_cols)    # 有幾個特徵就畫幾個子圖
        if n > 0:
            fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.8), dpi=160)
            if n == 1:
                axes = [axes]

            for ax, col in zip(axes, valid_feature_cols):
                vals = gdf[col].dropna().values
                if len(vals) == 0:
                    ax.set_title(col)
                    continue
                # A) 畫 boxplot
                ax.boxplot(vals, positions=[0], widths=0.35, patch_artist=True,
                           boxprops=dict(facecolor="lightgray", alpha=0.6))
                #   所有點本來都會落在 x=0，會重疊，所以加入一點左右隨機偏移
                x_jitter = np.random.uniform(-0.10, 0.10, size=len(gdf))
                #   一筆一筆畫 scatter(review 點更大, 不透明度更高, 有黑色外框)
                for i, (_, row) in enumerate(gdf.iterrows()):
                    is_review = row.get("decision") == "review"
                    ax.scatter(
                        x_jitter[i], row[col],
                        s=60 if is_review else 34,
                        color=color_map.get(row.get("decision", "keep"), "gray"),
                        alpha=0.95 if is_review else 0.45,
                        edgecolors="black" if is_review else "none",
                        linewidths=0.8 if is_review else 0.0,
                        marker="o"
                    )

                ax.set_title(col)
                ax.set_xticks([0])
                ax.set_xticklabels([group_name], rotation=20)
                ax.grid(True, alpha=0.25, axis="y")

            plt.suptitle(f"{group_name} - group-wise QC features")
            plt.tight_layout()

            if out_dir is not None:
                plt.savefig(out_dir / f"{stem}_boxplot.png", bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()


# ============================================================
# 畫 group-level median heatmap(group 層級的特徵摘要)
# ============================================================
def plot_group_feature_summary_heatmap(
    group_feature_summary,
    feature_cols=None,
    group_col="group",
    out_path=None,
    sort_by=None,
):
    """
    每一列 = 一個 group
    每一欄 = 這個 group 某個特徵的代表值（這裡假設是 median）
    顏色 = 這個 group 在該特徵上，相對其他 groups 偏高還是偏低
    -> 「哪些 group 在哪些特徵上，整體表現明顯偏離其他 group？」
    """

    df = group_feature_summary.copy()
    # 決定要畫哪些特徵,如果沒有指定 feature_cols，就用預設
    if feature_cols is None:
        feature_cols = [
            "log_rms_gap",
            "log_peak_gap",
            "log_peak_ratio",
            "b_all_5_70k_log_gap",
        ]
        
    # 只保留 DataFrame 裡真的存在的欄位
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols) == 0:
        print("⚠️ group-level heatmap 沒有可用特徵")
        return None

    # 如果要指定排序欄位，就先排(如果你指定了 sort_by，而且該欄位存在,就依那個欄位由大到小排序,否則就照 group 名稱排序)
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values(group_col).reset_index(drop=True)

    # 針對每個特徵，跨 group 做 robust z-score
    heat_df = df[[group_col] + feature_cols].copy()

    # 對每個特徵做跨-group robust z-score
    for col in feature_cols:
        med = heat_df[col].median()                    # 先算這個特徵在所有 groups 之間的中位數
        mad = _mad(heat_df[col].values)                # 此特徵在各 group 間的穩健離散程度
        scale = max(1.4826 * mad, 1e-6)                # 把 MAD 轉成穩健尺度，並避免分母為 0
        heat_df[col] = (heat_df[col] - med) / scale    # 得到跨-group 的 robust z-score

    # 畫 heatmap(圖的大小會根據：特徵數量,group 數量自動調整。)
    plt.figure(figsize=(1.4 * len(feature_cols) + 3, 0.55 * len(heat_df) + 2), dpi=160)
    # 把數值矩陣畫成熱圖
    im = plt.imshow(
        heat_df[feature_cols].values,
        aspect="auto",
        cmap="coolwarm",    # 藍色方向通常代表偏低,紅色方向通常代表偏高
        vmin=-4,
        vmax=4,
    )

    plt.title("Group-level Median Heatmap")
    
    # x 軸：顯示每個特徵名稱，並旋轉 45 度避免重疊
    plt.xticks(
        ticks=np.arange(len(feature_cols)),
        labels=feature_cols,
        rotation=45,
        ha="right"
    )
    
    # y軸：顯示每個 group 名稱
    plt.yticks(
        ticks=np.arange(len(heat_df)),
        labels=heat_df[group_col].tolist()
    )

    cbar = plt.colorbar(im)
    cbar.set_label("robust z-score across groups")

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"✅ group-level median heatmap 已儲存: {out_path}")
    else:
        plt.show()

    return heat_df


# ============================================================
# 畫 near/far 頻譜差異的 group-level median heatmap
# ============================================================
def plot_group_near_far_spectral_heatmap(
        spectral_report_df,
        feature_cols=None,
        out_path=None,
        sort_by="distance_cm",
    ):
    """
    1.先把每個 group 的 near/far 頻譜差異特徵取 median
    2.再把這些 group-level median 轉成「跨 group 的 robust z-score」
    3.最後畫成 heatmap，看哪個 group 在哪些頻譜差異特徵上特別高或特別低
    """
    df = spectral_report_df.copy()

    # 如果沒指定 feature_cols，就用預設這些 near/far 特徵
    if feature_cols is None:
        feature_cols = [
            "psd_log_corr",
            "psd_log_cosine_sim",
            "psd_js_div",
            "psd_log_l1_gap",
            "log_rms_ratio",
            "b_all_5_70k_log_ratio",
            "b3_20_70k_log_ratio",
            "b5_120_70k_log_ratio",
        ]
    # 只保留實際存在的欄位
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols) == 0:
        print("⚠️ 沒有可畫的 near/far spectral 特徵")
        return None
    
    # group-level 聚合:準備 groupby("group") 後每欄怎麼聚合
    agg_dict = {"label": "median", "leak_quantity": "median", "distance_cm": "median", "pressure": "median"}
    for c in feature_cols:
        agg_dict[c] = "median"    # 每個 group 都會算出：label, leak_quantity, distance_cm, pressure, 每個 near/far feature 的 median
    # 「每列一個 group」的摘要表
    gdf = df.groupby("group", as_index=False).agg(agg_dict)

    # 排序：如果 sort_by 欄位存在，就照那欄排序，預設是 distance_cm, 否則就照 group 名稱排序
    if sort_by in gdf.columns:
        gdf = gdf.sort_values(sort_by).reset_index(drop=True)
    else:
        gdf = gdf.sort_values("group").reset_index(drop=True)

    # 建立 heatmap 資料
    heat_df = gdf[["group"] + feature_cols].copy()
    
    # 跨 group 做 robust z-score
    for col in feature_cols:
        med = heat_df[col].median()                    # 算所有 groups 的中位數 med
        mad = _mad(heat_df[col].values)                # 算跨 group 的 MAD mad
        scale = max(1.4826 * mad, 1e-6)                # 用 1.4826 * mad 當穩健尺度
        heat_df[col] = (heat_df[col] - med) / scale    # 再把每個 group 的值轉成 robust z-score

    # 畫 heatmap(x 軸是特徵名，y 軸是 group 名稱)
    plt.figure(figsize=(1.5 * len(feature_cols) + 3, 0.55 * len(heat_df) + 2), dpi=160)
    im = plt.imshow(heat_df[feature_cols].values, aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)

    plt.title("Near/Far Spectral Difference Heatmap")
    plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.yticks(np.arange(len(heat_df)), heat_df["group"].tolist())

    cbar = plt.colorbar(im)
    cbar.set_label("robust z-score across groups")

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"✅ near/far spectral heatmap 已儲存: {out_path}")
    else:
        plt.show()

    return gdf, heat_df


# ============================================================
# 每個 group 各畫：ai0(far) vs ai1(near) 的平均 PSD + log(Pxx_near / Pxx_far)
# ============================================================
def plot_group_near_far_psd(
    records,
    out_dir,
    use_pressure=False,
    nperseg=NPERSEG,
    smooth_sigma=1.0,
    max_groups=None,
):
    """
    把資料先依 group 分組，然後對每個 group 各畫一張圖，比較 ai0(far) 和 ai1(near) 的平均 PSD，並畫出 log(Pxx_near / Pxx_far)
    1.同一組條件下，far 跟 near 的頻譜長什麼樣
    2.哪些頻率是 near 比 far 強，或反過來
    """
    from scipy.ndimage import gaussian_filter1d

    # 先建立輸出資料夾與基本設定
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12

    # 先把 records 按 group 分類
    group_map = {}
    for r in records:
        group_name = make_group_name(r["meta"], use_pressure=use_pressure)    # 每筆 record 用 make_group_name(...) 算出它屬於哪個 group
        group_map.setdefault(group_name, []).append(r)

    items = sorted(group_map.items())
    if max_groups is not None:
        items = items[:max_groups]
    # 逐 group 處理
    for group_name, group_records in items:
        # 每次處理一個 group，並初始化
        psd_ai0_list = []    # 存每筆 far 的 PSD
        psd_ai1_list = []    # 存每筆 near 的 PSD
        f_ref = None         # 記住頻率軸

        # 對 group 內每筆 record 算 PSD
        for r in group_records:    
            ai0 = np.asarray(r["signal"]["ai0"], dtype=float)
            ai1 = np.asarray(r["signal"]["ai1"], dtype=float)
            fs = float(r["fs"])

            # 分別對：far, near 算 PSD,並保留原始 PSD 尺度(normalize=False)
            f0, P0 = compute_psd(ai0, fs, nperseg=nperseg, normalize=False, smooth_sigma=0)
            f1, P1 = compute_psd(ai1, fs, nperseg=nperseg, normalize=False, smooth_sigma=0)

            # 對齊長度與頻率範圍
            n = min(len(P0), len(P1))
            f = f0[:n]
            P0 = P0[:n]
            P1 = P1[:n]

            # 把太低頻的部分去掉，只保留 FREQ_MIN 以上的頻率
            mask = f >= FREQ_MIN
            f = f[mask]
            P0 = P0[mask]
            P1 = P1[mask]
            
            # 把每筆 record 的 PSD 收集起來
            psd_ai0_list.append(P0)
            psd_ai1_list.append(P1)
            f_ref = f
        # 如果這個 group 沒資料就跳過
        if len(psd_ai0_list) == 0:
            continue
        
        # 計算 group 的幾何平均 PSD
        mean_psd_ai0 = np.exp(np.mean(np.log(np.array(psd_ai0_list) + eps), axis=0))
        mean_psd_ai1 = np.exp(np.mean(np.log(np.array(psd_ai1_list) + eps), axis=0))
        log_ratio = np.log((mean_psd_ai1 + eps) / (mean_psd_ai0 + eps))     # 計算 near/far 的 log ratio

        # 可選的平滑(如果 smooth_sigma > 0，就用 Gaussian filter 平滑曲線)
        if smooth_sigma and smooth_sigma > 0:
            mean_psd_ai0 = gaussian_filter1d(mean_psd_ai0, sigma=smooth_sigma)
            mean_psd_ai1 = gaussian_filter1d(mean_psd_ai1, sigma=smooth_sigma)
            log_ratio = gaussian_filter1d(log_ratio, sigma=smooth_sigma)
        
        # 畫兩張子圖
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=160, sharex=True)

        # 上圖：平均 PSD 比較(near / far 的整體頻譜形狀,哪些頻率能量比較高)
        axes[0].semilogy(f_ref, mean_psd_ai0, color="steelblue", label="ai0 = far", linewidth=1.5)
        axes[0].semilogy(f_ref, mean_psd_ai1, color="tomato", label="ai1 = near", linewidth=1.5)
        axes[0].set_title(f"{group_name} - Mean PSD")
        axes[0].set_ylabel("PSD")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)

        # 下圖：log ratio(near 相對 far 的差異,灰色虛線 y=0 是基準線)
        axes[1].plot(f_ref, log_ratio, color="purple", linewidth=1.3)
        axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
        axes[1].set_title("log(Pxx_near / Pxx_far)")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Log Ratio")
        axes[1].grid(True, alpha=0.25)

        plt.tight_layout()

        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(group_name))
        plt.savefig(out_dir / f"{stem}_near_far_psd.png", bbox_inches="tight")
        plt.close(fig)

    print(f"✅ 每個 group 的 near/far PSD 比較圖已存到: {out_dir}")


# ============================================================
# 每個 group 各畫一張 band energy 比較圖
# ============================================================
def plot_group_near_far_band_bars(
    records,
    bands,
    out_dir,
    use_pressure=False,
):
    """
    1.每個 group 各畫一張圖
    2.比較 far(ai0) 和 near(ai1) 在各個 band 的能量大小
    3.同時看 log(E_near / E_far)，確認哪個 band 是 near 明顯較強或較弱
    """
    # 建立輸出資料夾與基本參數
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12
    
    # 先把 records 依 group 分類
    group_map = {}
    for r in records:
        group_name = make_group_name(r["meta"], use_pressure=use_pressure)     # 每筆 record 用 make_group_name(...) 決定所屬 group
        group_map.setdefault(group_name, []).append(r)
        
    # 逐 group 處理
    for group_name, group_records in sorted(group_map.items()):
        rows = []    # rows 是用來收集這個 group 內每筆 record 的 band 特徵

        # 對 group 內每筆 record 算各 band energy
        for r in group_records:
            ai0 = np.asarray(r["signal"]["ai0"], dtype=float)
            ai1 = np.asarray(r["signal"]["ai1"], dtype=float)
            fs = float(r["fs"])
            
            # 先建立一列資料
            row = {"record_id": r["record_id"]}
            for band in bands:
                e0 = compute_band_energy(ai0, fs, band)     # far 在這個 band 的能量
                e1 = compute_band_energy(ai1, fs, band)     # near 在這個 band 的能量
                row[f"{band['name']}_e0"] = e0
                row[f"{band['name']}_e1"] = e1
                row[f"{band['name']}_log_ratio"] = np.log((e1 + eps) / (e0 + eps))
            rows.append(row)

        if len(rows) == 0:
            continue

        # 把 group 內所有 record 組成 DataFrame:每列 -> 一筆 record , 每欄 -> 某個 band 的 far/near energy 或 log ratio
        df_band = pd.DataFrame(rows)

        # 對每個 band 取 group median
        band_names = [b["name"] for b in bands]
        med_e0 = [df_band[f"{name}_e0"].median() for name in band_names]         # 每個 band 的 far median energy
        med_e1 = [df_band[f"{name}_e1"].median() for name in band_names]         # 每個 band 的 near median energy
        med_lr = [df_band[f"{name}_log_ratio"].median() for name in band_names]  # 每個 band 的 median log ratio

        # 建立 x 軸位置, x 是每個 band 在 x 軸上的位置
        x = np.arange(len(band_names))
        width = 0.36

        # 畫兩張子圖
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=160, sharex=True)

        # 上圖：far vs near 的 median band energy -> 在哪些 band，near 的能量比 far 高,差距大不大
        axes[0].bar(x - width / 2, med_e0, width=width, color="steelblue", label="ai0 = far")
        axes[0].bar(x + width / 2, med_e1, width=width, color="tomato", label="ai1 = near")
        axes[0].set_title(f"{group_name} - Median Band Energy")
        axes[0].set_ylabel("Energy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25, axis="y")

        # 下圖：median log(E_near / E_far) -> 快速判斷 band 差異方向
        axes[1].bar(x, med_lr, width=0.55, color="purple", alpha=0.85)
        axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
        axes[1].set_title("Median log(E_near / E_far)")
        axes[1].set_ylabel("Log Ratio")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(band_names, rotation=30, ha="right")
        axes[1].grid(True, alpha=0.25, axis="y")

        plt.tight_layout()

        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(group_name))
        plt.savefig(out_dir / f"{stem}_band_energy.png", bbox_inches="tight")
        plt.close(fig)

    print(f"✅ 每個 group 的 band energy 比較圖已存到: {out_dir}")


# ============================================================
# 畫多個特徵的熱圖 selected features vs (leak_quantity, distance_cm) heatmap
# ============================================================
def plot_selected_feature_heatmap_by_condition(
        record_df,
        selected_features,
        out_path=None,
        label_filter=1,
    ):
    """
    列 = feature
    欄 = leak_quantity × distance_cm
    值 = record-level median，並做 feature-wise robust z-score
    """
    df = record_df.copy()
    # 如果有指定 label_filter，就只保留某個類別
    if label_filter is not None:
        df = df[df["label"] == label_filter].copy()

    # 從你給的 selected_features 裡，挑出實際存在於 dataframe 欄位中的特徵
    use_features = [c for c in selected_features if c in df.columns]
    if len(use_features) == 0:    # 如果一個可用特徵都沒有，就直接結束
        print("⚠️ 沒有可畫的 selected features")
        return None, None

    # 依照 leak_quantity 和 distance_cm 分組 -> 每組對所有指定 feature 算中位數 -> 產生一張整理過的表
    cond_df = (
        df.groupby(["leak_quantity", "distance_cm"])[use_features]
        .median()
        .sort_index()
    )

    if cond_df.empty:
        print("⚠️ 沒有資料可畫 selected feature heatmap")
        return None, None

    heat_df = cond_df.T.copy()    # 表格轉置

    # feature-wise robust z-score：每個 feature 自己做標準化
    for feat in heat_df.index:
        vals = heat_df.loc[feat].values.astype(float)
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        scale = max(1.4826 * mad, 1e-6)
        heat_df.loc[feat] = (vals - med) / scale

    # 幫每個欄位做顯示標籤
    col_labels = [
        f"{'Baseline' if int(lq) == 0 else f'Baseline+{int(lq)}'}\n{int(dist)}cm"
        for lq, dist in heat_df.columns
    ]

    plt.figure(
        figsize=(max(10, 0.85 * len(col_labels)), max(10, 0.26 * len(use_features))),
        dpi=180
    )
    im = plt.imshow(
        heat_df.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-4,
        vmax=4,
    )

    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(use_features)), use_features)
    plt.title("Selected Features vs Leak Quantity / Distance (dev, record-level)")
    plt.xlabel("Condition")
    plt.ylabel("Feature")

    cbar = plt.colorbar(im)
    cbar.set_label("Feature-wise robust z-score")

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"✅ 已儲存: {out_path}")
    else:
        plt.show()

    return cond_df, heat_df


# ============================================================
# 正常 vs 洩漏的特徵狀態，並產生cohen's d
# ============================================================
def compare_normal_vs_leak_features(features_df, top_n=10, save_path=None):
    """
    對比正常 vs 洩漏的特徵分布 (根據leak_label分組)
    找出那些特徵最能區分[正常訊號 vs 洩漏訊號]，並用圖表展示
    適合用來做整體特徵區分性分析和可視化統計對比
    
    1️⃣ 計算每個特徵的區分能力
    2️⃣ 挑出 top N
    3️⃣ 畫箱型圖比較分布
   
    Parameters
    ----------
    features_df 包含所有特徵和 'leak_label' 的資料 : DataFrame        
    top_n 顯示前 N 個最有區分性的特徵 : int        
    save_path 圖片儲存路徑: str or None
        
    """
    
    print("\n" + "=" * 60)
    print("對比正常 vs 洩漏特徵分布")
    print("=" * 60)
    
    # 分離正常和洩漏樣本
    normal = features_df[features_df['label'] == 0]
    leak = features_df[features_df['label'] == 1]
    
    print(f"正常樣本: {len(normal)} 筆")
    print(f"洩漏樣本: {len(leak)} 筆")
    
    # 選擇數值特徵，排除metadata特徵
    exclude_cols = ['distance', 'pressure', 'leak_quantity', 'record_id', 'label', 'window_idx', "t_start", "label"]
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_features if c not in exclude_cols]
    
    # 計算區分度
    separability = []
    for feat in feature_cols:
        try:
            # 計算兩群的平均
            mean_normal = normal[feat].mean()    # 正常訊號的平均值
            mean_leak = leak[feat].mean()    # 洩漏訊號的平均值
            std_pooled = np.sqrt((normal[feat].std()**2 + leak[feat].std()**2) / 2)     # 合併標準差
            
            # 計算Cohen's d:數值越高代表該特徵具有越高的區分力
            if std_pooled > 0:
                cohen_d = abs(mean_leak - mean_normal) / std_pooled
                separability.append({
                    'feature': feat,
                    'cohen_d': cohen_d,
                    'mean_normal': mean_normal,
                    'mean_leak': mean_leak
                })
        except:
            continue
    
    # 尋找Top特徵
    sep_df = pd.DataFrame(separability).sort_values('cohen_d', ascending=False)
    top_features = sep_df.head(top_n)['feature'].tolist()
    
    print(f"\n前 {top_n} 個最具區分性的特徵:")
    print(sep_df.head(top_n)[['feature', 'cohen_d']].to_string(index=False))
    
    # 繪製箱型圖
    n_cols = 4
    n_rows = (top_n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    # 逐一畫每個重要的特徵
    for idx, feat in enumerate(top_features):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        data_to_plot = [normal[feat].dropna(), leak[feat].dropna()]
        
        # 畫箱型圖，patch_artist=True → 才能填顏色
        bp = ax.boxplot(data_to_plot, labels=['正常', '洩漏'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(f'{feat}\n(Cohen\'s d = {sep_df[sep_df["feature"]==feat]["cohen_d"].values[0]:.2f})',
                    fontsize=9)
        ax.set_ylabel('特徵值')
        ax.grid(True, alpha=0.3, axis='y')
    
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('正常 vs 洩漏 - 特徵分布對比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 對比圖已儲存: {save_path}")
        plt.close(fig)
    
    # 將計算好的Cohen's d的分數存成字典傳出，方便後續計算共線性使用
    cohen_dict = dict(zip(sep_df.feature, sep_df.cohen_d))
    
    return fig, sep_df, cohen_dict


# ============================================================
# 主程式入口
# ============================================================
def plot_all_features_pairwise_boxplots(
        record_df,    # record-level 特徵表
        level_a,      # 第一組，例如 0
        level_b,      # 第二組，例如 30
        out_dir,      # 儲存圖的路徑
        level_col="leak_quantity",    # 用哪個欄位分組，預設用洩漏量
        features_per_page=12,     # 每頁畫幾個特徵
        prefix=None,     # 輸出檔名前綴
        label_a=None,    # 圖上顯示的組名
        label_b=None,
    ):
    '''
    把兩組資料的所有 record-level 特徵，一頁一頁畫成箱型圖，並先用 Cohen's d 幫特徵排序
    '''
    # 把輸出路徑轉成 Path，並建立資料夾
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只保留你要比較的兩組資料，例如只留 0 和 30
    df = record_df[record_df[level_col].isin([level_a, level_b])].copy()

    # 這裡列出哪些欄位不是特徵，不要拿來畫箱型圖
    exclude_cols = {
        "filename", "record_id", "label",
        "leak_quantity", "distance_cm", "pressure",
        "condition", "group",
        "decision", "soft_score", "ratio_warning",
        "ratio_reason", "reason", "family_reason"
    }

    # 自動找出真正要分析的特徵欄(必須是數值型, 不能在排除清單裡)
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
    ]
    # 準備一個空 list，等等存每個特徵的排序分數
    rows = []
    # 逐個特徵處理
    for feat in feature_cols:
        # 把這個特徵拆成兩組：x0：第一組, x1：第二組
        x0 = df.loc[df[level_col] == level_a, feat].dropna().values
        x1 = df.loc[df[level_col] == level_b, feat].dropna().values
        
        # 如果其中一組樣本太少，就跳過，不畫
        if len(x0) < 2 or len(x1) < 2:
            continue

        # 計算 Cohen's d
        s0 = np.std(x0, ddof=1)    # 先算兩組標準差
        s1 = np.std(x1, ddof=1)
        pooled = np.sqrt((s0**2 + s1**2) / 2.0)    # 再算 pooled std
        d = 0.0 if pooled < 1e-12 else (np.mean(x1) - np.mean(x0)) / pooled     # 最後算效果量, d 越大，表示兩組差異越明顯

        rows.append({
            "feature": feat,
            "cohen_d": d,
            "abs_cohen_d": abs(d),
        })

    # 把所有特徵整理成表，並依照 abs_cohen_d 由大到小排序
    rank_df = pd.DataFrame(rows).sort_values("abs_cohen_d", ascending=False).reset_index(drop=True)
    # 如果最後沒有任何可畫特徵，就提示並結束
    if rank_df.empty:
        print(f"⚠️ {level_a} vs {level_b} 沒有可畫特徵")
        return None
    
    # 如果你沒有自己指定檔名前綴，就自動產生。
    if prefix is None:
        prefix = f"boxplot_{int(level_a)}_vs_{int(level_b)}"
        
    # 先把排序結果輸出成 CSV
    rank_df.to_csv(out_dir / f"{prefix}_ranking.csv", index=False, encoding="utf-8-sig")

    ranked_features = rank_df["feature"].tolist()     # 排序後的特徵清單
    n_pages = int(np.ceil(len(ranked_features) / features_per_page))     # 總共要畫幾頁

    # 如果你沒手動指定圖上的組名，就自動命名
    if label_a is None:
        label_a = "Baseline" if int(level_a) == 0 else f"Baseline+{int(level_a)}"
    if label_b is None:
        label_b = "Baseline" if int(level_b) == 0 else f"Baseline+{int(level_b)}"

    # 開始逐頁畫圖
    for page_idx in range(n_pages):
        # 取出這一頁要畫的特徵
        feats = ranked_features[page_idx * features_per_page:(page_idx + 1) * features_per_page]

        # 建立這一頁的子圖。
        # 每頁固定 4 欄 -> 列數依特徵數決定 -> flatten() 後方便逐格填圖
        n_cols = 4
        n_rows = int(np.ceil(len(feats) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.2), dpi=160)
        axes = np.atleast_1d(axes).flatten()
        
        # 逐個子圖、逐個特徵開始畫
        for ax, feat in zip(axes, feats):
            # 再次取出這個特徵在兩組的資料
            x0 = df.loc[df[level_col] == level_a, feat].dropna()
            x1 = df.loc[df[level_col] == level_b, feat].dropna()

            # 畫箱型圖
            bp = ax.boxplot(
                [x0, x1],
                tick_labels=[label_a, label_b],    # x 軸兩組名稱
                patch_artist=True,     # 讓盒子可以填色
                showfliers=False       # 不顯示離群點
            )
            # 把兩組箱型圖染成不同顏色
            bp["boxes"][0].set_facecolor("lightblue")
            bp["boxes"][1].set_facecolor("lightcoral")

            # 把這個特徵名和它的 Cohen's d 顯示在標題上
            d_val = rank_df.loc[rank_df["feature"] == feat, "cohen_d"].values[0]
            ax.set_title(f"{feat}\n(Cohen's d={d_val:.2f})", fontsize=9)
            # 加上 y 軸淡淡的格線，方便看分布高低
            ax.grid(True, alpha=0.3, axis="y")
            _set_robust_ylim(ax, [x0, x1], lower_pct=2, upper_pct=98, pad_ratio=0.08)    # 自動調整 y 軸範圍
        
        # 如果這一頁的格子比實際特徵數多，剩下空白格就關掉
        for ax in axes[len(feats):]:
            ax.axis("off")

        plt.suptitle(f"{label_a} vs {label_b} (record-level)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_page_{page_idx + 1:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"✅ {label_a} vs {label_b} 箱型圖已輸出，共 {n_pages} 頁")
    return rank_df



#                                               ****************************************************************
#                                                                            其他
#                                               ****************************************************************
# ***  Step2 正式流程輔助工具  ***
# ============================================================
# 從整張 window-level feature 表中，挑出可以拿來做 Step2b 擴充的「原始候選欄位」
# ============================================================
def select_step2_candidate_base_features(df):
    """
    從 window-level 特徵表中，挑出 Step2b 要拿來建立 rel / z / cusum / temporal 的原始候選欄位。
    注意：這個函式不是從 selected features 回推 raw features；
    它是假設 df 裡已經有 raw features，然後排除 rel/z/cusum/temporal 等衍生欄位。
    """
    # 建立一組要排除的欄位
    exclude_cols = {
        "filename", "record_id", "t_start", "label",
        "leak_quantity", "distance_cm", "pressure",
        "condition", "group",
    }

    # 建立空清單，用來存最後挑出來的候選原始特徵
    base_cols = []
    # 逐一檢查 df 裡的每個欄位名稱
    for c in df.columns:
        # 如果這個欄位是 metadata / label，就跳過
        if c in exclude_cols:
            continue
        # 如果這個欄位不是數值型態，也跳過
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        # 如果這個欄位已經是衍生特徵，也跳過
        if c.startswith("rel_") or c.startswith("z_") or c.startswith("cusum_"):
            continue
        # 如果這個欄位是 temporal 或 rolling 類衍生特徵，也跳過
        if c.startswith("delta_") or c.startswith("local_") or c.startswith("rolling_"):
            continue
        
        # 如果以上條件都沒有被排除，就把這個欄位加入候選原始特徵清單
        base_cols.append(c)

    return sorted(base_cols)


# ***  舊版 band scan / EDA 輔助區  ***
# ============================================================
# 定義 band grid
# ============================================================
def generate_fixed_bands(fs):
    """
    低頻細切（5k~20k，每 5kHz 一個 band）
    高頻粗切（20k~500k，每 50kHz 一個 band）
    """
    
    # Nyquist 與頻率上限控制,最高可分析頻率(基本 DSP 原則)
    nyq = fs / 2    
    f_max = int(min(500000, nyq * 0.95))     # range 只吃int整數資料，不吃float小數資料，最多到500k
    bands = []
    
    # 低頻細切：5k ~ 20k，每 5kHz
    # 低頻用高解析度（5k step）:低頻 → leak 特徵通常比較細緻
    low = 5000
    while low + 5000 <= 20000:
        bands.append({
            "f_low":  low,
            "f_high": low + 5000,
            "center": low + 2500
        })
        low += 5000

    # 高頻粗切：20k ~ 500k(f_max)，每 50kHz
    # 高頻用低解析度（50k step）:高頻 → 通常是 broadband noise
    low = 20000
    while low + 50000 <= f_max:
        bands.append({
            "f_low":  low,
            "f_high": low + 50000,
            "center": low + 25000
        })
        low += 50000

    # 尾段(最後一段不足 50k 的區間)
    if low < f_max:
        bands.append({
            "f_low":  low,
            "f_high": f_max,
            "center": (low + f_max) / 2
        })

    return bands


# ============================================================
# 把每一個固定頻段都拿來做一次 leak vs normal 比較，看看哪個頻段最有區分力
# ============================================================
def compute_band_scan_summary(
        records_leak,      # 洩漏資料，每筆 record 裡要有 ai0 / ai1 / diff / fs
        records_normal,    # 正常資料，每筆 record 裡要有 ai0 / ai1 / diff / fs
        bands,             # 掃描的固定頻段清單
        save_csv=None,     # 要不要把結果存成表格
    ):
    '''
    1.逐個 band 跑
    2.對每筆 leak / normal 算 band energy
    3.分別整理成 3 種指標
        - ai1 band energy
        - diff band energy
        - ai1/ai0 ratio
    4.對每種指標算
        - median
        - Cohen's d
        - log median gap
    5.最後把每個 band 的結果排個名
    '''
    eps = 1e-12

    rows = []     # 儲存每個 band 的摘要結果

    # 逐頻段掃描
    for band in bands:
        leak_ai1 = []           # 洩漏組在這個 band 的 ai1 能量
        normal_ai1 = []         # 正常組在這個 band 的 ai1 能量
        leak_diff = []          # 洩漏組在這個 band 的 diff 能量
        normal_diff = []        # 正常組在這個 band 的 diff 能量
        leak_ratio = []         # 洩漏組在這個 band 的 ai1/ai0 能量比
        normal_ratio = []       # 正常組在這個 band 的 ai1/ai0 能量比

        # 第一個內層迴圈：處理 leak, 對每一筆洩漏取出三個通道和fs
        for r in records_leak:
            fs = float(r["fs"])
            ai0 = np.asarray(r["signal"]["ai0"], dtype=float)
            ai1 = np.asarray(r["signal"]["ai1"], dtype=float)
            diff = np.asarray(r["signal"]["diff"], dtype=float)

            # 計算這筆 record 在這個 band 裡的能量
            e_ai0 = compute_band_energy(ai0, fs, band)
            e_ai1 = compute_band_energy(ai1, fs, band)
            e_diff = compute_band_energy(diff, fs, band)

            leak_ai1.append(e_ai1)
            leak_diff.append(e_diff)
            leak_ratio.append(e_ai1 / (e_ai0 + eps))

        # 第二個內層迴圈：處理 normal
        for r in records_normal:
            fs = float(r["fs"])
            ai0 = np.asarray(r["signal"]["ai0"], dtype=float)
            ai1 = np.asarray(r["signal"]["ai1"], dtype=float)
            diff = np.asarray(r["signal"]["diff"], dtype=float)

            e_ai0 = compute_band_energy(ai0, fs, band)
            e_ai1 = compute_band_energy(ai1, fs, band)
            e_diff = compute_band_energy(diff, fs, band)

            normal_ai1.append(e_ai1)
            normal_diff.append(e_diff)
            normal_ratio.append(e_ai1 / (e_ai0 + eps))

        # 建立基本資訊
        row = {
            "band_name": band.get("name", f"{band['f_low']}_{band['f_high']}"),
            "f_low": band["f_low"],
            "f_high": band["f_high"],
            "bandwidth": band["f_high"] - band["f_low"],

            # median 指標：這個 band 中，leak 跟 normal 的典型值是多少
            "ai1_leak_median": np.median(leak_ai1) if leak_ai1 else np.nan,
            "ai1_normal_median": np.median(normal_ai1) if normal_ai1 else np.nan,
            "diff_leak_median": np.median(leak_diff) if leak_diff else np.nan,
            "diff_normal_median": np.median(normal_diff) if normal_diff else np.nan,
            "ratio_leak_median": np.median(leak_ratio) if leak_ratio else np.nan,
            "ratio_normal_median": np.median(normal_ratio) if normal_ratio else np.nan,

            # Cohen's d 區分力
            "ai1_cohen_d": _cohen_d(leak_ai1, normal_ai1),                 # 單看 near sensor 的 band energy，這個 band 對 leak / normal 的區分力多大
            "diff_cohen_d": _cohen_d(leak_diff, normal_diff),              # 單看差分通道的 band energy，區分力多大
            "ratio_ai1_ai0_cohen_d": _cohen_d(leak_ratio, normal_ratio),   # 單看 near/far ratio，區分力多大

            # leak median 相對於 normal median 的對數比例差(看的是「差異方向與倍數關係」)
            "ai1_log_median_gap": np.log((np.median(leak_ai1) + eps) / (np.median(normal_ai1) + eps)),
            "diff_log_median_gap": np.log((np.median(leak_diff) + eps) / (np.median(normal_diff) + eps)),
            "ratio_log_median_gap": np.log((np.median(leak_ratio) + eps) / (np.median(normal_ratio) + eps)),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # 方便快速看哪幾段最值得關注：這個 band 的整體分離能力(把三種區分力取絕對值後平均)
    summary_df["score_abs_mean"] = summary_df[
        ["ai1_cohen_d", "diff_cohen_d", "ratio_ai1_ai0_cohen_d"]
    ].abs().mean(axis=1)

    # 前幾列就是最值得關注的關鍵頻段
    summary_df = summary_df.sort_values("score_abs_mean", ascending=False).reset_index(drop=True)

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(save_csv, index=False, encoding="utf-8-sig")
        print(f"✅ fixed-band scan summary 已儲存: {save_csv}")

    return summary_df


# *** 統計評估小工具 ***
# ============================================================
# 計算每個特徵區分力
# ============================================================
def _cohen_d(x1, x0):
    # 先轉成 numpy array
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    
    # 去掉 NaN:確保後面平均值、標準差不會被缺值干擾
    x1 = x1[~np.isnan(x1)]
    x0 = x0[~np.isnan(x0)]

    # 樣本太少就不算(少於 2 筆就沒辦法穩定估計變異)
    if len(x1) < 2 or len(x0) < 2:
        return np.nan

    # 計算標準差：ddof=1 表示用樣本標準差
    s1 = np.std(x1, ddof=1)
    s0 = np.std(x0, ddof=1)
    pooled = np.sqrt((s1**2 + s0**2) / 2.0)    # 合併標準差，用來把平均差正規化

    if pooled < 1e-12:
        return 0.0

    return (np.mean(x1) - np.mean(x0)) / pooled    # 正值：x1 比 x0 大, 負值：x1 比 x0 小, 絕對值越大：區分力越強


# ===========================================================
# 避免整個流程中斷
# ===========================================================
# AUC 需要 y_true 同時有正類和負類, 如果 y_true 只有一類，直接回傳 NaN, 如果計算失敗，也回傳 NaN
def _safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


# ============================================================
# 主程式入口
# ============================================================
"""
完整主流程（嚴格 dev / holdout 分離版）

Step 1. 讀原始資料（baseline / leak）
Step 2. hard QC
Step 3. 建立 diff channel
Step 4. record-level dev / holdout split

Step 4.2. Step2a：只在 dev 做候選頻帶探索
Step 4.3. split 後 soft QC（dev / holdout 分開做）
Step 4.4. 儲存三通道 TDMS（可選）

Step 5. dev-only EDA / 視覺化檢查
Step 6. 整理 dev / holdout 的 by-file file_specs
Step 7. 用 final_main_bands 做正式 window-level 特徵工程
Step 8. 建立正式擴充特徵
        （offband summary / baseline-relative / z / CUSUM / temporal）
Step 9. 壓成 record-level，做 ranking 與特徵篩選
Step 10. CV / holdout / LODO / permutation 驗證
Step 11. 輸出 final features / selecte

補充：
- Step2a 的目的是先收斂 leak-focused bands
- Step 7 ~ Step 10 的目的是在正式頻帶上收斂 final selected features
- 後續 transition / disturbance / runtime 不再自己重找頻帶或重選特徵，
  而是直接讀這支程式輸出的正式設定檔
"""

if __name__ == "__main__":
    # ============================================================
    # 繪圖輸出開關
    # 目的：
    # 1. 流程整合期先把必要圖和非必要圖分開控制
    # 2. 避免每次重跑都輸出大量探索期圖表
    # 3. 等現場資料進來後，再視需要打開特定圖做檢查
    # ============================================================
    
    SAVE_STEP5_SPECTROGRAM_PLOTS = False          # Step 5：是否輸出 ai0 / ai1 / diff 的時頻圖，偏肉眼檢查用
    SAVE_STEP5_BAND_ENERGY_PLOTS = False          # Step 5.1：是否輸出 Step2a 候選頻帶 / final_main_bands 的 band energy 比較圖
    SAVE_STEP5_PSD_PLOTS = False                  # Step 5.2：是否輸出 Baseline vs Leak 的 PSD 圖，作為整體頻譜差異參考
    
    SAVE_STEP9_RAW_BOXPLOTS = False               # Step 9：是否輸出依 rank_df_levels_raw 排名的基礎 boxplot（探索期用）
    SAVE_STEP9_DISTANCE_BOXPLOTS = False          # Step 9：是否輸出依距離穩定性排名的 boxplot（探索期用）
    SAVE_STEP9_DISTANCE_EFFECT_PLOTS = False      # Step 9：是否輸出 top features 的 distance effect 線圖
    SAVE_STEP9_DISTANCE_SENSITIVE_PLOTS = False   # Step 9：是否輸出最吃距離特徵的線圖，專門觀察距離敏感性
    SAVE_STEP9_FINAL_SUMMARY_PLOT = True          # Step 9：是否輸出 final ranking 的總結圖，建議保留
    
    SAVE_STEP11_SELECTED_FEATURE_HEATMAP = True   # Step 11：是否輸出最終保留特徵的 condition heatmap，建議保留
    SAVE_STEP11_SELECTED_FEATURE_LINE_PLOTS = False  # Step 11：是否輸出最終保留特徵的 leak/distance 線圖
    SAVE_STEP12_FINAL_SELECTED_DISTANCE_PLOTS = False  # Step 12：是否輸出最終特徵的距離泛化能力圖
    
    SAVE_MULTILEVEL_LINE_PLOTS = False            # 額外多漏量趨勢圖，偏探索期分析用
    SAVE_PAIRWISE_BOXPLOTS = False                # 額外 pairwise boxplot（例如 Baseline vs 30），偏探索期分析用
    SAVE_ALL_PAIRWISE_BOXPLOTS = False            # 輸出大量全特徵 pairwise boxplots，圖很多，通常只在深入探索時打開

    # 設定資料路徑與輸出資料夾
    data_root = Path("/Users/paul/Desktop/Final/normal_leak_test")               # 原始資料總資料夾
    out_root = Path("/Users/paul/Desktop/Final/test_result")             # 分析結果輸出資料夾
    # 先建立輸出資料夾(feature_plots, feature_selection)
    (out_root / "feature_plots").mkdir(parents=True, exist_ok=True) # 先建立兩個子資料夾：feature_plots, feature_selection
    (out_root / "feature_selection").mkdir(parents=True, exist_ok=True)

    # 設定常數變數
    extraction_cfg = load_feature_extraction_config(FEATURE_EXTRACTION_CONFIG_PATH)
    
    window_sec = float(extraction_cfg["window_sec"])
    hop_sec = float(extraction_cfg["hop_sec"])
    fs = float(extraction_cfg["fs"])
    nperseg = int(extraction_cfg["nperseg"])
    max_points = int(extraction_cfg["max_points"])
    
    base_feature_version = str(extraction_cfg["base_feature_version"])
    include_offband_summary = bool(extraction_cfg["include_offband_summary"])
    feature_extraction_config_hash = (calculate_file_sha256(FEATURE_EXTRACTION_CONFIG_PATH))

    # ============================================================
    # Step 1. 讀原始資料（只保留 normal / 30）
    # ============================================================
    # None：正式跑全部資料
    STEP2_RECORD_LIMIT = None     # 50：流程測試只取前 50 筆
    records_leak = load_tdms_data(data_root / "leak1")      # 讀洩漏資料
    records_normal = load_tdms_data(data_root / "leak0")    # 讀 baseline 資料
    
    if STEP2_RECORD_LIMIT is not None:
        records_leak = records_leak[:STEP2_RECORD_LIMIT]
        records_normal = records_normal[:STEP2_RECORD_LIMIT]

    # records_leak 裡面只保留要分析的洩漏量,目前的 TARGET_LEAK_LEVELS = {30}
    records_leak = [
        r for r in records_leak
        if int(round(r["meta"]["leak_quantity"])) in TARGET_LEAK_LEVELS
    ]
    # 只保留 baseline(正常) 那一類, records_normal 只留 0
    records_normal = [
        r for r in records_normal
        if int(round(r["meta"]["leak_quantity"])) == 0
    ]
    # 印出兩類資料筆數
    print(f"Baseline 資料: {len(records_normal)} 筆")
    print(f"Baseline+30 資料: {len(records_leak)} 筆")

    # 設定二元分類標籤：洩漏資料 label = 1 / baseline 資料 label = 0
    for r in records_leak:
        r["label"] = 1
    for r in records_normal:
        r["label"] = 0


    # ============================================================
    # Step 2. hard QC 硬性品質檢查
    # ============================================================
    # 檢查每筆資料有沒有：NaN / Inf / 全零 / RMS 太低
    abnormal_leak_quality = set(check_signal_quality(records_leak))
    abnormal_normal_quality = set(check_signal_quality(records_normal))

    # 刪掉不符合硬性品質的資料
    records_leak = [r for r in records_leak if r["record_id"] not in abnormal_leak_quality]
    records_normal = [r for r in records_normal if r["record_id"] not in abnormal_normal_quality]

    print(f"硬性排除 leak: {len(abnormal_leak_quality)} 筆")
    print(f"硬性排除 normal: {len(abnormal_normal_quality)} 筆")


    # ============================================================
    # Step 3. 建 diff channel
    # ============================================================
    records_leak = compute_diff_signal(records_leak)
    records_normal = compute_diff_signal(records_normal)
    
    
    # ============================================================
    # Step 4. 正式 dev / holdout split（建模專用）
    # 先在檔案層級切分，後續再由每個檔案切成 windows，可降低資料洩漏風險
    # ============================================================
    # 把洩漏跟 baseline 的 record 合併成一份完整清單 all_records
    all_records = records_leak + records_normal
    # 先把所有 record 的：record_id, label 取出來
    record_ids = np.array([r["record_id"] for r in all_records])
    record_labels = np.array([r["label"] for r in all_records])
    
    # 把資料切成(根據 ID 切分)：80% dev , 20% holdout
    dev_filenames, holdout_filenames = train_test_split(
        record_ids,
        test_size=0.2,
        stratify=record_labels,    # stratify 讓 normal/leak 比例維持一致
        random_state=42
    )

    dev_record_ids = set(dev_filenames)
    holdout_record_ids = set(holdout_filenames)

    # 把四組資料拆出來(因為後面 band 探索、特徵篩選都只能碰 dev)
    dev_records_leak = [r for r in records_leak if r["record_id"] in dev_record_ids]
    dev_records_normal = [r for r in records_normal if r["record_id"] in dev_record_ids]
    holdout_records_leak = [r for r in records_leak if r["record_id"] in holdout_record_ids]
    holdout_records_normal = [r for r in records_normal if r["record_id"] in holdout_record_ids]

    print(f"\nDev records: {len(dev_record_ids)}")
    print(f"Holdout records: {len(holdout_record_ids)}")
    
    # ============================================================
    # Step 4.1 判斷頻帶來源
    # ============================================================
    band_config_exists = BAND_CONFIG_PATH.exists()     # 檢查正式 band config 檔案存不存在
    need_band_discovery = (FORCE_BAND_DISCOVERY or not band_config_exists)     # 只要「使用者強制要求重新搜尋」（FORCE_BAND_DISCOVERY）或「正式 band config 根本不存在」，這次執行就需要跑頻帶搜尋
    
    print("\n=== 頻帶流程判斷 ===")
    print("正式 band config:", BAND_CONFIG_PATH)
    print("正式 band 是否存在:", band_config_exists)
    print("強制重新搜尋:", FORCE_BAND_DISCOVERY)
    print("本次是否執行頻帶搜尋:", need_band_discovery)
    
    # ============================================================
    # 支線 A：搜尋新頻帶
    # ============================================================
    # 如果需要頻帶搜尋，先用三元運算子決定原因文字：是「使用者強制要求」還是「檔案真的不存在」，印出來說明
    if need_band_discovery:
        reason = (
            "使用者要求強制重新搜尋"
            if FORCE_BAND_DISCOVERY
            else "找不到正式 band_config_step2.json"
        )
    
        print(f"\n開始 Step2a 頻帶搜尋，原因：{reason}")
    
        run_band_discovery_stage(
            records_dev=(dev_records_leak + dev_records_normal),     # 只餵 dev 資料進去
            output_dir=FINAL_OUTPUT_ROOT,
            resonance_json_path=RESONANCE_JSON_PATH,
            top_bands_to_keep=TOP_BANDS_TO_KEEP,
    
            # 正式特徵提取參數
            window_sec=window_sec,
            hop_sec=hop_sec,
            fs=fs,
            nperseg=nperseg,
            max_points=max_points,
            base_feature_version=base_feature_version,
        )
        # raise SystemExit(...) 直接終止程式, 頻帶搜尋只是產出「候選」結果，不能自動接著往下跑正式特徵工程，必須要人工介入（檢查候選頻帶、手動加 offband、定案存成正式檔、重新跑 baseline 建立腳本）
        raise SystemExit(
            "\nStep2a 頻帶探索已完成。\n"
            "已輸出 band_config_step2_candidate.json。\n"
            "請先檢查候選頻帶並加入 offband，\n"
            "再將定案結果存成 band_config_step2.json。\n"
            "接著重新執行 build_site_baseline.py，\n"
            "最後保持 FORCE_BAND_DISCOVERY=False，"
            "重新執行本程式完成 Step2b。"
        )
        
    # ============================================================
    # 支線 B: 使用既有且已定案的正式 band config
    # ============================================================
    print("\n找到正式 band_config_step2.json，跳過 Step2a 頻帶搜尋。")
    # 如果不需要頻帶搜尋, 把正式 band config 讀出來、驗證過，同時算出它的指紋 band_config_hash
    band_payload, band_config = load_frozen_band_config(BAND_CONFIG_PATH)
    band_config_hash = calculate_file_sha256(BAND_CONFIG_PATH)
 
    # 把 band_config 物件裡的三種頻段清單各自取出，存成獨立變數
    leak_focused_bands = list(band_config.leak_focused_bands)
    resonance_bands = list(band_config.resonance_bands)
    offband_bands_for_rule = list(band_config.offband_bands)      # Offband 只提供後續 Veto／抗干擾規則使用，不允許進入模型特徵選擇或模型訓練。
    
    # 把 leak-focused 跟 resonance 兩種頻段合併、處理重疊，得到最終要用來做特徵萃取的主頻段清單 final_main_bands
    final_main_bands = merge_band_groups(leak_focused_bands,resonance_bands,)
    
    # 用 set comprehension 分別把 leak-focused 頻段跟合併後主頻段的名稱各自收集成一個 set
    selected_leak_band_names = {str(band["name"])for band in leak_focused_bands}
    final_main_band_names = {str(band["name"])for band in final_main_bands}
    # 算出「原本在 leak-focused 裡、但合併之後卻不見了」的頻段名稱
    missing_leak_bands = sorted(selected_leak_band_names - final_main_band_names)
    # 如果真的有漏檢測頻段在合併過程中不見了，直接丟出錯誤終止程式
    if missing_leak_bands:
        raise ValueError(f"Step 2 選出的 leak-focused bands 在正式 main bands 中被移除：{missing_leak_bands}")
    
    print("Step 2 leak-focused bands：", len(leak_focused_bands),)
    print("正式 final_main_bands：",len(final_main_bands),)
    print("✅ 所有 Step 2 leak-focused bands 都已進入正式特徵提取")
    
    # 把 band config 的 temporal band 名稱,同步給 shared_feature_utils 的 raw feature extractor。
    configured_temporal_band_names = (configure_temporal_band_names(band_config.temporal_band_names))
    # 拿「同步之後回傳的結果」去反過來跟「原本 band config 裡的內容」比對，確保同步過程沒有出任何差錯
    if configured_temporal_band_names != set(band_config.temporal_band_names):
        raise ValueError("TEMPORAL_BAND_NAMES 與正式 band config 不一致")

    # 讀取固定 baseline，並且把「現在」的 band config 路徑跟特徵萃取設定路徑傳進去做指紋比對，確保 baseline 建立時的環境跟現在完全一致
    fixed_baseline = load_fixed_normal_baseline(
        FIXED_BASELINE_PATH,
        band_config_path=BAND_CONFIG_PATH,
        feature_extraction_config_path=(FEATURE_EXTRACTION_CONFIG_PATH),
        require_final=REQUIRE_FINAL_BASELINE,
    )
    
    # ============================================================
    # Step2b Baseline 資料隔離檢查
    # ============================================================
    
    # all_records 是切分前全部純正常／純洩漏 records，同時包含後續會進入 dev 與 holdout 的資料。
    # 一次檢查 all_records，可以確保固定 Baseline 不會與 dev 或 holdout 任一資料重複。
    validate_baseline_source_isolation(
        baseline_payload=fixed_baseline, analysis_records=all_records,
        context="Step2b dev/holdout", strict=STRICT_BASELINE_ISOLATION,
    )
    
    print("\n=== Step2b 正式輸入設定 ===")
    print("leak-focused bands:", len(leak_focused_bands))
    print("resonance bands:", len(resonance_bands))
    print("offband bands:", len(offband_bands_for_rule))
    print("baseline environment:",fixed_baseline.get("environment"),)
    print("baseline status:",fixed_baseline.get("schema_status"),)
    
    
    # ============================================================
    # Step 4.2. soft QC 軟性檢查（split 後分開做）
    # ============================================================
    # 把 dev、holdout 各自的 leak/normal 合併成完整清單
    dev_records_all = dev_records_leak + dev_records_normal
    holdout_records_all = holdout_records_leak + holdout_records_normal

    # Soft QC 同時檢查正式主頻帶與 Offband，用來觀察感測器一致性、寬頻干擾與異常訊號。
    qc_target_bands = final_main_bands + offband_bands_for_rule
    
    # 對 dev 和 holdout 分開做 soft QC，避免 dev/holdout 混在一起做判斷造成資訊洩漏
    dev_qc_report, dev_suspicious, dev_remove, dev_review, dev_keep, dev_group_summary = (
        check_leak_channel_consistency(
            dev_records_all,
            target_bands=qc_target_bands,
            save_csv=out_root / "feature_selection" / "soft_qc_dev.csv",
            verbose=True,
        )
    )
    
    holdout_qc_report, holdout_suspicious, holdout_remove, holdout_review, holdout_keep, holdout_group_summary = (
        check_leak_channel_consistency(
            holdout_records_all,
            target_bands=qc_target_bands,
            save_csv=out_root / "feature_selection" / "soft_qc_holdout.csv",
            verbose=True,
        )
    )
    
    # ============================================================
    # Step 4.3. 儲存三通道 TDMS（給 CNN-LSTM / 後續深度學習用）
    # ============================================================
    #save_diff_to_tdms(dev_records_all,out_root / "tdms_3ch_dev")
    #save_diff_to_tdms(holdout_records_all,out_root / "tdms_3ch_holdout")

    # ============================================================
    # Step 5. dev-only EDA / 視覺化檢查
    # ============================================================
    # 所有探索分析只用 dev, 不碰 holdout
    eda_records_leak = dev_records_leak
    eda_records_normal = dev_records_normal

    # 依洩漏量分組, 後面可以分別拿：Baseline vs 30
    eda_records_by_level = {
        lv: [r for r in eda_records_leak if int(round(r["meta"]["leak_quantity"])) == lv]
        for lv in ANALYSIS_LEAK_LEVELS
    }
    
    print(f"EDA Baseline 資料: {len(eda_records_normal)} 筆")
    for lv in ANALYSIS_LEAK_LEVELS:
        print(f"EDA Baseline+{lv} 資料: {len(eda_records_by_level[lv])} 筆")
    
    # 肉眼先看三通道 (ai0/ai1/diff) 的頻譜差異
    # 比較 baseline 的 ai0,ai1,diff 時頻圖
    if SAVE_STEP5_SPECTROGRAM_PLOTS:
        for i, r_normal in enumerate(eda_records_normal[10:25], start=1):
            plot_record_dual_channel_spectrogram(
                r_normal,
                save_path=out_root / "feature_plots" / "dual_channel_spectrogram" / f"dual_channel_baseline_{i:02d}.png",
                fmin=40_000,
                fmax=300_000,
            )
        
        # 比較 baseline+30 的 ai0,ai1,diff 時頻圖
        for i, r_leak in enumerate(eda_records_by_level[30][30:50], start=1):
            plot_record_dual_channel_spectrogram(
                r_leak,
                save_path=out_root / "feature_plots" / "dual_channel_spectrogram" / f"dual_channel_30ml_{i:02d}.png",
                fmin=40_000,
                fmax=300_000,
            )
        
        
    # ============================================================
    # Step 5.1. Step2a 候選頻帶視覺化（只用 dev）
    # ============================================================
    if SAVE_STEP5_BAND_ENERGY_PLOTS:
        for lv in ANALYSIS_LEAK_LEVELS:
            # 直接看 Step2a 選出的 leak-focused bands
            plot_band_energy_summary(
                eda_records_by_level[lv],
                eda_records_normal,
                leak_focused_bands,
                channel="ai1",
                label_leak=f"Baseline+{lv}",
                label_normal="Baseline",
                save_path=out_root / "feature_plots" / f"step2a_band_energy_leakfocused_vs_{lv}_ai1.png",
            )

            plot_band_energy_summary(
                eda_records_by_level[lv],
                eda_records_normal,
                leak_focused_bands,
                channel="diff",
                label_leak=f"Baseline+{lv}",
                label_normal="Baseline",
                save_path=out_root / "feature_plots" / f"step2a_band_energy_leakfocused_vs_{lv}_diff.png",
            )

            # 如果你也想看正式主頻帶（leak-focused + resonance）版本，可以保留這兩張
            plot_band_energy_summary(
                eda_records_by_level[lv],
                eda_records_normal,
                final_main_bands,
                channel="ai1",
                label_leak=f"Baseline+{lv}",
                label_normal="Baseline",
                save_path=out_root / "feature_plots" / f"step2b_band_energy_finalmain_vs_{lv}_ai1.png",
            )

            plot_band_energy_summary(
                eda_records_by_level[lv],
                eda_records_normal,
                final_main_bands,
                channel="diff",
                label_leak=f"Baseline+{lv}",
                label_normal="Baseline",
                save_path=out_root / "feature_plots" / f"step2b_band_energy_finalmain_vs_{lv}_diff.png",
            )


    # ============================================================
    # Step 5.2  PSD 視覺化（看大概關鍵頻段）
    # 目的：看 Baseline vs 30ml 的整體頻譜差異，作為 Step2a 結果的背景參考
    # ============================================================
    if SAVE_STEP5_PSD_PLOTS:
        for lv in ANALYSIS_LEAK_LEVELS:
            plot_cross_condition_same_channel_psd(
                eda_records_normal,
                eda_records_by_level[lv],
                channel="ai1",
                label_a="Baseline",
                label_b=f"Baseline+{lv}",
                save_path=out_root / "feature_plots" / f"psd_Baseline_vs_{lv}_ai1.png",
            )
        
            plot_cross_condition_same_channel_psd(
                eda_records_normal,
                eda_records_by_level[lv],
                channel="diff",
                label_a="Baseline",
                label_b=f"Baseline+{lv}",
                save_path=out_root / "feature_plots" / f"psd_Baseline_vs_{lv}_diff.png",
            )
    
        # 同一類別內 ai0 vs ai1 的 PSD
        plot_within_label_dual_channel_psd(
            eda_records_normal,
            label_name="Baseline",
            save_path=out_root / "feature_plots" / "normal_ai0_vs_ai1_psd.png",
        )
        
        for lv in ANALYSIS_LEAK_LEVELS:
            plot_within_label_dual_channel_psd(
                eda_records_by_level[lv],
                label_name=f"Baseline+{lv}",
                save_path=out_root / "feature_plots" / f"leak{lv}_ai0_vs_ai1_psd.png",
            )

    # ============================================================
    # Step 6. 整理 dev / holdout 的 by-file 輸入清單
    # ============================================================
    # 把 dev / holdout 整理成 file_specs 格式
    dev_records_all = dev_records_leak + dev_records_normal
    holdout_records_all = holdout_records_leak + holdout_records_normal
    
    # dev 和 holdout 都用完全相同的 band 設定 core_bands=final_main_bands , aux_bands=[] , low_bands=[]
    dev_file_specs = [
        {
            "filepath": r["filepath"],
            "record_id": r["record_id"],
            "label": r["label"],
        }
        for r in dev_records_all
    ]
    
    holdout_file_specs = [
        {
            "filepath": r["filepath"],
            "record_id": r["record_id"],
            "label": r["label"],
        }
        for r in holdout_records_all
    ]
    
    cache_dir = out_root / "feature_cache"
    
    # ============================================================
    # Step 7. by-file 正式特徵工程（使用 final_main_bands = leak_focused + resonance）
    # ============================================================
    # 針對 dev 資料做特徵工程
    features_dev = load_or_compute_features_by_file(
        cache_dir=cache_dir,
        file_specs=dev_file_specs,
        core_bands=final_main_bands,         # 主頻段（漏檢測+共振）
        aux_bands=offband_bands_for_rule,   # 輔助的 offband 頻段
        low_bands=[],
        split_name="dev",
        n_jobs=-2,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        target_levels=TARGET_LEVELS,
        feature_version=base_feature_version,
        max_points=max_points,
    )
    
    # 針對 holdout 資料做特徵工程
    features_holdout = load_or_compute_features_by_file(
        cache_dir=cache_dir,
        file_specs=holdout_file_specs,
        core_bands=final_main_bands,
        aux_bands=offband_bands_for_rule,
        low_bands=[],
        split_name="holdout",
        n_jobs=-2,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        target_levels=TARGET_LEVELS,
        feature_version=base_feature_version,
        max_points=max_points,
    )
    
    print(f"Dev feature rows: {len(features_dev)}")
    print(f"Holdout feature rows: {len(features_holdout)}")
    
    # 把回傳的 list 轉成 DataFrame, 先依 record_id 排序、同一個 record 內再依 t_start（視窗起始時間）
    df_dev = pd.DataFrame(features_dev).sort_values(["record_id", "t_start"]).reset_index(drop=True)
    df_holdout = pd.DataFrame(features_holdout).sort_values(["record_id", "t_start"]).reset_index(drop=True)
    
    # 檢查正式頻帶原始特徵覆蓋率
    audit_band_feature_coverage(df_dev, band_config)
    audit_band_feature_coverage(df_holdout, band_config)
    
    
    # ============================================================
    # Step 7.1. 把 metadata 接回 window-level 特徵表
    # ============================================================
    # 用 list comprehension 從原始 record 裡抽出四個 metadata 欄位（record_id、洩漏量、距離、壓力），組成一份 DataFrame, 依 record_id 去重
    record_meta_dev_df = pd.DataFrame([
        {
            "record_id": r["record_id"],
            "leak_quantity": r["meta"]["leak_quantity"],
            "distance_cm": int(round(r["meta"]["distance"] * 100)),
            "pressure": r["meta"]["pressure"],
        }
        for r in (dev_records_leak + dev_records_normal)]).drop_duplicates(subset=["record_id"])
    
    record_meta_holdout_df = pd.DataFrame([
        {
            "record_id": r["record_id"],
            "leak_quantity": r["meta"]["leak_quantity"],
            "distance_cm": int(round(r["meta"]["distance"] * 100)),
            "pressure": r["meta"]["pressure"],
        }
        for r in (holdout_records_leak + holdout_records_normal)]).drop_duplicates(subset=["record_id"])
    
    # 用 record_id 當 key，把 metadata 表 merge 回原本 window-level 的特徵表
    df_dev = df_dev.merge(record_meta_dev_df, on="record_id", how="left")
    df_holdout = df_holdout.merge(record_meta_holdout_df, on="record_id", how="left")
    
    # ============================================================
    # Step 7.2. 建立 offband / leak-focused / resonance 摘要特徵
    # ============================================================
    # 只有當最前面從 extraction_cfg 讀出來的 include_offband_summary 是 True 時才執行
    if include_offband_summary:
        # 產生 offband/leak_focus/resonance 的摘要與比值特徵
        df_dev = build_offband_summary_features(df_dev, band_config)
        df_holdout = build_offband_summary_features(df_holdout, band_config)
        
        # 檢查 offband summary coverage, 確認這些摘要欄位真的都建出來
        audit_offband_summary_coverage(df_dev)
        audit_offband_summary_coverage(df_holdout)
    
    # ============================================================
    # Step 7.3. 決定 Step2b 正式候選 raw base features
    # ============================================================
    # 從 df_dev 裡挑出「要進一步展開衍生特徵（rel/z/cusum/temporal）」的原始欄位清單，只針對 final_main_bands 這些頻段、且欄位後綴要符合 STEP2_EXPAND_SUFFIXES
    # 但正式擴充層（rel / z / cusum / temporal）先只對 E / RMS / EnvRMS 展開
    expand_base_feature_cols = select_step2_expand_base_features(
        df=df_dev,
        bands=final_main_bands,
        suffixes=STEP2_EXPAND_SUFFIXES,
    )
    # 印出總數，並列出前 20 個欄位名稱當抽樣預覽
    print("Step2 expand base features 數量 =", len(expand_base_feature_cols))
    for c in expand_base_feature_cols[:20]:
        print("  ", c)
    
    # ============================================================
    # Step 7.4 套用固定正常 baseline，建立 rel / z 特徵
    #
    # dev 與 holdout 使用同一份預先建立的固定 baseline。
    # 不使用 dev normal、holdout normal 或每筆資料前幾窗, 重新估計 baseline。
    # ============================================================
    # 對 dev、holdout 各自套用同一份固定 baseline
    df_dev = apply_fixed_normal_baseline(
        df=df_dev,
        feature_cols=expand_base_feature_cols,
        baseline_payload=fixed_baseline,
    )
    
    df_holdout = apply_fixed_normal_baseline(
        df=df_holdout,
        feature_cols=expand_base_feature_cols,
        baseline_payload=fixed_baseline,
    )
    
    expected_relative_features = [
        derived_name
        for feature_name in expand_base_feature_cols
        for derived_name in (
            f"rel_{feature_name}",
            f"z_{feature_name}",
        )
    ]
    
    # 用一個迴圈同時檢查 dev、holdout 兩份資料表
    # 對每一組,檢查前面算出來的衍生特徵是否真的都出現在對應的資料表裡,缺了就報錯,並在錯誤訊息裡指名是 dev 還是 holdout 出問題
    for dataset_name, dataset_df in [
        ("dev", df_dev),
        ("holdout", df_holdout),
    ]:
        missing = [
            feature_name
            for feature_name in expected_relative_features
            if feature_name not in dataset_df.columns
        ]
    
        if missing:
            raise ValueError(f"{dataset_name} 缺少固定 baseline 特徵：{missing}")
        
    # ============================================================
    # Step 7.5. 建立 CUSUM 特徵
    # ============================================================
    # 把每個原始欄位對應的 z_* 欄位名組成清單——CUSUM（累積和）特徵是建立在標準化後的 z-score 之上，所以需要先確定要用哪些 z_* 欄位當輸入。
    z_feature_cols = [
        f"z_{feature_name}"
        for feature_name in expand_base_feature_cols
    ]
    
    # 分別檢查 dev、holdout 兩邊是不是真的有這些 z_* 欄位
    missing_dev_z = [ c for c in z_feature_cols if c not in df_dev.columns]
    missing_holdout_z = [ c for c in z_feature_cols if c not in df_holdout.columns]
    if missing_dev_z or missing_holdout_z:
        raise ValueError(f"CUSUM 前置 z 特徵不完整：dev={missing_dev_z}, holdout={missing_holdout_z}")
    
    # 建立 CUSUM 特徵
    df_dev = add_cusum_features(
        df=df_dev,
        input_feature_cols=z_feature_cols,
        k=0.02,         # k=0.02 是 CUSUM 的漂移容忍參數
        decay=0.97,
    )
    
    df_holdout = add_cusum_features(
        df=df_holdout,
        input_feature_cols=z_feature_cols,
        k=0.02,
        decay=0.97,
    )
    
    # ============================================================
    # Step 7.6. 建立 Step2 正式 temporal features（window-level）
    # 目前 temporal 擴充先只對 leak-focused bands 展開，避免一開始把 resonance bands 也一起展開，造成特徵量與計算量過大
    # ============================================================
    # temporal 先只對 leak-focused bands 展開，避免一開始把 resonance 也一起展得太重
    temporal_base_feature_cols = select_step2_expand_base_features(
        df=df_dev,
        bands=leak_focused_bands,     # 只針對漏檢測頻段挑欄位，不含共振頻段
        suffixes=STEP2_EXPAND_SUFFIXES,
    )
    
    print("Step2 temporal base feature 數量 =", len(temporal_base_feature_cols))
    for c in temporal_base_feature_cols[:20]:
        print("  ", c)
    
    #  建立正式 temporal features
    df_dev = add_step2_temporal_features(
        df=df_dev,
        raw_feature_cols=temporal_base_feature_cols,
        group_col="record_id",
        time_col="t_ref",
    )
    
    df_holdout = add_step2_temporal_features(
        df=df_holdout,
        raw_feature_cols=temporal_base_feature_cols,
        group_col="record_id",
        time_col="t_ref",
    )
    '''
    temporal_feature_cols = [
        c for c in df_dev.columns
        if (
            c.startswith("delta_")
            or c.startswith("delta_prev3mean_")
            or c.startswith("delta_prev5mean_")
            or c.startswith("local_rel3_")
            or c.startswith("local_rel5_")
            or c.startswith("local_z3_")
            or c.startswith("local_z5_")
            or c.startswith("rolling_std3_")
            or c.startswith("rolling_std5_")
            or c.startswith("rolling_slope3_")
            or c.startswith("rolling_slope5_")
            or c.startswith("rolling_max_gap3_")
            or c.startswith("rolling_max_gap5_")
        )
    ]
    
    print("Step2 temporal feature 數量 =", len(temporal_feature_cols))
    '''

    # ============================================================
    # Step 8. 把 Step 7 算好的 window-level 特徵，壓成 record-level，然後只在 dev 資料上做特徵排名與視覺化（只在 dev 做）
    # ============================================================
    # 先把每一筆檔案的很多小窗特徵整理成「每筆檔案一列」 -> 再判斷哪些特徵最能分正常與洩漏 -> 再檢查這些特徵會不會其實只是吃距離 -> 最後把結果輸出成表格和圖
    
    # 整理 dev 的 metadata
    record_meta_dev_df = pd.DataFrame([
        {
            "record_id": r["record_id"],    # 這筆檔案的 ID
            "label": r["label"],     # 正常或洩漏類別
            "leak_quantity": r["meta"]["leak_quantity"],     # 洩漏量
            "distance_cm": int(round(r["meta"]["distance"] * 100)),    # 距離
            "pressure": r["meta"]["pressure"],    # 壓力
        }
        for r in (dev_records_leak + dev_records_normal)]).drop_duplicates(subset=["record_id"])
    
    # 把 window-level 特徵聚合成 record-level：同一個 record_id 底下所有 windows 用 median 壓成一列
    record_df_dev_all = build_record_level_feature_table(
        window_df=df_dev,
        qc_record_df=None,
        extra_meta_df=record_meta_dev_df,
        agg="median",   
    )
    
    # 整理 merge 後重複的 label 欄位
    if "label_x" in record_df_dev_all.columns:
        record_df_dev_all["label"] = record_df_dev_all["label_x"]
        drop_tmp_cols = [c for c in ["label_x", "label_y"] if c in record_df_dev_all.columns]
        record_df_dev_all = record_df_dev_all.drop(columns=drop_tmp_cols)
        
    # 看 dev 裡面各 leak quantity 的分布
    print("\n=== dev record-level leak_quantity 分布 ===")
    print(record_df_dev_all["leak_quantity"].value_counts().sort_index())

    # 只保留關注的 levels(TARGET_LEVELS = {0, 30})
    record_df_dev_levels = record_df_dev_all[record_df_dev_all["leak_quantity"].isin(TARGET_LEVELS)].copy()

    # 先做 baseline vs 30 的診斷:如果只看 baseline vs 30 ml，哪些特徵最有差？
    diag_30_df = diagnose_baseline_vs_30(
        record_df_dev_levels,
        out_dir=out_root / "feature_selection" / "baseline_vs_30_diagnostic",
        top_n=30,
    )
    
    # ------------------------------------------------------------
    # 9-1. 只按洩漏量做 ranking
    # 用途：看整體上哪些特徵最能反映 Baseline vs 30 的差異
    # 做多漏量特徵排名:會根據：normal vs 30 相鄰 level 差異, Spearman 單調趨勢, Kruskal-Wallis 綜合算出 score。
    # ------------------------------------------------------------
    rank_df_levels_raw = rank_features_for_leak_levels(
        record_df_dev_levels,
        target_levels=TARGET_LEVELS,
    )
    
    # ------------------------------------------------------------
    # 9-2. 固定距離做 ranking
    # 用途：檢查在每個 distance 下，normal vs leak 是否仍然有一致差異(分數越高 = 跨距離越穩定)
    # ------------------------------------------------------------
    rank_df_distance = rank_features_within_each_distance(
        record_df_dev_levels,
        target_levels=TARGET_LEVELS,
        min_samples=3,
    )
    
    # ------------------------------------------------------------
    # 9-3. 固定洩漏量做 ranking
    # 用途：檢查同一 leak level 下，不同 distance 對特徵影響有多大(分數越高 = 越容易受距離影響)
    # ------------------------------------------------------------
    rank_df_leaklevel = rank_features_within_each_leak_level(
        record_df_dev_levels,
        target_levels=tuple(ANALYSIS_LEAK_LEVELS),
        min_samples=3,
    )
    
    # ------------------------------------------------------------
    # 9-4. 合併三種 ranking 計算 final_score 設計邏輯：
    # + score                         -> 對 leak 量敏感
    # + distance_stability_score      -> 固定距離下也穩
    # - distance_sensitivity_score    -> 如果太吃距離，就扣分
    # ------------------------------------------------------------
    rank_df_final = rank_df_levels_raw.merge(
        rank_df_distance,
        on="feature",
        how="left"
    ).merge(
        rank_df_leaklevel,
        on="feature",
        how="left"
    )

    # 對 leak 敏感，加分 , 跨距離穩定，加分 , 太吃距離，扣分
    rank_df_final["final_score"] = (
        0.50 * rank_df_final["score"].fillna(0) +
        0.35 * rank_df_final["distance_stability_score"].fillna(0) -
        0.15 * rank_df_final["distance_sensitivity_score"].fillna(0)
    )
    
    # 至少要求大部分距離下 normal vs leak 方向一致
    rank_df_final = rank_df_final[rank_df_final["distance_sign_consistency"].fillna(0) >= 0.67].copy()

    # 觀察版：先不要刪高風險距離特徵(例如不能：某距離是 leak 比 normal 高, 另一距離卻變成 leak 比 normal 低)
    rank_df_final = rank_df_final.sort_values("final_score", ascending=False).reset_index(drop=True)
    
    # 去掉你已知的高風險距離敏感特徵
    #rank_df_final = rank_df_final[
        #~rank_df_final["feature"].isin(drop_distance_sensitive)
    #].sort_values("final_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------
    # 9-5. 存 ranking 結果
    # 分開存是為了之後可以回頭看：哪些特徵是 leak-sensitive , 哪些特徵是 distance-stable , 哪些特徵是 distance-sensitive
    # ------------------------------------------------------------
    # 哪些特徵對 leak 敏感
    rank_df_levels_raw.to_csv(      # EDA 用的特徵
        out_root / "feature_selection" / "rank_features_multilevel.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 哪些跨距離穩
    rank_df_distance.to_csv(
        out_root / "feature_selection" / "rank_features_by_distance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 哪些吃距離
    rank_df_leaklevel.to_csv(
        out_root / "feature_selection" / "rank_features_within_leaklevel.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 綜合後最後保留哪些
    rank_df_final.to_csv(
        out_root / "feature_selection" / "rank_features_final_combined.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 刪掉距離高風險特徵(建模用)
    rank_df_levels_model = rank_df_levels_raw[
        ~rank_df_levels_raw["feature"].isin(drop_distance_sensitive)
    ].reset_index(drop=True)
    
    # =====================================
    # 繪圖
    # =====================================
    # -- Boxplot --  看這些 top features 在 0 / 30 的分布是不是有規律
    # robust 版：不看極端值
    if SAVE_STEP9_RAW_BOXPLOTS:
        plot_top_features_multilevel_boxplot(
            record_df_dev_levels,
            rank_df_levels_raw,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=TOP_N_FEATURES_TO_PLOT,
            save_path=out_root / "feature_plots" / "top_features_boxplot_multilevel_robust.png",
            showfliers=False,
            robust_ylim=True,
            lower_pct=2,
            upper_pct=98,
        )
        
        # 含離群值
        plot_top_features_multilevel_boxplot(
            record_df_dev_levels,
            rank_df_levels_raw,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=TOP_N_FEATURES_TO_PLOT,
            save_path=out_root / "feature_plots" / "top_features_boxplot_multilevel_full.png",
            showfliers=True,
            robust_ylim=False,
        )
    
    # 跨距離的比較圖
    if SAVE_STEP9_DISTANCE_BOXPLOTS:
        plot_top_features_multilevel_boxplot(
            record_df_dev_levels,
            rank_df_distance,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=TOP_N_FEATURES_TO_PLOT,
            save_path=out_root / "feature_plots" / "top_features_selected_by_distance_stability_pooled_view.png",
            showfliers=False,
            robust_ylim=True,
            lower_pct=2,
            upper_pct=98,
        )
    
    # 特徵會不會隨距離改變太多
    if SAVE_STEP9_DISTANCE_EFFECT_PLOTS:
        plot_top_features_distance_effect_lines(
            record_df_dev_levels,
            rank_df_distance,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=TOP_N_FEATURES_TO_PLOT,
            out_dir=out_root / "feature_plots" / "distance_effect_lines",
        )
        
    # 把「最吃距離的前 10 個特徵」抓出來畫
    if SAVE_STEP9_DISTANCE_SENSITIVE_PLOTS:
        distance_sensitive_features = rank_df_leaklevel["feature"].head(10).tolist()
    
        plot_selected_feature_lines_by_leak_and_distance(
            record_df_dev_levels,
            features_to_plot=distance_sensitive_features,
            out_dir=out_root / "feature_plots" / "distance_sensitive_feature_lines",
            label_filter=None,
        )
    if SAVE_STEP9_FINAL_SUMMARY_PLOT:
        plot_top_features_multilevel_boxplot(
            record_df_dev_levels,
            rank_df_final,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=TOP_N_FEATURES_TO_PLOT,
            save_path=out_root / "feature_plots" / "top_features_boxplot_multilevel_final_summary.png",
            showfliers=False,
            robust_ylim=True,
            lower_pct=2,
            upper_pct=98,
        )

    # -- Trend Lines --
    #    看這些特徵是不是隨漏量增加而上升或下降
    if SAVE_MULTILEVEL_LINE_PLOTS:
        plot_top_features_multilevel_lines(
            record_df_dev_levels,
            rank_df_levels_raw,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=10,
            out_dir=out_root / "feature_plots" / "trend_lines_multilevel",
        )

    # -- pairwise 比較 --
    #    看像 Baseline vs 30 這種局部比較是不是也分得開
    if SAVE_PAIRWISE_BOXPLOTS:
        plot_top_features_pairwise_boxplot(
            record_df_dev_levels,
            rank_df_levels_raw,
            level_a=0,
            level_b=30,
            label_a="Baseline",
            label_b="Baseline+30ml",
            top_n=TOP_N_FEATURES_TO_PLOT,
            save_path=out_root / "feature_plots" / "top_features_boxplot_Baseline_vs_Baseline+30.png",
        )

    # 正常與不同洩漏組差異
    if SAVE_ALL_PAIRWISE_BOXPLOTS:
        pairwise_out_dir = out_root / "feature_plots" / "all_pairwise_record_level_boxplots"
    
        plot_all_features_pairwise_boxplots(
            record_df_dev_all,
            level_a=0,
            level_b=30,
            out_dir=pairwise_out_dir,
            prefix="Baseline_vs_Baseline+30"
        )
    
    
        plot_all_features_pairwise_boxplots(
            record_df_dev_all,
            level_a=0,
            level_b=1,
            level_col="label",
            out_dir=out_root / "feature_plots" / "baseline_vs_selected_leaks_boxplots",
            prefix="Baseline_vs_SelectedLeaks"
        )

    
    # ============================================================
    # Step 10. 用 ranking 結果挑候選特徵 + 模型重要性加權 + 去共線性
    # ============================================================
    # 建立模型禁止使用的特徵清單 
    # 從完整 window-level 特徵表中找出所有：
    # 1. Offband 原始特徵
    # 2. Offband summary
    # 3. Offband ratio
    # 4. 其他 Veto 專用特徵
    # 這些欄位保留給後處理規則，但禁止進模型。
    rule_only_features = {
        feature_name for feature_name in df_dev.columns
        if is_rule_only_feature(feature_name, offband_bands_for_rule,)
    }
    
    # 合併人工指定的距離高風險特徵。
    banned_model_features = ( set(drop_distance_sensitive) | rule_only_features)
    
    print("\n=== 模型禁用特徵檢查 ===")
    print("Rule-only／Offband 特徵數量：", len(rule_only_features),)
    print("模型總禁用特徵數量：", len(banned_model_features),)
    
    # rank_df_final 可保留完整排名供分析，但真正的模型候選池必須另外排除禁用特徵。
    rank_df_model = rank_df_final[ ~rank_df_final["feature"].isin(banned_model_features)].copy().reset_index(drop=True)
    
    if rank_df_model.empty:
        raise ValueError("排除 Offband、Veto 與距離高風險特徵後，沒有可供模型使用的候選特徵")
    
    rank_df_model.to_csv(
        out_root / "feature_selection" / "rank_features_model_eligible.csv", index=False, encoding="utf-8-sig",)
    
    candidate_features = rank_df_model["feature"].head(TOP_K_CANDIDATES).tolist()
    X_dev_candidate_df = df_dev[candidate_features].copy()
    
    # 統計分數(原本就有)
    stat_score_dict = dict(zip(rank_df_model["feature"], rank_df_model["final_score"]))
    
    # 模型重要性(新增)：對候選特徵訓練 RandomForest，取得 feature_importances_
    model_importance_dict = compute_model_feature_importance(
        df_dev=df_dev,
        candidate_features=candidate_features,
        random_state=42,
    )
    
    # 合併統計分數與模型重要性
    combined_score_dict, stat_norm_dict, model_norm_dict = combine_statistical_and_model_scores(
        statistical_score_dict=stat_score_dict,
        model_importance_dict=model_importance_dict,
        stat_weight=0.5,
        model_weight=0.5,
    )
    
    # 輸出合併分數表，方便檢查統計 vs 模型兩邊意見是否一致
    combined_score_df = pd.DataFrame({
        "feature": candidate_features,
        "stat_score_norm": [stat_norm_dict[f] for f in candidate_features],
        "model_importance_norm": [model_norm_dict[f] for f in candidate_features],
        "combined_score": [combined_score_dict[f] for f in candidate_features],
    }).sort_values("combined_score", ascending=False).reset_index(drop=True)
    
    combined_score_df.to_csv(
        out_root / "feature_selection" / "combined_statistical_model_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )
    
    print("\n=== 統計分數 vs 模型重要性（合併後排名前 20）===")
    print(combined_score_df.head(20).to_string(index=False))
    
    # 用合併後分數做共線性篩選（原本這裡只用 rank_score_dict 統計分數）
    X_dev_reduced, dropped_features, corr_pairs = remove_collinear_features(
        X_dev_candidate_df,
        combined_score_dict,
        threshold=0.95,
        method="pearson",
        verbose=True,
        save_path=out_root / "feature_selection"
    )
    top_feature_names = X_dev_reduced.columns.tolist()
    
    # 如果共線性篩選後特徵太少，補回 combined score 高的特徵
    if len(top_feature_names) < MIN_FINAL_FEATURES:
        supplemental_features = [
            f for f in combined_score_df["feature"].tolist()
            if f not in top_feature_names
        ]
    
        n_need = MIN_FINAL_FEATURES - len(top_feature_names)
        top_feature_names = top_feature_names + supplemental_features[:n_need]
    
        print(
            f"⚠️ 共線性篩選後只剩 {len(X_dev_reduced.columns)} 個特徵，"
            f"已補回到 {len(top_feature_names)} 個"
        )
    
    # 重新依照 top_feature_names 建立後續要用的資料
    X_dev_reduced = X_dev_candidate_df[top_feature_names].copy()
    
    
    # ============================================================
    # 最終模型特徵 Rule-only 防呆
    # ============================================================
    selected_rule_only_features = [
        feature_name
        for feature_name in top_feature_names
        if is_rule_only_feature(feature_name,offband_bands_for_rule,)]
    
    if selected_rule_only_features:
        raise ValueError(f"最終模型特徵包含禁止進模型的 Offband／Veto 欄位：{selected_rule_only_features}")
    
    print("✅ 最終模型特徵不包含 Offband／Veto 規則欄位")
        
    # ============================================================
    # Step 11. 看最終保留特徵的分布長相 heatmap（dev, record-level）
    # ============================================================
    # 把 top_feature_names 聚合成 record-level
    record_df_dev_selected = build_record_level_feature_table(
        window_df=df_dev,
        qc_record_df=None,
        feature_cols=top_feature_names,
        extra_meta_df=record_meta_dev_df,
        agg="median",    # 代表同一筆 record 的每個特徵，仍然用中位數當代表值
    )
    # 整理 merge 後重複的 label 欄位
    if "label_x" in record_df_dev_selected.columns:
        record_df_dev_selected["label"] = record_df_dev_selected["label_x"]
        drop_tmp_cols = [c for c in ["label_x", "label_y"] if c in record_df_dev_selected.columns]
        record_df_dev_selected = record_df_dev_selected.drop(columns=drop_tmp_cols)
    
    # 畫最終保留特徵的 heatmap(看哪些特徵在 Baseline vs 30 真的有穩定差異)
    # 畫:列 -> feature, 欄 -> Baseline / Baseline+30  × distance, 值 -> feature-wise robust z-score
    if SAVE_STEP11_SELECTED_FEATURE_HEATMAP:
        plot_selected_feature_heatmap_by_condition(
            record_df_dev_selected,
            selected_features=top_feature_names,
            out_path=out_root / "feature_selection" / "selected_features_condition_heatmap_dev_micro.png",
            label_filter=None,
        )
        
    # 畫最終保留特徵的線圖
    if SAVE_STEP11_SELECTED_FEATURE_LINE_PLOTS:
        plot_selected_feature_lines_by_leak_and_distance(
            record_df_dev_selected,
            features_to_plot=top_feature_names[:10],
            out_dir=out_root / "feature_selection" / "selected_feature_lines_dev_micro",
            label_filter=None,
        )
        
        
    # ============================================================
    # Step 12. 最終保留特徵的距離泛化能力圖
    # ============================================================
    # 建一張最終特徵的距離摘要表(最後留下來的每個特徵，距離穩定性評估分數是多少)
    final_selected_distance_df = (
        pd.DataFrame({"feature": top_feature_names})
        .merge(
            rank_df_final[[
                "feature",
                "final_score",
                "distance_stability_score",
                "distance_sensitivity_score",
                "distance_sign_consistency",
            ]],
            on="feature",
            how="left"
        )
    )
    # 摘要表存成 CSV
    final_selected_distance_df.to_csv(
        out_root / "feature_selection" / "final_selected_feature_distance_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    # 畫最終特徵的 distance effect lines
    if SAVE_STEP12_FINAL_SELECTED_DISTANCE_PLOTS:
        plot_top_features_distance_effect_lines(
            record_df_dev_levels,
            final_selected_distance_df,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=len(final_selected_distance_df),
            out_dir=out_root / "feature_plots" / "final_selected_distance_effect_lines",
        )
        # 畫最終特徵的 leak / distance 線圖
        plot_selected_feature_lines_by_leak_and_distance(
            record_df_dev_selected,
            features_to_plot=top_feature_names,
            out_dir=out_root / "feature_plots" / "final_selected_feature_lines",
            label_filter=None,
        )

    # ============================================================
    # Step 13. 準備模型資料（binary: normal vs all leak levels）
    # ============================================================
    # 從 df_dev、df_holdout 裡面，只抽出最後保留的特徵欄位
    # 準備：訓練特徵, 測試特徵, 標籤, group id
    X_dev_final = df_dev[top_feature_names].values
    X_holdout_final = df_holdout[top_feature_names].values

    y_dev = df_dev["label"].values
    y_holdout = df_holdout["label"].values
    groups_dev = df_dev["record_id"].values   # 同一個 record_id 的 windows 必須被分在同一 fold

    # ============================================================
    # Step 14. CV + holdout evaluation
    # ============================================================
    from sklearn.pipeline import Pipeline
    # 先建立 Pipeline
    pipeline_final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 決定 group CV 要切幾折
    n_groups = len(np.unique(groups_dev))
    n_splits = min(5, n_groups)

    # 建立 StratifiedGroupKFold 做 group CV
    gkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )
    
    # 做 reference CV
    scores = cross_val_score(
        pipeline_final,
        X_dev_final,
        y_dev,
        groups=groups_dev,
        cv=gkf,
        scoring="f1"
    )

    # 因為這個 reference CV 雖然有 group split，但它用的特徵 top_feature_names 是你先用整個 dev 資料選出來的，分數通常會比較樂觀。
    print(
        f"\nReference CV F1 (optimistic reference only): "
        f"{scores.mean():.3f} ± {scores.std():.3f}"
    )
    
    # 確認沒有明顯 leakage 的關鍵
    # 更嚴謹的 group CV:每個 fold 內自己做 ranking -> 每個 fold 內自己去共線性 -> 再評估
    foldwise_scores, foldwise_feature_info, feature_freq_df = run_group_cv_with_foldwise_feature_selection(
        df_dev=df_dev,
        record_meta_dev_df=record_meta_dev_df,
        target_levels=TARGET_LEVELS,
        top_k_candidates=TOP_K_CANDIDATES,
        corr_threshold=0.95,
        random_state=42,
        banned_features=sorted(banned_model_features),    # 每個 Fold 內部做特徵選擇時，也必須排除 Offband／Veto 專用特徵，避免 CV 流程與正式模型流程使用不同規則。
    )

    print(
        f"Fold-wise FS CV F1 (primary): "
        f"{foldwise_scores.mean():.3f} ± {foldwise_scores.std():.3f}"
    )
    
    # 輸出 feature stability 表
    feature_freq_df.to_csv(
        out_root / "feature_selection" / "foldwise_feature_stability.csv",
        index=False,
        encoding="utf-8-sig"
    )
    
    
    # ============================================================
    # Step 15. LODO：Leave-one-distance-out validation 看跨距離泛化
    # ============================================================
    print("LODO 使用全 dev 選出的固定特徵，結果作為輔助參考，不視為無洩漏主指標")
    # 跑 Leave-one-distance-out:每次留一個 distance_cm 當測試組, 其他距離當訓練組
    lodo_summary_df, lodo_window_pred_df, lodo_file_pred_df = (
        run_leave_one_distance_out_validation(
            df_dev=df_dev,
            feature_names=top_feature_names,
            base_pipeline=pipeline_final,
            threshold=0.5,
        )
    )
    # 每個距離 fold 的摘要結果
    lodo_summary_df.to_csv(
        out_root / "feature_selection" / "leave_one_distance_out_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    # window-level 的預測結果
    lodo_window_pred_df.to_csv(
        out_root / "feature_selection" / "leave_one_distance_out_window_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )
    # file-level 的預測結果
    lodo_file_pred_df.to_csv(
        out_root / "feature_selection" / "leave_one_distance_out_file_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    
    # 正式 fit 全部 dev
    pipeline_final.fit(X_dev_final, y_dev)

    print("\n=== Holdout 評估 ===")
    print(classification_report(y_holdout, pipeline_final.predict(X_holdout_final)))
    
    # 取出 holdout 預測值
    y_holdout_pred = pipeline_final.predict(X_holdout_final)
    

    # ============================================================
    # Step 16. 看 holdout 在不同距離下表現 Holdout by distance 把 holdout prediction 拿去按 distance_cm 分組分析
    # ============================================================
    # 看 holdout 在不同距離下表現如何
    evaluate_holdout_by_group(
        df_holdout=df_holdout,
        y_true=y_holdout,
        y_pred=y_holdout_pred,
        group_col="distance_cm",
    )
    
    
    # ============================================================
    # Step 17. 看 holdout 在不同漏量下表現 Holdout by leak quantity 把 holdout prediction 拿去按 leak quantity 分組分析
    # ============================================================
    # 看 holdout 在不同洩漏量下表現如何
    evaluate_holdout_by_group(
        df_holdout=df_holdout,
        y_true=y_holdout,
        y_pred=y_holdout_pred,
        group_col="leak_quantity",
    )
    
    
    # ============================================================
    # Step 18. Permutation test
    # ============================================================
    # 檢查：如果把 group-level label 打亂, 模型分數會不會還一樣高
    # 保持資料與 group 結構 -> 但把 label 打亂很多次 -> 每次重做 group CV -> 收集一堆「亂標籤下的 F1」
    perm_scores = run_permutation_test_group_cv(
        X=X_dev_final,
        y=y_dev,
        groups=groups_dev,
        pipeline=pipeline_final,
        cv=gkf,
        scoring="f1",
        n_iter=100,
        random_state=42,
    )
    # 取真實分數
    real_cv_mean_ref = scores.mean()
    real_cv_mean_foldwise = foldwise_scores.mean()
    
    # 計算 permutation p-value(看在亂標籤的情況下，有多少次分數居然還能大於等於你的真實分數。)
    p_value_ref = (np.sum(perm_scores >= real_cv_mean_ref) + 1) / (len(perm_scores) + 1)
    
    print(f"\nReference CV mean F1 = {real_cv_mean_ref:.4f}")
    print(f"Fold-wise CV mean F1 = {real_cv_mean_foldwise:.4f}")
    print(f"Permutation p-value (reference CV) = {p_value_ref:.4f}")
    print("Fold-wise CV permutation test: 目前尚未實作『每次 permutation 都重做 fold-wise 特徵選擇』，因此暫不報告 p-value")

    # ============================================================
    # 檢查特徵 selected feature alignment
    # ============================================================
    feature_config_audit = {
        "selected_feature_names": top_feature_names
    }
    
    audit_selected_feature_alignment(df_dev, feature_config_audit)
    audit_selected_feature_alignment(df_holdout, feature_config_audit)

    # ============================================================
    # Step 19. 匯出 selected features & full features
    # ============================================================
    # 先輸出 full feature 版
    df_dev.to_csv(out_root / "features_dev_full_micro_DC.csv", index=False)
    df_holdout.to_csv(out_root / "features_holdout_full_micro_DC.csv", index=False)
        
    # 再輸出 select feature 版
    df_dev_selected = df_dev[
        ["filename", "record_id", "t_start", "label", "leak_quantity", "distance_cm", "pressure"] + top_feature_names
    ]
    
    df_holdout_selected = df_holdout[
        ["filename", "record_id", "t_start", "label", "leak_quantity", "distance_cm", "pressure"] + top_feature_names
    ]

    # 存 selected feature 表
    df_dev_selected.to_csv(out_root / "features_dev_selected_micro_DC.csv", index=False)
    df_holdout_selected.to_csv(out_root / "features_holdout_selected_micro_DC.csv", index=False)

    # 存最終特徵名稱清單
    pd.Series(top_feature_names).to_csv(
        out_root / "selected_feature_names_micro.csv",
        index=False,
        header=["feature_name"]
    )

    # 存 dev / holdout 的 record id
    pd.Series(sorted(dev_record_ids)).to_csv(
        out_root / "dev_record_ids_micro.csv",
        index=False,
        header=["record_id"]
    )
    pd.Series(sorted(holdout_record_ids)).to_csv(
        out_root / "holdout_record_ids_micro.csv",
        index=False,
        header=["record_id"]
    )
    
    # 輸出特徵設定檔 JSON
    save_feature_config_json(
        output_json=out_root / "feature_config_micro.json",
        selected_feature_names=top_feature_names,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        max_points=max_points,
        target_levels=TARGET_LEVELS,
        top_k_candidates=TOP_K_CANDIDATES,
        corr_threshold=0.95,
        temporal_band_names=TEMPORAL_BAND_NAMES,
        
        baseline_feature_names=expand_base_feature_cols,
        cusum_input_feature_names=z_feature_cols,
        temporal_base_feature_names=temporal_base_feature_cols,
        
        band_config_path=BAND_CONFIG_PATH,
        band_config_hash=band_config_hash,
        feature_extraction_config_path=(FEATURE_EXTRACTION_CONFIG_PATH),
        feature_extraction_config_hash=(feature_extraction_config_hash),
        baseline_path=FIXED_BASELINE_PATH,
        baseline_payload=fixed_baseline,

        source_note="step2_micro_leak_feature_selection_final",
        score_col="leak_score",
        pred_col="pred_label",
        threshold=0.5,
        min_consecutive_windows=3,
        min_leak_duration_sec=0.4,
        disturbance_ratio_col="offband_to_leak_focus_E_ratio",
        disturbance_ratio_threshold=None,
        baseline_mode="fixed_normal_baseline",
        include_offband_summary=include_offband_summary,
        cusum_k=0.02,
        cusum_decay=0.97,
    )
    
    print(f"\n✅ Micro-leak 流程完成，結果已輸出到 {out_root}")














