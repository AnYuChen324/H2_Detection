#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2：頻帶探索與正式特徵選擇流程
================================

本程式使用純正常與純洩漏 TDMS，提供兩條互斥流程：

A. Step2a：候選洩漏頻帶探索（FORCE_BAND_DISCOVERY = True）
B. Step2b：正式特徵工程、特徵選擇與驗證（FORCE_BAND_DISCOVERY = False）

本程式不建立固定正常 Baseline。
Discovery 與 Final Baseline 必須由 build_site_baseline_v2.py 預先建立。


一、資料管理與隔離原則
----------------------

資料分成四種用途，彼此不可重複：

1. Baseline calibration normal：只用來建立固定正常 Baseline。
2. Step2 dev：頻帶探索、特徵排名、模型重要性、特徵選擇、交叉驗證。
3. Step2 holdout：不參與任何選擇，只在最後評估一次，不拿來調參。
4. 保留推論日（RESERVED_TEST_DATES）：Step2 完全不讀，留給最終模型推論。

進入 dev / holdout 切分前，依序做：

1. Hard QC：NaN / Inf、全零、RMS 過低 → 直接排除。
2. 重複檔案檢查（DUPLICATE_POLICY）：
   - "error"：發現內容相同的檔案就停止（新資料用）
   - "drop_conflicts"：同一份內容被標成不同 label 時整組排除（舊資料流程測試用）
3. 從檔名還原錄音場次（session_id）：同條件下相鄰檔案間隔超過 SESSION_GAP_SEC 視為新場次。
4. 排除保留推論日。
5. 日期 × 標籤檢查：若沒有任何一天同時錄了正常與洩漏，會印出警告，
   代表選出的頻帶與特徵可能是在分辨「哪天錄的」，結果只能當流程測試。

dev / holdout 以錄音場次為單位切分（每個「label × 距離」各自切）：

- 場次數 ≥ 3：整個場次給 holdout，同一場次不會被拆開。
- 場次數 < 3：每個場次各自依時間切，後段給 holdout，中間丟掉 HOLDOUT_GAP_FILES 支當緩衝。

目前限制：實驗室資料不足，Baseline 與 Step2 dev / holdout 共用同一批 leak0 正常資料
（STRICT_BASELINE_ISOLATION = False），holdout 的正常誤報率偏樂觀；
且舊資料正常與洩漏不同天錄製，頻帶與特徵排名會混入日期差異。


二、支線 A：Step2a 頻帶探索
---------------------------

啟動條件：FORCE_BAND_DISCOVERY = True，或 band_config_step2.json 不存在。

必要輸入：

- 純正常與純洩漏 TDMS
- band_config_step2_discovery.json（98 個掃描頻帶：每 10 kHz 一段、每 5 kHz 滑一次）
- discovery_baseline.json
- feature_extraction_config.json
- approved_resonance_bands.json（共振程式核准的共振頻帶）
- shared 的 OFFBAND_BANDS（抗干擾參照頻帶，全流程唯一定義）

執行流程（只使用 dev）：

1. 載入掃描頻帶，驗證 discovery baseline 的設定指紋與資料隔離。
2. 對全部掃描頻帶計算 window-level 原始特徵（有快取就直接讀）。
3. 只用原始特徵排名：每個頻帶 × 通道（ai0 / ai1 / diff）× 指標（STEP2_EXPAND_SUFFIXES）。
   rel_ / z_ 只是平移縮放，區分力和原始值完全相同，所以不納入排名。
4. 聚合成 record-level median，每個特徵算正常 vs 洩漏的 |Cohen's d|。
5. 彙整成頻帶分數：
   - 證據家族 = 通道（ai0 / ai1 / diff）
   - 一個通道的 |d| ≥ SUPPORT_MIN_IMPORTANCE 才算「支持」
   - 至少 MIN_FEATURE_FAMILY_COUNT 個通道支持才保留
   - band_score = 0.6 × 最強特徵 + 0.4 × 各通道平均
6. （BUILD_CROSS_WINDOW_FEATURES = True 時）對前 PRELIMINARY_BANDS_TO_KEEP 名
   加上跨窗特徵再排名一次；預設關閉，直接用第一輪排名。
7. 依分數由高到低挑選 TOP_BANDS_TO_KEEP 個候選洩漏頻帶，依序排除：
   - 上界超過 MAX_BAND_FREQ_HZ（接近奈奎斯特頻率）
   - 壓到 offband
   - 壓到共振頻帶（EXCLUDE_RESONANCE_OVERLAP = True 時；否則合併時共振頻帶會消失）
   - 和已選入的頻帶重疊（相鄰掃描頻帶重疊一半，只留分數高的）
8. 輸出候選設定檔（offband 自動帶入）後停止程式。

自動警告：

- 排名前 TOP_BANDS_TO_KEEP 的頻帶因 offband 或共振被排除 → 可能要檢查 offband / 共振設定
- 最高 |Cohen's d| > 3 → 對微小洩漏而言異常大，可能是日期或安裝條件差異

主要輸出：

- feature_importance_first_pass.csv：每個特徵的區分力
- band_importance_first_pass.csv：通過支持門檻的頻帶分數
- band_importance_summary.csv：最終頻帶排名，含 rank、selection_status（選入 / 排除原因）
- step2a_record_features_full.csv
- leak_candidate_bands.json
- band_config_step2_candidate.json


三、人工頻帶定案
----------------

Step2a 完成後必須人工處理：

1. 檢查 band_importance_summary.csv：
   - selection_status 為「選入」的頻帶
   - supporting_families 包含 ai1 與 diff 的頻帶最符合「洩漏在近端」
   - matched_resonance_bands：是否落在共振區
2. 確認 band_config_step2_candidate.json 的內容（offband 與共振頻帶已自動帶入）。
3. 將 schema_status 改成 "final"，另存為 band_config_step2.json。
4. 執行 build_site_baseline_v2.py，用正式頻帶建立 final baseline（site_baseline.json）。
5. 將 FORCE_BAND_DISCOVERY 改成 False，重新執行本程式進入 Step2b。

若要修改 offband，請改 shared 的 OFFBAND_BANDS（不要直接改設定檔），再重跑 Step2a。
offband 的選擇標準：對洩漏不敏感（Step2a 排名尾段）且對干擾敏感（共振程式頻譜的敲擊升幅明顯）。


四、支線 B：Step2b 正式特徵選擇
-------------------------------

啟動條件：FORCE_BAND_DISCOVERY = False，且 band_config_step2.json 存在並為 final。

必要輸入：

- band_config_step2.json
- feature_extraction_config.json
- site_baseline.json
- approved_resonance_bands.json（檢查 band config 的共振頻帶是否與正式版一致）
- 純正常與純洩漏 TDMS

執行流程：

1. 資料管理與切分（同第一節）。
2. 載入並驗證正式 band config、共振頻帶一致性、固定 baseline 指紋與資料隔離。
3. （RUN_SOFT_QC = True 時）dev / holdout 分開做 Soft QC，只輸出報表。
4. 對 final_main_bands（洩漏 + 共振）計算原始特徵；offband 一起算，只給規則用。
5. 建立 offband / leak_focus / resonance 摘要（只給規則用）。
6. 對頻帶的 E / RMS / EnvRMS 套用固定 baseline，建立 rel_ / z_。
7. （BUILD_CROSS_WINDOW_FEATURES = True 時）建立跨窗特徵，只供分析，不進模型。
8. 診斷與抗干擾假設檢查（洩漏 ÷ 正常的 offband 比值，各距離分開看）。
9. 建立可進模型的特徵清單（見第六節）。
10. 正式選特徵 select_model_features（見第七節）。
11. 驗證：reference CV、fold-wise CV、LODO、holdout（見第七節）。
12. 輸出特徵表與特徵契約。


五、固定 Baseline 原則
----------------------

Dev 與 holdout 使用同一份預先建立的固定正常 Baseline。

禁止：

- 使用 dev normal 或 holdout normal 重新 fit Baseline
- 使用每筆資料前幾個 window 建立 Baseline
- 缺值時使用 mean = 0、std = 1
- 使用當下資料作為 fallback

定義：

    rel_x = x - baseline_mean

    z_x = (x - baseline_mean) / baseline_std


六、模型特徵與規則特徵
----------------------

可進模型的特徵：

- final_main_bands（洩漏 + 共振）的原始特徵與 rel_ / z_
- 單一視窗內的時間結構特徵（temporal_band_names 指定的頻帶，例如 residual cusum_peak）
- 且 runtime 特徵抽取器算得出來（validate_runtime_extractable_features）

不可進模型的特徵（保留在完整特徵表，原因記錄在 model_excluded_features.csv）：

- rule_only：offband 原始特徵、offband 摘要與比值、leak_focus / resonance 群組摘要、
  計算時混入 offband 能量的特徵（high_low_ratio）→ 只給抗干擾規則
- cross_window：delta_、local_、rolling_、cusum_pos_、cusum_neg_（MODEL_EXCLUDED_PREFIXES）
  → 離線每支檔案只有 7 個視窗，和現場連續資料流的數值分布對不上；
    時間特性改由後規則處理（看模型分數是否持續偏高）
- distance_risk：DROP_DISTANCE_SENSITIVE 人工指定的距離高風險特徵
- not_runtime：runtime 特徵抽取器算不出來的特徵


七、選特徵與驗證
----------------

select_model_features()（正式流程、fold-wise CV、LODO 共用同一個函式）：

1. 每支檔案壓成一列（7 個視窗取中位數）。
2. 洩漏區分力：score = |Cohen's d|（多種洩漏量時以最小洩漏量為主）。
3. 跨距離穩定性：每個距離各自比較正常 vs 洩漏。
4. 跨天穩定性：只有「同一天同時有正常和洩漏」時才納入。
5. 綜合分數：
   - 有跨天資料：0.50 × 區分力 + 0.25 × 跨距離 + 0.25 × 跨天
   - 沒有跨天資料：0.60 × 區分力 + 0.40 × 跨距離
   （不扣距離敏感度：洩漏聲隨距離衰減是物理現象，只作報表參考）
6. 方向一致性：各距離（各天）至少 MIN_SIGN_CONSISTENCY 方向一致。
7. 取前 TOP_K_CANDIDATES 名 → 去共線性（Spearman，門檻 CORR_THRESHOLD，分數高者優先保留）。
8. 模型重要性：打亂測試法（permutation importance），以場次分組，在沒看過的場次上評估。
9. 統計分數與模型重要性各轉成排名百分比後各占一半，決定最終排序。
   不補回被去共線性刪掉的特徵。

驗證：

- Reference CV：用全部 dev 選出的特徵做場次分組 CV，分數偏樂觀，只當對照。
- Fold-wise CV（主要指標）：每折只用訓練折重新跑 select_model_features。
  至少 3 天且每天都有正常和洩漏 → 每折留整天；否則 → 每折留整個場次。
- LODO：每次留一個距離當驗證，每輪只用訓練距離重新選特徵。
- Holdout：用全部 dev 訓練，評估一次；依距離、洩漏量、場次分組，
  輸出誤報率、偵測率、F1、AUC，並標出最差的組別。
- 打亂標籤檢定（RUN_PERMUTATION_TEST）：預設關閉。特徵是用真實標籤選的，
  結果幾乎一定「顯著」，參考價值有限。

指標以 AUC 為主（不受分類門檻影響），F1 為輔。


八、Feature execution plan（feature_config_micro.json，schema_version 3）
-----------------------------------------------------------------------

執行計畫從最終特徵名單「反推」，只列下游真正需要算的東西：

- raw_required_features：每個視窗要先算的原始特徵（輸出前檢查 runtime 算得出來）
- baseline_features：要套固定 baseline 產生 rel_ / z_ 的原始特徵
- cusum_input_features：要做跨窗 CUSUM 的 z_ 欄位（跨窗特徵不進模型時為空）
- temporal_base_features：要做跨窗 delta / local / rolling 的原始特徵（同上，通常為空）
- final_keep_features：最終給模型的特徵

另外記錄上游指紋（band config、特徵萃取設定、baseline）與資料來源（data_info：
日期、場次數、切分方式、保留推論日、各項驗證分數）。

feature_config_micro.json 內的 threshold 與連續窗參數只是初始值。
正式模型 threshold、Two-stage 與 Veto 規則由後續：

- metadata.json
- postprocess_config.json
- runtime_feature_schema.json

完成定案。


九、正式輸出
------------

輸出資料夾（FINAL_OUTPUT_ROOT）：

- feature_config_micro.json：特徵契約
- selected_feature_names_micro.csv
- features_dev_full_micro_DC.csv / features_holdout_full_micro_DC.csv
- features_dev_selected_micro_DC.csv / features_holdout_selected_micro_DC.csv
- dev_record_ids_micro.csv / holdout_record_ids_micro.csv

feature_selection/：

- model_excluded_features.csv：不能進模型的特徵與原因
- rank_features_model_eligible.csv：通過方向一致性的特徵與各項分數
- final_selected_features.csv：最終特徵的分數明細
- corr_dropped_*.csv / corr_retained_*.csv：去共線性紀錄
- final_selected_feature_distance_summary.csv：最終特徵的距離 / 日期穩定性
- foldwise_cv_scores.csv / foldwise_feature_stability.csv
- leave_one_distance_out_*.csv
- holdout_by_distance_cm.csv / holdout_by_leak_quantity.csv / holdout_by_session_id.csv
- offband_assumption_check.csv：抗干擾假設檢查
- baseline_vs_leak_diagnostic/：每個特徵的 Cohen's d 與 AUC


十、主要開關
------------

    FORCE_BAND_DISCOVERY         True：跑 Step2a；False：跑 Step2b
    BUILD_CROSS_WINDOW_FEATURES  是否計算跨窗特徵（不進模型，預設 False）
    RUN_SOFT_QC                  是否輸出 Soft QC 報表（預設 False）
    RUN_PERMUTATION_TEST         是否跑打亂標籤檢定（預設 False）
    REQUIRE_FINAL_BASELINE       是否要求 baseline 為 final（現場部署時改 True）

新資料進來前要確認的設定：

    DUPLICATE_POLICY             舊資料 "drop_conflicts" → 新資料 "error"
    RESERVED_TEST_DATES          填入保留給最終推論的日期
    STRICT_BASELINE_ISOLATION    有獨立的 Baseline-only 正常資料後改 True
    MAX_BAND_FREQ_HZ             依 nyquist_check 結果確認


十一、本程式不負責
------------------

- 建立固定 Baseline（build_site_baseline_v2.py）
- 定義 Offband（shared 的 OFFBAND_BANDS）
- 共振頻帶搜尋與核准（find_pipe_resonance_from_disturbance.py）
- 正式模型訓練與儲存
- Disturbance 告警規則搜尋
- Transition 事件層級驗證
- 正式 Postprocess 設定定案
- Runtime Schema 生成與即時推論
"""

import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal
from nptdms import TdmsWriter, ChannelObject              # save_diff_to_tdms() 輸出三通道 TDMS

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

# 特徵工程共用函式（與 baseline、Transition、Disturbance、Runtime 共用同一份規則）
from shared_feature_utils import (
    # 讀檔與設定
    load_tdms_folder,
    load_band_config,
    load_feature_extraction_config,
    calculate_file_sha256,
    # 檢查
    check_signal_quality,
    # 特徵計算
    load_or_compute_features_by_file,
    configure_temporal_band_names,
    build_offband_summary_features,
    compute_band_energy,
    # 固定 baseline 與衍生特徵
    load_fixed_normal_baseline,
    apply_fixed_normal_baseline,
    validate_baseline_source_isolation,
    add_cusum_features,
    add_window_dynamic_features,
    add_local_rolling_features,
    validate_runtime_extractable_features,    # 確認 runtime 算得出這些特徵
    # 稽核
    audit_band_feature_coverage,
    audit_offband_summary_coverage,
    audit_selected_feature_alignment,
    # 全域設定
    REMOVE_DC,             # 是否去直流（save_diff_to_tdms 記錄用）
    OFFBAND_BANDS,         # 抗干擾規則用的 offband（全流程唯一定義）
)


from plotting import (
    plot_record_dual_channel_spectrogram,
    plot_band_energy_summary,
    plot_cross_condition_same_channel_psd,
    plot_within_label_dual_channel_psd,
    plot_top_features_multilevel_boxplot,
    plot_top_features_distance_effect_lines,
    plot_selected_feature_heatmap_by_condition,
    plot_selected_feature_lines_by_leak_and_distance,
)


# 中文字型
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Noto Sans CJK TC", "PingFang TC"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 路徑
# ============================================================
DATA_ROOT = Path("/Users/paul/Desktop/Final/normal_leak_test")    # 純正常（leak0）/ 純洩漏（leak1）資料
FINAL_OUTPUT_ROOT = Path("/Users/paul/Desktop/Final/test_result")
FEATURE_EXTRACTION_CONFIG_PATH = FINAL_OUTPUT_ROOT / "feature_extraction_config.json"

# Step2a（頻帶探索）
DISCOVERY_BAND_CONFIG_PATH = FINAL_OUTPUT_ROOT / "band_config_step2_discovery.json"
DISCOVERY_BASELINE_PATH = FINAL_OUTPUT_ROOT / "discovery_baseline.json"

# Step2b（正式特徵選擇）
BAND_CONFIG_PATH = FINAL_OUTPUT_ROOT / "band_config_step2.json"
FIXED_BASELINE_PATH = FINAL_OUTPUT_ROOT / "site_baseline.json"

# 共振程式核准的正式共振頻帶：Step2a 會自動帶入候選設定檔，Step2b 會檢查是否一致
RESONANCE_JSON_PATH = Path("/Users/paul/Desktop/Final/resonance/approved_resonance_bands.json")


# ============================================================
# 執行模式（切換流程時要改的地方）
# ============================================================
# True：執行 Step2a 頻帶探索（即使 band_config_step2.json 已存在）；False：執行 Step2b
FORCE_BAND_DISCOVERY = False

# 實驗室 baseline 是 candidate → False；現場正式部署時改 True
REQUIRE_FINAL_BASELINE = False

# baseline 與 Step2 資料重複時：False 只警告（目前實驗室資料共用）；現場資料到位後改 True 直接停止
STRICT_BASELINE_ISOLATION = False

N_JOBS = 6              # 特徵計算的平行數（記憶體不足時調小）
RUN_SOFT_QC = False     # Soft QC 報表只輸出 CSV 給人看，不影響任何篩選；需要檢查資料時再打開

# 打亂標籤檢定：特徵是用真實標籤選的，結果幾乎一定「顯著」，參考價值有限且很耗時
# 真正評估選特徵流程請看 fold-wise CV 與 LODO；需要時再打開
RUN_PERMUTATION_TEST = False


# ============================================================
# 資料管理
# ============================================================
# 保留給最終模型推論的日期：Step2 完全不讀（選頻帶、選特徵也算訓練的一部分）
RESERVED_TEST_DATES = []            # 例如 ["20261023"]；舊資料流程測試留空

# 內容完全相同的重複檔案："error" 直接停止（新資料）；"drop_conflicts" 排除矛盾的副本（舊資料流程測試）
DUPLICATE_POLICY = "drop_conflicts"

SESSION_GAP_SEC = 60                # 同一條件下，相鄰檔案間隔超過這個秒數就視為新的錄音場次
HOLDOUT_RATIO = 0.2                 # Step2 內部 holdout 的比例（以場次為單位）
HOLDOUT_GAP_FILES = 5               # 單一場次依時間切分時，dev 和 holdout 之間丟掉幾支當緩衝
RANDOM_STATE = 42


# ============================================================
# 分析對象
# ============================================================
TARGET_LEVELS = {0, 30}                            # 比較的洩漏量（0 = 正常）
TARGET_LEAK_LEVELS = {30}                          # 其中屬於洩漏的
ANALYSIS_LEVELS = sorted(TARGET_LEVELS)            # [0, 30]
ANALYSIS_LEAK_LEVELS = sorted(TARGET_LEAK_LEVELS)  # [30]


# ============================================================
# Step2a：頻帶探索
# ============================================================
PRELIMINARY_BANDS_TO_KEEP = 25    # 第一輪：98 個候選頻帶先保留幾個
TOP_BANDS_TO_KEEP = 12            # 第二輪（加入 temporal 後）：最後保留幾個
MIN_FEATURE_FAMILY_COUNT = 2      # 每個頻帶至少要有幾個通道（ai0 / ai1 / diff）明顯支持
SUPPORT_MIN_IMPORTANCE = 0.5      # 一個通道的 |Cohen's d| 至少多少，才算「支持」這個頻帶（0.5 = 中等效果）

# 候選洩漏頻帶的上限頻率：取樣率 1 MHz 的奈奎斯特是 500 kHz，越接近越受抗混疊濾波器影響
# 請依 nyquist_check 的結果調整
MAX_BAND_FREQ_HZ = 450_000

# 候選洩漏頻帶是否要避開共振頻帶：
# band config 合併時「洩漏頻帶優先、重疊的共振頻帶丟掉」，不避開的話共振頻帶會從正式特徵裡消失
EXCLUDE_RESONANCE_OVERLAP = True

if PRELIMINARY_BANDS_TO_KEEP < TOP_BANDS_TO_KEEP:
    raise ValueError("PRELIMINARY_BANDS_TO_KEEP 不可小於 TOP_BANDS_TO_KEEP")


# ============================================================
# Step2b：特徵選擇
# ============================================================
# 要建立 rel / z 的頻帶特徵種類。
# E、RMS、EnvRMS 三者高度相關（RMS = √E），共線性篩選通常只會留一個；
# 想加速可改成 ("E",)，想多一種不同角度可改成 ("E", "EnvStd")
STEP2_EXPAND_SUFFIXES = ("E", "RMS", "EnvRMS")

TOP_K_CANDIDATES = 200        # 統計排名後，取前幾名進入模型重要性與共線性篩選
CORR_THRESHOLD = 0.95         # 共線性門檻：相關係數超過這個值的兩個特徵只留一個
MIN_SIGN_CONSISTENCY = 0.67   # 方向一致性下限：各距離（各天）中，至少這個比例要是同一方向（例如都是洩漏較高）
MIN_FINAL_FEATURES = 30       # 共線性篩選後最少保留幾個特徵
TOP_N_FEATURES_TO_PLOT = 20   # 圖表顯示前幾名

# 跨窗特徵不進模型：離線每個檔案只有 7 窗，現場是連續資料流，數值分布對不上；
# 時間特性改由後規則處理（看模型分數是否持續偏高）。
# 注意 cusum_max / cusum_alert_rate 這類「單一視窗內」的 CUSUM 不受影響，所以這裡不寫 "cusum_"
MODEL_EXCLUDED_PREFIXES = ("delta_", "local_", "rolling_", "cusum_pos_", "cusum_neg_")

# 是否計算跨窗特徵（CUSUM / delta / local / rolling）：不進模型，只供分析參考，而且是 Step2b 最慢的部分
BUILD_CROSS_WINDOW_FEATURES = False

# 已知受距離影響很大的特徵，不允許進入模型
DROP_DISTANCE_SENSITIVE = [
    "ai0_crest",
    "ai0_std",
    "high_low_ratio",
    "centroid_ai1_global",
    "diff_crest",
    "ai1_rms",
]

# 不是特徵的欄位（識別、標籤、條件、時間、QC）：排名與聚合時一律排除
NON_FEATURE_COLS = {
    # 識別與標籤
    "filename", "record_id", "label",
    # 實驗條件
    "condition", "group", "leak_quantity", "distance_cm", "pressure",
    # 場次與日期（CV 分組用）
    "session_id", "file_time", "file_date",
    # 時間與索引
    "t_start", "t_end", "t_ref", "window_idx", "source_index", "fs",
    # Soft QC
    "decision", "soft_score", "ratio_warning", "ratio_reason", "reason", "family_reason",
}


# ============================================================
# CUSUM 容忍值
# ============================================================
CUSUM_K = 0.5        # CUSUM 容忍值（標準 CUSUM，約為想偵測偏移量的一半）
CUSUM_DECAY = 0.97


# ******************************  資料讀取/前處理/windowing  ******************************
#                                                 =====     三通道 TDMS 輸出（給深度學習用，目前未啟用）     =====
# ============================================================
# 三通道（ai0 / ai1 / diff）存成新的 TDMS（給深度學習用）
# ============================================================
def save_diff_to_tdms(records,             # 要儲存的資料清單
                      save_dir,            # 輸出的根目錄
                      dtype=np.float32,    # 輸出的訊號預設轉成 32 位元浮點數 
                      overwrite=False):    # 決定遇到已存在的輸出檔案時怎麼處理, overwrite=False(檔案已存在就跳過), overwrite=True(重新寫入並覆蓋原檔)
    """
    把每個 record 的 ai0 / ai1 / diff 存成三通道 TDMS。
    - 依 record_id（相對路徑，例如 leak0/xxx.tdms）建立子資料夾，避免不同資料夾的同名檔案互相覆蓋
    - 存的是已處理過的訊號（去直流、截取前 max_points 點），處理資訊與 metadata 寫在 properties
    - 預設 float32：檔案大小約為 float64 的一半，精度對深度學習已足夠
    - overwrite=False 時，已存在的檔案會跳過
    """
    # 建立輸出資料夾物件
    save_dir = Path(save_dir)
    
    # 準備結果紀錄
    # saved_paths:記錄成功寫入的檔案路徑, skipped:記錄因為檔案已存在而跳過幾個
    saved_paths, skipped = [], 0    

    # 每次從 records 取出一筆資料，放進變數 r
    for r in records:
        rel = Path(r["record_id"])                                        # 取得相對路徑, 例如 leak0/xxx.tdms
        fpath = save_dir / rel.parent / f"{rel.stem}_with_diff.tdms"      # 建立輸出檔案路徑
        if fpath.exists() and not overwrite:     # 已存在就跳過, 直接進入下一筆 record，不執行本次後面的寫檔流程
            skipped += 1
            continue         
        fpath.parent.mkdir(parents=True, exist_ok=True)     # 建立輸出資料夾
        
        # 從 record 取得 metadata
        meta = r.get("meta", {})
        # 建立 TDMS properties
        props = {
            "wf_increment": 1.0 / float(r["fs"]),                   # 取樣時間間隔:讓 shared 讀回時能通過取樣率檢查
            "wf_start_offset": 0.0,                                 # 波形開始時間
            "source": str(r["filepath"]),                           # 原始檔案完整路徑
            "record_id": str(r["record_id"]),                       # 保存原本的資料識別名稱
            "label": int(r["label"]) if r.get("label") is not None else -1,   # 有標籤就寫0/1, -1 代表未標記
            "leak_quantity": float(meta.get("leak_quantity", np.nan)),        # 洩漏量
            "distance_m": float(meta.get("distance", np.nan)),                # 感測距離
            "pressure_bar": float(meta.get("pressure", np.nan)),              # 壓力
            "preprocessing": f"remove_dc={REMOVE_DC}; diff=ai1-ai0; truncated_to_max_points",    # 前處理紀錄
            "session_id": str(r.get("session_id", "")),                       # 錄音場次（切分資料用）
            "file_time": r["file_time"].isoformat() if r.get("file_time") else "",   # 檔案時間（依日期切分用）
        }

        # 建立三個 TDMS 通道
        channels = [
            ChannelObject("signals", name, data=np.asarray(r["signal"][name], dtype=dtype), properties=props)
            for name in ("ai0", "ai1", "diff")
        ]
        # 寫入 TDMS
        with TdmsWriter(fpath) as writer:
            writer.write_segment(channels)
        saved_paths.append(fpath)

    print(f"✅ 已存 {len(saved_paths)} 個三通道 TDMS 到 {save_dir}"
          + (f"（{skipped} 個已存在而跳過）" if skipped else ""))
    return saved_paths



#                                        *****************************************************************************
#                                                                     QC / 資料一致性檢查
#                                        *****************************************************************************
'''
資料檢查分三層：
  0. 重複檔案檢查（check_duplicate_files）：內容完全相同的檔案 → 依 DUPLICATE_POLICY 停止或排除
  1. Hard QC（shared 的 check_signal_quality）：NaN / Inf、全零、RMS 過低 → 直接排除
  2. Soft QC（check_leak_channel_consistency）：在同一組（同洩漏量、同距離）內找離群檔案
     → 只輸出報表（keep / review），不排除任何資料；由 RUN_SOFT_QC 控制是否執行
若之後要用 soft QC 結果排除資料，必須在 dev / holdout 切分後分開執行，避免資料洩漏。
'''
# ============================================================
# 檢查內容完全相同的重複檔案
# ============================================================
def check_duplicate_files(records, policy="error"):
    """
    用檔案內容的 SHA-256 找出內容完全相同的檔案。
    policy：
      - "error"：發現任何重複就停止（新資料用這個）
      - "drop_conflicts"：同一份內容同時被標成不同 label 時，所有副本都排除；
                          label 相同的重複只保留一份（舊資料流程測試用）
      - "warn"：只提示，不排除
    """
    # 檢查是否 policy 輸入錯誤
    valid_policies = {"error", "drop_conflicts", "warn"}
    if policy not in valid_policies:
        raise ValueError(f"policy 必須是 {sorted(valid_policies)} 之一，目前為 {policy!r}")
    
    # 幫每支檔案算「指紋」，把指紋相同的放在一起
    by_hash = {}
    for r in tqdm(records, desc="檢查重複檔案", unit="file"):
        by_hash.setdefault(calculate_file_sha256(r["filepath"]), []).append(r)

    # 找出重複的組，沒有就直接結束
    dup_groups = [rs for rs in by_hash.values() if len(rs) > 1]
    if not dup_groups:
        print("✅ 沒有內容重複的檔案")
        return records

    # 區分兩種重複，並印出摘要，重複有兩種，嚴重程度不同：
    # label 一致的重複：同一份內容出現兩次，但都標成洩漏（或都標成正常）。只是多存了一份，不算錯，但會讓那段資料被算兩次。
    # label 矛盾的重複：同一份內容一份標正常、一份標洩漏。這就是你這次遇到的 347 組，資料本身是錯的。
    conflict_groups = [rs for rs in dup_groups if len({x["label"] for x in rs}) > 1]
    print(f"⚠️ 內容重複的檔案：{len(dup_groups)} 組，其中 label 互相矛盾：{len(conflict_groups)} 組")
    for rs in dup_groups[:3]:
        print("   例如：", [x["record_id"] for x in rs])

    # 依 policy 決定怎麼處理
    if policy == "error":      # "error"：直接停止程式，逼你先整理資料。新資料用這個。
        raise ValueError("發現內容重複的檔案，請先整理資料（或流程測試時改用 policy='drop_conflicts'）")
    if policy == "warn":       # "warn"：只印出提醒，資料原封不動繼續跑。
        return records

    # drop_conflicts：矛盾的整組排除；label 一致的只留第一份
    drop_ids = set()
    for rs in dup_groups:
        if len({x["label"] for x in rs}) > 1:
            drop_ids.update(x["record_id"] for x in rs)        # 矛盾：整組都排除
        else:
            drop_ids.update(x["record_id"] for x in rs[1:])    # 一致：只留第一份
    kept = [r for r in records if r["record_id"] not in drop_ids]
    print(f"   已排除 {len(drop_ids)} 支，剩 {len(kept)} 支")
    return kept


# ============================================================
# 從檔名還原錄音場次
# ============================================================
# 檔名拆解規則：cond：實驗條件（洩漏量、壓力、位置）/ date：8 位數日期 / time：6 位數時間
_SESSION_RE = re.compile(r"^(?P<cond>.+?)_(?P<date>\d{8})_(?P<time>\d{6})")

def assign_session_ids(records, gap_sec=SESSION_GAP_SEC):
    """
    檔案是連續錄音，每秒存一支 0.5 秒的片段；相鄰檔案幾乎一樣，不能拆到 dev / holdout 兩邊。
    依檔名的「條件」（時間戳之前的部分）與時間戳，把連續的檔案歸到同一個場次，
    寫入 r["session_id"] 與 r["file_time"]。
    """
    parsed = []
    
    # 逐支檔案拆解檔名
    for r in records:
        m = _SESSION_RE.match(Path(r["filename"]).stem)
        if m is None:
            raise ValueError(f"檔名無法解析場次（需要 _YYYYMMDD_HHMMSS）：{r['filename']}")
        ts = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
        parsed.append((m["cond"], ts, r))

    # 依條件、時間排序
    parsed.sort(key=lambda x: (x[0], x[1]))
    last_ts, counter = {}, {}
    
    # 判斷場次、編號
    for cond, ts, r in parsed:
        # 這個條件第一次出現，或者和上一支的時間差超過 60 秒 → 視為新場次，編號加 1。
        if cond not in last_ts or (ts - last_ts[cond]).total_seconds() > gap_sec:
            counter[cond] = counter.get(cond, 0) + 1
            
        # 間隔在 60 秒內 → 和上一支屬於同一場次，編號不變
        last_ts[cond] = ts
        r["session_id"] = f"{cond}_s{counter[cond]:02d}"
        r["file_time"] = ts
        
    return records


# ============================================================
# 以場次為單位切分 dev / holdout
# ============================================================
def split_dev_holdout_by_session(
    records,                              # 要切分的資料（每筆都要先跑過 assign_session_ids，才有 session_id / file_time）
    holdout_ratio=HOLDOUT_RATIO,          # holdout 大約佔多少比例（預設讀設定區，0.2 = 20%）
    gap_files=HOLDOUT_GAP_FILES,          # 依時間切分時，dev 和 holdout 中間丟掉幾支當緩衝
    min_sessions=3,                       # 一個組合至少要有幾個場次，才用「整場切分」
    random_state=RANDOM_STATE,            # 隨機種子：固定後每次挑到的 holdout 場次都一樣，結果可重現
):
    """
    在每個「label × 距離」組合內切分：
    - 場次數 ≥ min_sessions：隨機挑約 holdout_ratio 的場次，整個場次給 holdout
    - 場次數 <  min_sessions：每個場次各自依時間切，後段約 holdout_ratio 給 holdout，
                              中間丟掉 gap_files 支當緩衝（場次太少時整場拿走會讓 dev 少掉太多）
    """
    # 建立隨機數產生器（只用來挑 holdout 場次）
    rng = np.random.default_rng(random_state)

    # ---------- 1. 分組：先依「label × 距離」，組內再依場次 ----------
    # 結果長這樣：
    #   (0, 70) → {"leak0_..._70cm_s01": [檔案...], "leak0_..._70cm_s02": [檔案...]}
    #   (1, 70) → {"leak1_..._70cm_s01": [檔案...], ...}
    # 每個「label × 距離」各自切分，才能確保 holdout 裡每個距離的正常和洩漏都有資料
    groups = {}
    for r in records:
        key = (r["label"], int(round(r["meta"]["distance"] * 100)))
        groups.setdefault(key, {}).setdefault(r["session_id"], []).append(r)

    # ---------- 小工具：單一場次依時間切 ----------
    # 照時間排序後切成三段：[ dev 前段 ][ 丟掉的緩衝 ][ holdout 後段 ]
    # 緩衝的目的：相鄰檔案只差一秒、幾乎一樣，交界處不留空隔的話，holdout 的第一支等於 dev 看過
    def time_split(files):
        files = sorted(files, key=lambda r: r["file_time"])          # 依錄音時間由早到晚排序
        n_hold = max(1, round(len(files) * holdout_ratio))           # holdout 要幾支（至少 1 支）
        cut = len(files) - n_hold                                    # holdout 從第幾支開始
        dev_part = files[:max(0, cut - gap_files)]                   # dev：從頭到「cut 往前 gap_files 支」
        hold_part = files[cut:]                                      # holdout：最後 n_hold 支
        return dev_part, hold_part

    # ---------- 2. 逐組切分 ----------
    dev, holdout, report = [], [], []
    for (label, dist), sessions in sorted(groups.items()):
        names = sorted(sessions)       # 這個組合裡的所有場次名稱（排序後，確保每次結果一致）
        g_dev, g_hold = [], []         # 這個組合分到 dev / holdout 的檔案

        if len(names) >= min_sessions:
            # (A) 場次夠多：隨機挑幾個「整場」給 holdout
            #     holdout 場次數 = 約 holdout_ratio，至少 1 場，且 dev 至少要留 1 場
            n_hold = min(max(1, round(len(names) * holdout_ratio)), len(names) - 1)
            hold_names = set(rng.choice(names, size=n_hold, replace=False).tolist())
            # 被挑中的場次整個進 holdout，其餘整個進 dev（同一場次不會被拆開）
            for s in names:
                (g_hold if s in hold_names else g_dev).extend(sessions[s])
            mode = f"場次切分：holdout {n_hold}/{len(names)} 場"
        else:
            # (B) 場次太少（例如只有 1～2 場）：整場拿走會讓 dev 少掉一半以上，
            #     所以改成每個場次各自依時間切，每場都把後段給 holdout
            for s in names:
                d, h = time_split(sessions[s])
                g_dev.extend(d)
                g_hold.extend(h)
            mode = f"場次內時間切分：{len(names)} 場各取後段，緩衝 {gap_files} 支"
        
        # 防呆：場次太短時，扣掉 holdout 和緩衝後 dev 可能一支都不剩
        if not g_dev:
            raise ValueError(f"label={label}、{dist}cm 切分後 dev 沒有資料，場次太短，請調小 holdout_ratio 或 gap_files")

        dev.extend(g_dev)
        holdout.extend(g_hold)
        report.append({
            "label": label,                       # 0 = 正常、1 = 洩漏
            "distance_cm": dist,                  # 距離
            "n_sessions": len(names),             # 這個組合有幾個場次
            "n_dev_files": len(g_dev),            # 分到 dev 的檔案數
            "n_holdout_files": len(g_hold),       # 分到 holdout 的檔案數
            "方式": mode,                          # 用哪種方式切（整場 / 場次內時間切分）
        })

    # ---------- 3. 印出切分報表，方便確認每組分得合不合理 ----------
    print("\n=== dev / holdout 切分方式 ===")
    print(pd.DataFrame(report).to_string(index=False))
    return dev, holdout


# ============================================================
# 依 metadata 決定分組名稱（soft QC 在同一組內比較離群）
# ============================================================
def make_group_name(meta, use_pressure=False):    # use_pressure 預設為 False,是因為目前資料都是 4 bar
    """
    把洩漏量、距離（和壓力）組成分組名稱，例如 normal_20cm、leak30_50cm、leak30_50cm_4bar。
    目前資料都是 4 bar，所以預設不把壓力放進組名。
    """
    leak_q = int(round(meta["leak_quantity"]))
    distance_cm = int(round(meta["distance"] * 100))
    leak_tag = "normal" if leak_q == 0 else f"leak{leak_q}"

    if use_pressure:
        return f"{leak_tag}_{distance_cm}cm_{meta['pressure']:g}bar"   # :g 保留小數，例如 0.3bar、4bar
    return f"{leak_tag}_{distance_cm}cm"


# 計算以中位數為中心的穩健離散程度，值越大代表這組資料在組內本來就比較分散；越小代表很集中
# 用 nanmedian 跳過空值，避免一個 NaN 讓整組結果變成 NaN
def _mad(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))    # 表示每個數值離中位數多遠,再取絕對值,再根據這些數值取中位數


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
    
    注意：df 必須先有 group_size 欄位（由 check_leak_channel_consistency 建立）
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

    # 算穩健 z 分數：1.4826 × MAD ≈ 標準差（常態分布下），讓 robust z 分數和一般 z 分數的尺度相同
    # 下限要依數值大小調整：固定的 1e-6 對能量這種 1e-9 等級的特徵太大，會把 z 分數壓到接近 0。
    # 改成「組內中位數的百萬分之一」，再加一個極小的絕對下限，避免中位數也是 0
    min_scale = np.maximum(df[median_col].abs() * 1e-6, 1e-30)
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


#  軟性異常評分：檢查所有 record（正常與洩漏）是否有組內離群，產生完整報表
def check_leak_channel_consistency(
        records,                      # 要檢查的 records（正常與洩漏都可以）
        target_bands=None,            # 要檢查哪些頻帶；建議由主程式明確傳入，例如 final_main_bands + offband_bands_for_rule
        rms_ratio_min=0.8,            # 只作 warning
        peak_ratio_min=0.8,           # 只作 warning
        band_ratio_min=0.7,           # 只作 warning
        outlier_z_thresh=2.5,         # group-wise robust z-score 的離群門檻
        min_group_size=5,             # 每組至少幾筆才做離群判斷
        suspicious_score_thresh=2,    # 軟性異常累積幾分以上就列為 suspicious
        save_csv=None,                # 要不要把報表存成 CSV
        verbose=True,                 # 要不要印摘要
        group_by="session",           # "session"：同一錄音場次內比較（找壞檔案）；"condition"：同洩漏量 × 距離內比較
    ):
    """
    軟性一致性檢查（正常與洩漏資料都適用）：不直接刪除，只標記 suspicious

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
        
        # 根據設定決定分組：同一場次內比較，或同一條件（洩漏量 × 距離）內比較
        meta = r["meta"]
        condition_name = make_group_name(meta, use_pressure=False)     # 例如 leak30_50cm
        if group_by == "session":
            if "session_id" not in r:
                raise ValueError("group_by='session' 需要先執行 assign_session_ids()")
            group_name = r["session_id"]
        else:
            group_name = condition_name
        # 建一列基本資訊
        row = {
            "record_id": r["record_id"],                               # 唯一識別碼,其實也就是檔名
            "filename": r["filename"],                                 # 檔名
            "label": r["label"],                                       # 0 = 正常、1 = 洩漏（比值警示只對洩漏有意義）
            "condition": condition_name,                               # 實驗條件（洩漏量 × 距離），例如 leak30_50cm
            "group": group_name,                                       # 實際用來比較離群的分組（場次或條件）
            "session_id": r.get("session_id", ""),                     # 錄音場次
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

            row[f"{band['name']}_e0"] = e0         # ai0 在這段頻帶的能量（原始值，報表參考用）
            row[f"{band['name']}_e1"] = e1         # ai1 在這段頻帶的能量（原始值，報表參考用）
            # 能量差異是倍數關係，取 log10 後再判斷離群比較穩定
            row[f"{band['name']}_log_e0"] = np.log10(e0 + eps)
            row[f"{band['name']}_log_e1"] = np.log10(e1 + eps)
            row[f"{band['name']}_ratio"] = e1 / (e0 + eps)
            row[f"{band['name']}_log_gap"] = abs(np.log((e1 + eps) / (e0 + eps)))
            
        rows.append(row)

    # -- Step 4:整理成report_df -- 
    report_df = pd.DataFrame(rows)
    report_df["group_size"] = report_df.groupby("group")["filename"].transform("count")    # 新增 group_size，表示每個 group 有幾筆資料,後面判斷「小樣本組不要硬抓離群」的依據
    
    # 把所有 band ratio 欄位名稱整理成 list，方便後面重複使用
    band_ratio_cols = [f"{b['name']}_ratio" for b in target_bands]

    # -- Step 5:決定哪些欄位要做 group-wise outlier --
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
        # 頻帶能量類（用 log 能量判斷離群）
    ] + [f"{b['name']}_log_e0" for b in target_bands] \
      + [f"{b['name']}_log_e1" for b in target_bands] \
      + [f"{b['name']}_log_gap" for b in target_bands]

    # -- Step 6:建立特徵家族 --
    feature_families = {
        "amplitude": ["rms0", "rms1", "peak0", "peak1"],
        "gap": ["log_rms_gap", "log_peak_gap", "log_peak_ratio"] + [f"{b['name']}_log_gap" for b in target_bands],
        "band_energy": [f"{b['name']}_log_e0" for b in target_bands] + [f"{b['name']}_log_e1" for b in target_bands],
    }

    # -- Step 7:對每個特徵都做一次「在同 group 內抓離群」,算完後，每個欄位都會多出：xxx_robust_z, flag_xxx_group_outlier, xxx_outlier_side
    for col in feature_cols:
        report_df = add_group_robust_outlier_flags(
            report_df,
            group_col="group",     # 在同一場次（或同一條件，依 group_by）內比較
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
        
    # 「近端應該比遠端強」只在有洩漏時成立，正常資料不做這項警示
    report_df["ratio_warning"] = report_df["ratio_warning"] & (report_df["label"] == 1)
    
    # -- Step 11：整理真正的異常/warning 的原因 --
    #    (A)把 ratio warning 轉成文字(生成人類可讀原因)
    def summarize_ratio_warning(row):
        if row["label"] != 1:
            return "n/a（正常資料不檢查）"
        
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

    # 原因填回報表
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
    
    # 每個分組（場次）屬於哪個條件、有幾支檔案；
    # 依 condition 排序後，同一條件的幾個場次會排在一起，方便比較哪個場次整體偏掉
    group_meta_summary = (
        report_df.groupby("group").agg(
            condition=("condition", "first"),
            leak_quantity=("leak_quantity", "median"),
            distance_cm=("distance_cm", "median"),
            pressure=("pressure", "median"),
            n_files=("record_id", "size"),
        ).reset_index()
    )
    
    # 這個 group 是什麼條件 + 這個 group 的各種特徵中位數是多少
    group_feature_summary = (
        group_meta_summary.merge(
            report_df.groupby("group")[group_summary_cols].median().reset_index(), on="group", how="left")
        .sort_values(["condition", "group"])
        .reset_index(drop=True)
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
    # 先加 window-to-window dynamic features:前後窗變化量 delta_*
    df = add_window_dynamic_features(
        df=df,
        feature_cols=raw_feature_cols,
        group_col=group_col,
        time_col=time_col,
        prev_windows=(3, 5),
    )
    
    # 再加 local rolling features:局部統計與趨勢 (local_*、rolling_*)
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
    suffixes=STEP2_EXPAND_SUFFIXES,     # 預設讀設定區：是 E, RMS, EnvRMS
):
    """
    從指定 bands 中，挑出 Step2b 要做正式擴充的 base features。
    這裡不是挑所有 raw feature，
    而是挑要往下建立 rel / z / cusum / temporal 的那一批。
    """
    # 建立空清單，存最後找到的欄位
    cols = []

    # 頻帶特徵的欄位命名：{頻帶}_{通道}_{指標}，例如 scan_093_465_475k_ai1_E
    channel_patterns = [
        "{name}_ai0_{suffix}",
        "{name}_ai1_{suffix}",
        "{name}_diff_{suffix}",
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
# 群組摘要欄位（build_offband_summary_features 產生）：全部保留給抗干擾規則
RULE_ONLY_SUMMARY_FEATURES = {
    "leak_focus_E_mean", "leak_focus_E_max",
    "resonance_E_mean", "resonance_E_max",
}
# 計算時混入 offband 能量的特徵：同樣不可進模型
FEATURES_USING_OFFBAND = {"high_low_ratio"}

def is_rule_only_feature(
    feature_name,
    offband_bands,
):
    """
    判斷某個特徵是否屬於 Offband／Veto 規則專用欄位。

    Rule-only 特徵仍會被計算並保留在完整特徵表中，但不得參與模型排名、
    重要性評估、共線性篩選、最終特徵選擇與模型訓練，只供模型預測後的抗干擾規則使用。

    包含：
    1. 名稱含 "offband" 的摘要與比值（offband_E_mean、offband_to_leak_focus_E_ratio……）
    2. 單一 offband 頻帶的原始特徵（例如 off_010_020k_ai0_E）
    3. 洩漏 / 共振頻帶的群組摘要（leak_focus_E_mean、resonance_E_max……）
    4. 計算時混入 offband 能量的特徵（high_low_ratio）
    以上任一種加上 rel_ / z_ 前綴的衍生版本也一併視為規則專用。
    """
    
    feature_name = str(feature_name)

    # 去掉 rel_ / z_ 前綴，讓衍生版本和原始欄位用同一套判斷
    base_name = feature_name
    for prefix in ("rel_", "z_"):
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
            break

    # 1. offband 摘要與比值
    if "offband" in base_name.lower():
        return True

    # 3、4. 群組摘要、混入 offband 的特徵
    if base_name in RULE_ONLY_SUMMARY_FEATURES or base_name in FEATURES_USING_OFFBAND:
        return True

    # 2. 單一 offband 頻帶的原始特徵：名稱以「offband 頻帶名_」開頭
    offband_names = {str(band["name"]) for band in offband_bands}
    return any(base_name.startswith(f"{band_name}_") for band_name in offband_names)


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
# 判斷特徵屬於哪個通道（作為頻帶排名的「證據家族」）
# ==========================
def infer_feature_family(feature_name):
    """
    回傳特徵來自哪個通道：ai0（遠端）、ai1（近端）、diff（差分）。

    用通道當「證據家族」的原因：
    - 原始值、rel_、z_ 只是平移和縮放，區分力完全相同，不算獨立的證據
    - 同一個頻帶在不同通道上的差異，才是真正不同角度的證據
      （例如 ai1 和 diff 都有差異，比只有 ai1 有差異更可信）
    搭配 MIN_FEATURE_FAMILY_COUNT = 2，代表一個頻帶至少要有兩個通道支持才算候選。
    """
    match = re.search(r"_(ai0|ai1|diff)_", str(feature_name))
    return match.group(1) if match else "other"


# ==========================
# 從特徵重要性彙整成頻帶分數
# ==========================
def summarize_feature_importance_by_band(feature_importance_df, discovery_bands):
    """
    把「每個特徵的重要性」彙整成「每個頻帶的分數」。

    - 家族 = 通道（ai0 / ai1 / diff），由 infer_feature_family() 決定
    - 每個頻帶、每個通道只取前 2 名特徵，避免特徵多的頻帶分數虛高
    - supported_family_count：有幾個通道的最強特徵達到 SUPPORT_MIN_IMPORTANCE（明顯支持）
      → 由呼叫端用 MIN_FEATURE_FAMILY_COUNT 篩選，不放進分數
    - band_score 只看區分力：最強特徵 60% + 各通道平均 40%
    """
    rows = []

    # 建立 band 查表，後面可依 band_name 補回 f_low / f_high
    band_lookup = {b["name"]: b for b in discovery_bands}

    # ---------- 1. 每個特徵對應到頻帶與通道 ----------
    for _, row in feature_importance_df.iterrows():
        feat = row["feature"]
        band_name = infer_band_name_from_feature(feat)
        if band_name is None or band_name not in band_lookup:
            continue
        rows.append({
            "band_name": band_name,
            "feature": feat,
            "importance": row["importance"],
            "feature_family": infer_feature_family(feat),     # ai0 / ai1 / diff / other
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ---------- 2. 每個頻帶、每個通道只保留前 2 名 ----------
    family_top2 = (
        df.sort_values("importance", ascending=False)
        .groupby(["band_name", "feature_family"], group_keys=False)
        .head(2)
        .reset_index(drop=True)
    )

    # 原始特徵數：只是觀察用，看某個頻帶原本有多少特徵
    raw_feature_count_df = (
        df.groupby("band_name")
        .agg(raw_feature_count=("feature", "count"))
        .reset_index()
    )
    
    # ---------- 3. 通道層級：每個通道的平均與最強重要性 ----------
    family_summary = (
        family_top2.groupby(["band_name", "feature_family"])
        .agg(
            mean_top2_importance=("importance", "mean"),        # 這個通道前 2 名的平均
            max_importance_in_family=("importance", "max"),     # 這個通道最強特徵
        )
        .reset_index()
    )
    # 這個通道是否「明顯支持」這個頻帶（只算 ai0 / ai1 / diff，不算 other）
    family_summary["is_supporting"] = (
        family_summary["feature_family"].isin(["ai0", "ai1", "diff"])
        & (family_summary["max_importance_in_family"] >= SUPPORT_MIN_IMPORTANCE)
    )

    # ---------- 4. 頻帶層級 ----------
    summary = (
        family_summary.groupby("band_name")
        .agg(
            supported_family_count=("is_supporting", "sum"),               # 幾個通道明顯支持（0～3）
            feature_family_count=("feature_family", "nunique"),            # 總共有幾個通道有特徵（觀察用）
            mean_importance=("mean_top2_importance", "mean"),              # 各通道平均重要性再平均
            max_importance=("max_importance_in_family", "max"),            # 最強特徵的重要性
        )
        .reset_index()
    )
    summary["supported_family_count"] = summary["supported_family_count"].astype(int)

    # 支持的通道是哪些（方便人工檢查，例如 "ai1|diff"）
    supporting = (
        family_summary[family_summary["is_supporting"]]
        .groupby("band_name")["feature_family"]
        .apply(lambda s: "|".join(sorted(s)))
    )
    summary["supporting_families"] = summary["band_name"].map(supporting).fillna("")

    # 補回原始特徵數與頻率範圍
    summary = summary.merge(raw_feature_count_df, on="band_name", how="left")
    summary["f_low"] = summary["band_name"].map(lambda x: band_lookup[x]["f_low"])
    summary["f_high"] = summary["band_name"].map(lambda x: band_lookup[x]["f_high"])

    # ---------- 5. 綜合分數：只看區分力 ----------
    summary["band_score"] = 0.6 * summary["max_importance"] + 0.4 * summary["mean_importance"]

    summary = summary.sort_values(
        ["band_score", "supported_family_count"],
        ascending=False,
    ).reset_index(drop=True)

    return summary


# ============================================================
# 將 ranking 結果整理成統一的 feature / importance 格式
# 第一輪與第二輪共用
# ============================================================
def build_feature_importance_table(
    ranking_df,              # rank_features_for_leak_levels 之類的排名函式算出來的結果
    importance_col=None,     # 指定用哪一欄當重要性；None 時依優先順序自動判斷
):
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
    if importance_col is None:
        # 沒指定就依優先順序自動判斷
        importance_col = next(
            (column for column in importance_candidates if column in ranking_df.columns),
            None,
        )
        if importance_col is None:
            raise ValueError(f"找不到可用的 importance 欄位；目前 ranking 欄位：{ranking_df.columns.tolist()}")
    elif importance_col not in ranking_df.columns:
        raise ValueError(f"ranking_df 沒有指定的欄位 {importance_col}")
    
    print(f"ℹ️ 特徵重要性使用欄位：{importance_col}")

    # 格式統一
    feature_importance_df = (
        ranking_df[["feature", importance_col]]    # 從原表裡只挑出這兩欄 "feature", importance_col ，組成一個新的、較小的 DataFrame
        .rename(columns={importance_col: "importance",}).copy())    # 把剛剛動態找到的那個欄位（不管它原本叫 final_score 還是 score 還是 cohen_d）統一改名成 "importance"
    
    # 把 feature 這欄強制轉成字串型別
    feature_importance_df["feature"] = (feature_importance_df["feature"].astype(str))

    # 把 importance 欄強制轉成數值型別
    feature_importance_df["importance"] = (
        pd.to_numeric(feature_importance_df["importance"], errors="coerce",))
    
    # cohen_d 有正負號（方向），但排名只看差異大小，所以取絕對值
    if importance_col == "cohen_d":
        feature_importance_df["importance"] = feature_importance_df["importance"].abs()

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
# 讀取探索頻帶設定（與 baseline 程式使用同一套驗證）
# ==========================
def load_discovery_band_config(config_path):      # 接收一個參數 config_path——探索階段輸出的那份 schema_status="discovery"
    """
    讀取 band_config_step2_discovery.json。
    - 基本驗證交給 shared 的 load_band_config（和 build_site_baseline_v2.py 同一套規則）：
      schema、狀態必須是 discovery、每段頻帶的名稱與上下界、名稱不可重複
    - 額外確認 baseline_scope 是 discovery_scan_grid（探索版獨有的欄位）
    回傳 (原始 payload, 全部候選頻帶清單)
    """
    
    # 統一轉成 Path 物件
    config_path = Path(config_path)
    
    # 和 baseline 程式同一套驗證；探索版的候選頻帶放在 leak_focused_bands
    band_config = load_band_config(config_path, expected_status="discovery")

    # 讀原始內容，取得 temporal_band_names 等其他欄位，並確認是完整掃描網格
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("baseline_scope") != "discovery_scan_grid":
        raise ValueError("discovery band config 的 baseline_scope 必須是 discovery_scan_grid")

    return payload, list(band_config.leak_focused_bands)


# ==========================
# 讀取共振頻帶 JSON
# ==========================
def load_resonance_band_json(resonance_json_path):
    """讀取共振程式核准的正式共振頻帶（approved_resonance_bands.json）。"""
    # 先把路徑轉成 Path
    resonance_json_path = Path(resonance_json_path)

    # 檢查檔案在不在，如果找不到，就直接報錯
    if not resonance_json_path.exists():
        raise FileNotFoundError(f"找不到共振頻帶 JSON: {resonance_json_path}")

    # 把共振 JSON 檔讀進來，轉回 Python 字典
    payload = json.loads(resonance_json_path.read_text(encoding="utf-8"))
    # 檢查是否讀到正確檔案
    if payload.get("schema_name") != "approved_resonance_bands":
        raise ValueError(f"{resonance_json_path.name} 不是 approved_resonance_bands 格式（是否讀到舊版或提案檔？）")
    
    # 只保留 band config 需要的欄位（reason 等說明欄位不寫進 band config）
    resonance_bands = [
        {"name": b["name"], "f_low": b["f_low"], "f_high": b["f_high"], "band_type": b["band_type"]}
        for b in payload.get("resonance_bands", [])
    ]
    print(f"✅ 已讀取正式共振頻帶（核准於 {payload.get('approved_at')}）：{[b['name'] for b in resonance_bands]}")
    return resonance_bands


# ==========================
# 判斷兩個 band 是否重疊(leak candidate band 有沒有跟 resonance band 重疊)
# ==========================
def bands_overlap(b1, b2):
    """
    兩段頻帶有重疊就回傳 True（只碰到邊界不算，例如 10～20k 和 20～30k 不算重疊）。
    Step2a 用在三個地方：
      - 候選洩漏頻帶是否落在共振區
      - 候選洩漏頻帶是否壓到 offband（壓到就排除）
      - 候選洩漏頻帶彼此是否相鄰重疊（重疊只留分數最高的）
    """
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
    2. resonance_bands：共振程式核准的共振頻帶
    3. offband_bands：抗干擾規則用的參照頻帶（來自 shared 的 OFFBAND_BANDS）

    Step2a 輸出的是 candidate：人工檢查後，把 schema_status 改成 final，
    另存成 band_config_step2.json，才會被 final baseline 與 Step2b 使用。
    """
    # 防呆：洩漏頻帶不可壓到 offband（否則 final baseline 會報錯，offband 也失去參照意義）
    conflicts = [
        f"{leak['name']} ↔ {off['name']}"
        for leak in leak_focused_bands
        for off in offband_bands
        if bands_overlap(leak, off)
    ]
    if conflicts:
        raise ValueError(f"洩漏頻帶與 offband 重疊：{conflicts}")
    
    # 先建立要寫出去的內容
    payload = {
        "schema_name": "band_config",
        "schema_version": 2,
        "schema_status": schema_status,
        "source": source_note,                                            # 這份 band config 是怎麼來的
        
        "band_merge_policy": ("keep_primary_overlap_drop_secondary_overlap_v1"),
        
        "leak_focused_bands": leak_focused_bands,
        "resonance_bands": resonance_bands,                               # Step1 已經找好的共振頻帶
        "offband_bands": offband_bands,                                   # 保留給後面 disturbance 判斷用的 offband
        # 需要算「單一視窗內」時間結構特徵（residual 的 cusum_peak、burst 等）的頻帶；
        # 和跨窗特徵（BUILD_CROSS_WINDOW_FEATURES）無關
        "temporal_band_names": sorted(list(temporal_band_names)),         
    }
    # 把輸出路徑轉成 Path 物件, 如果資料夾不存在，就先建立
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    # 把整份 payload 寫成 JSON 檔
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    # 用 final baseline 同一套規則讀回來檢查（名稱重複、上下界、temporal 是否都在 core 頻帶內）
    load_band_config(output_json, expected_status=schema_status)

    print(f"✅ 已輸出完整 band config: {output_json}")
    
    
# ==========================
# 依分數挑選候選洩漏頻帶：避開 offband、相鄰重疊只留分數最高的
# ==========================
def select_non_overlapping_bands(
    band_summary_df,          # 已依 band_score 由高到低排序的頻帶排名表
    band_lookup,              # 頻帶名稱 → 頻率範圍
    offband_bands,            # 抗干擾參照頻帶
    top_n,                    # 最多選幾個
    resonance_bands=(),       # 共振頻帶（有傳就避開）
    max_freq_hz=None,         # 頻帶上界不可超過這個頻率（有傳才檢查）
):
    """
    從分數最高的頻帶開始，一個一個檢查，依序排除：
      1. 上界超過 max_freq_hz      → 太接近奈奎斯特頻率，受抗混疊濾波器影響
      2. 壓到 offband              → offband 是抗干擾參照，不能同時當洩漏頻帶
      3. 壓到共振頻帶              → 合併時共振頻帶會被丟掉，正式特徵就沒有共振了
      4. 和前面已選入的頻帶重疊    → 相鄰頻帶重疊一半，其實是同一段訊號，留分數高的就好
    都沒問題且還沒選滿 top_n 個 → 選入
    回傳 (選入的頻帶清單, 加上 rank 與 selection_status 欄位的排名表)
    """
    # 複製一份排名表（避免改到原本的），準備兩個清單：selected 放已選入的頻帶，status 記錄每個頻帶的處理結果。
    df = band_summary_df.copy().reset_index(drop=True)
    selected, status = [], []

    # 從分數最高的頻帶開始，一個一個檢查
    for band_name in df["band_name"]:
        band = band_lookup[band_name]     # 目前檢查的這個頻帶（含頻率範圍）
        # 目前這個頻帶壓到哪些 offband / 共振頻帶 / 已選入的頻帶（沒有就是空清單）
        hit_offband = [o["name"] for o in offband_bands if bands_overlap(band, o)]
        hit_resonance = [r["name"] for r in resonance_bands if bands_overlap(band, r)]
        hit_selected = [s["name"] for s in selected if bands_overlap(band, s)]    

        # 目前檢查的這個頻帶，上界太接近奈奎斯特頻率 → 排除
        if max_freq_hz is not None and band["f_high"] > max_freq_hz:
            status.append(f"排除：上界超過 {max_freq_hz / 1000:.0f} kHz（接近奈奎斯特）")
        # 目前檢查的這個頻帶，和 offband 重疊 → 排除
        elif hit_offband:
            status.append(f"排除：壓到 offband {'|'.join(hit_offband)}")
        # 目前檢查的這個頻帶，和共振頻帶重疊 → 排除（否則合併時共振頻帶會消失）
        elif hit_resonance:
            status.append(f"排除：壓到共振頻帶 {'|'.join(hit_resonance)}")
        # 目前檢查的這個頻帶，和前面已選入的頻帶重疊 → 排除（前面選入的分數一定比較高）
        elif hit_selected:
            status.append(f"排除：與已選 {'|'.join(hit_selected)} 重疊")
        # 都沒有問題，而且還沒選滿 → 選入
        elif len(selected) < top_n:
            selected.append(band)
            status.append("選入")
        # 都沒有問題，但已經選滿了 → 不選
        else:
            status.append("未選（名額已滿）")

    # 把結果寫回排名表, 加上兩欄：原本的排名 rank，和每個頻帶的處理結果 selection_status
    df.insert(0, "rank", range(1, len(df) + 1))      # 原始排名（1 = 分數最高）
    df["selection_status"] = status

    # 排名很前面卻因 offband 被排除：區分力很強的話，代表這段頻率和洩漏有關，offband 可能選得不好
    strong_conflict = df[
        (df["rank"] <= top_n)
        & df["selection_status"].str.contains("offband|共振", regex=True)
    ]
    
    if not strong_conflict.empty:
        print(f"\n⚠️ 以下頻帶排名在前 {top_n} 名，卻因壓到 offband 或共振頻帶被排除；"
              "若區分力很強，請檢查 OFFBAND_BANDS 或共振頻帶是否合適：")
        print(strong_conflict[["rank", "band_name", "band_score", "max_importance", "selection_status"]].to_string(index=False))

    if len(selected) < top_n:
        print(f"ℹ️ 只選到 {len(selected)} 個候選頻帶（目標 {top_n} 個）")

    return selected, df


# ==========================
# Step2a 頻帶探索主流程
# ==========================
def run_band_discovery_stage(
    records_dev,      # dev 資料（正常 + 洩漏）
    output_dir,       # 輸出資料夾
    *,
    window_sec,
    hop_sec,
    fs,
    nperseg,
    max_points,
    base_feature_version,
    resonance_json_path=None,                 # 共振程式核准的共振頻帶 JSON
    top_bands_to_keep=TOP_BANDS_TO_KEEP,      # 最後要保留幾個候選洩漏頻帶
):
    """
    Step2a 頻帶探索：
    1. 對全部 discovery 掃描頻帶算 window-level 特徵（有快取就直接讀）
    2. 聚合成 record-level，依「正常 vs 洩漏」的區分力（|Cohen's d|）排名特徵
    3. 彙整成頻帶分數（每個通道為一個證據家族），至少 MIN_FEATURE_FAMILY_COUNT 個通道明顯支持才保留
    4. （BUILD_CROSS_WINDOW_FEATURES = True 時）對初選頻帶加上跨窗特徵再排一次
    5. 依分數挑選候選頻帶：避開 offband、相鄰重疊只留分數最高的
    6. 輸出候選設定檔（offband 自動帶入），之後人工檢查定案

    排名只用原始特徵：rel_ / z_ 只是平移和縮放，區分力和原始值完全相同，算了也不會改變排名。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. 讀取頻帶設定
    # ============================================================
    # 讀共振頻帶（共振程式核准的）
    resonance_bands = load_resonance_band_json(resonance_json_path) if resonance_json_path is not None else []
    # 讀全部 98 個掃描頻帶，並建立「頻帶名稱 → 頻率範圍」的查表
    discovery_payload, discovery_bands = load_discovery_band_config(DISCOVERY_BAND_CONFIG_PATH)
    band_lookup = {str(b["name"]): b for b in discovery_bands}      # 頻帶名稱 → 頻率範圍
    print(f"Step2a 候選頻帶數量: {len(discovery_bands)}")

    # 單一視窗內的時間結構特徵（residual cusum_peak、burst 等）要算哪些頻帶
    configure_temporal_band_names(discovery_payload.get("temporal_band_names", []))

    # ============================================================
    # 2. 讀探索用 baseline（驗證設定一致）、資料隔離檢查
    # ============================================================
    # 讀 discovery baseline，同時確認它的設定指紋和現在一致（頻帶、特徵萃取設定都沒變）
    discovery_baseline = load_fixed_normal_baseline(
        DISCOVERY_BASELINE_PATH,
        band_config_path=DISCOVERY_BAND_CONFIG_PATH,
        feature_extraction_config_path=FEATURE_EXTRACTION_CONFIG_PATH,
        require_final=False,                     # 探索階段不要求正式版 baseline
    )
    # 確認是探索用的版本，不是正式版
    if discovery_baseline.get("baseline_scope") != "discovery_scan_grid":
        raise ValueError("Step2a 必須使用 baseline_scope=discovery_scan_grid 的 baseline")
    # 檢查 baseline 用的正常資料有沒有和 dev 重複（目前舊資料會重複，只會警告）
    validate_baseline_source_isolation(
        baseline_payload=discovery_baseline, analysis_records=records_dev,
        context="Step2a dev", strict=STRICT_BASELINE_ISOLATION,
    )

    # ============================================================
    # 3. window-level 特徵（全部掃描頻帶）
    # ============================================================
    # 把 dev 資料整理成檔案清單
    file_specs_dev = [
        {"filepath": r["filepath"], "record_id": r["record_id"], "label": r["label"]}
        for r in records_dev
    ]
    if not file_specs_dev:
        raise ValueError("Step2a 沒有可用的 dev 資料")
    # 對每支檔案、全部 98 個頻帶算特徵（有快取就直接讀，這是最花時間的一步）
    features_dev = load_or_compute_features_by_file(
        cache_dir=output_dir / "feature_cache",
        file_specs=file_specs_dev,
        core_bands=discovery_bands,
        aux_bands=[],                        # 探索階段不算 offband
        low_bands=[],
        split_name="band_discovery_dev",
        n_jobs=N_JOBS,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        target_levels=TARGET_LEVELS,
        feature_version=base_feature_version,
        max_points=max_points,
    )
    df_dev = pd.DataFrame(features_dev).reset_index(drop=True)
    del features_dev                         # 釋放記憶體

    # 整理每支檔案的洩漏量、距離、壓力，排名時要用
    record_meta_dev_df = pd.DataFrame([
        {
            "record_id": r["record_id"],
            "leak_quantity": r["meta"]["leak_quantity"],
            "distance_cm": int(round(r["meta"]["distance"] * 100)),
            "pressure": r["meta"]["pressure"],
        }
        for r in records_dev
    ]).drop_duplicates(subset=["record_id"])

    # 挑出要拿來排名的欄位：每個頻帶 × 每個通道 × E / RMS / EnvRMS
    base_feature_cols = select_step2_expand_base_features(df=df_dev, bands=discovery_bands)
    if not base_feature_cols:
        raise ValueError("Step2a 找不到候選頻帶的原始特徵")

    # ============================================================
    # 4. 第一輪：全部頻帶排名
    # ============================================================
    first_pass_feature_cols = list(base_feature_cols)

    # 如果開關打開，額外算 z 分數和跨窗 CUSUM，加進排名特徵
    if BUILD_CROSS_WINDOW_FEATURES:
        # 跨窗 CUSUM 需要 z 分數當輸入（只在開關打開時才算）
        df_dev = apply_fixed_normal_baseline(
            df=df_dev, feature_cols=base_feature_cols, baseline_payload=discovery_baseline,
        )
        z_cols = [f"z_{c}" for c in base_feature_cols]
        df_dev = add_cusum_features(df=df_dev, input_feature_cols=z_cols, k=CUSUM_K, decay=CUSUM_DECAY)
        first_pass_feature_cols += [f"cusum_{side}_{z}" for z in z_cols for side in ("pos", "neg")]

    print("\n=== Step2a 第一輪 ===")
    print("全部 discovery bands:", len(discovery_bands))
    print("第一輪特徵數:", len(first_pass_feature_cols))

    # window → record-level（每支檔案取中位數）
    first_pass_record_df = build_record_level_feature_table(
        window_df=df_dev,
        qc_record_df=None,
        feature_cols=first_pass_feature_cols,
        extra_meta_df=record_meta_dev_df,
        agg="median",
    )

    # 特徵排名：score = |Cohen's d|（正常 vs 洩漏）,每個特徵算「正常 vs 洩漏」的區分力（|Cohen’s d|），存成 feature_importance_first_pass.csv
    first_pass_ranking_df = rank_features_for_leak_levels(first_pass_record_df, target_levels=tuple(ANALYSIS_LEVELS))
    first_pass_ranking_df.to_csv(output_dir / "feature_importance_first_pass.csv", index=False, encoding="utf-8-sig")
    first_pass_feature_importance_df = build_feature_importance_table(first_pass_ranking_df, importance_col="score")

    # 彙整成頻帶分數；至少 MIN_FEATURE_FAMILY_COUNT 個通道明顯支持才保留
    first_pass_band_summary_df = summarize_feature_importance_by_band(
        first_pass_feature_importance_df, discovery_bands=discovery_bands,
    )
    first_pass_band_summary_df = first_pass_band_summary_df[
        first_pass_band_summary_df["supported_family_count"] >= MIN_FEATURE_FAMILY_COUNT
    ].copy().reset_index(drop=True)

    # 如果一個都沒留下，就報錯並建議你放寬條件
    if first_pass_band_summary_df.empty:
        raise ValueError(
            "Step2a 第一輪沒有頻帶達到支持門檻；"
            "可調低 SUPPORT_MIN_IMPORTANCE（例如 0.3）或 MIN_FEATURE_FAMILY_COUNT 再試"
        )
    first_pass_band_summary_df.to_csv(output_dir / "band_importance_first_pass.csv", index=False, encoding="utf-8-sig")

    # ============================================================
    # 5. 第二輪（可選）：初選頻帶加上跨窗特徵再排一次
    # ============================================================
    # 開關打開：取第一輪前 PRELIMINARY_BANDS_TO_KEEP 名，加上跨窗特徵（delta / local / rolling）重新排名（很慢，純資料上跨窗特徵接近 0）
    if BUILD_CROSS_WINDOW_FEATURES:
        # 從第一輪挑出初選頻帶(取前 25 名)
        preliminary_bands = [
            band_lookup[name]
            for name in first_pass_band_summary_df.head(PRELIMINARY_BANDS_TO_KEEP)["band_name"]
        ]
        # 把初選結果存檔
        save_leak_candidate_band_json(
            output_json=output_dir / "leak_candidate_bands_first_pass.json",
            selected_bands=preliminary_bands,
            source_note="step2a_first_pass",
        )

        # 只對初選頻帶建立跨窗特徵（delta / local / rolling）
        preliminary_base_cols = select_step2_expand_base_features(df=df_dev, bands=preliminary_bands)
        columns_before = set(df_dev.columns)
        df_dev = add_step2_temporal_features(df=df_dev, raw_feature_cols=preliminary_base_cols)
        temporal_cols = [c for c in df_dev.columns if c not in columns_before]

        # 第二輪特徵 = 初選頻帶的第一輪特徵 + 跨窗特徵
        preliminary_names = {b["name"] for b in preliminary_bands}
        second_pass_feature_cols = list(dict.fromkeys(
            [c for c in first_pass_feature_cols if infer_band_name_from_feature(c) in preliminary_names]
            + temporal_cols
        ))
        print("\n=== Step2a 第二輪 ===")
        print("初選 bands:", len(preliminary_bands), "／第二輪特徵數:", len(second_pass_feature_cols))

        # 把視窗資料壓成每支檔案一列
        record_df = build_record_level_feature_table(
            window_df=df_dev,
            qc_record_df=None,
            feature_cols=second_pass_feature_cols,
            extra_meta_df=record_meta_dev_df,
            agg="median",
        )
        
        # 重新排名特徵
        ranking_df = rank_features_for_leak_levels(record_df, target_levels=tuple(ANALYSIS_LEVELS))
        ranking_df.to_csv(output_dir / "feature_importance_summary.csv", index=False, encoding="utf-8-sig")
        feature_importance_df = build_feature_importance_table(ranking_df, importance_col="score")

        # 彙整成頻帶分數並篩選
        band_summary_df = summarize_feature_importance_by_band(feature_importance_df, discovery_bands=preliminary_bands)
        band_summary_df = band_summary_df[
            band_summary_df["supported_family_count"] >= MIN_FEATURE_FAMILY_COUNT
        ].copy().reset_index(drop=True)
        if band_summary_df.empty:
            raise ValueError("Step2a 第二輪沒有留下候選頻帶")
    else:     # 開關關閉時
        # 開關關閉：略過第二輪，直接用第一輪（只有原始特徵）的排名挑選頻帶
        # 不算跨窗特徵：第一輪排名就是最終排名
        print("ℹ️ BUILD_CROSS_WINDOW_FEATURES = False：略過第二輪，直接用第一輪排名挑選頻帶")
        record_df = first_pass_record_df
        ranking_df = first_pass_ranking_df
        band_summary_df = first_pass_band_summary_df

    record_df.to_csv(output_dir / "step2a_record_features_full.csv", index=False, encoding="utf-8-sig")

    # ============================================================
    # 6. 標註是否落在共振區（只是資訊，不影響選擇）
    # ============================================================
    # 對每個頻帶標記它有沒有和共振頻帶重疊。這只是資訊，方便你人工判斷，不影響選擇。
    if resonance_bands:
        band_summary_df["matched_resonance_bands"] = band_summary_df["band_name"].map(
            lambda name: "|".join(
                rb["name"] for rb in resonance_bands if bands_overlap(band_lookup[name], rb)
            )
        )
        band_summary_df["overlaps_resonance"] = band_summary_df["matched_resonance_bands"] != ""


    # ============================================================
    # 7. 挑選候選頻帶：避開 offband、相鄰重疊只留分數最高的
    # ============================================================
    # 避開 offband、相鄰重疊只留分數最高的，選出 12 個
    selected_bands, band_summary_df = select_non_overlapping_bands(
        band_summary_df, band_lookup, OFFBAND_BANDS, top_bands_to_keep,
        resonance_bands=resonance_bands if EXCLUDE_RESONANCE_OVERLAP else (),
        max_freq_hz=MAX_BAND_FREQ_HZ,
    )
    if not selected_bands:
        raise ValueError("Step2a 排除 offband 與重疊後沒有任何候選頻帶")
    # 存成 band_importance_summary.csv，並印出選入的頻帶清單
    band_summary_df.to_csv(output_dir / "band_importance_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== Step2a 選入的候選洩漏頻帶（{len(selected_bands)} 個）===")
    for band in selected_bands:
        print(f"  {band['name']}  {band['f_low']} ~ {band['f_high']} Hz")
        
    # 區分力異常高的提醒：微小洩漏通常不會讓兩組完全分開（|d| 超過 3 已經非常大）
    top_d = band_summary_df["max_importance"].max()
    if top_d > 3:
        print(f"\n⚠️ 最高區分力 |Cohen's d| = {top_d:.1f}，對微小洩漏而言異常大。")
        print("   請確認正常與洩漏是否在同一天、相同的感測器安裝條件下錄製（見程式開頭的「每天的正常 / 洩漏檔案數」）。")

    
    # ============================================================
    # 8. 輸出候選設定檔（人工檢查後改成 final，另存成 band_config_step2.json）
    # ============================================================
    # leak_candidate_bands.json：只有候選洩漏頻帶
    save_leak_candidate_band_json(
        output_json=output_dir / "leak_candidate_bands.json",
        selected_bands=selected_bands,
        source_note="step2a_micro_leak_band_discovery",
    )
    # band_config_step2_candidate.json：完整設定，包含洩漏頻帶、共振頻帶、offband（自動帶入）
    save_band_config_json(
        output_json=output_dir / "band_config_step2_candidate.json",
        leak_focused_bands=selected_bands,
        resonance_bands=resonance_bands,
        offband_bands=OFFBAND_BANDS,                                  # 自動帶入 shared 定義的 offband
        temporal_band_names={b["name"] for b in selected_bands},       # 洩漏頻帶算單一視窗內的時間結構特徵
        source_note="step2_band_discovery_with_resonance",
        schema_status="candidate",
    )

    print(f"✅ Step2a 完成，保留 {len(selected_bands)} 個候選頻帶")
    return {
        "discovery_bands": discovery_bands,
        "selected_bands": selected_bands,
        "resonance_bands": resonance_bands,
        "ranking_df": ranking_df,
        "band_summary_df": band_summary_df,
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
    1. 若 feature_cols = None：自動抓數值特徵，適合 PCA / QC / 全體分析
    2. 若 feature_cols 有指定：只聚合指定特徵，適合 selected features 分析
    3. 可選 extra_meta_df：額外接回 leak_quantity / distance_cm / pressure 等 metadata
    """
    df = window_df.copy()    # 複製資料

    if exclude_cols is None:
        exclude_cols = NON_FEATURE_COLS     # 不是特徵的欄位（統一定義在設定區）

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

    # 接回 QC 結果（只接 record_df 還沒有的欄位，避免同名欄位變成 _x / _y）
    if qc_record_df is not None:
        qc_cols = [
            c for c in [
                "filename", "condition", "group",
                "leak_quantity", "distance_cm", "pressure",
                "decision", "soft_score", "ratio_warning",
                "ratio_reason", "reason", "family_reason",
            ]
            if c in qc_record_df.columns and c not in record_df.columns
        ]
        if qc_cols:
            qc_map = qc_record_df[[record_id_col] + qc_cols].drop_duplicates(subset=[record_id_col])
            record_df = record_df.merge(qc_map, on=record_id_col, how="left")

    # 接回額外 metadata（洩漏量、距離、壓力、場次等），同樣只接還沒有的欄位
    if extra_meta_df is not None:
        extra_cols = [
            c for c in extra_meta_df.columns
            if c != record_id_col and c not in record_df.columns
        ]
        if extra_cols:
            extra_map = extra_meta_df[[record_id_col] + extra_cols].drop_duplicates(subset=[record_id_col])
            record_df = record_df.merge(extra_map, on=record_id_col, how="left")

    return record_df


# =============================================
# 診斷：每個特徵單獨看，能不能分開「正常 vs 指定洩漏量」
# =============================================
def diagnose_baseline_vs_30(record_df,                              # 已經整理好的 record-level 特徵表
                            out_dir,                                # 輸出資料夾
                            top_n=30,                               # 最後要印出前幾名特徵，預設 30 個
                            leak_level=ANALYSIS_LEAK_LEVELS[0],     # 要和正常比較的洩漏量（預設讀設定區，目前是 30）
                            ):     
    """
    逐一檢查每個特徵，在 record-level 上能不能分開「正常（0 ml）vs 洩漏（leak_level）」。
    同時算 Cohen's d 和 AUC：
      - Cohen's d：差了幾個標準差，容易被少數極端值撐大
      - AUC：只看排序，不受極端值影響；0.5 = 亂猜，1.0 = 完美分開
      兩者都高的特徵最可靠。
    結果只輸出報表，不影響選特徵。
    """
    
    # 把輸出路徑轉成 Path 物件，然後確保輸出資料夾存在
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 篩選資料 -> 只保留正常和指定洩漏量
    df = record_df[record_df["leak_quantity"].isin([0, leak_level])].copy()

    # 不是特徵的欄位（統一定義在設定區）
    exclude_cols = NON_FEATURE_COLS

    # 自動找出要分析的特徵欄(條件是：特徵是數值型，且不在 exclude_cols 裡)
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
    ]
    
    # 先準備一個空 list，等等每個特徵算出來的結果會一列一列放進去
    rows = []
    # 開始逐個特徵處理
    for feat in feature_cols:
        x0 = df.loc[df["leak_quantity"] == 0, feat].dropna().values     # x0：正常 Baseline 組的值
        x1 = df.loc[df["leak_quantity"] == leak_level, feat].dropna().values    # x1：洩漏組的值

        # 任一組少於 3 筆就不算（統計量不可靠）
        if len(x0) < 3 or len(x1) < 3:
            continue

        # 算這個特徵在 30 ml 相對 Baseline 的 Cohen's d(Cohen's d 是效果量，代表兩組差多大)
        d = _cohen_d(x1, x0)      # 正值：洩漏比正常大；負值：洩漏比正常小

        # 計算 AUC：正反兩個方向都試，取比較好的（所以一定 ≥ 0.5）
        y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]    # y：真實標籤 Baseline = 0 , 30 ml = 1
        v = np.r_[x0, x1]    # 把兩組特徵值接起來

        try:    # 有些特徵是：30 ml 越大越像 leak，有些特徵可能相反：30 ml 越小越像 leak
            auc_pos = roc_auc_score(y, v)     # 洩漏比較高時的分辨力
            auc_neg = roc_auc_score(y, -v)    # 洩漏比較低時的分辨力
            auc_best = max(auc_pos, auc_neg)    # 再取其中比較大的，這樣就能不管方向，先看「這個特徵最多能分多好」
            # 同時 direction 也會記錄方向:"higher_in_leak"：30 ml 較高比較有辨識力 , "lower_in_leak"：30 ml 較低比較有辨識力
            direction = "higher_in_leak" if auc_pos >= auc_neg else "lower_in_leak"
        except Exception:
            auc_best = np.nan
            direction = "unknown"

        rows.append({
            "feature": feat,                                                  # 特徵名
            "n_baseline": len(x0),                                            # Baseline 樣本數
            "n_leak": len(x1),                                                # 30 ml 樣本數
            "median_baseline": np.median(x0),                                 # Baseline 中位數
            "median_leak": np.median(x1),                                     # 30 ml 中位數
            "median_gap_leak_minus_baseline": np.median(x1) - np.median(x0),  # 中位數差值
            "cohen_d_leak_vs_baseline": d,                                    # 效果量
            "abs_cohen_d": abs(d) if pd.notna(d) else np.nan,                 # 效果量絕對值
            "auc_best_direction": auc_best,                                   # 最佳方向下的 AUC
            "direction": direction,                                           # 這個特徵是 30 ml 較高還是較低
        })
        
    if not rows:
        print(f"⚠️ 正常或 {leak_level} ml 的資料不足，無法做診斷")
        return pd.DataFrame()

    # 把前面收集的結果轉成 DataFrame，並排序(auc_best_direction 高的排前面, 若 AUC 相同，再看 abs_cohen_d)
    diag_df = (
        pd.DataFrame(rows)
        .sort_values(["auc_best_direction", "abs_cohen_d"], ascending=False)
        .reset_index(drop=True)
    )
    # 把診斷結果存成 CSV
    diag_df.to_csv(out_dir / f"diagnose_baseline_vs_{leak_level}_features.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== Baseline vs {leak_level} ml 診斷 ===")
    print(diag_df.head(top_n).to_string(index=False))

    return diag_df


# ============================================================
# 哪些特徵最能分開「正常 vs 洩漏」（record-level 排名）
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

    # 不是特徵的欄位（統一定義在設定區）
    exclude_cols = NON_FEATURE_COLS

    # 找出候選特徵欄位:只要是數值欄位，而且不在排除名單裡，就視為候選特徵
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
    ]

    # 準備分組資訊
    rows = []    # 儲存每個特徵的評分結果
    sorted_levels = sorted(target_levels)
    base_level = 0     # baseline/正常 當比較基準
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

    # ============================================================
    # 總分
    #   只有一種洩漏量：score = |Cohen's d|（正常 vs 洩漏）
    #   多種洩漏量  ：最小的洩漏量最重要（微小洩漏是偵測目標）占 70%，
    #                 相鄰洩漏量之間的區分力平均占 30%（看特徵能不能分出洩漏大小）
    # ============================================================
    if not leak_levels:
        raise ValueError("target_levels 至少要有一種洩漏量（非 0）")

    min_leak = int(min(leak_levels))                  # 最小的洩漏量，例如 30
    main_col = f"abs_d_0_{min_leak}"                  # 正常 vs 最小洩漏量的區分力

    # 相鄰等級之間的區分力（例如 0→30、30→60），取平均
    adjacent_cols = [
        f"abs_d_{int(a)}_{int(b)}"
        for a, b in zip(sorted_levels[:-1], sorted_levels[1:])
    ]
    rank_df["adjacent_effect_mean"] = rank_df[adjacent_cols].fillna(0).mean(axis=1)

    if len(leak_levels) == 1:
        rank_df["leak_effect_score"] = rank_df[main_col].fillna(0)
    else:
        rank_df["leak_effect_score"] = (
            0.7 * rank_df[main_col].fillna(0)
            + 0.3 * rank_df["adjacent_effect_mean"]
        )

    # 通用欄名 score：下游（build_feature_importance_table 等）都讀這一欄
    rank_df["score"] = rank_df["leak_effect_score"]

    # 依總分排序，同分時看「正常 vs 最小洩漏量」的區分力
    rank_df = rank_df.sort_values(["score", main_col], ascending=False).reset_index(drop=True)

    show_cols = ["feature", "score", f"d_0_{min_leak}", "rho_leakqty", "kw_stat"]
    print(f"\n=== 特徵排名：正常 vs 洩漏 {[int(lv) for lv in leak_levels]} ml（前 20 名）===")
    print(rank_df[show_cols].head(20).to_string(index=False))

    return rank_df


# ============================================================
# 在每個分組（距離、日期……）內分別比較正常 vs 洩漏，檢查特徵換了條件是否還穩定
# ============================================================
def rank_features_within_each_group(
        record_df,                              # record-level 特徵表（每支檔案一列）
        group_col,                              # 分組欄位，例如 "distance_cm"、"file_date"
        prefix,                                 # 輸出欄位的前綴，例如 "distance"、"date"
        target_levels=tuple(ANALYSIS_LEVELS),   # 要比較的洩漏量（含 0 = 正常），預設讀設定區
        min_samples=3,                          # 每個分組裡，正常和洩漏各至少要幾筆才計算
        ):
    
    """
    在每個分組內各自比較「正常 vs 洩漏」，再看特徵在不同分組之間是否一致：
      {prefix}_mean_abs_d        ：所有分組的平均區分力
      {prefix}_min_abs_d         ：最差那個分組的區分力（有沒有某個條件抓不到）
      {prefix}_sign_consistency  ：方向一致的比例（例如每個距離都是「洩漏比較高」→ 1.0）
      {prefix}_stability_score   ：0.45 × 平均 + 0.35 × 最差 + 0.20 × 方向一致
    只有「同一個分組裡同時有正常和洩漏」的分組才會被計算。
    """
    # ---------- 1. 準備資料 ----------
    # 只保留要比較的洩漏量
    df = record_df[record_df["leak_quantity"].isin(target_levels)].copy()
    if group_col not in df.columns:
        raise ValueError(f"record_df 沒有分組欄位 {group_col}")

    # 要檢查的特徵：數值欄位，且不是識別 / 標籤 / 條件 / 時間等欄位（統一定義在設定區）
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in NON_FEATURE_COLS
    ]
    groups = sorted(df[group_col].dropna().unique())          # 所有分組，例如 [0, 20, 50, 70, 90]（cm）
    leak_levels = [lv for lv in target_levels if lv != 0]     # 只留洩漏量，例如 [30]

    # 先算好每個「分組 × 洩漏量」的資料位置（True / False 遮罩），後面幾千個特徵的迴圈直接拿來用，不用每次重新篩選，速度快很多
    # 例如 masks[(20, 0)]  = 「20cm 的正常資料」在哪幾列
    #     masks[(20, 30)] = 「20cm 的 30 ml 洩漏」在哪幾列
    masks = {
        (g, lv): (df[group_col] == g) & (df["leak_quantity"] == lv)
        for g in groups
        for lv in [0] + leak_levels
    }

    # ---------- 2. 逐一特徵計算 ----------
    rows = []
    for feat in feature_cols:
        row = {"feature": feat}
        # 初始化特徵收集器
        all_abs_d = []       # 收集所有分組、所有洩漏量的 |Cohen's d|（算整體平均與最差）
        sign_hits = []       # 收集每個洩漏量的方向一致性（算整體方向一致性）

        # 逐個 leak level 處理
        for lv in leak_levels:
            # (A) 每個分組各自算一次 Cohen's d
            d_vals = []
            for g in groups:
                x0 = df.loc[masks[(g, 0)], feat].dropna().values      # 這個分組的正常資料
                x1 = df.loc[masks[(g, lv)], feat].dropna().values     # 這個分組的洩漏資料
                if len(x0) < min_samples or len(x1) < min_samples:
                    continue                                          # 這個分組資料不夠（或只有一種類別），跳過
                d = _cohen_d(x1, x0)                                  # 正值：洩漏較高；負值：洩漏較低
                if pd.notna(d):
                    d_vals.append(d)
                    
            # 記錄實際有幾個分組能計算（例如 5 個距離都有資料 → 5）
            row[f"{prefix}_n_groups_0_{int(lv)}"] = len(d_vals)
            if not d_vals:
                continue        

            # (B) 整理這個洩漏量在各分組的結果
            d_vals = np.asarray(d_vals, dtype=float)
            # 方向一致性：「洩漏較高」和「洩漏較低」哪一種占多數，就取那個比例
            # 例如 5 個距離中 4 個 d > 0、1 個 d < 0 → 0.8
            consistency = max(np.mean(d_vals > 0), np.mean(d_vals < 0))
            row[f"{prefix}_mean_abs_d_0_{int(lv)}"] = np.mean(np.abs(d_vals))    # 各分組平均區分力
            row[f"{prefix}_min_abs_d_0_{int(lv)}"] = np.min(np.abs(d_vals))      # 最差分組的區分力
            row[f"{prefix}_sign_consistency_0_{int(lv)}"] = consistency          # 方向一致性
            all_abs_d.extend(np.abs(d_vals).tolist())
            sign_hits.append(consistency)
            
        # (C) 彙整所有洩漏量：這個特徵整體的平均、最差、方向一致性
        row[f"{prefix}_mean_abs_d"] = np.mean(all_abs_d) if all_abs_d else np.nan
        row[f"{prefix}_min_abs_d"] = np.min(all_abs_d) if all_abs_d else np.nan
        row[f"{prefix}_sign_consistency"] = np.mean(sign_hits) if sign_hits else np.nan
        rows.append(row)
        
    out = pd.DataFrame(rows)
    
    # ---------- 3. 檢查是否完全無法評估 ----------
    # 沒有任何分組同時有正常和洩漏時（例如舊資料每天只有一種類別），所有值都是 NaN；
    # 印出提示，之後分數會全部變成 0，等於不影響排名
    if out[f"{prefix}_mean_abs_d"].isna().all():
        print(f"ℹ️ 沒有任何 {group_col} 同時有正常與洩漏，無法評估跨 {prefix} 穩定性")

    # ---------- 4. 穩定分數與排序 ----------
    out[f"{prefix}_stability_score"] = (
        0.45 * out[f"{prefix}_mean_abs_d"].fillna(0)            # 平均區分力：整體有沒有用
        + 0.35 * out[f"{prefix}_min_abs_d"].fillna(0)           # 最差分組：有沒有某個條件完全抓不到
        + 0.20 * out[f"{prefix}_sign_consistency"].fillna(0)    # 方向一致：換條件後趨勢會不會反過來
    )
    # 分數越高，代表這個特徵換了距離（或日期）之後越穩定
    out = out.sort_values(f"{prefix}_stability_score", ascending=False).reset_index(drop=True)

    print(f"\n=== 跨 {prefix} 穩定的特徵（前 20 名）===")
    print(out[[
        "feature",
        f"{prefix}_stability_score",
        f"{prefix}_mean_abs_d",
        f"{prefix}_min_abs_d",
        f"{prefix}_sign_consistency",
    ]].head(20).to_string(index=False))

    return out


# ============================================================
# 跨距離穩定性（保留原本的函式名稱，主程式的呼叫不用改）
# 跨日期穩定性則直接呼叫：rank_features_within_each_group(record_df, group_col="file_date", prefix="date")
# ============================================================
def rank_features_within_each_distance(record_df, target_levels=tuple(ANALYSIS_LEVELS), min_samples=3):
    return rank_features_within_each_group(
        record_df, group_col="distance_cm", prefix="distance",
        target_levels=target_levels, min_samples=min_samples,
    )


# ============================================================
# 固定洩漏量下比較不同距離，檢查特徵受距離影響多大（只作報表參考）
# ============================================================
def rank_features_within_each_leak_level(
    record_df,                              # record-level 特徵表（每支檔案一列）
    target_levels=tuple(ANALYSIS_LEVELS),   # 要分析的洩漏量（主程式傳入的是只有洩漏的等級，例如 (30,)）
    min_samples=3,                          # 每個距離至少要幾筆才計算
):
    """
    固定一種洩漏量，比較不同距離之間的差異：分數越高 = 越受距離影響。

    注意：洩漏聲會隨距離衰減，所以「越能抓到洩漏的特徵，通常也越受距離影響」，
    受距離影響本身不一定是壞事。換了距離還能不能分開正常與洩漏，
    請看 rank_features_within_each_distance 的 distance_min_abs_d 與 distance_sign_consistency。
    這個分數建議只作報表參考，不用來扣分。
    """
    # ---------- 1. 準備資料 ----------
    df = record_df[record_df["leak_quantity"].isin(target_levels)].copy()

    # 要檢查的特徵（不是特徵的欄位統一定義在設定區）
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in NON_FEATURE_COLS
    ]
    sorted_levels = sorted(target_levels)

    # 先算好每個「洩漏量」的距離清單，以及每個「洩漏量 × 距離」的資料位置（加速用）
    dist_by_level = {
        lv: sorted(df.loc[df["leak_quantity"] == lv, "distance_cm"].dropna().unique())
        for lv in sorted_levels
    }
    masks = {
        (lv, dist): (df["leak_quantity"] == lv) & (df["distance_cm"] == dist)
        for lv in sorted_levels
        for dist in dist_by_level[lv]
    }
    level_masks = {lv: df["leak_quantity"] == lv for lv in sorted_levels}

    # ---------- 2. 逐一特徵計算 ----------
    rows = []
    for feat in feature_cols:
        row = {"feature": feat}
        all_pair_abs_d = []     # 所有洩漏量、所有距離配對的 |Cohen's d|
        all_rho = []            # 每個洩漏量下，特徵和距離的單調關係強度 |rho|
        all_kw = []             # 每個洩漏量下，各距離分布差異的 Kruskal-Wallis 統計量

        for lv in sorted_levels:
            dist_levels = dist_by_level[lv]

            # 這個洩漏量下，每個距離的特徵值（先取好，下面的兩兩比較和 Kruskal 共用）
            values_by_dist = {
                dist: df.loc[masks[(lv, dist)], feat].dropna().values
                for dist in dist_levels
            }

            # (A) 兩兩距離比較：例如 0cm vs 20cm、0cm vs 50cm……，各算一次 |Cohen's d|
            pair_abs_d_vals = []
            for i, d0 in enumerate(dist_levels[:-1]):
                for d1 in dist_levels[i + 1:]:
                    x0, x1 = values_by_dist[d0], values_by_dist[d1]
                    if len(x0) < min_samples or len(x1) < min_samples:
                        continue
                    d_val = _cohen_d(x1, x0)
                    if pd.notna(d_val):
                        pair_abs_d_vals.append(abs(d_val))

            if pair_abs_d_vals:
                row[f"mean_abs_d_dist_{int(lv)}"] = np.mean(pair_abs_d_vals)    # 不同距離之間平均差多少
                row[f"max_abs_d_dist_{int(lv)}"] = np.max(pair_abs_d_vals)      # 差最多的那一對距離
                all_pair_abs_d.extend(pair_abs_d_vals)
            else:
                row[f"mean_abs_d_dist_{int(lv)}"] = np.nan
                row[f"max_abs_d_dist_{int(lv)}"] = np.nan

            # (B) 單調關係：距離越遠，特徵是不是系統性地變大或變小（例如能量隨距離衰減）
            sub = df.loc[level_masks[lv], ["distance_cm", feat]]
            try:
                if sub["distance_cm"].nunique() >= 2 and sub[feat].notna().sum() >= min_samples:
                    rho, _ = spearmanr(sub["distance_cm"], sub[feat], nan_policy="omit")
                else:
                    rho = np.nan
            except Exception:
                rho = np.nan
            row[f"rho_distance_{int(lv)}"] = rho
            if pd.notna(rho):
                all_rho.append(abs(rho))

            # (C) 整體分布差異：一次比較所有距離（兩兩比較的補充）
            try:
                groups = [v for v in values_by_dist.values() if len(v) >= min_samples]
                kw_stat, kw_p = kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)
            except Exception:
                kw_stat, kw_p = np.nan, np.nan
            row[f"kw_stat_dist_{int(lv)}"] = kw_stat
            row[f"kw_p_dist_{int(lv)}"] = kw_p
            if pd.notna(kw_stat):
                all_kw.append(kw_stat)

        # (D) 彙整所有洩漏量
        row["distance_pair_mean_abs_d"] = np.mean(all_pair_abs_d) if all_pair_abs_d else np.nan
        row["distance_pair_max_abs_d"] = np.max(all_pair_abs_d) if all_pair_abs_d else np.nan
        row["distance_rho_mean"] = np.mean(all_rho) if all_rho else np.nan
        row["distance_kw_mean"] = np.mean(all_kw) if all_kw else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        print("⚠️ 沒有足夠資料做固定洩漏量的跨距離比較")
        return out

    # ---------- 3. 距離敏感度分數 ----------
    # Kruskal 統計量的數值範圍很大，先除以最大值縮到 0～1
    kw_max = out["distance_kw_mean"].replace([np.inf, -np.inf], np.nan).max()
    out["distance_kw_mean_norm"] = out["distance_kw_mean"] / kw_max if pd.notna(kw_max) and kw_max > 0 else 0.0

    # 分數越高 = 越受距離影響
    out["distance_sensitivity_score"] = (
        0.45 * out["distance_pair_mean_abs_d"].fillna(0)      # 不同距離之間平均差多少
        + 0.25 * out["distance_pair_max_abs_d"].fillna(0)     # 差最多的那一對距離
        + 0.15 * out["distance_rho_mean"].fillna(0)           # 是否隨距離單調變化
        + 0.15 * out["distance_kw_mean_norm"].fillna(0)       # 各距離整體分布差異
    )
    out = out.sort_values("distance_sensitivity_score", ascending=False).reset_index(drop=True)

    print("\n=== 受距離影響最大的特徵（同一洩漏量下，前 20 名）===")
    print(out[[
        "feature",
        "distance_sensitivity_score",
        "distance_pair_mean_abs_d",
        "distance_pair_max_abs_d",
        "distance_rho_mean",
        "distance_kw_mean",
    ]].head(20).to_string(index=False))

    return out


# ============================================================
# 去除高共線性特徵：依分數由高到低，和已保留的太像就跳過
# ============================================================
def remove_collinear_features(
        X,                          # 候選特徵資料（每欄一個特徵）
        cohen_dict,                 # 每個特徵的分數（分數高的優先保留）
        threshold=CORR_THRESHOLD,   # 相關係數超過多少算「太像」，預設讀設定區
        method="spearman",          # 相關係數算法：spearman 只看排序，能抓到 E 和 RMS 這種非直線的關係
        verbose=True,
        save_path=None,
):
    """
    依分數由高到低逐一檢查每個特徵：
      - 和「已經保留的特徵」相關係數都不超過 threshold → 保留
      - 和其中任何一個太像 → 刪除（因為已保留的那個分數比較高）
    相關係數只算一次（兩個特徵之間的相關，不會因為刪掉別的特徵而改變）。

    回傳 (保留後的資料, 被刪除的特徵清單, 刪除紀錄 [(被刪特徵, 因為和誰太像, 相關係數)])
    """
    # 從表格 X 裡，挑出所有「數值型」欄位的名稱，存成一個清單
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("⚠️ 沒有數值特徵")
        return X, [], []

    if verbose:
        print("\n" + "=" * 60)
        print(f"去除高共線性特徵（{method}，門檻 {threshold}，分數高者優先保留）")
        print("=" * 60)
        print(f"原始特徵數: {len(numeric_cols)}")

    # 相關係數矩陣只算一次，取絕對值（正相關、負相關一樣算「太像」）
    corr = X[numeric_cols].corr(method=method).abs()

    # 依分數由高到低排序；同分時依名稱排序，確保每次結果一致
    order = sorted(numeric_cols, key=lambda f: (-cohen_dict.get(f, 0.0), f))

    kept, dropped_features, corr_pairs_history = [], [], []
    for feat in order:
        # 這個特徵和哪些已保留的特徵太像（相關係數是 NaN 時視為不像，例如整欄都是同一個值）
        too_close = [k for k in kept if corr.loc[feat, k] > threshold]
        if too_close:
            best = max(too_close, key=lambda k: corr.loc[feat, k])      # 最像的那一個
            dropped_features.append(feat)
            corr_pairs_history.append((feat, best, corr.loc[feat, best]))
        else:
            kept.append(feat)

    # 保留原本的欄位順序
    X_reduced = X.drop(columns=dropped_features)

    if verbose:
        print(f"\n移除 {len(dropped_features)} 個特徵（顯示前 20 個）：")
        for feat, best, c in corr_pairs_history[:20]:
            print(f"  - {feat}  ← 和 {best} 太像（{c:.3f}）")
        if len(dropped_features) > 20:
            print(f"  ... 還有 {len(dropped_features) - 20} 個")
        print(f"\n✅ 特徵數: {len(numeric_cols)} → {len(kept)}")

    # 輸出報表
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 被刪除的特徵，以及它是因為和哪個保留特徵太像
        pd.DataFrame(
            corr_pairs_history, columns=["dropped_feature", "kept_feature", "correlation"]
        ).to_csv(save_path / f"corr_dropped_{timestamp}.csv", index=False, encoding="utf-8-sig")

        # 保留的特徵（依分數由高到低）
        pd.DataFrame({
            "feature": kept,
            "score": [cohen_dict.get(f, 0.0) for f in kept],
        }).to_csv(save_path / f"corr_retained_{timestamp}.csv", index=False, encoding="utf-8-sig")

    return X_reduced, dropped_features, corr_pairs_history


# ============================================================
# 模型重要性：把一個特徵打亂後，模型在「沒看過的場次」上掉多少分
# ============================================================
def compute_model_feature_importance(
    df_dev,                        # dev 的 window-level 資料（含候選特徵與 label）
    candidate_features,            # 要評估的特徵（建議先去共線性，只剩彼此不同的特徵）
    groups=None,                   # 分組欄位的值（建議傳 session_id），同一組不會同時出現在訓練與驗證
    random_state=RANDOM_STATE,     # 隨機種子：固定後每次結果都一樣
    n_splits=5,                    # 交叉驗證折數
    n_repeats=5,                   # 每個特徵打亂幾次取平均（越多越穩，但越慢）
):
    """
    Permutation importance（打亂重要性）：
      1. 依 groups 分成訓練折與驗證折（驗證折是模型沒看過的場次）
      2. 在訓練折訓練隨機森林
      3. 在驗證折上，把某個特徵的值隨機打亂，看 AUC 掉多少
      4. 各折平均
    掉越多 = 越重要；打亂後分數沒變（≤ 0）= 這個特徵沒有提供額外資訊。

    和隨機森林內建的 feature_importances_ 相比：
      - 用沒看過的場次評估，不會因為模型記住特定場次而高估
      - 重複的特徵不會互相瓜分分數（打亂一個，模型還能用另一個，所以多餘的分數會很低）
    """
    from sklearn.inspection import permutation_importance

    # ---------- 1. 準備資料 ----------
    X = df_dev[candidate_features]
    y = df_dev["label"].values
    
    # 分組：同一組的視窗一定一起在訓練折或一起在驗證折
    if groups is None:
        groups = df_dev["record_id"].values           # 沒指定時退回以檔案分組（建議傳 session_id，避免同場次跨兩邊）
    groups = np.asarray(groups)

    # 折數受「每一類各有幾個獨立組」限制：每個場次只有一種 label，
    # 若折數 = 總組數，每折只剩一個場次 → 驗證折只有一種類別，所有折都會被跳過（和 fold-wise CV 同一套寫法）
    groups_per_class = {int(lbl): len(np.unique(groups[y == lbl])) for lbl in np.unique(y)}
    n_splits = min(n_splits, *groups_per_class.values())
    if n_splits < 2:
        raise ValueError(f"模型重要性：每一類至少需要 2 個獨立組，目前：{groups_per_class}")
    
    # StratifiedGroupKFold：同一組不拆開（Group），並盡量讓每折的正常 / 洩漏比例相近（Stratified）
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ---------- 2. 逐折計算 ----------
    importance_sum = np.zeros(len(candidate_features))      # 累加每折的重要性，最後取平均
    n_done = 0                                              # 實際完成幾折（有些折可能被跳過）
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        # 驗證折只有一種類別時算不了 AUC（AUC 需要同時有正常和洩漏），跳過這一折
        if len(np.unique(y[val_idx])) < 2:
            print(f"  ⚠️ 模型重要性第 {fold} 折的驗證資料只有一種類別，跳過")
            continue

        # (A) 在訓練折訓練隨機森林（隨機森林不需要標準化，直接用原始數值）
        clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=N_JOBS)
        clf.fit(X.iloc[train_idx], y[train_idx])

        # (B) 在驗證折上做打亂測試：
        #     每個特徵輪流被打亂 n_repeats 次，記錄 AUC 平均掉了多少
        result = permutation_importance(
            clf, X.iloc[val_idx], y[val_idx],
            scoring="roc_auc",                         # 用 AUC 評分：不受分類門檻影響，比 F1 穩定
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=N_JOBS,
        )
        importance_sum += result.importances_mean      # importances_mean：每個特徵打亂後 AUC 平均掉多少
        n_done += 1

    # 所有折都被跳過（例如資料太少、分組太集中）就無法計算
    if n_done == 0:
        raise ValueError("所有交叉驗證折都只有一種類別，無法計算模型重要性")

    # ---------- 3. 整理結果 ----------
    # 各折平均；負值代表打亂後分數反而變好（等於這個特徵沒用），統一當成 0
    importance = np.clip(importance_sum / n_done, 0, None)
    print(f"✅ 模型重要性（permutation，{n_done} 折平均）完成：{len(candidate_features)} 個特徵")

    # 回傳 {特徵名稱: 重要性}，給 combine_statistical_and_model_scores 合併統計分數用
    return dict(zip(candidate_features, importance))


# ============================================================
# 合併統計分數與模型重要性
# ============================================================
def combine_statistical_and_model_scores(
    statistical_score_dict,    # 統計分數 {特徵: 分數}，例如 final_score
    model_importance_dict,     # 模型重要性 {特徵: 分數}，compute_model_feature_importance 的輸出
    stat_weight=0.5,           # 統計分數的權重
    model_weight=0.5,          # 模型重要性的權重
):
    """
    把統計分數和模型重要性合併成一個綜合分數，用來決定最終特徵的排序。

    - 只合併「兩邊都有分數」的特徵（模型重要性通常只算了一部分特徵）
    - 兩者數值範圍完全不同，先各自轉成「排名百分比」（最高 = 1.0，最低接近 0）再加權：
      只看排名，不會因為某一個特徵分數特別高，就把其他特徵的差距壓扁

    回傳三個 dict：(綜合分數, 統計分數排名百分比, 模型重要性排名百分比)
    """
    # 只取兩邊都有分數的特徵，順序依模型重要性的輸入順序
    features = [f for f in model_importance_dict if f in statistical_score_dict]
    if not features:
        raise ValueError("統計分數與模型重要性沒有共同的特徵")

    missing = [f for f in model_importance_dict if f not in statistical_score_dict]
    if missing:
        print(f"⚠️ 有 {len(missing)} 個特徵只有模型重要性、沒有統計分數，已略過，例如 {missing[:5]}")

    stat_vals = pd.Series([statistical_score_dict[f] for f in features], index=features, dtype=float)
    model_vals = pd.Series([model_importance_dict[f] for f in features], index=features, dtype=float)

    # 轉成排名百分比：分數最高 = 1.0；同分給相同的排名
    stat_norm = stat_vals.rank(pct=True, method="average")
    model_norm = model_vals.rank(pct=True, method="average")

    combined = stat_weight * stat_norm + model_weight * model_norm

    return (
        combined.to_dict(),       # 綜合分數：用來決定最終特徵的排序
        stat_norm.to_dict(),      # 統計分數的排名百分比（輸出報表用）
        model_norm.to_dict(),     # 模型重要性的排名百分比（輸出報表用）
    )


# ============================================================
# 正式選特徵流程（主程式與 fold-wise CV 共用同一套）
# ============================================================
def select_model_features(
    df_windows,                            # window-level 資料（要有 label、record_id、session_id 與候選特徵）
    record_meta_df,                        # 每支檔案的 metadata（record_id、label、leak_quantity、distance_cm、file_date…）
    eligible_features,                     # 可以進模型的特徵清單（已排除跨窗、offband、runtime 算不出來的）
    *,                                     # 後面的參數都要用「名稱=值」的方式傳入，避免傳錯位置
    groups_col="session_id",               # 模型重要性的交叉驗證以什麼分組（同一組不會同時在訓練與驗證）
    target_levels=tuple(ANALYSIS_LEVELS),  # 要比較的洩漏量（含 0 = 正常）
    top_k=TOP_K_CANDIDATES,                # 去共線性前，取綜合分數前幾名
    corr_threshold=CORR_THRESHOLD,         # 共線性門檻：相關係數超過這個值算「太像」
    random_state=RANDOM_STATE,             # 隨機種子：固定後每次結果都一樣
    verbose=True,                          # False 時不印中間過程（fold-wise CV 用）
    save_dir=None,                         # 有指定才輸出共線性報表
):
    """
    完整的選特徵流程：
      1. 每支檔案壓成一列（中位數）
      2. 洩漏區分力排名（score = |Cohen's d|）
      3. 跨距離穩定性
      4. 跨天穩定性（有多天、而且同一天有正常和洩漏時才計算）
      5. 綜合分數 final_score
      6. 方向一致性篩選（各距離 / 各天的方向要一致）
      7. 取前 top_k 名 → 去共線性（分數高的優先保留）
      8. 剩下的特徵算模型重要性（打亂測試法，以 groups_col 分組）
      9. 統計分數與模型重要性合併，決定最終排序
    回傳 dict：selected_features（最終特徵，依重要性排序）、rank_df（所有可用特徵的各項分數）、final_df（最終特徵的分數）
    """
    import contextlib
    import io

    # verbose=False 時，把內部函式印出的訊息丟掉（fold-wise CV 每一折都會跑一次，不然畫面會被洗版）
    # verbose=True 時用 nullcontext()，等於什麼都不做，訊息照常印出
    quiet = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
    with quiet:
        # ---------- 1. 每支檔案壓成一列 ----------
        # 只處理可用特徵（不用把幾千個欄位都聚合），速度快很多
        feature_cols = [c for c in eligible_features if c in df_windows.columns]
        record_df = build_record_level_feature_table(
            window_df=df_windows,
            feature_cols=feature_cols,            
            extra_meta_df=record_meta_df,       # 接回洩漏量、距離、日期等資訊
            agg="median",                       # 同一支檔案的 7 個視窗取中位數
        )
        # 只保留要比較的洩漏量
        record_df = record_df[record_df["leak_quantity"].isin(target_levels)].copy()

        # ---------- 2. 洩漏區分力：每個特徵分開正常和洩漏的能力（score = |Cohen's d|）----------
        rank_df = rank_features_for_leak_levels(record_df, target_levels=target_levels)

        # ---------- 3. 跨距離穩定性：換了距離，還能不能分開正常和洩漏 ----------
        distance_df = rank_features_within_each_distance(record_df, target_levels=target_levels)
        rank_df = rank_df.merge(distance_df, on="feature", how="left")

        # ---------- 4. 跨天穩定性：換了一天，還能不能分開正常和洩漏 ----------
        # 只有「同一天同時有正常和洩漏」的日期才能比較；
        # 舊資料每天只有一種類別，算出來全是 NaN，這時就不納入
        use_date = False
        if "file_date" in record_df.columns:
            date_df = rank_features_within_each_group(
                record_df, group_col="file_date", prefix="date", target_levels=target_levels,
            )
            if date_df["date_mean_abs_d"].notna().any():
                rank_df = rank_df.merge(date_df, on="feature", how="left")
                use_date = True

        # ---------- 5. 綜合分數 ----------
        # 有跨天資料：區分力 50% + 跨距離 25% + 跨天 25%
        # 沒有跨天資料：區分力 60% + 跨距離 40%
        # （不再扣「距離敏感度」：洩漏聲隨距離衰減是物理現象，換距離還能不能分開，已由跨距離穩定性處理）
        if use_date:
            rank_df["final_score"] = (
                0.50 * rank_df["score"].fillna(0)
                + 0.25 * rank_df["distance_stability_score"].fillna(0)
                + 0.25 * rank_df["date_stability_score"].fillna(0)
            )
        else:
            rank_df["final_score"] = (
                0.60 * rank_df["score"].fillna(0)
                + 0.40 * rank_df["distance_stability_score"].fillna(0)
            )

        # ---------- 6. 方向一致性篩選 ----------
        # 例如某個特徵在 20cm 是「洩漏較高」、在 90cm 卻是「洩漏較低」，到現場就不可靠，直接淘汰
        # 要求至少 MIN_SIGN_CONSISTENCY（0.67）的距離 / 天數方向一致
        # 整欄都是 NaN（完全無法評估）時不篩，避免把所有特徵都刪光
        for col in ["distance_sign_consistency"] + (["date_sign_consistency"] if use_date else []):
            if col in rank_df.columns and rank_df[col].notna().any():
                rank_df = rank_df[rank_df[col].fillna(0) >= MIN_SIGN_CONSISTENCY]
        rank_df = rank_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        if rank_df.empty:
            raise ValueError("方向一致性篩選後沒有任何特徵；可調低 MIN_SIGN_CONSISTENCY 再試")

        # ---------- 7. 取前 top_k 名，去共線性 ----------
        # 依綜合分數由高到低檢查，和已保留的特徵太像（例如 x 和 z_x、E 和 RMS）就跳過
        candidates = rank_df["feature"].head(top_k).tolist()
        final_score_dict = dict(zip(rank_df["feature"], rank_df["final_score"]))
        X_reduced, _, _ = remove_collinear_features(
            df_windows[candidates], final_score_dict,
            threshold=corr_threshold, verbose=verbose, save_path=save_dir,
        )
        survivors = X_reduced.columns.tolist()       # 去共線性後留下來、彼此不重複的特徵

        # ---------- 8. 模型重要性 ----------
        # 打亂測試法：把一個特徵打亂後，模型在「沒看過的場次」上 AUC 掉多少
        # 放在去共線性之後算：重複的特徵已經刪掉，重要性不會被互相瓜分，計算量也小很多
        model_importance = compute_model_feature_importance(
            df_windows, survivors,
            groups=df_windows[groups_col].values,
            random_state=random_state,
        )

        # ---------- 9. 合併統計分數與模型重要性 ----------
        # 兩者各自轉成排名百分比後各占一半，決定最終特徵的排序
        combined, stat_norm, model_norm = combine_statistical_and_model_scores(
            {f: final_score_dict[f] for f in survivors}, model_importance,
        )
        final_df = pd.DataFrame({
            "feature": survivors,
            "final_score": [final_score_dict[f] for f in survivors],        # 統計綜合分數
            "model_importance": [model_importance[f] for f in survivors],   # 打亂後 AUC 掉多少
            "stat_score_rank": [stat_norm[f] for f in survivors],           # 統計分數的排名百分比
            "model_importance_rank": [model_norm[f] for f in survivors],    # 模型重要性的排名百分比
            "combined_score": [combined[f] for f in survivors],             # 最終排序依據
        }).sort_values("combined_score", ascending=False).reset_index(drop=True)

    # 印出每一步剩下多少特徵，方便確認哪一步刪掉最多
    if verbose:
        print(f"\n✅ 選特徵完成：可用 {len(feature_cols)} → 方向一致 {len(rank_df)} → "
              f"前 {len(candidates)} 名 → 去共線性後 {len(survivors)} 個"
              + ("（含跨天穩定性）" if use_date else "（無跨天資料，未納入跨天穩定性）"))

    return {
        "selected_features": final_df["feature"].tolist(),   # 最終特徵（依重要性排序）
        "rank_df": rank_df,                                   # 所有通過方向篩選的特徵與各項分數（報表用）
        "final_df": final_df,                                 # 最終特徵的分數明細（報表用）
        "used_date_stability": use_date,                      # 這次有沒有納入跨天穩定性
    }


# ============================================================
# 決定 CV 用什麼分組：能留整天就留整天，否則留整個場次
# ============================================================
def choose_cv_groups(df, min_days=3):
    """
    - 至少 min_days 天，而且每一天都同時有正常和洩漏 → 以日期分組（每折留整天當驗證，最接近推論情境）
    - 否則 → 以錄音場次分組（每折留整個場次當驗證）
    回傳 (每列的分組值, 說明文字)
    """
    if "file_date" in df.columns:
        # 每一天有幾種 label：2 = 同時有正常和洩漏，1 = 只有一種
        labels_per_day = df.groupby("file_date")["label"].nunique()
        if len(labels_per_day) >= min_days and (labels_per_day == 2).all():
            return df["file_date"].values, f"file_date（{len(labels_per_day)} 天，每折留整天）"
    # 舊資料（每天只有一種類別）或天數不夠：退回以場次分組
    return df["session_id"].values, "session_id（每折留整個場次）"


# ============================================================
# Fold-wise CV：每一折只用訓練折跑完整選特徵流程，評估這套流程的真實表現
# ============================================================
def run_group_cv_with_foldwise_feature_selection(
    df_dev,                      # dev 的 window-level 資料（要有 session_id，最好也有 file_date）
    record_meta_dev_df,          # dev 每支檔案的 metadata
    eligible_features,           # 可以進模型的特徵清單（和正式流程同一份）
    *,
    n_splits=5,                  # 交叉驗證折數
    random_state=RANDOM_STATE,
):
    """
    正式流程是「用全部 dev 選特徵，再評估」，分數會偏樂觀（選特徵時已經看過所有資料）。
    這裡每一折都只用訓練折跑 select_model_features()（和正式流程同一套），
    再用驗證折評估，得到的分數才是這套流程在「沒看過的資料」上的真實表現。

    驗證折是整天（資料夠時）或整個場次，不會和訓練折共用同一段錄音。
    回傳 (各折 F1, 各折選到的特徵與分數, 特徵被選中的次數表)
    """
    from sklearn.metrics import f1_score

    # ---------- 1. 決定分組方式與切法 ----------
    groups, group_desc = choose_cv_groups(df_dev)        # 整天或整個場次
    y_all = df_dev["label"].values
    # 折數受「每一類各有幾個獨立組」限制，避免驗證折只有一種類別
    g_arr = np.asarray(groups)
    groups_per_class = {int(lbl): len(np.unique(g_arr[y_all == lbl])) for lbl in np.unique(y_all)}
    n_splits = min(n_splits, *groups_per_class.values())
    if n_splits < 2:
        raise ValueError(f"Fold-wise CV：每一類至少需要 2 個獨立組（{group_desc}），目前：{groups_per_class}")
        
    # StratifiedGroupKFold：同一組不拆開，並盡量讓每折的正常 / 洩漏比例相近
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print("\n" + "=" * 60)
    print(f"Fold-wise CV：分組依 {group_desc}，共 {n_splits} 折")
    print("=" * 60)

    # ---------- 2. 逐折：只用訓練折選特徵 → 訓練 → 驗證折評估 ----------
    fold_scores, fold_selected_features = [], []
    # cv.split 只需要「列數」來切，所以 X 給一個全是 0 的假陣列即可
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y_all)), y_all, groups), start=1):
        df_train = df_dev.iloc[train_idx]      # 訓練折
        df_val = df_dev.iloc[val_idx]          # 驗證折（模型和選特徵都沒看過）

        # 驗證折只有一種類別時，F1 和 AUC 都算不出來
        if df_val["label"].nunique() < 2:
            print(f"Fold {fold_idx}: 驗證折只有一種類別，跳過")
            continue

        # (A) 只用訓練折跑完整選特徵流程（和正式流程同一個函式）
        meta_train = record_meta_dev_df[record_meta_dev_df["record_id"].isin(df_train["record_id"])]
        try:
            result = select_model_features(
                df_train, meta_train, eligible_features,
                random_state=random_state, verbose=False,
            )
        except ValueError as e:
            # 某一折資料太少導致選不出特徵時，跳過這一折，不讓整個 CV 中斷
            print(f"Fold {fold_idx}: 選特徵失敗，跳過（{e}）")
            continue
        selected = result["selected_features"]

        # (B) 用這一折選出的特徵，在訓練折訓練模型
        clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=N_JOBS)
        clf.fit(df_train[selected], df_train["label"])

        # (C) 在驗證折評估
        prob = clf.predict_proba(df_val[selected])[:, 1]     # 每個視窗是洩漏的機率
        y_val = df_val["label"].values

        fold_f1 = f1_score(y_val, (prob >= 0.5).astype(int))  # 以 0.5 為門檻轉成 0 / 1 後算 F1
        fold_auc = roc_auc_score(y_val, prob)                 # AUC 不受門檻影響，更能反映特徵本身的分辨力
        fold_scores.append(fold_f1)
        fold_selected_features.append({
            "fold": fold_idx,
            "n_features": len(selected),
            "features": selected,
            "f1": fold_f1,
            "auc": fold_auc,
        })
        print(f"Fold {fold_idx}: F1 = {fold_f1:.4f}, AUC = {fold_auc:.4f}, 特徵數 = {len(selected)}")

    if not fold_scores:
        raise ValueError("Fold-wise CV 沒有任何一折完成，請檢查資料分組")

    # ---------- 3. 彙整各折結果 ----------
    fold_scores = np.array(fold_scores)
    fold_aucs = np.array([item["auc"] for item in fold_selected_features])
    print(f"\nF1  mean = {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
    print(f"AUC mean = {fold_aucs.mean():.4f} ± {fold_aucs.std():.4f}")

    # ---------- 4. 特徵穩定度：每個特徵在幾折被選中 ----------
    # 次數越多，代表換了訓練資料還是會被選到，越值得信任；只在一折出現的可能是巧合
    feature_freq = {}
    for item in fold_selected_features:
        for feat in item["features"]:
            feature_freq[feat] = feature_freq.get(feat, 0) + 1
    feature_freq_df = (
        pd.DataFrame([{"feature": f, "n_folds_selected": n} for f, n in feature_freq.items()])
        .sort_values(["n_folds_selected", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    print("\n各折都常被選到的特徵（前 20 名）：")
    print(feature_freq_df.head(20).to_string(index=False))

    return fold_scores, fold_selected_features, feature_freq_df


# ============================================================
# 留一個距離驗證：模型完全沒看過某個距離時，還能不能抓到洩漏
# ============================================================
def run_leave_one_distance_out_validation(
    df_dev,                      # dev 的 window-level 資料（至少要有 record_id、label、distance_cm）
    feature_names,               # 固定特徵清單（不重新選特徵時使用）
    base_pipeline,               # 要用的模型流程（每輪都會複製一份乾淨的重新訓練）
    threshold=0.5,               # 機率超過多少算洩漏
    record_meta_df=None,         # 每支檔案的 metadata（要重新選特徵時才需要）
    eligible_features=None,      # 可用特徵清單；有傳就「每輪只用訓練距離重新選特徵」
):
    """
    每次留一個距離當驗證，其餘距離訓練。例如驗證 20cm：
      訓練：0、50、70、90cm；驗證：只有 20cm

    兩種模式：
      - 只傳 feature_names：所有輪都用同一組特徵（特徵是看過全部距離後選的，成績偏樂觀）
      - 另外傳 record_meta_df + eligible_features：每一輪只用訓練的距離重新跑 select_model_features，
        驗證距離完全沒參與選特徵，成績才是真正的「沒看過這個距離」

    同時計算視窗層級與檔案層級（每支檔案的視窗機率取平均）的成績。
    回傳 (每個距離的總表, 每個視窗的預測, 每支檔案的預測)
    """
    # ---------- 1. 檢查 ----------
    # 必要欄位：record_id（檔案層級聚合用）、label（標籤）、distance_cm（決定怎麼切）
    required_cols = {"record_id", "label", "distance_cm"}
    missing = required_cols - set(df_dev.columns)
    if missing:
        raise ValueError(f"df_dev 缺少欄位: {sorted(missing)}")

    # record_meta_df, eligible_features 兩個參數都有傳，才啟用「每輪重新選特徵」
    reselect = record_meta_df is not None and eligible_features is not None
    # 自動找出資料裡有哪些距離，例如 [0, 20, 50, 70, 90]
    distances = sorted(df_dev["distance_cm"].dropna().unique())

    summary_rows = []        # 每個距離一列總結
    window_pred_rows = []    # 每個視窗的預測（存成明細表）
    file_pred_rows = []      # 每支檔案的預測（存成明細表）

    print("\n" + "=" * 60)
    print("Leave-One-Distance-Out：" + ("每輪重新選特徵" if reselect else "使用固定特徵（成績偏樂觀，僅供參考）"))
    print("=" * 60)
    print(f"distance_cm: {[int(d) for d in distances]}")

    # ---------- 2. 每個距離輪流當驗證 ----------
    for dist in distances:
        train_df = df_dev[df_dev["distance_cm"] != dist]     # 訓練：這個距離以外的所有資料
        test_df = df_dev[df_dev["distance_cm"] == dist]      # 驗證：只有這個距離

        # 訓練資料要同時有正常和洩漏，模型才學得起來；任一邊沒資料也無法進行
        if train_df.empty or test_df.empty or train_df["label"].nunique() < 2:
            print(f"[跳過] {int(dist)} cm：訓練或驗證資料不足")
            continue
        # 驗證資料只有一種類別時還是可以算 F1，但 AUC 算不出來
        if test_df["label"].nunique() < 2:
            print(f"[提醒] {int(dist)} cm：驗證資料只有一種類別，AUC 無法計算")

        # (A) 決定這一輪用哪些特徵
        if reselect:
            # 只用訓練距離的資料跑完整選特徵流程（驗證距離完全沒參與）
            meta_train = record_meta_df[record_meta_df["record_id"].isin(train_df["record_id"])]
            try:
                features = select_model_features(
                    train_df, meta_train, eligible_features, verbose=False,
                )["selected_features"]
            except ValueError as e:
                # 訓練資料太少導致選不出特徵時，跳過這一輪，不讓整個驗證中斷
                print(f"[跳過] {int(dist)} cm：選特徵失敗（{e}）")
                continue
        else:
            # 使用主程式傳入的固定特徵（只保留資料裡真的存在的欄位）
            features = [c for c in feature_names if c in df_dev.columns]
            if not features:
                raise ValueError("feature_names 沒有任何欄位存在於 df_dev")

        # (B) 訓練：每輪複製一份乾淨的模型，避免上一輪訓練的狀態殘留
        model = clone(base_pipeline)
        model.fit(train_df[features].values, train_df["label"].values)

        # (C) 視窗層級：每個 0.2 秒視窗分別判斷
        y_test = test_df["label"].values
        y_prob = model.predict_proba(test_df[features].values)[:, 1]      # 每個視窗是洩漏的機率
        # Step 2 僅用固定 0.5 產生輔助 F1／Precision／Recall；
        # 本階段不調整分類門檻，正式偵測門檻與連續窗規則由後續模型訓練／後處理階段定案。
        y_pred = (y_prob >= threshold).astype(int)                        # 超過門檻 → 判為洩漏

        # precision：判為洩漏的視窗中，真的是洩漏的比例（誤報少 → 高）
        # recall   ：真正洩漏的視窗中，被抓到的比例（漏報少 → 高）
        # F1       ：precision 和 recall 的綜合
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        w_auc = _safe_auc(y_test, y_prob)      # 不受門檻影響的分辨力；只有一種類別時回傳 NaN

        # 存每個視窗的預測結果（之後可以回頭看哪些視窗判錯）
        tmp_win = test_df[["record_id", "label", "distance_cm"]].copy()
        tmp_win["y_prob"] = y_prob
        tmp_win["y_pred"] = y_pred
        tmp_win["held_out_distance_cm"] = dist
        window_pred_rows.append(tmp_win)

        # (D) 檔案層級：同一支檔案的視窗機率取平均，再判斷整支檔案
        #     比視窗層級穩定（單一視窗偶爾的誤判會被平均掉），比較接近「這段錄音是不是洩漏」
        file_df = tmp_win.groupby("record_id", as_index=False).agg(
            label=("label", "first"),
            distance_cm=("distance_cm", "first"),
            y_prob=("y_prob", "mean"),
        )
        file_df["y_pred"] = (file_df["y_prob"] >= threshold).astype(int)
        file_df["held_out_distance_cm"] = dist
        file_pred_rows.append(file_df)

        f_prec, f_rec, f_f1, _ = precision_recall_fscore_support(
            file_df["label"], file_df["y_pred"], average="binary", zero_division=0
        )
        f_auc = _safe_auc(file_df["label"].values, file_df["y_prob"].values)

        # 每個距離只印一行重點
        print(f"{int(dist):>3} cm｜特徵 {len(features):>3}｜"
              f"視窗 F1={w_f1:.3f} AUC={w_auc:.3f}｜檔案 F1={f_f1:.3f} AUC={f_auc:.3f}")

        # 這個距離的總結（存成總表）
        summary_rows.append({
            "distance_cm": dist,
            "n_features": len(features),                         # 這一輪用了幾個特徵
            "n_train_windows": len(train_df),
            "n_test_windows": len(test_df),
            "n_test_records": test_df["record_id"].nunique(),
            "test_positive_rate": test_df["label"].mean(),       # 驗證資料中洩漏的比例
            "window_precision": w_prec,
            "window_recall": w_rec,
            "window_f1": w_f1,
            "window_auc": w_auc,
            "file_precision": f_prec,
            "file_recall": f_rec,
            "file_f1": f_f1,
            "file_auc": f_auc,
        })

    # ---------- 3. 彙整所有距離 ----------
    summary_df = pd.DataFrame(summary_rows)
    window_pred_df = pd.concat(window_pred_rows, ignore_index=True) if window_pred_rows else pd.DataFrame()
    file_pred_df = pd.concat(file_pred_rows, ignore_index=True) if file_pred_rows else pd.DataFrame()

    if not summary_df.empty:
        # 所有距離的平均成績
        print(f"\n平均：視窗 F1 = {summary_df['window_f1'].mean():.4f}、AUC = {summary_df['window_auc'].mean():.4f}；"
              f"檔案 F1 = {summary_df['file_f1'].mean():.4f}、AUC = {summary_df['file_auc'].mean():.4f}")
        # 最差的距離：現場最需要注意的盲區
        worst = summary_df.loc[summary_df["window_auc"].idxmin()]
        print(f"最差的距離：{int(worst['distance_cm'])} cm（視窗 AUC = {worst['window_auc']:.3f}）")

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
    random_state=RANDOM_STATE,
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
# 模型在不同條件下有沒有表現一致（holdout 分組評估）
# ============================================================
def evaluate_holdout_by_group(
    df_holdout,          # holdout 的 window-level 資料（要有分組欄位）
    y_true,              # 真實標籤（0 = 正常、1 = 洩漏），順序要和 df_holdout 的列一致
    y_pred,              # 預測標籤（0 / 1），順序要和 df_holdout 的列一致
    group_col,           # 分組欄位，例如 "distance_cm"、"leak_quantity"、"session_id"、"file_date"
    y_prob=None,         # 預測機率（有傳才算 AUC）
    save_csv=None,       # 有指定就存成 CSV
):
    """
    依某個欄位分組，每組算一列重點指標，整理成一張表：
      - n_normal / n_leak：這組有幾個正常、幾個洩漏視窗
      - false_alarm_rate ：正常視窗被誤判成洩漏的比例（越低越好；沒有正常資料時為 NaN）
      - detection_rate   ：洩漏視窗被正確抓到的比例（越高越好；沒有洩漏資料時為 NaN）
      - f1、auc          ：同時有正常和洩漏時才算
    只有一種類別的組別（例如依洩漏量分組時，0 ml 全是正常），只會顯示對它有意義的那一項。
    回傳整理好的表。
    """
    # ---------- 1. 檢查分組欄位 ----------
    if group_col not in df_holdout.columns:
        print(f"⚠️ df_holdout 沒有欄位: {group_col}")
        return pd.DataFrame()

    # ---------- 2. 整理成一張評估用的表 ----------
    # 每列一個視窗：屬於哪一組、真實答案、模型預測（有傳的話再加上機率）
    eval_df = pd.DataFrame({
        "group": df_holdout[group_col].values,
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    })
    if y_prob is not None:
        eval_df["y_prob"] = np.asarray(y_prob)

    # ---------- 3. 逐組計算 ----------
    rows = []
    for group_value, sub in eval_df.groupby("group"):
        normal = sub[sub["y_true"] == 0]                    # 這組裡真正是正常的視窗
        leak = sub[sub["y_true"] == 1]                      # 這組裡真正是洩漏的視窗
        has_both = len(normal) > 0 and len(leak) > 0        # 是否同時有兩種類別

        row = {
            group_col: group_value,
            "n_windows": len(sub),
            "n_normal": len(normal),
            "n_leak": len(leak),
            # 誤報率：正常視窗中，被模型判成洩漏的比例（沒有正常資料時算不出來，填 NaN）
            "false_alarm_rate": (normal["y_pred"] == 1).mean() if len(normal) else np.nan,
            # 偵測率：洩漏視窗中，被模型正確判成洩漏的比例（沒有洩漏資料時算不出來，填 NaN）
            "detection_rate": (leak["y_pred"] == 1).mean() if len(leak) else np.nan,
        }

        # F1 和 AUC 需要同時有正常和洩漏才有意義；只有一種類別時填 NaN
        if has_both:
            _, _, f1, _ = precision_recall_fscore_support(
                sub["y_true"], sub["y_pred"], average="binary", zero_division=0
            )
            row["f1"] = f1
            # AUC 需要機率；沒傳 y_prob 時填 NaN
            row["auc"] = _safe_auc(sub["y_true"], sub["y_prob"]) if "y_prob" in sub else np.nan
        else:
            row["f1"] = np.nan
            row["auc"] = np.nan
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # ---------- 4. 印出結果 ----------
    print("\n" + "=" * 60)
    print(f"Holdout 依 {group_col} 分組評估")
    print("=" * 60)
    print(result_df.round(3).to_string(index=False))

    # 自動標出最需要注意的組別：誤報最高、偵測最低（現場最可能出問題的條件）
    if result_df["false_alarm_rate"].notna().any():
        worst_fa = result_df.loc[result_df["false_alarm_rate"].idxmax()]
        print(f"誤報最高：{group_col} = {worst_fa[group_col]}（{worst_fa['false_alarm_rate']:.1%}）")
    if result_df["detection_rate"].notna().any():
        worst_det = result_df.loc[result_df["detection_rate"].idxmin()]
        print(f"偵測最低：{group_col} = {worst_det[group_col]}（{worst_det['detection_rate']:.1%}）")

    # ---------- 5. 存檔 ----------
    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_csv, index=False, encoding="utf-8-sig")

    return result_df


# ============================================================
# 從最終特徵名單反推執行計畫：下游只需要算哪些東西
# ============================================================
# 衍生特徵的前綴 → 去掉前綴就是它依賴的原始特徵名稱
# 例如 z_scan_093_465_475k_ai1_E 去掉 "z_" 就是 scan_093_465_475k_ai1_E
_CUSUM_PREFIXES = ("cusum_pos_z_", "cusum_neg_z_")      # 跨窗 CUSUM（輸入是 z_ 欄位）
_BASELINE_PREFIXES = ("rel_", "z_")                     # 套固定 baseline 產生的相對值與 z 分數
_TEMPORAL_PREFIXES = (                                  # 跨窗時間特徵
    # 注意順序：較長的前綴要放前面，例如 "delta_prev3mean_" 要在 "delta_" 之前，
    # 否則 delta_prev3mean_X 會被誤當成 delta_ 開頭，去掉前綴後變成 prev3mean_X
    "delta_prev3mean_", "delta_prev5mean_", "delta_",
    "local_rel3_", "local_rel5_", "local_z3_", "local_z5_",
    "rolling_std3_", "rolling_std5_", "rolling_slope3_", "rolling_slope5_",
    "rolling_max_gap3_", "rolling_max_gap5_",
)


def build_feature_execution_plan(selected_feature_names):
    """
    從最終特徵名單反推下游要算什麼。例如選到 z_scan_093_465_475k_ai1_E：
      → raw_required_features 要有 scan_093_465_475k_ai1_E
      → baseline_features 要有 scan_093_465_475k_ai1_E（要套 baseline 才算得出 z_）
    回傳 dict：
      raw_required_features ：每個視窗要先算的原始特徵（給 runtime 特徵抽取器）
      baseline_features     ：要套固定 baseline 產生 rel_ / z_ 的原始特徵
      cusum_input_features  ：要做跨窗 CUSUM 的 z_ 欄位
      temporal_base_features：要做跨窗 delta / local / rolling 的原始特徵
      final_keep_features   ：最終給模型的特徵
    """
    raw = set()          # 每個視窗要算的原始特徵
    baseline = set()     # 要套 baseline 的原始特徵
    cusum = set()        # 要做跨窗 CUSUM 的 z_ 欄位
    temporal = set()     # 要做跨窗時間特徵的原始特徵
    # 用 set 是為了自動去重複：例如同時選到 rel_X 和 z_X，X 只會記一次

    for feat in selected_feature_names:
        if feat.startswith(_CUSUM_PREFIXES):
            # cusum_pos_z_X：需要 z_X → 需要套 baseline → 需要原始特徵 X
            z_col = "z_" + feat.split("_z_", 1)[1]       # cusum_pos_z_X → z_X（以第一個 "_z_" 切開，取後半）
            base = z_col[len("z_"):]                     # z_X → X
            cusum.add(z_col)
            baseline.add(base)
            raw.add(base)
        elif feat.startswith(_BASELINE_PREFIXES):
            # rel_X 或 z_X：需要套 baseline → 需要原始特徵 X
            base = feat.split("_", 1)[1]                 # 以第一個 "_" 切開，取後半：rel_X / z_X → X
            baseline.add(base)
            raw.add(base)
        elif feat.startswith(_TEMPORAL_PREFIXES):
            # delta_X、rolling_std3_X……：需要跨窗計算 → 需要原始特徵 X
            prefix = next(p for p in _TEMPORAL_PREFIXES if feat.startswith(p))   # 找出是哪個前綴（依清單順序，長的先比）
            base = feat[len(prefix):]
            temporal.add(base)
            raw.add(base)
        else:
            # 沒有任何衍生前綴：本身就是原始特徵，直接算
            raw.add(feat)

    return {
        "raw_required_features": sorted(raw),
        "baseline_features": sorted(baseline),
        "cusum_input_features": sorted(cusum),
        "temporal_base_features": sorted(temporal),
        "final_keep_features": list(selected_feature_names),     # 保留原本順序（依重要性排序）
    }


# ============================================================
# 輸出最終 feature config JSON（給 Transition / Disturbance / Runtime 讀取的特徵契約）
# ============================================================
def save_feature_config_json(
    output_json,                    # 輸出路徑
    selected_feature_names,         # 最終特徵名單
    window_sec,                     # 每個視窗多長（秒）
    hop_sec,                        # 每次滑動多少（秒）
    fs,                             # 取樣率
    nperseg,                        # Welch PSD 的分段長度
    max_points,                     # 每支檔案最多讀多少點
    target_levels,                  # 這次比較了哪些洩漏量
    top_k_candidates,               # 去共線性前取前幾名
    corr_threshold,                 # 共線性門檻
    temporal_band_names,            # 要算「單一視窗內」時間結構特徵的頻帶（residual cusum_peak、burst 等）

    # 正式上游特徵契約：下游載入時會比對這些，確保用的是同一套設定
    band_config_path,               # band config 檔案路徑
    band_config_hash,               # band config 檔案指紋
    feature_extraction_config_path, # 特徵萃取設定檔路徑
    feature_extraction_config_hash, # 特徵萃取設定檔指紋
    baseline_path,                  # 固定 baseline 檔案路徑
    baseline_payload,               # 固定 baseline 的內容（從裡面取版本、狀態等資訊）

    band_config=None,               # 有傳就檢查「runtime 算不算得出這些原始特徵」
    data_info=None,                 # 資料來源紀錄（日期、場次數、切分方式等），原樣寫進 JSON

    # 後處理的預設值（正式數值由後規則程式定案，這裡只是起始值）
    source_note="step2_micro_leak_feature_selection",   # 這份檔案是哪個流程產生的
    score_col="leak_score",                             # 模型輸出分數的欄位名稱
    pred_col="pred_label",                              # 模型預測標籤的欄位名稱
    threshold=0.5,                                      # 分數超過多少算洩漏
    min_consecutive_windows=3,                          # 至少連續幾個視窗才算洩漏事件
    min_leak_duration_sec=0.4,                          # 事件至少持續多久（秒）
    disturbance_ratio_col="offband_to_leak_focus_E_ratio",   # 抗干擾規則用的欄位
    disturbance_ratio_threshold=None,                   # 抗干擾門檻（由 Disturbance 程式決定）

    baseline_mode="fixed_normal_baseline",              # baseline 模式：固定正常 baseline
    include_offband_summary=True,                       # 是否計算 offband / 洩漏 / 共振摘要
    cusum_k=CUSUM_K,                                    # 跨窗 CUSUM 的容忍值（只有執行計畫裡有 CUSUM 時才用到）
    cusum_decay=CUSUM_DECAY,                            # 跨窗 CUSUM 的衰減
):
    """
    把最終特徵與下游需要的參數整理成 feature_config_micro.json。

    feature_execution_plan 是從最終特徵名單「反推」出來的，只列下游真正需要算的東西，
    runtime 不會去算模型用不到的特徵。
    有傳 band_config 時，會檢查 raw_required_features 全部都是 runtime 特徵抽取器支援的。
    """
    selected = list(selected_feature_names)

    # ---------- 1. 從最終特徵反推執行計畫 ----------
    plan = build_feature_execution_plan(selected)

    # ---------- 2. 最後一道檢查：runtime 算不算得出這些原始特徵 ----------
    # 前面主程式已經先過濾過一次；這裡再確認，萬一有漏，在 Step2 就報錯，而不是部署時才發現
    # 算不出來時，validate_runtime_extractable_features 會直接報錯並列出是哪些特徵
    if band_config is not None:
        validate_runtime_extractable_features(plan["raw_required_features"], band_config)
        print(f"✅ runtime 可抽取檢查通過：{len(plan['raw_required_features'])} 個原始特徵")

    # ---------- 3. 組成要寫出的內容 ----------
    payload = {
        "schema_name": "feature_config_micro",                    # 檔案種類
        "schema_version": 3,                                      # 3：執行計畫改為從最終特徵反推，並新增 raw_required_features
        "created_at": datetime.now().astimezone().isoformat(),    # 產生時間（含時區）
        "source": source_note,

        # ---------- 最終特徵 ----------
        "selected_feature_names": selected,

        # ---------- 上游契約：下游載入時要比對這些指紋，不一致就拒絕使用 ----------
        "band_config_path": str(Path(band_config_path)),
        "band_config_sha256": band_config_hash,
        "feature_extraction_config_path": str(Path(feature_extraction_config_path)),
        "feature_extraction_config_sha256": feature_extraction_config_hash,

        "baseline_mode": baseline_mode,
        "baseline_path": str(Path(baseline_path)),
        "baseline_schema_name": baseline_payload.get("schema_name"),
        "baseline_schema_status": baseline_payload.get("schema_status"),            # candidate（實驗室）/ final（現場正式）
        "baseline_environment": baseline_payload.get("environment"),                # laboratory / site
        "baseline_config_fingerprint": baseline_payload.get("config_fingerprint"),  # baseline 建立時的設定指紋
        "baseline_base_feature_version": baseline_payload.get("base_feature_version"),  # baseline 用的特徵公式版本

        # ---------- 訊號與切窗參數（下游必須用完全相同的值，否則特徵數值會對不上）----------
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "fs": fs,
        "nperseg": nperseg,
        "max_points": max_points,

        # ---------- Step2 選特徵過程（日後回頭查用）----------
        "target_levels": sorted(target_levels),
        "top_k_candidates": top_k_candidates,
        "corr_threshold": corr_threshold,
        "model_excluded_prefixes": list(MODEL_EXCLUDED_PREFIXES),    # 不進模型的跨窗特徵前綴
        "build_cross_window_features": BUILD_CROSS_WINDOW_FEATURES,  # 這次有沒有算跨窗特徵
        # 要算「單一視窗內」時間結構特徵的頻帶（和跨窗特徵無關）
        "temporal_band_names": sorted(temporal_band_names),

        # ---------- 資料來源（日期、場次、切分方式、保留推論日等）----------
        "data_info": data_info or {},          # 沒傳時寫入空字典

        # ---------- 下游特徵工程 ----------
        "include_offband_summary": include_offband_summary,
        "cusum_k": cusum_k,              # 只有執行計畫裡有 CUSUM 時才會用到
        "cusum_decay": cusum_decay,

        # ---------- 後處理預設值（正式數值由後規則程式定案）----------
        "score_col": score_col,
        "pred_col": pred_col,
        "threshold": threshold,
        "min_consecutive_windows": min_consecutive_windows,
        "min_leak_duration_sec": min_leak_duration_sec,
        "disturbance_ratio_col": disturbance_ratio_col,
        "disturbance_ratio_threshold": disturbance_ratio_threshold,

        # ---------- 執行計畫：下游只算這些 ----------
        "feature_execution_plan": plan,
    }

    # ---------- 4. 寫出 JSON ----------
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 印出摘要：最終特徵數，以及下游實際要算多少東西
    print(f"✅ 已輸出 feature config: {output_json}")
    print(f"   最終特徵 {len(selected)} 個 → 需要算原始特徵 {len(plan['raw_required_features'])} 個、"
          f"套 baseline {len(plan['baseline_features'])} 個、CUSUM {len(plan['cusum_input_features'])} 個、"
          f"跨窗 {len(plan['temporal_base_features'])} 個")



#                                               ****************************************************************
#                                                                            其他
#                                               ****************************************************************
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
Step2 完整主流程（嚴格 dev / holdout 分離版）

Step 1. 讀原始資料（正常 / 洩漏）
Step 2. hard QC
Step 3. 重複檔案檢查、還原錄音場次、排除保留推論日
Step 4. 以場次為單位切分 dev / holdout
  支線 A（FORCE_BAND_DISCOVERY=True）：Step2a 頻帶探索 → 人工定案 → 結束
  支線 B：讀正式 band config 與 final baseline，繼續 Step2b
Step 5. dev-only EDA（預設關閉）
Step 6～7. 正式特徵工程、套固定 baseline（跨窗特徵預設不算）
Step 8. record-level 診斷、抗干擾假設檢查
Step 9. 建立可進模型的特徵清單（排除 offband、跨窗、距離高風險、runtime 不支援）
Step 10. 正式選特徵（select_model_features）
Step 11～12. 最終特徵的分布與距離摘要
Step 13～15. reference CV、fold-wise CV（主要指標）、LODO（每輪重新選特徵）
Step 16. holdout 評估（依距離、洩漏量、場次分組）
Step 17. 打亂標籤檢定（預設關閉）
Step 19. 輸出特徵表與 feature_config_micro.json

後續 Transition / Disturbance / Runtime 不再自己找頻帶或選特徵，
直接讀這支程式輸出的設定檔。
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
    
    SAVE_STEP9_RAW_BOXPLOTS = False               # Step 10：是否輸出可用特徵排名前幾名的 boxplot（探索期用）
    SAVE_STEP9_DISTANCE_EFFECT_PLOTS = False      # Step 9：是否輸出 top features 的 distance effect 線圖
    SAVE_STEP9_FINAL_SUMMARY_PLOT = True          # Step 9：是否輸出 final ranking 的總結圖，建議保留
    
    SAVE_STEP11_SELECTED_FEATURE_HEATMAP = True   # Step 11：是否輸出最終保留特徵的 condition heatmap，建議保留
    SAVE_STEP11_SELECTED_FEATURE_LINE_PLOTS = False  # Step 11：是否輸出最終保留特徵的 leak/distance 線圖
    SAVE_STEP12_FINAL_SELECTED_DISTANCE_PLOTS = False  # Step 12：是否輸出最終特徵的距離泛化能力圖

    # 設定資料路徑與輸出資料夾
    data_root = DATA_ROOT            # 原始資料總資料夾（設定在檔案上方）
    out_root = FINAL_OUTPUT_ROOT     # 分析結果輸出資料夾（設定在檔案上方）
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
    # load_tdms_folder() 讀檔時已完成：去直流、兩聲道對齊、建立 diff（與 baseline / runtime 同一套規則）
    records_leak = load_tdms_folder(data_root / "leak1", max_points_per_file=max_points, fs_target=fs)      # 讀洩漏資料
    records_normal = load_tdms_folder(data_root / "leak0", max_points_per_file=max_points, fs_target=fs)    # 讀 baseline 資料
    
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
    print(f"洩漏資料（{sorted(TARGET_LEAK_LEVELS)} ml）: {len(records_leak)} 筆")

    # 設定二元分類標籤：洩漏資料 label = 1 / baseline 資料 label = 0
    for r in records_leak:
        r["label"] = 1
    for r in records_normal:
        r["label"] = 0
        
    # 檢查檔名解析結果：每個洩漏量 × 距離各有幾筆
    check_df = pd.DataFrame([
        {
            "label": r["label"],
            "leak_quantity": r["meta"]["leak_quantity"],
            "distance_cm": int(round(r["meta"]["distance"] * 100)),
        }
        for r in records_leak + records_normal
    ])
    print("\n=== 檔名解析結果（洩漏量 × 距離）===")
    print(check_df.groupby(["leak_quantity", "distance_cm"]).size().to_string())
    
    # ---------- 資料治理報表：每一步之後剩下幾支 ----------
    governance = []
    def log_counts(step, leak, normal, note=""):
        governance.append({"步驟": step, "洩漏檔案數": len(leak), "正常檔案數": len(normal), "說明": note})
    
    log_counts("原始讀取", records_leak, records_normal)


    # ============================================================
    # Step 2. hard QC 硬性品質檢查（規則見 shared 的 check_signal_quality）
    # ============================================================
    # 檢查每筆資料有沒有：NaN / Inf / 全零 / RMS 太低
    abnormal_leak = check_signal_quality(records_leak)
    abnormal_normal = check_signal_quality(records_normal)

    # 刪掉不符合硬性品質的資料
    records_leak = [r for r in records_leak if r["record_id"] not in abnormal_leak]
    records_normal = [r for r in records_normal if r["record_id"] not in abnormal_normal]

    print(f"硬性排除 leak: {len(abnormal_leak)} 筆、normal: {len(abnormal_normal)} 筆")
    log_counts("硬性 QC 後", records_leak, records_normal, "NaN / Inf / 全零 / RMS 太低")
    for rid, reason in {**abnormal_leak, **abnormal_normal}.items():
        print(f"  - {rid}：{reason}")
        
    
    # ============================================================
    # Step 3. 重複檔案檢查、還原錄音場次、排除保留推論日
    # ============================================================
    # 內容完全相同的檔案：依 DUPLICATE_POLICY 停止或排除（舊資料有 347 組 leak0 / leak1 重複）
    all_records = check_duplicate_files(records_leak + records_normal, policy=DUPLICATE_POLICY)
    
    log_counts("內容重複檢查後",
           [r for r in all_records if r["label"] == 1],
           [r for r in all_records if r["label"] == 0],
           "另有 347 組 leak0/leak1 重複檔已事先手動移出（694 支，見 duplicate_files.csv）")
    
    # 從檔名還原錄音場次，每筆加上 session_id、file_time
    assign_session_ids(all_records)
    
    # 保留給最終模型推論的日期：Step2 完全不使用
    if RESERVED_TEST_DATES:
        reserved = set(RESERVED_TEST_DATES)
        n_before = len(all_records)
        all_records = [r for r in all_records if r["file_time"].strftime("%Y%m%d") not in reserved]
        print(f"🔒 排除保留推論日 {sorted(reserved)}：{n_before} → {len(all_records)} 支")
    
    records_leak = [r for r in all_records if r["label"] == 1]
    records_normal = [r for r in all_records if r["label"] == 0]
    
    log_counts("排除保留推論日後", records_leak, records_normal,
           f"保留日：{sorted(RESERVED_TEST_DATES) if RESERVED_TEST_DATES else '無'}")
    
    # ---------- 日期 × 標籤檢查 ----------
    # 同一天要同時有正常和洩漏；如果某天只有一種，模型可能學到「哪天錄的」而不是「有沒有洩漏」
    date_label_df = pd.crosstab(
        pd.Series([r["file_time"].strftime("%Y%m%d") for r in all_records], name="file_date"),
        pd.Series([r["label"] for r in all_records], name="label"),
    ).reindex(columns=[0, 1], fill_value=0)
    print("\n=== 每天的正常（0）/ 洩漏（1）檔案數 ===")
    print(date_label_df.to_string())
    
    n_mixed_days = int(((date_label_df[0] > 0) & (date_label_df[1] > 0)).sum())
    if n_mixed_days == 0:
        print("\n" + "!" * 60)
        print("⚠️ 沒有任何一天同時錄了正常和洩漏：日期與標籤完全綁在一起。")
        print("   選出的頻帶與特徵可能是在分辨「哪天錄的」，結果只能當流程測試，不能定案。")
        print("!" * 60)
    elif n_mixed_days < len(date_label_df):
        print(f"ℹ️ {len(date_label_df)} 天中有 {n_mixed_days} 天同時有正常和洩漏，其餘只有一種類別")
        
    # 每個洩漏量 × 距離：幾支檔案、幾個場次
    session_check_df = pd.DataFrame([
        {
            "leak_quantity": r["meta"]["leak_quantity"],
            "distance_cm": int(round(r["meta"]["distance"] * 100)),
            "session_id": r["session_id"],
        }
        for r in all_records
    ])
    print("\n=== 整理後的資料（洩漏量 × 距離：檔案數 / 場次數）===")
    print(session_check_df.groupby(["leak_quantity", "distance_cm"])
          .agg(n_files=("session_id", "size"), n_sessions=("session_id", "nunique"))
          .to_string())


    # ============================================================
    # Step 4. dev / holdout 切分（以錄音場次為單位）
    # 同一場次不會被整場拆到兩邊；場次太少時，場次內依時間切並留緩衝
    # ============================================================
    dev_records, holdout_records = split_dev_holdout_by_session(all_records)
    dev_record_ids = {r["record_id"] for r in dev_records}
    holdout_record_ids = {r["record_id"] for r in holdout_records}

    dev_records_leak = [r for r in dev_records if r["label"] == 1]
    dev_records_normal = [r for r in dev_records if r["label"] == 0]
    holdout_records_leak = [r for r in holdout_records if r["label"] == 1]
    holdout_records_normal = [r for r in holdout_records if r["label"] == 0]

    dev_sessions = {r["session_id"] for r in dev_records}
    holdout_sessions = {r["session_id"] for r in holdout_records}
    print(f"\nDev records: {len(dev_record_ids)}（{len(dev_sessions)} 場次）")
    print(f"Holdout records: {len(holdout_record_ids)}（{len(holdout_sessions)} 場次）")
    shared_sessions = dev_sessions & holdout_sessions
    if shared_sessions:
        print(f"ℹ️ 以下場次依時間切分，前段在 dev、後段在 holdout（中間有緩衝）：{len(shared_sessions)} 個")
    
    log_counts("dev", dev_records_leak, dev_records_normal, f"{len(dev_sessions)} 場次")
    log_counts("holdout", holdout_records_leak, holdout_records_normal,
               f"{len(holdout_sessions)} 場次（其中 {len(shared_sessions)} 場與 dev 依時間切分共用）")
    governance_path = out_root / "feature_selection" / "data_governance_report.csv"
    pd.DataFrame(governance).to_csv(governance_path, index=False, encoding="utf-8-sig")
    print("✅ 資料治理報表：", governance_path)
    
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
        
        # 頻帶搜尋只產出「候選」結果，必須人工檢查定案、重建 baseline 後，才能繼續 Step2b，所以在這裡結束程式
        raise SystemExit(
            "\nStep2a 頻帶探索已完成。\n"
            "已輸出 band_config_step2_candidate.json（offband 已自動帶入）。\n"
            "請檢查 band_importance_summary.csv 與候選頻帶，\n"
            "確認後把 schema_status 改成 final，另存成 band_config_step2.json。\n"
            "接著執行 build_site_baseline_v2.py 建立 final baseline，\n"
            "最後把 FORCE_BAND_DISCOVERY 改成 False，重新執行本程式完成 Step2b。"
        )
        
    # ============================================================
    # 支線 B: 使用既有且已定案的正式 band config
    # ============================================================
    print("\n找到正式 band_config_step2.json，跳過 Step2a 頻帶搜尋。")
    # 如果不需要頻帶搜尋, 把正式 band config 讀出來、驗證過，同時算出它的指紋 band_config_hash
    band_config = load_band_config(BAND_CONFIG_PATH, expected_status="final", require_offband=True)
    band_config_hash = calculate_file_sha256(BAND_CONFIG_PATH)
 
    # 把 band_config 物件裡的三種頻段清單各自取出，存成獨立變數
    leak_focused_bands = list(band_config.leak_focused_bands)
    resonance_bands = list(band_config.resonance_bands)
    offband_bands_for_rule = list(band_config.offband_bands)      # Offband 只提供後續 Veto／抗干擾規則使用，不允許進入模型特徵選擇或模型訓練。
    
    # 檢查 band config 的共振頻帶是否和共振程式的正式版一致（手動複製時容易漏同步）
    if RESONANCE_JSON_PATH.exists():
        approved = {(b["name"], b["f_low"], b["f_high"]) for b in load_resonance_band_json(RESONANCE_JSON_PATH)}
        current = {(b["name"], b["f_low"], b["f_high"]) for b in resonance_bands}
        if approved != current:
            raise ValueError(
                "band_config_step2.json 的 resonance_bands 與 approved_resonance_bands.json 不一致；\n"
                f"  正式版：{sorted(approved)}\n  設定檔：{sorted(current)}\n"
                "請先同步共振頻帶，再重建 site_baseline.json。"
            )
    
    # 把 leak-focused 跟 resonance 兩種頻段合併、處理重疊，得到最終要用來做特徵萃取的主頻段清單 final_main_bands
    final_main_bands = band_config.final_main_bands
    
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
    
    if RUN_SOFT_QC:
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
        for i, r_leak in enumerate(eda_records_by_level[ANALYSIS_LEAK_LEVELS[0]][30:50], start=1):
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
    # 把 dev、holdout 各自的洩漏 / 正常合併成完整清單
    dev_records_all = dev_records_leak + dev_records_normal
    holdout_records_all = holdout_records_leak + holdout_records_normal

    # 特徵計算函式只需要知道「哪支檔案、叫什麼、標籤是什麼」，不用把整段訊號傳進去
    # （函式會自己讀檔，平行運算時比較省記憶體）
    dev_file_specs = [
        {"filepath": r["filepath"], "record_id": r["record_id"], "label": r["label"]}
        for r in dev_records_all
    ]
    holdout_file_specs = [
        {"filepath": r["filepath"], "record_id": r["record_id"], "label": r["label"]}
        for r in holdout_records_all
    ]

    cache_dir = out_root / "feature_cache"          # 特徵快取：同樣的檔案與設定，第二次直接讀，不用重算
    sel_dir = out_root / "feature_selection"        # 選特徵與驗證報表的資料夾
    # 注意：不要把變數取名為 fs，fs 已經是取樣率

    # ============================================================
    # Step 7. by-file 正式特徵工程
    #   core_bands = final_main_bands（洩漏頻帶 + 共振頻帶）：模型特徵的來源
    #   aux_bands  = offband：一起算，但只給抗干擾規則用，不進模型
    # ============================================================
    # dev 和 holdout 用完全相同的設定，先打包成一份，避免兩邊參數不小心寫得不一樣
    feature_kwargs = dict(
        cache_dir=cache_dir,
        core_bands=final_main_bands,
        aux_bands=offband_bands_for_rule,
        low_bands=[],
        n_jobs=N_JOBS,                         # 平行數（設定區）
        window_sec=window_sec,                 # 每個視窗 0.2 秒
        hop_sec=hop_sec,                       # 每次滑動 0.05 秒
        fs=fs,                                 # 取樣率
        nperseg=nperseg,                       # Welch PSD 分段長度
        target_levels=TARGET_LEVELS,           # 只算要比較的洩漏量
        feature_version=base_feature_version,  # 特徵公式版本（版本不同，快取不會被誤用）
        max_points=max_points,                 # 每支檔案最多讀多少點
    )
    # **feature_kwargs：把上面那包設定展開成「名稱=值」傳進函式
    features_dev = load_or_compute_features_by_file(file_specs=dev_file_specs, split_name="dev", **feature_kwargs)
    features_holdout = load_or_compute_features_by_file(file_specs=holdout_file_specs, split_name="holdout", **feature_kwargs)

    print(f"Dev feature rows: {len(features_dev)}")
    print(f"Holdout feature rows: {len(features_holdout)}")

    # 轉成表格，依 record_id、視窗起始時間排序（跨窗特徵與分組都依賴這個順序）
    df_dev = pd.DataFrame(features_dev).sort_values(["record_id", "t_start"]).reset_index(drop=True)
    df_holdout = pd.DataFrame(features_holdout).sort_values(["record_id", "t_start"]).reset_index(drop=True)
    del features_dev, features_holdout     # 已轉成表格，原本的 list 用不到了，釋放記憶體

    # 檢查正式頻帶的原始特徵是否都有算出來（漏算會直接報錯）
    audit_band_feature_coverage(df_dev, band_config)
    audit_band_feature_coverage(df_holdout, band_config)

    # ============================================================
    # Step 7.1. 把 metadata 接回 window-level 特徵表
    # ============================================================
    def build_record_meta(records):
        """每支檔案一列：洩漏量、距離、壓力、場次、日期（CV 分組、跨天穩定性都會用到）"""
        return pd.DataFrame([
            {
                "record_id": r["record_id"],
                "leak_quantity": r["meta"]["leak_quantity"],
                "distance_cm": int(round(r["meta"]["distance"] * 100)),    # 公尺 → 公分
                "pressure": r["meta"]["pressure"],
                "session_id": r["session_id"],                             # 錄音場次（assign_session_ids 加的）
                "file_date": r["file_time"].strftime("%Y%m%d"),            # 錄音日期，例如 "20261023"
            }
            for r in records
        ]).drop_duplicates(subset=["record_id"])     # 保險：同一支檔案只留一列

    record_meta_dev_df = build_record_meta(dev_records_all)
    record_meta_holdout_df = build_record_meta(holdout_records_all)

    # 用 record_id 對應，把 metadata 接到每個視窗上（同一支檔案的 7 個視窗拿到相同的資訊）
    df_dev = df_dev.merge(record_meta_dev_df, on="record_id", how="left")
    df_holdout = df_holdout.merge(record_meta_holdout_df, on="record_id", how="left")

    # ============================================================
    # Step 7.2. offband / leak-focused / resonance 摘要特徵
    # 只給抗干擾規則用，不進模型（Step 9 會自動排除）
    # ============================================================
    if include_offband_summary:
        df_dev = build_offband_summary_features(df_dev, band_config)
        df_holdout = build_offband_summary_features(df_holdout, band_config)
        # 確認摘要欄位都有建出來
        audit_offband_summary_coverage(df_dev)
        audit_offband_summary_coverage(df_holdout)

    # ============================================================
    # Step 7.3. 要套固定 baseline 的原始特徵
    # 每個正式頻帶 × 通道（ai0 / ai1 / diff）× 指標（STEP2_EXPAND_SUFFIXES，預設 E / RMS / EnvRMS）
    # ============================================================
    expand_base_feature_cols = select_step2_expand_base_features(df=df_dev, bands=final_main_bands)
    print(f"\n套用 baseline 的原始特徵數量 = {len(expand_base_feature_cols)}")

    # ============================================================
    # Step 7.4. 套用固定正常 baseline，建立 rel_ / z_ 特徵
    #   rel_X = X − 正常平均（比正常狀態多了多少）
    #   z_X   = (X − 正常平均) ÷ 正常標準差（偏離正常幾個標準差）
    # dev 與 holdout 用同一份預先建立的 baseline，不從 dev 或 holdout 重新估計
    # ============================================================
    df_dev = apply_fixed_normal_baseline(df=df_dev, feature_cols=expand_base_feature_cols, baseline_payload=fixed_baseline)
    df_holdout = apply_fixed_normal_baseline(df=df_holdout, feature_cols=expand_base_feature_cols, baseline_payload=fixed_baseline)

    # 檢查 rel_ / z_ 欄位是否都有建出來（缺了代表 baseline 少了某些特徵的統計值）
    expected_relative_features = [f"{p}{c}" for c in expand_base_feature_cols for p in ("rel_", "z_")]
    for dataset_name, dataset_df in [("dev", df_dev), ("holdout", df_holdout)]:
        missing = [c for c in expected_relative_features if c not in dataset_df.columns]
        if missing:
            raise ValueError(f"{dataset_name} 缺少固定 baseline 特徵：{missing[:20]}")

    # ============================================================
    # Step 7.5～7.6. 跨窗特徵（CUSUM / delta / local / rolling）
    # 不進模型（MODEL_EXCLUDED_PREFIXES 會排除）：離線每支檔案只有 7 個視窗，和現場連續資料流的數值分布對不上
    # 時間特性改由後規則處理（看模型分數是否持續偏高）
    # 只在 BUILD_CROSS_WINDOW_FEATURES = True 時才算，供分析參考
    # ============================================================
    if BUILD_CROSS_WINDOW_FEATURES:
        # CUSUM 的輸入是 z_ 欄位
        z_feature_cols = [f"z_{c}" for c in expand_base_feature_cols]
        df_dev = add_cusum_features(df=df_dev, input_feature_cols=z_feature_cols, k=CUSUM_K, decay=CUSUM_DECAY)
        df_holdout = add_cusum_features(df=df_holdout, input_feature_cols=z_feature_cols, k=CUSUM_K, decay=CUSUM_DECAY)

        # delta / local / rolling 只對洩漏頻帶展開（不含共振頻帶，避免計算量過大）
        temporal_base_feature_cols = select_step2_expand_base_features(df=df_dev, bands=leak_focused_bands)
        df_dev = add_step2_temporal_features(df=df_dev, raw_feature_cols=temporal_base_feature_cols)
        df_holdout = add_step2_temporal_features(df=df_holdout, raw_feature_cols=temporal_base_feature_cols)
        print(f"ℹ️ 已建立跨窗特徵（只供分析，不進模型）：CUSUM 輸入 {len(z_feature_cols)}、temporal 輸入 {len(temporal_base_feature_cols)}")
    else:
        print("ℹ️ BUILD_CROSS_WINDOW_FEATURES = False：略過跨窗特徵（不進模型，時間特性由後規則處理）")

    # ============================================================
    # Step 8. record-level 總表、診斷、抗干擾假設檢查（只用 dev）
    # ============================================================
    # 每支檔案一列（7 個視窗取中位數），包含所有特徵，給診斷與報表用
    record_df_dev_all = build_record_level_feature_table(
        window_df=df_dev, extra_meta_df=record_meta_dev_df, agg="median",
    )
    # 只保留要比較的洩漏量（目前是 0 和 30）
    record_df_dev_levels = record_df_dev_all[record_df_dev_all["leak_quantity"].isin(TARGET_LEVELS)].copy()

    print("\n=== dev record-level leak_quantity 分布 ===")
    print(record_df_dev_all["leak_quantity"].value_counts().sort_index().to_string())

    # 每個特徵單獨看能不能分開正常與洩漏（Cohen's d + AUC），只輸出報表，不影響選特徵
    diag_df = diagnose_baseline_vs_30(
        record_df_dev_levels,
        out_dir=sel_dir / "baseline_vs_leak_diagnostic",
        top_n=30,
    )

    # ---------- 抗干擾假設檢查：洩漏是否會讓 offband 一起升高（各距離分開看）----------
    # 比值 = 洩漏的中位數 ÷ 正常的中位數
    #   leak_focus_E_mean 大於 1：洩漏確實讓洩漏頻帶升高
    #   offband_E_mean 接近 1    ：洩漏不影響 offband，抗干擾規則可行
    #   offband_E_mean 明顯大於 1：這個距離的洩漏也會拉高 offband，規則可能誤把洩漏當干擾
    # （舊資料正常與洩漏不同天錄，比值會混入日期差異，只當參考）
    summary_cols = [c for c in ("offband_E_mean", "leak_focus_E_mean") if c in record_df_dev_all.columns]
    if summary_cols:
        # 每個「距離 × 標籤」的中位數，再把標籤轉成欄位：med[(欄位, 0)] = 正常、med[(欄位, 1)] = 洩漏
        med = record_df_dev_all.groupby(["distance_cm", "label"])[summary_cols].median().unstack("label")
        offband_check_df = pd.DataFrame({c: med[(c, 1)] / med[(c, 0)] for c in summary_cols})
        offband_check_df.to_csv(sel_dir / "offband_assumption_check.csv", encoding="utf-8-sig")
        print("\n=== 抗干擾假設檢查：洩漏 ÷ 正常（各距離）===")
        print(offband_check_df.round(2).to_string())

    # ============================================================
    # Step 9. 建立「可以進模型的特徵清單」
    # 正式選特徵、fold-wise CV、LODO 都用同一份，確保三者的規則一致
    # ============================================================
    import contextlib     # 用來暫時把 print 的輸出丟掉
    import io

    def _base_name(feature_name):
        """去掉 rel_ / z_ 前綴，取得它依賴的原始特徵（例如 z_X → X）"""
        for prefix in ("rel_", "z_"):
            if feature_name.startswith(prefix):
                return feature_name[len(prefix):]
        return feature_name

    # runtime 特徵抽取器算不算得出來
    # 同一個原始特徵只檢查一次（X、rel_X、z_X 共用結果），存在 _runtime_ok_cache 裡
    _runtime_ok_cache = {}
    def _runtime_ok(feature_name):
        base = _base_name(feature_name)
        if base not in _runtime_ok_cache:
            try:
                # 一個一個檢查時可能會印很多訊息，先丟掉
                with contextlib.redirect_stdout(io.StringIO()):
                    validate_runtime_extractable_features([base], band_config)
                _runtime_ok_cache[base] = True
            except ValueError:
                # 檢查函式遇到不支援的特徵會報錯 → 代表 runtime 算不出來
                _runtime_ok_cache[base] = False
        return _runtime_ok_cache[base]

    # 所有數值特徵（排除識別、標籤、條件、時間等欄位，清單定義在設定區）
    all_feature_cols = [
        c for c in df_dev.columns
        if pd.api.types.is_numeric_dtype(df_dev[c]) and c not in NON_FEATURE_COLS
    ]

    # 逐一判斷每個特徵能不能進模型；不能的話記下原因（只記第一個符合的原因）
    exclusion_rows = []
    eligible_features = []
    for c in all_feature_cols:
        if is_rule_only_feature(c, offband_bands_for_rule):
            reason = "rule_only（offband / 摘要，只給抗干擾規則）"
        elif c.startswith(MODEL_EXCLUDED_PREFIXES):
            reason = "cross_window（跨窗特徵，時間特性由後規則處理）"
        elif _base_name(c) in DROP_DISTANCE_SENSITIVE:
            reason = "distance_risk（已知距離高風險）"
        elif not _runtime_ok(c):
            reason = "not_runtime（runtime 特徵抽取器算不出來）"
        else:
            eligible_features.append(c)      # 通過所有檢查 → 可以進模型
            continue
        exclusion_rows.append({"feature": c, "reason": reason})

    # 被排除的特徵與原因存成 CSV，方便回頭查「為什麼某個特徵沒被選」
    exclusion_df = pd.DataFrame(exclusion_rows, columns=["feature", "reason"])
    exclusion_df.to_csv(sel_dir / "model_excluded_features.csv", index=False, encoding="utf-8-sig")
    banned_model_features = set(exclusion_df["feature"])

    print("\n=== 可進模型的特徵 ===")
    print(f"全部數值特徵：{len(all_feature_cols)}")
    if not exclusion_df.empty:
        print(exclusion_df["reason"].value_counts().to_string())     # 每種原因排除了幾個
    print(f"可用特徵：{len(eligible_features)}")
    if not eligible_features:
        raise ValueError("排除後沒有任何可進模型的特徵")
        
    # 注意：
    # Step 2 的 Random Forest 僅用於 group-wise permutation importance，
    # 目的是評估各候選特徵在未看過場次上的額外辨識貢獻，並完成特徵定案。
    # 它不是最終部署分類器。
    #
    # 本流程輸出的 feature_config_micro.json 與 selected_feature_names_micro.csv
    # 會由後續模型訓練流程讀取；最終模型以相同 dev / holdout 切分訓練與驗收 XGBoost。

    # ============================================================
    # Step 10. 正式選特徵
    # 和 fold-wise CV、LODO 用同一個函式，所以驗證評估的就是這套流程本身
    # 流程：區分力 → 跨距離（跨天）穩定性 → 方向一致性 → 前 TOP_K → 去共線性 → 模型重要性 → 合併排序
    # ============================================================
    selection = select_model_features(
        df_dev, record_meta_dev_df, eligible_features,
        verbose=True, save_dir=sel_dir,          # save_dir：輸出去共線性報表
    )
    top_feature_names = selection["selected_features"]   # 最終特徵（依重要性排序）
    selection_rank_df = selection["rank_df"]             # 所有通過方向一致性的特徵與各項分數
    final_feature_df = selection["final_df"]             # 最終特徵的分數明細

    selection_rank_df.to_csv(sel_dir / "rank_features_model_eligible.csv", index=False, encoding="utf-8-sig")
    final_feature_df.to_csv(sel_dir / "final_selected_features.csv", index=False, encoding="utf-8-sig")

    print("\n=== 最終特徵（依綜合分數排序）===")
    print(final_feature_df.round(4).to_string(index=False))

    # 防呆：最終特徵不可包含任何禁止進模型的欄位（正常情況不會發生，萬一前面邏輯被改壞會在這裡擋下）
    leaked = [f for f in top_feature_names if f in banned_model_features]
    if leaked:
        raise ValueError(f"最終模型特徵包含禁止進模型的欄位：{leaked}")
    print("✅ 最終模型特徵不包含 offband／規則專用／跨窗／runtime 不支援的欄位")

    # ---------- 排名相關圖（開關在主程式最上面，預設只畫總結圖）----------
    # 最終特徵在正常 / 洩漏的分布箱型圖
    if SAVE_STEP9_FINAL_SUMMARY_PLOT:
        plot_top_features_multilevel_boxplot(
            record_df_dev_levels, final_feature_df,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=min(TOP_N_FEATURES_TO_PLOT, len(final_feature_df)),
            save_path=out_root / "feature_plots" / "top_features_boxplot_final_summary.png",
            showfliers=False, robust_ylim=True, lower_pct=2, upper_pct=98,
        )
    # 可用特徵排名前幾名的箱型圖（探索用）
    if SAVE_STEP9_RAW_BOXPLOTS:
        plot_top_features_multilevel_boxplot(
            record_df_dev_levels, selection_rank_df,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=TOP_N_FEATURES_TO_PLOT,
            save_path=out_root / "feature_plots" / "top_features_boxplot_eligible_ranking.png",
            showfliers=False, robust_ylim=True, lower_pct=2, upper_pct=98,
        )
    # 跨距離最穩定的特徵，隨距離變化的線圖（探索用）
    if SAVE_STEP9_DISTANCE_EFFECT_PLOTS:
        plot_top_features_distance_effect_lines(
            record_df_dev_levels,
            selection_rank_df.sort_values("distance_stability_score", ascending=False),
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=TOP_N_FEATURES_TO_PLOT,
            out_dir=out_root / "feature_plots" / "distance_effect_lines",
        )

    # ============================================================
    # Step 11. 最終特徵的分布（dev, record-level）
    # ============================================================
    # 只聚合最終特徵，比總表小很多
    record_df_dev_selected = build_record_level_feature_table(
        window_df=df_dev, feature_cols=top_feature_names,
        extra_meta_df=record_meta_dev_df, agg="median",
    )

    # 熱圖：列 = 特徵、欄 = 正常 / 洩漏 × 距離，顏色 = 標準化後的數值
    # 一眼看出哪些特徵在每個距離都有穩定差異
    if SAVE_STEP11_SELECTED_FEATURE_HEATMAP:
        plot_selected_feature_heatmap_by_condition(
            record_df_dev_selected,
            selected_features=top_feature_names,
            out_path=sel_dir / "selected_features_condition_heatmap_dev_micro.png",
            label_filter=None,
        )
    # 前 10 個最終特徵隨洩漏量、距離變化的線圖
    if SAVE_STEP11_SELECTED_FEATURE_LINE_PLOTS:
        plot_selected_feature_lines_by_leak_and_distance(
            record_df_dev_selected,
            features_to_plot=top_feature_names[:10],
            out_dir=sel_dir / "selected_feature_lines_dev_micro",
            label_filter=None,
        )

    # ============================================================
    # Step 12. 最終特徵的距離摘要
    # 距離敏感度只作參考，不影響選擇（洩漏聲隨距離衰減是物理現象，受距離影響不一定是壞事）
    # ============================================================
    # 只對最終特徵算距離敏感度（很快）
    leaklevel_df = rank_features_within_each_leak_level(
        record_df_dev_selected, target_levels=tuple(ANALYSIS_LEAK_LEVELS), min_samples=3,
    )
    # 有算跨天穩定性時才會有 date_ 欄位，所以只取存在的欄位
    summary_keep_cols = [c for c in [
        "feature", "final_score", "distance_stability_score", "distance_sign_consistency",
        "date_stability_score", "date_sign_consistency",
    ] if c in selection_rank_df.columns]
    # 合併成一張表：綜合分數 + 跨距離（跨天）穩定性 + 距離敏感度
    final_selected_distance_df = (
        final_feature_df[["feature", "combined_score"]]
        .merge(selection_rank_df[summary_keep_cols], on="feature", how="left")
        .merge(leaklevel_df[["feature", "distance_sensitivity_score"]], on="feature", how="left")
    )
    final_selected_distance_df.to_csv(
        sel_dir / "final_selected_feature_distance_summary.csv", index=False, encoding="utf-8-sig",
    )

    if SAVE_STEP12_FINAL_SELECTED_DISTANCE_PLOTS:
        plot_top_features_distance_effect_lines(
            record_df_dev_levels, final_selected_distance_df,
            target_levels=tuple(ANALYSIS_LEVELS),
            top_n=len(final_selected_distance_df),
            out_dir=out_root / "feature_plots" / "final_selected_distance_effect_lines",
        )

    # ============================================================
    # Step 13. 準備模型資料（binary：正常 vs 洩漏）
    # ============================================================
    X_dev_final = df_dev[top_feature_names].values           # dev 的最終特徵（每列一個視窗）
    X_holdout_final = df_holdout[top_feature_names].values
    y_dev = df_dev["label"].values                           # 0 = 正常、1 = 洩漏
    y_holdout = df_holdout["label"].values
    groups_dev = df_dev["session_id"].values                 # 同一場次的視窗一定在同一折（相鄰檔案幾乎一樣，不能拆開）

    # 隨機森林：樹模型只看大小順序，不需要標準化
    # cross_val_score、LODO 會自動複製一份重新訓練，所以這裡只是定義「用什麼模型」
    pipeline_final = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS)

    # ============================================================
    # Step 14. 交叉驗證
    # ============================================================
    # 折數受「每一類各有幾個獨立場次」限制，避免某一折完全沒有正常或洩漏
    y_arr = np.asarray(y_dev)
    g_arr = np.asarray(groups_dev)
    groups_per_class = {int(lbl): len(np.unique(g_arr[y_arr == lbl])) for lbl in np.unique(y_arr)}
    n_splits = min(5, *groups_per_class.values())
    print(f"CV 折數：{n_splits}（每類獨立場次數：{groups_per_class}）")
    if n_splits < 2:
        raise ValueError(f"每一類至少需要 2 個獨立場次才能做 Group CV，目前：{groups_per_class}")
    # 同一場次不拆開，並盡量讓每折的正常 / 洩漏比例相近
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # (A) reference CV：特徵是用全部 dev 選的（已經看過每一折），分數偏樂觀，只當對照
    ref_scores = cross_val_score(
        pipeline_final, X_dev_final, y_dev,
        groups=groups_dev, cv=gkf, scoring="roc_auc",
    )
    print(f"\nReference CV AUC（樂觀參考，以場次分組）: {ref_scores.mean():.3f} ± {ref_scores.std():.3f}")

    # (B) fold-wise CV：每折只用訓練折重新跑完整選特徵流程（主要指標）
    #     分組自動決定：每天都有正常和洩漏、且至少 3 天 → 留整天；否則 → 留整個場次
    foldwise_scores, foldwise_feature_info, feature_freq_df = run_group_cv_with_foldwise_feature_selection(
        df_dev=df_dev,
        record_meta_dev_df=record_meta_dev_df,
        eligible_features=eligible_features,      # 和正式流程同一份可用特徵清單
        random_state=RANDOM_STATE,
    )
    foldwise_aucs = np.array([item["auc"] for item in foldwise_feature_info])
    print(f"Fold-wise CV（主要指標）：AUC {foldwise_aucs.mean():.3f} ± {foldwise_aucs.std():.3f}、"
          f"F1 {foldwise_scores.mean():.3f} ± {foldwise_scores.std():.3f}")
    # reference 和 fold-wise 差很多，代表「用全部 dev 選特徵」讓分數灌水了

    # 每折的分數（不含特徵清單，太長）
    pd.DataFrame([{k: v for k, v in item.items() if k != "features"} for item in foldwise_feature_info]).to_csv(
        sel_dir / "foldwise_cv_scores.csv", index=False, encoding="utf-8-sig",
    )
    # 每個特徵在幾折被選中：次數越多越穩定
    feature_freq_df.to_csv(sel_dir / "foldwise_feature_stability.csv", index=False, encoding="utf-8-sig")

    # ============================================================
    # Step 15. LODO：留一個距離驗證（每輪只用訓練距離重新選特徵）
    # 資料裡有哪些距離會自動找出來，每個距離輪流當一次驗證
    # ============================================================
    lodo_summary_df, lodo_window_pred_df, lodo_file_pred_df = run_leave_one_distance_out_validation(
        df_dev=df_dev,
        feature_names=top_feature_names,           # 不重新選特徵時才會用到（這裡有傳下面兩個，所以會重新選）
        base_pipeline=pipeline_final,
        threshold=0.5,
        record_meta_df=record_meta_dev_df,
        eligible_features=eligible_features,
    )
    lodo_summary_df.to_csv(sel_dir / "leave_one_distance_out_summary.csv", index=False, encoding="utf-8-sig")
    lodo_window_pred_df.to_csv(sel_dir / "leave_one_distance_out_window_predictions.csv", index=False, encoding="utf-8-sig")
    lodo_file_pred_df.to_csv(sel_dir / "leave_one_distance_out_file_predictions.csv", index=False, encoding="utf-8-sig")

    
    # 注意：
    # 此處為 RF 特徵篩選流程的 holdout 驗證，
    # 用於確認「特徵選擇 + RF 基準模型」在未看過資料上的分辨能力。
    # 它不是最終部署模型的驗收成績；
    # 正式 XGBoost 模型將在後續訓練流程中讀取本流程輸出的特徵契約，
    # 並沿用相同 dev / holdout 切分完成最終訓練與驗收。
    # ============================================================
    # Step 16. Holdout 評估
    # 用全部 dev 訓練，在 holdout 評估一次；holdout 不拿來調任何參數，否則就失去「沒看過」的意義
    # ============================================================
    pipeline_final.fit(X_dev_final, y_dev)
    y_holdout_prob = pipeline_final.predict_proba(X_holdout_final)[:, 1]   # 每個視窗是洩漏的機率
    y_holdout_pred = (y_holdout_prob >= 0.5).astype(int)                   # 超過 0.5 判為洩漏
    holdout_auc = _safe_auc(y_holdout, y_holdout_prob)                     # 不受門檻影響的分辨力

    print("\n=== Holdout 評估 ===")
    print(classification_report(y_holdout, y_holdout_pred, digits=3, zero_division=0))
    print(f"Holdout AUC = {holdout_auc:.4f}")

    # 依距離、洩漏量、場次分組，各輸出一張表：每組的誤報率、偵測率、F1、AUC
    # 會自動標出誤報最高、偵測最低的組別（現場最可能出問題的條件）
    for group_col in ("distance_cm", "leak_quantity", "session_id"):
        evaluate_holdout_by_group(
            df_holdout=df_holdout,
            y_true=y_holdout,
            y_pred=y_holdout_pred,
            y_prob=y_holdout_prob,
            group_col=group_col,
            save_csv=sel_dir / f"holdout_by_{group_col}.csv",
        )

    # ============================================================
    # Step 17. 打亂標籤檢定（預設關閉）
    # 特徵是用真實標籤選的，打亂後當然分不好，所以結果幾乎一定「顯著」，參考價值有限且很耗時
    # 真正評估選特徵流程請看 Step 14 的 fold-wise CV 與 Step 15 的 LODO
    # ============================================================
    if RUN_PERMUTATION_TEST:
        perm_scores = run_permutation_test_group_cv(
            X=X_dev_final, y=y_dev, groups=groups_dev,
            pipeline=pipeline_final, cv=gkf,
            scoring="roc_auc", n_iter=100, random_state=RANDOM_STATE,
        )
        # p 值：100 次亂猜中，有幾次分數 ≥ 真實分數（+1 是避免 p 值剛好為 0 的標準寫法）
        p_value_ref = (np.sum(perm_scores >= ref_scores.mean()) + 1) / (len(perm_scores) + 1)
        print(f"Permutation p-value（reference CV，偏樂觀）= {p_value_ref:.4f}")

    # ============================================================
    # Step 18. 最終特徵對齊檢查
    # 確認 dev 和 holdout 都有最終特徵的每一個欄位，而且沒有缺值或無限大
    # ============================================================
    feature_config_audit = {"selected_feature_names": top_feature_names}
    audit_selected_feature_alignment(df_dev, feature_config_audit)
    audit_selected_feature_alignment(df_holdout, feature_config_audit)

    # ============================================================
    # Step 19. 匯出特徵表與特徵契約
    # ============================================================
    # 完整特徵表（所有欄位，給事後分析用；檔案很大）
    df_dev.to_csv(out_root / "features_dev_full_micro_DC.csv", index=False)
    df_holdout.to_csv(out_root / "features_holdout_full_micro_DC.csv", index=False)

    # 只含最終特徵的表（給模型訓練、Transition 對照用）
    meta_cols_out = ["filename", "record_id", "t_start", "label", "leak_quantity",
                     "distance_cm", "pressure", "session_id", "file_date"]
    df_dev[[c for c in meta_cols_out if c in df_dev.columns] + top_feature_names].to_csv(
        out_root / "features_dev_selected_micro_DC.csv", index=False)
    df_holdout[[c for c in meta_cols_out if c in df_holdout.columns] + top_feature_names].to_csv(
        out_root / "features_holdout_selected_micro_DC.csv", index=False)

    # 最終特徵名稱、dev / holdout 的檔案清單（日後重現切分用）
    pd.Series(top_feature_names).to_csv(out_root / "selected_feature_names_micro.csv", index=False, header=["feature_name"])
    pd.Series(sorted(dev_record_ids)).to_csv(out_root / "dev_record_ids_micro.csv", index=False, header=["record_id"])
    pd.Series(sorted(holdout_record_ids)).to_csv(out_root / "holdout_record_ids_micro.csv", index=False, header=["record_id"])

    # 資料來源紀錄：日後回頭看，知道這組特徵是用哪些資料、什麼條件選出來的，成績多少
    # float(...)：numpy 的數字型別不能直接寫進 JSON，先轉成一般小數
    data_info = {
        "data_root": str(data_root),
        "reserved_test_dates": list(RESERVED_TEST_DATES),             # 保留給最終推論的日期（Step2 完全沒用）
        "duplicate_policy": DUPLICATE_POLICY,
        "dev_dates": sorted(record_meta_dev_df["file_date"].unique().tolist()),
        "holdout_dates": sorted(record_meta_holdout_df["file_date"].unique().tolist()),
        "n_dev_records": len(dev_record_ids),
        "n_holdout_records": len(holdout_record_ids),
        "n_dev_sessions": len(dev_sessions),
        "n_holdout_sessions": len(holdout_sessions),
        "cv_grouping": choose_cv_groups(df_dev)[1],                   # fold-wise CV 用天還是場次分組
        "used_date_stability": selection["used_date_stability"],      # 選特徵時有沒有納入跨天穩定性
        "n_eligible_features": len(eligible_features),
        "reference_cv_auc": float(ref_scores.mean()),
        "foldwise_cv_auc": float(foldwise_aucs.mean()),
        "foldwise_cv_f1": float(foldwise_scores.mean()),
        "lodo_window_auc_mean": float(lodo_summary_df["window_auc"].mean()) if not lodo_summary_df.empty else None,
        "rf_feature_selection_holdout_auc": float(holdout_auc),
    }

    # 特徵契約：Transition / Disturbance / Runtime 讀這份檔案決定要算哪些特徵
    # 執行計畫會從最終特徵自動反推，並檢查 runtime 算不算得出來（有傳 band_config）
    # 後處理參數（門檻、連續窗數等）用函式預設值，正式數值由後規則程式定案
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
        corr_threshold=CORR_THRESHOLD,
        temporal_band_names=band_config.temporal_band_names,   # 單一視窗內時間結構特徵的頻帶
        band_config_path=BAND_CONFIG_PATH,
        band_config_hash=band_config_hash,
        feature_extraction_config_path=FEATURE_EXTRACTION_CONFIG_PATH,
        feature_extraction_config_hash=feature_extraction_config_hash,
        baseline_path=FIXED_BASELINE_PATH,
        baseline_payload=fixed_baseline,
        band_config=band_config,
        data_info=data_info,
        source_note="step2_micro_leak_feature_selection_final",
        include_offband_summary=include_offband_summary,
    )

    print(f"\n✅ Micro-leak 流程完成，結果已輸出到 {out_root}")














