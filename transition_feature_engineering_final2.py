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
6. Feature config 與 Site Baseline SHA-256 必須一致。
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
7. 使用 window 中心時間 t_ref 貼上標籤：
   - t_ref <= normal_end_sec：label=0
   - normal_end_sec < t_ref < leak_start_sec：灰區
   - t_ref >= leak_start_sec：label=1
8. 建立 Offband、Leak-focus、Resonance summary。
9. 保留完整正常、灰區及洩漏序列。
10. 依 Execution Plan 套用固定 Baseline，建立 rel/z。
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
)


# ============================================================
# Baseline 資料隔離設定
# ============================================================
# Transition 不應與建立固定 Baseline 的正常 TDMS 重複。
# True：發現相同 TDMS 路徑或 record_id 時直接停止。
# False：只顯示警告，僅供舊資料流程測試。
STRICT_BASELINE_ISOLATION = True


# ============================================================
# 備用工具：從 selected_feature_names 回推依賴欄位
# 目前正式流程已改為：
# 在正式 bands 內先建立完整特徵鏈，最後再保留 selected features
# 這個函式先保留，供未來若要做「只算必要特徵」精簡版時使用
# ============================================================
def parse_required_features(selected_feature_names):     # selected_feature_names :Step2 最後選出來、寫進 feature_config.json 裡的那些正式特徵
    # 從模型最後要的特徵名稱，反推前面特徵工程至少要先算哪些基礎欄位
    raw_base_features = set()    # 前面最先要先算出來的原始欄位
    z_base_features = set()      # 哪些欄位後面還要先建成 z_*
    final_keep_features = set(selected_feature_names)    # 最後真的要保留給模型的正式特徵欄位

    # 只要看到這些前綴，就把前綴剝掉，找回原始欄位名稱
    prefix_map = {
        "rel_": "",
        "z_": "",
        "delta_prev3mean_": "",
        "delta_prev5mean_": "",
        "delta_": "",
        "local_rel3_": "",
        "local_rel5_": "",
        "local_z3_": "",
        "local_z5_": "",
        "rolling_std3_": "",
        "rolling_std5_": "",
        "rolling_slope3_": "",
        "rolling_slope5_": "",
        "rolling_max_gap3_": "",
        "rolling_max_gap5_": "",
    }

    # 把模型最後要的每個特徵，一個一個拿來分析
    for feat in selected_feature_names:
        # 先特別處理 cusum, 因為 cusum 的依賴關係比別的特徵多一層
        if feat.startswith("cusum_pos_") or feat.startswith("cusum_neg_"):
            z_feat = feat.replace("cusum_pos_", "", 1).replace("cusum_neg_", "", 1)
            if z_feat.startswith("z_"):
                raw_base_features.add(z_feat[len("z_"):])
                z_base_features.add(z_feat)
            continue

        # 一般衍生特徵處理:模型最後選到的是衍生特徵，這裡就把它往回追到原始欄位
        matched = False
        for prefix in prefix_map:
            if feat.startswith(prefix):
                raw_base_features.add(feat[len(prefix):])
                matched = True
                break
        # 如果不是任何衍生前綴, 那它本身就當成原始欄位
        if not matched:
            raw_base_features.add(feat)

    return (
        sorted(raw_base_features),      # 前面最先要先算的原始欄位
        sorted(z_base_features),        # 哪些 z_* 欄位後面還要拿去做 cusum
        sorted(final_keep_features),    # 最後真正要留下給模型的欄位
    )


# ============================================================
# 讀 band 設定檔
# ============================================================
def load_band_config(band_config_json):
    # 轉成 Path 物件
    band_config_json = Path(band_config_json)
    
    if not band_config_json.exists():
        raise FileNotFoundError(f"找不到正式 band config：{band_config_json}")
    
    payload = json.loads(Path(band_config_json).read_text(encoding="utf-8"))
    
    # 檢查
    if payload.get("schema_name") != "band_config":
        raise ValueError("band_config_step2.json 的 schema_name 必須是 band_config")

    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("目前只支援 band config schema_version=2")

    if payload.get("schema_status") != "final":
        raise ValueError("Transition 特徵工程只能載入 schema_status=final 的 band config")

    required_keys = [
        "leak_focused_bands", "resonance_bands",
        "offband_bands", "temporal_band_names",
    ]

    missing_keys = [key for key in required_keys if key not in payload]

    if missing_keys:
        raise ValueError(f"band_config_step2.json 缺少：{missing_keys}")

    for key in required_keys:
        if not isinstance(payload[key], list):
            raise ValueError(f"band_config_step2.json 的 {key} 必須是 list")

    if not payload["leak_focused_bands"]:
        raise ValueError("正式 band config 的 leak_focused_bands 不可為空")

    # 目前正式流程一定需要 Offband 建立抗干擾規則，因此 Transition 也必須保留相同的 Offband 特徵，才能驗證正式 Veto 規則。
    if not payload["offband_bands"]:
        raise ValueError("正式 band config 的 offband_bands 不可為空")

    # 回傳頻段
    return BandConfig(
        leak_focused_bands=list(payload["leak_focused_bands"]),
        resonance_bands=list(payload["resonance_bands"]),
        offband_bands=list(payload["offband_bands"]),
        temporal_band_names=set(payload["temporal_band_names"]),
    )


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
    use_center=True,          # 用 window 中心時間還是起始時間判斷
    drop_transition=False,    # 是否丟掉中間灰區
):
    '''把每個 window 的時間位置，對照 transition_meta 裡的事件時間，幫每一列貼上分類標籤：
        0：正常段
        1：洩漏段
        中間區間：過渡灰區
    '''
    
    # 複製一份，避免直接改原始 DataFrame
    df = df_windows.copy()
    meta = df_meta.copy()

    # 用 record_id 把 window 特徵表和 transition_meta.csv 接起來
    df = df.merge(
        meta[
            [
                "record_id",
                "interval_name",
                "distance_cm",
                "leak_quantity",
                "pressure_bar",
                "output_tdms",
                "normal_end_sec",
                "leak_start_sec",
                "onset_gt_sec",
            ]
        ],
        on="record_id",
        how="left"
    )

    # 決定用哪個時間點貼標:如果 use_center=True，就用 window 中心時間 = t_start + 0.1
    if use_center:
        df["t_ref"] = df["t_start"] + window_sec / 2.0
    else:   # 否則用 window 開始時間
        df["t_ref"] = df["t_start"]

    # 先把所有 label 設成空值
    df["label"] = np.nan
    
    # 正常區
    df.loc[df["t_ref"] <= df["normal_end_sec"], "label"] = 0    # 如果 window 參考時間在 normal_end_sec 之前，標 0
    # 洩漏區
    df.loc[df["t_ref"] >= df["leak_start_sec"], "label"] = 1    # 如果在 leak_start_sec 之後，標 1

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
# 抓全部 raw 特徵欄位
# ============================================================
def select_all_raw_feature_columns(df):    # df 是目前累積到某個階段的完整特徵表(可能已經包含 band 原始特徵、offband 摘要特徵,但還沒加上rel_/z_/cusum_這些衍生特徵)
    # 建立排除欄位清單(metadata/標註/時間欄位,不是真正的訊號特徵)
    exclude_cols = {
        "filename", "record_id", "t_start", "t_end", "t_ref", "label",
        "interval_name", "distance_cm", "leak_quantity", "pressure_bar",
        "output_tdms", "normal_end_sec", "leak_start_sec", "onset_gt_sec",
    }

    # 列出六種衍生特徵前綴,凡是這幾種開頭的欄位都算「後面才算出來的衍生特徵」,不是原始特徵
    derived_prefixes = (
        "rel_", "z_", "cusum_", "delta_", "local_", "rolling_"
    )

    # 逐一掃過df.columns,三個條件都要滿足才留下:不在黑名單裡、是數值型態、不是那六種前綴開頭
    raw_cols = [
        c for c in df.columns
        if (
            c not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[c])
            and not c.startswith(derived_prefixes)
        )
    ]
    return raw_cols


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
    band_config = load_band_config(band_config_json)          # 讀 Step2 的正式 band 設定
    extraction_cfg = load_feature_extraction_config(feature_extraction_config_path)
    feature_cfg = load_feature_config(feature_config_json)    # 讀 Step2 的正式 feature 設定:這是 feature_config.json 的流程設定，不是特徵資料表本身

    fixed_baseline = load_fixed_normal_baseline(
        baseline_json,
        band_config_path=band_config_json,
        feature_extraction_config_path=(
            feature_extraction_config_path
        ),
        require_final=require_final_baseline,
    )
    
    current_band_hash = calculate_file_sha256(band_config_json)
    
    current_extraction_hash = calculate_file_sha256(feature_extraction_config_path)
    
    if (feature_cfg.get("band_config_sha256") != current_band_hash):
        raise ValueError("feature_config_micro.json 與目前 band_config_step2.json 不一致")
    
    if (feature_cfg.get("feature_extraction_config_sha256") != current_extraction_hash):
        raise ValueError("feature_config_micro.json 與目前 feature_extraction_config 不一致")
    
    if (feature_cfg.get("baseline_config_fingerprint") != fixed_baseline.get("config_fingerprint")):
        raise ValueError("feature_config_micro.json 與目前 固定 baseline 不一致")
    
    # 把 band config 的 temporal band 名稱,同步給 shared_feature_utils 的 raw feature extractor。
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
    
    baseline_feature_names = list(dict.fromkeys(execution_plan["baseline_features"]))
    cusum_input_feature_names = list(dict.fromkeys(execution_plan["cusum_input_features"])) 
    temporal_base_feature_names = list(dict.fromkeys(execution_plan["temporal_base_features"])) 
    planned_final_features = list(dict.fromkeys(execution_plan["final_keep_features"]))
    
    if planned_final_features != final_keep_features:
        raise ValueError("feature_execution_plan.final_keep_features 與 selected_feature_names 不一致")
    
    print("固定 baseline 特徵數量:",len(baseline_feature_names),)    
    print("CUSUM 輸入特徵數量:",len(cusum_input_feature_names),)
    print("temporal base 特徵數量:",len(temporal_base_feature_names),)
    print("最終 selected features 數量:", len(final_keep_features),)
    print("需要的 final selected features 數量:", len(final_keep_features))

    
    # ============================================================
    # -----  4. 從 feature_config.json 取正式參數(不是模型本體，是正式特徵工程流程規格)  -----
    # ============================================================
    window_sec = float(extraction_cfg["window_sec"])
    hop_sec = float(extraction_cfg["hop_sec"])
    fs = float(extraction_cfg["fs"])
    nperseg = int(extraction_cfg["nperseg"])
    max_points = int(extraction_cfg["max_points"])
    base_feature_version = str(extraction_cfg["base_feature_version"])
    include_offband_summary = bool(extraction_cfg["include_offband_summary"])
    
    if "cusum_k" not in feature_cfg:
        raise ValueError("feature_config_micro.json 缺少 cusum_k")
    if "cusum_decay" not in feature_cfg:
        raise ValueError("feature_config_micro.json 缺少 cusum_decay")
    
    cusum_k = float(feature_cfg["cusum_k"])
    cusum_decay = float(feature_cfg["cusum_decay"])


    # ============================================================
    # -----  5. 從 meta 裡決定哪些 TDMS 要處理  -----
    # ============================================================
    valid_record_ids = set(df_meta["record_id"])         # 先從 meta_csv 抓出合法的 record_id
    all_tdms_files = sorted(tdms_dir.rglob("*.tdms"))    # 掃 tdms_dir 底下所有 .tdms
    file_specs = [                                       # 只保留檔名主體有出現在 meta 裡的檔案
        {"filepath": str(p), "record_id": p.stem}
        for p in all_tdms_files
        if p.stem in valid_record_ids
    ]

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

    
    # ============================================================
    # -----  6. 真正 band 特徵抽取  -----
    # ============================================================
    # 對每個 TDMS 切窗 -> 在core_bands(洩漏頻帶+共振頻帶合併)跟aux_bands(offband)上算出各種基礎頻帶特徵,low_bands留空代表不再用舊版低頻設計
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
    
    # 檢查
    audit_band_feature_coverage(df, band_config)
    
    # 清除記憶體
    del features
    gc.collect()

    
    # ============================================================
    # -----  7. 根據 meta 幫每個 window 貼 transition 標註  -----
    # ============================================================
    # 把每個 window 對回 transition meta, 決定這個 window 是正常還是洩漏, 並補上後面會需要的欄位(label, normal_end_sec, leak_start_sec, onset_gt_sec, distance_cm, pressure_bar)
    df = attach_window_labels_from_transition_meta(
        df_windows=df,
        df_meta=df_meta,
        window_sec=window_sec,
        use_center=True,
        # 先保留灰區，讓 CUSUM/temporal 看見完整連續序列
        drop_transition=False,
    )
    
    # transition metadata 對齊檢查
    required_transition_meta_cols = [
        "normal_end_sec", "leak_start_sec", "onset_gt_sec",
    ]
    
    missing_meta_values = {
        column: int(df[column].isna().sum())
        for column in required_transition_meta_cols
        if (
            column not in df.columns
            or df[column].isna().any()
        )
    }
    
    if missing_meta_values:
        raise ValueError(f"Transition windows 無法完整對齊 metadata：{missing_meta_values}")
        
    # 可選擇建立 offband / leak-focused / resonance 摘要欄位(預設是)
    if include_offband_summary:
        df = build_offband_summary_features(df, band_config)
        # 檢查
        audit_offband_summary_coverage(df)
        
    # 保留作為報表:從目前的df裡挑出所有「原始特徵欄位」(排除 metadata、排除衍生特徵前綴),這份清單接下來會被拿去當作rel_/z_/CUSUM/temporal 特徵鏈的計算基礎
    all_raw_feature_cols = select_all_raw_feature_columns(df)
    print("正式 raw feature 數量:", len(all_raw_feature_cols))
    

    # ============================================================
    # Step 8：使用固定正常 baseline 建立 rel / z 特徵
    #
    # baseline_feature_names 由 Step 2 的 execution plan 提供。
    # dev、holdout、transition、disturbance 與 Runtime
    # 都使用同一份固定 baseline。
    # normal_end_sec 只負責 transition 標註，不參與 baseline 統計。
    # ============================================================
    # Baseline 前動態檢查
    validate_feature_stage_requirements(df, baseline_feature_names,
        stage_name=("Transition / fixed baseline 前置 raw"),)
    
    validate_feature_stage_requirements(df, temporal_base_feature_names,
        stage_name=("Transition / temporal 前置 raw"),)

    # 固定 baseline
    df = apply_fixed_normal_baseline(df=df, feature_cols=baseline_feature_names, baseline_payload=fixed_baseline,)
    
    # 檢查 rel/z
    expected_baseline_derived_features = [
        derived_feature
        for raw_feature in baseline_feature_names
        for derived_feature in ( f"rel_{raw_feature}", f"z_{raw_feature}",)
    ]
    
    validate_feature_stage_requirements(df, expected_baseline_derived_features,
        stage_name=("Transition / fixed baseline rel-z 輸出"),
    )
    
    
    # ============================================================
    # Step 9. 建 CUSUM 特徵
    # 把所以 z_* 欄位後面還需要再建 cusum, CUSUM 的作用是：把持續偏移或累積偏移的趨勢放大出來
    # ============================================================
    # CUSUM 前置檢查
    validate_feature_stage_requirements(df, cusum_input_feature_names,
        stage_name=("Transition / CUSUM 輸入"),)
    
    # 建 CUSUM 特徵
    if cusum_input_feature_names:
        df = add_cusum_features(df=df, input_feature_cols=(cusum_input_feature_names), k=cusum_k, decay=cusum_decay,)
        
    # CUSUM 後檢查輸出
    expected_cusum_output_features = [
        output_feature
        for input_feature in cusum_input_feature_names
        for output_feature in (f"cusum_pos_{input_feature}", f"cusum_neg_{input_feature}",)]
    
    validate_feature_stage_requirements(df, expected_cusum_output_features,
        stage_name=( "Transition / CUSUM 輸出"),)

    
    # ============================================================
    # Step 10. 建 temporal 類特徵
    # 針對原始欄位建立時間上的衍生特徵
    # delta_*/delta_prev*mean_*(window 間變化量)
    # local_rel*/local_z*/rolling_std*/rolling_slope*/rolling_max_gap*(局部滾動統計)這一整批 temporal 特徵
    # ============================================================
    # Temporal 前置檢查
    validate_feature_stage_requirements(
        df,
        temporal_base_feature_names,
        stage_name=("Transition / temporal 輸入"),
    )
    
    # 建 temporal 類特徵
    if temporal_base_feature_names:
        df = add_window_dynamic_features(
            df=df,
            feature_cols=temporal_base_feature_names,
            group_col="record_id",
            time_col="t_ref",
        )

        df = add_local_rolling_features(
            df=df,
            feature_cols=temporal_base_feature_names,
            group_col="record_id",
            time_col="t_ref",
        )

    # 最後 selected features 檢查
    validate_feature_stage_requirements(
        df,
        final_keep_features,
        stage_name=("Transition / final selected features"),
    )
        
    
    # ============================================================
    # Step 11：完成連續時間特徵後，才處理 transition 灰區
    # ============================================================
    # 完整連續序列：保留 normal、transition 灰區與 leak，可供時間序列檢查與 onset 分析
    df_continuous = df.copy()
    gray_mask = df_continuous["label"].isna()
    
    print("完整連續 windows：",len(df_continuous),)
    print("transition 灰區 windows：", int(gray_mask.sum()),)
    
    # 模型訓練／分類評估資料：
    # 只保留具有明確 0/1 label 的 windows
    df_labeled = df_continuous.loc[~gray_mask].copy()
    
    df_labeled["label"] = (df_labeled["label"].astype(int))
    
    print("明確標籤 windows：",len(df_labeled),)
    
    # 定義輸出要保留哪些欄位(13 個), base_output_cols基本 metadata / 標註欄位 , final_keep_features模型真正要吃的特徵欄位
    base_output_cols = [
        "filename", "record_id", "t_start", "t_ref", "label",
        "interval_name", "distance_cm", "leak_quantity", "pressure_bar",
        "output_tdms", "normal_end_sec", "leak_start_sec", "onset_gt_sec",
    ]
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # 檢查
    audit_selected_feature_alignment(df, feature_cfg)
    
    # 完整連續版本:先輸出一份「完整版」csv(檔名加上_full後綴),保留這次特徵工程算出來的所有欄位(不只是模型要的 5 個),供你之後核對
    full_output_csv = output_csv.with_name(output_csv.stem + "_full.csv")      # 如果你想確認 offband 摘要欄位到底有沒有被建立,直接打開這份_full.csv
    
    # 這份 _full.csv 會包含灰區，可以檢查真正連續的：- raw , rel/z , CUSUM , temporal , onset 前後變化
    df_continuous.to_csv(full_output_csv,index=False, encoding="utf-8-sig",)
    print(f"✅ full continuous feature csv 已輸出: {full_output_csv}")

    # 明確標籤 selected 版本:真正要交給推論程式用的精簡版欄位清單(基本欄位+模型需要的 5 個特徵)
    requested_output_cols = (base_output_cols + final_keep_features)
    output_cols = list(dict.fromkeys(c for c in requested_output_cols if c in df_labeled.columns))
    
    
    # ============================================================
    # 連續 selected 版本：供 transition onset 推論使用
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
        print("⚠️ 輸出欄位順序或內容和 Step2 selected_feature_names 不完全一致")
    else:
        print("✅ 輸出特徵欄位與 Step2 selected_feature_names 完全一致")
    
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
    
        # esonance 頻帶相對 baseline 的偏移量與 z-score。用來補充判斷共振頻帶是否異常。
        "rel_resonance_E_mean",
        "z_resonance_E_mean",
        "rel_resonance_E_max",
        "z_resonance_E_max",
    ]
    
    # 也保留每個 offband 原始頻帶特徵，方便未來規則直接指定單一 offband 欄位
    extra_rule_cols += [ c for c in df_labeled.columns if c.startswith("off_")]
    # 去除重複欄位，而且保留原本順序，過濾掉不存在的欄位
    extra_rule_cols = [ c for c in dict.fromkeys(extra_rule_cols) if c in df_labeled.columns]
    # 決定 rule CSV 最後要輸出的欄位(基本 metadata 欄位 + 模型 selected features + offband / veto 規則欄位)
    rule_output_cols = list(dict.fromkeys(base_output_cols + final_keep_features + extra_rule_cols))
    # 從 df_labeled 裡取出剛剛決定的欄位，建立 rule 版輸出表
    df_rule_out = df_labeled[rule_output_cols].copy()
    # 建立 rule CSV 的檔名
    rule_output_csv = output_csv.with_name(output_csv.stem.replace("_model", "_rule") + ".csv")
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
        n_jobs=2,
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


