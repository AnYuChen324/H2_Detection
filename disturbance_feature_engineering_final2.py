#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正式 Disturbance 特徵工程流程
============================

本程式讀取連續 Disturbance TDMS、Session Manifest
與 Event 標註，使用 Step2 定案的正式頻帶、
特徵萃取設定、固定正常 Baseline 與
Feature Execution Plan，建立模型與抗干擾規則
需要的 window-level 特徵表。

本程式不執行：

- 頻帶搜尋
- 特徵選擇
- Baseline 建立
- 模型訓練
- 模型機率預測
- 最終告警／Veto 規則定案


一、正式輸入
------------

1. Disturbance TDMS
2. Session Manifest CSV
3. Disturbance Event CSV
4. band_config_step2.json
5. feature_extraction_config.json
6. site_baseline.json
7. feature_config_micro.json


二、執行前契約驗證
------------------

1. Band config 必須為 schema_status=final。
2. Leak-focused bands 不可為空。
3. Offband bands 不可為空。
4. Feature config 與 Band config SHA-256 必須一致。
5. Feature config 與 Extraction config SHA-256 必須一致。
6. Feature config 與固定 Baseline 契約必須一致。
7. Feature Execution Plan 必須完整。
8. selected_feature_names 必須與 final_keep_features 一致。
9. Disturbance TDMS 不得與 Baseline calibration TDMS 重複。
10. Manifest 與 Event 欄位必須完整。
11. 每支 TDMS 必須能唯一對應到 Manifest。

任一契約不一致時立即停止，不使用 fallback。


三、Session 與時間軸整理
------------------------

1. 掃描每個 Disturbance Session。
2. 配對 TDMS、Manifest 與 Event CSV。
3. 同一 Session 的多支 TDMS 使用相同 record_id。
4. 保留每支檔案的局部時間：
   - t_start_local
   - t_ref_local
5. 使用 capture_start_elapsed_sec 轉換成 Session 全域時間：
   - t_start
   - t_ref
6. 依 record_id 與全域時間排序。
7. CUSUM 與 Temporal 不可跨不同 Session。


四、Disturbance 標註
--------------------

對每個 Window 計算與 Event 的時間重疊：

- disturbance_flag
- disturbance_type
- disturbance_strength
- overlap_sec
- overlap_ratio
- eval_class

目前只要重疊時間大於 0：

    disturbance_flag = 1
    eval_class = disturbance_negative

沒有重疊：

    disturbance_flag = 0
    eval_class = clean_normal

所有 Disturbance 資料的 Leak ground truth 均為：

    label = 0

baseline_end_sec 僅保留作為後續分析 metadata，
不參與固定 Baseline mean/std 計算。


五、特徵計算流程
----------------

1. 使用正式 Leak-focused、Resonance 與 Offband
   建立完整 Raw features。
2. 建立 Offband、Leak-focus、Resonance summary 與 ratio。
3. 依 Execution Plan 取得模型 Baseline 輸入欄位。
4. 額外加入規則需要的 Summary Baseline 欄位。
5. 使用固定 Site Baseline 建立 rel/z。
6. 對 Execution Plan 指定的 z 特徵建立 CUSUM。
7. 對 Execution Plan 指定的基礎特徵建立：
   - Delta
   - Local relative/Z
   - Rolling standard deviation
   - Rolling slope
   - Rolling max gap
8. 驗證所有 final_keep_features 均存在且有效。

正式頻帶的 Raw features 保持完整；
衍生特徵依 Execution Plan 針對性建立。


六、Baseline 與時間狀態原則
---------------------------

Disturbance 與 Transition 使用同一份固定正常 Baseline：

    rel_x = x - baseline_mean

    z_x = (
        x - baseline_mean
    ) / baseline_std

不使用：

- 每條 Session 的前段資料重建 Baseline
- baseline_end_sec 重新估計 mean/std
- 每支 TDMS 的前幾個 Windows
- 當下資料 mean/std
- mean=0、std=1 fallback

本離線流程將完整 Session 串接後計算 CUSUM/Temporal。
Runtime 的跨 Buffer 狀態由 Runtime 程式另外管理。


七、模型與規則特徵分離
----------------------

Model features：

- 僅使用 feature_execution_plan.final_keep_features
- 不允許加入 Offband／Veto Rule-only features

Rule features：

- 模型 selected features
- Offband Raw features
- Offband/Leak-focus/Resonance summary
- Offband ratio
- Summary rel/z

Offband 只供模型預測後的抗干擾規則使用，
不得進入模型訓練。


八、正式輸出
------------

1. inference_features_disturbance_full.csv

- 全部 Raw、Summary 與衍生特徵
- 全部 Session/Event metadata
- 用於稽核與除錯

2. inference_features_disturbance_model.csv

- Metadata
- final_keep_features
- 用於模型訓練或模型機率推論
- 不包含 Offband Rule-only 特徵

3. inference_features_disturbance.csv

- Metadata
- final_keep_features
- Offband／Veto 規則候選欄位
- 用於 Disturbance 告警規則比較


九、後續流程
------------

Disturbance 特徵工程
→ 模型機率推論
→ 候選告警／Veto 規則比較
→ 輸出前幾名候選規則
→ Transition 回測
→ 人工選定正式規則
→ 產生 postprocess_config.json
→ 產生 runtime_feature_schema.json
"""

# ============================================================
# 載入套件
# ============================================================
from pathlib import Path, PureWindowsPath
import sys
import gc
import pandas as pd
import numpy as np
import json
import warnings

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

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
    build_continuous_segments,    # v22 連續段：依取樣序號把首尾相接的 0.5 秒檔組成一段
)


# ============================================================
# Baseline 資料隔離設定
# ============================================================
# Disturbance 不應與建立固定 Baseline 的正常 TDMS 重複。
# True：發現相同 TDMS 路徑、record_id 或檔案內容時直接停止
# False：只顯示警告，僅供舊資料流程測試。
STRICT_BASELINE_ISOLATION = True


# ============================================================
# 已知有問題、暫時排除的單支 TDMS（寫清楚原因；確認資料後再移除）
# ============================================================
EXCLUDED_FILES = {
    "leak0_0ml_4bar_straight0cm_20260624_153744_dual_2ch":  "adjacent_vibration/detection01：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_left90cm_20260624_162214_dual_2ch":     "adjacent_vibration/detection08：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_straight50cm_20260702_113640_dual_2ch": "adjacent_vibration/detection11：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_straight50cm_20260702_113937_dual_2ch": "adjacent_vibration/detection14：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_straight0cm_20260707_162817_dual_2ch":  "adjacent_vibration/detection25：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_straight0cm_20260707_162931_dual_2ch":  "adjacent_vibration/detection26：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_left90cm_20260624_161924_dual_2ch":     "cable_pull/detection08：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_straight50cm_20260624_160109_dual_2ch": "tap_pipe/detection06：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_straight0cm_20260707_152721_dual_2ch":  "tap_pipe/detection24：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_straight50cm_20260702_104904_dual_2ch": "touch_pipe/detection10：manifest 位置錯置（檔名早於 session 開始）",
    "leak0_0ml_4bar_straight50cm_20260702_105352_dual_2ch": "touch_pipe/detection15：manifest 位置錯置（檔名早於 session 開始）",
}


# ============================================================
# 內容重複檢查（比檔案內容，不比檔名；改過檔名的複本也抓得到）
# ============================================================
def check_content_duplicates(file_specs, baseline_paths, strict=True):
    """
    1. Disturbance 檔案彼此內容相同 → 同一段訊號被算兩次
    2. Disturbance 跟建立 baseline 的正常檔內容相同 → baseline 資料洩漏
    同一 session 的多支 TDMS 共用 record_id，所以回報時用檔名。
    """
    seen = {}             # 指紋 → 第一支出現的檔名
    dup_pairs = []        # disturbance 內部重複
    baseline_hits = []    # 和 baseline 來源檔內容相同

    for spec in file_specs:
        name = Path(spec["filepath"]).stem
        h = calculate_file_sha256(spec["filepath"])
        if h in seen:
            dup_pairs.append((seen[h], name))
        else:
            seen[h] = name

    missing_baseline = 0
    for bp in baseline_paths:
        if not Path(bp).exists():
            missing_baseline += 1
            continue
        h = calculate_file_sha256(bp)
        if h in seen:
            baseline_hits.append((Path(bp).stem, seen[h]))

    if missing_baseline:
        print(f"⚠️ baseline 來源檔有 {missing_baseline} 支找不到，這些無法比對內容")

    if not dup_pairs and not baseline_hits:
        print(f"✅ 內容重複檢查通過：{len(seen)} 支 disturbance 彼此不重複，也沒有跟 baseline 檔內容相同")
        return

    msg = (f"Disturbance 內容重複 {len(dup_pairs)} 組：{dup_pairs[:5]}；"
           f"與 baseline 內容相同 {len(baseline_hits)} 組：{baseline_hits[:5]}")
    if strict:
        raise ValueError(msg)
    print("⚠️ " + msg)


# ============================================================
# 讀 feature 設定檔
# ============================================================
def load_feature_config(feature_config_json):
    # 把 feature_config.json 讀進來，回傳一個字典
    return json.loads(Path(feature_config_json).read_text(encoding="utf-8"))


# ============================================================
# 把一個檔名字串整理成「不含副檔名的主檔名」
# ============================================================
def normalize_filename_stem(x):
    '''
    如果 x 是空值，就回傳空字串,否則把它轉成檔名主體，也就是去掉副檔名
    這通常是為了之後拿來做：features 跟 manifest 的檔名對齊
    '''
    if pd.isna(x):
        return ""
    return PureWindowsPath(str(x)).stem


# 計算算兩個時間區間到底重疊了幾秒
def overlap_seconds(a_start, a_end, b_start, b_end):
    '''
    計算：兩個時間區間重疊了幾秒
    用來判斷：
     - 某個 window 的時間區間
     - 跟某個干擾事件的時間區間
    '''
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# ============================================================
# 掃描 disturbance 資料夾
# 結構範例：
# root_dir/
#   adjacent_vibration/
#     disturbance01/
#       *.tdms
#       dual_session_manifest_*.csv
#       disturbance_events_*.csv
# ============================================================
def collect_disturbance_sessions(
    root_dir: Path,
    input_mode,      # "segments" 或 "per_file_legacy"
    fs,              # feature_extraction_config 的 fs（segments 模式換算點數與秒數用）
):
    """
    掃描每個 disturbance session，配對 TDMS、manifest、event。
    所有問題先收集起來，最後一次列出並停止（人工下載漏檔通常不只一個，方便一次補齊）。
    """
    file_specs = []       # 後面萃取特徵時要處理哪些 TDMS 檔（segments 模式是一段一筆）
    file_records = []     # segments 模式：一支檔一筆，給資料隔離與內容重複檢查用
    manifest_rows = []    # 每個 TDMS 對應的時間資訊與 metadata
    event_rows = []       # 每個干擾事件的標註資訊
    problems = []         # 收集所有問題，最後一次報錯
    session_dirs = {}     # session_id → 第一次出現的資料夾，用來檢查跨資料夾重複
    gap_info = []         # 每個 session 檔案之間的時間間隙，只供參考
    excluded = []         # 依 EXCLUDED_FILES 排除的檔案

    # 路徑統一轉成 Path 物件，並確認資料夾真的存在
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"找不到 disturbance 根目錄：{root_dir}")

    # 定義 manifest 與 event 必要欄位
    manifest_need = ["session_id", "capture_start_elapsed_sec", "capture_end_elapsed_sec", "capture_mid_elapsed_sec"]
    # 每個干擾事件必須知道：幾秒開始、幾秒結束、干擾類型(敲擊、震動、拉扯)、干擾強度
    event_need = ["start_elapsed_sec", "end_elapsed_sec", "disturbance_type", "disturbance_strength"]

    # 第一層：干擾種類資料夾 -> adjacent_vibration / cable_pull / tap_pipe / touch_pipe
    for disturbance_group_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        # 取得該類別的名稱
        disturbance_group = disturbance_group_dir.name

        # 第二層：每次收集的子資料夾 -> disturbance01 / disturbance02 ...
        for disturbance_dir in sorted(p for p in disturbance_group_dir.iterdir() if p.is_dir()):
            disturbance_id = disturbance_dir.name
            where = f"{disturbance_group}/{disturbance_id}"

            # ---- 尋找 manifest / event 檔：每個 session 必須剛好各 1 份 ----
            manifest_files = sorted(
                list(disturbance_dir.glob("dual_session_manifest_*.csv")) +
                list(disturbance_dir.glob("transition_dual_manifest_*.csv"))
            )
            event_files = sorted(
                list(disturbance_dir.glob("disturbance_events_*.csv")) +
                list(disturbance_dir.glob("transition_dual_disturbance_events_*.csv"))
            )
            
            # manifest, event CSV 必須剛好一份
            if len(manifest_files) != 1:
                problems.append(f"{where}：manifest 應該剛好 1 份，目前 {len(manifest_files)} 份 {[p.name for p in manifest_files]}")
                continue
            if len(event_files) != 1:
                problems.append(f"{where}：event 檔應該剛好 1 份，目前 {len(event_files)} 份 {[p.name for p in event_files]}")
                continue
            
            # 讀取兩份 CSV 成 DataFrame
            manifest_path = manifest_files[0]
            event_path = event_files[0]
            df_manifest = pd.read_csv(manifest_path)
            df_events = pd.read_csv(event_path)

            # ---- 檢查 manifest 結構 ----
            if df_manifest.empty:
                problems.append(f"{where}：manifest 是空的")
                continue
            # 檢查必要欄位是否缺失
            miss = [c for c in manifest_need if c not in df_manifest.columns]
            if miss:
                problems.append(f"{where}：manifest 缺少欄位 {miss}")
                continue
            # 從多種可能欄位名稱中，找出哪一欄存有 TDMS 檔名
            file_col = next((c for c in ["dual_file", "merged_file", "tdms_file", "filename"] if c in df_manifest.columns), None)
            if file_col is None:
                problems.append(f"{where}：manifest 缺少檔名欄位，現有欄位 {df_manifest.columns.tolist()}")
                continue
            
            # 排除已知有問題的單支檔：同時從 manifest 拿掉，後面的時間檢查才不會再算到它
            row_stems = df_manifest[file_col].map(lambda x: PureWindowsPath(str(x)).stem)
            drop_mask = row_stems.isin(EXCLUDED_FILES)
            
            # 連續段不能挖掉中間的檔（會變成兩段，而且 manifest 的時間判斷要重做），v22 資料有問題請直接修 manifest 或整個 session 不用
            if input_mode == "segments" and drop_mask.any():
                problems.append(f"{where}：segments 模式不支援 EXCLUDED_FILES，但這個 session 有 {int(drop_mask.sum())} 支在排除名單裡")
                continue
            
            for s in row_stems[drop_mask]:
                excluded.append((where, s, EXCLUDED_FILES[s]))
            df_manifest = df_manifest.loc[~drop_mask].reset_index(drop=True)
            if df_manifest.empty:
                problems.append(f"{where}：排除檔案後 manifest 沒有任何資料")
                continue

            # 驗證 session ID 唯一性：manifest 內只能有一個，而且不能和別的資料夾重複
            sids = df_manifest["session_id"].astype(str).str.strip().unique()
            if len(sids) != 1:
                problems.append(f"{where}：manifest 裡有 {len(sids)} 個不同的 session_id {list(sids)[:5]}")
                continue
            # 取得這次實驗的唯一 ID
            session_id = sids[0]
            # 若這個 ID 已在其他資料夾出現，代表兩個不同資料夾會被後續程式串成同一條時間序列，所以必須停止。
            if session_id in session_dirs:
                problems.append(f"session_id「{session_id}」重複出現在 {session_dirs[session_id]} 和 {where}，會被串成同一條時間序列")
                continue
            session_dirs[session_id] = where

            # 檢查 TDMS 的全域時間範圍:manifest 時間必須是數字，而且結束 > 開始
            cap_start = pd.to_numeric(df_manifest["capture_start_elapsed_sec"], errors="coerce")
            cap_end = pd.to_numeric(df_manifest["capture_end_elapsed_sec"], errors="coerce")
            cap_mid = pd.to_numeric(df_manifest["capture_mid_elapsed_sec"], errors="coerce")
            # 標示不合法的資料列：時間是空值；時間是文字；結束時間小於或等於開始時間。
            bad_time = cap_start.isna() | cap_end.isna() | cap_mid.isna() | (cap_end <= cap_start)
            # 只要有任何一列不合法，整個 session 不採用
            if bad_time.any():
                problems.append(f"{where}：manifest 有 {int(bad_time.sum())} 列時間是空值、非數字，或結束 ≤ 開始")
                continue

            # 確認 TDMS 檔案沒有時間重疊:同一 session 內的檔案時間不可互相重疊（重疊會讓全域時間重複、排序錯亂）
            order = cap_start.sort_values().index                   # 依每支檔案的開始時間排序，取得原始列索引順序
            # 依排序後順序取出開始、結束時間，轉成 NumPy 陣列
            s_sorted = cap_start.loc[order].to_numpy()
            e_sorted = cap_end.loc[order].to_numpy()
            # 比較相鄰檔案：若後一支在前一支結束前就開始，代表時間重疊。
            overlap = s_sorted[1:] < e_sorted[:-1] - 1e-3
            if overlap.any():
                problems.append(f"{where}：有 {int(overlap.sum())} 處檔案時間互相重疊")
                continue
            # 記錄檔案間的時間缺口：計算相鄰 TDMS 中間是否有空白時間
            gaps = s_sorted[1:] - e_sorted[:-1]
            n_gap = int((gaps > 1e-3).sum())
            # 有缺口不立刻停止，只記錄：哪一個 session；有幾個缺口；最大缺口多少秒。
            if n_gap:
                gap_info.append((session_id, n_gap, float(gaps.max())))

            # ---- 檢查 event CSV（event 檔存在但沒有內容 = 這次沒有干擾，合法） ----
            miss = [c for c in event_need if c not in df_events.columns]
            if miss:      # 即使沒有干擾事件，event CSV 也應保有欄位標題
                problems.append(f"{where}：event 檔缺少欄位 {miss}")
                continue
            df_events = df_events.copy()
            # event CSV 沒有資料列是合法的，代表這次 session 沒有任何干擾事件，可視為乾淨的無洩漏正常資料
            if df_events.empty:
                print(f"ℹ️ {where}：event 檔沒有任何干擾事件，整段視為 clean_normal")
            else:
                # 若有干擾事件，就檢查它們的開始／結束時間是否合法。
                ev_start = pd.to_numeric(df_events["start_elapsed_sec"], errors="coerce")
                ev_end = pd.to_numeric(df_events["end_elapsed_sec"], errors="coerce")
                bad_ev = ev_start.isna() | ev_end.isna() | (ev_end <= ev_start)
                if bad_ev.any():
                    problems.append(f"{where}：event 有 {int(bad_ev.sum())} 筆時間是空值、非數字，或結束 ≤ 開始")
                    continue
                # 將驗證後的數值時間寫回 DataFrame，確保後續使用的是數字型別
                df_events["start_elapsed_sec"] = ev_start
                df_events["end_elapsed_sec"] = ev_end
                
            # ---- segments 模式：用取樣序號組連續段，取代下面「逐支配對 + 檔名時間順序」的檢查 ----
            if input_mode == "segments":
                # 首尾相接、取樣點重疊、manifest 列了但找不到檔、同名檔，都在 build_continuous_segments 裡檢查
                try:
                    seg_specs, df_seg_files = build_continuous_segments(manifest_path, disturbance_dir, fs)
                except ValueError as e:
                    problems.append(f"{where}：{e}")
                    continue
                # 資料夾裡多出 manifest 沒列的 TDMS：來源不明，不允許
                extra_tdms = sorted({p.name for p in disturbance_dir.glob("*.tdms")} - set(df_seg_files["tdms_name"]))
                if extra_tdms:
                    problems.append(f"{where}：資料夾有 {len(extra_tdms)} 支 TDMS 不在 manifest 裡 {extra_tdms[:5]}")
                    continue

                # 和下面舊流程相同：baseline_end_sec 只當 metadata；事件補上 session 資訊
                if not df_events.empty:
                    baseline_end_sec = float(df_events["start_elapsed_sec"].min())
                else:
                    baseline_end_sec = float(cap_end.max())
                df_events["session_id"] = session_id
                df_events["disturbance_group"] = disturbance_group
                df_events["disturbance_id"] = disturbance_id
                event_rows.append(df_events)

                # 一段一筆交給特徵抽取；檔案層級的資訊另外記，用來把窗對回它起點所在的那支檔
                file_specs.extend(seg_specs)
                for r in df_seg_files.itertuples(index=False):
                    file_records.append({"filepath": r.filepath, "record_id": r.segment_id})
                    manifest_rows.append({
                        "session_id": session_id,
                        "record_id": r.segment_id,                                     # 一段一個 record：告警規則、CUSUM／temporal 依它分組
                        "disturbance_group": disturbance_group,
                        "disturbance_id": disturbance_id,
                        "filename_stem": Path(r.filepath).stem,
                        "tdms_file": str(r.filepath),
                        "capture_start_elapsed_sec": float(r.capture_start_elapsed_sec),
                        "capture_end_elapsed_sec": float(pd.to_numeric(r.capture_end_elapsed_sec)),
                        "capture_mid_elapsed_sec": float(pd.to_numeric(r.capture_mid_elapsed_sec)),
                        "offset_in_segment_sec": float(r.offset_in_segment_sec),       # 這支檔在段內從第幾秒開始
                        "baseline_end_sec": baseline_end_sec,
                        "label": 0,
                        "leak_status": "leak0",
                        "disturbance_session_manifest": str(manifest_path),
                        "disturbance_event_csv": str(event_path),
                    })
                continue            

            # ---- TDMS 配對：manifest 列的每支都要找到；資料夾裡多出來的也要列出 ----
            row_paths = []         # manifest 每一列對到的檔案路徑
            listed_stems = []      # 成功配對的檔名主體
            missing_tdms = []      # 找不到的檔案名稱
            # 逐一讀取 manifest 中記載的 TDMS 名稱
            for name in df_manifest[file_col]:
                # 取出檔名
                dual_file_name = PureWindowsPath(str(name)).name
                # 在目前 session 資料夾尋找該檔案
                tdms_path = disturbance_dir / dual_file_name
                # 若 manifest 裡副檔名格式不一致，再嘗試以檔名主體加 .tdms 尋找
                if not tdms_path.exists():
                    alt = disturbance_dir / f"{Path(dual_file_name).stem}.tdms"
                    # 替代檔也不存在，就標成找不到
                    tdms_path = alt if alt.exists() else None
                    
                # 每一列最後都保留一個位置；若找不到就是 None，最後再統一報錯
                if tdms_path is None:
                    missing_tdms.append(dual_file_name)
                else:
                    listed_stems.append(tdms_path.stem)
                row_paths.append(tdms_path)

            # 檢查 manifest 和資料夾檔案是否一一對應
            if missing_tdms:
                problems.append(f"{where}：manifest 列了但找不到 {len(missing_tdms)} 支 TDMS {missing_tdms[:5]}")
                continue
            # 找出 manifest 是否重複列同一支 TDMS
            dup_stems = sorted({s for s in listed_stems if listed_stems.count(s) > 1})
            # 同一檔案若被列兩次，會讓訊號被重複算進特徵工程
            if dup_stems:
                problems.append(f"{where}：manifest 同一支 TDMS 列了不只一次 {dup_stems[:5]}")
                continue
            # 找出資料夾中存在、但 manifest 沒有列出的 TDMS(資料夾裡被排除的那支 TDMS 不能被當成「多出來的檔案」。)
            extra_tdms = sorted({p.stem for p in disturbance_dir.glob("*.tdms")} - set(listed_stems) - set(EXCLUDED_FILES))
            # 不允許來源不明的 TDMS 被忽略或偷偷進入分析
            if extra_tdms:
                problems.append(f"{where}：資料夾有 {len(extra_tdms)} 支 TDMS 不在 manifest 裡 {extra_tdms[:5]}")
                continue
            
            # 檔名裡的錄製時間，依 manifest 時間排序後必須是遞增的（抓「放錯位置但剛好沒重疊」的檔案）
            ts = pd.to_datetime(
                pd.Series(listed_stems).str.extract(r"_(\d{8}_\d{6})_")[0],
                format="%Y%m%d_%H%M%S", errors="coerce",
            )
            if ts.notna().all():
                ts_sorted = ts.iloc[cap_start.argsort().to_numpy()].reset_index(drop=True)
                n_back = int((ts_sorted.diff().dt.total_seconds() < 0).sum())
                if n_back:
                    problems.append(f"{where}：有 {n_back} 支檔的檔名時間比前一支還早，manifest 位置可能錯置")
                    continue
            else:
                print(f"ℹ️ {where}：有檔名解析不出錄製時間，跳過時間順序檢查")

            # ---- 全部檢查通過，才收進結果，找出第一個干擾發生前的位置 ----
            # baseline_end_sec：有干擾事件就用第一個事件開始時間；沒有就用整段最後時間（只當 metadata，不參與 baseline 計算）
            if not df_events.empty:
                baseline_end_sec = float(df_events["start_elapsed_sec"].min())
            else:
                baseline_end_sec = float(cap_end.max())

            # 補上 session 與干擾條件資訊:屬於哪個 session；干擾大類；第幾次實驗。
            df_events["session_id"] = session_id
            df_events["disturbance_group"] = disturbance_group
            df_events["disturbance_id"] = disturbance_id
            event_rows.append(df_events)

            # 建立後續特徵工程需要的檔案清單
            for i, tdms_path in enumerate(row_paths):
                # 將檔案路徑與 record_id 放進特徵工程清單
                file_specs.append({
                    "filepath": str(tdms_path),
                    "record_id": session_id,
                })
                # 將每支 TDMS 的完整資訊加入 manifest 總表。
                manifest_rows.append({
                    "session_id": session_id,                                      # 屬於哪一個收集 session
                    "record_id": session_id,                                       # 同一 session 共用 record_id
                    "disturbance_group": disturbance_group,                        # 干擾大類
                    "disturbance_id": disturbance_id,                              # 哪一次干擾收集
                    "filename_stem": tdms_path.stem,                               # TDMS 檔名主體
                    "tdms_file": str(tdms_path),                                   # TDMS 完整路徑
                    # 全域時間位置
                    "capture_start_elapsed_sec": float(cap_start.iloc[i]),         # 在 session 時間軸上的開始時間
                    "capture_end_elapsed_sec": float(cap_end.iloc[i]),             # 在 session 時間軸上的結束時間
                    "capture_mid_elapsed_sec": float(cap_mid.iloc[i]),             # 在 session 時間軸上的中間時間
                    "baseline_end_sec": baseline_end_sec,                          # 第一個干擾開始前的時間點（只當 metadata）
                    "label": 0,                                                    # 真實 leak label：全部是無洩漏
                    "leak_status": "leak0",
                    "disturbance_session_manifest": str(manifest_path),
                    "disturbance_event_csv": str(event_path),
                })
                
    matched = {s for _, s, _ in excluded}
    not_found = sorted(set(EXCLUDED_FILES) - matched)
    if not_found:
        print(f"⚠️ EXCLUDED_FILES 有 {len(not_found)} 支在 manifest 裡找不到（檔名可能打錯）：{not_found}")

    if excluded:
        print(f"⚠️ 依 EXCLUDED_FILES 排除 {len(excluded)} 支 TDMS：")
        for w, s, reason in excluded:
            print(f"   - {w}：{s}")
            
    # ---- 一次列出所有問題 ----
    if problems:
        raise ValueError(
            f"Disturbance 資料有 {len(problems)} 個問題，請先修正：\n  - " + "\n  - ".join(problems)
        )
    # 如果最後完全沒有可用 TDMS，就停止
    if not file_specs:
        raise ValueError("找不到任何可處理的 disturbance TDMS")
    # 檔案之間有時間間隙：CUSUM/temporal 會把間隙前後的窗當成相鄰，只提示不停止
    if gap_info:
        worst = max(gap_info, key=lambda x: x[2])
        print(f"ℹ️ {len(gap_info)} 個 session 的檔案之間有時間間隙；最大間隙 {worst[2]:.3f} 秒（{worst[0]}）")
    # 把每支 TDMS 的 manifest 資訊整理成 DataFrame
    df_manifest_all = pd.DataFrame(manifest_rows)
    # 把所有 session 的事件資料合併成一張表；若所有 session 都沒有事件，則回傳空表
    df_events_all = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()

    if input_mode == "segments":
        print(f"✅ Disturbance session 檢查通過：{len(session_dirs)} 個 session、{len(file_specs)} 段、{len(file_records)} 支 TDMS")
    else:
        file_records = file_specs        # 舊流程一筆就是一支檔
        print(f"✅ Disturbance session 檢查通過：{len(session_dirs)} 個 session、{len(file_specs)} 支 TDMS")
    return file_specs, df_manifest_all, df_events_all, file_records


# ============================================================
# 依據 window 與干擾事件時間區間的重疊關係，把每個 window 標成 clean_normal 或 disturbance_negative
# 注意：呼叫前 t_start 必須已經換成 session 全域時間
# （t_start = capture_start_elapsed_sec + 檔內局部時間），
# 才能和 event 的 start/end_elapsed_sec 放在同一條時間軸比較。
# ============================================================
def attach_window_labels_from_disturbance_meta(
    df_windows: pd.DataFrame,    # 特徵表，每一列代表一個 window, 裡面至少要有：session_id,t_start
    df_events: pd.DataFrame,     # disturbance_events.csv 整理後的表。裡面至少要有：session_id, start_elapsed_sec, end_elapsed_sec, disturbance_type, disturbance_strength
    window_sec: float = 0.2,
):
    '''
    對每一個 window，判斷它有沒有落在干擾事件時間區間內。
     - 如果有，就標成：disturbance_negative
     - 如果沒有，就標成：clean_normal
    '''
    df = df_windows.copy()
    
    # 建立輸出欄位的暫存清單
    disturbance_flag = []         # 是否碰到干擾，1 是、0 否
    disturbance_type = []         # 重疊最多的干擾類型
    disturbance_strength = []     # 重疊最多事件的強度
    overlap_sec_list = []         # 與事件重疊幾秒
    overlap_ratio_list = []       # 重疊秒數除以 window 長度
    eval_class = []               # clean_normal 或 disturbance_negative

    # 先把 event 依 session 分組:把所有干擾事件依 session_id 分好
    if df_events.empty or "session_id" not in df_events.columns:
        event_map = {}    # 完全沒有干擾事件：所有窗都是 clean_normal
    else:
        # 將事件依 session 分組
        event_map = {
            sid: sub.sort_values("start_elapsed_sec").copy()
            for sid, sub in df_events.groupby("session_id")
        }
    
    # 開始逐個 window 處理
    for row in df.itertuples(index=False):
        # window 的基本資訊
        sid = str(row.session_id)        # 這個 window 屬於哪個 session
        w_start = float(row.t_start)     # window 起始時間
        w_end = w_start + window_sec     # window 結束時間

        # 初始化「目前找到的最佳重疊」,假設:目前沒有任何重疊,也不知道是哪種干擾、什麼強度
        best_overlap = 0.0      # 假設：一開始重疊 0 秒
        best_type = ""          # 假設：還不知道是哪種干擾
        best_strength = ""      # 假設：還不知道強度
        
        # 找這個 session 的干擾事件
        session_events = event_map.get(sid, None)    # 從剛剛那個 event_map 裡，取出這個 window 所屬 session 的所有干擾事件。
        # 如果這個 session 有事件，就逐一比較
        if session_events is not None and not session_events.empty:
            # 逐一檢查每個事件
            for ev in session_events.itertuples(index=False):
                # 計算這個 window 和某個事件重疊多少秒
                ov = overlap_seconds(
                    w_start, w_end,
                    float(ev.start_elapsed_sec), 
                    float(ev.end_elapsed_sec),
                )
                # 只保留重疊最多的那個事件,這個 window 會被歸到「與它重疊最多的那個干擾事件」
                if ov > best_overlap:
                    best_overlap = ov
                    best_type = str(ev.disturbance_type)
                    best_strength = str(ev.disturbance_strength)

        # 浮點誤差：例如 2.4 + 0.2 = 2.6000000000000005，剛好「碰到」事件起點的窗會被算成重疊 5e-16 秒；
        # 重疊不到 1 微秒（1 個取樣點）視為沒有重疊
        flag = 1 if best_overlap > 1e-6 else 0
        ratio = best_overlap / window_sec if window_sec > 0 else 0.0     # 算重疊比例

        # 把這個 window 的結果存進 list
        disturbance_flag.append(flag)
        disturbance_type.append(best_type if flag else "")
        disturbance_strength.append(best_strength if flag else "")
        overlap_sec_list.append(best_overlap)
        overlap_ratio_list.append(ratio)
        # 產生最終分類:disturbance_negative -> 有干擾，但不是 leak，仍屬負樣本, clean_normal -> 完全正常、沒有干擾的負樣本
        eval_class.append("disturbance_negative" if flag else "clean_normal")    
    
    # 最後把結果寫回 DataFrame
    df["label"] = 0                                      # 這整批資料在 leak / non-leak 的真實標籤上都屬於正常(label=0)；干擾資訊另外放在 disturbance_flag / eval_class
    df["disturbance_flag"] = disturbance_flag            # 1 表示碰到干擾，0 表示沒有
    df["disturbance_type"] = disturbance_type            # 干擾類型
    df["disturbance_strength"] = disturbance_strength    # 干擾強度
    df["overlap_sec"] = overlap_sec_list                 # 和最佳干擾事件重疊幾秒
    df["overlap_ratio"] = overlap_ratio_list             # 重疊比例
    df["eval_class"] = eval_class                        # disturbance_negative 或 clean_normal

    return df


# ============================================================
# 主流程
# ============================================================
def build_disturbance_inference_feature_csv(
    root_dir,                         # 干擾資料的根目錄
    band_config_json,                 # Step2 定案的主頻帶、共振頻帶、offband 設定
    feature_extraction_config_path,   # 切窗、取樣率、PSD 等特徵抽取規則
    baseline_json,                    # 固定正常 baseline，保存正常特徵的 mean / std 與來源資訊
    feature_config_json,              # Step2 選出的模型特徵與 Feature Execution Plan
    output_csv,                       # 輸出的推論特徵表路徑
    cache_dir,                        # TDMS 特徵計算結果的快取位置
    input_mode,                       # "segments"（v22 連續錄製，正式）或 "per_file_legacy"（舊資料）；必填
    split_name="disturbance_infer",   # 快取與輸出識別名稱
    n_jobs=2,                         # 平行處理的工作數
    require_final_baseline=False,     # 是否強制 baseline 必須標記為 final
):
    """將 disturbance TDMS、session manifest 與干擾事件標註，
        依 Step2 已定案的頻帶、固定 baseline 與特徵計畫，
        轉成可供模型與抗干擾規則評估的逐窗特徵表。"""
    # segments：v22 連續錄製（正式流程），同一段的 0.5 秒檔在記憶體串起來再切窗，跨檔的窗也會切出來
    # per_file_legacy：舊資料，每支檔各自切窗，再用 capture_start_elapsed_sec 排到 session 時間軸上
    if input_mode not in ("segments", "per_file_legacy"):
        raise ValueError(f"input_mode 只能是 'segments' 或 'per_file_legacy'，目前：{input_mode}")    
    
    # 把路徑轉成 Path
    root_dir = Path(root_dir)
    band_config_json = Path(band_config_json)
    feature_config_json = Path(feature_config_json)
    output_csv = Path(output_csv)
    cache_dir = Path(cache_dir)
    feature_extraction_config_path = Path(feature_extraction_config_path)
    baseline_json = Path(baseline_json)

    # 讀進 Step2 輸出的設定檔
    # 讀取正式頻帶設定，並強制兩件事：頻帶設定必須是 final + 必須有 offband
    band_config = load_band_config(band_config_json, expected_status="final", require_offband=True)     # 正式頻帶：必須是 final，而且要有 offband
    # 讀取共同的特徵抽取規則，例如：取樣率,視窗長度,Welch／PSD 的 nperseg,特徵版本。
    extraction_config = (load_feature_extraction_config(feature_extraction_config_path))
    # 讀取 Step2 選定的最終特徵，以及 Feature Execution Plan
    feature_config = load_feature_config(feature_config_json)
    
    # 讀取固定 normal baseline, 這個 baseline 會在後續將原始頻帶特徵轉成：rel_特徵, z_特徵
    fixed_baseline = load_fixed_normal_baseline(
        baseline_json,
        band_config_path=band_config_json,
        feature_extraction_config_path=(
            feature_extraction_config_path
        ),
        require_final=require_final_baseline,
    )
    
    # 驗證 Step 2 契約, 替目前使用的設定檔計算 SHA-256 指紋
    current_band_hash = calculate_file_sha256(band_config_json)
    current_extraction_hash = calculate_file_sha256(feature_extraction_config_path)
    
    # 檢查
    # 確認目前的頻帶設定，和 Step2 當初選特徵時用的頻帶設定完全一致
    if (feature_config.get("band_config_sha256") != current_band_hash):
        raise ValueError("feature_config_micro.json 與目前 band_config_step2.json 不一致")
    # 確認目前的切窗、取樣率、PSD 等規則，也和 Step2 當初選特徵時一致。
    if (feature_config.get("feature_extraction_config_sha256") != current_extraction_hash):
        raise ValueError("feature_config_micro.json 與目前 feature_extraction_config 不一致")
    
    # 取出：fp_cfg：Step2 特徵設定所記錄的 baseline 指紋, fp_base：現在實際載入的 baseline 指紋。
    fp_cfg = feature_config.get("baseline_config_fingerprint")
    fp_base = fixed_baseline.get("config_fingerprint")
    if not fp_cfg or not fp_base or fp_cfg != fp_base:
        raise ValueError(f"feature_config_micro.json 與目前固定 baseline 不一致：{fp_cfg} vs {fp_base}")
    
    # 把 band config 的 temporal band 名稱,同步給 shared_feature_utils 的 raw feature extractor。
    configure_temporal_band_names(band_config.temporal_band_names)
    
    # 載入 Step2 已定案的特徵清單
    selected_feature_names = feature_config.get("selected_feature_names", [],)
    # 若沒有這份清單，程式不知道最後要輸出哪些欄位給模型，因此停止。
    if not selected_feature_names:
        raise ValueError("feature_config_micro.json 缺少 selected_feature_names")
    
    # 建立正式保留的模型特徵欄位與順序
    final_keep_features = list(selected_feature_names)
    # 載入 Feature Execution Plan
    execution_plan = feature_config.get("feature_execution_plan")
    # 檢查
    if not isinstance(execution_plan, dict):
        raise ValueError("feature_config_micro.json 缺少 feature_execution_plan")
    
    # 強制 execution plan 必須包含四個區塊
    required_plan_keys = [
        "baseline_features",          # 要套用 baseline、計算 rel／z 的原始特徵
        "cusum_input_features",       # 要建立 CUSUM 的輸入欄位，通常是 z 特徵
        "temporal_base_features",     # 要建立 delta、rolling 等時間特徵的欄位
        "final_keep_features",        # 最終模型真正使用的特徵及順序
    ]
    # 檢查
    missing_plan_keys = [key for key in required_plan_keys if key not in execution_plan]
    if missing_plan_keys:
        raise ValueError(f"feature_execution_plan 缺少欄位：{missing_plan_keys}")
    
    # 確認每一個區塊都是清單，不會讀到格式錯誤的 JSON。＃
    for key in required_plan_keys:
        if not isinstance(execution_plan[key], list):
            raise ValueError(f"feature_execution_plan.{key} 必須是 list")
    
    # 去除重複特徵+保留原本特徵順序。
    baseline_feature_names = list(dict.fromkeys(execution_plan["baseline_features"]))
    cusum_input_feature_names = list(dict.fromkeys(execution_plan["cusum_input_features"]))   
    temporal_base_feature_names = list(dict.fromkeys(execution_plan["temporal_base_features"]))   
    planned_final_features = list(dict.fromkeys(execution_plan["final_keep_features"]))
    
    # 確認 execution plan 裡的最終特徵，和 selected_feature_names 完全一致，連欄位順序也一致。
    if planned_final_features != final_keep_features:
        raise ValueError("feature_execution_plan.final_keep_features 與 selected_feature_names 不一致")
    
    # 顯示這次要做多少特徵
    print("固定 baseline 輸入特徵數量:", len(baseline_feature_names))
    print("CUSUM 輸入特徵數量:", len(cusum_input_feature_names))
    print("temporal base 特徵數量:", len(temporal_base_feature_names))
    print("final selected features 數量:", len(final_keep_features))

    # 從特徵設定讀取切窗與頻譜參數
    window_sec = float(extraction_config["window_sec"])                             # 決定每個 window 多長
    hop_sec = float(extraction_config["hop_sec"])                                   # window 之間跳多少
    fs = float(extraction_config["fs"])                                             # 取樣率
    nperseg = int(extraction_config["nperseg"])                                     # 頻譜分析（像 STFT/Welch）時每段的長度
    max_points = int(extraction_config["max_points"])                               # 安全上限，避免單一檔案點數過多
    base_feature_version = str(extraction_config["base_feature_version"])           # 用特徵版本控制快取；若特徵計算邏輯改版，舊 cache 不應被誤用
    include_offband_summary = bool(extraction_config["include_offband_summary"])    # 決定要不要額外算 offband 摘要特徵
    
    # Disturbance 流程強制需要 offband
    if not include_offband_summary:
        raise ValueError("disturbance 規則比較需要 offband summary；請將 feature_extraction_config.json 的 include_offband_summary 設為 true")
    if "cusum_k" not in feature_config:
        raise ValueError("feature_config_micro.json 缺少 cusum_k")
    if "cusum_decay" not in feature_config:
        raise ValueError("feature_config_micro.json 缺少 cusum_decay")
    cusum_k = float(feature_config["cusum_k"])             # CUSUM 容忍值（0.5）
    cusum_decay = float(feature_config["cusum_decay"])     # CUSUM 衰減（0.97）
    
    # 檢查 CUSUM 範圍
    if not (cusum_k > 0):              # cusum_k：小偏移容忍區；小於此程度的波動不容易持續累積。
        raise ValueError(f"cusum_k 必須 > 0，目前 {cusum_k}")
    if not (0 < cusum_decay <= 1):     # cusum_decay：舊偏移淡化速度。
        raise ValueError(f"cusum_decay 必須在 (0, 1]，目前 {cusum_decay}")
    print(f"CUSUM 參數：k = {cusum_k}，decay = {cusum_decay}")

    # 掃描 disturbance 資料夾，整理出三份資料：
    # 1. file_specs：file_specs  ：交給特徵抽取的清單（segments 模式是一段一筆，舊模式是一支檔一筆）
    # 2. df_manifest：每個 TDMS 在整個 session 裡的時間位置
    # 3. df_events：每個 session 裡的干擾段落資訊
    # 4.file_records：一支檔一筆，給資料隔離與內容重複檢查用
    file_specs, df_manifest, df_events, file_records = collect_disturbance_sessions(
        root_dir, input_mode=input_mode, fs=fs,
    )
    print("TDMS 檔數量:", len(file_specs))
    print("manifest rows:", len(df_manifest))
    print("event rows:", len(df_events))
    
    
    # ============================================================
    # Disturbance 與固定 Baseline 資料隔離檢查
    # ============================================================
    # file_specs 包含本次真正會進入 Disturbance 特徵工程的所有 TDMS。
    # Disturbance 資料可能用於：模型負樣本訓練、Disturbance holdout、告警／Veto 候選規則比較，
    # 因此不得與建立 site_baseline.json 的正常 calibration TDMS 使用相同檔案、record_id 或內容。
    validate_baseline_source_isolation(
        baseline_payload=fixed_baseline,
        analysis_records=file_records,      # 逐支檔檢查（連續段裡的每一支都要查）
        context="Disturbance feature engineering",
        strict=STRICT_BASELINE_ISOLATION,
    )

    # 內容層級的重複與洩漏檢查：比檔案內容的 SHA-256，改過檔名的複本也抓得到
    check_content_duplicates(
        file_records,
        fixed_baseline.get("source_tdms_paths", []),
        strict=STRICT_BASELINE_ISOLATION,
    )
    

    # ========================================================
    # 1. 萃取特徵
    # ========================================================
    # 每個 TDMS 依 window_sec / hop_sec 切窗，在 core_bands（洩漏＋共振）和 aux_bands（offband）上算特徵
    features = load_or_compute_features_by_file(
        cache_dir=cache_dir,
        file_specs=file_specs,
        core_bands=band_config.final_main_bands,
        aux_bands=band_config.offband_bands,
        low_bands=[],
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

    df = pd.DataFrame(features).reset_index(drop=True)
    del features
    gc.collect()

    # 檢查有沒有 filename（同一 session 共用 record_id，只能靠檔名對回 manifest）
    if "filename" not in df.columns:
        raise ValueError("特徵表缺少 filename 欄位，無法對齊 manifest")
    
    if input_mode == "segments":
        # 連續段：窗的 t_start 是「段內秒數」，窗也可能跨兩支檔。
        # 依窗起點落在哪一支檔，對回那支檔的 manifest 列（filename_stem、capture_start_elapsed_sec…），
        # 再把 t_start 換成「那支檔內的秒數」，後面「全域時間 = capture_start + 檔內時間」的算法就完全不用改。
        # 例：段內 0.40 秒起的窗 → 落在第 1 支（offset 0.0），檔內 0.40 秒；它會一路延伸到第 2 支的 0.10 秒，這是正常的。
        file_offsets = (
            df_manifest[["record_id", "filename_stem", "offset_in_segment_sec"]]
            .sort_values("offset_in_segment_sec")
        )
        df = df.sort_values("t_start").reset_index(drop=True)
        df = pd.merge_asof(
            df, file_offsets,
            left_on="t_start", right_on="offset_in_segment_sec",
            by="record_id", direction="backward",
        )
        if df["filename_stem"].isna().any():
            raise ValueError("有窗對不到段內的任何一支檔（段的起點和 manifest 不一致）")
        # 浮點誤差：剛好落在檔案邊界的窗（例如 0.5 - 0.5 = 1e-17）四捨五入到奈秒
        df["t_start"] = (df["t_start"] - df["offset_in_segment_sec"]).round(9)
        df = df.drop(columns=["offset_in_segment_sec"])   # manifest 合併時會再帶回來
    else:
        df["filename_stem"] = df["filename"].map(normalize_filename_stem)

    # 檢查：每支 TDMS 都有算出窗（用檔名判斷，不能用 record_id）
    expected_stems = {Path(s["filepath"]).stem for s in file_records}    # 逐支檔檢查（segments 模式的 file_specs 是一段一筆）
    lost_stems = sorted(expected_stems - set(df["filename_stem"]))
    if lost_stems:
        raise ValueError(f"有 {len(lost_stems)} 支 TDMS 沒有算出任何窗：{lost_stems[:10]}")

    # 檢查：訊號品質（門檻和 baseline 程式相同）
    bad = pd.Series(False, index=df.index)
    for ch in ("ai0", "ai1"):
        rms = pd.to_numeric(df[f"{ch}_rms"], errors="coerce")
        bad |= ~np.isfinite(rms) | (rms < SIGNAL_RMS_MIN)
    if bad.any():
        if input_mode == "segments":
            # 連續段：只拿掉中間幾支檔會在段裡挖出空洞，後面的告警規則會把洞前後當成相鄰窗，所以整段排除
            bad_ids = sorted(df.loc[bad, "record_id"].unique())
            print(f"⚠️ 訊號 QC：{len(bad_ids)} 段有壞窗（共 {int(bad.sum())} 窗），整段排除：{bad_ids[:10]}")
            df = df[~df["record_id"].isin(bad_ids)].reset_index(drop=True)
        else:
            # 舊資料：一支檔只有 0.5 秒、session 內檔案之間本來就有間隙，所以排除「整支檔」，不排除整個 session
            bad_stems = sorted(df.loc[bad, "filename_stem"].unique())
            print(f"⚠️ 訊號 QC：{len(bad_stems)} 支 TDMS 有壞窗（共 {int(bad.sum())} 窗），整支排除：{bad_stems[:10]}")
            df = df[~df["filename_stem"].isin(bad_stems)].reset_index(drop=True)
        if df.empty:
            raise ValueError("訊號 QC 後沒有任何可用的 TDMS")

    # 檢查 raw band 欄位有沒有真的抽出來
    missing_raw = audit_band_feature_coverage(df, band_config)
    if missing_raw:
        raise ValueError(f"Disturbance 缺少原始 band 特徵：{missing_raw[:20]}")


    # ========================================================
    # 2. 把 manifest 合進 features
    # ========================================================
    # 抽取器會從檔名解析出 label；disturbance 的 label 以 meta 為準（全部是 0），後面貼標時會統一設定，所以先拿掉抽取器的版本，避免 merge 撞名
    df = df.drop(columns=["label"], errors="ignore")
    
    # 特徵表裡不應該已經有 manifest 欄位，否則 merge 會變成 _x / _y
    merge_keys = ["record_id", "filename_stem"]
    clash = [c for c in df_manifest.columns if c not in merge_keys and c in df.columns]
    if clash:
        raise ValueError(f"特徵表已經有這些欄位，merge 會撞名：{clash}")

    # 用 record_id + filename_stem 對回每支 TDMS；many_to_one：manifest 每支檔只能有一列
    df = df.merge(df_manifest, on=merge_keys, how="left", validate="many_to_one")
    # 檢查是否有對不上的manifest
    missing_manifest = df["capture_start_elapsed_sec"].isna()
    if missing_manifest.any():
        miss_stems = sorted(df.loc[missing_manifest, "filename_stem"].unique())
        raise ValueError(f"有 {int(missing_manifest.sum())} 窗對不到 manifest，例如：{miss_stems[:5]}")

    # 保留檔內局部時間
    df["t_start_local"] = df["t_start"].astype(float)
    # t_ref 一律用窗中心（和 transition 一致），不沿用抽取器可能給的其他定義
    df["t_ref_local"] = df["t_start_local"] + window_sec / 2.0

    # 換成 session 全域時間 = 這支檔在 session 的開始時間 + 檔內局部時間
    df["t_start"] = df["capture_start_elapsed_sec"].astype(float) + df["t_start_local"]
    df["t_ref"] = df["capture_start_elapsed_sec"].astype(float) + df["t_ref_local"]

    # 排序成 session 時間序列
    df = df.sort_values(["record_id", "t_start"]).reset_index(drop=True)

    # 檢查：同一 session 內不可有重複的全域時間（代表檔案時間重疊或 manifest 填錯）
    dup_time = df.duplicated(["record_id", "t_start"], keep=False)
    if dup_time.any():
        dup_sessions = sorted(df.loc[dup_time, "record_id"].unique())
        raise ValueError(f"有 {int(dup_time.sum())} 窗的全域時間重複，session：{dup_sessions[:5]}")


    # ========================================================
    # 3. 貼 disturbance 標註(disturbance_flag/eval_class)
    # ========================================================
    #    逐個 window 去看它有沒有落在干擾段落裡,然後新增欄位：disturbance_flag, disturbance_type, disturbance_strength, overlap_sec, overlap_ratio, eval_class
    df = attach_window_labels_from_disturbance_meta(
        df_windows=df,
        df_events=df_events,
        window_sec=window_sec,
    )

    
    # ========================================================
    # 4.offband 摘要特徵
    # ========================================================
    # 若 feature_extraction_config 啟用 offband 摘要，建立主頻帶、共振頻帶與 offband 的能量摘要及比值。
    if include_offband_summary:
        df = build_offband_summary_features(df, band_config)
        
        # 確認 offband 摘要與比值欄位都有成功建立
        missing_summary = audit_offband_summary_coverage(df)
        if missing_summary:
            raise ValueError(
                f"Disturbance 缺少 offband 摘要特徵：{missing_summary}"
            )
        
    rule_baseline_feature_names = [
        "offband_E_mean",
        "offband_E_max",
        "leak_focus_E_mean",
        "leak_focus_E_max",
        "resonance_E_mean",
        "resonance_E_max",
    ]
    
    # 模型與規則需要的 baseline 欄位取聯集
    baseline_apply_feature_names = list(dict.fromkeys(
        baseline_feature_names
        + rule_baseline_feature_names
    ))
    
    validate_feature_stage_requirements(
        df,
        baseline_apply_feature_names,
        stage_name=("Disturbance / fixed baseline 前置 raw"),
    )
    
    # ========================================================
    # 5. 建 rel_ / z_
    # ========================================================
    # 套用固定 baseline
    df = apply_fixed_normal_baseline(
        df=df,
        feature_cols=baseline_apply_feature_names,
        baseline_payload=fixed_baseline,
    )

    # rel_ / z_ 都使用固定 site_baseline 計算；baseline_end_sec 只當 metadata，不參與計算
    # 檢查完整 rel/z
    expected_baseline_derived_features = [
        derived_feature
        for raw_feature in baseline_apply_feature_names
        for derived_feature in (
            f"rel_{raw_feature}",
            f"z_{raw_feature}",
        )
    ]
    
    validate_feature_stage_requirements(
        df,
        expected_baseline_derived_features,
        stage_name=(
            "Disturbance / fixed baseline rel-z 輸出"
        ),
    )

    # ========================================================
    # 6. 建 CUSUM 累積統計量
    # ========================================================
    validate_feature_stage_requirements(
        df,
        cusum_input_feature_names,
        stage_name="Disturbance / CUSUM 輸入",
    )
    
    if cusum_input_feature_names:
        df = add_cusum_features(
            df=df,
            input_feature_cols=(
                cusum_input_feature_names
            ),
            k=cusum_k,
            decay=cusum_decay,
        )
    
    expected_cusum_output_features = [
        output_feature
        for input_feature in cusum_input_feature_names
        for output_feature in (
            f"cusum_pos_{input_feature}",
            f"cusum_neg_{input_feature}",
        )
    ]
    
    validate_feature_stage_requirements(
        df,
        expected_cusum_output_features,
        stage_name="Disturbance / CUSUM 輸出",
    )
            
    # ========================================================
    # 7. 建立 dynamic / rolling 時間結構特徵
    # ========================================================
    validate_feature_stage_requirements(
        df,
        temporal_base_feature_names,
        stage_name="Disturbance / temporal 輸入",
    )
    
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
    
    # 檢查 rel / z / cusum / temporal 衍生鏈有沒有漏
    validate_feature_stage_requirements(
        df,
        final_keep_features,
        stage_name=(
            "Disturbance / final selected features"
        ),
    )
    

    # ========================================================
    # 8. 驗證並輸出
    # ========================================================
    # 基礎欄位清單：metadata/標註欄位
    base_output_cols = [
        "filename",
        "filename_stem",
        "record_id",
        "session_id",
        "disturbance_group",
        "disturbance_id",
        "tdms_file",
        "t_start_local",
        "t_ref_local",
        "t_start",
        "t_ref",
        "capture_start_elapsed_sec",
        "capture_end_elapsed_sec",
        "capture_mid_elapsed_sec",
        "baseline_end_sec",
        "label",
        "disturbance_flag",
        "disturbance_type",
        "disturbance_strength",
        "overlap_sec",
        "overlap_ratio",
        "eval_class",
        "leak_status",
        "disturbance_session_manifest",
        "disturbance_event_csv",
    ]

    # ============================================================
    # 收集所有 Offband／Veto 規則候選欄位
    # ============================================================
    offband_band_names = {str(band["name"]) for band in band_config.offband_bands}
    
    # 最終模型特徵不得包含任何 Offband／Veto 專用欄位。
    rule_only_model_features = [
        feature_name for feature_name in final_keep_features
        if ("offband" in feature_name.lower() or any(f"{band_name}_" in feature_name for band_name in offband_band_names)
        )
    ]
    
    if rule_only_model_features:
        raise ValueError(f"final_keep_features 包含 Offband Rule-only 特徵，Offband 不可進入模型：{rule_only_model_features}")
    
    extra_veto_cols = [
        column
        for column in df.columns
        if ("offband" in column.lower() or any(f"{band_name}_" in column for band_name in offband_band_names)
        )]
    
    extra_veto_cols += [
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
    
        "rel_offband_E_mean",
        "z_offband_E_mean",
        "rel_offband_E_max",
        "z_offband_E_max",
    
        "rel_leak_focus_E_mean",
        "z_leak_focus_E_mean",
        "rel_leak_focus_E_max",
        "z_leak_focus_E_max",
    
        "rel_resonance_E_mean",
        "z_resonance_E_mean",
        "rel_resonance_E_max",
        "z_resonance_E_max",
    ]
    
    missing_rule_cols = [c for c in dict.fromkeys(extra_veto_cols) if c not in df.columns]
    if missing_rule_cols:
        print(f"ℹ️ 以下規則欄位這次沒有建立，不會輸出：{missing_rule_cols}")
    
    # 去重複
    extra_veto_cols = [c for c in dict.fromkeys(extra_veto_cols) if c in df.columns]
    # 確保輸出資料夾存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 第一層: full 版(全部欄位)
    full_output_csv = output_csv.with_name(output_csv.stem + "_full.csv")
    df.to_csv(full_output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ disturbance full feature csv 已輸出: {full_output_csv}")

    # 第二層:model 版(只留模型訓練/合併需要的欄位)
    model_requested_cols = list(dict.fromkeys(base_output_cols + final_keep_features))
    
    missing_model_output_cols = [
        column
        for column in model_requested_cols
        if column not in df.columns
    ]
    
    if missing_model_output_cols:
        raise ValueError(f"Disturbance model CSV 缺少必要欄位：{missing_model_output_cols}")
    
    model_output_cols = model_requested_cols
    df_model = df[model_output_cols].copy()
    
    # 驗證是否跟STEP2輸出的特徵一樣
    model_feature_cols_only = [c for c in df_model.columns if c not in base_output_cols]
    missing_model = [c for c in final_keep_features if c not in model_feature_cols_only]
    extra_model = [c for c in model_feature_cols_only if c not in final_keep_features]
    
    print("=== model csv 檢查 ===")
    print("selected features 預期數量 =", len(final_keep_features))
    print("selected features 實際輸出數量 =", len(model_feature_cols_only))
    print("missing model features =", missing_model[:20])
    print("extra model features =", extra_model[:20])

    if missing_model:
        raise ValueError(f"model csv 缺少 selected features: {missing_model[:20]}")
    elif model_feature_cols_only != final_keep_features:
        raise ValueError(f"model csv 的模型特徵順序和 Step2 不一致：extra={extra_model[:10]}")
    else:
        print("✅ model csv 與 Step2 selected_feature_names 完全一致")
        
    # 檢查 Step2 selected features 有沒有都存在
    audit_selected_feature_alignment(df, feature_config)
    
    # 輸出
    model_output_csv = output_csv.with_name(output_csv.stem + "_model.csv")
    df_model.to_csv(model_output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ disturbance model feature csv 已輸出: {model_output_csv}")

    # 第三層:rule 版(保留模型特徵 + offband 規則欄位)
    rule_requested_cols = (base_output_cols + final_keep_features + extra_veto_cols)
    
    rule_output_cols = list(dict.fromkeys(
        c
        for c in rule_requested_cols
        if c in df.columns
    ))
    df_rule = df[rule_output_cols].copy()
    
    required_rule_cols = [
        "offband_E_mean", "offband_E_max",
        "leak_focus_E_mean", "leak_focus_E_max",
        "resonance_E_mean", "resonance_E_max",
        "offband_to_leak_focus_E_ratio", "offbandmax_to_leak_focusmax_E_ratio",
        "offband_to_resonance_E_ratio", "offbandmax_to_resonancemax_E_ratio",
        "z_offband_E_mean","z_offband_E_max",
    ]
    
    # 檢查 Rule CSV 所需欄位是否完整。
    missing_required_rule_cols = (
        audit_rule_feature_alignment(df_rule, required_rule_cols,)
    )

    if missing_required_rule_cols:
        raise ValueError(f"disturbance rule csv 缺少必要規則欄位：{missing_required_rule_cols}")

    rule_model_feature_cols = [c for c in df_rule.columns if c in final_keep_features]
    rule_extra_cols = [
        c for c in df_rule.columns
        if c not in base_output_cols and c not in final_keep_features
    ]
    missing_rule = [c for c in final_keep_features if c not in rule_model_feature_cols]

    print("=== rule csv 檢查 ===")
    print("selected features 預期數量 =", len(final_keep_features))
    print("selected features 實際輸出數量 =", len(rule_model_feature_cols))
    print("missing rule features =", missing_rule[:20])
    print("extra veto/offband features =", rule_extra_cols[:20])

    if missing_rule:
        raise ValueError(f"rule csv 缺少 selected features: {missing_rule[:20]}")
    elif rule_model_feature_cols != final_keep_features:
        raise ValueError("rule csv 的模型特徵順序和 Step2 不一致")
    else:
        print("✅ rule csv 內的模型特徵與 Step2 selected_feature_names 完全一致")

    df_rule.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ disturbance rule feature csv 已輸出: {output_csv}")

    print("model shape =", df_model.shape)
    print("rule shape =", df_rule.shape)
    print(df_rule["eval_class"].value_counts(dropna=False))
    print(df_rule.groupby("disturbance_type")["disturbance_flag"].sum())

    return df_rule


if __name__ == "__main__":
    # Step2(純正常/純洩漏資料的頻帶探索流程)跑完之後輸出的正式頻帶設定檔,裡面存的是leak_focused_bands(洩漏關鍵頻帶)、resonance_bands(共振頻帶)、offband_bands(參照頻帶)
    BAND_CONFIG_JSON = "/Users/paul/Desktop/Final/test_result/band_config_step2.json"
    
    # Step2 輸出的特徵層面的設定——selected_feature_names(最終選定要給模型用的特徵)
    FEATURE_CONFIG_JSON = "/Users/paul/Desktop/Final/test_result/feature_config_micro.json"
    
    FEATURE_EXTRACTION_CONFIG = ("/Users/paul/Desktop/Final/test_result/feature_extraction_config.json")

    BASELINE_JSON = ("/Users/paul/Desktop/Final/test_result/site_baseline.json")

    # 執行 Disturbance window-level 特徵工程。
    #
    # 本程式只輸出供模型推論與抗干擾規則比較，使用的特徵 CSV，本身不載入模型、不執行預測。
    build_disturbance_inference_feature_csv(
        root_dir="/Users/paul/Desktop/Final/disturbance_test_0624_0702_0707",
        band_config_json=BAND_CONFIG_JSON,
        feature_extraction_config_path=(FEATURE_EXTRACTION_CONFIG),
        baseline_json=BASELINE_JSON,
        feature_config_json=FEATURE_CONFIG_JSON,
        output_csv="/Users/paul/Desktop/Final/disturbance_feature_engineering/inference_features_disturbance.csv",
        cache_dir="/Users/paul/Desktop/Final/disturbance_feature_engineering/feature_cache",
        split_name="disturbance_final",
        n_jobs=6,
        require_final_baseline=False,
        input_mode="per_file_legacy",   # 舊資料：每支檔各自切窗；v22 連續錄製改成 "segments"
    )  
    
    



