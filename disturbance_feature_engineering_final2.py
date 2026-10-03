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
from dataclasses import dataclass
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
)


# ============================================================
# Baseline 資料隔離設定
# ============================================================
# Disturbance 不應與建立固定 Baseline 的正常 TDMS 重複。
# True：發現相同 TDMS 路徑或 record_id 時直接停止。
# False：只顯示警告，僅供舊資料流程測試。
STRICT_BASELINE_ISOLATION = True


# ============================================================
# 讀取並驗證正式 Band Config
# ============================================================
def load_band_config(band_config_json):
    """
    讀取正式 band_config_step2.json。

    Disturbance 正式特徵工程只接受：
    1. schema_name = band_config
    2. schema_version = 2
    3. schema_status = final
    4. leak_focused_bands 不可為空
    5. offband_bands 不可為空
    """
    # 轉成 Path 物件
    band_config_json = Path(band_config_json)
    # 讀取 band 資料
    payload = json.loads(band_config_json.read_text(encoding="utf-8"))

    # 檢查
    if payload.get("schema_name") != "band_config":
        raise ValueError(f"band_config 的 schema_name 必須是 band_config，目前檔案：{band_config_json}")

    if payload.get("schema_version") != 2:
        raise ValueError(f"band_config 的 schema_version 必須是 2，目前值：{payload.get('schema_version')}")

    if payload.get("schema_status") != "final":
        raise ValueError(f"Disturbance 正式特徵工程只能讀取 schema_status=final 的 band config，目前值：{payload.get('schema_status')}")
    
    # 四種頻段：如果 JSON 裡有這個欄位，就讀它, 如果沒有，就先給空清單，避免直接報錯
    leak_focused_bands = payload.get("leak_focused_bands",[],)
    resonance_bands = payload.get("resonance_bands",[],)
    offband_bands = payload.get("offband_bands",[],)
    temporal_band_names = payload.get("temporal_band_names",[],)
    
    if not leak_focused_bands:
        raise ValueError("正式 band config 的 leak_focused_bands 不可為空")

    if not offband_bands:
        raise ValueError("Disturbance 抗干擾規則需要 Offband，正式 band config 的 offband_bands 不可為空")
        
    return BandConfig(
            leak_focused_bands=leak_focused_bands,
            resonance_bands=resonance_bands,
            offband_bands=offband_bands,
            temporal_band_names=set(temporal_band_names),
        )


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
    return Path(str(x)).stem


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
def collect_disturbance_sessions(root_dir: Path):
    file_specs = []       # 後面萃取特徵時要處理哪些 TDMS 檔
    manifest_rows = []    # 每個 TDMS 對應的時間資訊與 metadata
    event_rows = []       # 每個干擾事件的標註資訊

    root_dir = Path(root_dir)    # root_dir 轉成 Path

    # 第一層迴圈：掃干擾種類資料夾 -> adjacent_vibration / cable_pull / tap_pipe / touch_pipe
    for disturbance_group_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        disturbance_group = disturbance_group_dir.name
        
        # 第二層迴圈：掃每次收集的子資料夾 -> disturbance01 / disturbance02 / disturbance03
        for disturbance_dir in sorted([p for p in disturbance_group_dir.iterdir() if p.is_dir()]):
            disturbance_id = disturbance_dir.name
            
            # 找這個 disturbance session 對應的兩份標註檔：
            # 1. dual_session_manifest：每筆 TDMS 在整個 session 裡的時間位置
            # 2. disturbance_events：session 裡每段干擾事件的開始/結束時間
            manifest_files = sorted(                                            # 記每個 TDMS 的時間區間
                list(disturbance_dir.glob("dual_session_manifest_*.csv")) +
                list(disturbance_dir.glob("transition_dual_manifest_*.csv"))
            )
            
            event_files = sorted(                                               # 記每次干擾開始/結束時間
                list(disturbance_dir.glob("disturbance_events_*.csv")) +
                list(disturbance_dir.glob("transition_dual_disturbance_events_*.csv"))
            )

            # 如果找不到，就跳過
            if not manifest_files:
                print(f"[skip] 找不到 dual_session_manifest: {disturbance_dir}")
                continue
            if not event_files:
                print(f"[skip] 找不到 disturbance_events: {disturbance_dir}")
                continue

            # 取第一個 manifest 和 event 檔(通常一個 session 裡應該只有一份對應的 manifest 跟 event csv)
            manifest_path = manifest_files[0]
            event_path = event_files[0]

            df_manifest = pd.read_csv(manifest_path)
            df_events = pd.read_csv(event_path)

            # 如果 manifest 是空的，就跳過
            if df_manifest.empty:
                print(f"[skip] manifest 為空: {manifest_path}")
                continue
            # 取得 session_id
            session_id = str(df_manifest["session_id"].iloc[0])

            # 決定 baseline_end_sec
            # 如果有干擾事件，就把第一個干擾開始時間當成 baseline 結束點
            # 如果沒有干擾事件，就退而求其次，把整段資料的最後時間當 baseline_end_sec
            if not df_events.empty:
                baseline_end_sec = float(df_events["start_elapsed_sec"].min())
            else:
                baseline_end_sec = float(df_manifest["capture_end_elapsed_sec"].max())

            # 收集 events 資料原本事件表只有事件本身資訊，這裡把它補成「知道自己屬於哪個 session、哪種干擾、哪一次收集」。
            df_events = df_events.copy()
            df_events["session_id"] = session_id
            df_events["disturbance_group"] = disturbance_group
            df_events["disturbance_id"] = disturbance_id
            event_rows.append(df_events)

            # 根據 dual_session_manifest 的每一列，整理每筆 TDMS 的 metadata
            # 先決定 manifest 裡哪個欄位存的是 TDMS 檔名
            file_col = None
            for candidate in ["dual_file", "merged_file", "tdms_file", "filename"]:
                if candidate in df_manifest.columns:
                    file_col = candidate
                    break
            
            if file_col is None:
                raise ValueError(f"manifest 缺少可用的檔名欄位，現有欄位: {df_manifest.columns.tolist()}")
            
            # 根據 dual_session_manifest 的每一列，整理每筆 TDMS 的 metadata
            for row in df_manifest.itertuples(index=False):
                dual_file_name = PureWindowsPath(str(getattr(row, file_col))).name
                tdms_path = disturbance_dir / dual_file_name
                            
                # 如果組出來的檔案不存在，就再試一次備案
                if not tdms_path.exists():
                    alt = disturbance_dir / f"{Path(dual_file_name).stem}.tdms"
                    if alt.exists():
                        tdms_path = alt
                    else:
                        print(f"[skip] 找不到 TDMS: {dual_file_name} @ {disturbance_dir}")
                        continue

                # 建立 file_specs
                file_specs.append({
                    "filepath": str(tdms_path),
                    # 整個 disturbance session 共用同一個 record_id，讓 CUSUM 與 temporal 特徵沿完整 session 時間軸計算。
                    # rel/z 使用固定 site baseline，不依賴 session 前段。
                    "record_id": session_id,
                })

                # 建立 manifest_rows:把每筆 TDMS 需要的 metadata 整理成一列表格
                manifest_rows.append({
                    "session_id": session_id,                                              # 這筆 TDMS 屬於哪一個收集 session
                    "record_id": session_id,                                               # 同一個 session 內所有 TDMS 共用同一個 record_id，後面 rel_/z_/cusum 才會沿同一條時間序列建立
                    "disturbance_group": disturbance_group,                                # 這筆 TDMS 屬於哪一種干擾大類
                    "disturbance_id": disturbance_id,                                      # 這筆 TDMS 屬於哪一次干擾 session
                    "filename_stem": tdms_path.stem,                                       # TDMS 檔名去掉副檔名後的主體
                    "tdms_file": str(tdms_path),                                           # 這筆 TDMS 的完整路徑字串
                    "capture_start_elapsed_sec": float(row.capture_start_elapsed_sec),     # 這筆 TDMS 在整個 session 時間軸上的開始時間
                    "capture_end_elapsed_sec": float(row.capture_end_elapsed_sec),         # 這筆 TDMS 在整個 session 時間軸上的結束時間
                    "capture_mid_elapsed_sec": float(row.capture_mid_elapsed_sec),         # 這筆 TDMS 在整個 session 時間軸上的中間時間
                    "baseline_end_sec": baseline_end_sec,                                  # baseline 區段到哪一秒為止(baseline以第一段為主)
                    "label": 0,                                                            # 這筆資料的真實 leak label
                    "leak_status": "leak0",                                                # 這筆資料的洩漏狀態是正常、無洩漏
                    "disturbance_session_manifest": str(manifest_path),                    # 這筆 TDMS 是從哪一份 dual_session_manifest.csv 來的
                    "disturbance_event_csv": str(event_path),                              # 這筆 TDMS 對應的是哪一份 disturbance_events.csv
                })
    # 最後檢查是不是完全沒找到任何 TDMS
    if not file_specs:
        raise ValueError("找不到任何可處理的 disturbance TDMS")

    # 最後把資料整理成 DataFrame
    df_manifest_all = pd.DataFrame(manifest_rows)    # 每筆 TDMS 一列
    df_events_all = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()    # 每個干擾事件一列

    # file_specs:給特徵萃取函式用,告訴它要處理哪些 TDMS 檔
    # df_manifest_all:給後面 merge 用,把每筆 TDMS 的 session 時間資訊帶進 features
    # df_events_all:給後面標註用,判斷每個 window 有沒有落在干擾區間內
    return file_specs, df_manifest_all, df_events_all


# ============================================================
# 依據 window 與干擾事件時間區間的重疊關係，把每個 window 標成 clean_normal 或 disturbance_negative
# 這裡假設：
# - features 的 t_start / t_ref 是「檔內局部時間」
# - manifest 的 capture_*_elapsed_sec 是「整個 session 的全域時間」
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

    disturbance_flag = []
    disturbance_type = []
    disturbance_strength = []
    overlap_sec_list = []
    overlap_ratio_list = []
    eval_class = []

    # 先把 event 依 session 分組:把所有干擾事件依 session_id 分好
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
            for ev in session_events.itertuples(index=False):
                # 計算這個 window 和某個事件重疊多少秒
                ov = overlap_seconds(
                    w_start, w_end,
                    float(ev.start_elapsed_sec), float(ev.end_elapsed_sec),
                )
                # 只保留重疊最多的那個事件,這個 window 會被歸到「與它重疊最多的那個干擾事件」
                if ov > best_overlap:
                    best_overlap = ov
                    best_type = str(ev.disturbance_type)
                    best_strength = str(ev.disturbance_strength)

        # 判斷這個 window 有沒有被視為干擾窗,只要有任何重疊，就算干擾窗
        flag = 1 if best_overlap > 0 else 0
        ratio = best_overlap / window_sec if window_sec > 0 else 0.0     # 算重疊比例

        # 把這個 window 的結果存進 list
        disturbance_flag.append(flag)
        disturbance_type.append(best_type if flag else "")
        disturbance_strength.append(best_strength if flag else "")
        overlap_sec_list.append(best_overlap)
        overlap_ratio_list.append(ratio)
        eval_class.append("disturbance_negative" if flag else "clean_normal")    # 產生最終分類:disturbance_negative -> 有干擾，但不是 leak，仍屬負樣本, clean_normal -> 完全正常、沒有干擾的負樣本
    
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
# 從整合好的資料中挑出 raw 特徵
# ============================================================
def select_all_raw_feature_columns(df):    # 整合好的特徵 DataFrame，裡面同時混著 metadata 欄位、原始特徵欄位、以及後面衍生出來的 rel_/z_/cusum_ 等欄位）
    # 建立要排除的欄位    
    exclude_cols = {
        # 識別欄位
        "filename",
        "filename_stem",
        "record_id",
        "session_id",
        "disturbance_group",
        "disturbance_id",
        "tdms_file",
        # 時間欄位
        "t_start_local",
        "t_ref_local",
        "t_start",
        "t_ref",
        "capture_start_elapsed_sec",
        "capture_end_elapsed_sec",
        "capture_mid_elapsed_sec",
        "baseline_end_sec",
        # 標註/標籤欄位
        "label",
        "disturbance_flag",
        "disturbance_type",
        "disturbance_strength",
        "overlap_sec",
        "overlap_ratio",
        "eval_class",
        "leak_status",
        # manifest / events csv 路徑
        "disturbance_session_manifest",
        "disturbance_event_csv",
    }

    # 列出所有「衍生特徵」的欄位名稱前綴
    derived_prefixes = (
        "rel_",        # 相對 baseline 的偏移量
        "z_",          # 標準化後的偏移量
        "cusum_",      # 累積和
        "delta_",      # 差分
        "local_",      # 局部 rolling 相關
        "rolling_"     # rolling 統計量
    )

    # 逐一掃過 df.columns 所有欄位名稱, 同時要滿足三個條件才會被留下來
    raw_cols = [
        c for c in df.columns
        if (
            c not in exclude_cols      # 這個欄位不在剛剛定義的排除單裡
            and pd.api.types.is_numeric_dtype(df[c])    # 資料型態是數值型
            and not c.startswith(derived_prefixes)    # 欄位名稱不是以 rel_、z_、cusum_、delta_、local_、rolling_ 這六種前綴開頭的(排除掉所有衍生特徵)
        )
    ]
    return raw_cols


# ============================================================
# 主流程
# ============================================================
def build_disturbance_inference_feature_csv(
    root_dir,                         # 干擾資料的根目錄
    band_config_json,                 # Step2 產生的頻段設定檔
    feature_extraction_config_path,
    baseline_json,
    feature_config_json,              # Step2 產生的特徵設定檔
    output_csv,                       # 輸出的推論特徵表路徑
    cache_dir,                        # 特徵計算的快取資料夾路徑
    split_name="disturbance_infer",   # 這批資料的識別名稱
    n_jobs=2,                         # 平行處理的工作數
    require_final_baseline=False,
):
    '''把 disturbance 資料夾裡的很多 TDMS + 干擾標註 csv，整理成一份可以拿去模型推論的特徵表'''
    # 把路徑轉成 Path
    root_dir = Path(root_dir)
    band_config_json = Path(band_config_json)
    feature_config_json = Path(feature_config_json)
    output_csv = Path(output_csv)
    cache_dir = Path(cache_dir)
    feature_extraction_config_path = Path(feature_extraction_config_path)
    baseline_json = Path(baseline_json)

    # 讀 Step2 輸出的設定檔
    band_config = load_band_config(band_config_json)     # band_config 是一個 BandConfig dataclass，裡面裝著洩漏頻帶、共振頻帶、offband 頻帶
    extraction_config = (load_feature_extraction_config(feature_extraction_config_path))
    feature_config = load_feature_config(feature_config_json)
    
    fixed_baseline = load_fixed_normal_baseline(
        baseline_json,
        band_config_path=band_config_json,
        feature_extraction_config_path=(
            feature_extraction_config_path
        ),
        require_final=require_final_baseline,
    )
    
    # 驗證 Step 2 契約
    current_band_hash = calculate_file_sha256(band_config_json)
    current_extraction_hash = calculate_file_sha256(feature_extraction_config_path)
    
    if (feature_config.get("band_config_sha256") != current_band_hash):
        raise ValueError("feature_config_micro.json 與目前 band_config_step2.json 不一致")
    
    if (feature_config.get("feature_extraction_config_sha256") != current_extraction_hash):
        raise ValueError("feature_config_micro.json 與目前 feature_extraction_config 不一致")
    
    if (feature_config.get("baseline_config_fingerprint") != fixed_baseline.get("config_fingerprint")):
        raise ValueError("feature_config_micro.json 與目前 固定 baseline 不一致")
    
    # 把 band config 的 temporal band 名稱,同步給 shared_feature_utils 的 raw feature extractor。
    configure_temporal_band_names(band_config.temporal_band_names)
    
    
    # 讀取 execution plan
    selected_feature_names = feature_config.get("selected_feature_names", [],)
    
    if not selected_feature_names:
        raise ValueError("feature_config_micro.json 缺少 selected_feature_names")
    
    final_keep_features = list(selected_feature_names)
    
    execution_plan = feature_config.get("feature_execution_plan")
    
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
    
    print("固定 baseline 模型特徵數量:",len(baseline_feature_names),)
    
    print("CUSUM 輸入特徵數量:", len(cusum_input_feature_names),)
    
    print("temporal base 特徵數量:", len(temporal_base_feature_names),)
    
    print("final selected features 數量:", len(final_keep_features),)
    
    print("需要的 final selected features 數量:", len(final_keep_features))
    
    # 從 feature_config 裡取出一串跟訊號處理有關的超參數
    window_sec = float(extraction_config["window_sec"])     # 決定每個 window 多長
    hop_sec = float(extraction_config["hop_sec"])          # window 之間跳多少
    fs = float(extraction_config["fs"])               # 取樣率
    nperseg = int(extraction_config["nperseg"])          # 頻譜分析（像 STFT/Welch）時每段的長度
    max_points = int(extraction_config["max_points"])  # 安全上限，避免單一檔案點數過多
    base_feature_version = str(extraction_config["base_feature_version"])
    include_offband_summary = bool(extraction_config["include_offband_summary"])    # 決定要不要額外算 offband 摘要特徵
    
    if not include_offband_summary:
        raise ValueError("disturbance 規則比較需要 offband summary；請將 feature_extraction_config.json 的 include_offband_summary 設為 true")
    if "cusum_k" not in feature_config:
        raise ValueError("feature_config_micro.json 缺少 cusum_k")
    if "cusum_decay" not in feature_config:
        raise ValueError("feature_config_micro.json 缺少 cusum_decay")
    cusum_k = float(feature_config["cusum_k"])             # CUSUM 累積
    cusum_decay = float(feature_config["cusum_decay"])     # 演算法的參數



    # 掃描 disturbance 資料夾，整理出三份資料：
    # 1. file_specs：後面萃特徵要處理哪些 TDMS 檔
    # 2. df_manifest：每個 TDMS 在整個 session 裡的時間位置
    # 3. df_events：每個 session 裡的干擾段落資訊
    file_specs, df_manifest, df_events = collect_disturbance_sessions(root_dir)

    print("TDMS 檔數量:", len(file_specs))
    print("manifest rows:", len(df_manifest))
    print("event rows:", len(df_events))
    
    
    # ============================================================
    # Disturbance 與固定 Baseline 資料隔離檢查
    # ============================================================
    # file_specs 包含本次真正會進入 Disturbance特徵工程的所有 TDMS。
    # Disturbance 資料可能用於：
    # 1. 模型負樣本訓練
    # 2. Disturbance holdout
    # 3. 告警／Veto 候選規則比較
    # 因此不得與建立 site_baseline.json 的正常 calibration TDMS 使用相同檔案或 record_id。
    validate_baseline_source_isolation(
        baseline_payload=fixed_baseline,
        analysis_records=file_specs,
        context="Disturbance feature engineering",
        strict=STRICT_BASELINE_ISOLATION,
    )
    

    # ========================================================
    # 1. 萃取特徵
    # ========================================================
    #    每個 TDMS 檔案，依 window_sec/hop_sec 切成很多小段，對每段訊號在 core_bands（洩漏頻帶+共振頻帶合併後的正式主頻帶）跟 aux_bands（offband 參照頻帶）上算能量之類的特徵
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
    
    # 轉成 DataFrame 格式
    df = pd.DataFrame(features).reset_index(drop=True)
    del features
    gc.collect()
    
    # 檢查 raw band 欄位有沒有真的抽出來
    audit_band_feature_coverage(df, band_config)

    # 檢查有沒有 filename
    if "filename" not in df.columns:
        raise ValueError("特徵表缺少 filename 欄位，無法對齊 manifest")

    # ========================================================
    # 2. 把 manifest 合進 features
    # ========================================================
    #    先把特徵表裡的檔名去副檔名，變成 filename_stem
    df["filename_stem"] = df["filename"].map(normalize_filename_stem)

    #    用 record_id + filename_stem 去 merge:特徵表裡每個 window,對應回它原本屬於哪個 TDMS
    df = df.merge(df_manifest, on=["record_id", "filename_stem"], how="left", validate="many_to_one",)

    # 檢查是不是有些特徵列對不到 manifest:如果有些 window 根本找不到對應的 manifest，那後面全域時間就無法建立，所以直接報錯
    missing_manifest = df["capture_start_elapsed_sec"].isna().sum()
    if missing_manifest > 0:
        raise ValueError(f"有 {missing_manifest} 列 features 無法對齊 dual_session_manifest")

    # 保留局部時間
    df["t_start_local"] = df["t_start"].astype(float)
    
    if "t_ref" not in df.columns:
        df["t_ref"] = df["t_start"].astype(float) + window_sec / 2.0
    
    df["t_ref_local"] = df["t_ref"].astype(float)

    # 把局部時間改成 session 的全域時間:原本 window 的 t_start 是在 TDMS 檔內部的相對時間, 現在把它加上這個 TDMS 在整個 session 的開始時間, 於是就變成整個 session 的全域時間
    df["t_start"] = df["capture_start_elapsed_sec"].astype(float) + df["t_start_local"]
    df["t_ref"] = df["capture_start_elapsed_sec"].astype(float) + df["t_ref_local"]

    # 排序成 session 時間序列
    df = df.sort_values(["record_id", "t_start"]).reset_index(drop=True)

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
    # 如果 feature_config 裡設定要算 offband 摘要（預設是 True），就呼叫外部模組的 build_offband_summary_features，用洩漏頻帶、共振頻帶、offband 頻帶三組能量互相比較，算出像 offband_E_mean、offband_to_leak_focus_E_ratio 這種摘要指標，用來判斷「是不是很多頻段一起升高（比較像寬頻機械干擾）」還是「只有洩漏頻帶升高（比較像真的洩漏）」
    if include_offband_summary:
        df = build_offband_summary_features(df, band_config)
        
        # 檢查 offband 摘要欄位有沒有建出來
        audit_offband_summary_coverage(df)
        
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
    
    # 驗證固定 baseline 的 rel_ / z_ 輸出
    all_raw_feature_cols = select_all_raw_feature_columns(df)
    
    summary_base_cols = [
        c for c in [
            "offband_E_mean",
            "offband_E_max",
            "leak_focus_E_mean",
            "leak_focus_E_max",
            "resonance_E_mean",
            "resonance_E_max",
        ]
        if c in df.columns
    ]
    
    for c in summary_base_cols:
        if c not in all_raw_feature_cols:
            all_raw_feature_cols.append(c)
    
    print("正式 raw feature 數量:", len(all_raw_feature_cols))

    # apply_fixed_normal_baseline() 已經使用 site_baseline.json
    # 建立完成 rel_* 與 z_*。
    #
    # baseline_end_sec 只保留作為事件前正常區間的
    # QC／分析邊界，不參與 rel/z baseline 計算。
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
        print("⚠️ model csv 的模型特徵順序和 Step2 selected_feature_names 不完全一致")
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
        print("⚠️ rule csv 的模型特徵順序和 Step2 selected_feature_names 不完全一致")
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
        n_jobs=2,
        require_final_baseline=False,
    )  
    
    
    
'''
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
'''



