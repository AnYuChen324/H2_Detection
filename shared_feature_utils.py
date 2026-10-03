#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:19:21 2026

@author: paul
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
import hashlib
from functools import lru_cache



# ======================================
# 全域常數/頻段設定
# ======================================
NPERSEG  = 2048          # Welch PSD 每段長度
FREQ_MIN = 5000          # 找 peak 或做頻譜分析時，只看 5 kHz 以上
FS       = 1_000_000     # 預設採樣率 1MHz

# 設定 DC 前處理
REMOVE_DC = True

# 先初始化為空集合，再根據本次資料實際收斂出的 selected_leak_bands 動態指定
TEMPORAL_BAND_NAMES = set()


# ======================================
# 正式 band config 的 temporal band 名稱同步
# ======================================
def configure_temporal_band_names(
    temporal_band_names,
):
    """
    將正式 band config 的 temporal band 名稱，同步到基礎 window 特徵提取器使用的全域設定。
    """
    # 清理輸入：把每個名稱轉成字串，並過濾掉空字串，避免空白名稱被加進去造成後續特徵提取出錯
    normalized_names = {
        str(name) for name in temporal_band_names if str(name).strip()
    }

    # 更新全域變數:先清空再更新，確保不會殘留舊的頻段名稱
    TEMPORAL_BAND_NAMES.clear()
    TEMPORAL_BAND_NAMES.update(normalized_names)

    print("✅ temporal raw band 設定完成：", len(TEMPORAL_BAND_NAMES),)
    return set(TEMPORAL_BAND_NAMES)     # 回傳目前的集合


# ==========================
# 合併關鍵頻段 + 共振頻段
# ==========================
def _bands_overlap(b1, b2):
    return max(b1["f_low"], b2["f_low"]) < min(b1["f_high"], b2["f_high"])


# ==========================
# 合併關鍵頻段 + 共振頻段
# ==========================
def merge_band_groups(*groups):
    """
    把多組頻段清單合併成一個，有優先順序概念：第一組最高優先，完整保留；後面的組別如果跟前面的頻段重疊就跳過

    規則：
    1. 同名頻帶只保留第一次。
    2. 第一組是最高優先群組，完整保留，允許第一組內的頻帶彼此重疊。
    3. 第二組之後的頻帶，如果和較高優先群組已保留的頻帶重疊，才跳過。

    正式呼叫慣例：
        merge_band_groups(
            leak_focused_bands,
            resonance_bands,
        )

    因此：
        - leak-focused bands 全部保留
        - resonance 與 leak-focused 重疊時才排除
    """
    # 初始化
    merged = []           # 最後合併後的結果
    seen_names = set()    # 記錄哪些 band name 已經出現過，避免重複

    # 逐組 band 去處理
    for group_index, group in enumerate(groups):
        # 每一組裡面再逐段 band 去看
        for band in group:
            name = str(band["name"])    # 用 band 的 name 當作唯一識別
            # 如果這個 band name 已經出現過，就跳過，不要重複加入
            if name in seen_names:
                continue

            # 重疊判斷（只對第二組以後）
            # 第一組（group_index == 0）完全不做重疊檢查，全部保留，包含組內互相重疊的頻段。
            # 第二組以後才做：如果這個頻段跟已經加入的任何頻段有重疊（呼叫 _bands_overlap），就跳過不加。
            if group_index > 0:
                overlaps_higher_priority = any(_bands_overlap(band, kept) for kept in merged)

                if overlaps_higher_priority:
                    continue
            # 加入結果
            seen_names.add(name)
            merged.append(dict(band))

    return merged


# ==========================
# 自動產生候選頻帶
# ==========================
def generate_sliding_bands(f_min=5_000,        # 從 5 kHz 開始掃
                           f_max=500_000,      # 最多掃到 500 kHz
                           band_width=10_000,  # 每一段 band 寬度 10 kHz
                           stride=5_000):      # 每次往前滑 5 kHz
    # 儲存所有候選頻帶
    bands = []
    cur = f_min    # 目前掃描到的起始頻率
    idx = 1        # 候選頻帶編號，從 1 開始
    
    # 只要目前這段 band 還沒超出上限，就持續往前滑動
    while cur + band_width <= f_max:
        # 把這一段 band 用字典的形式加到 bands 裡
        bands.append({
            # scan：代表這是 sliding scan 產生的候選 band
            # idx：band 編號
            # 5、15k：代表這段 band 的頻率範圍是 5~15 kHz
            "name": f"scan_{idx:03d}_{int(cur/1000)}_{int((cur+band_width)/1000)}k", 
            "f_low": int(cur),     # band 的下界頻率
            "f_high": int(cur + band_width),    # band 的上界頻率
        })
        cur += stride    # 把起點往前滑動 stride
        idx += 1     # band 編號加 1
    return bands


# ============================================================
# 驗證固定 Basline資料與分析分析資料是否隔離
# ============================================================
def validate_baseline_source_isolation(
    baseline_payload,    # discovery_baseline.json 或 site_baseline.json 讀取後的完整內容。
    analysis_records,    # 本次分析使用的 record 清單。每一 record 至少需要包含包含：filepath, record_id
    *,
    context,             # 顯示在錯誤或警告訊息中的流程名稱，例如：Step2a dev / Step2b dev/holdout / Transition / Disturbance
    strict=True,         # True：發現重複資料就直接停止，正式流程使用。
                         # False：只顯示警告但繼續執行，適合目前舊實驗室資料的流程測試。
):
    """
    檢查建立固定 Baseline 的正常 TDMS，是否又被放進頻帶探索、特徵選擇、模型模型訓練或 Holdoutout資料。
    """
    # 檢查 baseline_payload  
    if not isinstance(baseline_payload, dict):
        raise TypeError("baseline_payload 必須是 dict")

    # Baseline JSON 在建立時應記錄實際使用的正常 TDMS 完整路徑。
    baseline_source_paths = (baseline_payload.get("source_tdms_paths", []))

    # 檢查 baseline.source_tdms_paths
    if not isinstance(baseline_source_paths, list):
        raise ValueError("baseline.source_tdms_paths 必須是 list")

    # 正式模式下，如果 Baseline 沒記錄來源，就無法確認是否和分析資料重複。
    if not baseline_source_paths:
        message = (f"{context}：Baseline 沒有 source_tdms_paths，無法驗證資料隔離")

        if strict:
            raise ValueError(message)

        print(f"⚠️ {message}")

        return {
            "overlap_paths": [],
            "overlap_record_ids": [],
            "source_tracking_available": False,
        }

    # --------------------------------------------------------
    # 1. 比較實際完整路徑
    # --------------------------------------------------------
    baseline_paths = {str(Path(path).resolve()) for path in baseline_source_paths}

    analysis_paths = {
        str(Path(record["filepath"]).resolve())
        for record in analysis_records if record.get("filepath")
    }

    overlap_paths = sorted(baseline_paths & analysis_paths)

    # --------------------------------------------------------
    # 2. 再比較 record_id
    # 即使檔案被移到另一個資料夾，只要檔名／record_id 沒有改變，仍有機會被這一層抓出來。
    # --------------------------------------------------------
    baseline_record_ids = {Path(path).stem for path in baseline_source_paths}

    # 同時收集：
    # 1. 分析流程使用的 record_id
    # 2. filepath 的檔名主體
    # Disturbance 的 record_id 通常是 session_id，不一定等於單一 TDMS 檔名，因此兩種都必須比較。
    analysis_record_ids = set()
    
    for record in analysis_records:
        record_id = record.get("record_id")
        filepath = record.get("filepath")
    
        if record_id:
            analysis_record_ids.add(str(record_id))
    
        if filepath:
            analysis_record_ids.add(Path(filepath).stem)

    overlap_record_ids = sorted(baseline_record_ids & analysis_record_ids)

    # 如果路徑及 record_id 都沒有重複，代表本次資料隔離檢查通過。
    if not overlap_paths and not overlap_record_ids:
        print(f"✅ {context} 與固定 Baseline 資料完全隔離")

        return {
            "overlap_paths": [],
            "overlap_record_ids": [],
            "source_tracking_available": True,
        }

    message = (
        f"{context} 與固定 Baseline 資料重複："
        f"路徑重複 {len(overlap_paths)} 支，"
        f"record_id 重複 {len(overlap_record_ids)} 支；"
        f"重複 record_id 前 20 筆="
        f"{overlap_record_ids[:20]}"
    )

    if strict:
        raise ValueError(message)

    # 非嚴格模式只提示，不停止。
    # 這種模式只能用來串接舊資料流程，不應將結果視為完全獨立的正式評估。
    print(f"⚠️ {message}")
    print(
        "⚠️ 目前允許重複僅供流程測試；"
        "正式訓練與效能報告前，"
        "請改用獨立的 Baseline-only 正常資料。"
    )

    return {
        "overlap_paths": overlap_paths,
        "overlap_record_ids": overlap_record_ids,
        "source_tracking_available": True,
    }



#                                        *****************************************************************************
#                                                                     資料讀取 / cache 工具
#                                        *****************************************************************************
# ============================================================
# Band hash + Cache 工具(特徵計算暫存結果)
# ============================================================
# 這份 feature cache 是用哪一套 band, window_sec, hop_sec... 設定算出來的
def make_feature_cache_hash(
    core_bands,    
    aux_bands,
    low_bands,
    window_sec,    
    hop_sec,
    fs=FS,         
    nperseg=NPERSEG,
    target_levels=None,
    feature_version="v2_global_dc",    
):
    config = {
        # 這次用哪些 band
        "core": core_bands,
        "aux": aux_bands,
        "low": low_bands,
        # 切窗方式
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        # 頻譜計算參數
        "fs": fs,
        "nperseg": nperseg,
        "target_levels": sorted(list(target_levels)) if target_levels is not None else None,
        "feature_version": feature_version,
        # 這版特徵工程的版本名
        "temporal_band_names": sorted(list(TEMPORAL_BAND_NAMES)),
    }
    # 把剛剛那個設定字典轉成字串
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


# ============================================================
# 查看某個 cache 資料夾裡目前有哪些 cache 檔
# ============================================================
def list_feature_caches(cache_dir):
    # 把路徑轉成 Path 物件
    cache_dir = Path(cache_dir)
    # 如果資料夾根本不存在，就提示你目前還沒有 cache
    if not cache_dir.exists():
        print("⚠️ 尚無 cache 資料夾")
        return
    # 去找資料夾裡所有符合 features_*.pkl 的檔案，並排序
    files = sorted(cache_dir.glob("features_*.pkl"))
    # 如果資料夾有了，但裡面還沒有特徵 cache，也直接提示
    if not files:
        print("⚠️ 尚無 cache 檔案")
        return
    print(f"\n📦 現有 feature cache ({cache_dir}):")
    # 逐個印出：檔名 , 檔案大小（MB）
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.1f} MB)")


# ============================================================
# 替「已經讀進記憶體的 records」做簽章
# ============================================================
def make_records_signature(records):
    '''
    判斷這批 records 是不是同一批資料
     - 同一批 records -> 同一個 signature
     - 只要 records 內容有差異 -> signature 就會變
    '''
    # 從每個 record 裡抓幾個可以代表這筆資料身份的欄位
    payload = [
        {
            "record_id": r["record_id"],
            "fs": float(r["fs"]),
            "n_ai1": len(r["signal"]["ai1"]),   # ai1 的長度
            "t_start": r.get("t_start", None),  # 如果有時間起點就一起放
        }
        for r in records
    ]
    # 把 payload 轉成字串，再做 MD5，取前 12 碼當 records signature
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ============================================================
# by_file cache 不能再用 records/windows 簽章,要改成用檔案清單來簽
# ============================================================
def make_file_specs_signature(file_specs):
    payload = [
        {
            "record_id": f["record_id"],
            "filepath": str(f["filepath"]),
            "size": Path(f["filepath"]).stat().st_size,   # 其中 size 很重要，因為即使檔名一樣，只要檔案內容被更新，檔案大小通常也會變
        }
        for f in file_specs
    ]
    # 先依 record_id 排序，再轉成字串
    s = json.dumps(sorted(payload, key=lambda x: x["record_id"]), sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ============================================================
# 把一個 TDMS 檔讀進來，整理成你後面特徵工程統一使用的 record 格式
# ============================================================
def load_one_tdms_record(
    filepath,        # TDMS 路徑
    record_id=None,  # 如果你想手動指定 record 名稱
    label=None,      # 這筆資料的標籤
    max_points_per_file=8_000_000,    # 最多讀多少點
    fs_target=FS,    # 如果 TDMS 裡抓不到取樣率，就用這個預設值
):
    # 把路徑轉成 Path，然後讀取 TDMS 檔
    filepath = Path(filepath)
    tdms = TdmsFile.read(filepath)
    # 先取第一個 group, 再把這個 group 裡的 channels 整理成字典，方便用名字查
    group = tdms.groups()[0]
    channels = {ch.name: ch for ch in group.channels()}
    
    # 如果有明確叫 ai0、ai1 的 channel，就用它們, 否則就退回成「取前兩個 channel」
    if "ai0" in channels and "ai1" in channels:
        channel_ai0 = channels["ai0"]
        channel_ai1 = channels["ai1"]
    else:
        ch_list = group.channels()
        channel_ai0 = ch_list[0]
        channel_ai1 = ch_list[1]
    # 讀出兩個聲道資料，並轉成 numpy array
    ai0 = np.asarray(channel_ai0[:max_points_per_file], dtype=float)
    ai1 = np.asarray(channel_ai1[:max_points_per_file], dtype=float)
    
    # 嘗試從 TDMS 的 metadata 裡面讀取取樣率
    fs = None
    if hasattr(channel_ai0, "properties"):
        inc = channel_ai0.properties.get("wf_increment")    # wf_increment 通常是每點的時間間隔
        if inc:
            fs = 1.0 / inc
    if fs is None:     # 如果 TDMS 裡沒有取樣率資訊，就退回用你傳進來的預設值
        fs = fs_target
        
    # 用檔名去解析 metadata
    meta = parse_filename(filepath.name)
    # 保證兩個聲道長度一致，如果兩邊長度不同，就截成相同最短長度
    n = min(len(ai0), len(ai1))
    ai0 = ai0[:n]
    ai1 = ai1[:n]
    
    # 如果有啟用 REMOVE_DC，就先把每個聲道做去直流
    if REMOVE_DC:
        ai0 = ai0 - np.mean(ai0)
        ai1 = ai1 - np.mean(ai1)

    # 建立第三條訊號：差分通道
    diff = ai1 - ai0

    return {
        "filename": filepath.name,
        "filepath": str(filepath),
        "record_id": record_id if record_id is not None else filepath.stem,
        "fs": fs,
        "meta": meta,
        "signal": {
            "ai0": ai0,
            "ai1": ai1,
            "diff": diff,
        },
        "label": label,
    }


# ============================================================
# 一次處理一個檔案 -> 讀 TDMS -> 切 windows -> 算這個檔案 window 的特徵
# ============================================================
def _process_one_file_worker(args):
    (
        file_spec,     # 檔案的基本資料
        core_bands,    # 主 band 清單
        aux_bands,     # 輔助 band 清單
        low_bands,     # 低頻 band 清單
        window_sec,    # 視窗長度
        hop_sec,       # 視窗滑動步長
        fs,            # 取樣率
        max_points,    # 單檔最大讀取點數
    ) = args
    # 讀一個 TDMS 檔案，整理成一筆 record
    rec = load_one_tdms_record(
        filepath=file_spec["filepath"],        # TDMS 檔案路徑
        record_id=file_spec.get("record_id"),  # 這筆資料的名稱或 id
        label=file_spec.get("label"),          # 這筆資料是不是 leak
        max_points_per_file=max_points,        # 避免一次讀太大
        fs_target=fs,                          # 用指定取樣率讀取
    )

    # 把整段訊號切成很多小視窗
    windows = sliding_window_records(
        [rec],
        fs=fs,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    
    # 計算特徵
    feats = extract_window_features_core_bands(
        windows,
        core_bands=core_bands,    # 算 core_bands(final_main_bands) 的 band 特徵
        aux_bands=aux_bands,      # 算 aux_bands(offband_bands_for_model) 的 band 特徵
        low_bands=low_bands,      # 算 low_bands([]) 的 band 特徵
        verbose=False,
    )
    return feats


# ============================================================
# 把很多個檔案平行丟給 _process_one_file_worker() 去算特徵
# ============================================================
def extract_features_from_files_parallel(
    file_specs,              # 多個檔案
    core_bands,              # 要用哪組 band
    aux_bands,
    low_bands,
    window_sec=0.2,          # 切窗設定
    hop_sec=0.05,
    fs=FS,                   # 取樣率
    max_points=8_000_000,    # 單檔最大讀取點數
    n_jobs=-2,               # 平行工作數
):
    # 每個檔案準備好一包參數
    args_list = [
        (
            f,
            core_bands,
            aux_bands,
            low_bands,
            window_sec,
            hop_sec,
            fs,
            max_points,
        )
        for f in file_specs
    ]
    # 平行運算核心
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_file_worker)(args)
        for args in tqdm(args_list, desc="特徵提取中(file)", unit="file")
    )

    # results 是：檔案1的一堆 windows 特徵 , 檔案2的一堆 windows 特徵 , 檔案3的一堆 windows 特徵 , 這裡是在把它攤平成一個大 list。
    feature_list = [feat for one_file_feats in results for feat in one_file_feats]
    print(f"✅ 平行特徵提取完成：{len(file_specs)} files -> {len(feature_list)} windows")
    return feature_list


# ============================================================
# 先看 cache 有沒有算過，如果有就直接讀；沒有才真的去平行計算：吃 records
# ============================================================
def load_or_compute_features_by_file(
    cache_dir,
    file_specs,
    core_bands,
    aux_bands,
    low_bands,
    split_name="dev",
    n_jobs=-2,
    window_sec=None,
    hop_sec=None,
    fs=FS,
    nperseg=NPERSEG,
    target_levels=None,
    feature_version="v2_global_dc",
    max_points=8_000_000,
):
    # 先確保 cache 資料夾存在
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 幫這次特徵設定做一個 hash 把：band 設定 , 視窗設定 , PSD 設定 , target levels , feature version，整理成一個唯一指紋
    cache_hash = make_feature_cache_hash(
        core_bands=core_bands,
        aux_bands=aux_bands,
        low_bands=low_bands,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        target_levels=target_levels,
        feature_version=feature_version,
    )

    # 幫這批檔案本身做一個 signature(因為就算 band 沒變，只要檔案清單變了，也不應該共用同一份 cache)
    file_sig = make_file_specs_signature(file_specs)
    cache_path = cache_dir / f"features_{split_name}_{cache_hash}_{file_sig}.pkl"

    # 如果 cache 已經存在：直接讀 cache, 不重新算特徵
    if cache_path.exists():
        print(f"✅ 讀取 cache ({split_name}, hash={cache_hash}, files={file_sig}): {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
        
    # 如果沒有 cache，就印出訊息，表示現在要重新算
    print(f"⚙️ cache miss，開始計算 {split_name}（hash={cache_hash}）...")
    # 進入平行計算, 把前面準備好的 band 設定和檔案丟進去跑
    result = extract_features_from_files_parallel(
        file_specs=file_specs,
        core_bands=core_bands,
        aux_bands=aux_bands,
        low_bands=low_bands,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        max_points=max_points,
        n_jobs=n_jobs,
    )
    # 算完後把結果存成 cache
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    print(f"✅ 已存 cache: {cache_path.name}")
    return result


# ============================================================
#  cache 的主入口(吃 file_specs):建立 cache 資料夾 -> 根據 band 設定算出 band_hash -> 組出 cache 檔名
# ============================================================
def load_or_compute_features(
    cache_dir,           # cache 要存在哪個資料夾
    records,             # 這次要處理的資料
    core_bands,          # 主 band 清單
    aux_bands,           # 輔助 band 清單
    low_bands,           # 低頻 band 清單
    split_name="dev",    # 這批資料的名字，預設叫 dev
    n_jobs=-2,           # 平行運算的工作數
    window_sec=None,     # 每個 window 多長
    hop_sec=None,        # 每次滑動多少
    fs=FS,               # 取樣率
    nperseg=NPERSEG,     # Welch PSD 的參數
    target_levels=None,  # 這次分析的 leak levels
    feature_version="v2_global_dc",   # 這版特徵工程的版本名字
):
    # 把 cache_dir 轉成 Path 物件, 如果資料夾不存在，就先建立
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 產生一個 cache_hash(特徵工程設定的指紋), 就是把：用了哪些 band, 視窗多長, hop 多大, 取樣率多少, PSD 參數多少, 分析哪幾個 level, 特徵版本是什麼, 全部綁在一起，算成一個 hash。
    cache_hash = make_feature_cache_hash(
        core_bands=core_bands,
        aux_bands=aux_bands,
        low_bands=low_bands,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        target_levels=target_levels,
        feature_version=feature_version,
    )
    # records 算一個 signature(這次拿來算特徵的是哪一批資料)
    records_sig = make_records_signature(records)
    # 組出 cache 檔名
    cache_path = cache_dir / f"features_{split_name}_{cache_hash}_{records_sig}.pkl"

    # 檢查：這份 cache 檔案是不是已經存在, 如果 cache 有，就完全不重算，直接結束
    if cache_path.exists():
        print(f"✅ 讀取 cache ({split_name}, hash={cache_hash}, records={records_sig}): {cache_path.name}")
        # 直接把 cache 檔打開讀進來
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"⚙️ cache miss，開始計算 {split_name}（hash={cache_hash}）...")
    # cache miss，要開始重新算特徵
    result = _extract_features_parallel(
        records,
        core_bands=core_bands,
        aux_bands=aux_bands,
        low_bands=low_bands,
        n_jobs=n_jobs,
    )
    # 特徵算完後，馬上把結果存成 cache
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    print(f"✅ 已存 cache: {cache_path.name}")
    return result


# ============================================================
# 平行化包裝層（不動原本的 extract_window_features_core_bands）
# ============================================================
def _extract_one_window(args):
    """
    joblib 用的單一 window worker
    把參數打包成 tuple 傳進來，避免 pickle 問題
    「原本一次吃很多 windows 的函式」包成「一次只處理一個 window 的 worker」，方便平行化
    """
    r, core_bands, aux_bands, low_bands = args
    # 直接呼叫原本的函式，只是把單一 window 包成 list 傳進去
    result = extract_window_features_core_bands(
        [r],
        core_bands=core_bands,
        aux_bands=aux_bands,
        low_bands=low_bands,
        verbose=False
    )
    return result[0]  # 只有一個 window，取第一個


# 平行特徵提取
def _extract_features_parallel(records, core_bands, aux_bands, low_bands, n_jobs=-2):
    # 把每個 window 跟 band 參數包成 args_list
    args_list = [(r, core_bands, aux_bands, low_bands) for r in records]
    
    # 用 Parallel 平行跑
    feature_list = Parallel(n_jobs=n_jobs, prefer="threads")(    # n_jobs=-2 通常表示用掉大部分 CPU core，但保留一顆
        delayed(_extract_one_window)(args)    # delayed(_extract_one_window)(args) 表示每個 window 都交給 _extract_one_window 處理
        for args in tqdm(args_list, desc="特徵提取中", unit="window")
    )
    print(f"✅ 平行特徵提取完成：{len(feature_list)} 筆")
    return feature_list



#                                        *****************************************************************************
#                                                                       特徵工程共用函式
#                                        *****************************************************************************
#                                                                 =====     檔名拆解     =====
# 檔名拆解
def parse_filename(filename):
    """
    從 TDMS 檔名中解析出實驗 metadata（距離、壓力、洩漏量）並轉成數值格式
    解析檔名，獲取距離（m）、壓力（bar）、洩漏量（ml）
     - 距離（distance, 單位：公尺）
     - 壓力（pressure, 單位：bar）
     - 洩漏量（leak_quantity, 單位：ml）

    支援格式範例：
    1. leak1_20ml_03bar_straight50cm_ss_test.tdms
    
    Returns
    -------
    dict : {'distance': float (m), 'pressure': float (bar), 'leak_quantity': float (ml)}
    """
    filename_lower = filename.lower()   # 檔名轉小寫,避免大小寫問題
    
    # -------  洩漏量(優先從 ml 單位解析)  -------
    leak_quantity = 0.0    # 預測沒有洩漏
    
    if "normal" in filename_lower:
        leak_quantity = 0.0
    
    else:
        # 新格式：50_leak-near 20 cm_pair010
        direct_match = re.search(r'(\d+)_leak', filename_lower)
        
        if direct_match:
            # group(1)是指第一個括號抓到的內容
            leak_quantity = float(direct_match.group(1)) 
        else:
            # 舊格式：20ml
            ml_match = re.search(r'(\d+)\s*ml', filename_lower)   # (\d+)ml 表示「抓數字後面接 ml」
            if ml_match:  # 如果檔名有明確標註 ml，直接用這個值。
                leak_quantity = float(ml_match.group(1))
            else:   # 如果都沒有，就用 leak code 映射表
                leak_match = re.search(r'leak(\d+)', filename_lower)
                if leak_match:
                    leak_code = int(leak_match.group(1))
                    leak_map = {
                        1:20, 
                        2:50,
                        3:100,
                        4:150,
                        5:200,
                        6:400
                        }
                    leak_quantity = float(leak_map.get(leak_code, 0))
                    
    #  -------  距離(轉成公尺)  -------
    distance = 0.0
    
    cm_match = re.search(r'(\d+)\s*cm', filename_lower)
    # 優先抓cm(轉公尺),在抓m
    if cm_match:
        distance = float(cm_match.group(1)) / 100.0
    else:
        m_match = re.search(r'(\d+)m', filename_lower)
        if m_match:
            distance = float(m_match.group(1))
            
    #  -------  壓力  -------
    pressure = 0.0
    pressure_match = re.search(r'(\d+)\s*bar', filename_lower)
    if pressure_match:
        pressure = float(pressure_match.group(1))
        
    #print(f"\n完成檔名拆解，獲取距離（m）、壓力（bar）、洩漏量（ml）")
    return{
        "distance": distance,
        "pressure": pressure,
        "leak_quantity": leak_quantity
        }


# =====    訊號分離     =====
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


# =======================================
# DC remove
# =======================================
def remove_dc(ai0: np.ndarray, ai1: np.ndarray):
    """去除直流偏移，跟訓練時保持一致"""
    return ai0 - np.mean(ai0), ai1 - np.mean(ai1)


# =======================================
# sliding window
# =======================================
# 將一個 tdms 檔案切成小視窗 window slide
def sliding_window_records(records, fs=1_000_000, window_sec=0.2, hop_sec=0.05):
    """
    把每個 record 切成多個 window
    records   ：原始資料，每一筆 record 裡面有 ai0、ai1、diff 等資訊。
    window_sec: 每個 window 的長度（秒）,每個window看200ms
    hop_sec   : 每次移動的步長（秒），overlap = 1 - hop/window,每次往前滑50ms,也就是 75% overlap
    """
    if not records:
        print("⚠️ 沒有 records 可做 windowing")
        return []
    
    fs_values = np.array([r["fs"] for r in records], dtype=float)

    if not np.allclose(fs_values, fs):
        warnings.warn(
            f"部分 record 的 fs 不等於指定 fs={fs}。"
            f"實際範圍: {fs_values.min():.2f} ~ {fs_values.max():.2f} Hz"
        )
    
    # 把秒轉成樣本點數(每個window有20萬點,每次往前移5萬點)
    window_size = int(fs * window_sec)     # 每個 window_sec = 0.2 秒,若 fs=1,000,000，那就代表每個window有20萬點
    hop_size    = int(fs * hop_sec)        # 75% overlap,優點：時間解析度高,leak onset 比較容易抓到,probability stream 會更平滑細緻

    windowed = []
    for r in records:
        signal_len = len(r["signal"]["ai1"])    # 取這筆資料中 ai1 的長度，也就是這個 record 總共有多少點
        record_id  = r["record_id"]              # 把原始檔名存下來，後面每個 window 都要知道自己來自哪個原始檔

        start = 0      # 第一個 window 從第0點開始
        win_idx = 0    # 第一個 window 編號是0
        
        # 只要還能切就一直切
        while start + window_size <= signal_len:
            end = start + window_size    # 算出該 window 的結尾位置
            
            # 建立時間點
            t_start = start / fs
            t_end = end / fs
            t_ref = (t_start + t_end) / 2.0
            
            # 建立一個 window 的資料結構
            windowed.append({
                "filename":  f"{r['filename']}_w{win_idx:03d}",     # 每個小視窗的新名稱
                "record_id": record_id,                             # 原檔名 -> GroupKFold 用
                "t_start":   t_start,                          # 起始秒數
                "t_end":     t_end,
                "t_ref":     t_ref,
                "fs":        fs,                               # 採樣率
                "label":     r["label"],                            # label標籤
                "meta":      r["meta"],                             # metedata
                "signal": {
                    "ai0":  r["signal"]["ai0"][start:end],          # 這個 window 的 ai0
                    "ai1":  r["signal"]["ai1"][start:end],          # 這個 window 的 ai1
                    "diff": r["signal"]["diff"][start:end],          # 這個 window 的 diff
                }
            })
            start   += hop_size
            win_idx += 1

    print(f"✅ windowing 完成：{len(records)} records → {len(windowed)} windows")
    print(f"   window={window_sec}s, hop={hop_sec}s, overlap={1-hop_sec/window_sec:.0%}")
    return windowed


# ============================================================
# 特徵工程：核心頻段優先版本
# ============================================================
def extract_window_features_core_bands(records, core_bands, aux_bands=None, low_bands=None, verbose=True):
    """
    把每一個切好的訊號視窗，轉成一列可丟進機器學習模型的特徵資料。
    而且它不是對所有頻段一視同仁，而是採用你現在的策略：
        - 低頻段
        - 核心頻段 core_bands：抽很多、抽完整
        - 輔助頻段 aux_bands：只抽少量摘要
    目的就是讓模型優先看你 EDA 已經證明重要的頻段
    """
    # 避免你沒有傳 aux_bands, low_band 時報錯,如果沒給，就當成空列表。
    if aux_bands is None:
        aux_bands = []
    if low_bands is None:
        low_bands = []
        
    temporal_bands = [
        b for b in core_bands
        if b["name"] in TEMPORAL_BAND_NAMES
    ]

    feature_list = []   # 建立最終要輸出的特徵表
    # 從切好的視窗拿出 ai0,ai1,diff,fs
    for r in records:
        ai0  = r["signal"]["ai0"]     # far
        ai1  = r["signal"]["ai1"]     # near
        diff = r["signal"]["diff"]    # 差分
        fs   = r["fs"]                # 採樣率
        
        # 建立一個feat dict 儲存 filename/record_id/t_start/label
        feat = {
            "filename":  r["filename"],
            "record_id": r["record_id"],
            "t_start":   r["t_start"],
            "t_end":     r.get("t_end", np.nan),
            "t_ref":     r.get("t_ref", r["t_start"]),
            "label":     r["label"],
        }
        
        # ===== ReferenceSubtractor（用 ai0 消共模噪音）=====
        # 核心概念：ai0 和 ai1 都聽到同樣的背景噪音，把共同的部分消掉，剩下的才是洩漏訊號。
        beta, _, _, _ = lstsq(ai0.reshape(-1, 1), ai1, rcond=None)    # lstsq 在找一個數字 β，讓 β × ai0 盡量接近 ai1。數學上就是最小化
        residual = ai1 - beta[0] * ai0     # 找到 β 之後，residual = ai1 - β × ai0 就是「ai1 裡面，ai0 解釋不了的部分」
        
        feat["ref_sub_beta"] = float(beta[0])                                      # 比例係數，正常狀態下兩個感測器靈敏度相近，β ≈ 1；洩漏時 near 能量多，β 會偏移
        feat["ref_sub_residual_rms"] = float(np.sqrt(np.mean(residual**2)))        # 消噪後殘差的整體能量，洩漏時應該比正常大
        feat["ref_sub_residual_std"] = float(np.std(residual))                     # 殘差的波動程度
        feat["ref_sub_residual_kurtosis"] = float(kurtosis(residual, fisher=True, bias=False))      # 峰度，洩漏訊號通常是脈衝型，峰度會比純噪音高很多
        
        # ===== global residual temporal =====
        sig_global = bandpass_filter(residual, fs, 1_000, 40_000)
        feat.update(
            extract_temporal_event_features(
                sig_global, fs, prefix="residual_global"
            )
        )
        
        # backward-compatible aliases
        feat["cusum_max"] = feat["residual_global_cusum_peak"]
        feat["cusum_alert_rate"] = feat["residual_global_cusum_alert_rate"]
        feat["cusum_rise_count"] = feat["residual_global_cusum_rise_count"]
        feat["burst_rate"] = feat["residual_global_burst_rate"]
        feat["burst_event_count"] = feat["residual_global_burst_event_count"]
        
        # ===== selected-band residual temporal =====
        for band in temporal_bands:
            name = band["name"]
            residual_band = bandpass_filter(residual, fs, band["f_low"], band["f_high"])
        
            # ReferenceSubtractor 的 band 版摘要：看 global residual 在這個 band 的能量/波動
            feat[f"{name}_residual_rms"] = float(np.sqrt(np.mean(residual_band ** 2)))
            feat[f"{name}_residual_std"] = float(np.std(residual_band))
            feat[f"{name}_residual_kurtosis"] = float(kurtosis(residual_band, fisher=True, bias=False))
        
            # CUSUM / BurstRate 的 select-band 版本
            feat.update(
                extract_temporal_event_features(
                    residual_band, fs, prefix=f"{name}_residual"
                )
            )
        
        # ===== 全域時間域(三通道) =====
        feat.update(extract_basic_time_features(ai0,  "ai0", fs=fs))
        feat.update(extract_basic_time_features(ai1,  "ai1", fs=fs))
        feat.update(extract_basic_time_features(diff, "diff", fs=fs))
                
        # ===== 三通道間相對強度 ===== 
        feat["rms_ratio_ai1_ai0"] = feat["ai1_rms"] / (feat["ai0_rms"] + 1e-12)
        feat["rms_ratio_diff_ai1"] = feat["diff_rms"] / (feat["ai1_rms"] + 1e-12)
        feat["rms_ratio_diff_ai0"] = feat["diff_rms"] / (feat["ai0_rms"] + 1e-12)
        
        # ===== ai0/ai1 原始訊號互相關峰值 =====
        # 跟上面的 ReferenceSubtractor 互補：ref_sub 看的是「消噪後殘差還剩多少」，這裡看的是「兩通道波形本身有多同步/對齊」，量的東西不一樣。
        E_ai0 = float(np.sum(ai0 ** 2)) + 1e-12
        E_ai1 = float(np.sum(ai1 ** 2)) + 1e-12
        xc = correlate(ai0, ai1, mode="same", method="fft")
        feat["xcorr_peak_ai0_ai1"] = float(np.max(np.abs(xc)) / np.sqrt(E_ai0 * E_ai1))

        # ===== ai1 全域頻譜重心：代表整段 ai1 頻譜能量偏向哪個頻率 =====
        f, Pxx1 = compute_psd(ai1, fs, normalize=False, smooth_sigma=0)
        feat["centroid_ai1_global"] = np.sum(f * Pxx1) / (np.sum(Pxx1) + 1e-12)
        
        
        # ===== 低頻頻段：探索性摘要(可拿掉，因為命名不同了) =====
        for band in low_bands:
            name = band["name"]
        
            lf_ai0 = extract_band_features_one_channel(ai0, fs, band, f"{name}_ai0", compute_env=False)
            lf_ai1 = extract_band_features_one_channel(ai1, fs, band, f"{name}_ai1", compute_env=False)
            lf_diff = extract_band_features_one_channel(diff, fs, band, f"{name}_diff", compute_env=False)
                    
            feat[f"{name}_ai1_E"] = lf_ai1[f"{name}_ai1_E"]
            feat[f"{name}_ai1_RMS"] = lf_ai1[f"{name}_ai1_RMS"]
            feat[f"{name}_ai1_DomFreq"] = lf_ai1[f"{name}_ai1_DomFreq"]
            feat[f"{name}_diff_E"] = lf_diff[f"{name}_diff_E"]
            feat[f"{name}_ratio_ai1_ai0"] = (
                lf_ai1[f"{name}_ai1_E"] / (lf_ai0[f"{name}_ai0_E"] + 1e-12)
            )

        # ===== 核心頻段(洩漏+共振)：完整抽 =====
        core_ai1_energy_map = {}
        core_diff_energy_map = {}
        
        for band in core_bands:
            name = band["name"]
            # 對每個 core_bands：ai0/ai1/diff抽完整特徵
            # 每個核心 band 都有 E, RMS, Peak, EnvMean, EnvRMS, EnvStd, DomFreq, Centroid, Entropy, Flatness
            f_ai0  = extract_band_features_one_channel(ai0,  fs, band, f"{name}_ai0")
            f_ai1  = extract_band_features_one_channel(ai1,  fs, band, f"{name}_ai1")
            f_diff = extract_band_features_one_channel(diff, fs, band, f"{name}_diff")

            feat.update(f_ai0)
            feat.update(f_ai1)
            feat.update(f_diff)
            
            core_ai1_energy_map[name] = f_ai1[f"{name}_ai1_E"]
            core_diff_energy_map[name] = f_diff[f"{name}_diff_E"]
            
            # 加上三種band energy ratio
            feat[f"{name}_ratio_ai1_ai0"] = (feat[f"{name}_ai1_E"] / (feat[f"{name}_ai0_E"] + 1e-12))
            feat[f"{name}_ratio_diff_ai1"] = (feat[f"{name}_diff_E"] / (feat[f"{name}_ai1_E"] + 1e-12))
            feat[f"{name}_ratio_diff_ai0"] = (feat[f"{name}_diff_E"] / (feat[f"{name}_ai0_E"] + 1e-12))
            
            # 加上 logratio 能量
            feat[f"{name}_logratio_ai1_ai0"] = np.log((feat[f"{name}_ai1_E"] + 1e-12) / (feat[f"{name}_ai0_E"] + 1e-12))
            feat[f"{name}_logratio_diff_ai1"] = np.log((feat[f"{name}_diff_E"] + 1e-12) / (feat[f"{name}_ai1_E"] + 1e-12))
            feat[f"{name}_logratio_diff_ai0"] = np.log((feat[f"{name}_diff_E"] + 1e-12) / (feat[f"{name}_ai0_E"] + 1e-12))

            # 加上 shape - only PSD
            shape_ai0 = extract_shape_only_psd_features(ai0, fs, band["f_low"], band["f_high"], f"{name}_ai0")
            shape_ai1 = extract_shape_only_psd_features(ai1, fs, band["f_low"], band["f_high"], f"{name}_ai1")
            shape_diff = extract_shape_only_psd_features(diff, fs, band["f_low"], band["f_high"], f"{name}_diff")
            
            feat.update(shape_ai0)
            feat.update(shape_ai1)
            feat.update(shape_diff)
            
        # 加上 ai1/diff 在 core bands 的總能量
        total_core_ai1_E = sum(core_ai1_energy_map.values())
        total_core_diff_E = sum(core_diff_energy_map.values())
        
        for band in core_bands:
            name = band["name"]
            feat[f"{name}_ai1_frac_core"] = (
                core_ai1_energy_map[name] / (total_core_ai1_E + 1e-12)
            )
            feat[f"{name}_diff_frac_core"] = (
                core_diff_energy_map[name] / (total_core_diff_E + 1e-12)
            )

        # ===== 輔助頻段：少量摘要 =====
        for band in aux_bands:
            name = band["name"]

            aux_ai0 = extract_band_features_one_channel(ai0, fs, band, f"{name}_ai0", compute_env=False)
            aux_ai1 = extract_band_features_one_channel(ai1, fs, band, f"{name}_ai1", compute_env=False)
            
            # 對 aux bands，同時保留ai0,ai1 的 E, RMS, DomFreq, Entropy,以及 ai1/ai0 的 energy ratio。
            feat[f"{name}_ai0_E"] = aux_ai0[f"{name}_ai0_E"]
            feat[f"{name}_ai0_RMS"] = aux_ai0[f"{name}_ai0_RMS"]
            feat[f"{name}_ai0_DomFreq"] = aux_ai0[f"{name}_ai0_DomFreq"]
            feat[f"{name}_ai0_Entropy"] = aux_ai0[f"{name}_ai0_Entropy"]
            
            feat[f"{name}_ai1_E"] = aux_ai1[f"{name}_ai1_E"]
            feat[f"{name}_ai1_RMS"] = aux_ai1[f"{name}_ai1_RMS"]
            feat[f"{name}_ai1_DomFreq"] = aux_ai1[f"{name}_ai1_DomFreq"]
            feat[f"{name}_ai1_Entropy"] = aux_ai1[f"{name}_ai1_Entropy"]
            
            feat[f"{name}_ratio_ai1_ai0"] = (
                aux_ai1[f"{name}_ai1_E"] / (aux_ai0[f"{name}_ai0_E"] + 1e-12)
            )
        # ===== ratio feature（高低頻比）=====
        # 不寫死頻率：把目前所有 band（core=leak_focused+resonance, aux=offband）
        # 依中心頻率排序，用中位數當分界分成低頻群/高頻群，各自取 ai1 能量平均後相除。
        # 頻段隨資料收斂改變時，分界自動跟著調整。
        
        # 蒐集每個 band 的「中心頻率」和「能量」
        _ratio_bands = list(core_bands) + list(aux_bands)      # 把所有 band(core = scan 洩漏/共振頻段 + aux = offband 參照頻段)合起來
        _band_centers = []
        # 逐個算出兩個東西——中心頻率(f_low 跟 f_high 的中間值)和這個 band 的 ai1 能量。存成一堆 (中心頻率, 能量) 的配對
        for _b in _ratio_bands:
            _e = feat.get(f"{_b['name']}_ai1_E", None)
            if _e is None:
                continue
            _center = (float(_b["f_low"]) + float(_b["f_high"])) / 2.0
            _band_centers.append((_center, float(_e)))

        # 按頻率排序,找「分界線」
        if len(_band_centers) >= 2:     # 先確認至少有 2 個 band
            # 那堆配對按中心頻率由低到高排序,拆成兩個陣列:一個裝所有頻率、一個裝對應的能量。
            _band_centers.sort(key=lambda t: t[0])   
            _centers = np.array([c for c, _ in _band_centers])
            _energies = np.array([e for _, e in _band_centers])
            # 最後算所有中心頻率的中位數當「分界線」_split
            _split = np.median(_centers)

            # 用分界線分成「低頻群 / 高頻群」
            _low_mask = _centers < _split      # 頻率低於分界的歸「低頻群」
            _high_mask = _centers >= _split    # 高於等於分界的歸「高頻群」

            # 萬一中位數剛好讓某一群變空的(所有 band 都被分到同一邊),就改用最笨但保證有效的辦法——照數量對半切
            if _low_mask.sum() == 0 or _high_mask.sum() == 0:
                _n = len(_centers)
                _low_mask = np.arange(_n) < (_n // 2)
                _high_mask = ~_low_mask
            
            # 各群能量取平均,相除得 ratio
            _low_E = float(_energies[_low_mask].mean())            # 低頻群所有 band 的能量取平均
            _high_E = float(_energies[_high_mask].mean())          # 高頻群取平均
            feat["high_low_ratio"] = _high_E / (_low_E + 1e-12)
        else:
            feat["high_low_ratio"] = np.nan
        
        # ===== log energy feature =====
        # 對每個核心 band、每個通道再多做一個 logE,因為能量可能差很多階，取 log 對模型通常比較穩。
        for band in core_bands:
            name = band["name"]
            for ch in ["ai0", "ai1", "diff"]:
                key     = f"{name}_{ch}_E"
                log_key = f"{name}_{ch}_logE"
                feat[log_key] = np.log(max(feat.get(key, 0.0), 0.0) + 1e-12)
        
        feature_list.append(feat)

    if verbose:
        print(f"✅ 核心頻段優先特徵完成：{len(feature_list)} 筆")
    return feature_list


# 計算基本時域特徵(RMS/Peak/Std/Mean absolute value/Crest factor/impulse factor/Skewness/ZCR/包絡波峰因子/包絡峰度)
def extract_basic_time_features(signal, prefix, suffix="", fs=None):    
    '''
    輸入一段訊號 x，計算幾個最基本的時間域特徵，並用 prefix 幫特徵命名，若是ai1,則輸出會是ai1_rms,ai1_pesk
    輸入參數：
        signal: 1D 訊號 -> ai0,ai1,diff
        fs: 取樣率，給 zero_crossing_rate 換算成 Hz 用；不給的話退回每點過零比例
    '''
    signal = np.asarray(signal, dtype=float)
    
    rms = np.sqrt(np.mean(signal ** 2))       # RMS 代表訊號的整體能量強度(訊號振幅越大，RMS 通常也越大)
    peak = np.max(np.abs(signal))             # Peak（峰值）:這段訊號中最大的瞬間振幅是多少
    std = np.std(signal)                      # 標準差:訊號波動程度
    mean_abs = np.mean(np.abs(signal))        # 平均絕對值:振幅大小
    
    crest = peak / (rms + 1e-12)              # Crest Factor（波峰因子）:這段訊號的最大尖峰，相對於整體平均能量，有多突出
    impulse = peak / (mean_abs + 1e-12)
    
    # ── 偏度 skewness：訊號分布左右不對稱的程度 ──
    mu = np.mean(signal)
    if std > 1e-12:
        skewness = float(np.mean(((signal - mu) / std) ** 3))
    else:
        skewness = 0.0
        
    # ── 過零率 zero_crossing_rate：訊號正負號翻轉得多快，洩漏訊號通常比背景噪音更高頻、翻轉更頻繁 ──
    sign = np.sign(signal)
    zero_crossings = int(np.sum(np.abs(np.diff(sign)) > 0))
    if fs is not None:
        zcr = zero_crossings / max(1e-9, len(signal) / fs)   # 換算成每秒過零次數(Hz)
    else:
        zcr = zero_crossings / max(1, len(signal))

    # ── 包絡波峰因子 env_peak_factor、包絡峰度 env_kurtosis：用 Hilbert 包絡看能量outline的突出程度 ──
    env = np.abs(hilbert(signal))
    env_mean = np.mean(env)
    env_std = np.std(env)
    env_peak_factor = float(env.max() / (env_mean + 1e-12))
    if env_std > 1e-12:
        env_kurtosis = float(np.mean(((env - env_mean) / env_std) ** 4))
    else:
        env_kurtosis = 0.0
        
    return {
        f"{prefix}_rms{suffix}": rms,
        f"{prefix}_peak{suffix}": peak,
        f"{prefix}_std{suffix}": std,
        f"{prefix}_mean_abs{suffix}": mean_abs,
        f"{prefix}_crest{suffix}": crest,
        f"{prefix}_impulse{suffix}": impulse,
        f"{prefix}_skewness{suffix}": skewness,
        f"{prefix}_zero_crossing_rate{suffix}": zcr,
        f"{prefix}_env_peak_factor{suffix}": env_peak_factor,
        f"{prefix}_env_kurtosis{suffix}": env_kurtosis,
    }


# 頻帶內的頻譜形狀：只看某個頻帶裡的頻譜「長相」，不要看它整體有多強
def extract_shape_only_psd_features(signal, fs, f_low, f_high, prefix):     # f_low、f_high 是「這個頻帶的下限和上限頻率」,是指每個頻帶的屬性
    """
    只保留頻譜形狀，不保留絕對能量尺度
    在指定頻帶內把 PSD 正規化成機率分布後，抽 shape features
    - 能量集中在哪裡
    - 分布有多寬
    - 頻譜有多亂
    - 頻譜像尖峰還是像平坦雜訊
    """
    # 先算 PSD,先保留原始PSD,不使用平滑
    f, Pxx = compute_psd(signal, fs, normalize=False, smooth_sigma=0)

    # 只取指定頻帶
    mask = (f >= f_low) & (f <= f_high)
    f_band = f[mask]
    p_band = Pxx[mask]

    # 如果這段頻帶沒有資料或總功率不合理，就回傳 0
    if len(f_band) == 0 or np.sum(p_band) <= 0:
        return {
            f"{prefix}_ShapeCentroid": 0.0,
            f"{prefix}_ShapeSpread": 0.0,
            f"{prefix}_ShapeEntropy": 0.0,
            f"{prefix}_ShapeFlatness": 0.0,
        }

    # 把 band 內 PSD 正規化成機率分布
    p = p_band / (np.sum(p_band) + 1e-12)

    # ShapeCentroid
    centroid = np.sum(f_band * p)                               # 頻譜形狀的重心:這段 band 的能量分布，平均偏向哪個頻率位置
    spread = np.sqrt(np.sum(((f_band - centroid) ** 2) * p))    # 頻譜分布的寬度:這段 band 的能量是集中在某一小段，還是分散在整個頻帶裡
    entropy = -np.sum(p * np.log(p + 1e-12))                    # 頻譜形狀的熵:這段 band 的能量分布有多亂、多分散
    flatness = np.exp(np.mean(np.log(p_band + 1e-12))) / (np.mean(p_band) + 1e-12)    # 頻譜平坦度:這段頻譜像白雜訊那樣平平的，還是像共振峰那樣尖尖的
 
    return {
        f"{prefix}_ShapeCentroid": centroid,      # 頻譜形狀的重心
        f"{prefix}_ShapeSpread": spread,          # 頻譜分布的寬度
        f"{prefix}_ShapeEntropy": entropy,        # 頻譜形狀的熵
        f"{prefix}_ShapeFlatness": flatness,      # 頻譜平坦度
    }


# ============================================================
# 計算時間結構特徵
# ============================================================
def extract_temporal_event_features(signal,             # 原始訊號，一維 array
                                    fs,                 # 取樣率
                                    prefix,             # 輸出特徵名稱前綴，例如 ai1、diff
                                    rms_win_sec=0.01,   # 每個 RMS 小窗長度，預設 0.01 秒
                                    k=0.5,              # CUSUM 的容忍偏移量
                                    h=5.0):             # CUSUM 警報門檻
    '''
    「窗內 CUSUM/burst」本來就不是設計來抓「穩定洩漏」的,它抓的是「窗內發生變化的那一刻」：也就是「洩漏正在發生的瞬間」
    【窗內 CUSUM】此處的 cusum 是「單一窗訊號內部」的變化偵測（前1/5當參考），產出 xxx_cusum_peak/alert_rate。
     與跨窗的 add_cusum_features（cusum_pos_z_xxx，用固定 baseline）完全不同，別混淆。
    
    這個函式會把一段 signal 切成很多小段，先算每小段的 RMS，形成一條 rms_series。
    然後再從這條 RMS 序列上算兩類特徵：
        1.CUSUM 類：看能量是不是持續往上偏，像慢慢累積的異常
        2.burst 類：看有沒有突然衝很高的爆發事件
    把原始波形轉成：「這段訊號在時間上是不是有持續異常、是不是常常爆發、爆發幾次」
    '''
    # 先建立一個空字典 feat，等等把所有計算出的特徵放進去
    feat = {}  
    # 把 RMS 小窗秒數轉成樣本點數
    win_len = int(rms_win_sec * fs)
    if win_len <= 0:    # 如果因為某些設定讓 win_len 算出來小於等於 0，就至少設成 1 點，避免後面出錯
        win_len = 1
    # 計算這段訊號總共可以切成幾個 RMS 小窗
    n_win = len(signal) // win_len
    
    # 如果這段訊號太短，連 5 個小窗都切不到，就認為資料太少，不適合算這些 temporal 特徵
    if n_win < 5:
        # 如果小窗太少，就直接回傳 0
        feat[f"{prefix}_cusum_peak"] = 0.0          # CUSUM 最大值
        feat[f"{prefix}_cusum_alert_rate"] = 0.0    # 超過警報門檻的比例
        feat[f"{prefix}_cusum_rise_count"] = 0      # CUSUM 警報事件出現幾次
        feat[f"{prefix}_burst_rate"] = 0.0          # burst 窗比例
        feat[f"{prefix}_burst_event_count"] = 0     # burst 事件次數
        return feat
    
    # 算 RMS 序列(每個時間小段的能量大小變化):把 signal 切成 n_win 個小窗 -> 每個小窗都算一次 RMS -> 最後組成一條 rms_series
    rms_series = np.array([
        np.sqrt(np.mean(signal[i * win_len:(i + 1) * win_len] ** 2))
        for i in range(n_win)
    ])
    
    # 取前面 1/5 的小窗數量，當作 baseline 參考區
    n_ref = max(1, n_win // 5)
    mu_c = np.mean(rms_series[:n_ref])     # 用前面參考區算:baseline RMS 平均
    sigma_c = max(np.std(rms_series[:n_ref]), 1e-9)    # 用前面參考區算:baseline RMS 標準差

    # 初始化 CUSUM 計算用的變數
    S = 0.0       # 目前累積分數
    s_trace = []  # 整條 CUSUM 軌跡
    alerts = []   # 每個窗有沒有超過警報門檻

    # 逐個 RMS 小窗往下掃
    for x in rms_series:
        # CUSUM 核心公式：持續偏高的能量累積, 而不是單一短暫尖峰
        S = max(0.0, S + (x - mu_c) / sigma_c - k)
        s_trace.append(S)    # 把這一刻的 CUSUM 值存下來
        alerts.append(S > h)    # 同時記錄這一窗是否超過警報門檻 h
    # 把 alerts 轉成布林陣列，方便後面運算
    alerts = np.array(alerts, dtype=bool)

    # 計算 burst 特徵
    med = np.median(rms_series)                           # RMS 序列的中位數
    mad = np.median(np.abs(rms_series - med)) * 1.4826    # median absolute deviation，中位數絕對偏差
    burst_thresh = med + 5 * mad                          # 設定 burst 門檻:如果某個 RMS 窗高到超過「中位數 + 5 倍 MAD」，就算是爆發窗
    burst_flags = rms_series > burst_thresh               # 逐窗判斷哪些窗屬於 burst

    # CUSUM 軌跡中的最大值:代表這段訊號裡，累積異常曾經衝多高
    feat[f"{prefix}_cusum_peak"] = float(np.max(s_trace))
    # 超過 CUSUM 門檻的窗比例，如果很多窗都在警報區，這個值就高
    feat[f"{prefix}_cusum_alert_rate"] = float(np.mean(alerts))
    # CUSUM 警報事件出現了幾次
    feat[f"{prefix}_cusum_rise_count"] = int(np.diff(np.r_[0, alerts.astype(int)]).clip(0).sum())
    # burst 窗比例:代表整段訊號裡，有多少比例的小窗屬於爆發狀態
    feat[f"{prefix}_burst_rate"] = float(np.mean(burst_flags))
    # burst 事件出現幾次
    feat[f"{prefix}_burst_event_count"] = int(np.diff(np.r_[0, burst_flags.astype(int)]).clip(0).sum())
    
    return feat


# =======================================
# 單一頻帶、單一通道特徵計算
# =======================================
def extract_band_features_one_channel(signal, 
                                      fs, 
                                      band, 
                                      prefix, 
                                      compute_env=True, 
                                      required_metrics=None,):     # 先把訊號濾到指定band,再用Hilbert算包絡
    """
    把一段訊號濾到指定頻帶後，算出這個頻帶的特徵，最後回傳成一個 dict
    required_metrics=None：維持原本完整特徵提取行為，供 Step2 與離線研究流程使用。
    required_metrics 有指定：只計算指定指標，供 Runtime 降低運算量。
    """
    from scipy.signal import hilbert
    
    # 決定要算哪些特徵
    all_metrics = {
        "E",
        "RMS",
        "Peak",
        "EnvMean",
        "EnvRMS",
        "EnvStd",
        "DomFreq",
        "Centroid",
        "Entropy",
        "Flatness",
        "ShapeCentroid",
        "ShapeSpread",
    }
    
    # 如果沒有指定就算全部 
    if required_metrics is None:
        metrics = set(all_metrics)
    else:
        metrics = {str(metric) for metric in required_metrics}
        unknown_metrics = (metrics - all_metrics)

        # 檢查有沒有不支援的特徵名稱
        if unknown_metrics:
            raise ValueError(f"不支援的 band metrics：{sorted(unknown_metrics)}")
    if not metrics:
        return {}

    #  Step1:先濾波到指定band
    # 原本的 signal 可能包含很多頻率成分,現在只保留 band["f_low"] ~ band["f_high"] 之間的成分
    # 同一組 band/channel 只濾波一次。
    x = bandpass_filter(signal, fs, band["f_low"], band["f_high"])

    features = {}

    # --------------------------------------------------------
    # 基礎振幅與能量特徵
    # --------------------------------------------------------
    if "E" in metrics:     # E 是能量
        features[f"{prefix}_E"] = float(np.mean(x ** 2))
    if "RMS" in metrics:   # RMS 是均方根振幅
        features[f"{prefix}_RMS"] = float(np.sqrt(np.mean(x ** 2)))
    if "Peak" in metrics:  # Peak 是最大瞬間振幅
        features[f"{prefix}_Peak"] = float(np.max(np.abs(x)))

    # --------------------------------------------------------
    # 包絡特徵：只有真的需要時才執行 Hilbert
    # --------------------------------------------------------
    env_metrics = {"EnvMean", "EnvRMS", "EnvStd"}
    needed_env_metrics = (metrics & env_metrics)

    if needed_env_metrics:
        if compute_env:
            env = np.abs(hilbert(x))
            # 包絡平均
            if "EnvMean" in needed_env_metrics:
                features[f"{prefix}_EnvMean"] = float(np.mean(env))
            # 包絡能量
            if "EnvRMS" in needed_env_metrics:
                features[f"{prefix}_EnvRMS"] = float(np.sqrt(np.mean(env ** 2)))
            # 包絡變動程度
            if "EnvStd" in needed_env_metrics:
                features[f"{prefix}_EnvStd"] = float(np.std(env))

        else:
            # 保留舊版 compute_env=False 的相容行為。
            for metric_name in needed_env_metrics:
                features[f"{prefix}_{metric_name}"] = 0.0

    # --------------------------------------------------------
    # 頻譜特徵：只有真的需要時才執行 Welch PSD
    # --------------------------------------------------------
    psd_metrics = {
        "DomFreq",
        "Centroid",
        "Entropy",
        "Flatness",
        "ShapeCentroid",
        "ShapeSpread",
    }

    needed_psd_metrics = (metrics & psd_metrics)
    # 如果需要頻譜特徵，才做 PSD
    if needed_psd_metrics:
        frequencies, power = compute_psd(x, fs, normalize=False, smooth_sigma=0,)
        # 只取目前 band 內的頻率
        mask = ((frequencies >= band["f_low"]) & (frequencies <= band["f_high"]))

        f_band = frequencies[mask]
        p_band = power[mask]
        total_power = float(np.sum(p_band))

        if (len(f_band) > 0 and total_power > 0):
            # 能量最大的主頻
            if "DomFreq" in needed_psd_metrics: 
                features[f"{prefix}_DomFreq"] = float(f_band[np.argmax(p_band)])
            # 頻譜重心
            if "Centroid" in needed_psd_metrics:
                features[f"{prefix}_Centroid"] = float(np.sum(f_band * p_band) / (total_power + 1e-12))
            # 頻譜分散程度
            if "Entropy" in needed_psd_metrics:
                normalized_power = (p_band / (total_power + 1e-12))
                features[f"{prefix}_Entropy"] = float(-np.sum(normalized_power * np.log(normalized_power + 1e-12)))
            # 頻譜平坦程度
            if "Flatness" in needed_psd_metrics:features[f"{prefix}_Flatness"] = float(
                    np.exp(np.mean(np.log(p_band + 1e-12))) / (np.mean(p_band) + 1e-12))
            
            if "ShapeCentroid" in needed_psd_metrics:
                p = p_band / (total_power + 1e-12)
                features[f"{prefix}_ShapeCentroid"] = float(np.sum(f_band * p))
            
            if "ShapeSpread" in needed_psd_metrics:
                p = p_band / (total_power + 1e-12)
                shape_centroid = float(np.sum(f_band * p))
                features[f"{prefix}_ShapeSpread"] = float(
                    np.sqrt(np.sum(((f_band - shape_centroid) ** 2) * p))
                )

        else:
            for metric_name in needed_psd_metrics:
                features[f"{prefix}_{metric_name}"] = 0.0

    return features


# ============================================================
# 檢查函式
# ============================================================
# selected_feature_names 反推回來的 raw_base_features，runtime parser 能不能解析
def validate_runtime_extractable_features(
    raw_base_features,
    band_config,
):
    """
    部署前檢查：
    確認模型選到的底層 raw features，都能被 runtime 的選擇性特徵提取器解析。

    這個函式只做名稱解析，不真的計算訊號特徵。
    """
    # runtime 認得的「統計指標」名稱（像 RMS、峰值、主頻等）
    supported_metrics = {
        "E", "RMS", "Peak",
        "EnvMean", "EnvRMS", "EnvStd",
        "DomFreq", "Centroid", "Entropy",
        "Flatness", "ShapeCentroid", "ShapeSpread",
    }

    # 一份是三個特殊的、寫死的固定特徵名稱
    supported_basic_time_features = {
        "ai0_env_kurtosis",
        "ai1_env_kurtosis",
        "diff_env_kurtosis",
    }
    
    # 把設定檔裡三種類型的頻帶（洩漏關注頻帶、共振頻帶、頻帶外）全部合併成一份清單，取出每個頻帶的名字。
    all_bands = (
        list(band_config.leak_focused_bands)
        + list(band_config.resonance_bands)
        + list(band_config.offband_bands)
    )
    band_names = sorted(
        [str(band["name"]) for band in all_bands],
        key=len,
        reverse=True,
    )

    unsupported = []

    # 大方向：這是整個函式的核心比對迴圈，逐一檢查模型要的每個特徵名稱，看看能不能透過「三種比對方式」的其中一種找到對應規則。
    for feature_name in dict.fromkeys(str(x) for x in raw_base_features):     # 去重複，同時保留原本的順序。
        matched = False     # 每個名字開始檢查前，先假設「還沒比對到」
        
        # 第一種比對：固定名稱
        if feature_name in supported_basic_time_features:    # 直接看這個名字是不是三個寫死的固定特徵之一，是的話直接算過關。
            matched = True

        # 第二種比對：_frac_core 結尾的特殊格式
        if feature_name.endswith("_frac_core"):
            # 如果名字是以 _frac_core 結尾，就用雙層迴圈把「頻帶名稱＋通道名稱」組合起來（例如 低頻帶_ai0_frac_core），看有沒有跟現在這個名字完全一樣。
            for band_name in band_names:
                for channel_name in ("ai0", "ai1", "diff"):
                    if feature_name == f"{band_name}_{channel_name}_frac_core":
                        matched = True
                        break
                if matched:
                    break

        # 第三種比對：一般的「頻帶+通道+統計指標」格式
        if not matched:    # 只有前兩種都沒比對到，才會進來跑這段（if not matched）
            for band_name in band_names:
                for channel_name in ("ai0", "ai1", "diff"):
                    # 把每個「頻帶名稱_通道名稱_」組成前綴，看這個特徵名是不是用這個前綴開頭。如果不是，continue 換下一個通道試試看。
                    prefix = f"{band_name}_{channel_name}_"
                    if not feature_name.startswith(prefix):
                        continue
                    
                    # 如果前綴對上了，就把前綴後面剩下的字串（metric_name）取出來，檢查它是不是在 supported_metrics 這份白名單裡，是的話才算真的比對到。
                    metric_name = feature_name[len(prefix):]
                    if metric_name in supported_metrics:
                        matched = True

                    break
                # 比對到就不用再試其他頻帶
                if matched:
                    break
        # 三種方式都試過還是沒比對到，就記錄成「不支援」
        if not matched:
            unsupported.append(feature_name)

    if unsupported:
        raise ValueError(
            "Runtime 無法產生下列模型 raw features，"
            "請補 runtime 特徵萃取器或重新選特徵："
            f"{sorted(unsupported)[:30]}"
        )

    return True


# ============================================================
# Runtime 專用的精簡版特徵提取器：只計算模型與規則真正需要的 Raw features
# ============================================================
def extract_required_window_features(
    records,                  # 一批 window 資料，每個 record 代表一個 window
    *,
    band_config,              # 頻帶設定，裡面包含 leak-focused、resonance、offband
    required_feature_names,   # 真正需要計算的 raw feature 名稱
    n_jobs=1,
    verbose=True,             # 控制要不要印出完成訊息
):
    """
    它會根據 required_feature_names，反推出需要哪些 band/channel/metric，然後對每個 window 只計算那些必要的 raw features。
    注意：
    required_feature_names 必須是已經反推出來的底層特徵，例如：

    - scan_093_465_475k_ai0_EnvMean
    - scan_094_470_480k_ai0_RMS
    - off_010_020k_ai0_E

    rel/z、CUSUM、Temporal 與 Summary 不在這裡計算，由 runtime_streaming.py 後續依序建立。
    """
    # 列出 Runtime 版本支援的底層特徵種類
    supported_metrics = {
        "E", "RMS", "Peak",
        "EnvMean", "EnvRMS", "EnvStd",
        "DomFreq", "Centroid", "Entropy",
        "Flatness", "ShapeCentroid", "ShapeSpread",
    }

    # 建立頻段查找表：合併所有正式頻帶，並避免同名頻帶重複，同名頻段只保留第一個（優先順序：leak_focused > resonance > offband）。
    all_bands = (
        list(band_config.leak_focused_bands)
        + list(band_config.resonance_bands)
        + list(band_config.offband_bands)
    )
    # 建立空字典，用來存 band 設定
    band_lookup = {}

    # 逐一把 band 放進 band_lookup, 如果有同名 band，只保留第一個(優先順序：leak_focused > resonance > offband)
    for band in all_bands:
        band_name = str(band["name"])

        if band_name not in band_lookup:
            band_lookup[band_name] = dict(band)

    # 依 band name 長度由長到短比對，避免名稱前綴相似時誤判。
    sorted_band_names = sorted(band_lookup, key=len, reverse=True,)

    # requests 格式：
    # {
    #     ("scan_xxx", "ai0"): {"RMS", "EnvMean"},
    #     ("off_xxx", "ai1"): {"E"},
    # }
    # 這樣同一個頻段+通道的多個指標只需要做一次帶通濾波，不重複計算。
    
    # 反推每個特徵需要哪個頻段、通道、指標
    requests = {}      # 儲存真正需要算的東西
    unsupported_features = []         # 存放無法解析的 feature 名稱
    basic_time_requests = set()       # 全頻時域特徵需求，例如 ai0_env_kurtosis，這類特徵不是某個 band 的特徵，所以不會進 requests。
    frac_core_requests = []           # frac_core 需求，例如 res_sec_167_187k_diff_frac_core，這類特徵需要先算 band energy，再用 core energy 當分母。

    # 逐一檢查每個 required feature
    for feature_name in dict.fromkeys(str(name) for name in required_feature_names):
        matched = False    # 預設目前這個 feature 還沒有成功解析
        
        # --------------------------------------------------------
        # 全頻時域特徵，例如 ai0_env_kurtosis
        # --------------------------------------------------------
        if feature_name in {
            "ai0_env_kurtosis",
            "ai1_env_kurtosis",
            "diff_env_kurtosis",
        }:
            channel_name = feature_name.replace("_env_kurtosis", "")
            basic_time_requests.add(channel_name)
            matched = True
            continue
        
        # --------------------------------------------------------
        # frac_core 特徵，例如 res_sec_167_187k_diff_frac_core
        # --------------------------------------------------------
        if feature_name.endswith("_frac_core"):
            for band_name in sorted_band_names:
                for channel_name in ("ai0", "ai1", "diff"):
                    expected_name = f"{band_name}_{channel_name}_frac_core"
        
                    if feature_name != expected_name:
                        continue
        
                    frac_core_requests.append(
                        (band_name, channel_name, feature_name)
                    )
        
                    # 先確保目標 band 的 energy 會被算出來。
                    requests.setdefault(
                        (band_name, channel_name),
                        set(),
                    ).add("E")
        
                    # frac_core 的分母是全部 leak-focused core bands 的能量，
                    # 所以同一個 channel 的所有 leak-focused band E 都要算。
                    for core_band in band_config.leak_focused_bands:
                        core_band_name = str(core_band["name"])
                        requests.setdefault(
                            (core_band_name, channel_name),
                            set(),
                        ).add("E")
        
                    matched = True
                    break
        
                if matched:
                    break
        
            if matched:
                continue
        
        # 嘗試用每個 band name、每個 channel 去比對 feature name
        for band_name in sorted_band_names:
            for channel_name in ("ai0", "ai1", "diff",):
                # 組出可能的 feature 前綴
                prefix = (
                    f"{band_name}_"
                    f"{channel_name}_"
                )
                
                # 如果 feature 名稱不是這個前綴開頭，就換下一個 band/channel
                if not feature_name.startswith(prefix):
                    continue
                # 把前綴拿掉，剩下的就是 metric
                metric_name = feature_name[len(prefix):]

                # metric 必須在 supported_metrics 裡（E / RMS / Peak / EnvMean 等）
                if metric_name not in supported_metrics:
                    # 如果 metric 不是支援項目，就記錄為不支援
                    unsupported_features.append(feature_name)
                    matched = True    # matched = True 的意思是：它有成功找到 band/channel，只是 metric 不合法
                    break
                # 建立 request key
                request_key = (band_name, channel_name)
                # 把這個 metric 加進 request
                requests.setdefault(request_key, set(),).add(metric_name)
                # 代表這個 feature 已經成功解析，不用繼續找其他 channel
                matched = True
                break
            # 跳出 band 迴圈，處理下一個 feature
            if matched:
                break
        # 如果所有 band/channel 都對不上，代表這個 feature 不是 Runtime raw feature 格式，就丟到 unsupported
        if not matched:
            unsupported_features.append(feature_name)
    # 如果有任何不能解析的 raw feature，直接報錯
    if unsupported_features:
        raise ValueError(f"Runtime 選擇性提取器無法解析下列 Raw features：{sorted(unsupported_features)[:30]}")

    # 確保 records 可以安全取得長度並重複走訪。
    records = list(records)
    
    
    # =========================
    # 單一 Window 特徵計算
    # =========================
    def extract_one_record(record):
        """
        計算單一 Window 所需的底層 Raw features。
        不在這裡更新 CUSUM／Temporal；因為這些時間特徵必須等全部 Window 計算完成並排序後，再依序更新。
        """
        # 取出這個 window 的訊號和取樣率
        signal_map = record["signal"]
        fs = int(record["fs"])
    
        row = {
            "filename": record.get("filename", ""),
            "record_id": record.get("record_id", ""),
            "t_start": record.get("t_start", np.nan),
            "t_end": record.get("t_end", np.nan),
            "t_ref": record.get("t_ref",record.get("t_start", np.nan)),      # t_ref 如果沒有，就用 t_start 當替代
            "label": record.get("label", -1),      # label 如果沒有，就用 -1
        }
        
        # --------------------------------------------------------
        # 全頻時域特徵
        # --------------------------------------------------------
        # 例如 ai0_env_kurtosis。
        # 這些特徵不是 bandpass 後的特徵，而是直接對整個 window 的 ai0/ai1/diff 訊號計算。
        for channel_name in basic_time_requests:
            if channel_name not in signal_map:
                raise ValueError(f"Window 缺少訊號通道：{channel_name}")
        
            basic_features = extract_basic_time_features(
                signal=signal_map[channel_name],
                prefix=channel_name,
                fs=fs,
            )
        
            row.update(basic_features)
                
            
        # 逐一處理前面整理好的 request
        for (band_name, channel_name,), required_metrics in requests.items():
            # 如果這個 window 沒有需要的通道，就直接報錯
            if channel_name not in signal_map:
                raise ValueError(f"Window 缺少訊號通道：{channel_name}")
            # 拿到這個 band 的頻率設定
            band = band_lookup[band_name]
            # 建立輸出欄位名稱前綴
            prefix = (
                f"{band_name}_"
                f"{channel_name}"
            )
            # 單一頻帶、單一通道特徵計算器:只針對 required_metrics 裡面指定的特徵做 bandpass filter / E / RMS / Env / PSD 特徵計算
            band_features = (
                extract_band_features_one_channel(
                    signal=signal_map[channel_name],
                    fs=fs,
                    band=band,
                    prefix=prefix,
                    compute_env=True,
                    required_metrics=(required_metrics),
                )
            )
            # 把算出來的 feature 加進這個 window 的 row
            row.update(band_features)
            
        # --------------------------------------------------------
        # frac_core 特徵
        # --------------------------------------------------------
        # 例如 res_sec_167_187k_diff_frac_core。
        # 定義：該 band 的 E / 所有 leak-focused core bands 的 E 總和。
        core_energy_by_channel = {}
    
        for channel_name in ("ai0", "ai1", "diff"):
            core_energy_cols = [
                f"{str(core_band['name'])}_{channel_name}_E"
                for core_band in band_config.leak_focused_bands
                if f"{str(core_band['name'])}_{channel_name}_E" in row
            ]
    
            core_energy_by_channel[channel_name] = sum(
                float(row[col]) for col in core_energy_cols
            )
    
        for band_name, channel_name, feature_name in frac_core_requests:
            energy_col = f"{band_name}_{channel_name}_E"
            denom = core_energy_by_channel.get(channel_name, 0.0)
    
            row[feature_name] = (
                float(row.get(energy_col, 0.0)) / (denom + 1e-12)
            )
        
        # 每次呼叫只回傳這一個 Window 的結果。
        return row


    # ============================================================
    # 單執行緒或多執行緒計算
    # ============================================================
    # 如果只用單執行緒，或 records 只有一筆，就直接逐筆計算
    if (int(n_jobs) == 1 or len(records) <= 1):
        feature_rows = [extract_one_record(record) for record in records]
    # 如果 n_jobs > 1，就用多執行緒計算
    else:
        # 這裡用 threads 而不是 processes，是為了避免大量 raw signal 被複製到不同 process，記憶體比較省
        feature_rows = Parallel(             # Parallel 會維持 records 原本的輸入順序。
            n_jobs=int(n_jobs),
            prefer="threads",)(delayed(extract_one_record)(record) for record in records)
    
    # 如果 verbose=True，就印出完成訊息
    if verbose:
        print(f"✅ Runtime 選擇性特徵完成：{len(feature_rows)} Windows，{len(requests)} 組 band/channel")
    
    return feature_rows


# ==========================
# 頻域濾波
# ==========================
# ** 只負責「建立濾波器係數」 **
# 做 bandpass filter 的快取，目的是加速
# 根據 order、lowcut、highcut、fs 建立一個穩定的 bandpass filter
@lru_cache(maxsize=256)     # 這行是快取裝飾器。意思是：如果同樣的參數以前算過，就不要重算，直接拿之前的結果
def _cached_sos(order,      # 濾波器階數
                lowcut_i,   # 低截止頻率
                highcut_i,  # 高截止頻率
                fs_i):      # sampling rate / 取樣率
    # 計算 Nyquist frequency
    nyq = 0.5 * fs_i
    # [lowcut_i / nyq, highcut_i / nyq] 是正規化後的頻率範圍
    return butter(order, [lowcut_i/nyq, highcut_i/nyq], btype='band', output="sos")

# ** 真正執行濾波 **
# 把訊號濾到指定頻帶
def bandpass_filter(signal, fs, lowcut= 200_000, highcut= 450_000, order=4):     # 雖然函式預設值是 200k~450k，但在你的特徵工程裡實際呼叫時，會把 band["f_low"] 和 band["f_high"] 傳進來，所以不會真的固定只濾 200–450k。
    signal = np.asarray(signal, dtype=np.float64)    

    nyq = 0.5 * fs    # 若fs=1_000_000,最高安全分析道500kHz
    # 設定 Nyquist 保護,避免高截止頻率碰到 Nyquist
    highcut = min(highcut, nyq * 0.999) 
    
    # 避免頻率設定不合法時直接炸掉
    if lowcut <= 0 or highcut <= 0 or lowcut >= highcut:
        return signal
    
    # 避免 window 太短時 sosfiltfilt 不穩或報錯
    if len(signal) < order * 30:
        return signal
    
    # 建立一個指定頻帶的濾波器設定，準備拿來濾你的訊號
    sos = _cached_sos(order, int(lowcut), int(highcut), int(fs))
    
    # 如果濾波失敗，就回傳原訊號，讓流程不中斷。
    try:
        return sosfiltfilt(sos, signal)
    except Exception:
        return signal


# ==========================
# 統一 PSD 計算（獨立函式）：可決定要原始訊號 or 標準化 or 平滑smooth
# ==========================
def compute_psd(signal, fs, nperseg=NPERSEG, normalize=False, smooth_sigma=0):
    from scipy.signal import welch
    from scipy.ndimage import gaussian_filter1d
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg, detrend="constant")    # 對訊號做 Welch PSD
    # 如果指定標準化，就把最大值縮成 1
    if normalize:
        Pxx = Pxx / (np.max(Pxx) + 1e-12)
    # 如果指定平滑，就對頻譜做 Gaussian 平滑
    if smooth_sigma > 0:
        Pxx = gaussian_filter1d(Pxx, sigma=smooth_sigma)

    return f, Pxx


# 計算頻帶能量：某一段頻帶裡的平均能量
def compute_band_energy(signal, fs, band):
    """
    計算指定頻帶的內的平均能量
    需要你前面已經有 bandpass_filter()，找出之主要關鍵頻段
    """
    x = bandpass_filter(signal, fs, band["f_low"], band["f_high"])
    return float(np.mean(x ** 2))


# ============================================================
# offband、洩漏頻帶、共振頻帶的摘要與比值
# ============================================================
def build_offband_summary_features(df, band_config):
    """
    建立 offband / leak-focused band / resonance band 的摘要特徵，方便後續做 disturbance veto 與規則判斷。

    目前 leak-focused 與 resonance 先分開保留：
    - leak-focused：正式主判斷頻帶
    - resonance：先保留觀察，之後再決定是否納入主規則
    
    offband：拿來看是不是寬頻干擾
    """
    eps = 1e-12    # 做比值時避免分母剛好是 0
    
    # 從特徵表（df）的欄位裡，找出指定頻段的所有能量特徵欄位（_E 結尾的）
    def collect_energy_cols(bands):
        # 取出頻段名稱
        names = [b["name"] for b in bands]
        # 搜尋對應的能量欄位
        cols = []
        for name in names:
            cols.extend([
                c for c in df.columns
                if c.startswith(f"{name}_") and c.endswith("_E")    # 欄位名稱以這個頻段名稱開頭並以 _E 結尾（代表能量特徵）
            ])
        return cols

    # 找出 offband 頻帶的能量欄位
    offband_e_cols = collect_energy_cols(band_config.offband_bands)
    # 找出洩漏頻帶的能量欄位
    leak_focus_e_cols = collect_energy_cols(band_config.leak_focused_bands)
    # 找出 resonance 頻帶的能量欄位
    resonance_e_cols = collect_energy_cols(band_config.resonance_bands)

    # 建立 offband 摘要
    if offband_e_cols:
        df["offband_E_mean"] = df[offband_e_cols].mean(axis=1)      # 這一列所有 offband 能量的平均
        df["offband_E_max"] = df[offband_e_cols].max(axis=1)        # 這一列所有 offband 能量的最大值

    # 建立 leak-focused 摘要
    if leak_focus_e_cols:
        df["leak_focus_E_mean"] = df[leak_focus_e_cols].mean(axis=1)     # 這些主洩漏頻帶平均多高
        df["leak_focus_E_max"] = df[leak_focus_e_cols].max(axis=1)       # 這些主洩漏頻帶中最高的是哪一個、多高

    # 建立 resonance 摘要
    if resonance_e_cols:
        df["resonance_E_mean"] = df[resonance_e_cols].mean(axis=1)
        df["resonance_E_max"] = df[resonance_e_cols].max(axis=1)

    # offband vs leak-focused：正式主規則先用這組
    if offband_e_cols and leak_focus_e_cols:
        # 整體 offband 平均能量 / 洩漏主頻帶平均能量
        # 如果值很高，通常代表：不是只有洩漏頻帶高，而是 offband 也一起很高，那就比較像干擾、碰撞、寬頻雜訊。
        df["offband_to_leak_focus_E_ratio"] = (df["offband_E_mean"] / (df["leak_focus_E_mean"] + eps))
        
        # offband 最大能量 / 洩漏主頻帶最大能量
        df["offbandmax_to_leak_focusmax_E_ratio"] = (df["offband_E_max"] / (df["leak_focus_E_max"] + eps))

    # offband vs resonance：先保留觀察
    # 看某些干擾是不是其實把 offband 拉得比共振還高或看共振是不是只是背景結構，不一定是主要判斷依據
    if offband_e_cols and resonance_e_cols:
        # 如果這個值很高，代表：offband 整體比 resonance 還明顯, 這可能表示事件不是單純共振，而是更寬頻的干擾
        df["offband_to_resonance_E_ratio"] = (df["offband_E_mean"] / (df["resonance_E_mean"] + eps))
        
        # 如果這個值很高，代表：某個 offband 爆得比 resonance 裡最強的頻帶還高, 這也會偏向寬頻干擾或異常碰撞。
        df["offbandmax_to_resonancemax_E_ratio"] = (df["offband_E_max"] / (df["resonance_E_max"] + eps))

    return df


# ============================================================
# CUSUM:幫每個 record_id 的時間序列特徵加上 CUSUM 累積變化特徵
# ============================================================
def add_cusum_features(
    df,
    input_feature_cols,
    k=0.02,          # k=0.0 太容易累積噪聲,k 是容忍值，用來過濾小波動，避免噪聲一直被累積,微小洩漏建議先從 0.02~0.05 試
    decay=0.97,      # 保留前面累積，不要一下子歸零
):
    '''
    不是只看單一 window 當下的值，而是看：這個特徵有沒有在連續幾個 window 裡持續偏高？
    對洩漏偵測很有用，因為真正 leak 通常不是一個瞬間 spike，而是會在 onset 之後持續存在
    
    本身 沒有重新建立 baseline。
    它吃進來的 input_feature_cols 通常應該已經是前面處理好的特徵, 也就是已經用固定 baseline 校正過的結果
    '''
    out = df.copy()
    # 先依照個體和時間排序，確保累積順序正確
    out = out.sort_values(["record_id", "t_ref"]).reset_index(drop=True)
    
    # 對 input_feature_cols 中的每個特徵，先建立兩個新欄位
    for feat in input_feature_cols:
        out[f"cusum_pos_{feat}"] = np.nan    # 累積正向變化(通常比較會用 cusum_pos，因為 leak 常常是特徵變高)
        out[f"cusum_neg_{feat}"] = np.nan    # 累積負向變化
        
    # 每個 record_id 分開算(不同檔案/不同 session 不會互相累積，避免資料串在一起)
    for record_id, sub in out.groupby("record_id", sort=False):
        idx = sub.index     # 記住這個 record 在原表中的 index，後面要把結果寫回去
    
        # 取出這個特徵在同一個 record 裡的時間序列
        for feat in input_feature_cols:
            x = sub[feat].fillna(0.0).values      # 把缺值補成 0，轉成陣列方便運算
            # 建立兩條累積序列
            s_pos = np.zeros(len(x), dtype=float)   # 記錄正向與負向的累積量
            s_neg = np.zeros(len(x), dtype=float)
    
            # 從第 2 個 window 開始算，因為第 1 個 window 沒有前面可累積
            for i in range(1, len(x)):
                pos_inc = max(0.0, x[i] - k)    # 如果目前值 x[i] 比容忍值 k 還大，就把超過的部分拿來累積
                neg_inc = max(0.0, -x[i] - k)
                # 累積更新
                s_pos[i] = decay * s_pos[i - 1] + pos_inc    # 目前正向累積 = 前一刻累積量 * decay + 這一刻新增正向量
                s_neg[i] = decay * s_neg[i - 1] + neg_inc
            
            # 把算好的 CUSUM 結果寫回原本資料表對應的位置
            out.loc[idx, f"cusum_pos_{feat}"] = s_pos
            out.loc[idx, f"cusum_neg_{feat}"] = s_neg
    
    return out


# ============================================================
# A. window-to-window dynamic features
# ============================================================
def add_window_dynamic_features(
    df,                        # 原始特徵表
    feature_cols,              # 要拿來做時間變化特徵的欄位
    group_col="record_id",     # 每一筆檔案或 session 的分組欄位
    time_col="t_ref",          # 時間排序欄位
    prev_windows=(3, 5),       # 要看前 3 窗、前 5 窗的平均
):
    """
    建立 window-to-window 動態特徵：
    1. delta_*：當前窗相對前一窗的變化量
    2. delta_prev{k}mean_*：當前窗相對前 k 窗平均的偏移量
    """
    # 先複製與排序
    out = df.copy()
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)

    # 過濾可用欄位:挑真正可以做動態特徵的欄位(欄位真的存在於 df 裡並且為數值型態)
    valid_cols = [
        c for c in feature_cols
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]

    # 逐筆 record 處理(每一個 record_id 各自獨立做)
    for _, sub in out.groupby(group_col, sort=False):
        idx = sub.index
        
        # 逐欄位做動態特徵
        for col in valid_cols:
            x = pd.to_numeric(sub[col], errors="coerce")
            
            # (a) 建立 delta_ 特徵:「這一窗是不是突然跳高 / 跳低」
            out.loc[idx, f"delta_{col}"] = x.diff()

            # (b) 建立 delta_prev{k}mean_ 特徵:當前窗跟「前 k 窗平均」差多少
            for k in prev_windows:
                prev_mean = x.shift(1).rolling(k, min_periods=1).mean()
                out.loc[idx, f"delta_prev{k}mean_{col}"] = x - prev_mean

    return out


# ============================================================
# B. 局部時間特徵：local normalization / rolling trend features
# ============================================================
def add_local_rolling_features(
    df,                      # 原始特徵表，每一列通常是一個 window
    feature_cols,            # 做 temporal 特徵的欄位
    group_col="record_id",   # 同一筆資料的分組欄位
    time_col="t_ref",        # 時間排序依據
    windows=(3, 5),          # 表示要看前 3 窗、前 5 窗兩種局部尺度
):
    """
    對每個 record_id，沿著時間軸，幫指定欄位建立 rolling temporal 特徵：
    1. local_rel{k}_*：相對前 k 窗平均的偏移
    2. local_z{k}_*：相對前 k 窗平均的 z-score
    3. rolling_std{k}_*：局部波動程度
    4. rolling_slope{k}_*：局部上升 / 下降趨勢
    5. rolling_max_gap{k}_*：相對最近局部最大值的差距
    """
    # 複製並排序
    out = df.copy()
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)

    # 過濾可用欄位：這個欄位名稱真的存在，而且它是數值型
    valid_cols = [
        c for c in feature_cols
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]

    # 逐筆 record 處理
    for _, sub in out.groupby(group_col, sort=False):
        idx = sub.index

        # 逐欄位做 temporal 特徵
        for col in valid_cols:
            x = pd.to_numeric(sub[col], errors="coerce")

            # 對每個局部窗長 k 去算
            for k in windows:
                prev = x.shift(1)    # 整條序列往後移一格, 目的：只看前面的窗，不把當前窗自己算進去
                prev_mean = prev.rolling(k, min_periods=1).mean()       # 前 k 個窗的平均
                prev_std = prev.rolling(k, min_periods=2).std(ddof=1)   # 前 k 個窗的標準差
                rolling_std = x.rolling(k, min_periods=2).std(ddof=1)   # 最近 k 窗本身的波動程度
                rolling_max = x.rolling(k, min_periods=1).max()         # 最近 k 窗的最大值

                # 真正建立哪些新特徵
                out.loc[idx, f"local_rel{k}_{col}"] = x - prev_mean     # 目前值 - 前 k 窗平均
                out.loc[idx, f"local_z{k}_{col}"] = (x - prev_mean) / (prev_std + 1e-12)    # 做局部 z-score, 這一窗比前面幾窗平均高了多少個局部標準差
                out.loc[idx, f"rolling_std{k}_{col}"] = rolling_std            # 最近 k 窗的波動程度
                out.loc[idx, f"rolling_max_gap{k}_{col}"] = x - rolling_max    # 現在距離最近局部高點差多少

                # 最近 k 窗的趨勢斜率:斜率 > 0：最近在上升
                slope = x.rolling(k, min_periods=2).apply(
                    lambda arr: np.polyfit(np.arange(len(arr)), arr, 1)[0]
                    if np.isfinite(arr).sum() >= 2 else np.nan,
                    raw=True,
                )
                out.loc[idx, f"rolling_slope{k}_{col}"] = slope

    return out


#                                               ****************************************************************
#                                                                 共用程式碼：檢查特徵欄位程式碼
#                                               ****************************************************************
# =============================================
# 檢查原始 band 能量欄位有沒有齊全
# =============================================
def audit_band_feature_coverage(df,                                       # 要檢查的特徵表 
                                band_config,                              # BandConfig（裡面有 leak_focused_bands、resonance_bands、offband_bands 這些頻段清單）
                                sensor_suffixes=("_ai0_E", "_ai1_E")):    # 感測器後綴，預設是雙感測器
    """
    檢查 band_config 裡每一個 band,在每一種感測器後綴下,對應的原始能量欄位是不是真的存在於 df 裡。
    """
    # 把band_config裡三種 band 清單合併成一份完整清單all_bands
    all_bands = (
        list(band_config.final_main_bands)
        + list(band_config.offband_bands)
    )
        
    # 依 band name 去重, 避免同一個 band 名稱如果不小心出現在多個清單裡
    seen = set()
    unique_bands = []
    for b in all_bands:
        name = b["name"]
        if name in seen:
            continue
        seen.add(name)
        unique_bands.append(b)

    # 對每一個去重後的 band b，再對每一種感測器後綴 suf，組出理論上應該存在的欄位名稱
    expected = [f'{b["name"]}{suf}' for b in unique_bands for suf in sensor_suffixes]
    # 再用一個 list comprehension，掃過 expected 裡的每個欄位名，只留下「不在 df.columns 裡」的那些，存進 missing
    missing = [c for c in expected if c not in df.columns]

    print(f"理論上應該有 {len(expected)} 個原始 band 能量欄位")
    print(f"實際缺少 {len(missing)} 個:", missing[:20])
    return missing      # 回傳完整的缺漏清單


# =============================================
# 檢查 offband 摘要/比值欄位有沒有被建立
# =============================================
def audit_offband_summary_coverage(df):
    # 把「應該要有的摘要/比值欄位」硬編碼成一份固定清單
    summary_cols = [
        "offband_E_mean", "offband_E_max",
        "leak_focus_E_mean", "leak_focus_E_max",
        "resonance_E_mean", "resonance_E_max",
        "offband_to_leak_focus_E_ratio", "offbandmax_to_leak_focusmax_E_ratio",
        "offband_to_resonance_E_ratio", "offbandmax_to_resonancemax_E_ratio",
    ]
    # 掃過固定清單，抓出「不在 df.columns 裡」的特徵欄位
    missing = [c for c in summary_cols if c not in df.columns]
    print("缺少的摘要/比值欄位:", missing if missing else "無,全部都在")
    
    return missing     # 回傳缺漏清單（可能是空的，代表全部都在）


# =============================================
# 檢查每個原始特徵,衍生特徵鏈有沒有跑完整
# =============================================
def audit_derived_feature_coverage(df,               # 特徵表
                                   raw_base_cols):   # 要檢查的原始特徵欄位清單
    """
    針對每個原始特徵欄位,檢查 rel_/z_/cusum_/delta_/local_/rolling_
    這些衍生版本是不是都建立出來了。
    """
    # 對每一個傳進來的原始特徵欄位c,套用 17 種衍生前綴樣板,組出理論上應該存在的衍生欄位名稱
    templates = [
        "rel_{c}", "z_{c}",
        "cusum_pos_z_{c}", "cusum_neg_z_{c}",
        "delta_{c}", "delta_prev3mean_{c}", "delta_prev5mean_{c}",
        "local_rel3_{c}", "local_rel5_{c}",
        "local_z3_{c}", "local_z5_{c}",
        "rolling_std3_{c}", "rolling_std5_{c}",
        "rolling_slope3_{c}", "rolling_slope5_{c}",
        "rolling_max_gap3_{c}", "rolling_max_gap5_{c}",
    ]

    # 先建立一個空字典，因為這支函式要記錄的是「每個原始欄位分別缺了哪些衍生版本」
    missing = {}
    # 逐一走過每個原始欄位 c, 檢查這個組出來的欄位名是否「不在 df.columns 裡」，把缺的存進 gaps
    for c in raw_base_cols:
        gaps = [tmpl.format(c=c) for tmpl in templates if tmpl.format(c=c) not in df.columns]
        if gaps:
            missing[c] = gaps

    print(f"檢查 {len(raw_base_cols)} 個原始欄位,{len(missing)} 個有缺漏")
    # 逐一印出「哪個原始欄位」缺了「哪些衍生欄位」
    for c, gaps in list(missing.items())[:5]:
        print(f"  {c} 缺少: {gaps}")
    return missing


# =============================================
# Step3 / Step4 / runtime 輸出的表，有沒有漏掉 Step2 最終選定特徵？
# =============================================
def audit_selected_feature_alignment(df, feature_config):
    # 直接從 feature_config_micro.json 讀出selected_feature_names,檢查這些 Step2 選定的正式特徵有沒有全部出現在df裡
    selected = feature_config.get("selected_feature_names", [])
    missing = [c for c in selected if c not in df.columns]

    print(f"Step2 selected features 應有 {len(selected)} 個")
    print(f"實際缺少 {len(missing)} 個:", missing if missing else "無, 全部齊全")
    return missing


# =============================================
# 模型特徵都有了，但 runtime / disturbance rule 需要的 offband 規則欄位有沒有漏？
# =============================================
def audit_rule_feature_alignment(df, required_rule_cols):
    missing = [c for c in required_rule_cols if c not in df.columns]
    print(f"規則欄位應有 {len(required_rule_cols)} 個")
    print(f"缺少 {len(missing)} 個:", missing if missing else "無, 全部齊全")
    return missing


# ============================================================
# 驗證某個特徵工程階段需要的欄位是否完整
# ============================================================
def validate_feature_stage_requirements(
    df,
    required_columns,
    *,
    stage_name,                 # 驗證哪個階段
    check_all_invalid=True,     # 是否要做「完全沒有有效值」這一關檢查，預設開啟
):
    """
    驗證某個特徵工程階段的必要欄位。

    檢查：
    1. DataFrame 是否有重複欄名。
    2. execution plan 的必要欄位是否存在。
    3. 必要欄位是否完全沒有有效數值。
    """
    # 先把每個欄位名轉成字串（防止型別不一致），再用 dict.fromkeys(...) 去重、同時保留原始順序，最後轉回 list。
    required_columns = list(dict.fromkeys(str(column) for column in required_columns))
    # 這份 df 裡有哪些欄位名稱是重複的清單
    duplicate_columns = (df.columns[df.columns.duplicated()].unique().tolist())

    # 如果真的有重複欄名，直接丟出 ValueError
    if duplicate_columns:
        raise ValueError(f"{stage_name} 出現重複欄名：{duplicate_columns}")

    # 用 list comprehension 一次抓出所有「該有但不在 df.columns 裡」的必要欄位，缺了就把完整清單丟進錯誤訊息
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{stage_name} 缺少必要特徵：{missing_columns}")

    # 建立一個空清單，準備裝「型別檢查有過、但整欄數值全是無效值」的欄位名稱。
    invalid_columns = []

    # 檢查是可選的:對每個欄位做轉型
    if check_all_invalid:
        # 逐一走過每個必要欄位。把這一欄強制轉成數值型別，任何轉不成數字的值都會變成 NaN
        for column in required_columns:
            values = (pd.to_numeric(df[column],errors="coerce",).replace([np.inf, -np.inf],np.nan,))
            # 標記每一列是不是「非缺失值」；.sum() 把 True 當 1、False 當 0 加總，得到「這一欄總共有幾個有效值」
            if values.notna().sum() == 0:
                invalid_columns.append(column)
    # 如果有任何欄位落入這個「全部無效」的狀況，丟出錯誤,把完整清單列出來
    if invalid_columns:
        raise ValueError(f"{stage_name} 以下特徵完全沒有有效數值：{invalid_columns}")

    print(f"✅ {stage_name}：{len(required_columns)} 個必要特徵全部存在")

    # 回報結果，順便留一份這次驗證的完整記錄
    return {
        "stage_name": stage_name,
        "required_count": len(required_columns),
        "missing_columns": missing_columns,
        "invalid_columns": invalid_columns,
        "duplicate_columns": duplicate_columns,
    }


#                                               ****************************************************************
#                                                                固定 baseline 與特徵設定契約工具
#                                               ****************************************************************
# ============================================================
# 固定 baseline 與特徵設定契約工具
# ============================================================
def calculate_file_sha256(path):      # 接收一個參數 path，可以是字串路徑也可以是 Path 物件
    '''
    把一個檔案的內容變成一組獨一無二的「指紋」- 這份設定檔的內容有沒有變，之後只要重新對同一個檔案算一次，指紋不一樣就代表內容變了
    
    主要用來鎖住這幾種東西：
     - band_config_step2.json
     - feature_extraction_config.json
     - feature_config_micro.json
     - site_baseline.json
     
    SHA256 指紋 = 設定檔版本鎖
    它保護你不要發生：
     - baseline 是用舊頻帶算的
     - 但推論是用新頻帶跑的
    
     - 模型是用舊特徵工程訓練的
     - 但 inference 是用新特徵工程產生的
    '''
    # 轉成路徑格式
    path = Path(path)

    # 檢查這個路徑指到的檔案是否真的存在於硬碟上
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")    # 如果檔案不存在，主動丟出一個 FileNotFoundError，並用 f-string 把實際找不到的路徑塞進錯誤訊息裡

    return hashlib.sha256(path.read_bytes()).hexdigest()     


# ============================================================
# 檢查「算特徵之前的規格」
# ============================================================
def load_feature_extraction_config(config_path):  
    '''
    讀取 feature_extraction_config.json，並確認它是一份合法、完整、可用的特徵萃取設定檔。
    '''
    # config_path——上游特徵萃取設定檔的路徑, 統一轉成 Path 物件
    config_path = Path(config_path)

    # 先確認檔案真的存在，不存在就主動丟出明確的 Error
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 feature extraction config：{config_path}")
    # 把整個檔案讀成一個字串，再把這個字串解析成 Python 的 dict，存進 payload
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    # 檢查「這份檔案是不是真的是 feature extraction config」——防止不小心把別種設定檔的路徑傳進來
    if payload.get("schema_name") != "feature_extraction_config":
        raise ValueError("schema_name 必須是 feature_extraction_config")

    # 列出這份設定檔一定要有的欄位
    required_keys = [
        "schema_version", "base_feature_version",
        "fs", "window_sec",
        "hop_sec", "nperseg",
        "max_points", "include_offband_summary",
    ]

    # 掃過 required_keys 裡的每一個欄位名，只留下「不在 payload 裡」的那些，存進 missing_keys
    missing_keys = [key for key in required_keys if key not in payload]

    # 如果 missing_keys 不是空清單（也就是至少缺一個），就把整份缺失清單一次列在錯誤訊息裡
    if missing_keys:
        raise ValueError(f"feature_extraction_config 缺少欄位：{missing_keys}")
    # 檢查 schema_version 版本
    if int(payload["schema_version"]) != 1:
        raise ValueError("目前只支援 feature_extraction_config schema_version=1")
    # 檢查取樣率，必須為正
    if float(payload["fs"]) <= 0:
        raise ValueError("fs 必須大於 0")
    # 檢查視窗長度 window_sec，必須為正
    if float(payload["window_sec"]) <= 0:
        raise ValueError("window_sec 必須大於 0")
    # 檢查窗移 hop_sec，必須為正
    if float(payload["hop_sec"]) <= 0:
        raise ValueError("hop_sec 必須大於 0")
    # 檢查：hop_sec 不可以大於 window_sec
    if (float(payload["hop_sec"]) > float(payload["window_sec"])):
        raise ValueError("hop_sec 不可大於 window_sec")
    # 檢查 nperseg（算頻譜時每一段的點數，影響頻率解析度），必須為正
    if int(payload["nperseg"]) <= 0:
        raise ValueError("nperseg 必須大於 0")
    # 檢查 max_points（訊號點數上限），必須為正
    if int(payload["max_points"]) <= 0:
        raise ValueError("max_points 必須大於 0")
    # include_offband_summary 是一個開關（要不要計算 offband 摘要特徵）
    if not isinstance(payload["include_offband_summary"], bool,):
        raise ValueError("include_offband_summary 必須是 JSON boolean")

    return payload


# ============================================================
# 載入已經建立好的 baseline(site_baseline.json),並確認設定一致
# ============================================================
def load_fixed_normal_baseline(
    baseline_path,                      # 固定 baseline 檔案路徑
    *,                                  # 強制後面三個參數都只能用關鍵字傳入
    band_config_path,                   # 目前正在用的 band 設定檔路徑
    feature_extraction_config_path,     # 前正在用的特徵萃取設定檔路徑
    require_final=False,                # 是否要求 baseline 一定要是正式版，預設 False
):
    '''
    讀取固定正常 baseline 檔案，並確認它跟目前這次特徵工程使用的設定完全一致
    '''
    # Path 化、確認檔案存在，不存在就丟出明確的 FileNotFoundError
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        raise FileNotFoundError(f"找不到固定 baseline：{baseline_path}")

    # 讀檔＋解析 JSON，存進 payload
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    
    # 確認這份檔案真的是「site_baseline」，不是誤傳其他種設定檔
    if payload.get("schema_name") != "site_baseline":
        raise ValueError("baseline schema_name 必須是 site_baseline")
    # 把 schema_status 取出來存成一個變數, 供後面確認 baseline 狀態合法
    schema_status = payload.get("schema_status")

    # 確認 schema_status 至少是這兩個合法值其中之一（不能是空的、拼錯字，或亂填其他字串）
    if schema_status not in {"candidate", "final"}:
        raise ValueError("baseline schema_status 必須是 candidate 或 final")
        
    # require_final=False（預設值）的情境下，candidate 版的 baseline 也可以讀；但如果呼叫端明確要求 require_final=True（例如正式跑 inference 的流程），還在候選階段的 baseline 就會被擋掉，逼你只能用已經定案的版本。
    if require_final and schema_status != "final":
        raise ValueError("正式執行只能使用 schema_status=final 的 baseline")
        
    # 取出 feature_stats 這個欄位
    feature_stats = payload.get("feature_stats")

    # 檢查 feature_stats 的型別必須是 dict
    if not isinstance(feature_stats, dict):
        raise ValueError("baseline.feature_stats 必須是 dict")

    # 檢查 feature_stats 不是空的
    if not feature_stats:
        raise ValueError("baseline.feature_stats 不可為空")
        
    # 分別對「現在」的 band 設定檔跟特徵萃取設定檔各算出一組指紋
    current_band_hash = calculate_file_sha256(band_config_path)
    current_extraction_hash = calculate_file_sha256(feature_extraction_config_path)

    # 拿「現在」算出的 band 設定檔指紋，去跟這份 baseline JSON 裡「當初建立 baseline 時記錄下來的」指紋比對。只要對不上，就代表 band 定義已經跟建立 baseline 當時不一樣
    if (payload.get("band_config_sha256") != current_band_hash):
        raise ValueError("baseline 與目前 band_config_step2.json 不一致，請重新建立 baseline")
    # 換成比對特徵萃取設定檔的指紋
    if (payload.get("feature_extraction_config_sha256") != current_extraction_hash):
        raise ValueError("baseline 與目前 feature_extraction_config 不一致，請重新建立 baseline")

    return payload


# ================================
# 用正常的資料當 baseline，對特徵做校正
# ================================
def apply_fixed_normal_baseline(
    df,
    feature_cols,
    baseline_payload,
):
    '''
    使用已經算好的固定 baseline 統計數值，把目前 df 的 raw feature 轉成 rel_* / z_*。
    '''
    # 複製，不動原始資料
    out = df.copy()

    # 特徵欄位清單去重、統一成字串
    feature_cols = list(dict.fromkeys(     # dict.fromkeys(...) 是 Python 裡一個常見的「去重但保留原始順序」的技巧
        str(column) for column in feature_cols     # 先把每個欄位名稱都轉成字串，防止外面不小心傳進非字串型別的欄位名
    ))

    # 檢查這些特徵欄位在資料表裡真的存在
    missing_raw_columns = [feature_name for feature_name in feature_cols if feature_name not in out.columns]
    if missing_raw_columns:
        raise ValueError(f"特徵表缺少 baseline 前置 raw 特徵：{missing_raw_columns}")

    # 檢查 baseline 裡有沒有涵蓋這些特徵的統計量
    feature_stats = baseline_payload.get("feature_stats", {},)
    missing_baseline_features = [feature_name for feature_name in feature_cols if feature_name not in feature_stats]

    if missing_baseline_features:
        raise ValueError(f"固定 baseline 缺少必要特徵：{missing_baseline_features}")

    # 逐一特徵做校正，過程中對 mean/std 本身也做防呆
    for feature_name in feature_cols:
        stat = feature_stats[feature_name]
        try:
            mean_value = float(stat["mean"])
            std_value = float(stat["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{feature_name} 的 baseline mean/std 格式錯誤") from exc

        if not np.isfinite(mean_value):
            raise ValueError(f"{feature_name} 的 baseline mean 無效")

        if (not np.isfinite(std_value) or std_value <= 0):
            raise ValueError(f"{feature_name} 的 baseline std 無效")

        # 真正計算 rel_* 跟 z_*
        values = pd.to_numeric(out[feature_name],errors="coerce",)
        out[f"rel_{feature_name}"] = (values - mean_value)               # rel_* 是原始值減掉 baseline 的平均值（絕對偏移量,單位跟原始特徵一樣）
        out[f"z_{feature_name}"] = (values - mean_value) / std_value     # z_* 是再除以 baseline 的標準差（標準化偏移量,單位變成「偏離幾個標準差」
   
    # 最後再驗一次，確保衍生欄位真的都建出來了
    expected_features = [
        derived_name
        for feature_name in feature_cols
        for derived_name in (
            f"rel_{feature_name}",
            f"z_{feature_name}",
        )
    ]

    missing_derived_features = [feature_name for feature_name in expected_features if feature_name not in out.columns]

    if missing_derived_features:
        raise RuntimeError(f"固定 baseline 衍生特徵建立不完整：{missing_derived_features}")

    return out      # 把加上所有 rel_*/z_* 衍生欄位的完整 DataFrame 回傳出去



#                                        *****************************************************************************
#                                                                      轉換/干擾共用函式
#                                        *****************************************************************************
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
# 定義 baseline-relative:看每個 window 相對於自己正常基線偏了多少
# ============================================================
def build_baseline_relative_features(
    df,
    feature_cols,
    baseline_end_col="normal_end_sec",
    time_col="t_ref",
):
    '''把每個 window 的特徵值，改寫成「相對於該 session 自己正常基線的偏移量」'''
    out = df.copy()

    # 對每個 record 分組,每次處理一筆 transition session
    for record_id, sub in out.groupby("record_id"):
        idx = sub.index
        # 先找 baseline 區段
        # 把這筆 record 中，t_ref <= normal_end_sec 的窗當成 baseline normal 段
        baseline_df = sub[sub[time_col] <= sub[baseline_end_col].iloc[0]]

        if len(baseline_df) < 3:
            continue
        
        # 算 baseline 的中位數和標準差
        baseline_med = baseline_df[feature_cols].median()       # 每個特徵在正常段的典型值
        # zscore 概念
        baseline_std = baseline_df[feature_cols].std(ddof=1).replace(0, np.nan)      # 每個特徵在正常段平常會波動多少
        
        # 對每個 window 建立新特徵
        for feat in feature_cols:
            out.loc[idx, f"rel_{feat}"] = sub[feat] - baseline_med[feat]     # rel_*:實際偏移量,單位還是原本特徵單位
            out.loc[idx, f"z_{feat}"] = (sub[feat] - baseline_med[feat]) / (baseline_std[feat] + 1e-12)     # z_*:標準化偏移量,表示這個 window 離 baseline 幾個標準差

    return out


# ============================================================
# 把 band 設定整理成一個乾淨物件
# ============================================================
@dataclass(frozen=True)         # 這份 band 設定是正式設定，不應該在程式中途被亂改
class BandConfig:
    leak_focused_bands: list    # 正式主判斷的洩漏頻帶
    resonance_bands: list       # Step1 找出來的共振頻帶
    offband_bands: list         # 額外拿來做 broadband / veto 判斷的頻帶
    temporal_band_names: set    # 目前先保留給未來 temporal feature gating 用，這支程式尚未實際使用

    @property
    # 正式主頻帶 = leak-focused bands + resonance bands
    def final_main_bands(self):   
        return merge_band_groups(self.leak_focused_bands, self.resonance_bands)


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
