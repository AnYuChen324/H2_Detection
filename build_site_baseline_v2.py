#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
固定正常 Baseline 建立流程
=========================

本程式讀取一批已確認為正常狀態的 calibration TDMS，對原始 window/base features 建立固定正常參考統計，
供後續離線特徵工程與 Runtime 使用。

Baseline 保存的內容包括：
- 各 raw/base feature 的 mean
- 各 raw/base feature 的 std
- 有效 window 數量
- 正常資料來源
- 頻帶設定與特徵萃取設定的 SHA-256
- 設定組合的 config fingerprint

本程式不直接建立 rel、z、CUSUM 或 temporal 衍生特徵。
這些特徵由後續流程使用 Baseline 統計與連續 windows 建立。


一、兩種執行模式
----------------

1. discovery 模式
     %runfile build_site_baseline_v2.py --args "--mode discovery"

用途：
- 產生與 Step2a 相同的完整滑動候選頻帶
- 建立 discovery_baseline.json
- 提供 Step2a 計算候選頻帶的 rel、z、CUSUM 與 temporal 特徵
- 解決正式頻帶尚未定案前無法建立 Baseline 的循環依賴

輸出：
- band_config_step2_discovery.json
- discovery_baseline.json

此模式產生的候選頻帶尚未定案，不可直接作為正式 Runtime 頻帶設定。


2. final 模式
    %runfile build_site_baseline_v2.py

用途：
- 讀取 schema_status=final 的 band_config_step2.json
- 使用正式 leak-focused、resonance 與 offband 頻帶
- 建立正式頻帶版 site_baseline.json
- 視 feature_extraction_config.json 設定建立群組摘要特徵

輸出：
- site_baseline.json

3. 測試模式
    %runfile build_site_baseline_v2.py --args "--tdms-limit 10"

    只使用前 N 支 TDMS，輸出到 *_test.json，不會覆蓋正式的 baseline。

注意：
--mode final 表示使用正式頻帶，
不代表 site_baseline.json 的 schema_status 一定是 final。

Baseline 的 schema_status 應依正常資料來源與品質決定：
- 實驗室或測試資料：candidate
- 現場正常資料且通過品質檢查：final


二、完整執行順序
----------------
Step 1：
使用正常 calibration TDMS 執行 discovery Baseline。
    discovery 模式
    %runfile build_site_baseline_v2.py --args "--mode discovery"

輸出 discovery_baseline.json。

Step 2：
執行 Step2a 頻帶探索。

    FORCE_BAND_DISCOVERY = True

Step2a 使用 discovery_baseline.json 對候選頻帶建立 raw、rel、z、CUSUM 與 temporal 特徵並進行排名。

Step 3：
人工檢查候選結果：
- 定案 leak-focused bands
- 從 approved_resonance_bands.json 複製正式共振頻帶（由共振程式提案、人工核准）
- 加入 offband
- 確認 temporal_band_names
- 另存為 band_config_step2.json
- 將 band config 的 schema_status 設為 final

Step 4：
使用正式頻帶重新建立 Baseline。
    final 模式（預設，直接執行即可）
    %runfile build_site_baseline_v2.py

輸出 site_baseline.json。

Step 5：
執行 Step2b 正式特徵工程與特徵選擇。

    FORCE_BAND_DISCOVERY = False

Step 6：
Transition、Disturbance 與 Runtime 均使用同一份：

- band_config_step2.json
- feature_extraction_config.json
- site_baseline.json
- feature_config_micro.json


三、本程式的實際處理順序
------------------------

1. 決定 discovery 或 final 模式
2. 決定頻帶設定檔與 Baseline 輸出路徑
3. 載入並驗證 feature_extraction_config.json
4. 載入並驗證 band config
5. 同步 temporal_band_names
6. 建立 TDMS file specs
7. 計算設定檔 SHA-256 與 config fingerprint
8. 讀取 TDMS 並提取 raw/base window features
9. Final 模式視設定建立頻帶群組摘要
10. 篩選適合建立 Baseline 的 raw/base features
11. 計算每個特徵的 mean、std 與 n_valid
12. 記錄無效或變異過小的特徵
13. 輸出 discovery_baseline.json 或 site_baseline.json


四、資料隔離原則
----------------

建立 Baseline 的正常 calibration 資料不得與以下資料重複：

- model holdout
- transition holdout
- disturbance holdout
- 最終測試資料

否則可能造成資料洩漏，使模型評估結果過度樂觀。


五、本程式不負責
----------------

本程式不負責：

- 頻帶重要性排名
- 正式頻帶選擇
- Offband 人工定案
- rel／z 特徵表輸出
- CUSUM 與 temporal 狀態更新
- 模型特徵選擇
- 模型訓練
- 告警與 Veto 規則搜尋
- 模型推論
- Runtime Schema 生成
"""


from pathlib import Path
import json
import gc
import numpy as np
import pandas as pd
from dataclasses import dataclass
import hashlib
from datetime import datetime
from collections import Counter
import argparse
import shutil

from shared_feature_utils import (
    BandConfig,
    calculate_file_sha256,
    load_feature_extraction_config,
    load_or_compute_features_by_file,
    load_band_config,
    build_offband_summary_features,
    configure_temporal_band_names,
    generate_sliding_bands,
    select_all_raw_feature_columns,
    GRID_F_MIN, GRID_F_MAX, GRID_WIDTH, GRID_STRIDE,
    SIGNAL_RMS_MIN,
)



# ============================================================
# 設定全域參數
# ============================================================
# 頻帶搜尋網格：沿用 shared_feature_utils 的共用設定（與 Step2a、共振程式一致）
DISCOVERY_F_MIN = GRID_F_MIN
DISCOVERY_F_MAX = GRID_F_MAX
DISCOVERY_BAND_WIDTH = GRID_WIDTH
DISCOVERY_STRIDE = GRID_STRIDE


# ============================================================
# 儲存探索用頻帶設定
# ============================================================
def save_discovery_band_config(
        output_path,         # 輸出路徑（band_config_step2_discovery.json）
        discovery_bands,     # generate_sliding_bands() 產生的完整掃描網格
    ):
    """把完整掃描網格寫成 discovery 版的 band config，供 discovery baseline 與 Step2a 使用。

    注意：
    - 內容只由網格參數決定，不要加入時間戳記等每次都會變的欄位。
      baseline 會用這份檔案的 SHA-256 產生快取指紋；內容不變，快取才能重複使用。
    - schema_status = "discovery"，load_band_config() 在 final 模式會拒絕它，避免誤用在正式流程。
    """
    
    # 建立要寫出去的字典 payload
    payload = {
        # 和正式版 band config 同一個 schema 家族，由 load_band_config() 驗證
        "schema_name": "band_config",
        "schema_version": 2,
        "schema_status": "discovery",                   # 尚未篩選的原始掃描結果，不可用於正式流程
        "baseline_scope": "discovery_scan_grid",        # 地毯式掃描網格，不是針對洩漏訊號挑選的結果
        "band_merge_policy": "keep_primary_overlap_drop_secondary_overlap_v1",
        
        # 為了沿用 band_config 的資料結構，探索階段暫時把完整掃描頻帶放在 leak_focused_bands；
        # 這些只是候選，不代表已定案為正式洩漏頻帶。
        "leak_focused_bands": list(discovery_bands),    # 為了沿用 band_config 的共同資料結構，探索階段暫時把完整掃描頻帶放在 leak_focused_bands。此時這些只是 discovery candidates，不代表已經定案為正式洩漏頻帶。
        "resonance_bands": [],
        "offband_bands": [],
        
        # 探索階段尚未篩選，全部候選都建立 temporal 特徵（約 98 個頻帶，計算量較大）
        "temporal_band_names": sorted(band["name"] for band in discovery_bands),
        
        # 探索階段獨有：記錄這份網格是用什麼參數產生的
        "discovery_parameters": {                       
            "f_min": DISCOVERY_F_MIN,                   # 掃描的最低/最高頻率 f_min/f_max
            "f_max": DISCOVERY_F_MAX,
            "band_width": DISCOVERY_BAND_WIDTH,         # 每個候選頻帶的寬度 band_width
            "stride": DISCOVERY_STRIDE,                 # 掃描時每次往前移動多少頻率的步進值
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True,)
    # 把這串 JSON 字串寫入檔案，指定用 UTF-8 編碼寫入
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",)

    return output_path


# ============================================================
# 特徵萃取器使用的三組頻帶（離線與 runtime 共用的分組方式）
# ============================================================
@dataclass(frozen=True)
class RuntimeBandGroups:
    """load_or_compute_features_by_file() 需要的三組頻帶。
    名稱有 Runtime，但離線特徵萃取也使用同一種分組。"""
    low_bands: list[dict]     # 舊版低頻頻帶（目前固定為空）
    core_bands: list[dict]    # 洩漏頻帶 + 共振頻帶：完整萃取特徵
    aux_bands: list[dict]     # offband：只萃取少量摘要特徵


# ============================================================
# 轉換函式：語意版 BandConfig → 特徵萃取器使用的 RuntimeBandGroups
# ============================================================
def resolve_runtime_band_groups(band_config: BandConfig) -> RuntimeBandGroups:
    """
    把語意版的 BandConfig 轉成特徵萃取器使用的 low / core / aux 三組頻帶。
    - core = 洩漏頻帶 + 共振頻帶（共振頻帶和洩漏頻帶重疊時會被排除，並印出提示）
    - aux  = offband（和 core 重疊時直接報錯：offband 必須是 core 以外的參照頻帶）
    """
    # 使用 BandConfig.final_main_bands，讓「core 怎麼合併」只定義在一個地方
    core_bands = band_config.final_main_bands
    
    # 檢查有沒有共振頻帶因為和洩漏頻帶重疊而被排除
    kept = {str(b["name"]) for b in core_bands}
    dropped = [str(b["name"]) for b in band_config.resonance_bands if str(b["name"]) not in kept]
    if dropped:
        print(f"⚠️ 以下共振頻帶與洩漏頻帶重疊，未納入 core bands：{dropped}")
        
    # 檢查 offband 有沒有和 core 頻帶重疊（重疊時同一段頻率會被算兩次，offband 也失去參照意義）
    def overlap(a, b):
        return max(a["f_low"], b["f_low"]) < min(a["f_high"], b["f_high"])
    
    offband_overlaps = [
        f"{off['name']} ↔ {core['name']}"
        for off in band_config.offband_bands
        for core in core_bands
        if overlap(off, core)
    ]
    if offband_overlaps:
        raise ValueError(f"offband 不可與 core 頻帶重疊：{offband_overlaps}")
    
    return RuntimeBandGroups(
        low_bands=[],                           # 不再另外保留舊版 low band
        core_bands=core_bands,
        aux_bands=band_config.offband_bands,    # aux_bands直接對應offband_bands,原封不動搬過去
    )


# ============================================================
# 內容重複檢查（比檔案內容，不比檔名；改過檔名的複本也抓得到）
# ============================================================
def check_baseline_duplicates(tdms_paths, other_dirs):
    """
    1. 正常資料內部：內容完全相同 → 只保留一支（避免同一段訊號被算兩次）
    2. 正常資料 vs 其他資料夾（洩漏等）：內容相同 → 直接停止（label 矛盾，不能拿來建 baseline）
    """
    hashes = {}        # hashes：記住「指紋 → 第一支出現的正常檔」
    kept = []          # kept：最後真正要拿去建 baseline 的正常檔
    dup_within = []    # dup_within：正常資料夾內發現的重複副本，留給報告使用
    
    # 正常資料內去重
    for p in tdms_paths:
        h = calculate_file_sha256(p)
        if h in hashes:
            dup_within.append((p, hashes[h]))
        else:
            hashes[h] = p
            kept.append(p)
            
    if dup_within:
        print(f"⚠️ 正常資料內部重複 {len(dup_within)} 支，只保留一份，例如：{dup_within[:3]}")
    # 掃描 leak1 等其他資料夾；若任何檔案內容和保留的正常檔相同，就記為衝突
    conflicts = []
    for d in other_dirs:
      if not Path(d).is_dir():
          raise FileNotFoundError(f"--check-against 資料夾不存在：{d}")
          
      for q in sorted(Path(d).rglob("*.tdms")):
            if calculate_file_sha256(q) in hashes:
                conflicts.append(str(q))
    # 一旦發現「同一內容同時出現在正常與洩漏資料」，就停止 baseline 建立
    if conflicts:
        raise ValueError(
            f"正常資料與其他資料夾有 {len(conflicts)} 支內容完全相同（label 矛盾），"
            f"請先移出再建 baseline，例如：{conflicts[:3]}"
        )

    print(f"✅ 內容重複檢查通過：保留 {len(kept)} 支正常資料，"
          f"與 {len(other_dirs)} 個資料夾沒有內容重複")
    return kept


# ============================================================
# TDMS 路徑整理成統一格式
# ============================================================
def build_file_specs(tdms_paths):     # tdms_paths：一個 TDMS 路徑清單
    '''把這些輸入 TDMS 路徑整理成後面特徵工程函式看得懂的 file_specs 格式'''
    # 檢查：如果沒傳任何檔案進來，就直接報錯
    if not tdms_paths:
        raise ValueError("至少要提供一個 TDMS")
    # 先建立一個空清單，等等把整理好的結果放進去
    file_specs = []
    
    # 開始逐一處理每個 TDMS 路徑
    for p in tdms_paths:
        path = Path(p)    # 把每個路徑字串轉成 Path 物件
        # 檢查這個檔案是不是真的存在
        if not path.exists():
            raise FileNotFoundError(f"找不到 TDMS 檔案: {path}")

        # 把每個檔案整理成一個 dictionary
        file_specs.append(
            {
                "filepath": str(path),    # 完整檔案路徑
                "record_id": path.stem,   # 檔名主體，不含副檔名
            }
        )

    # 檢查 record_id（檔名主體）是否重複：後續會依 record_id 分組計算 CUSUM、rolling 等跨視窗特徵，重複時兩個檔案會被串成同一條時間序列
    counts = Counter(s["record_id"] for s in file_specs)
    duplicated = sorted(r for r, n in counts.items() if n > 1)
    
    if duplicated:
        raise ValueError(f"有 TDMS 檔名重複（record_id 會衝突）：{duplicated[:10]}")

    return file_specs


# ============================================================
# 最底層負責把 TDMS 變成原始 window 特徵表
# ============================================================
def extract_base_feature_table(
    *,                         # 全部都是關鍵字參數，而且沒有預設值：呼叫端一個都不能漏
    file_specs,                # 整理好的 TDMS 清單
    cache_dir: Path,           # 快取資料夾
    split_name: str,           # 這批資料名稱（會成為快取檔名的一部分）
    band_config: BandConfig,   # 語意版頻帶設定（洩漏 / 共振 / offband）
    n_jobs: int,               # 平行處理數
    window_sec: float,         # 窗長
    hop_sec: float,            # hop 長
    fs: float,                 # 取樣率
    nperseg: int,              # Welch / PSD 相關設定
    max_points: int,           # 最大讀取點數
    feature_version: str,      # 特徵公式版本（會成為快取指紋的一部分）
):
    """從 file_specs 抽出原始 window 特徵表，並排除訊號異常的視窗。"""
    # 把語意版的 BandConfig 轉成特徵萃取器使用的 low / core / aux 三組頻帶
    band_groups = resolve_runtime_band_groups(band_config)
    
    # 讀每個 TDMS → 切成很多 window → 依頻帶算各種基礎特徵（有快取就直接讀）
    features = load_or_compute_features_by_file(
        cache_dir=cache_dir,
        file_specs=file_specs,             # 每一筆 TDMS 讀進來
        core_bands=band_groups.core_bands,
        aux_bands=band_groups.aux_bands,
        low_bands=band_groups.low_bands,
        split_name=split_name,
        n_jobs=n_jobs,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        target_levels=None,
        feature_version=feature_version,
        max_points=max_points,
    )
    
    # 轉成 DataFrame
    df = pd.DataFrame(features).reset_index(drop=True)
    del features            # 處理大量 TDMS 時盡快釋放記憶體
    gc.collect()
    
    # 硬性 QC：排除訊號異常的視窗（門檻與特徵工程的 check_signal_quality() 一致）
    # 全零訊號的 RMS 為 0，含 NaN 的訊號 RMS 為 NaN，都會在這裡被排除
    rms_min = SIGNAL_RMS_MIN
    bad = pd.Series(False, index=df.index)
    for ch in ("ai0", "ai1"):
        rms = pd.to_numeric(df[f"{ch}_rms"], errors="coerce")
        bad |= ~np.isfinite(rms) | (rms < rms_min)
        
    if bad.any():
        bad_files = sorted(df.loc[bad, "record_id"].unique())
        print(f"⚠️ 訊號 QC：排除 {int(bad.sum())} 個視窗（來自 {len(bad_files)} 個檔案），"
              f"例如 {bad_files[:5]}")
        df = df[~bad].reset_index(drop=True)
        
    # 檢查
    if df.empty:
        raise ValueError("訊號 QC 後沒有任何可用的視窗，請檢查正常資料是否正確")

    return df


# ============================================================
# 對所有候選原始特徵建立正常 baseline 統計
# ============================================================
def build_site_baseline_stats(df_features: pd.DataFrame,               # 正常資料的特徵表
                              raw_feature_cols: list[str],             # raw 特徵（select_all_raw_feature_columns()）+ 群組摘要欄位
                              heavy_tail_ratio: float = 3.0,
                        ) -> tuple[dict, list[dict], list[dict]]:
    """
    這個函式負責 baseline 最核心的工作：對每一個特徵，算出「正常狀態下長什麼樣子」。
    
    對每個 raw feature 算出 baseline 的 mean / std（後續 z-score 使用）。
    另外記錄穩健統計量（中位數、MAD 換算的標準差、1% / 99% 分位數）供診斷，
    並標出 std 明顯大於穩健標準差的「長尾特徵」——代表 std 可能被少數極端視窗撐大。
    注意：視窗之間有 75% 重疊，n_valid 會高估實際獨立樣本數。
    """
    # 建立三個清單
    stats = {}                # 成功建立 baseline 的特徵 → 它的統計量
    invalid_features = []     # 無法建立 baseline 的特徵，以及原因
    heavy_tailed = []         # 診斷用：分布形狀需要注意的特徵

    # 逐欄計算 baseline 統計量
    for col in raw_feature_cols:
        # 先把欄位轉成數值並去掉空值
        x = pd.to_numeric(df_features[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        # 檢查１：如果這個特徵所有視窗都是 NaN
        if x.empty:
            invalid_features.append({"feature_name": col,"reason": "沒有有效數值","n_valid": 0})
            continue
        
        # 有效值比例不足：大部分視窗都是 NaN 時，這個特徵的統計量不可靠
        valid_ratio = len(x) / len(df_features)
        if valid_ratio < 0.95:
            invalid_features.append({
                "feature_name": col,
                "reason": "有效數值比例不足",
                "n_valid": int(len(x)),
                "valid_ratio": round(valid_ratio, 4),
            })
            continue

        # 計算平均值跟標準差
        mean_val = float(x.mean())       
        std_val = float(x.std(ddof=1))   

        # 檢查2：平均值不是有限數值（例如極端值導致溢位）
        if not np.isfinite(mean_val):
            invalid_features.append({"feature_name": col, "reason": "mean 不是有限數值", "n_valid": int(len(x)),})
            continue

        std_floor = max(abs(mean_val) * 1e-12, np.finfo(np.float64).tiny,)
        
        # 檢查3：std 太小或無效，所有視窗的值幾乎一樣
        if (not np.isfinite(std_val) or std_val <= std_floor):
            invalid_features.append({
                "feature_name": col,
                "reason": "std 無效或相對尺度接近 0",
                "n_valid": int(len(x)),
                "mean": mean_val,
                "std": std_val,
                "std_floor": std_floor,
            })
            continue
        
        # 穩健統計量（只供診斷，不影響後續 z-score），用中位數計算不受極端值影響
        median_val = float(x.median())
        robust_std = float(1.4826 * (x - median_val).abs().median())   # MAD 換算成和 std 同尺度

        # 算正式的統計量：把這個特徵的 baseline 統計量存進字典
        stats[col] = {
            "mean": mean_val,                      # 後續 z-score 使用
            "std": std_val,                        # 後續 z-score 使用
            "n_valid": int(len(x)),                # 用了幾個視窗
            "median": median_val,                  # 以下僅供診斷
            "robust_std": robust_std,
            "p01": float(x.quantile(0.01)),
            "p99": float(x.quantile(0.99)),
        }
        
        # 找出需要注意的特徵
        # std 是穩健標準差的 3 倍以上 → 很可能被少數極端視窗撐大
        if robust_std > 0 and std_val / robust_std > heavy_tail_ratio:
            heavy_tailed.append({
                "feature_name": col,
                "std_to_robust_std": round(std_val / robust_std, 2),
            })
            
        # 穩健標準差為 0：超過一半的視窗數值相同（常見於計數型特徵），std 完全由少數不同的視窗決定，z-score 分布會很不連續
        elif robust_std == 0:
            heavy_tailed.append({
                "feature_name": col,
                "std_to_robust_std": None,
                "note": "超過一半的視窗數值相同",
            })
            
    # 排序並回傳:把需要注意的特徵依比值由大到小排序（None 排最前面），然後回傳三樣東西：正式統計量、無效特徵清單、診斷清單。
    heavy_tailed.sort(
        key=lambda d: float("inf") if d["std_to_robust_std"] is None else d["std_to_robust_std"],
        reverse=True,
    )
    return stats, invalid_features, heavy_tailed


# ============================================================
# 算好的 baseline 存成 site_baseline.json
# ============================================================
def save_site_baseline(
    *,
    output_path,                       # 最後 baseline 檔要存去哪裡
    baseline_stats,                    # 每個特徵的 mean / std（及診斷用的穩健統計量）
    tdms_paths,                        # 這次拿來建 baseline 的正常 TDMS 清單
    band_config_path,                  # 這次用的是哪份 band 設定
    feature_extraction_config_path,    # 這次用的特徵萃取設定
    invalid_features,                  # 無法建立 baseline 的特徵與原因
    band_config_hash,
    feature_extraction_config_hash,
    config_fingerprint,
    raw_feature_cols,                  # 候選欄位（raw + 群組摘要）
    n_windows,                         # QC 後實際使用的視窗數
    window_sec,                        # 這次建立 baseline 用的參數
    hop_sec,
    fs,
    nperseg,
    max_points,
    baseline_environment,              # 目前是實驗室資料 laboratory / site
    baseline_schema_status,            # candidate / final
    baseline_scope,
    base_feature_version,
    heavy_tailed_features              # 診斷用：分布需要注意的特徵
):
    """
    把 baseline 統計量與建立時的所有設定寫成 JSON。
    - 先寫暫存檔再替換，避免寫到一半中斷時弄壞原本的 baseline
    - NaN / Inf 轉成 null，確保輸出是標準 JSON
    """
    # 把所有非有限的數值轉成 null
    def to_json_safe(obj):
        """把 NaN / Inf 轉成 None（JSON 的 null），讓檔案符合標準 JSON。"""
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_safe(v) for v in obj]
        return obj
    
    # 建立 payload：把所有 metadata（來源檔案、用的 config 路徑、訊號參數）連同算好的 baseline_stats 一起包成一個 payload 寫成 JSON
    payload = {
        "schema_name": "site_baseline",
        "schema_version": 1,
        "schema_status": baseline_schema_status,
        "environment": baseline_environment,
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline_type": "fixed_normal_baseline",                   # 標記這份檔案的用途:場域正常 baseline
        "baseline_scope": baseline_scope,
        
        # 上游特徵契約：後續流程會比對這些指紋，不一致就拒絕使用這份 baseline
        "base_feature_version": base_feature_version,              # baseline 版本
        "band_config_path": str(Path(band_config_path)),
        "band_config_sha256":band_config_hash,
        "feature_extraction_config_path": str(Path(feature_extraction_config_path)),
        "feature_extraction_config_sha256": (feature_extraction_config_hash),
        "config_fingerprint":config_fingerprint,
        
        # 正常資料來源（資料隔離檢查會用到）
        "source_tdms_count": len(tdms_paths),
        "source_tdms_paths": [str(Path(p)) for p in tdms_paths],
        
        # 訊號與切窗參數
        "window_sec": window_sec,                    
        "hop_sec": hop_sec,
        "fs": fs,
        "nperseg": nperseg,
        "max_points": max_points,
        "n_windows": int(n_windows),
                
        # baseline 統計特徵結果
        "feature_stats": baseline_stats,                            # 每個 raw feature 的：mean, std
        "candidate_raw_feature_columns":list(raw_feature_cols),     # 候選欄位：raw 特徵 + 群組摘要（名稱保留以相容其他程式）
        "baseline_feature_columns":sorted(baseline_stats.keys()),   # 真正具有有效 baseline 統計的欄位              
        "invalid_features": invalid_features,
        # 診斷用：std 可能被極端視窗撐大的特徵
        "heavy_tailed_features": heavy_tailed_features,    
        
        }
    
    # 把輸出路徑轉成 Path
    output_path = Path(output_path)
    # 建立輸出資料夾
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 先寫到暫存檔，完成後再整個替換，避免寫到一半中斷時把原本的 baseline 弄壞
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(to_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp_path.replace(output_path)

    print(f"{baseline_scope} baseline 已輸出：{output_path}")
    print(f"  有效特徵 {len(baseline_stats)} 個、無效 {len(invalid_features)} 個、"
          f"診斷提示 {len(heavy_tailed_features)} 個、視窗 {int(n_windows)} 個")


# ============================================================
# 讀取命令列參數
# ============================================================
def parse_args() -> argparse.Namespace:
    """讀取命令列參數。參數依用途分組，執行 --help 時會分區顯示。"""
    # 建立命令列解析器
    parser = argparse.ArgumentParser(description="建立固定正常 baseline（discovery / final）")

    # ---------- 模式 ----------
    # 選擇頻帶模式，只允許兩種模式:discovery, final
    mode = parser.add_argument_group("模式")
    mode.add_argument(
        "--mode", choices=["discovery", "final"], default="final",
        help="discovery：完整掃描候選頻帶，供 Step2a；final：使用定案的 band_config_step2.json",
    )

    # ---------- 正常資料 ----------
    data = parser.add_argument_group("正常資料")
    data.add_argument(
        "--normal-dir", type=Path,
        default=Path("/Users/paul/Desktop/Final/normal_leak_test/leak0"),
        help="建立 baseline 用的正常 TDMS 資料夾",
    )
    
    # 限制讀取多少支 TDMS 檔
    data.add_argument(
        "--tdms-limit", type=int, default=None,
        help="只用前 N 支 TDMS（測試流程用）；不指定則使用全部",
    )
    
    # 正常資料不可與這些資料夾的檔案內容重複
    data.add_argument(
        "--check-against", type=Path, nargs="*",
        default=[Path("/Users/paul/Desktop/Final/normal_leak_test/leak1")],
        help="檢查正常資料不可與這些資料夾的檔案內容重複（例如洩漏資料）",
    )
        
    # 標記正常資料來自哪裡：實驗室或現場
    data.add_argument(
        "--environment", choices=["laboratory", "site"], default="laboratory",
        help="資料來源：實驗室或現場",
    )
    
    # 表示這份 Baseline 的狀態
    # Candidat 候選版本：- 實驗室資料 - 流程測試 - 尚未正式部署
    # final 正式現場版本。
    data.add_argument(
        "--schema-status", choices=["candidate", "final"], default="candidate",
        help="baseline 狀態；final 只能用於現場資料、且使用全部檔案",
    )

    # ---------- 路徑與執行 ----------
    paths = parser.add_argument_group("路徑與執行")     # 指定程式從哪裡讀取設定檔，以及把結果存到哪裡
    paths.add_argument(
        "--result-root", type=Path, default=Path("/Users/paul/Desktop/Final/test_result"),
        help="設定檔與 baseline 輸出的根目錄",
    )
    paths.add_argument(
        "--cache-dir", type=Path, default=Path("/Users/paul/Desktop/runtime_outputs/site_baseline_cache"),
        help="特徵快取資料夾",
    )
    paths.add_argument(
        "--n-jobs", type=int, default=6,
        help="平行處理數",
    )
    # 解析所有輸入參數
    args = parser.parse_args()
    
    # 防呆：discovery 不能標成 final
    if args.mode == "discovery" and args.schema_status == "final":
        parser.error("discovery 模式不可使用 --schema-status final（探索用的 baseline 不是正式版）")
    
    # 防呆：final 狀態的 baseline 必須是現場資料，而且用全部檔案
    if args.schema_status == "final":
        if args.environment != "site":
            parser.error("--schema-status final 只能搭配 --environment site")
        # Final Baseline 不可只用部分檔案
        if args.tdms_limit is not None:
            parser.error("--schema-status final 不能搭配 --tdms-limit（必須使用全部檔案）")
    # tdms-limit 必須是正整數
    if args.tdms_limit is not None and args.tdms_limit <= 0:
        parser.error("--tdms-limit 必須大於 0")

    return args



# ============================================================
# baseline 建立程式的主流程入口
# ============================================================
def main():
    '''
    ① 設定資料來源（實驗室 / 現場）、找出正常 TDMS
    ② 依模式決定頻帶設定檔與輸出位置（discovery / final）
    ③ 讀取特徵萃取設定（窗長、取樣率、nperseg 等）
    ④ 讀取並驗證頻帶設定
    ⑤ 準備檔案清單、同步 temporal 頻帶、計算設定指紋
    ⑥ 算特徵（含訊號 QC）
    ⑦ 挑出要建 baseline 的欄位（raw ＋ 群組摘要）
    ⑧ 算 mean / std 與診斷統計
    ⑨ 存成 JSON
    '''
    # ============================================================
    # 讀取執行參數
    # ============================================================
    args = parse_args()
    # 把常用參數取出來
    baseline_mode = args.mode
    baseline_environment = args.environment
    baseline_schema_status = args.schema_status

    # -- 正常 TDMS 資料 --
    tdms_paths = sorted(str(p) for p in args.normal_dir.rglob("*.tdms"))   # v22 的檔在 session 子資料夾裡
    if not tdms_paths:     # 沒有 TDMS 就停止
        raise ValueError(f"在 {args.normal_dir} 找不到任何 .tdms 檔案")
    if args.tdms_limit is not None:     # 測試模式只使用前 N 支
        tdms_paths = tdms_paths[:args.tdms_limit]
        print(f"⚠️ 測試模式：只使用前 {len(tdms_paths)} 支 TDMS")
        
    # 內容重複檢查：內部重複只留一份；跟洩漏資料重複就停止
    tdms_paths = check_baseline_duplicates(tdms_paths, other_dirs=args.check_against)
    
    # 最低資料量:正式 Baseline 至少要有 10 支 TDMS
    MIN_FINAL_FILES = 10
    if baseline_schema_status == "final" and len(tdms_paths) < MIN_FINAL_FILES:
        raise ValueError(f"正式 baseline 至少需要 {MIN_FINAL_FILES} 支 TDMS，目前只有 {len(tdms_paths)} 支")

    # ============================================================
    # 設定結果根目錄
    # ============================================================
    # -- 依模式決定頻帶來源與輸出位置 --
    result_root = args.result_root
    # 組成特徵設定檔路徑
    feature_extraction_config_path = result_root / "feature_extraction_config.json"

    # ============================================================
    # 自動建立完整掃描頻帶
    # ============================================================
    # Discovery 模式
    if baseline_mode == "discovery":
        discovery_bands = generate_sliding_bands(
            f_min=DISCOVERY_F_MIN,
            f_max=DISCOVERY_F_MAX,
            band_width=DISCOVERY_BAND_WIDTH,
            stride=DISCOVERY_STRIDE,
        )
        # 設定 discovery 頻帶檔，並儲存
        band_config_path = result_root / "band_config_step2_discovery.json"
        save_discovery_band_config(output_path=band_config_path, discovery_bands=discovery_bands)
        output_path = result_root / "discovery_baseline.json"
        baseline_scope = "discovery_scan_grid"
    else:
        # Final 頻帶模式
        band_config_path = result_root / "band_config_step2.json"
        output_path = result_root / "site_baseline.json"
        baseline_scope = "finalized_band_config"
        # 正式頻帶檔不存在就停止
        if not band_config_path.exists():
            raise FileNotFoundError(f"mode=final，但找不到正式設定：{band_config_path}")
    
    # ============================================================
    # 測試模式改用 _test.json
    # ============================================================
    # 測試模式輸出到另一個檔名，避免覆蓋正式的 baseline
    if args.tdms_limit is not None:
        output_path = output_path.with_name(f"{output_path.stem}_test{output_path.suffix}")
    
    # 保護正式版：既有的 site_baseline.json 若是 final，不允許被非 final 的 baseline 覆蓋
    if baseline_mode == "final" and output_path.exists() and baseline_schema_status != "final":
        existing_status = json.loads(output_path.read_text(encoding="utf-8")).get("schema_status")
        if existing_status == "final":
            output_path = output_path.with_name(f"{output_path.stem}_candidate{output_path.suffix}")
            print(f"⚠️ 既有的 baseline 是正式版（final），本次 candidate 改輸出到：{output_path.name}")
    # 顯示本次執行資訊
    print("\n=== Baseline 建立模式 ===")
    print("mode:", baseline_mode)
    print("scope:", baseline_scope)
    print("band config:", band_config_path)
    print("output:", output_path)

    # ============================================================
    # 載入特徵設定檔
    # ============================================================
    extraction_cfg = load_feature_extraction_config(feature_extraction_config_path)

    # 定義訊號處理參數（從 feature_extraction_config.json 讀取）
    window_sec = float(extraction_cfg["window_sec"])          # 每個窗多長
    hop_sec = float(extraction_cfg["hop_sec"])                # 每次滑動多少
    fs = float(extraction_cfg["fs"])                          # 取樣率
    nperseg = int(extraction_cfg["nperseg"])                  # 頻譜分析參數
    max_points = int(extraction_cfg["max_points"])            # 最多讀多少點
    include_offband_summary = bool(extraction_cfg["include_offband_summary"])    # 是否建立 offband、leak-focused、resonance 群組摘要
    base_feature_version = str(extraction_cfg["base_feature_version"])           # 基礎特徵計算公式版本，用於快取與設定指紋
    n_jobs = args.n_jobs               # 平行處理數
      
    # ============================================================
    # 載入並驗證本次模式對應的 band config
    # ============================================================
    # Discovery 模式只能讀取尚未篩選的探索頻帶設定；
    # Final 模式只能讀取已經人工加入 Offband 並定案的正式設定。
    expected_band_status = ("discovery" if baseline_mode == "discovery" else "final")
    # 載入並驗證頻帶
    band_config = load_band_config(band_config_path, expected_status=expected_band_status,require_offband=(baseline_mode == "final"))
    
    # 顯示本次實際載入的頻帶數量，方便核對設定
    print("\n=== Band config 驗證完成 ===")
    print("expected status:", expected_band_status)
    print("leak-focused bands:", len(band_config.leak_focused_bands),)
    print("resonance bands:", len(band_config.resonance_bands),)
    print("offband bands:", len(band_config.offband_bands),)
    print("temporal bands:", len(band_config.temporal_band_names),)
    
    # ============================================================
    # 準備資料與快取
    # ============================================================
    # 整理正常 TDMS 檔案清單
    file_specs = build_file_specs(tdms_paths)
    
    # 把 band config 的 temporal band 名稱同步到 shared_feature_utils 的基礎 window 特徵提取器。
    configure_temporal_band_names(band_config.temporal_band_names)
    
    # 快取名稱綁定設定檔版本
    band_config_hash = calculate_file_sha256(band_config_path)
    feature_extraction_config_hash = calculate_file_sha256(feature_extraction_config_path)
    
    # 組合本次設定指紋
    fingerprint_source = (
        f"{baseline_mode} : {baseline_scope} : {band_config_hash} : {feature_extraction_config_hash} : {base_feature_version}")
    
    # 再次計算 SHA-256, 把文字轉成雜湊函式可處理的 bytes, 轉成十六進位文字
    config_fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    cache_split_name = (f"{baseline_mode}_baseline_build_{config_fingerprint}")

    # ============================================================
    # 特徵工程:把 TDMS 轉成特徵表
    # ============================================================
    # 抽出最原始 base features:讀正常 TDMS -> 切窗 -> 依照 band config 算出原始 band 特徵表
    df_features = extract_base_feature_table(
        file_specs=file_specs,
        cache_dir=args.cache_dir,
        split_name=cache_split_name,
        band_config=band_config,
        n_jobs=n_jobs,
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        max_points=max_points,
        feature_version=base_feature_version,
    )
    
    # 最低視窗數:正式 Baseline 至少 100 個有效視窗
    MIN_FINAL_WINDOWS = 100
    if baseline_schema_status == "final" and len(df_features) < MIN_FINAL_WINDOWS:
        raise ValueError(
            f"正式 baseline 至少需要 {MIN_FINAL_WINDOWS} 個有效視窗，目前只有 {len(df_features)} 個"
        )
        
    # ============================================================
    # 挑出要建 baseline 的欄位
    # ============================================================
    # 先抓原始 raw base features（這時還沒加 summary）
    raw_feature_cols = select_all_raw_feature_columns(df_features)
    print(f"baseline raw 特徵欄位：{len(raw_feature_cols)} 個")
    
    # 再建立 offband / leak_focus / resonance 的 summary base features
    build_summary_features = (include_offband_summary and baseline_mode == "final")
    
    # 建立頻帶群組摘要
    if build_summary_features:
        df_features = build_offband_summary_features(df_features, band_config,)
        
    # 收集實際存在的摘要欄位
    summary_base_cols = [
        c for c in [
            "offband_E_mean",
            "offband_E_max",
            "leak_focus_E_mean",
            "leak_focus_E_max",
            "resonance_E_mean",
            "resonance_E_max",
        ]
        if c in df_features.columns
    ]
    
    # baseline 真正要存的欄位 = 原始 raw features + summary base features
    baseline_feature_cols = sorted(set(raw_feature_cols + summary_base_cols))
    
    # ============================================================
    # 計算 baseline 統計量
    # ============================================================
    baseline_stats, invalid_features, heavy_tailed = build_site_baseline_stats(df_features, baseline_feature_cols)

    print("baseline 候選特徵數：",len(baseline_feature_cols))
    print("baseline 有效特徵數：",len(baseline_stats))
    print("baseline 無效特徵數：",len(invalid_features))
    
    # 顯示長尾診斷
    if heavy_tailed:
        # 多數視窗值完全相同的特徵。
        sparse = [d for d in heavy_tailed if d["std_to_robust_std"] is None]
        
        # 一般標準差比穩健標準差大很多的長尾特徵
        long_tail = [d for d in heavy_tailed if d["std_to_robust_std"] is not None]
        print(f"\n⚠️ 診斷：長尾特徵 {len(long_tail)} 個（std 超過穩健標準差 3 倍）、"
              f"多數值相同的特徵 {len(sparse)} 個")
        for item in long_tail[:10]:
            print(f"  - {item['feature_name']}：std / robust_std = {item['std_to_robust_std']}")
    
    # 顯示無效特徵原因
    if invalid_features:
        print("\nbaseline 無效特徵：")
        for item in invalid_features:
            print(
                "-",
                item["feature_name"],
                ":",
                item["reason"],
            )
            
    # ============================================================
    # 存檔 Baseline
    # ============================================================
    # 最後把所有東西打包存檔，存成 site_baseline.json
    save_site_baseline(
        output_path=output_path,
        baseline_stats=baseline_stats,
        tdms_paths=tdms_paths,
        band_config_path=band_config_path,
        feature_extraction_config_path=(feature_extraction_config_path),
        invalid_features=invalid_features,
        band_config_hash=band_config_hash,
        feature_extraction_config_hash=(feature_extraction_config_hash),
        config_fingerprint=config_fingerprint,
        raw_feature_cols=baseline_feature_cols,
        n_windows=len(df_features),
        window_sec=window_sec,
        hop_sec=hop_sec,
        fs=fs,
        nperseg=nperseg,
        max_points=max_points,
        baseline_environment=baseline_environment,
        baseline_schema_status=baseline_schema_status,
        baseline_scope=baseline_scope,
        base_feature_version=base_feature_version,
        heavy_tailed_features=heavy_tailed
    )
    
    # 測試模式不建立歷史備份
    # 每次建立正式流程用的 baseline，都另存一份帶時間戳記的歷史版本（測試模式不存）
    if args.tdms_limit is None:
        history_dir = output_path.parent / "baseline_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = history_dir / f"{output_path.stem}_{baseline_schema_status}_{stamp}{output_path.suffix}"
        shutil.copy2(output_path, archived)
        print(f"  歷史版本：{archived}")
        


if __name__ == "__main__":
    main()
    
    
    
        
