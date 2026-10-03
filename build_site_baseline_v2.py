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

    BASELINE_MODE = "discovery"

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

    BASELINE_MODE = "final"

用途：
- 讀取 schema_status=final 的 band_config_step2.json
- 使用正式 leak-focused、resonance 與 offband 頻帶
- 建立正式頻帶版 site_baseline.json
- 視 feature_extraction_config.json 設定建立群組摘要特徵

輸出：
- site_baseline.json

注意：

BASELINE_MODE="final" 表示使用正式頻帶，
不代表 site_baseline.json 的 schema_status 一定是 final。

Baseline 的 schema_status 應依正常資料來源與品質決定：
- 實驗室或測試資料：candidate
- 現場正常資料且通過品質檢查：final


二、完整執行順序
----------------
Step 1：
使用正常 calibration TDMS 執行 discovery Baseline。

    BASELINE_MODE = "discovery"

輸出 discovery_baseline.json。

Step 2：
執行 Step2a 頻帶探索。

    FORCE_BAND_DISCOVERY = True

Step2a 使用 discovery_baseline.json 對候選頻帶建立 raw、rel、z、CUSUM 與 temporal 特徵並進行排名。

Step 3：
人工檢查候選結果：
- 定案 leak-focused bands
- 保留需要的 resonance bands
- 加入 offband
- 確認 temporal_band_names
- 另存為 band_config_step2.json
- 將 band config 的 schema_status 設為 final

Step 4：
使用正式頻帶重新建立 Baseline。

    BASELINE_MODE = "final"

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

from shared_feature_utils import (
    load_or_compute_features_by_file,
    merge_band_groups,
    build_offband_summary_features,
    configure_temporal_band_names,
    generate_sliding_bands
)



# ============================================================
# 設定全域參數
# ============================================================
# BASELINE_MODE 決定使用哪一套頻帶範圍：
# discovery：完整掃描候選頻帶
# final：已定案的正式頻帶
# 注意：BASELINE_MODE="final" 不代表輸出的 baseline schema_status, 一定是 final；schema_status 仍由資料來源與品質決定。
#BASELINE_MODE = "discovery"
BASELINE_MODE = "final"

# 使用 Step2a 相同的頻段拆解
DISCOVERY_F_MIN = 5_000
DISCOVERY_F_MAX = 500_000
DISCOVERY_BAND_WIDTH = 10_000
DISCOVERY_STRIDE = 5_000


# ============================================================
# 儲存探索用頻帶設定
# ============================================================
def save_discovery_band_config(
        output_path,         # 檔案路徑輸出
        discovery_bands,     # 頻帶搜尋（Step2a）掃出來的候選頻帶清單
    ):
    # 建立要寫出去的字典 payload
    payload = {
        # schema_name 跟 schema_version 都跟你正式版 band config（load_frozen_band_config 那支）驗證的欄位對得起來——同一個 schema 家族，版本一致
        "schema_name": "band_config",
        "schema_version": 2,
        "schema_status": "discovery",                   # 確保這份「還沒篩選過」的原始掃描結果不會被誤用在正式流程裡
        "baseline_scope": "discovery_scan_grid",        # 這份頻帶設定的產生範疇/依據，是一個地毯式掃描網格（grid），而不是針對特定漏檢測訊號精挑出來的結果
        "band_merge_policy": (
            "keep_primary_overlap_"
            "drop_secondary_overlap_v1"
        ),
        "leak_focused_bands": list(discovery_bands),    # 為了沿用 band_config 的共同資料結構，探索階段暫時把完整掃描頻帶放在 leak_focused_bands。此時這些只是 discovery candidates，不代表已經定案為正式洩漏頻帶。
        "resonance_bands": [],
        "offband_bands": [],
        "temporal_band_names": sorted(                  # 既然這是探索階段、還沒篩選，乾脆全部候選都先納入 temporal 名單
            band["name"]
            for band in discovery_bands
        ),
        "discovery_parameters": {                       # 探索階段獨有、正式版完全沒有
            "f_min": DISCOVERY_F_MIN,                   # 掃描的最低/最高頻率 f_min/f_max
            "f_max": DISCOVERY_F_MAX,
            "band_width": DISCOVERY_BAND_WIDTH,         # 每個候選頻帶的寬度 band_width
            "stride": DISCOVERY_STRIDE,                 # 掃描時每次往前移動多少頻率的步進值
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True,)

    # 把這串 JSON 字串寫入檔案，指定用 UTF-8 編碼寫入
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,), encoding="utf-8",)

    return output_path


# ============================================================
# 定義一個資料類別 BandConfig (專門拿來裝 band 設定的小盒子)
# ============================================================
@dataclass(frozen=True)             # 這份 band 設定是正式設定，不應該在程式中途被亂改
class BandConfig:
    '''
    專門拿來裝 band 設定的小盒子。
    frozen=True 的意思是：建立之後，內容不能隨便改。
    好處是：避免你後面程式不小心把 band 改掉。
    '''
    leak_focused_bands: list[dict]   # 正式主判斷的洩漏頻帶
    resonance_bands: list[dict]      # Step1 找出來的共振頻帶
    offband_bands: list[dict]        # 額外拿來做 broadband / veto 判斷的頻帶
    temporal_band_names: list[str]   # 指定基礎特徵提取器需要建立 temporal raw/base 特徵的頻帶
    

# ============================================================
# 定義一個新的資料類別存放的是load_or_compute_features_by_file實際需要的三種 band :low_bands、core_bands、aux_bands
# ============================================================
@dataclass(frozen=True)
class RuntimeBandGroups:
    low_bands: list[dict]
    core_bands: list[dict]
    aux_bands: list[dict]


# ============================================================
# 轉換函：:輸入是語意版的BandConfig,輸出是load_or_compute_features_by_file可以直接吃的RuntimeBandGroups
# ============================================================
def resolve_runtime_band_groups(band_config: BandConfig) -> RuntimeBandGroups:
    return RuntimeBandGroups(
        low_bands=[],     # low_bands固定給空清單(對應你所有腳本裡一直重複出現的「不再另外保留舊版 low band」這個設計決定)
        core_bands=merge_band_groups(       # core_bands把leak_focused_bands跟resonance_bands合併起來
            band_config.leak_focused_bands,
            band_config.resonance_bands,
        ),
        aux_bands=band_config.offband_bands,    # aux_bands直接對應offband_bands,原封不動搬過去
    )


# ============================================================
# 讀 band 設定檔：config loaders
# ============================================================
def load_band_config(
        band_config_path,
        *,
        expected_status=None,
    ):
    payload = json.loads(Path(band_config_path).read_text(encoding="utf-8"))
    
    # 檢查 payload 內容
    if payload.get("schema_name") != "band_config":
        raise ValueError("band config 的 schema_name 必須是 band_config")

    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("目前只支援 band_config schema_version=2")
        
    # 檢查 schema_status
    schema_status = payload.get("schema_status")

    if expected_status is not None:
        if schema_status != expected_status:
            raise ValueError(f"{band_config_path.name} 的 schema_status 必須是 {expected_status}，目前為 {schema_status}")

    band_config = BandConfig(
        leak_focused_bands=payload.get("leak_focused_bands", [],),
        resonance_bands=payload.get("resonance_bands", [],),
        offband_bands=payload.get("offband_bands", [],),
        temporal_band_names=payload.get("temporal_band_names", [],),
    )

    if not band_config.leak_focused_bands:
        raise ValueError("band config 缺少 leak_focused_bands")

    return band_config    


# ============================================================
# 讀上游特徵設定檔.json：決定「怎麼把 TDMS 原始訊號切成一段一段、算出頻譜」的參數
# ============================================================
def load_feature_extraction_config(config_path):
    # 讀檔＋解析 JSON
    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    # 檢查 schema 種類
    if payload.get("schema_name") != "feature_extraction_config":
        raise ValueError(
            "schema_name 必須是 feature_extraction_config"
        )

    # 一次列出「必要欄位清單」，再用 list comprehension 抓出缺了哪些
    required_keys = [
        "schema_version",
        "base_feature_version",
        "fs",
        "window_sec",
        "hop_sec",
        "nperseg",
        "max_points",
        "include_offband_summary",
    ]

    missing_keys = [key for key in required_keys if key not in payload]

    if missing_keys:
        raise ValueError(f"feature_extraction_config.json 缺少欄位：{missing_keys}")
        
    # 對「已經確定存在」的欄位，逐一檢查數值/型別是否合理
    if int(payload["schema_version"]) != 1:
        raise ValueError("目前只支援 feature_extraction_config schema_version=1")
    # 檢查取樣率 fs
    if float(payload["fs"]) <= 0:
        raise ValueError("fs 必須大於 0")
    # 檢查頻譜點數 nperseg
    if int(payload["nperseg"]) <= 0:
        raise ValueError("nperseg 必須大於 0")
    # 檢查上限點數 max_points
    if int(payload["max_points"]) <= 0:
        raise ValueError("max_points 必須大於 0")
    # 要不要計算 offband 摘要特徵,只要不是「真正的」JSON boolean（true/false），一律擋下來
    if not isinstance(payload["include_offband_summary"],bool,):
        raise ValueError("include_offband_summary 必須是 JSON boolean")
    # 檢查窗長 window_sec
    if float(payload["window_sec"]) <= 0:
        raise ValueError("window_sec 必須大於 0")
    # 檢查窗移 hop_sec
    if float(payload["hop_sec"]) <= 0:
        raise ValueError("hop_sec 必須大於 0")
    # 檢查hop_sec（滑動視窗每次往前移動的秒數）不可以大於 window_sec（每個視窗的長度）
    if float(payload["hop_sec"]) > float(payload["window_sec"]):
        raise ValueError("hop_sec 不可大於 window_sec")

    return payload


# ============================================================
# 計算設定檔 hash，避免 band/config 改版後誤讀舊快取
# ============================================================
def calculate_file_hash(path):
    '''
    計算檔案內容的 SHA-256。
    頻帶或特徵設定內容改變時，會產生不同的設定指紋。
    '''
    # 路徑正規化
    file_path = Path(path)
    
    # 檢查
    if not file_path.exists():
        raise FileNotFoundError(f"找不到要計算 hash 的檔案：{file_path}")
    
    # 建立一個 SHA-256 的雜湊器
    digest = hashlib.sha256()
    # 把整個檔案的「原始位元組」餵給雜湊器
    digest.update(path.read_bytes())

    return digest.hexdigest()


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
    return file_specs


# ============================================================
# 最底層負責把 TDMS 變成原始 window 特徵表
# ============================================================
def extract_base_feature_table(
    *,                         # 全關鍵字參數（*, 開頭），而且全部都沒有預設值——代表這十個參數呼叫端一個都不能漏
    file_specs,                # 整理好的 TDMS 清單
    cache_dir: Path,           # 快取資料夾
    split_name: str,           # 這批資料名稱
    band_config: BandConfig,   # low/core/aux bands 設定
    n_jobs: int,               # 平行處理數
    window_sec: float,         # 窗長
    hop_sec: float,            # hop 長
    fs: float,                 # 取樣率
    nperseg: int,              # Welch / PSD 相關設定
    max_points: int,           # 最大讀取點數
    feature_version: str,
):
    
    '''從 file_specs 抽出原始 window 特徵表。'''
    # 把語意版的 BandConfig（leak_focused_bands/resonance_bands/offband_bands）轉成 load_or_compute_features_by_file 實際看得懂的機械版格式（low_bands/core_bands/aux_bands）
    band_groups = resolve_runtime_band_groups(band_config)
    
    # 讀每個 TDMS -> 切成很多 window -> 依照 band_config 的頻帶去算各種基礎特徵 -> 把結果整理成一個 features 清單
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
    
    # 處理大量 TDMS 時盡快釋放記憶體
    del features
    gc.collect()
    return df


# ============================================================
# 決定「哪些欄位值得拿來建 baseline」
# ============================================================
def select_raw_feature_columns(df: pd.DataFrame):     # df：你前面從正常 TDMS 抽出來的特徵表
    # 先定義不該算進去的欄位
    exclude_cols = {
        "filename",    # 檔名
        "record_id",   # 資料編號
        "t_start",     # 窗起始時間
        "t_ref",       # 窗代表時間
        "t_end",       # 結束時間
        "label",
        "target",
        "window_idx",
        "source_index",
        "fs",
    }

    # 建立空清單, 等等把篩出來的 raw feature 名稱放進去
    raw_cols = []
    # 開始逐欄檢查
    for c in df.columns:
        # 如果是排除欄位(filename, record_id, t_start, t_ref)，就跳過
        if c in exclude_cols:
            continue
        # 如果不是數值欄位，也跳過
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        # 如果是 rel_、z_、cusum_ 類欄位，也跳過 -> 排除已經是「衍生後」的欄位
        # 建立場域 baseline 時，應該先針對最底層 raw feature 建 baseline
        if c.startswith("rel_") or c.startswith("z_") or c.startswith("cusum_"):
            continue
        # 如果是 delta_、local_、rolling_ 類欄位，也跳過
        if c.startswith("delta_") or c.startswith("local_") or c.startswith("rolling_"):
            continue
        # 真正的原始特徵欄位, 加進 raw_cols
        raw_cols.append(c)

    return raw_cols


# ============================================================
# 對所有候選原始特徵建立正常 baseline 統計
# ============================================================
def build_site_baseline_stats(df_features: pd.DataFrame,               # 正常資料的特徵表
                              raw_feature_cols: list[str]              # 剛剛 select_raw_feature_columns() 篩出來的 raw feature 名單
                              ) -> tuple[dict, list[dict]]:
    '''對每個 raw feature 算出 baseline 的 mean / std'''
    # 建立空 dictionary, 存放每個特徵的 baseline 統計量
    stats = {}
    invalid_features = []

    # 逐欄計算 baseline 統計量
    for col in raw_feature_cols:
        # 先把欄位轉成數值並去掉空值
        x = pd.to_numeric(df_features[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        
        # 如果這欄完全沒資料
        if x.empty:
            invalid_features.append({
                "feature_name": col,
                "reason": "沒有有效數值",
                "n_valid": 0,
            })
            continue

        mean_val = float(x.mean())       # 算平均值
        std_val = float(x.std(ddof=1))   # 算標準差

        # 如果標準差不合理
        if not np.isfinite(mean_val):
            invalid_features.append({
                "feature_name": col,
                "reason": "mean 不是有限數值",
                "n_valid": int(len(x)),
            })
            continue

        std_floor = max(abs(mean_val) * 1e-12,np.finfo(np.float64).tiny,)
        
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


        # 把這個特徵的 baseline 統計量存進字典
        stats[col] = {
            "mean": mean_val,
            "std": std_val,
            "n_valid": int(len(x)),
        }

    return stats, invalid_features


# ============================================================
# 算好的 baseline 存成 site_baseline.json
# ============================================================
def save_site_baseline(
    *,
    output_path,               # 最後 baseline 檔要存去哪裡
    baseline_stats,            # 每個 raw feature 算好的 mean / std
    tdms_paths,                # 這次拿來建 baseline 的正常 TDMS 清單
    band_config_path,          # 這次用的是哪份 band 設定
    feature_extraction_config_path,
    invalid_features,
    band_config_hash,
    feature_extraction_config_hash,
    config_fingerprint,
    raw_feature_cols,
    n_windows,
    window_sec,                # 這次建立 baseline 用的參數
    hop_sec,
    fs,
    nperseg,
    max_points,
    baseline_environment,      # 目前是實驗室資料
    baseline_schema_status,
    baseline_scope,
    base_feature_version,
):
    # 建立 payload：把所有 metadata（來源檔案、用的 config 路徑、訊號參數）連同算好的 baseline_stats 一起包成一個 payload 寫成 JSON
    payload = {
        "schema_name": "site_baseline",
        "schema_status": baseline_schema_status,
        "environment": baseline_environment,
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline_type": "fixed_normal_baseline",                   # 標記這份檔案的用途:場域正常 baseline
        "baseline_scope": baseline_scope,
        
        # 上游特徵契約
        "base_feature_version": base_feature_version,              # baseline 版本
        "band_config_path": str(Path(band_config_path)),
        "band_config_sha256":band_config_hash,
        "feature_extraction_config_path": str(Path(feature_extraction_config_path)),
        "feature_extraction_config_sha256": (feature_extraction_config_hash),
        "config_fingerprint":config_fingerprint,
        
        # 正常資料來源
        "source_tdms_count": len(tdms_paths),
        "source_tdms_paths": [str(Path(p)) for p in tdms_paths],
        
        # 訊號與切窗參數
        "window_sec": window_sec,                    
        "hop_sec": hop_sec,
        "fs": fs,
        "nperseg": nperseg,
        "max_points": max_points,
        "feature_stats": baseline_stats,                            # 每個 raw feature 的：mean, std
        "n_windows": int(n_windows),
                
        # baseline 特徵結果
        "candidate_raw_feature_columns":list(raw_feature_cols),     # 原始候選範圍
        "baseline_feature_columns":sorted(baseline_stats.keys()),   # 真正具有有效 baseline 統計的欄位              
        "invalid_features": invalid_features,
        
        }
    
    # 把輸出路徑轉成 Path
    output_path = Path(output_path)
    # 建立輸出資料夾
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 把 JSON 寫出去
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"{baseline_scope} baseline 已輸出："
        f"{output_path}"
    )

# ============================================================
# baseline 建立程式的主流程入口
# ============================================================
def main():
    # 設定資料來源(實驗室)
    baseline_environment = "laboratory"
    baseline_schema_status = "candidate"
    
    # 設定資料來源(現場)
    #baseline_environment = "site"
    #baseline_schema_status = "final"
    
    # --  正常資料路徑，並讀取  --
    # None：正式執行全部資料, 正式建立現場 Baseline 時要改成：BASELINE_TDMS_LIMIT = None
    # 30：只用前 30 支測試流程
    BASELINE_TDMS_LIMIT = None
    normal_data_dir = Path("/Users/paul/Desktop/Final/normal_leak_test/leak0")
    tdms_paths = sorted(str(p) for p in normal_data_dir.glob("*.tdms"))
    
    if BASELINE_TDMS_LIMIT is not None:
        tdms_paths = tdms_paths[:BASELINE_TDMS_LIMIT]
    
    if not tdms_paths:
        raise ValueError(f"在 {normal_data_dir} 找不到任何 .tdms 檔案")
    
    # ============================================================
    # 依 baseline 模式決定頻帶來源與輸出位置
    # ============================================================
    # 設定 result_root，這次執行所有輸出/輸入檔案的根目錄
    result_root = Path("/Users/paul/Desktop/Final/test_result")
    # 特徵上游設定檔路徑
    feature_extraction_config_path = (result_root / "feature_extraction_config.json")
    
    # 分支 1：對象是 baseline 建立流程,
    if BASELINE_MODE == "discovery":
        # 產生候選頻帶
        discovery_bands = generate_sliding_bands(
            f_min=DISCOVERY_F_MIN,
            f_max=DISCOVERY_F_MAX,
            band_width=DISCOVERY_BAND_WIDTH,
            stride=DISCOVERY_STRIDE,
        )
        # 組出 band_config.json 路徑
        band_config_path = (result_root / "band_config_step2_discovery.json")
        # 輸出路徑
        save_discovery_band_config(output_path=band_config_path, discovery_bands=discovery_bands,)
        output_path = (result_root / "discovery_baseline.json")
    
        baseline_scope = "discovery_scan_grid"
    
    # 分支 2：不產生新的滑動頻帶,而是直接指向已經存在的正式版 band config（band_config_step2.json
    elif BASELINE_MODE == "final":
        # band_config.json 路徑
        band_config_path = (result_root / "band_config_step2.json")
        # 輸出路徑
        output_path = (result_root / "site_baseline.json")    
        baseline_scope = "finalized_band_config"
        if not band_config_path.exists():
            raise FileNotFoundError(f"BASELINE_MODE=final，但找不到正式設定：{band_config_path}")
    
    else:
        raise ValueError(f"BASELINE_MODE 只能是 discovery 或 final，目前為：{BASELINE_MODE}")
    
    print("\n=== Baseline 建立模式 ===")
    print("mode:", BASELINE_MODE)
    print("scope:", baseline_scope)
    print("band config:", band_config_path)
    print("output:", output_path)

    # --  載入設定檔  --
    extraction_cfg = load_feature_extraction_config(feature_extraction_config_path)

    # 定義訊號處理參數（從 feature_extraction_config.json 讀取）
    window_sec = float(extraction_cfg["window_sec"])          # 每個窗多長
    hop_sec = float(extraction_cfg["hop_sec"])                # 每次滑動多少
    fs = float(extraction_cfg["fs"])                          # 取樣率
    nperseg = int(extraction_cfg["nperseg"])                  # 頻譜分析參數
    max_points = int(extraction_cfg["max_points"])            # 最多讀多少點
    include_offband_summary = bool(extraction_cfg["include_offband_summary"])    # 是否建立 offband、leak-focused、resonance 群組摘要
    base_feature_version = str(extraction_cfg["base_feature_version"])           # 基礎特徵計算公式版本，用於快取與設定指紋
    n_jobs = 2               # 平行處理數
      
    # ============================================================
    # 載入並驗證本次模式對應的 band config
    # ============================================================
    # Discovery 模式只能讀取尚未篩選的探索頻帶設定；
    # Final 模式只能讀取已經人工加入 Offband 並定案的正式設定。
    expected_band_status = ("discovery" if BASELINE_MODE == "discovery" else "final")
    band_config = load_band_config(band_config_path, expected_status=expected_band_status,)
    
    # ============================================================
    # Final 模式強制要求正式 Offband
    # ============================================================
    # 正式流程會使用 Offband 建立：
    # 1. Offband 原始特徵
    # 2. Offband 群組摘要
    # 3. Offband-to-leak ratio
    # 4. Disturbance Veto 抗干擾規則
    #
    # 因此 BASELINE_MODE="final" 時，band_config_step2.json 不允許缺少 Offband。
    # Discovery 模式尚未進行人工 Offband 定案，所以不執行這項檢查。
    if ( BASELINE_MODE == "final" and not band_config.offband_bands):
        raise ValueError(
            "BASELINE_MODE=final，但 band_config_step2.json 沒有 offband_bands；"
            "請先在候選 band config 中加入正式 Offband，並將結果定案為 band_config_step2.json。"
        )
    
    # 顯示本次實際載入的頻帶數量，方便核對設定
    print("\n=== Band config 驗證完成 ===")
    print("expected status:", expected_band_status)
    print("leak-focused bands:", len(band_config.leak_focused_bands),)
    print("resonance bands:", len(band_config.resonance_bands),)
    print("offband bands:", len(band_config.offband_bands),)
    print("temporal bands:", len(band_config.temporal_band_names),)
    
    # 整理正常 TDMS 檔案清單
    file_specs = build_file_specs(tdms_paths)
    
    # 把 band config 的 temporal band 名稱同步到
    # shared_feature_utils 的基礎 window 特徵提取器。
    configure_temporal_band_names(band_config.temporal_band_names)
    
    # --  快取名稱綁定設定檔版本  --
    band_config_hash = calculate_file_hash(band_config_path)
    feature_extraction_config_hash = calculate_file_hash(feature_extraction_config_path)
    
    fingerprint_source = (
        f"{BASELINE_MODE} : {baseline_scope} : {band_config_hash} : {feature_extraction_config_hash} : {base_feature_version}")
    
    config_fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    cache_split_name = (f"{BASELINE_MODE}_baseline_build_{config_fingerprint}")

    # --  特徵工程  --
    # 抽出最原始 base features:讀正常 TDMS -> 切窗 -> 依照 band config 算出原始 band 特徵表
    df_features = extract_base_feature_table(
        file_specs=file_specs,
        cache_dir=Path("/Users/paul/Desktop/runtime_outputs/site_baseline_cache"),
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
    
    # 先抓原始 raw base features（這時還沒加 summary）
    raw_feature_cols = select_raw_feature_columns(df_features)
    
    # 再建立 offband / leak_focus / resonance 的 summary base features
    build_summary_features = (include_offband_summary and BASELINE_MODE == "final")
    
    if build_summary_features:
        df_features = build_offband_summary_features(df_features, band_config,)
        
    # 加上 6 個摘要欄位（offband/leak_focus/resonance 各自的 mean/max），用 if c in df_features.columns 防呆（避免某個 summary 欄位因為 band_config 沒設某類 band 而沒被算出來，篩選時漏掉會直接 KeyError）
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
    
    # 計算 baseline 統計量
    baseline_stats, invalid_features = build_site_baseline_stats(df_features, baseline_feature_cols)

    print("baseline 候選特徵數：",len(baseline_feature_cols))
    print("baseline 有效特徵數：",len(baseline_stats))
    print("baseline 無效特徵數：",len(invalid_features))

    if invalid_features:
        print("\nbaseline 無效特徵：")
        for item in invalid_features:
            print(
                "-",
                item["feature_name"],
                ":",
                item["reason"],
            )

    # --  存檔  --
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
    )
    


if __name__ == "__main__":
    main()
    
