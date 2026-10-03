#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正式離線 / 即時推論流程
============================================================

目的：
將新進 TDMS 資料依照「已定版模型」對應的頻帶、特徵、baseline
與後處理規則，輸出每個滑動視窗與整支資料的洩漏判定結果。

版本一致性原則：
1. feature_config.json、band_config.json、runtime_feature_schema.json、
   model.pkl、postprocess_config.json 必須屬於同一模型版本。
2. feature_config["selected_feature_names"] 必須與
   runtime_feature_schema["model_feature_names"] 完全一致。
3. 若特徵清單不一致，直接中止，不可用新版特徵硬餵舊模型，
   也不可用舊頻帶硬湊新版模型欄位。

整體流程：

Step 1. 載入正式設定與模型配套檔案
    - band_config：頻帶定義
    - feature_config：window_sec、hop_sec、PSD、CUSUM 等特徵工程參數
    - runtime_feature_schema：模型與規則需要的正式欄位
    - site_baseline：正常場域的 baseline mean / std
    - postprocess_config：正式告警與抗干擾規則
    - metadata / model.pkl：訓練好的模型與模型 threshold

Step 2. 檢查設定版本一致性
    - 確認 feature_config 的 selected_feature_names
      與 runtime_feature_schema 的 model_feature_names 完全一致。
    - 不一致代表前段特徵設定已更新、但模型未同步重訓，
      必須恢復舊版設定或重跑完整特徵工程與模型訓練流程。

Step 3. 整理新進 TDMS
    - 將單一檔案或資料夾中的 TDMS 整理為 file_specs。
    - 每支檔案建立 record_id，作為後續逐檔切窗、推論與彙整依據。

Step 4. 建立基礎頻帶特徵
    - 讀取 TDMS 原始雙感測器訊號。
    - 依 window_sec 切窗、依 hop_sec 滑動更新。
    - 依 band_config 計算各頻帶能量、RMS、Entropy、Peak 等基礎特徵。
    - 每個 window 建立 t_start 與 t_ref。

Step 5. 建立正式 runtime 特徵
    - 套用 site_baseline 建立 rel_* 與 z_* 特徵。
    - 對指定 z_* 特徵建立 CUSUM 累積特徵。
    - 建立 delta、local、rolling 等時間動態特徵。
    - 建立 offband、leak_focus、resonance 摘要與頻帶比值。
    - 對摘要能量補建 rel_* 與 z_* 特徵。
    - 最後只保留模型、規則與時間識別所需欄位。

Step 6. 檢查正式欄位完整性
    - 確認所有 model_feature_names 都存在。
    - 確認所有 rule_feature_names 都存在。
    - 缺少任一欄位即中止，避免模型在錯誤特徵上推論。

Step 7. 模型逐窗推論
    - 以 model_feature_names 取出正式特徵矩陣。
    - 載入 model.pkl，輸出每個 window 的 leak_score 與 pred_label。
    - threshold 以模型 metadata / thresholds.json 為準。

Step 8. 告警規則與後處理
    - 依 postprocess_config 套用 consecutive、smooth 或其他 alarm rule。
    - 依 min_leak_duration_sec 排除過短告警。
    - 依 offband ratio 等規則標記 disturbance_suspect。
    - 每個 window 輸出 normal、leak 或 disturbance_suspect。

Step 9. 整支 TDMS 結果彙整與輸出
    - 將逐窗狀態彙整為每支 TDMS 的最終狀態。
    - 輸出逐窗推論表、逐檔摘要表、設定與執行資訊。

共用設計：
build_runtime_feature_table()
    TDMS -> 基礎頻帶特徵 -> 正式 runtime 特徵表

build_runtime_features_from_base_table()
    已切窗基礎特徵 -> baseline / CUSUM / temporal /
    summary 特徵 -> 正式 runtime 特徵表

第二個函式也供 buffer_sec.py 使用，
以確保離線 buffer 模擬與正式 runtime 的特徵工程邏輯一致。
"""


# ============================================================
# 載入套件
# ============================================================
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from collections import deque
from pathlib import Path


# ============================================================
# 專案路徑設定
# ============================================================
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
    remove_dc,
    configure_temporal_band_names,
    load_feature_extraction_config,
    calculate_file_sha256,
    validate_runtime_extractable_features,
)

from alarm_rule_utils import apply_alarm_method_by_record


# ============================================================
# 收集 TDMS 函式
# ============================================================
def collect_runtime_tdms_paths(path_inputs):
    """
    path_inputs 可以是：
    - 單一檔案路徑字串
    - 單一資料夾路徑字串
    - 多個檔案/資料夾組成的 list
    
    把使用者傳進來、格式可能很不一致的路徑輸入，統一整理成一份「確定存在、確定是 .tdms 檔案」的完整路徑清單
    """
    # 輸入格式的彈性處理
    if isinstance(path_inputs, (str, Path)):       # 先判斷 path_inputs 是不是「單一個」字串或 Path 物件
        path_inputs = [path_inputs]                # 如果是單一個，就包成只有一個元素的 list

    # 逐項檢查並收集路徑
    tdms_paths = []

    for item in path_inputs:
        p = Path(item)          # 每個項目轉成 Path 物件

        # 先確認這個路徑真的存在——如果使用者打錯路徑或檔案被移走了，這裡會直接丟出 FileNotFoundError
        if not p.exists():      
            raise FileNotFoundError(f"找不到路徑: {p}")
            
        # 如果這個路徑是「單一檔案」，就檢查副檔名（轉小寫比對，避免 .TDMS 這種大小寫差異被誤判）是不是 .tdms
        if p.is_file():
            if p.suffix.lower() != ".tdms":
                raise ValueError(f"不是 TDMS 檔案: {p}")
            tdms_paths.append(str(p))
        
        # 如果這個路徑是「資料夾」，就用 rglob("*.tdms") 遞迴（recursive glob）搜尋這個資料夾底下所有子資料夾裡的 .tdms 檔案
        elif p.is_dir():
            tdms_paths.extend(str(x) for x in sorted(p.rglob("*.tdms")))
            
    # 如果跑完整個迴圈後 tdms_paths 還是空的（例如給的資料夾裡根本沒有任何 .tdms 檔），就直接報錯提醒使用者
    if not tdms_paths:
        raise ValueError("找不到任何 .tdms 檔案")

    return tdms_paths


# ============================================================
# 帶入父資料夾
# ============================================================
def build_runtime_file_specs(tdms_paths):
    '''
    把上一步 collect_runtime_tdms_paths 收集到的純路徑字串清單，轉換成後面特徵工程管線真正需要的格式——每個檔案除了路徑之外，還要有一個能唯一識別它的 record_id
    '''
    file_specs = []
    
    # 存在性檢查
    for p in tdms_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"找不到 TDMS 檔案: {path}")

        # path.with_suffix("")：把副檔名拿掉, .parts：把整條路徑拆成一個一個資料夾名稱的 tuple, [-4:]：只取「最後 4 層」
        record_id = "__".join(path.with_suffix("").parts[-4:])      

        # 最後把每個檔案整理成一個 {"filepath": ..., "record_id": ...} 的字典，蒐集成 list 回傳
        file_specs.append(
            {
                "filepath": str(path),
                "record_id": record_id,
            }
        )

    return file_specs


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
    temporal_band_names: list[str]   # 目前先保留給未來 temporal feature gating 用，這支程式尚未實際使用
    

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
    

#                                                 ********************  config loaders  ********************
#                                                 **********************************************************
# ============================================================
# 讀 band 設定檔：config loaders
# ============================================================
def load_band_config(band_config_path):
    payload = json.loads(Path(band_config_path).read_text(encoding="utf-8"))
    return BandConfig(
        # 如果 JSON 裡有這個欄位，就讀它, 如果沒有，就先給空清單，避免直接報錯
        leak_focused_bands=payload.get("leak_focused_bands", []),
        resonance_bands=payload.get("resonance_bands", []),
        offband_bands=payload.get("offband_bands", []),
        temporal_band_names=payload.get("temporal_band_names", []),
    )


# ============================================================
# 讀 feature_config.json
# ============================================================
def load_feature_config(feature_config_path):
    """
    從 feature_config.json 讀正式特徵與參數設定
    它不只是讀 feature name，還會讀一些正式流程參數，例如：
    window_sec , hop_sec , cusum_k , cusum_decay , threshold , min_consecutive_windows
    """
    # 把輸入的路徑轉成 Path 物件
    feature_config_path = Path(feature_config_path)
    return json.loads(feature_config_path.read_text(encoding="utf-8"))     # 把 JSON 檔案讀成文字 -> 把文字轉成 Python dictionary


# ============================================================
# 讀取場域 baseline 設定檔
# ============================================================
def load_site_baseline(site_baseline_path):    # 輸入是：site_baseline_path 也就是你的 baseline 設定檔路徑
    """
    讀取場域 baseline 設定檔。
    裡面應包含每個 raw base feature 的 mean / std。
    """
    # 把路徑轉成 Path 物件
    site_baseline_path = Path(site_baseline_path)
    # 讀取 JSON 檔：先讀成文字字串，再轉成字典格式
    payload = json.loads(site_baseline_path.read_text(encoding="utf-8"))
    # 取出 feature_stats
    feature_stats = payload.get("feature_stats", {})
    
    # 檢查 feature_stats 是否存在
    if not feature_stats:
        raise ValueError("site_baseline.json 缺少 feature_stats")

    return payload


# ============================================================
# 讀取正式 runtime 後處理規則:模型已經說疑似 leak 了，接下來要不要真的判成 leak)
# ============================================================
def load_postprocess_config(postprocess_config_path):
    # 讀取整份 postprocess_config.json，轉成 Python dict
    payload = json.loads(Path(postprocess_config_path).read_text(encoding="utf-8"))
    # 確認裡面真的有 baseline_rule_settings 這個子物件，而且不是空的。如果沒有，就直接 raise 停掉
    rule_cfg = payload.get("baseline_rule_settings", {})
    if not rule_cfg:
        raise ValueError("postprocess_config.json 缺少 baseline_rule_settings")
    return payload


# ============================================================
# 專門處理「數值到底算不算『沒提供』」這件事
# ============================================================
def _clean_optional_value(v, cast=None):
    # 先擋 None，再用 pd.isna 擋 JSON 的 NaN，兩種缺值都會回傳 None；如果值是有效的，才用 cast 轉型
    if v is None:
        return None
    try:
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            return v
        if pd.isna(v):
            return None
    except Exception:
        pass
    return cast(v) if cast is not None else v


# ============================================================
# 整理正式 Runtime 使用的告警與抗干擾規則
# ============================================================
def build_runtime_postprocess_settings(feature_payload,            # 上游的 feature_config_micro.json
                                       baseline_rule_settings):    # postprocess_config.json 規則
    # 1. 先整理正式規則
    # 一次性把整個 baseline_rule_settings 掃過一遍，對每一個 key 的 value 都套用 _clean_optional_value(v)（沒給 cast，所以只做「缺值偵測」，不做型別轉換），組出一份全新的 rule dict
    rule = {
        k: _clean_optional_value(v)
        for k, v in dict(baseline_rule_settings).items()
    }
    
    # 2. 再讀取 method / two-stage
    method = str(rule.get("method", "consecutive"))     # 如果規則裡沒填 method，預設用 "consecutive"（連續視窗判斷）
    
    # 第一階段（stage1）:先對整條機率序列做一次「初判」，完全不套 veto
    stage1 = rule.get("stage1")
    # 第二階段（stage2）再用一個比較嚴格、需要更長時間或更強證據的規則,去確認「這是不是真的洩漏,而不是一時的雜訊突波」，veto 機制（用 offband/broadband 比值去否決）
    stage2 = rule.get("stage2")
    
    # 只有當 method == "two_stage" 時才檢查這兩個欄位一定要是 dict
    if method == "two_stage":
        if not isinstance(stage1, dict) or not isinstance(stage2, dict):
            raise ValueError("method=two_stage 時，postprocess_config.json 必須包含 stage1 與 stage2")
    
    # 3. Threshold
    # 取出 window_threshold, 如果規則裡沒有這個值，就退回用 feature_payload（也就是 feature_config_micro.json）裡的 threshold，再不行就用寫死的 0.5 保底
    window_threshold = _clean_optional_value(rule.get("window_threshold"), float)
    if window_threshold is None:
        window_threshold = float(feature_payload.get("threshold", 0.5))

    # 理論上「單一 window 分類門檻」跟「告警規則用的門檻」可以是不同數值，但因為 postprocess_config.json 裡的 baseline_rule_settings 只定義了 window_threshold，沒有另外定義 alarm_threshold，所以會直接把 alarm_threshold 設成跟 window_threshold 一樣
    alarm_threshold = _clean_optional_value(rule.get("alarm_threshold"), float)
    if alarm_threshold is None:
        alarm_threshold = window_threshold

    # 4. Alarm rule 參數
    # smooth_window 沒讀到的話，保底值是 1——代表「不做平滑」
    smooth_window = _clean_optional_value(rule.get("smooth_window"), int)
    if smooth_window is None:
        smooth_window = 1

    # consecutive 連續告警視窗數:沒讀到的話，退回 feature_config_micro.json 的 min_consecutive_windows，再不行才用寫死的 3
    consecutive = _clean_optional_value(rule.get("consecutive"), int)
    if consecutive is None:
        consecutive = int(feature_payload.get("min_consecutive_windows", 3))

    # 因為這四個只有在 method="n_of_m"（n_hits/m_window）或 method="hysteresis"（high_threshold/low_threshold）時才有意義
    n_hits = _clean_optional_value(rule.get("n_hits"), int)
    m_window = _clean_optional_value(rule.get("m_window"), int)
    high_threshold = _clean_optional_value(rule.get("high_threshold"), float)
    low_threshold = _clean_optional_value(rule.get("low_threshold"), float)

    # 5. 最短警報持續時間
    # min_leak_duration_sec 用來判斷「這段連續告警持續夠不夠久才算真正 leak」的秒數門檻
    min_leak_duration_sec = _clean_optional_value(rule.get("min_leak_duration_sec"), float)
    if min_leak_duration_sec is None:
        min_leak_duration_sec = float(feature_payload.get("min_leak_duration_sec", 0.4))

    # 6. Veto 規則(沒填預設 False)
    # use_veto：決定「要不要用 offband ratio 去否決模型判斷的 leak」
    use_veto = bool(rule.get("use_veto", False))
    veto_column = rule.get("veto_column")
    veto_threshold = _clean_optional_value(rule.get("veto_threshold"), float)
    veto_rules = rule.get("veto_rules") or []
    # 型別檢查,確保它真的是 list
    if not isinstance(veto_rules, list):
        raise ValueError("veto_rules 必須是 list")
    # 如果啟用了 veto（use_veto=True），但舊版跟新版兩種 veto 機制都沒有任何一種被設定（not veto_column 且 not veto_rules）,就報錯
    if use_veto and not veto_column and not veto_rules:
        raise ValueError("use_veto=True，但 postprocess_config.json  沒有 veto_column 或 veto_rules")
        
    # 使用單一數值 veto_column 時，沒有門檻就使用 0.5
    if use_veto and veto_column and veto_threshold is None:
        veto_threshold = 0.5

    # 7. 依 method 檢查必要參數
    if method == "n_of_m":
        if n_hits is None or m_window is None:
            raise ValueError("method=n_of_m 時必須提供 n_hits 與 m_window")

    if method == "hysteresis":
        if high_threshold is None or low_threshold is None:
            raise ValueError("method=hysteresis 時必須提供 high_threshold 與 low_threshold")
    
    # 確認 method 本身是這五種合法值之一
    supported_methods = {"threshold", "consecutive", "n_of_m", "hysteresis", "two_stage",}
    if method not in supported_methods:
        raise ValueError(f"不支援的 alarm method: {method}")

    return {
        "alarm_method": method,
        "window_threshold": float(window_threshold),
        "alarm_threshold": float(alarm_threshold),
        "smooth_window": int(smooth_window),
        "consecutive": int(consecutive),
        "n_hits": n_hits,
        "m_window": m_window,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
    
        "use_veto": use_veto,
        "veto_column": veto_column,
        "veto_threshold": veto_threshold,
        "veto_rules": veto_rules,
    
        "stage1": stage1,
        "stage2": stage2,
    
        "min_leak_duration_sec": float(min_leak_duration_sec),
    }


# ============================================================
# 讀取正式 runtime 欄位契約
#
# candidate schema 由 disturbance inference 產生；
# 正式 runtime_feature_schema.json 則在 transition
# 驗證正式規則成功後產生。
# ============================================================
def load_runtime_feature_schema(schema_path):
    schema_path = Path(schema_path)     # 轉成路徑物件

    # 讀取並解析 JSON
    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    # 禁止使用候選 schema:正式 runtime 禁止載入 candidate schema
    if payload.get("schema_status") != "final":
        raise ValueError(f"runtime 只能載入 schema_status=final 的 schema，目前檔案: {schema_path}")

    # 確認 schema 類型
    # 正式檔案應包含："schema_name": "runtime_feature_schema"  ,  候選檔案則應是："schema_name": "runtime_feature_schema_candidate"
    if payload.get("schema_name") != "runtime_feature_schema":
        raise ValueError("schema_name 必須是 runtime_feature_schema")

    # 確認有模型 ID
    if not payload.get("model_id"):       # model_id 用來標示這份 schema 屬於哪一顆模型
        raise ValueError("runtime_feature_schema.json 缺少 model_id")

    # 確認有模型特徵清單
    if not payload.get("model_feature_names"):
        raise ValueError("runtime_feature_schema.json 缺少 model_feature_names")

    return payload


# ============================================================
# 一堆預設參數整理成一包統一設定
# ============================================================
def load_runtime_configs(
    *,                                            # 代表這個函式的所有參數都必須用關鍵字傳入
    band_config_path=None,                        # 正式推論真正要讀的頻段設定檔
    feature_extraction_config_path=None,
    feature_config_path=None,                     # 正式推論真正要讀的特徵設定檔
    postprocess_config_path=None,                 # 正式推論真正要讀的offband設定檔
    runtime_feature_schema_path=None,             # 這顆模型實際需要哪些模型欄位與規則欄位
    site_baseline_path=None,                      # 場域正常 baseline 設定檔
    fallback_window_sec=0.2,             
    fallback_hop_sec=0.05,
    fallback_fs=1_000_000,
    fallback_nperseg=2048,
    fallback_max_points=8_000_000,
    fallback_cusum_k=0.02,
    fallback_cusum_decay=0.97,
    fallback_include_offband_summary=True,
):
    '''負責讀取所有設定檔、檢查版本一致性，最後整理成一個 runtime_cfg 字典給後面流程使用'''
    # 確認有 band config
    if band_config_path is None:
        raise ValueError("正式 runtime 流程需要 band_config_path")
    
    # 確認有 feature_config_path
    if feature_config_path is None:
        raise ValueError("正式 runtime 流程需要 feature_config_path")
        
    # 確認有 feature_extraction_config_path
    if feature_extraction_config_path is None:
        raise ValueError("正式 runtime 流程需要 feature_extraction_config_path")

    # 確認有 postprocess_config 
    if postprocess_config_path is None:
        raise ValueError("正式 runtime 流程需要 postprocess_config_path")
    
    # 確認有正式離線版的最終資料
    if runtime_feature_schema_path is None:
        raise ValueError("正式 runtime 流程需要 runtime_feature_schema_path")

    # 確認有透過正常資料建立的 baseline
    if site_baseline_path is None:
        raise ValueError("正式 runtime 需要 site_baseline.json，不能省略")
        
    # 讀取5份設定檔
    band_config = load_band_config(band_config_path)
    extraction_cfg = load_feature_extraction_config(feature_extraction_config_path)
    feature_payload = load_feature_config(feature_config_path)
    postprocess_payload = load_postprocess_config(postprocess_config_path)
    runtime_feature_schema = load_runtime_feature_schema(runtime_feature_schema_path)
    site_baseline_payload = load_site_baseline(site_baseline_path)                           # 包含正常狀態下各特徵的平均值與標準差
    
    # 把 band config 的 temporal band 名稱,同步給 shared_feature_utils 的 raw feature extractor。
    configure_temporal_band_names(band_config.temporal_band_names)
    
    # 後處理規則整理：把 postprocess JSON 中目前選定的正式規則整理成容易使用的數值
    baseline_rule_settings = postprocess_payload["baseline_rule_settings"]     # 取出 baseline_rule_settings
    postprocess_settings = build_runtime_postprocess_settings(
        feature_payload,
        baseline_rule_settings,
    )

    # ============================================================
    # 特徵版本一致性檢查
    # ============================================================
    config_feature_names = feature_payload.get("selected_feature_names",[])             # feature_config_micro.json 裡記錄的模型特徵清單。
    model_feature_names = runtime_feature_schema.get("model_feature_names",[])          # runtime_feature_schema.json 裡記錄的模型實際輸入欄位。
    current_extraction_hash = calculate_file_sha256(feature_extraction_config_path)     # 計算目前 runtime 讀進來的 feature_extraction_config.json 檔案指紋。

    # # 確認 feature_config_micro.json 是基於目前這份 feature_extraction_config.json 建立的。
    # 如果 hash 不一致，代表模型特徵設定和現在 runtime 的底層特徵萃取參數不同步。
    if feature_payload.get("feature_extraction_config_sha256") != current_extraction_hash:
        raise ValueError("feature_config_micro.json 與目前 feature_extraction_config.json 不一致")
    
    # 確認 site_baseline.json 也是基於目前這份 feature_extraction_config.json 建立的。
    # baseline 的 mean/std 必須和 runtime 現在的切窗、頻譜參數、特徵公式一致。
    if site_baseline_payload.get("feature_extraction_config_sha256") != current_extraction_hash:
        raise ValueError("site_baseline.json 與目前 feature_extraction_config.json 不一致")
    
    # feature_config_micro.json 在不同版本中可能使用不同欄位名稱記錄底層特徵版本。
    feature_base_version = (
        feature_payload.get("base_feature_version") or feature_payload.get("baseline_base_feature_version"))
    
    # 如果兩種欄位都沒有，代表 feature_config_micro.json 沒有記錄底層特徵公式版本。
    # 這種情況無法保證 runtime 和模型訓練時的特徵計算方式一致，所以直接停止。
    if feature_base_version is None:
        raise ValueError("feature_config_micro.json 缺少 base_feature_version 或 baseline_base_feature_version")
    
    # 確認 feature_config_micro.json 記錄的底層特徵版本，和目前 feature_extraction_config.json 的 base_feature_version 一致。
    if feature_base_version != extraction_cfg.get("base_feature_version"):
        raise ValueError(
            "feature_config_micro.json 與 feature_extraction_config.json 的 base_feature_version 不一致："
            f"feature_config={feature_base_version}, "
            f"extraction_config={extraction_cfg.get('base_feature_version')}"
        )
        
    # 確認 site_baseline.json 的底層特徵版本也和 feature_extraction_config.json 一致
    if site_baseline_payload.get("base_feature_version") != extraction_cfg.get("base_feature_version"):
        raise ValueError("site_baseline.json 與 feature_extraction_config.json 的 base_feature_version 不一致")
        
    # ============================================================
    # 模型特徵欄位一致性檢查
    # ============================================================
    # 檢查欄位一致：只要 band／feature config 被新版流程覆寫、但模型沒有同步重訓，程式會立刻停止，而不是後面才顯示「缺少模型欄位」
    if not config_feature_names:
        raise ValueError("feature_config.json 缺少 selected_feature_names")
    
    # 如果 runtime_feature_schema.json 沒有 model_feature_names，代表部署 schema 不完整，runtime 不知道模型實際需要哪些欄位。
    if not model_feature_names:
        raise ValueError("runtime_feature_schema.json 缺少 model_feature_names")
    
    # 確認 feature_config_micro.json 的 selected_feature_names 和 runtime_feature_schema.json 的 model_feature_names 完全一致。
    if config_feature_names != model_feature_names:
        raise ValueError(
            "feature_config 與模型 schema 的特徵清單不一致。"
            "請使用同一版 band config / feature config 重新做特徵工程與模型訓練，或恢復此模型訓練時使用的設定檔。"
        )
    # 通過檢查後兩者已確認相同，因此以模型 schema 的清單作為正式依據
    selected_feature_names = model_feature_names
    
    # 如果沒有 selected features，就直接報錯
    if not selected_feature_names:
        raise ValueError("正式模型沒有可用的特徵欄位")
    # 從 selected features 回推出前面要先算哪些上游特徵
    raw_base_features, z_base_features, final_keep_features = parse_required_features(selected_feature_names)

    # 檢查 runtime parser 能不能解析全部 raw features
    validate_runtime_extractable_features(raw_base_features=raw_base_features, band_config=band_config)

    return {
        "band_config": band_config,                                               # 正式要使用的 band 設定(leak_focused_bands/resonance_bands/offband_bands/temporal_band_names)
        "feature_payload": feature_payload,                                       # 保留整份原始 feature config，方便後面如果還想再讀其他欄位
        "selected_feature_names": selected_feature_names,                         # 模型最終要吃的特徵欄位清單
        "raw_base_features": raw_base_features,                                   # 最底層要先算出來的原始特徵（衍生特徵的上游）
        "z_base_features": z_base_features,                                       # 哪些 z_* 欄位後面還要再做 CUSUM
        "final_keep_features": final_keep_features,                               # 最後輸出表要保留的欄位
        
        # ========================================================
        # 底層特徵萃取參數(這些參數正式來源是 feature_extraction_config.json)
        # ========================================================
        # window / PSD 相關參數
        "window_sec": float(extraction_cfg.get("window_sec", fallback_window_sec)),     # 切窗長度 
        "hop_sec": float(extraction_cfg.get("hop_sec", fallback_hop_sec)),              # 移動步長  
        "fs": float(extraction_cfg.get("fs", fallback_fs)),                             # 取樣率
        "nperseg": int(extraction_cfg.get("nperseg", fallback_nperseg)),              # 做頻譜分析（Welch/PSD）時每段的點數
        "max_points": int(extraction_cfg.get("max_points", fallback_max_points)),     # 讀 TDMS 時最多讀多少點，避免超大檔案吃爆記憶體
        
        # ========================================================
        # CUSUM 參數
        # ========================================================
        # CUSUM 參數
        "cusum_k": feature_payload.get("cusum_k", fallback_cusum_k),              # 累積偏移特徵的靈敏度跟衰減係數
        "cusum_decay": feature_payload.get("cusum_decay", fallback_cusum_decay),
        
        # 是否建立 offband / leak-focused / resonance 的 summary features。
        "include_offband_summary": bool(
            extraction_cfg.get("include_offband_summary", fallback_include_offband_summary)
        ),
        
        # 記錄這次 runtime 實際使用的底層特徵版本與設定檔指紋。
        # 後面輸出 run_info 時可以用來追蹤這次推論到底用了哪一版特徵工程。
        "base_feature_version": extraction_cfg.get("base_feature_version"),
        "feature_extraction_config_path": str(feature_extraction_config_path),
        "feature_extraction_config_sha256": current_extraction_hash,
        
        # 現場場域正常 baseline
        "site_baseline_payload": site_baseline_payload,
        
        # runtime baseline fallback 參數：若沒有 site baseline，正式推論才會改用新資料前幾窗或前幾秒自行估 baseline
        "baseline_mode": feature_payload.get("baseline_mode", "first_n_windows"),     # 沒有 site baseline 時，用「前幾窗」還是「前幾秒」自估 baseline
        "baseline_n_windows": feature_payload.get("baseline_n_windows", 5),           # 前幾窗模式要看幾個窗
        "baseline_duration_sec": feature_payload.get("baseline_duration_sec", None),  # 前幾秒模式要看幾秒
        
        # 模型輸出欄位
        "score_col": feature_payload.get("score_col", "leak_score"),   # 模型分數要存在哪個欄位名（預設 "leak_score"）
        "pred_col": feature_payload.get("pred_col", "pred_label"),     # 模型 0/1 預測要存在哪個欄位名（預設 "pred_label"）
        
        # 正式後處理規則參數
        "window_threshold": postprocess_settings["window_threshold"],  # 單一 window 判斷洩漏的機率門檻
        "alarm_method": postprocess_settings["alarm_method"],          # 告警邏輯用哪一種
        "alarm_threshold": postprocess_settings["alarm_threshold"],    # 告警規則本身用的門檻
        "smooth_window": postprocess_settings["smooth_window"],        # 平滑視窗大小
        "consecutive": postprocess_settings["consecutive"],            # 連續命中法要求的連續窗數
        "n_hits": postprocess_settings["n_hits"],                      # n_of_m 方法專用，目前用不到
        "m_window": postprocess_settings["m_window"],
        "high_threshold": postprocess_settings["high_threshold"],      # hysteresis 方法專用，目前用不到
        "low_threshold": postprocess_settings["low_threshold"],
        
        # veto / 干擾規則相關
        "use_veto": postprocess_settings["use_veto"],                       # 是否啟用 offband ratio 否決機制
        "veto_column": postprocess_settings["veto_column"],
        "veto_threshold": postprocess_settings["veto_threshold"],
        "veto_rules": postprocess_settings["veto_rules"],
        "min_consecutive_windows": postprocess_settings["consecutive"],     # 連續命中法要求的連續窗數
        "min_leak_duration_sec": postprocess_settings["min_leak_duration_sec"],    # apply_postprocessing 判斷「這段連續告警要持續多久才算真正 leak」的秒數門檻
            
        # two-stage
        "stage1": postprocess_settings["stage1"],
        "stage2": postprocess_settings["stage2"],
        
        # offband
        "postprocess_payload": postprocess_payload,        # 整份 postprocess_config.json 原始內容
        "baseline_rule_settings": baseline_rule_settings,  # postprocess_payload["baseline_rule_settings"] 這個子物件，方便直接存取
        
        # 正式離線需要的特徵
        "runtime_feature_schema": runtime_feature_schema,       # 整份 runtime_feature_schema.json 原始內容
        "model_feature_names": runtime_feature_schema.get("model_feature_names", []),      # 從 schema 裡拆出來的三份欄位清單
        "rule_feature_names": runtime_feature_schema.get("rule_feature_names", []),        # 從 schema 裡拆出來的三份欄位清單
        "all_required_columns": runtime_feature_schema.get("all_required_columns", []),    # 從 schema 裡拆出來的三份欄位清單
    }


#                                           ********************  feature engineering utils  ********************
#                                           *********************************************************************
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
        feature_version="v2_global_dc",
        max_points=max_points,
    )
    
    # 轉成 DataFrame
    df = pd.DataFrame(features).reset_index(drop=True)
    del features
    gc.collect()
    return df


# ============================================================
# 從 selected features 反推出前面至少要先算哪些基礎特徵
# ============================================================
def parse_required_features(selected_feature_names: list[str]):    # 輸入是模型最後真正要用的特徵名稱清單
    """
    從 selected features 回推出：
    1. 一開始至少要先算哪些 raw 特徵
    2. 哪些 z_ 欄位後面還要拿去做 CUSUM
    3. 最後要保留哪些欄位
    """
    raw_base_features = set()    # 先建立一個空集合, 存放最原始、最先要先算出來的特徵欄位
    z_base_features = set()      # 建立一個空集合，存放哪些 z_... 欄位後面還需要再往上做 CUSUM
    final_keep_features = set(selected_feature_names)     # 把模型最後選中的欄位存成集合

    # 列出：哪些前綴代表「這是一種衍生特徵」,這張表的用途是：看到這些前綴時，就知道要把前綴拿掉，回推出原始 base feature 是誰
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

    # 逐一檢查模型最後選到的每個特徵名稱, 一個一個看：這個欄位是原始欄位嗎？還是衍生欄位？如果是衍生欄位，它的上游原始欄位是誰？
    for feat in selected_feature_names:
        # 1.處理 CUSUM 特徵：CUSUM 這種欄位比較多包一層
        if feat.startswith("cusum_pos_") or feat.startswith("cusum_neg_"):
            # 如果是 cusum 特徵，先把最前面的 cusum_pos_ 或 cusum_neg_ 去掉
            z_feat = feat.replace("cusum_pos_", "", 1).replace("cusum_neg_", "", 1)
            # 再檢查這個 cusum 底下是不是接 z_...
            if z_feat.startswith("z_"):
                # 如果 z_feat 是像 z_xxx，那就把 z_ 去掉，取出原始特徵 xxx
                raw_base_features.add(z_feat[len("z_"):])
                # 也把這個 z_... 欄位記下來，後面我不只要先算 raw xxx，還要再算 z_xxx，因為它還要拿去做 CUSUM
                z_base_features.add(z_feat)
            continue

        # 先設一個旗標 matched = False。等等如果這個 feature 符合某個前綴，就會把它改成 True。
        matched = False
        # 逐一檢查剛剛那張 prefix_map 裡的每一種前綴
        for prefix in prefix_map:
            # 如果這個特徵名稱是以某個已知前綴開頭，代表它是衍生特徵
            if feat.startswith(prefix):
                # 把前綴拿掉，只留下原始 base feature 名稱
                raw_base_features.add(feat[len(prefix):])
                matched = True    # 把 matched 設成 True
                break

        # 如果前面全部前綴都沒對上，表示這個 feat 本身就可能是原始特徵，不是衍生特徵。
        if not matched:
            raw_base_features.add(feat)    # 直接把它自己加進 raw_base_features

    return (
        sorted(raw_base_features),      # 一開始至少要先算哪些原始特徵
        sorted(z_base_features),        # 哪些 z_ 特徵後面還要再做 CUSUM
        sorted(final_keep_features),    # 模型最後真正要保留哪些欄位
    )


# ============================================================
# 建立時間序列特徵
# ============================================================
def add_window_dynamic_features(
    df,                      # 原始特徵表
    feature_cols,            # 哪些欄位要做這些動態特徵
    group_col="record_id",   # 以哪個 record 為單位分組
    time_col="t_ref",        # 時間排序依據
    prev_windows=(3, 5),     # 要回看前 3 個窗、前 5 個窗
):
    """建立 delta 與相對前幾窗平均的變化量特徵：delta_* , delta_prev3mean_* , delta_prev5mean_*。"""
    out = df.copy()    # 先複製一份 DataFrame，避免直接改原始資料
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)     # 先按照：record_id , t_ref 排序。

    # 過濾真正可用的欄位:這個欄位名真的存在, 而且它是數值欄位
    valid_cols = [
        c for c in feature_cols
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]

    # 依照 record_id 分組:每一筆 record 的時間序列要分開算，不能把不同 record 接在一起算時間差
    for _, sub in out.groupby(group_col, sort=False):
        idx = sub.index     # 先把這個 group 在原本 DataFrame 裡對應的索引位置存起來。等等算出來的結果要寫回 out 時，會用到
        
        # 逐個特徵欄位處理
        for col in valid_cols:
            x = pd.to_numeric(sub[col], errors="coerce")    # 欄位轉成數值型態
            out.loc[idx, f"delta_{col}"] = x.diff()         # 建立：相鄰兩個 window 的差值。也就是：本窗值 - 前一窗值

            # 再依序處理：回看前 3 個窗 , 回看前 5 個窗
            for k in prev_windows:
                # 計算：前 k 個窗的平均值。先 shift(1) 是為了排除「當前窗」本身，只看之前的資料
                prev_mean = x.shift(1).rolling(k, min_periods=1).mean()
                out.loc[idx, f"delta_prev{k}mean_{col}"] = x - prev_mean    # 建立：目前這一窗，相對於前 k 窗平均偏了多少

    return out


# ============================================================
# 對每個原始特徵欄位，再往上算一批「局部時間統計特徵」
# ============================================================
def add_local_rolling_features(
    df,                       # 原始特徵表
    feature_cols,             # 哪些欄位要做這些時間特徵
    group_col="record_id",    # 以哪個 record 分組
    time_col="t_ref",         # 時間排序依據
    windows=(3, 5),           # 要看前 3 窗、前 5 窗
):
    """建立 local relative / local z / rolling trend 類特徵。"""
    out = df.copy()     # 先複製一份資料，避免直接改原始 df
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)     # 先按照：record_id , t_ref 排序。
    
    # 過濾真正能算的欄位:欄位真的存在 , 欄位是數值型
    valid_cols = [
        c for c in feature_cols
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]

    # 依 record_id 分組。每一筆 record 的時間序列分開算，不把不同 record 混在一起。
    for _, sub in out.groupby(group_col, sort=False):
        idx = sub.index    # 記住這個分組在原表中的索引，等等算完後可以把結果寫回 out
        # 開始逐個特徵欄位處理
        for col in valid_cols:
            # 把這個欄位轉成數值格式,如果有奇怪值就變成 NaN
            x = pd.to_numeric(sub[col], errors="coerce")
            
            # 依序處理：回看 3 窗 , 回看 5 窗
            for k in windows:
                prev = x.shift(1)     # 把序列往後移一格。意思是：只看前面的窗，不包含目前這一窗自己。
                prev_mean = prev.rolling(k, min_periods=1).mean()          # 算前 k 個窗的平均值
                prev_std = prev.rolling(k, min_periods=2).std(ddof=1)      # 算前 k 個窗的標準差
                rolling_std = x.rolling(k, min_periods=2).std(ddof=1)      # 算「包含當前窗」的 rolling standard deviation:最近這一小段的波動程度
                rolling_max = x.rolling(k, min_periods=1).max()            # 算最近 k 窗中的最大值。比較：當前值離最近最大值差多少。

                # 建立 local_rel 特徵
                out.loc[idx, f"local_rel{k}_{col}"] = x - prev_mean    # 當前值 - 前 k 窗平均
                out.loc[idx, f"local_z{k}_{col}"] = (x - prev_mean) / (prev_std + 1e-12)    # 建立 local_z 特徵:(當前值 - 前 k 窗平均) / 前 k 窗標準差
                out.loc[idx, f"rolling_std{k}_{col}"] = rolling_std       # 建立 rolling_std 特徵：最近 k 窗整體波動大不大。
                out.loc[idx, f"rolling_max_gap{k}_{col}"] = x - rolling_max     # 建立 rolling_max_gap 特徵：當前值 - 最近 k 窗最大值

                # 算 rolling_slope:對最近 k 個點做一條直線擬合，取它的斜率，洩漏訊號是否有持續上升趨勢，干擾是不是只是瞬間尖峰
                # 如果斜率是正的：代表最近在往上升
                slope = x.rolling(k, min_periods=2).apply(
                    lambda arr: np.polyfit(np.arange(len(arr)), arr, 1)[0]
                    if np.isfinite(arr).sum() >= 2 else np.nan,
                    raw=True,
                )
                out.loc[idx, f"rolling_slope{k}_{col}"] = slope

    return out


# ============================================================
# 把前面兩種 temporal 特徵都一起加上去
# ============================================================
def add_common_temporal_features(df, raw_base_features):
    """transition / disturbance 共用的 temporal 特徵層。"""
    # 先從 raw_base_features 裡挑出：真的存在於 df，而且是數值型的欄位。
    temporal_base_cols = [
        c for c in raw_base_features
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 先呼叫前面那個 add_window_dynamic_features(...)。先建立：delta_* , delta_prev3mean_* , delta_prev5mean_*
    df = add_window_dynamic_features(
        df=df,
        feature_cols=temporal_base_cols,
        group_col="record_id",
        time_col="t_ref",
    )
    
    # 呼叫剛剛那個 add_local_rolling_features(...)。再建立：local_rel* , local_z* , rolling_std* , rolling_max_gap* , rolling_slope*
    df = add_local_rolling_features(
        df=df,
        feature_cols=temporal_base_cols,
        group_col="record_id",
        time_col="t_ref",
    )
    return df


# ============================================================
# 將已切窗的基礎頻帶特徵表，轉成正式 runtime 特徵表
# ============================================================
def build_runtime_features_from_base_table(base_df, runtime_cfg):
    """
    將已切窗的基礎頻帶特徵表，轉成正式 runtime 特徵表。
    可供完整 TDMS 與最近 H 秒 rolling buffer 共用。
    """
    df = base_df.copy()

    # 確認必要欄位存在(後面所有 temporal feature 和後處理都需要：record_id , t_start)
    if "record_id" not in df.columns or "t_start" not in df.columns:
        raise ValueError("base_df 必須包含 record_id 與 t_start")
    # 如果沒有 t_ref，就自己補(t_ref 通常表示：這個 window 的代表時間點，這裡用：window 起點 + 半個 window 長度，設成窗中心時間。)
    if "t_ref" not in df.columns:
        df["t_ref"] = (
            pd.to_numeric(df["t_start"], errors="coerce")
            + runtime_cfg["window_sec"] / 2.0
        )

    # 先按時間排序，後面才會依時間順序正常運作
    df = df.sort_values(["record_id", "t_ref"]).reset_index(drop=True)

    # 建立 runtime baseline relative 特徵:用前幾窗或前幾秒當 baseline, 建立 rel_*, 建立 z_*
    df = build_runtime_baseline_features(
        df=df,
        raw_base_features=runtime_cfg["raw_base_features"],
        site_baseline_payload=runtime_cfg["site_baseline_payload"],
        group_col="record_id",
        time_col="t_ref",
        baseline_mode=runtime_cfg["baseline_mode"],
        baseline_n_windows=runtime_cfg["baseline_n_windows"],
        baseline_duration_sec=runtime_cfg["baseline_duration_sec"],
    )

    # 找出哪些 z_* 需要再做 CUSUM
    needed_z_cols = [
        f"z_{c}"
        for c in runtime_cfg["raw_base_features"]
        if f"z_{c}" in runtime_cfg["z_base_features"]
    ]

    # 如果有需要，就加 CUSUM, 把前面選定的 z_* 欄位做成：cusum_pos_* , cusum_neg_* , 這種累積偏移特徵。
    if needed_z_cols:
        df = add_cusum_features(
            df=df,
            input_feature_cols=needed_z_cols,
            k=runtime_cfg["cusum_k"],
            decay=runtime_cfg["cusum_decay"],
        )
        
    # 再補上其他 temporal 特徵:會再加：delta_* , delta_prev3mean_* , local_rel* , local_z* , rolling_std* , rolling_slope* 這些時間變化特徵。
    df = add_common_temporal_features(df, runtime_cfg["raw_base_features"])

    # 如果設定有開，就建立 offband / leak / resonance 摘要特徵
    if runtime_cfg["include_offband_summary"]:
        df = build_offband_summary_features(df, runtime_cfg["band_config"])
        
        # 把多個頻帶整理成「洩漏能量、共振能量、非目標頻帶能量，以及它們的比例」
        summary_base_cols = [
        c for c in [
            "offband_E_mean", "offband_E_max",
            "leak_focus_E_mean", "leak_focus_E_max",
            "resonance_E_mean", "resonance_E_max",
        ]
        if c in df.columns
    ]

    if summary_base_cols:
        # 替這些摘要能量建立特徵
        df = build_runtime_baseline_features(
            df=df,
            raw_base_features=summary_base_cols,
            site_baseline_payload=runtime_cfg["site_baseline_payload"],
            group_col="record_id",
            time_col="t_ref",
            baseline_mode=runtime_cfg["baseline_mode"],
            baseline_n_windows=runtime_cfg["baseline_n_windows"],
            baseline_duration_sec=runtime_cfg["baseline_duration_sec"],
        )

    runtime_extra_cols = [
        "offband_E_mean", "offband_E_max",
        "leak_focus_E_mean", "leak_focus_E_max",
        "resonance_E_mean", "resonance_E_max",
        "offband_to_leak_focus_E_ratio",
        "offbandmax_to_leak_focusmax_E_ratio",
        "offband_to_resonance_E_ratio",
        "offbandmax_to_resonancemax_E_ratio",
        "rel_offband_E_mean", "z_offband_E_mean",
        "rel_offband_E_max", "z_offband_E_max",
        "rel_leak_focus_E_mean", "z_leak_focus_E_mean",
        "rel_leak_focus_E_max", "z_leak_focus_E_max",
        "rel_resonance_E_mean", "z_resonance_E_mean",
        "rel_resonance_E_max", "z_resonance_E_max",
    ]
    
    # 保留每個視窗的識別資訊與時間資訊，讓後續知道這列特徵來自哪支檔案、哪個時間點
    base_output_cols = ["filename", "record_id", "t_start", "t_ref"]
    
    # 合併需要輸出的欄位
    requested_output_cols = (
        base_output_cols                          # 基本識別與時間欄位
        + runtime_cfg["final_keep_features"]      # 模型訓練時決定的正式特徵欄位
        + runtime_extra_cols                      # 規則與除錯需要的額外摘要欄位
    )
    
    # 去除重複欄位，同時保留第一次出現的順序
    output_cols = list(dict.fromkeys(
        c for c in requested_output_cols
        if c in df.columns                        # 某些設定沒有產生某欄位時，不會直接報錯
    ))
    
    return df[output_cols].copy()


# ============================================================
# 正式 runtime 建立 baseline relative 特徵 , 優先使用 site baseline，沒有才 fallback 到前幾個窗
# ============================================================
def build_runtime_baseline_features(
    df,                                 # 目前已經算出基礎 band 特徵的表
    raw_base_features,                  # 哪些原始特徵要拿來做 rel_* 和 z_*
    site_baseline_payload=None,
    group_col="record_id",              # 每筆訊號的分組欄位
    time_col="t_ref",                   # 時間排序欄位
    baseline_mode="first_n_windows",    # baseline 要怎麼選
    baseline_n_windows=5,               # 如果用前幾窗，就預設前 5 窗
    baseline_duration_sec=None,         # 如果用前幾秒，就看這個秒數
):
    """
    建立正式 runtime 的 rel_* 與 z_* 特徵。

    優先順序：
    1. 若提供 site_baseline.json，優先使用場域正常資料建立的 mean / std
    2. 若沒有 site baseline，才 fallback 到每筆新資料前幾窗自行估 baseline
    """
    # 先複製一份，避免直接改原始 df。然後把資料按照：record_id , t_ref 排序。
    out = df.copy()
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)

    # 避免後面做除法時分母太小或等於 0
    eps = 1e-12

    # 過濾真正可以拿來做 baseline relative 的欄位
    # 條件是：
    #   欄位名稱在 raw_base_features 裡
    #   這個欄位真的存在於 out
    #   這個欄位是數值型
    valid_cols = [
        c for c in raw_base_features
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]
    
    # 如果有場域 baseline 優先使用
    feature_stats = {}
    if site_baseline_payload is not None:
        feature_stats = site_baseline_payload.get("feature_stats", {})

    # 開始逐筆訊號分開處理
    for _, sub in out.groupby(group_col, sort=False):
        idx = sub.index      # 記住這一組資料在原表中的索引位置，等等好把結果寫回去
        t = pd.to_numeric(sub[time_col], errors="coerce")    # 把時間欄位轉成數值型，方便做 baseline 篩選

        # baseline 怎麼選
        # 如果設定的是 first_seconds，就代表：把每筆訊號最前面幾秒當 baseline
        if baseline_mode == "first_seconds" and baseline_duration_sec is not None:
            baseline_mask = t <= float(baseline_duration_sec)
            # 如果因為某些原因，前幾秒根本沒有選到任何窗，那就退回成：至少拿前 baseline_n_windows 個窗來當 baseline
            if baseline_mask.sum() == 0:
                baseline_mask = pd.Series(False, index=sub.index)
                baseline_mask.iloc[: max(1, baseline_n_windows)] = True
        # 如果不是 first_seconds 模式，那就走預設：直接拿最前面 baseline_n_windows 個窗當 baseline
        else:
            baseline_mask = pd.Series(False, index=sub.index)
            baseline_mask.iloc[: max(1, baseline_n_windows)] = True

        # 對每個特徵欄位建立 rel_* 和 z_*
        for col in valid_cols:
            x = pd.to_numeric(sub[col], errors="coerce")

            # 有現場場域 site_baseline.json 就先用它
            if col in feature_stats:
                mu = float(feature_stats[col].get("mean", np.nan))
                sigma = float(feature_stats[col].get("std", np.nan))
                
            else:
                baseline_values = x[baseline_mask]     # 把剛剛選中的 baseline 那幾窗取出來
                # 算 baseline 的：mu：平均值 , sigma：標準差
                mu = baseline_values.mean()
                sigma = baseline_values.std(ddof=1)

            # 如果 baseline 平均值算不出來，例如全是 NaN，那就退回整段訊號的平均值
            if pd.isna(mu):
                mu = x.mean()

            # 如果 baseline 標準差算不出來，或太小，那就退回整段訊號的標準差。
            if pd.isna(sigma) or sigma < eps:
                sigma = x.std(ddof=1)
            
            # 如果連整段標準差都不可靠，那最後硬設成 1.0。這樣至少後面算 z-score 不會炸掉。
            if pd.isna(sigma) or sigma < eps:
                sigma = 1.0

            # 真正建立新特徵
            out.loc[idx, f"rel_{col}"] = x - mu       # 建立 rel_* 特徵:每一窗的值，相對於 baseline 平均偏了多少
            out.loc[idx, f"z_{col}"] = (x - mu) / (sigma + eps)     # 建立 z_* 特徵：每一窗相對於 baseline 偏了幾個標準差

    return out


# ============================================================
# 正式推論真正的特徵工程入口
# ============================================================
def build_runtime_feature_table(
    *,
    file_specs,                   # 新進 TDMS 清單
    runtime_cfg,                  # 剛剛 load_runtime_configs() 整理好的設定
    cache_dir,                    # 快取資料夾
    split_name="runtime_infer",   # 這批資料的識別名字
    n_jobs=2,                     # 平行處理數量
):
    """
    正式 runtime 特徵工程入口
    """
    # 先抽基礎 band 特徵:對新進 TDMS 切窗, 按 band_config 算基礎 band 特徵
    df = extract_base_feature_table(
        file_specs=file_specs,
        cache_dir=cache_dir,
        split_name=split_name,
        band_config=runtime_cfg["band_config"],
        n_jobs=n_jobs,
        window_sec=runtime_cfg["window_sec"],
        hop_sec=runtime_cfg["hop_sec"],
        fs=runtime_cfg["fs"],
        nperseg=runtime_cfg["nperseg"],
        max_points=runtime_cfg["max_points"],
    )
    
    return build_runtime_features_from_base_table(df, runtime_cfg)


#                                               ********************  runtime pipeline  ********************
#                                               ************************************************************
# ============================================================
# 模型先對每個 window 做預測
# ============================================================
def run_model_inference(
    df_features,               # 前面特徵工程做好的特徵表
    model_bundle,              # 訓練好的模型檔位置
    score_col="leak_score",    # 模型分數欄位名稱，預設叫 leak_score
    pred_col="pred_label",     # 模型預測標籤欄位名稱，預設叫 pred_label
    threshold_override=None,   
):
    # 載入模型:把你前面訓練好的模型讀進來
    model = model_bundle["model"]
    # 如果呼叫端有明確傳 threshold_override，就優先用它；沒有的話才退回用模型自己的 window_threshold 
    selected_feature_names = model_bundle["feature_names"]
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(model_bundle["window_threshold"])
    )
    
    # 檢查 runtime 特徵表本身是否有重複欄名
    duplicate_cols = (df_features.columns[df_features.columns.duplicated()].unique().tolist())
    
    if duplicate_cols:
        raise ValueError(f"runtime 特徵表出現重複欄名: {duplicate_cols}")
    
    # 檢查模型需要的特徵是否完整
    missing_features = [
        c for c in selected_feature_names
        if c not in df_features.columns
    ]
    
    if missing_features:
        raise ValueError(f"缺少模型需要的特徵欄位: {missing_features[:10]}")
    
    # 嚴格按照模型訓練時的欄位名稱及順序建立 X
    X = df_features[selected_feature_names].copy()
    
    # 再確認實際輸入模型的欄位數量
    if X.shape[1] != len(selected_feature_names):
        raise ValueError(
            "模型輸入欄位數量不一致："
            f"預期 {len(selected_feature_names)}，"
            f"實際 {X.shape[1]}"
        )

    # 複製一份原表
    df_out = df_features.copy()

    # ---  三層 fallback  ---
    # 如果模型支援 predict_proba:優先用 predict_proba 取「洩漏類別（class 1）」的機率
    if hasattr(model, "predict_proba"):
        # 如果這個模型可以輸出機率，就直接取「第 1 類」的機率作為分數, 第 1 類就是你定義的 leak 類
        y_score = model.predict_proba(X)[:, 1]
    
    # 如果模型沒有 predict_proba，但有 decision_function(有些模型不直接給機率，只給 decision score。例如某些 SVM 類模型)
    elif hasattr(model, "decision_function"):
        # 先拿到原始 decision score, 再用 sigmoid 轉成 0 到 1 之間的值
        raw_score = model.decision_function(X)
        y_score = 1.0 / (1.0 + np.exp(-raw_score))
    
    # 如果兩個都沒有，就退回用 predict, 就直接用預測類別當分數
    else:
        y_score = model.predict(X).astype(float)

    # 把分數寫回表格, 把模型分數存進你指定的欄位名。預設就是：leak_score
    df_out[score_col] = y_score
    # 根據 threshold 轉成 0/1 預測
    df_out[pred_col] = (df_out[score_col] >= threshold).astype(int)

    return df_out


# ============================================================
# 把很多 window 的結果整理成比較合理的事件判斷
# ============================================================
def apply_postprocessing(
    *,
    df_pred,                  # 已經跑完模型推論的表
    score_col="leak_score",
    pred_col="pred_label",    # 窗級 0/1 預測欄位
    group_col="record_id",    # 每筆訊號的分組欄位
    time_col="t_ref",         # 時間欄位
    window_sec=0.2,           # 窗長
    min_consecutive_windows=3,# 至少要連續幾窗才像真正 leak
    min_leak_duration_sec=0.4,# 至少要持續多久才像真正 leak
    disturbance_ratio_col="offband_to_leak_focus_E_ratio",    # 要拿哪個 ratio 幫忙判斷干擾
    disturbance_ratio_threshold=None,   # ratio 超過多少時偏向干擾
):
    '''
    這段是整個流程的關鍵，因為模型窗級預測通常很碎，不能直接拿來當最後結論。
    它要做的是：
        - 把很多個 pred_label=1 的窗串起來看
        - 判斷它是穩定 leak
        - 還是只是短暫干擾尖峰
    '''
    # 先複製並排序(按照 record_id 和時間排序), 先把每一窗預設成 "normal"
    df = df_pred.copy()
    df = df.sort_values([group_col, time_col]).reset_index(drop=True)
    df["event_state"] = "normal"

    # 逐筆訊號分開處理(依 record_id 分組獨立處理)
    for _, sub in df.groupby(group_col, sort=False):
        # 抓出這筆訊號的索引與預測結果
        idx = sub.index.tolist()    # 這筆訊號在整張表中的列位置
        preds = sub[pred_col].fillna(0).astype(int).tolist()    # 這筆訊號每個窗的預測結果，轉成整數清單

        # 找出連續的 1 區段(哪些窗是連續一整段都被模型判成 1 的), 這些段落就是後面要檢查的候選事件
        start = None
        runs = []
        for i, val in enumerate(preds):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(preds) - 1))

        # 逐段處理每個候選事件
        for s, e in runs:
            # 找出這段對應的列索引與長度
            run_idx = idx[s:e + 1]    # 這段連續正例在整張表中的列位置
            run_len = e - s + 1       # 這段有幾個窗

            # 算這段持續多久
            t0 = float(sub.iloc[s][time_col])
            t1 = float(sub.iloc[e][time_col])
            duration_sec = (t1 - t0) + window_sec    # 因為 t_ref 是窗的中心時間（不是起點），所以區段真正涵蓋的物理時間長度，應該是「第一個窗的起點」到「最後一個窗的終點」，也就是 (t1 + window_sec/2) - (t0 - window_sec/2) = (t1 - t0) + window_sec

            # 先預設這段是 leak:既然模型已經把這段連續窗都判成 1，那先暫時假設它是 leak，但後面還要再過濾。
            state = "leak"
            
            # 第一層過濾:如果這段連續窗數不夠,持續時間太短, 那就不要直接叫它 leak, 而是判成：disturbance_suspect
            # 同時滿足「窗數夠多」且「持續夠久」才能保住 "leak" 這個判定
            if run_len < min_consecutive_windows or duration_sec < min_leak_duration_sec:
                state = "disturbance_suspect"   
            
            # 第二層過濾:再看 ratio 規則:如果你有設定干擾 ratio 門檻，而且這個 ratio 欄位真的存在，就進一步做第二層檢查
            if (
                disturbance_ratio_threshold is not None
                and disturbance_ratio_col in sub.columns
            ):
                # 算這段 run 的 ratio 中位數
                run_ratio = pd.to_numeric(
                    sub.iloc[s:e + 1][disturbance_ratio_col],
                    errors="coerce",
                ).median()
                # 如果 ratio 太高，也改成 disturbance_suspect
                # 如果這段窗雖然模型判得像 leak，但它的 offband_to_leak_focus_E_ratio 太高，代表 offband 能量相對 leak 主頻帶太強，比較像寬頻干擾而不是真正穩定 leak。
                if pd.notna(run_ratio) and run_ratio >= disturbance_ratio_threshold:
                    state = "disturbance_suspect"
            # 把這段結果寫回去, 所以最後每個窗不只是有：pred_label, 還會有：event_state("normal","leak","disturbance_suspect")
            df.loc[run_idx, "event_state"] = state

    return df


# ============================================================
# 把很多個 window 的結果濃縮成「這整筆訊號最後是什麼狀態」
# ============================================================
def summarize_event_state(
    *,
    df_post,                   # 已經做完後處理的結果表
    group_col="record_id",
    state_col="event_state",
    score_col="leak_score",
):
    # 先建立空清單, 把每一筆訊號的摘要結果一列一列存起來, 最後會變成一張 summary 表
    rows = []

    # 逐筆訊號分組(依照 record_id 分組)
    for record_id, sub in df_post.groupby(group_col, sort=False):
        # 把這筆訊號所有窗的事件狀態抓出來
        states = sub[state_col].astype(str).tolist()

        # 決定這整筆訊號的最終狀態
        # 規則 1:只要這筆訊號裡有任何窗被判成 "leak", 那整筆訊號最後就叫 leak，本身已經是通過 apply_postprocessing 那一關「連續窗數/持續時間/offband ratio」層層過濾之後才留下來的
        if "leak" in states:
            final_state = "leak"
        # 規則 2:如果沒有 "leak"，但有 "disturbance_suspect"，那整筆訊號最後就叫 disturbance_suspect
        elif "disturbance_suspect" in states:
            final_state = "disturbance_suspect"
        # 規則 3:如果兩者都沒有，那就代表整段都很正常，所以是 normal
        else:
            final_state = "normal"

        # 把這筆訊號的摘要資訊存成一列
        rows.append({
            "record_id": record_id,      # 這筆 summary 是哪一筆訊號
            "final_state": final_state,  # 這整筆訊號的最後結論(可能是：normal , leak , disturbance_suspect)
            "max_leak_score": pd.to_numeric(sub[score_col], errors="coerce").max(),    # 這筆訊號所有窗裡，最高的 leak score 是多少
            "n_total_windows": len(sub),         # 這筆訊號總共有幾個 window
            "n_leak_windows": int((sub[state_col] == "leak").sum()),     # 這筆訊號裡，有多少窗最後被判成 leak
            "n_disturbance_windows": int((sub[state_col] == "disturbance_suspect").sum()),    # 有多少窗被判成 disturbance_suspect
            "n_normal_windows": int((sub[state_col] == "normal").sum()),    # 有多少窗維持 normal
        })

    return pd.DataFrame(rows)


# ============================================================
# 把結果輸出成檔案
# ============================================================
def export_runtime_result(
    *,
    df_windows,   # window-level 詳細結果
    df_summary,   # event / record-level 摘要結果
    output_dir,   # 輸出資料夾
    window_csv_name="runtime_window_predictions.csv",    # window 結果檔名
    summary_csv_name="runtime_event_summary.csv",        # summary 結果檔名
): 
    # 把輸出資料夾轉成 Path
    output_dir = Path(output_dir)
    # 如果資料夾不存在就建立
    output_dir.mkdir(parents=True, exist_ok=True)

    # 組出兩個輸出檔路徑
    window_csv = output_dir / window_csv_name     # window-level 輸出檔
    summary_csv = output_dir / summary_csv_name   # summary-level 輸出檔

    # 存檔
    df_windows.to_csv(window_csv, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"✅ window-level 結果已輸出: {window_csv}")
    print(f"✅ event summary 已輸出: {summary_csv}")

    # 回傳輸出資訊
    return {
        "window_csv": window_csv,
        "summary_csv": summary_csv,
    }


# ============================================================
# 把結果輸出成檔案
# ============================================================
def load_model_bundle(metadata_json_path):
    metadata_json_path = Path(metadata_json_path)
    model_dir = metadata_json_path.parent     # 從 metadata.json 的路徑反推出模型資料夾 model_dir

    # 讀出整份 metadata 內容
    metadata = json.loads(metadata_json_path.read_text(encoding="utf-8"))
    model = joblib.load(model_dir / "model.pkl")

    # thresholds.json 是選擇性存在的檔案——回頭看 ModelManager.save_model，它只有在 threshold_payload 裡至少有一個值不是 None 時才會真的寫出這個檔案
    thresholds_path = model_dir / "thresholds.json"
    thresholds = {}
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

    # 讀書特徵清單
    feature_names = metadata["features"]["feature_names"]
    # 優先看 thresholds.json 裡有沒有 window_threshold；沒有的話，退回去看 metadata.json 裡 performance.window_level.threshold；兩個都沒有，最後保底用 0.5
    window_threshold = thresholds.get(
        "window_threshold",
        metadata.get("performance", {}).get("window_level", {}).get("threshold", 0.5),
    )

    return {
        "model": model,
        "metadata": metadata,
        "feature_names": feature_names,
        "window_threshold": float(window_threshold),
        "model_dir": model_dir,
    }


#                                                  ********************  main entry  ********************
#                                                  ******************************************************
# ============================================================
# 正式推論流程入口
# ============================================================
def run_runtime_pipeline(
    *,
    tdms_paths,                    # 要推論的一筆或多筆 TDMS 路徑
    output_dir,                    # 結果要輸出到哪裡
    model_metadata_json,           # 訓練好的模型檔
    band_config_path,              # 正式使用的 band 設定檔
    feature_extraction_config_path,
    feature_config_path,           # 正式使用的特徵與參數設定檔
    postprocess_config_path,       # 正式 offband 設定檔
    runtime_feature_schema_path,
    site_baseline_path=None,       # 場域正常 baseline 設定檔，可選；若提供則優先使用
    split_name="runtime_infer",    # 這批推論資料的名字
    n_jobs=2,                      # 平行處理數量
):
    """
    正式離線推論的完整端到端流程：讀五份正式設定 → 整理新資料 → 跑封裝好的特徵工程 → 雙重欄位一致性檢查 → 模型推論（用規則端 threshold 覆寫） → 逐 record 套告警規則（含 veto/two-stage） → 額外的事件層級後處理（連續/持續時間關卡刻意調鬆,因為已在告警規則階段處理過） → 壓成摘要表並輸出。
    
    正式即時 / 離線推論入口。
    輸入是一筆或多筆新進 TDMS，
    直接讀 band_config、feature_config、model，
    並優先使用 site_baseline 做 baseline-relative 特徵建立。
    若未提供 site baseline，才 fallback 到新資料前幾窗估 baseline。
    
    新的 TDMS
       ↓
    整理 TDMS 路徑
       ↓
    讀取「以前已經決定好的設定」
       ├── band_config
       ├── feature_config
       ├── postprocess_config
       ├── runtime_feature_schema
       ├── site_baseline
       └── model
       ↓
    TDMS 切成很多 window
       ↓
    計算基本特徵
       ↓
    建立 baseline-relative 特徵
       ├── rel_*
       └── z_*
       ↓
    CUSUM 特徵
       ↓
    Temporal 特徵
       ├── delta
       ├── local_rel
       ├── local_z
       ├── rolling_std
       └── rolling_slope
       ↓
    offband / leak / resonance 摘要
       ↓
    檢查「模型需要的欄位」有沒有全部準備好
       ↓
    XGBoost 模型
       ↓
    每個 window 得到
       ├── leak_score
       └── pred_label
       ↓
    Alarm rule
       ↓
    連續 window / 持續時間 / 干擾 ratio
       ↓
    每個 window 最終：
       ├── normal
       ├── leak
       └── disturbance_suspect
       ↓
    整筆 TDMS 最終：
       ├── normal
       ├── leak
       └── disturbance_suspect
       ↓
    輸出 CSV
        
    """
    # ------------ 讀設定 ---------------
    # 輸出路徑與 cache 路徑
    output_dir = Path(output_dir)
    cache_dir = output_dir / "feature_cache"     # 在輸出資料夾底下，定義一個：feature_cache，這是給特徵工程快取用的，意思是如果某些特徵已經算過，可以減少重複計算。

    # 讀正式 runtime 設定,得到一包 runtime_cfg，裡面已經整理好：band config、feature config、postprocess config、runtime feature schema、site baseline
    runtime_cfg = load_runtime_configs(
        band_config_path=band_config_path,
        feature_extraction_config_path=(feature_extraction_config_path),
        feature_config_path=feature_config_path,
        postprocess_config_path=postprocess_config_path,
        runtime_feature_schema_path=runtime_feature_schema_path,
        site_baseline_path=site_baseline_path,
    )

    # ------------ 整理新進 TDMS ---------------
    # 整理新進 TDMS 輸入(通常是加上 record_id、感測器對應等 metadata 的結構, 新進來的推論資料,不是訓練用的 dev/holdout)
    file_specs = build_runtime_file_specs(tdms_paths)
    # 印出一些基本資訊
    print("正式 runtime TDMS 數量 =", len(file_specs))
    print("selected features 數量 =", len(runtime_cfg["selected_feature_names"]))
    print("raw base features 數量 =", len(runtime_cfg["raw_base_features"]))

    # ------------ 特徵工程 ---------------
    # 讀 TDMS -> 切窗 -> 算 band base features -> 建 baseline relative 特徵 -> 算 CUSUM -> 算 temporal features -> 算 offband / leak / resonance 摘要特徵
    df_features = build_runtime_feature_table(
        file_specs=file_specs,
        runtime_cfg=runtime_cfg,
        cache_dir=cache_dir,
        split_name=split_name,
        n_jobs=n_jobs,
    )
    
    # ------------ 特徵一致性檢查 ---------------
    # 檢查 Step3 / Step4 / runtime 輸出的表，有沒有漏掉 Step2 最終選定特徵？
    audit_selected_feature_alignment(df_features,{"selected_feature_names": runtime_cfg["model_feature_names"]})
    audit_rule_feature_alignment(df_features,runtime_cfg["rule_feature_names"])
    
    # 分別算出「模型要的欄位」跟「規則要的欄位」裡，有哪些在 df_features.columns 裡找不到
    required_model_cols = runtime_cfg["model_feature_names"]
    required_rule_cols = runtime_cfg["rule_feature_names"]
    missing_model_cols = [c for c in required_model_cols if c not in df_features.columns]
    missing_rule_cols = [c for c in required_rule_cols if c not in df_features.columns]
    
    print("\n" + "=" * 80)
    print("正式 runtime 特徵欄位檢查")
    print("=" * 80)
    print("模型欄位數量:", len(required_model_cols))
    print("規則欄位數量:", len(required_rule_cols))
    print("缺少模型欄位:", missing_model_cols)
    print("缺少規則欄位:", missing_rule_cols)
    
    # 任一邊有缺,就報錯終止,不讓不完整的特徵表繼續往下跑進模型推論
    if missing_model_cols:
        raise ValueError(f"正式 runtime 缺少模型欄位: {missing_model_cols}")
    
    if missing_rule_cols:
        raise ValueError(f"正式 runtime 缺少規則欄位: {missing_rule_cols}")

    # ------------ 跑模型推論 ---------------
    # 載入模型：把剛剛的特徵表丟進模型, 會產生：leak_score, pred_label, 就會有每個 window 的模型預測結果了
    model_bundle = load_model_bundle(model_metadata_json)
    
    # 從 model_bundle 取出模型自己記錄的 window_threshold（訓練時決定的最佳門檻），跟 runtime_cfg 取出的 window_threshold
    model_threshold = float(model_bundle["window_threshold"])
    rule_threshold = float(runtime_cfg["window_threshold"])
    
    # 一致性檢查：確認 threshold 是否一致(訓練時決定的最佳閾值 vs 規則搜尋時用的閾值)
    if abs(model_threshold - rule_threshold) > 1e-12:
        print("⚠️ 模型 threshold 與 postprocess rule 的 window_threshold 不一致")
        print("model threshold =", model_threshold)
        print("rule threshold  =", rule_threshold)

    # 1. 模型只負責產生每個 window 的機率：模型推論，對每個時間窗算出機率分數（score_col）跟二元判斷（pred_col）
    df_pred = run_model_inference(
        df_features=df_features,
        model_bundle=model_bundle,
        score_col=runtime_cfg["score_col"],
        pred_col=runtime_cfg["pred_col"],
        threshold_override=runtime_cfg["window_threshold"],    # 用的是 runtime_cfg["window_threshold"]（也就是 rule 那邊的閾值）當作 threshold_override，而不是 model_bundle["window_threshold"]——這跟前面那個一致性檢查的用意是呼應的：即使兩者理論上該相等
    )
    
    # ------------ 正式規則 alarm stage ---------------
    df_rule_input = df_pred.copy()

    # 把 runtime_cfg["score_col"] 指到的實際欄位內容,複製一份、轉數值、把轉不成數字的值用 0.0 補上，存進固定名稱 pred_proba
    df_rule_input["pred_proba"] = pd.to_numeric(
        df_rule_input[runtime_cfg["score_col"]],
        errors="coerce",).fillna(0.0)
    
    # 從 runtime_cfg 裡挑出你剛剛在 build_runtime_postprocess_settings 看過的那份回傳字典
    formal_rule_cfg = {
        "method": runtime_cfg["alarm_method"],
        "alarm_threshold": runtime_cfg["alarm_threshold"],
        "smooth_window": runtime_cfg["smooth_window"],
        "consecutive": runtime_cfg["consecutive"],
        "n_hits": runtime_cfg["n_hits"],
        "m_window": runtime_cfg["m_window"],
        "high_threshold": runtime_cfg["high_threshold"],
        "low_threshold": runtime_cfg["low_threshold"],
        "min_leak_duration_sec": runtime_cfg["min_leak_duration_sec"],
    
        "use_veto": runtime_cfg["use_veto"],
        "veto_column": runtime_cfg["veto_column"],
        "veto_threshold": runtime_cfg["veto_threshold"],
        "veto_rules": runtime_cfg["veto_rules"],
    
        "stage1": runtime_cfg["stage1"],
        "stage2": runtime_cfg["stage2"],
    }
    
    # 用整理好的 formal_rule_cfg 對每一個 record_id 分別跑一次告警規則（consecutive/n_of_m/hysteresis/two_stage 其中一種，取決於 formal_rule_cfg["method"]）
    df_alarm = apply_alarm_method_by_record(
        df_pred=df_rule_input,
        cfg=formal_rule_cfg,
        window_threshold=runtime_cfg["window_threshold"],
    )
    
    df_alarm["alarm_detected"] = df_alarm["detected_rule"]                # 套完 veto 的正式結果
    df_alarm["alarm_detected_raw"] = df_alarm["detected_rule_raw"]        # 未套 veto
    df_alarm["alarm_score_smooth"] = df_alarm["smoothed_proba_rule"]      # 平滑後機率
         
    # ------------ 事件層級後處理 ---------------
    # 2. 正式告警規則讀取 leak_score
    # 把 window-level 結果再整理成比較合理的事件判斷。會根據：連續窗數夠不夠, 持續時間夠不夠, offband ratio 高不高, 把某些短暫尖峰改判成：disturbance_suspect
    df_post = apply_postprocessing(
        df_pred=df_alarm,
        score_col=runtime_cfg["score_col"],
        pred_col="alarm_detected",     # 根據套用告警規則之後的判斷結果做事件層級整理
        group_col="record_id",
        time_col="t_ref",
        window_sec=runtime_cfg["window_sec"],
    
        # 前面共用規則已經完成連續窗與最短持續時間判斷
        min_consecutive_windows=1,
        min_leak_duration_sec=0.0,
    
        # veto 已在共用 alarm rule 逐 window 套用, 「會根據：連續窗數夠不夠, 持續時間夠不夠, offband ratio 高不高, 把某些短暫尖峰改判成：disturbance_suspect」
        disturbance_ratio_col=runtime_cfg.get(
            "disturbance_ratio_col",
            "offband_to_leak_focus_E_ratio",),
        disturbance_ratio_threshold=None,
    )

    # ------------ 輸出結果 ---------------
    # 把每個 window 的結果濃縮成每筆訊號的總結
    df_summary = summarize_event_state(
        df_post=df_post,
        group_col="record_id",
        state_col="event_state",
        score_col=runtime_cfg["score_col"],
    )

    # 把結果輸出成檔案, 存到 output_dir
    export_info = export_runtime_result(
        df_windows=df_post, df_summary=df_summary, output_dir=output_dir,)
    
    # 印出結果
    print("\n" + "=" * 80)
    print("正式離線推論 summary")
    print("=" * 80)
    print(df_summary)
    
    print("\nwindow-level 最後 20 筆")
    print(
        df_post[
            ["record_id", "t_ref", runtime_cfg["score_col"], runtime_cfg["pred_col"], "event_state"]
        ].tail(20)
    )

    return {
        "runtime_cfg": runtime_cfg,
        "df_features": df_features,
        "df_pred": df_pred,
        "df_post": df_post,
        "df_summary": df_summary,
        "export_info": export_info,
    }




if __name__ == "__main__":
    # 資料路徑
    PROJECT_ROOT = Path("/Users/paul/Desktop/Final")

    # 設定模型
    MODEL_DIR = (
        PROJECT_ROOT
        / "test_result"
        / "saved_models_transition_disturbance_final"
        / "xgb"
        / "xgb_20260922_102829"
    )

    runtime_inputs = [
        PROJECT_ROOT / "transition_test",      # 轉換資料
        PROJECT_ROOT / "disturbance_test",     # 干擾資料
    ]
    
    # 目前只是整合測試，所以各取前 5 支
    transition_paths = collect_runtime_tdms_paths([runtime_inputs[0]])[:5]
    disturbance_paths = collect_runtime_tdms_paths([runtime_inputs[1]])[:5]

    tdms_paths = transition_paths + disturbance_paths
    
    result = run_runtime_pipeline(
        # 跑正式 runtime 推論的 TDMS
        tdms_paths=tdms_paths,
        # 輸出目錄
        output_dir="/Users/paul/Desktop/runtime_outputs/formal_runtime_case",
        # 模型本身+threshold
        model_metadata_json=(MODEL_DIR / "metadata.json"),
        # 頻帶
        band_config_path=(PROJECT_ROOT / "test_result" / "band_config_step2.json"),
        # 上游特徵
        feature_extraction_config_path=(PROJECT_ROOT / "test_result" / "feature_extraction_config.json"),
        # 決定窗長、hop、要抽哪些特徵
        feature_config_path=(PROJECT_ROOT / "test_result" / "feature_config_micro.json"),
        # 模型訓練當下配套選出的告警規則
        postprocess_config_path=(MODEL_DIR / "postprocess_config.json"),
        # 模型訓練時實際用的欄位清單
        runtime_feature_schema_path=(MODEL_DIR / "runtime_feature_schema.json"),
        # 正式正常 baseline
        site_baseline_path=(PROJECT_ROOT / "test_result" / "site_baseline.json"),
        # 標記快取/輸出檔名的字串
        split_name="formal_runtime_case",
        n_jobs=2,
    )
    
    df_summary = result["df_summary"]
    df_post = result["df_post"]
    

