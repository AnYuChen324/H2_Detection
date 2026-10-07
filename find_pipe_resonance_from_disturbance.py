#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出管子自身固定的共振候選頻帶。
1. 先用「敲擊管子」資料，找出哪些頻率在扣除寬頻趨勢後仍有局部凸起
2. 用 0713 做跨天驗證，並分位置找峰、統計每個 10 kHz 格子在幾個位置都通過
3. 自動產生提案 proposed_resonance_bands.json，人工確認後用 --approve-proposal 核准，
   寫入正式版 approved_resonance_bands.json（每次核准都會留下紀錄）

原始 TDMS -> 全頻段 PSD -> 扣除寬頻趨勢找局部峰 -> 跨天驗證 -> 分位置投票 -> 提案 -> 人工核准 -> 回灌特徵工程

分析邏輯：
1. 敲擊管子（tap_pipe / tab_pipe / tappipe）可視為一種寬頻激發。
   敲擊會讓很寬的頻段整體升高，因此不能只看「敲擊比背景高多少」，
   而是先扣除頻譜上的寬頻趨勢，只保留比局部趨勢更凸出的頻率，作為共振候選。

2. 本資料集的 disturbance 資料不是單一長錄音檔，而是一段段 TDMS 小片段，
   並且每個 session 都已經有標註檔可用，例如：
   - dual_session_manifest_*.csv
   - disturbance_events_*.csv
   - transition_dual_manifest_*.csv
   - transition_dual_disturbance_events_*.csv

3. 因此這支程式不需要手動列出 TAP_TDMS_FILES 與 BACKGROUND_TDMS_FILES，
   而是直接：
   - 從資料夾中自動找到敲擊類型的 session
   - 用 manifest 找到每個 TDMS 小片段對應的時間範圍與檔案路徑
   - 用 disturbance events 判斷哪些片段屬於敲擊期間
   - 用同一批 session 中、離敲擊事件夠遠的片段當背景噪音
   - 敲擊與背景都只使用 leak0（無洩漏）片段，避免混入洩漏訊號

4. 最後分別計算：
   - 敲擊片段的平均 PSD
   - 背景片段的平均 PSD

   再用兩者的 dB 差異，扣除寬頻趨勢後找出局部凸起的峰值，並用 0713 資料做跨天驗證。
   順序是「先找峰、後套網格」：
    先在整條頻譜上（使用預設 fs=1 MHz、nperseg=8192 時約為 122 Hz）找出局部凸起並做跨天驗證，
    再看每個驗證通過的峰落在哪一個 10 kHz 格子裡（與特徵工程的搜尋網格一致），
    最後依「在幾個位置、幾個聲道都通過」決定提案中的 core / secondary。
    這和洩漏頻帶的做法相反（洩漏頻帶是先切好 10 kHz 格子，再評估每一格）：
    共振峰可能只有幾百 Hz 寬，若先切格子再比較，窄峰會被整格平均掉。

5. 這支程式找到的是「共振候選頻帶」，不是直接等同於「洩漏判別頻帶」；
   後續仍需搭配正常/洩漏資料，評估共振頻帶特徵對洩漏判斷是否有幫助。
   
其他：
    - 與 offband 重疊的格子（FIXED_EXCLUDED_RANGES）一律預設 include = false，並標示 fixed_excluded
    - 上次核准時排除的格子（EXCLUDED_RANGES）會預設 include = false，並標示 previously_excluded
    - diff_vs_approved 以「範圍」比對正式版，分成新增 / 移除 / 類型改變；預設不納入的格子不計入

說明：
本程式找出的不是嚴格物理定義下的固有模態頻率，而是根據實際敲擊量測資料，在扣除寬頻趨勢後仍局部凸起、且可跨天重現的頻帶，因此稱為「共振候選頻帶」。

敲擊造成的寬頻整體升高（可能包含感測器本身的頻率響應），不視為共振。
共振候選也會受到感測器位置、敲擊位置、阻尼、邊界條件與量測噪音影響，所以不一定與理論上的單一固有頻率完全相同。

名詞區分：
- Welch PSD 的 detrend='constant'（scipy 預設）：在每一小段時域訊號內去除直流成分
- 本程式的「扣除寬頻趨勢」（detrend_db）：在頻譜上用中位數濾波扣掉整體高原
兩者是不同的處理。


使用限制：
- TDMS 片段假設：
  本程式會讀取每個 TDMS 檔的完整聲道資料。
  manifest 的 t_start / t_end 只用於事件分類與追查，不會拿來裁切波形。
  因此每個 manifest 片段必須對應獨立的 TDMS 檔。

- 取樣率假設：
  頻率軸使用 --fs 指定的取樣率計算，目前不會自動從 TDMS metadata
  驗證真實取樣率。--fs 設定錯誤會使所有峰值頻率一起偏移。

- 平均方式：
  搜尋資料與驗證資料各自只保留 Tap、Background 都有的位置。
  程式先在每個位置內平均 PSD，再將不同位置等權重平均，
  避免片段數較多的位置主導結果。

- 跨天驗證：
  對每個搜尋候選峰，在驗證資料的 ±validation_tol_hz 範圍內，
  選擇局部凸起最大的頻率點，並在同一頻率點檢查 local dB
  與 Tap/Background dB 差。

- 輸出注意：
  重複使用相同的 --output-csv 名稱時會寫入相同輸出資料夾；
  執行前應確認其中沒有被誤認為本次結果的舊檔案。

本程式的目的，是從真實量測資料中找出穩定且可重現的共振候選頻帶，作為後續特徵工程、洩漏分析與模型驗證的候選依據。
"""

from __future__ import annotations
from pathlib import Path, PureWindowsPath
import argparse
from dataclasses import dataclass
from typing import Iterable, Literal
import os
import tempfile
import json
from scipy.ndimage import median_filter
from datetime import datetime
from functools import lru_cache

# 避免 matplotlib 字型快取寫到無權限的位置
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks
from nptdms import TdmsFile
from collections import Counter


# ============================================================
# 參數預設
# ============================================================
DEFAULT_ROOT = Path("/Users/paul/Desktop/AE Data/disturbance")                           # 干擾資料根目錄
DEFAULT_OUTPUT = Path("/Users/paul/Desktop/Final/resonance/pipe_resonance_bands.csv")    # 輸出檔的預設位置

# 相容資料夾中曾使用過的不同敲擊名稱拼法
TAP_FOLDER_KEYWORDS = ("tap_pipe", "tab_pipe", "tappipe")


# ============================================================
# 預設批次設定:前三天用來找共振候選峰，0713 保留做驗證
# ============================================================
DEFAULT_SEARCH_BATCHES = {
    "disturbance_ann_0624",
    "disturbance_ann_0702",
    "disturbance_ann_0707",
}

DEFAULT_VALIDATION_BATCHES = {
    "disturbance_ann_0713",
}


# ============================================================
# 共振頻帶網格常數（與 build_site_baseline_v2.py 的 DISCOVERY_* 一致）
# ============================================================
GRID_F_MIN = 5_000         # 涵蓋的頻率範圍，和 discovery 一樣是 5～500 kHz。
GRID_F_MAX = 500_000       # 涵蓋的頻率範圍，和 discovery 一樣是 5～500 kHz。
GRID_WIDTH = 10_000        # 每一格寬 10 kHz，和洩漏頻帶的寬度一致。
RES_GRID_STRIDE = 10_000   # 共振頻帶用「不重疊」的格子，避免特徵數量暴增
MIN_OVERLAP_HZ = 5_000     # 格子和共振頻帶重疊至少一半才納入


# ============================================================
# 共振頻帶設定（半自動）
# - 分析時自動產生提案：<輸出>/configs/proposed_resonance_bands.json
# - 人工確認後用 --approve-proposal 核准，寫入正式版 approved_resonance_bands.json
# - 每次核准都會在正式版旁邊的 resonance_bands_approval_log.jsonl 新增一筆紀錄
# 程式碼裡不再寫死頻帶，下面四個變數會在 main() 開頭從正式版載入
# ============================================================
DEFAULT_APPROVED_JSON = Path("/Users/paul/Desktop/Final/resonance/approved_resonance_bands.json")

CORE_BANDS: list[tuple] = []
SECONDARY_BANDS: list[tuple] = []
ALL_BANDS: list[tuple] = []                # 目前正式版的 ALL_BANDS
EXCLUDED_RANGES: set[tuple] = set()
# 固定排除：已定為 offband 的頻段（與 band_config_step2.json 的 offband_bands 一致）
# 和這些範圍有重疊的共振格子，提案時一律預設不納入
FIXED_EXCLUDED_RANGES = [(10_000, 20_000), (50_000, 60_000), (210_000, 220_000), (315_000, 325_000)]


# ============================================================
# 從正式版載入共振頻帶到 CORE_BANDS / SECONDARY_BANDS / ALL_BANDS / EXCLUDED_RANGES
# ============================================================
def load_approved_bands(path: Path) -> None:
    """從正式版載入共振頻帶到 CORE_BANDS / SECONDARY_BANDS / ALL_BANDS，以及上次核准時排除的頻帶範圍。"""
    # 函式裡對變數賦值時，預設會建立一個只在函式內有效的新變數。
    # 加上 global，代表這裡要改的是檔案最上面那四個變數，其他函式才看得到載入後的內容。
    global CORE_BANDS, SECONDARY_BANDS, ALL_BANDS, EXCLUDED_RANGES
    
    # 第一次執行時還沒有正式版，就當作「沒有既定頻帶」繼續執行，不會中斷。這時落差稽核會把所有驗證通過的峰都列為「落在頻帶外」，這是正常的。
    if not path.exists():
        print(f"⚠️ 找不到正式版共振頻帶 {path}，本次視為「沒有既定頻帶」，跑完後請核准提案")
        CORE_BANDS, SECONDARY_BANDS = [], []
        EXCLUDED_RANGES = set()
        
    else:
        # 讀取 JSON，並檢查 schema_name
        payload = json.loads(path.read_text(encoding="utf-8"))
        
        # 防呆：確認是正式版的格式，避免把提案檔或其他 JSON 傳進來
        if payload.get("schema_name") != "approved_resonance_bands":
            raise ValueError(f"{path} 不是 approved_resonance_bands 格式")
            
        # 把每一筆頻帶轉成 (名稱, 下界, 上界, 類型)；上下界轉成 float，和峰值頻率比較時型別一致
        bands = [(b["name"], float(b["f_low"]), float(b["f_high"]), b["band_type"])
                 for b in payload["resonance_bands"]]
        
        # 防呆：band_type 只能是 core 或 secondary，否則會被默默丟掉
        bad = [b[0] for b in bands if b[3] not in ("core", "secondary")]
        if bad:
            raise ValueError(f"{path} 有 band_type 不是 core/secondary 的頻帶：{bad}")
                
        # 依照欄位（band_type）分成兩組
        CORE_BANDS = [b for b in bands if b[3] == "core"]
        SECONDARY_BANDS = [b for b in bands if b[3] == "secondary"]
        
        # 上次核准時刻意排除的頻帶範圍；產生下一份提案時，這些格子會預設維持排除
        EXCLUDED_RANGES = {(int(b["f_low"]), int(b["f_high"]))
                           for b in payload.get("excluded_bands", [])}
        
        print(f"已載入正式版共振頻帶（核准於 {payload.get('approved_at')}）：{[b[0] for b in bands]}")
        if EXCLUDED_RANGES:
            print(f"  上次核准時排除的範圍：{sorted(EXCLUDED_RANGES)}")
        
    ALL_BANDS = CORE_BANDS + SECONDARY_BANDS


# ============================================================
# 用來存每一個 TDMS 小片段的整理後資訊
# ============================================================
@dataclass(frozen=True)
class SliceRecord:
    # 每一個 TDMS 小片段，不只是原始檔而已，還帶有：
    session_dir: Path           # 這個 session 的資料夾
    tdms_path: Path             # 片段的 TDMS 檔路徑
    session_id: str             # 它來自哪個 session（只用於追查）
    batch_name: str             # 哪一天批次
    disturbance_dir: str        # session 上一層的資料夾名稱（只用於追查）
    disturbance_id: str         # 哪個 disturbance id
    position: str               # 位置，例如 straight20cm（分位置分析會用到）
    pressure_bar: float         # 用於追查及混合壓力警告，目前不會自動篩選
    dual_index: int             # manifest 裡的片段編號(只用於追查)
    t_start: float              # 片段開始時間（只用於追查，不用來裁切訊號）
    t_end: float                # 片段結束時間（只用於追查）
    label: Literal["tap", "background"]                  # "tap" 或 "background"


# ============================================================
# 讀取命令列參數
# ============================================================
def parse_args() -> argparse.Namespace:
    """讀取命令列參數。參數依用途分組，執行 --help 時會分區顯示。"""
    parser = argparse.ArgumentParser(
        description="從 disturbance 的敲擊資料找出管子的共振候選頻帶（半自動：提案 → 人工核准）"
    )

    # ---------- 資料來源 ----------
    data = parser.add_argument_group("資料來源")
    data.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="disturbance 根目錄",
    )
    data.add_argument(
        "--channels", nargs="+", default=["ai0", "ai1"],
        help="要分析的聲道，可指定多個；每個聲道都會輸出各自的峰值表、頻譜與圖",
    )
    data.add_argument(
        "--batch-filter", nargs="*", default=None,
        help="指定搜尋批次（覆蓋預設的 0624/0702/0707），例如 disturbance_ann_0624 disturbance_ann_0702；"
             "不可包含驗證批次 0713",
    )
    data.add_argument(
        "--position-filter", nargs="*", default=None,
        help="只分析特定位置，例如 straight20cm straight50cm；不指定則全部納入",
    )

    # ---------- 片段標記 ----------
    slicing = parser.add_argument_group("片段標記（敲擊 / 背景）")
    slicing.add_argument(
        "--background-guard-sec", type=float, default=0.5,
        help="背景片段要避開敲擊事件前後多少秒，避免沾到敲擊尾波",
    )
    slicing.add_argument(
        "--event-overlap-sec", type=float, default=0.0,
        help="判斷片段是否屬於敲擊事件時，事件邊界額外放寬的秒數（標註切得太緊時可調大）",
    )

    # ---------- 頻譜計算 ----------
    spectrum = parser.add_argument_group("頻譜計算")
    spectrum.add_argument(
        "--fs", type=float, default=1_000_000,
        help="取樣率（Hz）",
    )
    spectrum.add_argument(
        "--nperseg", type=int, default=8192,
        help="Welch PSD 的 nperseg；刻意比特徵工程（2048）大，以細解析度找峰，最後再對齊 10 kHz 格子",
    )

    # ---------- 找峰 ----------
    peak = parser.add_argument_group("找峰（扣除寬頻趨勢後）")
    peak.add_argument(
        "--trend-window-hz", type=float, default=20_000.0,
        help="估計寬頻趨勢用的中位數視窗寬度（Hz）",
    )
    peak.add_argument(
        "--local-peak-db", type=float, default=3.0,
        help="峰要比局部趨勢高出多少 dB 才算候選峰；跨天驗證也用同一門檻",
    )
    peak.add_argument(
        "--peak-prominence", type=float, default=2.0,
        help="find_peaks 的 prominence（在扣除趨勢後的曲線上計算）",
    )
    peak.add_argument(
        "--peak-db-threshold", type=float, default=6.0,
        help="原始 tap/背景 dB 差的最低門檻；找峰和跨天驗證都要求超過此值，確保敲擊確實高於背景",
    )
    peak.add_argument(
        "--min-freq-hz", type=float, default=5_000.0,
        help="低於這個頻率的峰不採用；實際下限取此值與趨勢視窗一半（--trend-window-hz / 2）的較大者",
    )
    peak.add_argument(
        "--peak-min-width-hz", type=float, default=0.0,
        help="峰的最小寬度（Hz，在半個 prominence 高度量測）；0 表示只記錄寬度、不篩選",
    )
    peak.add_argument(
        "--validation-tol-hz", type=float, default=250.0,
        help="跨天驗證時，容許峰在驗證資料中偏移的範圍（Hz）；0 表示只看同一頻率點。"
             "預設 250 Hz（約 ±2 個頻率點），依隨機通過率檢查選定",
    )

    # ---------- 提案規則 ----------
    vote = parser.add_argument_group("提案規則（分位置 / 分聲道投票）")
    vote.add_argument(
        "--min-positions", type=int, default=2,
        help="某聲道在一個 10 kHz 格子裡，至少要幾個位置跨天驗證通過，才算該聲道的強證據",
    )
    vote.add_argument(
        "--min-channels", type=int, default=None,
        help="列為 core 至少需要幾個聲道有強證據；預設為全部聲道",
    )

    # ---------- 頻帶管理 ----------
    bands = parser.add_argument_group("頻帶管理（正式版 / 核准 / 落差稽核）")
    bands.add_argument(
        "--approved-json", type=Path, default=DEFAULT_APPROVED_JSON,
        help="正式版共振頻帶檔",
    )
    bands.add_argument(
        "--approve-proposal", type=Path, default=None,
        help="核准模式：指定要核准的 proposed_resonance_bands.json（只核准，不做分析）",
    )
    bands.add_argument(
        "--approve-note", default="",
        help="核准理由（核准模式必填，會寫進核准紀錄）",
    )
    bands.add_argument(
        "--band-boundary-margin-hz", type=float, default=500.0,
        help="落差稽核用：驗證通過的峰離所在頻帶邊界小於這個值（Hz），就提示邊界可能需要調整",
    )
    
    bands.add_argument(
        "--allow-empty-approval", action="store_true",
        help="允許核准沒有任何頻帶的提案（會清空正式版）",
    )

    # ---------- 輸出 ----------
    output = parser.add_argument_group("輸出")
    output.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT,
        help="輸出的主檔名與路徑；其他輸出會放在同名資料夾下的 tables/spectra/figures/configs",
    )

    return parser.parse_args()


# ============================================================
# 字型設定函式:專門處理畫圖中文字型
# ============================================================
def configure_matplotlib_fonts() -> None:    # -> None 表示這個函式不回傳值
    """
    設定 matplotlib 中文字型，避免標題、座標軸、圖例出現亂碼或方塊字。
    """
    # 建立「希望優先使用」的中文字型候選名單
    candidate_fonts = [
        "Heiti TC",
        "PingFang TC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK TC",
        "Microsoft JhengHei",
    ]
    
    # 目前這台機器 matplotlib 看得到的所有字型名稱
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    # 先建立一個變數 chosen_font，初始值是 None
    chosen_font = None
    # 依序檢查 candidate_fonts 裡的每一個字型名稱
    for font_name in candidate_fonts:
        if font_name in available_fonts:    # 檢查：這個候選字型 font_name，有沒有出現在目前系統可用字型 available_fonts 裡
            chosen_font = font_name         # 如果找到這個字型，就把它記錄下來
            break

    # 如果 chosen_font 不是 None，就代表前面真的找到至少一個可用字型。
    if chosen_font is not None:
        matplotlib.rcParams["font.family"] = chosen_font    # 把 matplotlib 的預設字型改掉
    else:
        print("⚠️ 找不到可用的中文字型，圖上的中文可能顯示成方框。"
              f"候選字型：{candidate_fonts}")

    matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# 小工具：名稱標準化
# ============================================================
# 把名字轉成容易比對的格式
def normalize_name(text: str) -> str:
    return text.lower().replace("-", "_")      # 輸入一段文字 text, 全部轉成小寫, 再把 - 改成 _


# 判斷資料夾是不是敲擊類型，像 tap_pipe、tab_pipe、tappipe
def is_tap_session_dir(path: Path) -> bool:
    """
    判斷某個資料夾是否屬於敲擊類型 session. 只要這個路徑的任何一段名稱看起來像敲擊類型資料夾，就把它當成 tap session。
    """
    # 把路徑 path 拆成一段一段, 對每一段都做 normalize_name()
    lowered_parts = [normalize_name(part) for part in path.parts] 
    return any(any(key in part for key in TAP_FOLDER_KEYWORDS) for part in lowered_parts)


# 真正去掃整個資料夾，把符合條件的 session 找出來
def iter_session_dirs(root: Path, batch_filter: set[str] | None = None) -> Iterable[Path]:     
    # set[str] | None,意思是「一個字串的集合,或者是 None」
    # -> Iterable[Path] 表示這個函式回傳一個「可疊代」的 Path 物件序列
    """
    自動掃描所有含有 manifest 的 session 資料夾，並只保留敲擊類型、且屬於指定批次的 session。
    """
    # 如果外面有傳進 batch_filter，就用外面傳的, 否則就用預設的 DEFAULT_SEARCH_BATCHES
    active_batch_filter = batch_filter or DEFAULT_SEARCH_BATCHES

    # 整個 root 下面遞迴搜尋兩種 manifest, 同時支援舊版與 0713 的 manifest 命名, 先把兩種結果合併, 再 sorted(...) 排序
    manifest_paths = sorted(
        list(root.rglob("dual_session_manifest_*.csv")) +
        list(root.rglob("transition_dual_manifest_*.csv"))
    )

    # 建立一個空集合
    seen_session_dirs = set()
    
    # 開始逐一處理每一個找到的 manifest 檔
    for manifest_path in manifest_paths:
        session_dir = manifest_path.parent    # 這個 manifest 檔所在的資料夾
        
        # 避免同一個 session 被重複加入
        if session_dir in seen_session_dirs:   # 如果這個 session_dir 之前已經處理過，就跳過
            continue
        seen_session_dirs.add(session_dir)     # 把這個資料夾記錄到 seen_session_dirs 裡

        # 只保留 tap / tab / tappipe 類型資料夾
        if not is_tap_session_dir(session_dir):
            continue

        # 只保留指定批次
        batch_name = next(
            (part for part in session_dir.parts if part.startswith("disturbance_ann_")),
            None,
        )
        if batch_name not in active_batch_filter:
            continue
        
        # 找這個 session 的兩個重要標註檔
        manifest_path, events_path = find_session_annotation_files(session_dir)
        
        # 只有當 manifest 和 events 都找得到, 才把這個 session_dir 輸出出去
        if manifest_path is not None and events_path is not None:
            yield session_dir


# 判斷時間區間重疊
def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """判斷兩個時間區間是否重疊；區間採左閉右開 [start, end)。
    瞬間事件（b_start == b_end）只要落在片段時間範圍內，就算重疊。"""
    if b_start == b_end:
        return a_start <= b_start < a_end
    return max(a_start, b_start) < min(a_end, b_end)


# ============================================================
# 扣除寬頻趨勢
# ============================================================
def detrend_db(diff_db: np.ndarray, freqs: np.ndarray, window_hz: float):
    """
    用中位數濾波估計寬頻趨勢，回傳 (趨勢, 扣除趨勢後的局部凸起)。
    敲擊會讓一整片頻率的能量上升；為避免把整片寬頻升高誤判為共振，本方法先估計每個頻率附近的背景趨勢，再找出相對周圍特別突出的局部峰值。
    不是找最高的頻率，而是找比自己附近頻率更突出的頻率。
    
    - freqs 必須是等間隔的頻率軸（Welch 的輸出即是）
    - 視窗會換算成奇數點數，讓趨勢對準每個頻率點
    - 寬度小於約半個視窗的凸起會保留在殘差裡；更寬的結構會被視為趨勢
    - 頻譜兩端約半個視窗內，因 mode="nearest" 補值，趨勢估計較不可靠
    """
    # 確認頻譜每一格相差多少Hz：頻率軸上相鄰兩點的間隔，也就是頻率解析度。Welch 的頻率軸是等間隔的，所以取前兩點相減就好
    df_hz = freqs[1] - freqs[0]
    
    # 將「想看20 kHz附近」換算成「要看幾個頻率點」。20,000 ÷ 122 ≈ 163個點；並確保視窗是奇數，才能有一個正中央的頻率點。
    win = max(3, int(window_hz / df_hz) | 1)     # 轉成點數並取奇數
    
    # 對每個頻率點，都看它附近約20 kHz的資料，取中位數當作當地的寬頻背景高度，最後連成一條趨勢線。
    trend = median_filter(diff_db, size=win, mode="nearest")
    
    # 回傳兩條曲線：趨勢本身，以及「原始曲線減掉趨勢」的殘差，也就是局部凸起。
    return trend, diff_db - trend


# ============================================================
# 讀取 manifest / events
# ============================================================
# 讀 manifest，而且把 0713 的欄位名稱先轉成跟舊版一致
def read_manifest(manifest_path: Path) -> pd.DataFrame:    # 輸入是 manifest_path, 型別標註 Path 表示你希望它是一個路徑物件, -> pd.DataFrame 表示這個函式最後會回傳一個 pandas 表格
    # 把 manifest_path 這個 CSV 檔讀進來，存成 df
    df = pd.read_csv(manifest_path)
    df.columns = df.columns.str.strip()          # 欄位名稱去除前後空白

    # 0713 的 transition_dual_manifest 欄位名稱和前幾天不同，先統一改成後面流程共用的名字。
    rename_map = {}

    # 舊版: dual_index vs 0713: output_index
    if "dual_index" not in df.columns and "output_index" in df.columns:
        rename_map["output_index"] = "dual_index"

    # 舊版: dual_file vs 0713: merged_file
    if "dual_file" not in df.columns and "merged_file" in df.columns:
        rename_map["merged_file"] = "dual_file"

    # 如果 rename_map 不是空的，代表真的有欄位要改名，就用 df.rename(...) 把欄位名稱改掉
    if rename_map:
        df = df.rename(columns=rename_map)

    # 後面流程一定會用到哪些欄位
    required = {
        "session_id",
        "dual_index",
        "position",
        "pressure_bar",
        "leak_status",
        "capture_start_elapsed_sec",
        "capture_end_elapsed_sec",
        "dual_file",
    }
    # 檢查：目前這個 CSV 讀進來後，有沒有缺少必要欄位。
    missing = sorted(required - set(df.columns))
    
    # 如果真的缺欄位，就直接報錯並停止
    if missing:
        raise ValueError(f"manifest 缺少欄位 {missing}: {manifest_path}")
        
    # 文字欄位去除前後空白，避免 "leak0 " 或 "straight20cm " 這類值造成比對失敗
    for col in ("leak_status", "position"):
        df[col] = df[col].astype(str).str.strip()

    # 時間欄位必須是數字，而且結束時間要晚於開始時間
    t0 = pd.to_numeric(df["capture_start_elapsed_sec"], errors="coerce")
    t1 = pd.to_numeric(df["capture_end_elapsed_sec"], errors="coerce")
    bad = t0.isna() | t1.isna() | (t1 <= t0)
    if bad.any():
        print(f"⚠️ {manifest_path.name}：{int(bad.sum())} 列時間欄位無效，已排除")
        df = df[~bad].copy()
        
    # 前提檢查：每一列都必須對應一個獨立的 TDMS 檔
    # （之後計算 PSD 會讀整個檔案；若一個檔案對應多列，篩選後即使只剩一列，檔案裡仍混有其他片段的資料）
    filenames = df["dual_file"].astype(str).str.strip().map(lambda s: PureWindowsPath(s).name)
    dup = filenames[filenames.duplicated(keep=False)].unique()
    if len(dup):
        raise SystemExit(
            f"{manifest_path.name}：有 {len(dup)} 個 TDMS 檔對應到多列 manifest，例如 {list(dup[:5])}。\n"
            "目前的 PSD 計算是讀整個檔案，這種資料需要先依時間切出片段才能分析。"
        )

    return df

# ============================================================
# 讀取 events 檔
# ============================================================
def read_events(events_path: Path) -> pd.DataFrame:
    # 把事件 CSV 讀進來, 這份表通常是在記錄：干擾種類, 開始時間, 結束時間
    df = pd.read_csv(events_path)
    df.columns = df.columns.str.strip()          # 欄位名稱去除前後空白

    # 定義事件檔最少要有的欄位
    required = {"disturbance_type", "start_elapsed_sec", "end_elapsed_sec"}
    # 檢查事件檔有沒有缺必要欄位
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"events 檔缺少欄位 {missing}: {events_path}")

    # 時間欄位必須是數字，而且結束時間不可早於開始時間（瞬間事件允許兩者相同）
    t0 = pd.to_numeric(df["start_elapsed_sec"], errors="coerce")
    t1 = pd.to_numeric(df["end_elapsed_sec"], errors="coerce")
    bad = t0.isna() | t1.isna() | (t1 < t0)
    if bad.any():
        print(f"⚠️ {events_path.name}：{int(bad.sum())} 個事件時間無效，已排除")
        df = df[~bad].copy()
        
    return df


# ============================================================
# 找出 session 的兩份標註檔
# ============================================================
# 從候選檔中挑出一份
def _pick_one(candidates: list[Path], kind: str, session_dir: Path) -> Path | None:
    """從候選檔中挑一個：依檔名排序後取第一個；有多個時印出提示。"""
    # 去除重複，並依路徑的字母順序排序
    candidates = sorted(set(candidates))
    # 一份都沒找到就回傳 None
    if not candidates:
        return None
    # 如果有多份時印出提示
    if len(candidates) > 1:
        print(f"⚠️ {session_dir.name} 有 {len(candidates)} 份 {kind}，"
              f"使用 {candidates[0].name}；其他：{[c.name for c in candidates[1:]]}")
    return candidates[0]      # 回傳排序後的第一份


# 找出 session 的兩份標註檔
@lru_cache(maxsize=None)
def find_session_annotation_files(session_dir: Path) -> tuple[Path | None, Path | None]:
    """根據新舊兩種命名規則，找出 session 的 manifest 與 events 檔。
    - 舊版：dual_session_manifest_*.csv / disturbance_events_*.csv
    - 0713：transition_dual_manifest_*.csv / transition_dual_disturbance_events_*.csv
    有多份候選檔時，依檔名排序取第一份並印出提示，確保每次結果一致。
    """
    # 找 manifest 候選檔案, 資料有兩種命名
    manifest_path = _pick_one(
        list(session_dir.glob("dual_session_manifest_*.csv"))          # 舊版：dual_session_manifest_*.csv
        + list(session_dir.glob("transition_dual_manifest_*.csv")),    # 0713：transition_dual_manifest_*.csv
        "manifest", session_dir,
    )
    # 找 events 檔案。也是同時支援：舊版命名, 0713 命名
    events_path = _pick_one(
        list(session_dir.glob("disturbance_events_*.csv"))
        + list(session_dir.glob("transition_dual_disturbance_events_*.csv")),
        "events", session_dir,
    )

    return manifest_path, events_path


# ============================================================
# 把 manifest 裡記錄的檔案路徑，轉換成這台 Mac 上實際的檔案位置
# ============================================================
@lru_cache(maxsize=None)     # 記住結果的裝飾器
def _tdms_index(session_dir: Path) -> dict[str, list[Path]]:
    """掃描一次 session 資料夾（含子資料夾），建立「檔名 → 路徑清單」對照表。"""
    # 先建立一個空的字典
    index: dict[str, list[Path]] = {}
    
    # rglob 會往下找所有層的子資料夾，"*.tdms" 代表所有副檔名是 .tdms 的檔案
    for p in session_dir.rglob("*.tdms"):
        # 檢查字典裡有沒有這個檔名。有的話，就取出已經存在的清單；沒有的話，就先放一個空清單進去，再取出這個空清單。
        index.setdefault(p.name, []).append(p)     # .append(p)：把這個檔案的完整路徑加到清單裡。
    return {name: sorted(paths) for name, paths in index.items()}


def resolve_dual_tdms_path(session_dir: Path,                # session_dir：這個 session 的資料夾
                           dual_file_value: str) -> Path:    # dual_file_value：manifest 裡某一列記錄的 TDMS 路徑字串
    """manifest 記錄的是擷取電腦（Windows）上的完整路徑；這裡只取檔名，
    再到本機的 session 資料夾找對應的 TDMS 檔。"""    
    # 先把 dual_file_value 轉成字串，再把前後空白去掉    
    raw = str(dual_file_value).strip()

    # # 用 Windows 規則解析，才能正確取出檔名
    filename = PureWindowsPath(raw).name

    # 正常情況：檔案就在 session 資料夾最上層
    candidate = session_dir / filename
    if candidate.exists():
        return candidate

    # 備用：到子資料夾找（整個 session 只掃描一次）
    matches = _tdms_index(session_dir).get(filename, [])
    if len(matches) > 1:
        print(f"⚠️ {session_dir.name} 有 {len(matches)} 個同名檔 {filename}，"
              f"使用 {matches[0].relative_to(session_dir)}")
        
    # 如果有找到符合的檔案，就回傳第一個
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"找不到對應 TDMS 檔案，session_dir={session_dir}, dual_file={dual_file_value}, filename={filename}")


# ============================================================
# 把 session 中每個片段標成 tap 或 background
# ============================================================
def build_slice_records(
    session_dir: Path,                   # 這個 session 的資料夾
    manifest_path: Path,                 # 這個 session 的 manifest CSV 路徑
    events_path: Path,                   # 這個 session 的 events CSV 路徑
    position_filter: set[str] | None,    # 如果你只想保留某些位置，就用這個篩
    background_guard_sec: float,         # 背景要離敲擊多遠，才算安全
    event_overlap_sec: float,            # 判斷 tap 時，事件邊界要不要稍微放寬
) -> list[SliceRecord]:                  # 最後會回傳「很多個片段紀錄」
    # 讀進兩份資料：manifest 片段清單, events 事件清單
    manifest = read_manifest(manifest_path)
    events = read_events(events_path)

    # 只保留敲擊事件：和資料夾判斷使用同一套名稱規則（tap / tab_pipe / tappipe）
    types = events["disturbance_type"].astype(str).map(normalize_name)
    tap_events = events[types.apply(
        lambda s: any(k in s for k in ("tap",) + TAP_FOLDER_KEYWORDS))].copy()

    # 如果這個 session 裡根本沒有敲擊事件，那就不用分析了，直接回傳空清單
    if tap_events.empty:
        found = events["disturbance_type"].astype(str).value_counts().to_dict()
        print(f"    ⚠️ {session_dir.parent.name}/{session_dir.name}：事件中沒有敲擊，"
              f"實際的事件類型為 {found}（資料夾分類或事件標註可能不一致）")
        return []

    # 先準備一個空清單 records，等等每判斷完一個片段，就把結果塞進去。
    records: list[SliceRecord] = []

    # 從路徑中抓出批次名稱
    batch_name = next(
        (part for part in session_dir.parts if part.startswith("disturbance_ann_")),"unknown_batch",)
    
    # 補充這個 session 的身分資訊
    disturbance_dir = session_dir.parent.name    # 上一層資料夾名稱
    disturbance_id = session_dir.name            # 這個 session 自己的資料夾名稱

    # 逐列掃描 manifest 裡的每一個片段。
    for row in manifest.itertuples(index=False):
        # 只拿 leak0，因為現在是要找管子的共振響應，避免混入真的洩漏訊號
        if str(row.leak_status).lower() != "leak0":
            continue
        # 如果你有設定只看某些位置，那不在指定位置的片段也跳過
        if position_filter and row.position not in position_filter:
            continue
        
        # 把這個片段的開始時間、結束時間抓出來
        t_start = float(row.capture_start_elapsed_sec)
        t_end = float(row.capture_end_elapsed_sec)
  
        # 判斷 1：判斷這個片段是否有落在敲擊事件內
        # 把這個片段的時間區間 t_start ~ t_end -> 去和每一個 tap event 的時間區間比較 -> 只要有任一個重疊，就算 in_tap = True
        in_tap = any(overlaps(t_start, t_end,
                float(ev.start_elapsed_sec) - event_overlap_sec,    # event_overlap_sec 是額外放寬邊界:用途是避免標註切得太緊，漏掉邊界附近的 tap 片段
                float(ev.end_elapsed_sec) + event_overlap_sec,
            )
            for ev in tap_events.itertuples(index=False)
        )

        # 判斷 2：判斷這個片段是否太靠近敲擊事件，若太靠近，就不要拿來當背景，避免背景被敲擊尾波污染
        # 如果某片離 tap 太近 -> 可能會沾到敲擊尾波、殘響 -> 那就不適合拿來當乾淨背景
        near_tap = any(overlaps(t_start, t_end,
                float(ev.start_elapsed_sec) - background_guard_sec,     #  background_guard_sec：背景要離敲擊多遠，才算安全
                float(ev.end_elapsed_sec) + background_guard_sec,
            )
            for ev in tap_events.itertuples(index=False)
        )

        # 最後分類規則
        if in_tap:
            label = "tap"    # 如果片段落在 tap 事件裡 -> 標成 "tap"
        elif near_tap:
            continue         # 否則，如果它太靠近 tap -> 直接丟掉，不用
        else:
            label = "background"    # 否則 -> 標成 "background"
        
        # 確定要使用這個片段後，才去找對應的 TDMS 檔
        tdms_path = resolve_dual_tdms_path(session_dir, row.dual_file) 

        # 如果這個片段最後被判定為 tap 或 background，就把它包成一個 SliceRecord 加進 records
        records.append(
            SliceRecord(
                session_dir=session_dir,               # 這片屬於哪個 session
                tdms_path=tdms_path,                   # TDMS 檔在哪
                session_id=str(row.session_id),
                batch_name=batch_name,                 # 是哪個 batch
                disturbance_dir=disturbance_dir,
                disturbance_id=disturbance_id,
                position=str(row.position),
                pressure_bar=float(row.pressure_bar),  # 壓力是多少
                dual_index=int(row.dual_index),        # 在原始檔裡是第幾片
                t_start=t_start,                       # 開始時間、結束時間
                t_end=t_end,   
                label=label,                           # 最後被標成 tap 還是 background
            )
        )

    return records


# ============================================================
# 讀取 TDMS 指定聲道，把整段波形抓出來
# ============================================================
def read_tdms_channel(filepath: Path,                      # filepath：TDMS 檔案路徑
                      channel_name: str) -> np.ndarray:    # channel_name：你想讀的聲道名稱，例如 ai1
    """從 TDMS 檔讀出指定聲道。用 open 串流讀取，只載入需要的那個聲道。"""
    target = channel_name.lower()
    with TdmsFile.open(filepath) as tdms:
        for group in tdms.groups():
            for channel in group.channels():
                if channel.name.lower() == target:
                    return np.asarray(channel[:], dtype=np.float64)
        available = [c.name for g in tdms.groups() for c in g.channels()]
    raise ValueError(f"在 {filepath} 找不到聲道 {channel_name}，可用聲道有: {available}")


# ============================================================
# 對多個片段做平均 PSD(每支 TDMS 的 PSD 只算一次，分位置分析時直接重用)
# ============================================================
_PSD_CACHE: dict = {}    # 記住算過的結果
# 算一個檔案的頻譜，並記起來
def file_psd(tdms_path, channel_name, fs, nperseg):
    # 用「檔案 + 聲道 + 取樣率 + nperseg」當作 key。第一次遇到這個組合時才真的讀檔、計算；之後再遇到就直接回傳記住的結果。
    key = (str(tdms_path), channel_name, fs, nperseg)
    if key not in _PSD_CACHE:
        # 讀出訊號，用 Welch 方法算頻譜
        signal = read_tdms_channel(tdms_path, channel_name)
        # 太短的片段直接跳過，避免頻率軸長度不一致
        _PSD_CACHE[key] = None if len(signal) < nperseg else welch(signal, fs=fs, nperseg=nperseg)
    return _PSD_CACHE[key]

# 把多個片段的頻譜平均起來
def averaged_psd(records: list[SliceRecord],    # 很多個片段清單
                 channel_name: str,             # 要讀哪個聲道
                 fs: float,                     # 取樣率
                 nperseg: int):                 # Welch method 的分段長度
    # 如果 records 是空的，代表根本沒有可用片段，那就直接報錯。
    if not records:
        raise ValueError("沒有可用片段可計算 PSD")
    freqs, psd_sum, n_used, n_skipped = None, None, 0, 0
    for record in records:
        res = file_psd(record.tdms_path, channel_name, fs, nperseg)
        if res is None:
            n_skipped += 1
            continue
        
        # 第一個可用的片段：記下頻率軸，並把它的頻譜複製一份當作累加的起點
        f, p = res
        if psd_sum is None:
            freqs, psd_sum = f, p.copy()     # 一定要複製，否則下面的 += 會改到快取裡的資料
        else:
            psd_sum += p
        n_used += 1
        
    # 所有片段都不能用就報錯；否則回傳頻率軸和平均頻譜。
    if n_used == 0:
        raise ValueError(f"所有片段都短於 nperseg={nperseg}")
        
    if n_skipped:
        print(f"  ⚠️ {channel_name}：{n_skipped} 個片段短於 nperseg={nperseg}，已略過（使用 {n_used} 個）")
        
    return freqs, psd_sum / n_used


# 先各距離平均，再把各距離等權重平均
def averaged_psd_balanced(records: list[SliceRecord], channel_name: str, fs: float, nperseg: int):
    """先在每個位置內平均，再把各位置的平均頻譜等權重平均。
    避免片段數較多的位置主導結果，也避免敲擊組和背景組的位置組成不同而造成偏差。"""
    if not records:
        raise ValueError("沒有可用片段可計算 PSD")
    # 找出這批片段裡有哪些位置，會自動去除重複
    positions = sorted({r.position for r in records})
    
    # 對每個位置：先篩出這個位置的片段，交給原本的 averaged_psd() 算出這個位置的平均頻譜，再把結果存進 per_pos 清單
    freqs, per_pos = None, []
    for pos in positions:
        f, p = averaged_psd([r for r in records if r.position == pos], channel_name, fs, nperseg)
        freqs = f
        per_pos.append(p)
    return freqs, np.mean(per_pos, axis=0)


# ============================================================
# 把使用到的片段整理成表格，方便後續追查
# ============================================================ 
def build_summary_df(records: list[SliceRecord]) -> pd.DataFrame:       # 輸入：records 也就是很多個 SliceRecord
    # 把原本一筆一筆的 SliceRecord，整理成一張可以輸出成 CSV 的表    
    rows = [
        {
            "label": r.label,                       # 這片是 tap 還是 background
            "batch_name": r.batch_name,             # 屬於哪一天/哪一批，例如 disturbance_ann_0707
            "disturbance_dir": r.disturbance_dir,   # 上一層干擾資料夾名稱
            "disturbance_id": r.disturbance_id,     # 這個 session 自己的名字
            "position": r.position,                 # 位置，例如 straight20cm
            "pressure_bar": r.pressure_bar,         # 壓力值
            "session_id": r.session_id,             # session 編號
            "dual_index": r.dual_index,             # manifest 裡的索引
            "tdms_path": str(r.tdms_path),          # 對應的 TDMS 檔路徑
            "t_start": r.t_start,                   # 片段開始時間
            "t_end": r.t_end,                       # 片段結束時間
        }
        for r in records
    ]
    return pd.DataFrame(rows)


# ============================================================
# 判斷某個峰值落在哪個你定義好的頻帶
# ============================================================
def find_band_for_frequency(freq_hz: float):
    """
    判斷某個峰值頻率屬於哪一段建議頻帶。
    """
    # 開始逐一檢查 ALL_BANDS 裡定義的每一段頻帶, 每次迴圈都會拿到：band_name：頻帶名稱 , f_lo：下界 , f_hi：上界 , band_type：類型，例如 core 或 secondary
    for band_name, f_lo, f_hi, band_type in ALL_BANDS:
        # 如果這個輸入頻率 freq_hz 落在這段頻帶範圍內，就表示找到了。
        if f_lo <= freq_hz < f_hi:
            return band_name, f_lo, f_hi, band_type
    # 如果整個 ALL_BANDS 都找完了，還是沒有任何一段包含這個頻率，那就回傳："off_band"：不在已定義頻帶內 , np.nan, np.nan：沒有頻帶上下界 , "其他"：類型歸到其他
    return "off_band", np.nan, np.nan, "其他"


# ============================================================
# 依正式版共振頻帶，把驗證通過的峰整理成摘要表（每段一列）
# ============================================================
def build_band_summary_table(peaks_df: pd.DataFrame) -> pd.DataFrame:    # 輸入：peaks_df 也就是峰值清單表
    """
    依正式版共振頻帶整理摘要表，每段頻帶一列。
    - 只統計跨天驗證通過的峰
    - 代表峰：該段內局部凸起（local_peak_dB）最大的峰
    - 正式版裡這次沒有任何驗證通過峰的頻帶也會列出（peak_count = 0），方便發現「已核准、但這次沒有證據支持」的頻帶
    """
    # 如果沒有正式版頻帶清單就直接結束
    if not ALL_BANDS:
        return pd.DataFrame()
    
    # 只留下驗證通過的峰值
    # 穩定共振帶：先篩資料，只保留 0713 驗證通過的峰值，不是把所有搜尋到的峰都拿來整理，而是只整理那些「跨天驗證後仍然成立」的峰。
    valid_df = peaks_df[peaks_df["validation_0713_pass"]]
    
    # 逐一檢查每段頻帶，找出落在裡面的峰
    rows = []      # 先準備一個空清單，等等每整理出一段頻帶，就加一列進去。
    # 開始逐一掃過你定義好的每一段頻帶，每段頻帶是 (名稱, 下界, 上界, 類型)
    for band_name, f_lo, f_hi, band_type in ALL_BANDS:
        # 從 valid_df 裡挑出：「所有落在這段頻帶範圍內的峰值」，sub 可以理解成：某一段頻帶內的子集合。
        sub = valid_df[(valid_df["frequency_Hz"] >= f_lo) & (valid_df["frequency_Hz"] < f_hi)]

        # 建立這個頻帶的一列摘要資料
        row = {
            # 頻帶本身的資訊(名稱、類型、範圍)
            "band_name": band_name,
            "band_type": band_type,
            "band_start_Hz": f_lo,
            "band_end_Hz": f_hi,
            "band_range": f"{int(f_lo)}-{int(f_hi)} Hz",
            "peak_count": len(sub),                       # 這段裡驗證通過的峰數
            "representative_peak_Hz": np.nan,
            "representative_peak_width_Hz": np.nan,
            "max_local_peak_dB": np.nan,                  # 搜尋資料中，局部凸起最大值
            "max_validation_0713_local_dB": np.nan,       # 0713 中，局部凸起最大值
        }
        
        # 有峰的話，填入代表峰的資訊
        if not sub.empty:
            rep = sub.loc[sub["local_peak_dB"].idxmax()]     # 找出局部凸起最大那個峰的位置，把那一整列取出來，當作代表峰。
            # 把摘要資料預設的 nan 換成實際數值：代表峰的頻率和寬度，以及這段裡搜尋資料和 0713 的最大局部凸起。
            row.update(
                representative_peak_Hz=rep["frequency_Hz"],
                representative_peak_width_Hz=rep["peak_width_Hz"],
                max_local_peak_dB=sub["local_peak_dB"].max(),
                max_validation_0713_local_dB=sub["validation_0713_local_dB"].max(),
            )
        rows.append(row)

    # 組成表格並排序
    return (pd.DataFrame(rows)
            .sort_values(["band_type", "band_start_Hz"])
            .reset_index(drop=True))

# ============================================================
# 繪圖函式
# ============================================================
# 畫前三天搜尋頻譜 vs 0713 驗證頻譜疊圖
def plot_search_vs_validation_overlay(
    search_spectrum_df: pd.DataFrame,
    validation_spectrum_df: pd.DataFrame,
    output_png: Path,
    channel: str = "",
    local_peak_db: float | None = None,     # 局部凸起門檻，畫成水平虛線
    min_freq_hz: float | None = None,       # 不評估的低頻範圍，畫成灰色陰影
) -> None:
    """
    畫「前三天 vs 0713」的對照圖，分上下兩張：
    - 上：原始的敲擊/背景 dB 差，以及寬頻趨勢（看高原與趨勢長什麼樣子）
    - 下：扣除趨勢後的局部凸起（找峰與跨天驗證實際使用的曲線）
    兩張圖都用陰影標出正式版頻帶：core 紅色、secondary 紫色。
    """
    from matplotlib.patches import Patch

    # 換成 kHz，數字比較好讀
    f_s = search_spectrum_df["frequency_Hz"] / 1000        
    f_v = validation_spectrum_df["frequency_Hz"] / 1000

    # 建立圖畫
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 上圖：原始曲線 + 寬頻趨勢
    ax1.plot(f_s, search_spectrum_df["tap_vs_bg_dB"], lw=1.2, color="tab:blue",
             label="搜尋（0624+0702+0707）")
    ax1.plot(f_s, search_spectrum_df["tap_vs_bg_trend_dB"], lw=1.2, ls="--", color="tab:blue",
             alpha=0.6, label="搜尋的寬頻趨勢")
    ax1.plot(f_v, validation_spectrum_df["validation_0713_tap_vs_bg_dB"], lw=1.0,
             color="tab:orange", alpha=0.9, label="驗證（0713）")
    ax1.set_ylabel("敲擊 / 背景 (dB)")
    ax1.set_title(f"原始曲線與寬頻趨勢（{channel}）")

    # 下圖：扣除趨勢後的局部凸起
    ax2.plot(f_s, search_spectrum_df["tap_vs_bg_local_dB"], lw=1.2, color="tab:blue",
             label="搜尋：局部凸起")
    ax2.plot(f_v, validation_spectrum_df["validation_0713_local_dB"], lw=1.0,
             color="tab:orange", alpha=0.9, label="驗證：局部凸起")
    if local_peak_db is not None:
        ax2.axhline(local_peak_db, color="gray", ls=":", lw=1, label=f"門檻 {local_peak_db:g} dB")
    ax2.set_ylabel("局部凸起 (dB)")
    ax2.set_xlabel("頻率 (kHz)")
    ax2.set_title("扣除寬頻趨勢後的局部凸起（找峰與跨天驗證實際使用的曲線）")

    # 兩張圖共同的標示：正式版頻帶、不評估的低頻範圍
    band_patches = [
        Patch(color="tab:red", alpha=0.15, label="正式版 core"),
        Patch(color="tab:purple", alpha=0.15, label="正式版 secondary"),
    ]
    for ax in (ax1, ax2):
        for _, f_lo, f_hi, band_type in ALL_BANDS:
            ax.axvspan(f_lo / 1000, f_hi / 1000,
                       color="tab:red" if band_type == "core" else "tab:purple", alpha=0.15)
        if min_freq_hz:
            ax.axvspan(0, min_freq_hz / 1000, color="gray", alpha=0.1)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + (band_patches if ALL_BANDS else []), fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 畫出所有候選峰，依「是否通過驗證」與「屬於哪段正式版頻帶」區分顏色
# ============================================================
def plot_peak_grouped_by_band(
    peaks_df: pd.DataFrame,    # 峰值表（某一個聲道的整體分析結果）
    output_png: Path,          # 輸出圖檔路徑
    channel: str = "",         # 聲道名稱，放在標題裡
) -> None:
    """畫出所有候選峰，依「是否通過跨天驗證」與「屬於哪段正式版頻帶」區分：
        - 淺灰空心：未通過跨天驗證
        - 深灰實心：通過驗證，但不在正式版頻帶內（例如尚未核准、或刻意排除）
        - 紅 / 紫實心：通過驗證，且落在正式版 core / secondary 頻帶內
        y 軸是局部凸起（local_peak_dB），也就是找峰實際使用的數值。
        第一次執行（尚無正式版）時，所有通過驗證的峰都會以深灰顯示，可用來決定要核准哪些頻帶。
    """
    # 一個候選峰都沒有，就不用畫圖
    if peaks_df.empty:
        return

    # 複製一份，避免新增欄位時改到原本的峰值表
    df = peaks_df.copy()
    
    # 查出每個峰屬於正式版的哪一段頻帶
    # find_band_for_frequency() 回傳 (名稱, 下界, 上界, 類型)；不屬於任何頻帶時名稱是 "off_band"
    band_info = df["frequency_Hz"].apply(find_band_for_frequency)
    df["band_name"] = [x[0] for x in band_info]     # 取第 1 個值：頻帶名稱
    df["band_type"] = [x[3] for x in band_info]     # 取第 4 個值：core / secondary
    df["f_kHz"] = df["frequency_Hz"] / 1000         # 頻率換成 kHz，x 軸的數字比較好讀

    # 建立一張寬 14、高 6 的圖
    fig, ax = plt.subplots(figsize=(14, 6))

    # 正式版頻帶的陰影
    # 先畫正式版頻帶的陰影（放在最底層，點會畫在陰影上面）
    # core 用淡紅色、secondary 用淡紫色；第一次執行時 ALL_BANDS 是空的，就不會畫任何陰影
    for _, f_lo, f_hi, band_type in ALL_BANDS:
        ax.axvspan(f_lo / 1000, f_hi / 1000,
                   color="tab:red" if band_type == "core" else "tab:purple", alpha=0.08)

    # 未通過跨天驗證的峰：淺灰空心點，當作背景參考
    # ~ 是「反轉」：validation_0713_pass 為 False 的才留下
    failed = df[~df["validation_0713_pass"]]
    ax.scatter(failed["f_kHz"], failed["local_peak_dB"], s=22,
               facecolors="none",          # 不填色 → 空心
               edgecolors="lightgray",     # 外框淺灰
               label="未通過跨天驗證")

    # 通過跨天驗證的峰
    passed = df[df["validation_0713_pass"]]
    
    # 其中不在正式版任何頻帶裡的：深灰實心點
    # 第一次執行時，所有通過驗證的峰都會在這裡；核准後，被刻意排除的頻帶（例如 10~20k）也會在這裡
    off = passed[passed["band_name"] == "off_band"]
    if not off.empty:
        ax.scatter(off["f_kHz"], off["local_peak_dB"], s=30,
                   color="dimgray", label="通過驗證、不在正式版")

    # 其中落在正式版頻帶裡的：依頻帶分組，core 紅色、secondary 紫色
    # groupby("band_name") 會把同一段頻帶的峰分成一組，每組畫一次，圖例就會列出每段頻帶的名稱
    for band_name, sub in passed[passed["band_name"] != "off_band"].groupby("band_name"):
        is_core = sub["band_type"].iloc[0] == "core"
        ax.scatter(sub["f_kHz"], sub["local_peak_dB"], s=40,
                   color="#d62728" if is_core else "#9467bd", label=band_name)

    # 標題、座標軸名稱
    ax.set_title(f"候選峰分布與正式版頻帶（{channel}）")
    ax.set_xlabel("頻率 (kHz)")
    ax.set_ylabel("局部凸起 local_peak_dB")
    
    # 顯示圖例（放右上角）和淡淡的格線
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)

    # 自動調整邊距、存檔（解析度 180、裁掉多餘空白），存完關閉以釋放記憶體
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    

# ============================================================
# 畫資料覆蓋圖：各批次、各位置用了多少敲擊 / 背景片段
# ============================================================
def plot_data_coverage(
    slice_summary_df: pd.DataFrame,    # 片段總表（build_summary_df() 的結果，含搜尋與驗證批次）
    output_png: Path,                  # 圖片儲存路徑
) -> None:
    """
    畫兩張長條圖，確認資料的分布：
    - 上：各批次（日期）的敲擊 / 背景片段數 → 確認資料不是集中在某一天
    - 下：各位置的敲擊 / 背景片段數 → 確認位置組成，以及敲擊與背景是否每個位置都有
    """
    if slice_summary_df.empty:
        return
    
    labels = ["background", "tap"]                 # 固定欄位順序，顏色才不會對錯
    colors = ["#a6cee3", "#fb9a99"]                # 背景淡藍、敲擊淡紅
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    # 上下兩張圖的做法一樣，只是分組的欄位不同：上圖依批次、下圖依位置
    for ax, by, title in [
        (ax1, "batch_name", "各批次的片段數"),
        (ax2, "position", "各位置的片段數"),
    ]:
        # 依「分組欄位 + 標籤」計數，再轉成「每組一列、敲擊與背景各一欄」的寬格式
        # reindex(columns=labels) 固定欄位順序；某種標籤完全沒有時補成 0
        pivot = (slice_summary_df.groupby([by, "label"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=labels, fill_value=0))

        pivot.plot(kind="bar", ax=ax, color=colors, rot=20)   # rot=20：x 軸文字稍作旋轉
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("片段數")
        ax.legend(["背景", "敲擊"])
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 把頻帶摘要表畫成圖，方便貼報告
# ============================================================
def save_band_summary_png(
    band_summary_df: pd.DataFrame,    # 頻帶摘要表（含 channel 欄位）
    output_png: Path,                 # 輸出的圖片路徑
) -> None:
    """
    把頻帶摘要表另存成表格圖片，方便直接貼報告或簡報。
    底色：表頭淡藍、core 淡藍白、secondary 淡米橘；
    峰數為 0 的列（已核准、但這次沒有證據支持）改用灰底灰字標出。
    """
    # 如果表格是空的，就直接不畫
    if band_summary_df.empty:
        return
    
    df = band_summary_df.reset_index(drop=True)
    
    # 整理成適合閱讀的表格：中文表頭、頻率換成 kHz
    # 挑出要顯示的欄位
    display_df = pd.DataFrame({
        "聲道": df["channel"],
        "頻帶": df["band_name"],
        "類型": df["band_type"],
        "範圍": [f"{lo / 1000:g}–{hi / 1000:g} kHz"
                 for lo, hi in zip(df["band_start_Hz"], df["band_end_Hz"])],
        "峰數": df["peak_count"].astype(int),
        "代表峰 (kHz)": (df["representative_peak_Hz"] / 1000).round(2),
        "搜尋 局部凸起 (dB)": df["max_local_peak_dB"].round(2),
        "0713 局部凸起 (dB)": df["max_validation_0713_local_dB"].round(2),
    })
    
    # 沒有峰的欄位顯示「—」，而不是 nan
    display_df = display_df.astype(object).where(display_df.notna(), "—")

    # 依列數決定圖的高度，表格不會被壓扁或留下大片空白
    fig_height = 0.5 * len(display_df) + 1.2
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")    # 純表格，不需要座標軸

    # 真正建立表格
    table = ax.table(
        cellText=display_df.values,    # 表格內容
        colLabels=display_df.columns,  # 欄位名稱當表頭
        cellLoc="center",              # 每格文字置中
        loc="center",                  # 整張表放在圖中央
    )
    # 調整表格外觀
    table.auto_set_font_size(False)    # 不要自動縮字
    table.set_fontsize(9)              # 字體大小設成 9
    table.scale(1, 1.4)                # 表格高度拉高一點，讓列與列之間比較不擠
    table.auto_set_column_width(col=list(range(display_df.shape[1])))  # 依內容自動調整欄寬

    # 逐格上色：r 是第幾列（0 是表頭）、c 是第幾欄
    no_evidence = (df["peak_count"] == 0).to_numpy()   # 峰數為 0 的列
    for (r, c), cell in table.get_celld().items():
        if r == 0:                                      # 表頭
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")
        elif no_evidence[r - 1]:                        # 已核准、但這次沒有證據支持
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(color="#888888")
        elif df.loc[r - 1, "band_type"] == "core":      # core
            cell.set_facecolor("#f7fbff")
        else:                                           # secondary
            cell.set_facecolor("#fff8f2")

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 收集片段的總入口：掃描 session → 整理每個 session 的片段 → 合併
# ============================================================
def collect_records(
    root: Path,                         # disturbance 的根目錄
    batch_filter: set[str],             # 要分析哪些批次，例如 0624、0702、0707
    position_filter: set[str] | None,   # 如果只想看某些位置，就傳進來；不限制就用 None
    background_guard_sec: float,        # 背景片段要離 tap 多遠才算安全
    event_overlap_sec: float,           # 判斷 tap 片段時，要不要把事件邊界放寬
) -> list[SliceRecord]:
    """
    收集指定批次中的所有敲擊 / 背景片段，並印出各批次的收集摘要。
    - 指定的批次完全找不到 session 時會警告（可能是批次名稱打錯）
    - session 存在、但沒有產生任何片段時會警告（可能是事件標註有問題）
    """
    # 找出符合條件的 session：敲擊類型、屬於指定批次、有 manifest 和 events
    session_dirs = list(iter_session_dirs(root, batch_filter=batch_filter))
    
    # 先準備一個空清單 all_records。等等每個 session 整理出來的片段，都會往這裡加。
    all_records: list[SliceRecord] = []
    stats: dict[str, dict] = {}         # 批次 → {session 數, 敲擊片段數, 背景片段數, 沒有片段的 session}

    # 開始逐個處理每個 session
    for session_dir in session_dirs:
        # 對這個 session_dir，去找它裡面的兩份標註檔：manifest_path, events_path
        manifest_path, events_path = find_session_annotation_files(session_dir)
        # 如果這個 session 缺檔：找不到 manifest, 或找不到 events, 那就直接跳過這個 session。
        if manifest_path is None or events_path is None:
            continue

        # 整理這個 session 的片段（讀標註、只留 leak0、分成敲擊 / 背景）
        # 讀 manifest -> 讀 events -> 只保留 leak0 -> 判斷哪些片段是 tap -> 哪些是 background -> 哪些因為太靠近 tap 要丟掉 -> 最後回傳這個 session 的 records。
        records = build_slice_records(
            session_dir=session_dir,
            manifest_path=manifest_path,
            events_path=events_path,
            position_filter=position_filter,
            background_guard_sec=background_guard_sec,
            event_overlap_sec=event_overlap_sec,
        )
        # 把這個 session 整理出來的 records，全部加進總清單 all_records
        # append(records)：會把整包 records 當成一個元素塞進去
        # extend(records)：會把裡面每一筆 SliceRecord 一筆一筆攤開加入
        all_records.extend(records)
        
        # 統計這個 session 屬於哪個批次、產生了多少片段
        batch = next((p for p in session_dir.parts if p.startswith("disturbance_ann_")), "unknown_batch")
        s = stats.setdefault(batch, {"sessions": 0, "tap": 0, "background": 0, "empty": []})
        s["sessions"] += 1
        s["tap"] += sum(r.label == "tap" for r in records)
        s["background"] += sum(r.label == "background" for r in records)
        if not records:
            s["empty"].append(f"{session_dir.parent.name}/{session_dir.name}")

    # 印出各批次的收集摘要
    for batch in sorted(batch_filter):
        if batch not in stats:
            print(f"  ⚠️ {batch}：找不到任何敲擊 session（批次名稱是否正確？）")
            continue
        s = stats[batch]
        print(f"  {batch}：session {s['sessions']} 個，敲擊 {s['tap']}、背景 {s['background']} 個片段")
        if s["empty"]:
            print(f"    ⚠️ 以下 session 沒有產生任何片段（原因見上方個別訊息）：{s['empty']}")


    return all_records


# ============================================================
# 安全檢查：確認「每個 TDMS 檔只被一個片段使用」
# ============================================================
def check_one_file_per_slice(records: list[SliceRecord], name: str) -> None:
    """
    確認每個 TDMS 檔只對應一個片段。
    averaged_psd() 會讀整個檔案算 PSD，所以若同一檔案對應多個片段，
    敲擊與背景會混在一起，結果就不正確。
    """
    # 把每個片段的檔案路徑拿出來計數
    counts = Counter(r.tdms_path for r in records)
    
    # 只留下出現超過一次的檔案(正常情況下 dup 會是空的)
    dup = {p: n for p, n in counts.items() if n > 1}
    
    # 只要找到任何一個重複的檔案，程式就會停止，一個片段都不會被拿去計算，並最多列出五個重複檔案。
    if dup:
        examples = [f"{p.name}（{n} 次）" for p, n in list(dup.items())[:5]]
        raise SystemExit(
            f"{name}：有 {len(dup)} 個 TDMS 檔對應到多個片段，例如 {examples}。\n"
            "目前的 PSD 計算是讀整個檔案，這種資料需要先依 t_start / t_end 切出片段才能分析。"
        )


# ============================================================
# 把任意範圍對齊到 10 kHz 搜尋網格（核准時使用）
# ============================================================
def snap_band_to_grid(band: dict) -> list[dict]:
    """
    把任意範圍換成對齊搜尋網格的 10 kHz 格子（起點為 10k 的倍數）。

    注意：這不是找共振頻帶的主要流程。程式自動產生的提案本來就是 10 kHz 格子，
    會直接原樣回傳；只有在核准前「人工修改了提案檔的範圍」時，才會實際進行對齊。
    - 和原範圍重疊至少 MIN_OVERLAP_HZ 的格子都納入
    - 範圍太窄、沒有格子重疊夠多時，取包含中心頻率的那一格
    """
    # 檢查頻段是否已經對齊
    lo, hi = int(band["f_low"]), int(band["f_high"])
    if hi - lo == GRID_WIDTH and lo % RES_GRID_STRIDE == 0:
        return [band]                                   # 已經對齊

    # 決定名稱前綴
    prefix = "res_core" if band["band_type"] == "core" else "res_sec"

    # 定義「產生一格」的小函式
    def make_cell(c_lo):     # c_lo為起點
        c_hi = c_lo + GRID_WIDTH
        return {
            **band,     # 先把原本頻帶的所有欄位複製過來
            "name": f"{prefix}_{c_lo // 1000:03d}_{c_hi // 1000:03d}k",
            "f_low": c_lo,
            "f_high": c_hi,
            "reason": f"{band.get('reason', '')}（由 {lo}~{hi} Hz 對齊）",      # reason 則會在原本的理由後面註明「由 25000~37000 Hz 對齊」，方便之後追查
        }

    # 找出重疊夠多的格子
    start = (lo // RES_GRID_STRIDE) * RES_GRID_STRIDE
    
    # 重疊長度的算法是「兩段的右端取較小者，減掉兩段的左端取較大者」
    cells = [
        make_cell(c_lo)
        for c_lo in range(start, hi, RES_GRID_STRIDE)
        if min(hi, c_lo + GRID_WIDTH) - max(lo, c_lo) >= MIN_OVERLAP_HZ
    ]
    
    # 範圍太窄時的備案:取中心所在的格子
    if not cells:                                      
        center = (lo + hi) / 2
        cells = [make_cell(int(center // RES_GRID_STRIDE) * RES_GRID_STRIDE)]
        
    # 檢查：只接受有效範圍內的格子：下限為共振格子的起點（10k），上限為 GRID_F_MAX
    first_cell = -(-GRID_F_MIN // RES_GRID_STRIDE) * RES_GRID_STRIDE
    valid = [c for c in cells if c["f_low"] >= first_cell and c["f_high"] <= GRID_F_MAX]
    if not valid:
        raise SystemExit(
            f"{band.get('name', '')} 的範圍 {lo}~{hi} Hz 超出可用的共振格子範圍"
            f"（{first_cell}~{GRID_F_MAX} Hz），請修正提案後再核准"
        )
    if len(valid) < len(cells):
        dropped = [c["name"] for c in cells if c not in valid]
        print(f"  ⚠️ {band.get('name', '')}：超出範圍的格子已略過 {dropped}")
    return valid

    
# ============================================================
# 比對這次驗證通過的峰，跟目前正式版共振頻帶有沒有落差
# ============================================================
def audit_band_drift(
    peaks_df: pd.DataFrame,                      # 前面找到的候選峰清單,也就是 result_df
    all_bands: list[tuple],                      # 要比對的頻帶，通常是目前正式版的 ALL_BANDS
    boundary_margin_hz: float = 500.0,           # 峰離格子邊緣小於此值（Hz）就提示
    excluded_ranges: set[tuple] | None = None,   # 上次核准時刻意排除的範圍，這些格子裡的峰不列為落差
) -> dict:
    """
    拿這次的結果和正式版比對，只負責回報，不會修改 approved_resonance_bands.json。
    檢查兩件事：
    1. uncovered_peaks：跨天驗證通過，但不在任何正式版頻帶裡的峰（可能漏掉的共振）
    2. boundary_close_peaks：落在頻帶內，但離格子邊緣太近的峰（共振可能橫跨兩格，可考慮是否也納入相鄰的格子）
    """
    # 只留下跨天驗證通過的峰
    valid_df = peaks_df[peaks_df["validation_0713_pass"] == True].copy()
    # 如果篩完一個都不剩,就直接回傳兩張空表
    if valid_df.empty:
        return {"uncovered_peaks": pd.DataFrame(), "boundary_close_peaks": pd.DataFrame()}
    
    # 用傳進來的 all_bands 判斷每個峰屬於哪一段（左閉右開，與網格一致）
    def lookup(freq_hz):     # 給一個頻率，查它屬於 all_bands 裡的哪一段頻帶
        for name, f_lo, f_hi, _ in all_bands:     # 逐一檢查每一段頻帶。每一段都是 (名稱, 下界, 上界, 類型) 這樣的 tuple
            # 左閉右開：包含下界、不包含上界。所以 30,000 Hz 屬於 30～40k，不屬於 20～30k，每個頻率只會屬於一格
            if f_lo <= freq_hz < f_hi:
                return name, f_lo, f_hi
        return "off_band", np.nan, np.nan

    # 標註每個峰落在哪個頻帶，不屬於任何頻帶的峰，名稱會是 "off_band"。
    band_info = valid_df["frequency_Hz"].apply(lookup)
    valid_df["band_name"] = [x[0] for x in band_info]
    valid_df["band_low"] = [x[1] for x in band_info]
    valid_df["band_high"] = [x[2] for x in band_info]

    # 1. 不在任何正式版頻帶裡的峰，依局部凸起由強到弱排序
    uncovered = (valid_df[valid_df["band_name"] == "off_band"].sort_values("local_peak_dB", ascending=False))

    # 刻意排除的格子裡的峰，已經人工判斷過，不再列為落差
    if excluded_ranges and not uncovered.empty:
        in_excluded = uncovered["frequency_Hz"].apply(
            lambda f: any(lo <= f < hi for lo, hi in excluded_ranges))
        uncovered = uncovered[~in_excluded]
        
    # 2. 在頻帶內、但離格子邊緣太近的峰
    # 先取出「有被涵蓋到」的峰(covered)。對每一個峰,算它離「所在頻帶下界」的距離、跟離「所在頻帶上界」的距離,兩者取比較小的那個當作 dist_to_edge_Hz——也就是「這個峰離它所在頻帶最近的那條邊,有多近」。
    covered = valid_df[valid_df["band_name"] != "off_band"].copy()
    covered["dist_to_edge_Hz"] = np.minimum(
        (covered["frequency_Hz"] - covered["band_low"]).abs(),
        (covered["frequency_Hz"] - covered["band_high"]).abs(),
    )
    
    # 接著篩出 dist_to_edge_Hz <= boundary_margin_hz(預設小於等於 500 Hz)的峰
    boundary_close = (covered[covered["dist_to_edge_Hz"] <= boundary_margin_hz]
                      .sort_values("dist_to_edge_Hz"))

    # uncovered_peaks:完全沒被涵蓋到的峰, boundary_close_peaks:落在頻帶內、但太靠近邊界的峰
    return {"uncovered_peaks": uncovered, "boundary_close_peaks": boundary_close}


# ============================================================
# 接收 audit_band_drift 算出來的結果，負責把結果印出來
# ============================================================
def report_band_drift(drift: dict) -> bool:
    """
    印出頻帶落差警告。回傳 True 代表這次有偵測到落差，值得你回去人工檢查。
    """
    # 先把兩個 DataFrame 分別取出來：被漏掉的峰、貼近邊界的峰
    uncovered = drift["uncovered_peaks"]             # 完全沒被涵蓋到的峰
    boundary_close = drift["boundary_close_peaks"]   # 落在頻帶內、但太靠近邊界的峰
    # 只要任一張表有內容，就算有落差。
    has_drift = not uncovered.empty or not boundary_close.empty     

    # 如果兩個 DataFrame 都是空的，就印出一個綠色勾勾的訊息安心通知你「這次沒問題」，然後直接回傳 False
    if not has_drift:
        print("\n✅ 頻帶比對：這次驗證通過的峰全部落在目前正式版共振頻帶內，且離邊界夠遠，沒有偵測到落差。")
        return False

    print("\n" + "!" * 70)
    print("⚠️ 頻帶比對：偵測到和目前正式版共振頻帶有落差，建議人工檢查")
    print("!" * 70)

    # 如果「完全落在頻帶外」的峰不是空的，就印出總數
    if not uncovered.empty:
        print(f"\n有 {len(uncovered)} 個跨天驗證通過的峰，落在目前所有頻帶範圍之外：")
        print(uncovered[["channel", "frequency_Hz", "local_peak_dB", "validation_0713_local_dB"]].head(10).to_string(index=False))
        
    # 落在頻帶內、但離邊界太近的峰
    if not boundary_close.empty:
        print(f"\n有 {len(boundary_close)} 個峰落在頻帶內，但離格子邊緣不到容許值，共振可能橫跨兩格，可考慮是否也納入相鄰的格子：")
        print(boundary_close[["channel", "frequency_Hz", "band_name", "dist_to_edge_Hz", "local_peak_dB"]].head(10).to_string(index=False))

    print("\n這不會自動修改正式版 approved_resonance_bands.json。")
    print("落差稽核看的是各聲道整體分析的單一峰，僅供參考；是否調整頻帶，請以提案（分位置、分聲道投票）為準。")
    print("若確定要調整，可編輯本次的 proposed_resonance_bands.json 後再核准。")
    print("!" * 70 + "\n")
    return True


# ============================================================
# 找峰（扣除寬頻趨勢）＋ 0713 跨天驗證，整體分析與分位置分析共用
# ============================================================
def find_validated_peaks(tap_records, bg_records, val_tap_records, val_bg_records, args):
    """找峰 + 0713 驗證。回傳 (峰值表, 搜尋頻譜表, 驗證頻譜表)。"""
    eps = 1e-20    # 避免分母或分子剛好接近 0 時出錯
    
    # 敲擊和背景只使用兩邊都有的位置，避免等權重平均時兩組的位置組成不同
    def _keep_common_positions(a, b, name):
        pos_a, pos_b = {r.position for r in a}, {r.position for r in b}
        common = pos_a & pos_b
        if (pos_a | pos_b) - common:
            print(f"  ⚠️ {name}：以下位置只有敲擊或只有背景，已略過 {sorted((pos_a | pos_b) - common)}")
        return [r for r in a if r.position in common], [r for r in b if r.position in common]
    
    tap_records, bg_records = _keep_common_positions(tap_records, bg_records, "搜尋資料")
    if not tap_records:
        raise SystemExit("搜尋資料中，敲擊與背景沒有共同的位置")
    val_tap_records, val_bg_records = _keep_common_positions(val_tap_records, val_bg_records, "驗證資料")
    
    # -- 1. 算出「敲擊比背景高多少」 --
    # tap_records 算出 psd_tap , bg_records 算出 psd_bg
    freqs, psd_tap = averaged_psd_balanced(tap_records, args.channel, args.fs, args.nperseg)
    _, psd_bg = averaged_psd_balanced(bg_records, args.channel, args.fs, args.nperseg)
    # 計算 tap 相對於 background 的 dB 差值，可以觀察哪些頻率敲擊時特別明顯，而且比背景高很多
    diff_db = 10 * np.log10((psd_tap + eps) / (psd_bg + eps))
    
    # -- 2. 扣除寬頻趨勢、找局部凸起 --
    trend_db, local_db = detrend_db(diff_db, freqs, args.trend_window_hz)

    df_hz = freqs[1] - freqs[0]
    # width=0 代表不篩選，但 find_peaks 仍會計算每個峰的寬度
    peak_idx, props = find_peaks(local_db, 
                                 height=args.local_peak_db,              # 比局部趨勢至少高 3 dB - 預設值
                                 distance=5,                             # 兩個峰至少相隔 5 個頻率點（約 610 Hz）- 預設值，避免同一個峰被重複算成好幾個。
                                 prominence=args.peak_prominence,        # 峰要比左右兩側的谷底至少高 2 dB - 預設值，排除「站在小坡上」的假峰。
                                 width=args.peak_min_width_hz / df_hz)   # 目前設 0，只記錄寬度、不篩選
    widths_hz = props["widths"] * df_hz      # 在半個 prominence 高度量測的峰寬（Hz）

    # -- 3. 再過濾一次 --
    # 頻率下限取 --min-freq-hz 與「趨勢視窗的一半」的較大者：
    # 中位數濾波要往左右各看半個視窗，低於這個頻率時左側資料不足，趨勢估計不可靠。
    # 預設 20 kHz 視窗 → 下限 10 kHz，也剛好是共振格子（10k 的倍數）的起點。
    min_freq = max(args.min_freq_hz, args.trend_window_hz / 2)
    keep = (diff_db[peak_idx] >= args.peak_db_threshold) & (freqs[peak_idx] >= min_freq)
    peak_idx, widths_hz = peak_idx[keep], widths_hz[keep]

    result_df = pd.DataFrame({
        "frequency_Hz": freqs[peak_idx],
        "tap_vs_bg_dB": diff_db[peak_idx],       # 原始 dB 差（下游函式還會用到，欄位名稱不變）
        "local_peak_dB": local_db[peak_idx],     # 比局部趨勢高多少
        "peak_width_Hz": widths_hz,              # 峰寬（Hz），目前只記錄、不篩選
    }).sort_values("local_peak_dB", ascending=False)
    
    result_df["validation_0713_freq_Hz"] = np.nan
    result_df["validation_0713_tap_vs_bg_dB"] = np.nan
    result_df["validation_0713_local_dB"] = np.nan
    result_df["validation_0713_pass"] = False

    # 先建立一個空表, 如果後面 0713 成功算出頻譜，就把它填進去；如果沒有，就維持空表
    validation_spectrum_df = pd.DataFrame()
    
    # 確認：有驗證資料 + 有驗證 tap + 有驗證 background + 前三天搜尋也真的找到峰，四個都成立才做驗證。
    if val_tap_records and val_bg_records and not result_df.empty:
        # 先把 0713 的 tap/background 平均 PSD 算出來
        val_freqs, val_psd_tap = averaged_psd_balanced(val_tap_records, args.channel, args.fs, args.nperseg)
        _, val_psd_bg = averaged_psd_balanced(val_bg_records, args.channel, args.fs, args.nperseg)
        
        # 再算出 0713 自己的 tap_vs_bg_dB 曲線
        val_diff_db = 10 * np.log10((val_psd_tap + eps) / (val_psd_bg + eps))
        _, val_local_db = detrend_db(val_diff_db, val_freqs, args.trend_window_hz)

        # 頻率軸必須相同，才能直接用索引對照
        if len(val_freqs) != len(freqs) or not np.allclose(val_freqs, freqs):
            raise SystemExit("搜尋與驗證的頻率軸不一致，請確認兩邊的 fs 與 nperseg 相同")

        # 在 ±validation_tol_hz 範圍內，找出局部凸起最明顯的頻率點；
        # 兩個驗證數值都從「同一個點」讀取，避免局部凸起和原始 dB 差分別取自不同頻率
        for row_idx, f0 in result_df["frequency_Hz"].items():
            lo = np.searchsorted(val_freqs, f0 - args.validation_tol_hz, side="left")
            hi = np.searchsorted(val_freqs, f0 + args.validation_tol_hz, side="right")
            if lo >= hi:
                continue
            j = lo + int(np.argmax(val_local_db[lo:hi]))
            v_diff, v_local = float(val_diff_db[j]), float(val_local_db[j])
            result_df.at[row_idx, "validation_0713_freq_Hz"] = float(val_freqs[j])   # 0713 對應到的頻率
            result_df.at[row_idx, "validation_0713_tap_vs_bg_dB"] = v_diff
            result_df.at[row_idx, "validation_0713_local_dB"] = v_local
            result_df.at[row_idx, "validation_0713_pass"] = (
                v_local >= args.local_peak_db and v_diff >= args.peak_db_threshold)
        result_df["validation_0713_pass"] = result_df["validation_0713_pass"].astype(bool)

        validation_spectrum_df = pd.DataFrame({
            "frequency_Hz": val_freqs,
            "validation_0713_tap_PSD": val_psd_tap,
            "validation_0713_background_PSD": val_psd_bg,
            "validation_0713_tap_vs_bg_dB": val_diff_db,
            "validation_0713_local_dB": val_local_db,
        })

    spectrum_df = pd.DataFrame({
        "frequency_Hz": freqs,
        "tap_PSD": psd_tap,
        "background_PSD": psd_bg,
        "tap_vs_bg_dB": diff_db,
        "tap_vs_bg_trend_dB": trend_db,
        "tap_vs_bg_local_dB": local_db,
    })
    return result_df, spectrum_df, validation_spectrum_df


# ============================================================
# 分位置投票：每個 10 kHz 格子在幾個位置都跨天驗證通過
# ============================================================
def vote_grid_cells(per_pos_df):
    """
    把驗證通過的峰放進 10 kHz 格子，並統計每格在幾個位置都有出現。

    這一步發生在找峰「之後」：峰是在細解析度（約 122 Hz）的整條頻譜上找到的，
    這裡只是依頻率判斷每個峰落在哪一格（左閉右開 [f_low, f_high)）。
    格子起點為 10k 的倍數、彼此不重疊，都屬於特徵工程的 discovery 網格。
    """
    # 只留下驗證通過的峰
    valid = per_pos_df[per_pos_df["validation_0713_pass"]]
    # 算出第一格的起點
    start = -(-GRID_F_MIN // RES_GRID_STRIDE) * RES_GRID_STRIDE
    rows = []
    
    # 逐格統計：從 10000 開始，每次加 10000，產生 10～20k、20～30k、……、490～500k，總共 49 格
    for lo in range(start, GRID_F_MAX - GRID_WIDTH + 1, RES_GRID_STRIDE):
        hi = lo + GRID_WIDTH
        # sub 是落在這一格裡的峰（左閉右開）。hit 是這些峰來自哪些位置，unique() 會去除重複，所以同一個位置在這格裡有好幾個峰，也只算一次。
        sub = valid[(valid["frequency_Hz"] >= lo) & (valid["frequency_Hz"] < hi)]
        hit = sorted(sub["position"].unique())
        
        # 整理成一列
        rows.append({
            "cell": f"{lo // 1000:03d}_{hi // 1000:03d}k",
            "f_low": lo,
            "f_high": hi,
            "n_positions": len(hit),          # 幾個位置有通過驗證的峰
            "n_peaks": len(sub),                    # 這格裡驗證通過的峰總數（所有位置合計）
            "positions": ",".join(hit),
            "max_local_peak_dB": float(sub["local_peak_dB"].max()) if len(sub) else np.nan,    # 這格裡最強的局部凸起
        })
    return pd.DataFrame(rows)


# ============================================================
# 複製一份參數，只把聲道換掉（分聲道分析時使用）
# ============================================================
def args_for_channel(args, channel):
    """複製一份參數，只把聲道換掉，讓 find_validated_peaks() 分析指定聲道。"""
    return argparse.Namespace(**{**vars(args), "channel": channel})


# ============================================================
# 把各聲道的投票表並排合併成一張表
# ============================================================
def combine_channel_votes(votes_by_channel: dict) -> pd.DataFrame:
    """把各聲道的格子投票結果並排合併成一張表。
    每個聲道的欄位名稱會加上聲道後綴，例如 n_positions_ai0、n_positions_ai1。"""
    combined = None
    for ch, v in votes_by_channel.items():
        # 挑出需要的欄位，並在名稱後面加上聲道，避免兩個聲道的欄位撞名
        v = v[["cell", "f_low", "f_high", "n_positions", "positions", "n_peaks", "max_local_peak_dB"]].rename(columns={
            "n_positions": f"n_positions_{ch}",
            "n_peaks": f"n_peaks_{ch}",
            "positions": f"positions_{ch}",
            "max_local_peak_dB": f"max_local_peak_dB_{ch}",
        })
        # 第一個聲道直接當作起點；之後的聲道用格子（cell、f_low、f_high）對齊合併
        combined = v if combined is None else combined.merge(v, on=["cell", "f_low", "f_high"])
    return combined


# ============================================================
# 產生共振頻帶提案（提案 → 人工確認 → 核准）
# ============================================================
def build_resonance_proposal(vote_df, overall_by_channel, channels,
                             min_positions, min_channels, run_info):
    """根據各聲道的分位置投票和整體分析，產生共振頻帶提案。
    對每個 10 kHz 格子、每個聲道分別判斷：
    - 強證據：該聲道至少 min_positions 個位置跨天驗證通過
    - 弱證據：該聲道至少 1 個位置通過，且該聲道整體平均分析在同一格也通過
    提案類型：
    - core：至少 min_channels 個聲道有強證據
    - secondary：未達 core，但至少一個聲道有強或弱證據
    """
    # 把數字取到兩位小數；如果是 nan 就回傳 None
    def to_float_or_none(x):
        return None if pd.isna(x) else round(float(x), 2)

    # 先把每個聲道整體分析裡驗證通過的峰挑出來備用（判斷弱證據時要用）
    overall_valid = {
        ch: df[df["validation_0713_pass"]] for ch, df in overall_by_channel.items()
    }

    bands = []

    # 逐格、逐聲道判斷證據強弱，對每一格，兩個聲道分別判斷：
    # 強證據：這個聲道有至少 2 個位置在這格通過驗證。
    # 弱證據：只有 1 個位置通過，但整體分析在這格也通過，等於有兩種分析方式都支持。
    # 兩者都不符合，這個聲道就不算支持這一格。
    for row in vote_df.to_dict(orient="records"):    # to_dict(orient="records") 把投票表轉成「每格一個字典」，方便用 row["f_low"] 這種方式讀取。
        strong, weak = [], []
        for ch in channels:
            n_pos = int(row[f"n_positions_{ch}"])
            ov = overall_valid[ch]
            in_cell = ((ov["frequency_Hz"] >= row["f_low"]) & (ov["frequency_Hz"] < row["f_high"])).any()
            if n_pos >= min_positions:
                strong.append(ch)
            elif n_pos >= 1 and in_cell:
                weak.append(ch)
                
        # 決定類型
        # 強證據的聲道數夠多（預設兩個都要）就是 core；有任何聲道支持（不管強弱）就是 secondary；完全沒有支持就跳過，不放進提案。
        if len(strong) >= min_channels:
            band_type = "core"        # 強證據的聲道數 ≥ min_channels（預設 2）
        elif strong or weak:
            band_type = "secondary"   # 有任何聲道支持（強或弱）
        else:
            continue
        prefix = "res_core" if band_type == "core" else "res_sec"
        
        # 檢查是否上次被排除過
        previously_excluded = (int(row["f_low"]), int(row["f_high"])) in EXCLUDED_RANGES
        
        # 和 offband 有重疊的格子，固定不納入（用「有重疊」判斷，因為 315～325k 沒有對齊 10k 格子）
        fixed_excluded = any(
            min(row["f_high"], hi) - max(row["f_low"], lo) > 0 for lo, hi in FIXED_EXCLUDED_RANGES)
                
        # 整理成提案的一筆
        bands.append({
            "include": not (previously_excluded or fixed_excluded),          # 上次核准時排除的，預設維持排除
            "previously_excluded": previously_excluded,                      # 標示這格是上次核准時排除的
            "fixed_excluded": fixed_excluded,                                # 與 offband 重疊，固定不納入
            "name": f"{prefix}_{row['cell']}",
            "f_low": int(row["f_low"]),
            "f_high": int(row["f_high"]),
            "band_type": band_type,
            "strong_channels": strong,
            "weak_channels": weak,
            # 每個聲道的詳細數據，給你核准前檢查用
            "evidence": {
                ch: {
                    "n_positions": int(row[f"n_positions_{ch}"]),
                    "n_peaks": int(row[f"n_peaks_{ch}"]),
                    "positions": row[f"positions_{ch}"],
                    "max_local_peak_dB": to_float_or_none(row[f"max_local_peak_dB_{ch}"]),
                }
                for ch in channels
            },
            "reason": (f"強證據聲道：{'、'.join(strong) or '無'}；"
                       f"弱證據聲道：{'、'.join(weak) or '無'}"
                       f"（core 需至少 {min_channels} 個聲道有強證據）"),
        })

    # 用「範圍」比對，才能區分「新增 / 移除」和「同一格只是類型改變」
    def cell_name(lo, hi):
        return f"{int(lo) // 1000:03d}_{int(hi) // 1000:03d}k"

    # 和正式版比對差異
    current_types = {(int(lo), int(hi)): t for _, lo, hi, t in ALL_BANDS}
    proposed_types = {(b["f_low"], b["f_high"]): b["band_type"] for b in bands if b["include"]}

    diff = {
        # 提案有、正式版沒有的範圍
        "added": [cell_name(*k) for k in sorted(proposed_types) if k not in current_types],
        # 正式版有、提案沒有的範圍
        "removed": [cell_name(*k) for k in sorted(current_types) if k not in proposed_types],
        # 兩邊都有、但類型不同的範圍
        "type_changed": [
            f"{cell_name(*k)}：{current_types[k]} → {proposed_types[k]}"
            for k in sorted(proposed_types)
            if k in current_types and proposed_types[k] != current_types[k]
        ],
    }

    # 組成完整的提案
    return {
        "schema_name": "proposed_resonance_bands",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "run_info": run_info,           # 這次分析用的參數
        "diff_vs_approved": diff,       # 和正式版的差異
        "resonance_bands": bands,       # 每一格的提案內容
    }


# ============================================================
# 核准提案：把人工確認過的提案寫成正式版，並留下備份與紀錄
# ============================================================
def approve_proposal(proposal_path: Path, approved_path: Path, note: str,
                     allow_empty: bool = False) -> None:
    """
    人工確認後執行：把提案寫成正式版，並新增一筆核准紀錄。
    - 提案會完整備份到正式版旁邊的 approval_history/，避免之後執行分析時被覆蓋
    - 正式版與核准紀錄都指向這份備份，確保日後能追溯當時核准的依據
    """
    # ---------- 1. 檢查輸入 ----------
    # 核准理由必填，會寫進正式版和核准紀錄
    if not note.strip():
        raise SystemExit("核准時請用 --approve-note 寫下理由，這會留在紀錄裡")
        
    # 讀取提案，並確認格式正確（避免傳錯檔案，例如把正式版當成提案）
    proposal_path = Path(proposal_path)
    proposal_text = proposal_path.read_text(encoding="utf-8")
    proposal = json.loads(proposal_text)
    if proposal.get("schema_name") != "proposed_resonance_bands":
        raise SystemExit(f"{proposal_path} 不是 proposed_resonance_bands 格式")

    # ---------- 2. 整理要核准的頻帶 ----------
    # 只保留 include 為 true 的頻帶（人工可在提案檔裡改成 false 排除）
    bands = [b for b in proposal["resonance_bands"] if b.get("include", True)]
    
    # 自動對齊搜尋網格；人工改過範圍的頻帶，對齊前後的差異會印出來讓人確認
    snapped = []
    for b in bands:
        new = snap_band_to_grid(b)
        if len(new) != 1 or new[0]["f_low"] != b["f_low"] or new[0]["f_high"] != b["f_high"]:
            print(f"  對齊：{b['name']} {b['f_low']}~{b['f_high']} Hz → {[n['name'] for n in new]}")
        snapped.extend(new)
    
    # 不同頻帶對齊到同一格時只留一筆，core 優先
    by_range = {}
    for b in snapped:
        key = (b["f_low"], b["f_high"])
        if key not in by_range or (b["band_type"] == "core" and by_range[key]["band_type"] != "core"):
            by_range[key] = b
    bands = sorted(by_range.values(), key=lambda b: b["f_low"])
    
    # 名稱一律依「類型 + 範圍」重新產生，避免人工改了 band_type 但名稱沒跟著改
    for b in bands:
        prefix = "res_core" if b["band_type"] == "core" else "res_sec"
        b["name"] = f"{prefix}_{b['f_low'] // 1000:03d}_{b['f_high'] // 1000:03d}k"
    
    # 防呆：沒有任何要納入的頻帶時，核准會清空正式版，必須明確同意才繼續
    if not bands and not allow_empty:
        raise SystemExit("提案中沒有任何 include = true 的頻帶，核准後正式版會被清空。\n"
                         "若確定要這麼做，請加上 --allow-empty-approval")
    
    # ---------- 3. 記下核准前的正式版內容（寫進紀錄，方便比對前後差異） ----------
    previous, old = [], None
    if approved_path.exists():
        old = json.loads(approved_path.read_text(encoding="utf-8"))
        previous = [b["name"] for b in old["resonance_bands"]]
        
    # 排除紀錄要延續：上次排除、但這次提案沒有再出現的格子，仍保留在排除清單中
    proposal_ranges = {(b["f_low"], b["f_high"]) for b in proposal["resonance_bands"]}
    carried_excluded = [
        b for b in (old or {}).get("excluded_bands", [])
        if (b["f_low"], b["f_high"]) not in proposal_ranges
    ]

    # ---------- 4. 備份提案 ----------
    # 分析的輸出資料夾名稱是固定的，每次執行都會覆蓋 proposed_resonance_bands.json。
    # 所以核准時把提案「原封不動」複製一份到 approval_history/，檔名加上核准時間，
    # 正式版和紀錄都指向這份備份，日後才查得到當時核准的完整依據（含各聲道、各位置的證據）。
    now = datetime.now().astimezone()
    approved_at = now.isoformat()
    history_dir = approved_path.parent / "approval_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    archived_proposal = history_dir / f"proposed_resonance_bands_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    archived_proposal.write_text(proposal_text, encoding="utf-8")
    
    # ---------- 5. 寫入正式版 ----------
    payload = {
        "schema_name": "approved_resonance_bands",
        "schema_version": 1,
        "approved_at": approved_at,
        "approved_from": str(archived_proposal),          # 指向備份，不會被之後的分析覆蓋
        "original_proposal_path": str(proposal_path),     # 原始提案的位置（僅供參考，內容可能已被覆蓋）
        "note": note,
        "run_info": proposal.get("run_info", {}),         # 產生提案時的參數（門檻、聲道、位置等）
        "resonance_bands": [
            {"name": b["name"], "f_low": b["f_low"], "f_high": b["f_high"],
             "band_type": b["band_type"], "reason": b.get("reason", "")}
            for b in bands
        ],
        "excluded_bands": [
            {"name": b["name"], "f_low": b["f_low"], "f_high": b["f_high"], "band_type": b["band_type"]}
            for b in proposal["resonance_bands"] if not b.get("include", True)
        ] + carried_excluded,
    }
    approved_path.parent.mkdir(parents=True, exist_ok=True)
    approved_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 6. 新增一筆核准紀錄（每次核准一行，只會往後加，不會改到舊紀錄） ----------
    log_path = approved_path.parent / "resonance_bands_approval_log.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "approved_at": approved_at,
            "proposal": str(archived_proposal),           # 指向備份
            "note": note,
            "previous": previous,                         # 核准前的正式版
            "approved": [b["name"] for b in bands],       # 核准後的正式版
        }, ensure_ascii=False) + "\n")

    # ---------- 7. 印出結果 ----------
    print(f"✅ 已核准，正式版：{approved_path}")
    print(f"   原本：{previous}")
    print(f"   現在：{[b['name'] for b in bands]}")
    print(f"   提案備份：{archived_proposal}")
    print(f"   紀錄：{log_path}")
    

# ============================================================
# 主流程
# ============================================================
'''
開始
 ├─ 核准模式？ → 是：核准提案後結束
 └─ 分析模式：
     0. 載入正式版、整理參數、檢查批次
     1. 收集前三天的片段（搜尋用）
     2~3. 收集 0713 的片段（驗證用），每個聲道做整體分析
     4. 建立輸出資料夾
     5. 輸出峰值表和頻譜 CSV
     5b. 分位置分析 → 投票 → 產生提案
     6. 摘要表、落差稽核、畫圖
     7. 在終端機印出摘要
'''
def main() -> None:
    # -- 判斷要做哪件事 --
    # 前面設定的參數讀進來，之後像 args.root、args.channels、args.nperseg 都是從這裡拿
    args = parse_args()
    # 核准模式：只把提案寫成正式版，不做分析
    if args.approve_proposal is not None:
        approve_proposal(args.approve_proposal, args.approved_json, args.approve_note,
                         args.allow_empty_approval)
        return
    
    # 分析模式：先載入目前的正式版，作為落差稽核和畫圖的基準
    load_approved_bands(args.approved_json)
        
    # -- 整理參數、檢查批次 --
    # 位置篩選條件
    position_filter = set(args.position_filter) if args.position_filter else None
    # 決定：哪些批次拿來「找共振候選峰」,哪些批次拿來「做驗證」
    # 如果你手動傳 --batch-filter，就覆蓋預設搜尋批次
    search_batches = set(args.batch_filter) if args.batch_filter else DEFAULT_SEARCH_BATCHES
    validation_batches = DEFAULT_VALIDATION_BATCHES
    # 檢查:如果搜尋和驗證用到同一天，跨天驗證就沒有意義了，直接停止。
    overlap = search_batches & validation_batches
    if overlap:
        raise SystemExit(f"搜尋批次和驗證批次重疊：{sorted(overlap)}，跨天驗證會失去意義")
    

    # ============================================================
    # 1. 收集前三天資料，用來找共振候選峰
    # ============================================================
    # 收集指定批次中的所有 tap/background 片段
    all_records = collect_records(
        root=args.root,
        batch_filter=search_batches,
        position_filter=position_filter,
        background_guard_sec=args.background_guard_sec,
        event_overlap_sec=args.event_overlap_sec,
    )
    
    # 如果一筆資料都沒收到，就直接停止
    if not all_records:
        raise SystemExit("找不到可用的搜尋批次資料")
        
    # 安全檢查：確認每個 TDMS 檔只被一個片段使用
    check_one_file_per_slice(all_records, "搜尋批次")
        
    # 把總片段再分成兩類：tap_records：敲擊片段 , bg_records：背景片段
    tap_records = [r for r in all_records if r.label == "tap"]
    bg_records = [r for r in all_records if r.label == "background"]

    # 沒有 tap 片段 -> 沒辦法找共振
    if not tap_records:
        raise SystemExit("沒有找到敲擊片段，請檢查事件標註或篩選條件")
    # 沒有 background 片段 -> 沒辦法做 tap vs background 比較
    if not bg_records:
        raise SystemExit("沒有找到背景片段，請放寬條件或縮小 background_guard_sec")

    print(f"敲擊片段數量   : {len(tap_records)}")
    print(f"背景片段數量   : {len(bg_records)}")
    print(f"使用 session 數: {len({r.session_dir for r in all_records})}")

    
    # ============================================================
    # 2~3. 收集 0713 驗證資料，每個聲道做整體分析
    # ============================================================
    validation_records = collect_records(
        root=args.root,
        batch_filter=validation_batches,
        position_filter=position_filter,
        background_guard_sec=args.background_guard_sec,
        event_overlap_sec=args.event_overlap_sec,
    )
    
    # 安全檢查：確認每個 TDMS 檔只被一個片段使用
    check_one_file_per_slice(validation_records, "驗證批次")
    val_tap_records = [r for r in validation_records if r.label == "tap"]
    val_bg_records = [r for r in validation_records if r.label == "background"]
    
    # 檢查壓力：不同壓力的共振頻率可能不同，混在一起會被平均掉
    pressures = sorted({r.pressure_bar for r in all_records + validation_records})
    if len(pressures) > 1:
        print(f"  ⚠️ 資料混有多種壓力：{pressures} bar，結果會被平均在一起；"
              "如需分開分析，請先依壓力整理資料")

    # 整理要分析的聲道
    channels = list(dict.fromkeys(args.channels))       # 去除重複、保留順序
    min_channels = args.min_channels or len(channels)   # min_channels 沒指定時，預設是「全部聲道都要有強證據」
    if not 1 <= min_channels <= len(channels):
        raise SystemExit(f"--min-channels 必須介於 1 和聲道數 {len(channels)} 之間")

    print("\n=== 整體分析 ===")
    overall_by_channel = {}      # 聲道 → 峰值表(給提案、摘要表、落差稽核用)
    overall_outputs = {}         # 聲道 → (峰值表, 搜尋頻譜, 驗證頻譜)
    # 整體分析：每個聲道把所有位置的片段放在一起（等權重平均），找峰並跨天驗證
    for ch in channels:
        res, spec, vspec = find_validated_peaks(
            tap_records, bg_records, val_tap_records, val_bg_records, args_for_channel(args, ch))
        overall_by_channel[ch] = res
        overall_outputs[ch] = (res, spec, vspec)
        print(f"  {ch}: 候選峰 {len(res)}，驗證通過 {int(res['validation_0713_pass'].sum())}")

    if all(vspec.empty for _, _, vspec in overall_outputs.values()):
        print("0713 驗證資料不足，這次先只輸出前三天的共振候選峰")
    
    
    # ============================================================
    # 4. 整理輸出資料夾結構
    # ============================================================
    # 把用到的片段清單整理成一張追查表
    summary_df = build_summary_df(all_records + validation_records)
    
    # 決定輸出資料夾結構
    output_anchor = args.output_csv
    run_name = output_anchor.stem
    output_root = output_anchor.parent / run_name
    
    # 建立四個子資料夾
    tables_dir = output_root / "tables"       # 表格
    spectra_dir = output_root / "spectra"     # 完整頻譜
    figures_dir = output_root / "figures"     # 圖檔
    configs_dir = output_root / "configs"     # 輸出共振 json 檔

    # 如果這些資料夾不存在，就建立起來
    for d in [output_root, tables_dir, spectra_dir, figures_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 表格類
    slice_summary_csv = tables_dir / f"{run_name}_slice_summary.csv"
    band_summary_csv = tables_dir / f"{run_name}_band_summary.csv"

    # 圖檔類
    band_summary_png = figures_dir / f"{run_name}_band_summary.png"
    coverage_png = figures_dir / f"{run_name}_data_coverage.png"

    # ============================================================
    # 5. 輸出 CSV（每個聲道各一份）
    # ============================================================
    summary_df.to_csv(slice_summary_csv, index=False, encoding="utf-8-sig")
    
    # 輸出各聲道的峰值表
    for ch, (res, spec, vspec) in overall_outputs.items():
        res.to_csv(tables_dir / f"{run_name}_peaks_{ch}.csv", index=False, encoding="utf-8-sig")
        spec.to_csv(spectra_dir / f"{run_name}_full_spectrum_{ch}.csv", index=False, encoding="utf-8-sig")
        if not vspec.empty:
            vspec.to_csv(spectra_dir / f"{run_name}_validation_0713_spectrum_{ch}.csv",
                         index=False, encoding="utf-8-sig")

    # ============================================================
    # 5b. 分位置找峰，再統計每個 10 kHz 格子在幾個位置都有出現
    #     （只用於「建立頻帶設定」，實際檢測時不需要位置資訊）
    # ============================================================
    search_pos = {r.position for r in all_records}
    val_pos = {r.position for r in validation_records}
    # 找出能做跨天驗證的位置
    positions = sorted(search_pos & val_pos)     # & 是交集：兩邊都有的位置才能分析
    skipped = sorted(search_pos ^ val_pos)       # ^ 是對稱差：只出現在其中一邊的位置，會印出來告訴你被略過了
    if skipped:
        print(f"  以下位置只出現在搜尋或驗證其中一邊，無法跨天驗證，略過：{skipped}")
    
    print("\n=== 分位置分析 ===")
    per_pos_frames = []
    analyzed_positions = set()        # 實際有執行分析的位置（即使沒找到峰也算）
    
    # 每個聲道、每個位置分別找峰
    for ch in channels:
        args_ch = args_for_channel(args, ch)
        for pos in positions:
            subsets = [
                [r for r in recs if r.position == pos]
                for recs in (tap_records, bg_records, val_tap_records, val_bg_records)
            ]
            if not all(subsets):
                print(f"  {ch} / {pos}: 搜尋或驗證缺少 tap/background，略過")
                continue
            pos_df, _, _ = find_validated_peaks(*subsets, args_ch)
            analyzed_positions.add(pos)
            pos_df["channel"] = ch
            pos_df["position"] = pos
            per_pos_frames.append(pos_df)
            print(f"  {ch} / {pos}: 候選峰 {len(pos_df)}，驗證通過 {int(pos_df['validation_0713_pass'].sum())}")
    
    if per_pos_frames:
        per_pos_df = pd.concat(per_pos_frames, ignore_index=True)
        per_pos_df.to_csv(tables_dir / f"{run_name}_per_position_peaks.csv",
                          index=False, encoding="utf-8-sig")
        # 投票：每個聲道各自統計「每個 10 kHz 格子在幾個位置通過」，再把兩個聲道的投票表左右並排合併。
        votes = {
            ch: vote_grid_cells(per_pos_df[per_pos_df["channel"] == ch])
            for ch in channels
        }
        vote_df = combine_channel_votes(votes)
        vote_df.to_csv(tables_dir / f"{run_name}_position_vote.csv",
                       index=False, encoding="utf-8-sig")
    
        n_pos_analyzed = len(analyzed_positions)
        # 產生提案並印出:先檢查位置數夠不夠產生 core，接著依規則產生提案，並把這次的所有參數記在 run_info 裡，一起寫成 proposed_resonance_bands.json
        if args.min_positions > n_pos_analyzed:
            print(f"  ⚠️ --min-positions={args.min_positions} 大於實際分析的位置數（{n_pos_analyzed} 個），"
                  "本次不可能產生 core；如需比較，請調低 --min-positions")
            
        proposal = build_resonance_proposal(
            vote_df, overall_by_channel, channels, args.min_positions, min_channels,
            run_info={
                "output_root": str(output_root),
                "search_batches": sorted(search_batches),
                "validation_batches": sorted(validation_batches),
                "positions_analyzed": sorted(analyzed_positions),
                "channels": channels,
                "min_channels": min_channels,
                "nperseg": args.nperseg,
                "trend_window_hz": args.trend_window_hz,
                "local_peak_db": args.local_peak_db,
                "peak_prominence": args.peak_prominence,
                "peak_db_threshold": args.peak_db_threshold,
                "min_freq_hz": args.min_freq_hz,
                "effective_min_freq_hz": max(args.min_freq_hz, args.trend_window_hz / 2),
                "peak_min_width_hz": args.peak_min_width_hz,
                "min_positions": args.min_positions,
                "validation_tol_hz": args.validation_tol_hz,
            },
        )
        proposal_json = configs_dir / "proposed_resonance_bands.json"
        proposal_json.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    
        print(f"\n=== 共振頻帶提案（core 需 {min_channels} 個聲道、各至少 {args.min_positions} 個位置）===")
        if proposal["resonance_bands"]:
            for b in proposal["resonance_bands"]:
                tag = ("（與 offband 重疊，固定不納入）" if b["fixed_excluded"]
                       else "（上次排除，預設不納入）" if b["previously_excluded"] else "")
                print(f"  {b['name']:<20} 強：{b['strong_channels']}  弱：{b['weak_channels']}{tag}")
        else:
            print("  （沒有符合條件的格子）")
    
        diff = proposal["diff_vs_approved"]
        print(f"\n共振頻帶提案：{proposal_json}")
        if not any(diff.values()):                                  
            print("  提案和目前的正式版相同，不需要核准。")
        else:
            print(f"  相較正式版新增：{diff['added']}")
            print(f"  相較正式版移除：{diff['removed']}")
            print(f"  類型改變：{diff['type_changed']}")            
            print("  確認後執行：")
            print(f'  %runfile {Path(__file__).resolve()} --args '
                  f'"--approve-proposal {proposal_json} --approve-note \'寫下理由\'"')
    else:
        print("\n⚠️ 沒有任何位置同時具備搜尋與驗證資料，本次不產生共振頻帶提案。")
        
    # ============================================================
    # 6. 摘要表、落差稽核、畫圖
    # ============================================================
    # 摘要表：每個聲道各做一份，再合併（加上 channel 欄位）
    summaries = [
        build_band_summary_table(res).assign(channel=ch)
        for ch, res in overall_by_channel.items()
    ]
    summaries = [s for s in summaries if not s.empty]
    band_summary_df = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(["band_type", "band_start_Hz", "channel"])
        .reset_index(drop=True)
        if summaries else pd.DataFrame()
    )
    if not band_summary_df.empty:
        band_summary_df.to_csv(band_summary_csv, index=False, encoding="utf-8-sig")
        save_band_summary_png(band_summary_df=band_summary_df, output_png=band_summary_png)
    
    # 頻帶落差稽核：每個聲道各自比對正式版，再合併成一份
    # 沒有正式版時，直接略過稽核
    if ALL_BANDS:
        parts = {"uncovered_peaks": [], "boundary_close_peaks": []}
        for ch, res in overall_by_channel.items():
            d = audit_band_drift(res, ALL_BANDS, boundary_margin_hz=args.band_boundary_margin_hz, excluded_ranges=EXCLUDED_RANGES | set(FIXED_EXCLUDED_RANGES))
            for key in parts:
                if not d[key].empty:
                    parts[key].append(d[key].assign(channel=ch))   # 加上 channel 欄位，標記來自哪個聲道

        band_drift = {
            "uncovered_peaks": (
                pd.concat(parts["uncovered_peaks"], ignore_index=True)
                .sort_values("local_peak_dB", ascending=False)
                if parts["uncovered_peaks"] else pd.DataFrame()
            ),
            "boundary_close_peaks": (
                pd.concat(parts["boundary_close_peaks"], ignore_index=True)
                .sort_values("dist_to_edge_Hz")
                if parts["boundary_close_peaks"] else pd.DataFrame()
            ),
        }
        
        # 接收 audit_band_drift 算出來的結果，負責把結果印出來
        has_drift = report_band_drift(band_drift)
    else:
        print("\n（尚無正式版共振頻帶，略過落差稽核；請先查看提案並核准）")
        band_drift, has_drift = None, False
    
    band_drift_json = None
    
    # 如果有落差,就把細節存成 JSON
    if has_drift:
        # 組出輸出路徑
        band_drift_json = configs_dir / f"{run_name}_band_drift_report.json"
        
        def to_records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
            """空表直接回傳空清單，避免選欄位時出錯。"""
            return [] if df.empty else df[cols].to_dict(orient="records")
        
        # 準備寫進 JSON 的內容, 選出幾個真正需要的欄位作為「人工判讀落差」最關鍵的欄位
        drift_payload = {
            "uncovered_peaks": to_records(
                band_drift["uncovered_peaks"],
                ["channel", "frequency_Hz", "local_peak_dB", "tap_vs_bg_dB", "validation_0713_local_dB"],
            ),
            "boundary_close_peaks": to_records(
                band_drift["boundary_close_peaks"],
                ["channel", "frequency_Hz", "band_name", "dist_to_edge_Hz", "local_peak_dB"],
            ),
        }
        
        # 把整個 drift_payload 用 json.dumps() 轉成文字寫進檔案
        band_drift_json.write_text( 
            json.dumps(drift_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    
    # 每個聲道各畫一份疊圖和峰值分組圖
    for ch, (res, spec, vspec) in overall_outputs.items():
        if not vspec.empty:
            plot_search_vs_validation_overlay(
                search_spectrum_df=spec,
                validation_spectrum_df=vspec,
                output_png=figures_dir / f"{run_name}_search_vs_validation_overlay_{ch}.png",
                channel=ch,
                local_peak_db=args.local_peak_db,
                min_freq_hz=max(args.min_freq_hz, args.trend_window_hz / 2),
            )
        plot_peak_grouped_by_band(
            peaks_df=res,
            output_png=figures_dir / f"{run_name}_peak_grouped_by_band_{ch}.png",
            channel=ch,
        )
    
    # 資料覆蓋圖和聲道無關，只畫一張
    plot_data_coverage(slice_summary_df=summary_df, output_png=coverage_png)
        

    # ============================================================
    # 7. 終端列印摘要
    # ============================================================
    print()
    print(f"輸出資料夾          : {output_root}")
    print(f"片段清單            : {slice_summary_csv}")
    print(f"各聲道峰值表        : {tables_dir}（*_peaks_<聲道>.csv）")
    print(f"各聲道頻譜          : {spectra_dir}（檔名結尾為聲道名稱）")
    print(f"各聲道圖            : {figures_dir}（檔名結尾為聲道名稱）")

    if not band_summary_df.empty:
        print(f"頻帶摘要表          : {band_summary_csv}")
        print(f"頻帶摘要表 PNG      : {band_summary_png}")
    if band_drift_json is not None:
        print(f"頻帶落差報告 JSON   : {band_drift_json}")

    print(f"正式版共振頻帶      : {args.approved_json}")
    if per_pos_frames:
        print(f"共振頻帶提案        : {configs_dir / 'proposed_resonance_bands.json'}")

    for ch, res in overall_by_channel.items():
        if not res.empty:
            print(f"\n{ch} 前 20 個共振候選頻率：")
            print(res.head(20).to_string(index=False))
            

        
if __name__ == "__main__":
    configure_matplotlib_fonts()
    main()