#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出管子自身固定的共振候選頻帶。
1.先用「敲擊管子」資料去找哪些頻率最容易被激發
2.把這些高響應頻率整理成一段一段的候選頻帶
3.再依穩定性與集中程度，人工整理成 CORE_BANDS 跟 SECONDARY_BANDS

原始 TDMS -> 全頻段 PSD -> 找共振峰 -> 整理候選頻帶 -> 回灌特徵工程/模型驗證

分析邏輯：
1. 敲擊管子（tap_pipe / tab_pipe / tappipe）可視為一種寬頻激發。
   管子被敲擊後，在某些固定頻率會特別容易響應，這些位置就是管子本身可能的共振候選頻帶。

2. 你的 disturbance 資料不是單一長錄音檔，而是一段段 TDMS 小片段，
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
   - 用同一批 session 中、離敲擊事件夠遠的 leak0 片段當背景噪音

4. 最後分別計算：
   - 敲擊片段的平均 PSD
   - 背景片段的平均 PSD

   再用兩者的 dB 差異找出明顯凸起的峰值，
   作為共振候選頻帶，並再用 0713 資料做跨天驗證。

5. 這支程式找到的是「共振候選頻帶」，不是直接等同於「洩漏判別頻帶」；
   後續仍需搭配正常/洩漏資料，評估哪些頻帶最適合做洩漏特徵。
   
說明：
本程式找出的不是嚴格物理定義下的固有模態頻率，而是根據實際敲擊量測資料所觀察到的高響應頻帶，因此較適合稱為「共振候選頻帶」。

這些頻帶通常與管路本體的共振行為有關，但同時也會受到感測器位置、敲擊位置、阻尼、邊界條件與量測噪音影響，所以不一定與理論上的單一固有頻率完全相同。

這裡依賴 scipy 預設 detrend='constant' 做逐段去 DC

本程式的目的，是從真實量測資料中找出穩定且可重現的高響應頻帶，作為後續特徵工程、洩漏分析與模型驗證的候選依據。
"""

from __future__ import annotations
from pathlib import Path, PureWindowsPath
import argparse
from dataclasses import dataclass
from typing import Iterable
import os
import tempfile
import json

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


# ============================================================
# 參數預設
# ============================================================
DEFAULT_ROOT = Path("/Users/paul/Desktop/AE Data/disturbance")      # 干擾資料根目錄
DEFAULT_OUTPUT = Path("/Users/paul/Desktop/Final/resonance/pipe_resonance_bands.csv")    # 輸出檔的預設位置

# 你的資料夾命名不完全一致，所以把幾種常見拼法都納入(已有改成 tap_pipe)
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
# 建議頻帶：先透過初步分析，整理出的建議頻帶
#
# 這些頻帶不是事先指定來找峰的，而是先做完以下探索流程後，再根據結果人工整理出的「建議共振候選頻帶」：
# 1. 使用 2026-06-24、2026-07-02、2026-07-07 的敲擊資料，在全頻段上計算 tap_vs_bg_dB，找出候選峰值
# 2. 再用 2026-07-13 的敲擊資料做跨天驗證，檢查這些候選峰在不同日期下是否仍然明顯存在
# 3. 最後觀察「驗證通過峰」在頻率軸上的群聚情況，將相近的峰整理成幾段較穩定、較有代表性的建議頻帶
#
# 因此：
# - CORE_BANDS   = 跨天較穩定、峰值較密集的核心候選頻帶
# - SECONDARY_BANDS = 也有出現，但穩定性或集中程度略次的次要候選頻帶
#
# 對應的探索結果可參考輸出：
# - pipe_resonance_bands.csv
# - pipe_resonance_bands_full_spectrum.csv
# - pipe_resonance_bands_validation_0713_spectrum.csv
# - pipe_resonance_bands_peak_grouped_by_band.png
# - pipe_resonance_bands_search_vs_validation_overlay.png
# - pipe_resonance_bands_band_summary.csv
# ============================================================
CORE_BANDS = [
    ("core_01", 100000, 192000, "core"),
    # 原本 core_01~04(118500-164000)是在 CORE_BANDS/SECONDARY_BANDS 留空、
    # 用 prominence=2/4/6 三種門檻反覆測試後,發現 100k~192k Hz 這個範圍
    # 不管門檻怎麼拉高都連續存在、沒有真正斷層,所以合併成一段寬頻帶,
    # 不再勉強切成四段獨立頻帶。
]

SECONDARY_BANDS = [
    ("secondary_01", 60000, 100000, "secondary"),
    # 強度中等(27~34 dB),持續存在但比核心區弱一階

    ("secondary_02", 335000, 348000, "secondary"),
    # 只在 prominence=4 存活、prominence=6 消失,中等孤立峰,信心中等

    ("secondary_03", 405000, 415000, "secondary"),
    # 連 prominence=6 都存活,局部形狀最孤立,但絕對強度弱(約 19~20 dB),
    # 信心較低,待更多資料驗證
]

ALL_BANDS = CORE_BANDS + SECONDARY_BANDS


# ============================================================
# 用來存每一個 TDMS 小片段的整理後資訊
# ============================================================
@dataclass(frozen=True)
class SliceRecord:
    # 每一個 TDMS 小片段，不只是原始檔而已，還帶有：
    session_dir: Path 
    tdms_path: Path
    session_id: str             # 它來自哪個 session
    batch_name: str             # 哪一天批次
    disturbance_dir: str     
    disturbance_id: str         # 哪個 disturbance id
    position: str               # 哪個位置
    pressure_bar: float         # 哪個壓力
    dual_index: int
    t_start: float              # 時間範圍
    t_end: float
    label: str                  # "tap" 或 "background"


# ============================================================
# 讀取命令列參數
# ============================================================
def parse_args() -> argparse.Namespace:
    # 把你執行程式時可能輸入的參數讀進來，最後整理成一個 args 物件回傳。
    # -> argparse.Namespace：表示這個函式最後回傳的是 argparse 整理好的設定物件，可以用 args.xxx 的方式去取值。

    parser = argparse.ArgumentParser(
        # 如果你之後在終端機輸入：python xxx.py --help，這句 description 就會顯示出來。
        description="從 disturbance 的敲擊資料自動找出管子的共振頻段"
    )

    parser.add_argument(
        "--root",     # --root：指定資料根目錄
        type=Path,    # type=Path：把輸入的字串自動轉成 Path 物件，後面比較好處理路徑
        default=DEFAULT_ROOT,       # default=DEFAULT_ROOT：如果你沒有特別指定，就用程式前面設定好的預設路徑
        help="disturbance 根目錄"    # help=...：這個參數的說明文字，會在 --help 時顯示
    )

    parser.add_argument(
        "--channel",      # --channel：指定要分析哪個感測器聲道
        default="ai1",    # default="ai1"：如果沒指定，就用 ai1
        help="要分析的聲道名稱，例如 ai0 或 ai1"
    )

    parser.add_argument(
        "--fs",               # --fs：sampling rate（取樣率）
        type=float,           # type=float：把輸入值轉成浮點數
        default=1_000_000,    # default=1_000_000：預設是 1 MHz
        help="取樣率"
    )

    parser.add_argument(
        "--nperseg",          # --nperseg：Welch PSD 每一段切多長
        type=int,             # type=int：這個值要是整數
        default=8192,         # default=8192：預設每段 8192 點，這個值會影響頻率解析度與平滑程度
        help="Welch PSD 的 nperseg"
    )

    parser.add_argument(
        "--peak-db-threshold",    # --peak-db-threshold：找峰值時的最低門檻
        type=float,
        default=6.0,              # 例如設成 6 dB，就表示至少要高過背景 6 dB 才算候選峰，門檻高：峰比較少、比較嚴格
        help="峰值門檻（dB）"
    )
    
    parser.add_argument(
        "--band-boundary-margin-hz",      # 從外部控制「多接近邊界才算太接近」這個判斷門檻
        type=float,                       # 表示你在終端機輸入 --band-boundary-margin-hz 300 這種值時，會自動轉成浮點數
        default=500.0,                    # 如果你沒有特別指定，就用 500 Hz 當預設容許值——也就是「一個峰如果離它所在頻帶的上緣或下緣不到 500 Hz，就視為這個邊界可能需要微調」
        help="頻帶邊界比對容許值（Hz），峰值離邊界小於這個值，就視為頻帶可能該微調",
    )

    parser.add_argument(
        "--background-guard-sec",    # --background-guard-sec：當程式在挑 background 片段時，會避開敲擊事件前後一段時間，避免把敲擊殘留振動誤當背景
        type=float,
        default=0.5,                 # default=0.5：預設前後各避開 0.5 秒
        help="背景片段要避開敲擊事件前後多少秒，避免沾到敲擊尾巴",
    )

    parser.add_argument(
        "--event-overlap-sec",       # --event-overlap-sec：判斷某個片段是否算 tap 時，事件時間範圍可以額外放寬多少秒，如果你懷疑事件標註切得太緊，就可以把這個值調大一點
        type=float,
        default=0.0,                 # default=0.0：預設不額外放寬
        help="判斷片段是否屬於敲擊事件時，額外放寬的重疊秒數",
    )

    parser.add_argument(
        "--batch-filter",    # --batch-filter：指定只分析哪些批次
        nargs="*",           # nargs="*"：表示這個參數可以一次接收多個值
        default=None,        # default=None：如果沒指定，就代表使用程式裡的預設批次設定
        help="只分析特定批次，例如 disturbance_ann_0624 disturbance_ann_0702",
    )

    parser.add_argument(
        "--position-filter",    # --position-filter：指定只分析哪些位置，也是可以一次輸入多個值
        nargs="*",
        default=None,           # default=None：如果沒指定，就表示所有位置都納入分析
        help="只分析特定位置，例如 straight0cm straight20cm",
    )

    parser.add_argument(
        "--output-csv",         # --output-csv：指定輸出的主檔名與路徑
        type=Path,
        default=DEFAULT_OUTPUT,
        help="輸出峰值清單 CSV"
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
    """
    判斷兩個時間區間是否有重疊。
    兩段區間的「較晚開始時間」,是否還早於兩段區間的「較早結束時間」,如果是，就代表中間有重疊。
    """
    return max(a_start, b_start) < min(a_end, b_end)


# ============================================================
# 讀取 manifest / events
# ============================================================
# 讀 manifest，而且把 0713 的欄位名稱先轉成跟舊版一致
def read_manifest(manifest_path: Path) -> pd.DataFrame:    # 輸入是 manifest_path, 型別標註 Path 表示你希望它是一個路徑物件, -> pd.DataFrame 表示這個函式最後會回傳一個 pandas 表格
    # 把 manifest_path 這個 CSV 檔讀進來，存成 df
    df = pd.read_csv(manifest_path)

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

    return df


# 讀事件檔
def read_events(events_path: Path) -> pd.DataFrame:
    # 把事件 CSV 讀進來, 這份表通常是在記錄：干擾種類, 開始時間, 結束時間
    df = pd.read_csv(events_path)

    # 定義事件檔最少要有的欄位
    required = {"disturbance_type", "start_elapsed_sec", "end_elapsed_sec"}
    # 檢查事件檔有沒有缺必要欄位
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"events 檔缺少欄位 {missing}: {events_path}")

    return df


# 負責找到某個 session 的標註檔
def find_session_annotation_files(session_dir: Path) -> tuple[Path | None, Path | None]:
    """
    根據不同批次的命名規則，自動找到 manifest 與 events 檔。
    """
    # 找 manifest 候選檔案, 資料有兩種命名
    manifest_candidates = (
        list(session_dir.glob("dual_session_manifest_*.csv")) +     # 舊版：dual_session_manifest_*.csv
        list(session_dir.glob("transition_dual_manifest_*.csv"))    # 0713：transition_dual_manifest_*.csv
    )
    # 找 events 檔案。也是同時支援：舊版命名, 0713 命名
    event_candidates = (
        list(session_dir.glob("disturbance_events_*.csv")) +
        list(session_dir.glob("transition_dual_disturbance_events_*.csv"))
    )

    # 如果有找到候選檔，就拿第一個, 如果完全沒找到，就設成 None。
    manifest_path = manifest_candidates[0] if manifest_candidates else None
    events_path = event_candidates[0] if event_candidates else None
    return manifest_path, events_path


# manifest 裡是 Windows 路徑，但你現在在 macOS 跑
def resolve_dual_tdms_path(session_dir: Path,                # session_dir：這個 session 的資料夾
                           dual_file_value: str) -> Path:    # dual_file_value：manifest 裡某一列記錄的 TDMS 路徑字串
    # 先把 dual_file_value 轉成字串，再把前後空白去掉    
    raw = str(dual_file_value).strip()

    # 先用 Windows 路徑規則取 basename
    filename = PureWindowsPath(raw).name

    # 正常情況下，TDMS 檔就在 session 資料夾底下
    candidate = session_dir / filename
    if candidate.exists():
        return candidate

    # 保險一點：如果不在最外層，就往下找同名檔
    matches = list(session_dir.rglob(filename))
    # 如果有找到符合的檔案，就回傳第一個
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"找不到對應 TDMS 檔案，session_dir={session_dir}, dual_file={dual_file_value}, filename={filename}")


# ============================================================
# 檢查 manifest 是「一列一檔」還是「一檔多片段」
# ============================================================
def inspect_manifest_structure(manifest_path: Path) -> pd.DataFrame:
    """
    檢查 manifest 的資料結構：
    1. 如果大多數 dual_file 只出現 1 次，通常代表每列對應一個獨立 TDMS 小檔
    2. 如果很多 dual_file 重複出現，通常代表同一個 TDMS 大檔被切成多個時間片段
    """
    df = read_manifest(manifest_path)

    print("=" * 60)
    print(f"檢查檔案: {manifest_path}")
    print(f"總列數: {len(df)}")
    print(f"dual_file 不重複數: {df['dual_file'].nunique()}")
    print()

    # 統計每個 dual_file 出現幾次
    summary = (
        df.groupby("dual_file")
        .agg(     # agg(欄位名=(來源, 函式)) 對每一組同時做多種聚合(aggregate)運算,並且用具名的方式一次產生好幾個新欄位
            row_count=("dual_file", "size"),                 # 用 size 算這組裡有幾筆資料列,結果放進新欄位 row_count
            dual_index_min=("dual_index", "min"),            # 這組裡 dual_index 欄位的最小值
            dual_index_max=("dual_index", "max"),            # 這組裡 dual_index 欄位的最大值
            start_min=("capture_start_elapsed_sec", "min"),  # 這組裡 capture_start_elapsed_sec 的最小值,也就是最早的起始時間
            end_max=("capture_end_elapsed_sec", "max"),      # 這組裡 capture_end_elapsed_sec 的最大值,也就是最晚的結束時間
        )
        .sort_values("row_count", ascending=False)           # 依照剛算出來的 row_count(每個 dual_file 出現的次數)由大到小排序,出現次數最多的排在最前面
        .reset_index()
    )

    print("前 20 個 dual_file 統計：")
    print(summary.head(20).to_string(index=False))
    print()

    repeated_count = (summary["row_count"] > 1).sum()
    single_count = (summary["row_count"] == 1).sum()

    print(f"只出現 1 次的 dual_file 數量: {single_count}")
    print(f"重複出現超過 1 次的 dual_file 數量: {repeated_count}")
    print()

    if repeated_count == 0:
        print("判斷結果：看起來接近「一列對應一個獨立 TDMS 檔」")
        print("目前 averaged_psd() 直接讀整個 TDMS 的寫法，大致合理。")
    else:
        print("判斷結果：看起來有「同一個 TDMS 檔對應多列 manifest」的情況")
        print("這表示可能需要先根據 t_start / t_end 切出片段，再算 PSD。")

    print("=" * 60)
    return summary


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

    # 只保留敲擊事件
    tap_events = events[
        # 把 disturbance_type 這欄轉成字串, 轉小寫, 看有沒有包含 "tap"
        events["disturbance_type"].astype(str).str.lower().str.contains("tap", na=False)].copy()

    # 如果這個 session 裡根本沒有敲擊事件，那就不用分析了，直接回傳空清單
    if tap_events.empty:
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
        tdms_path = resolve_dual_tdms_path(session_dir, row.dual_file)    # 找到這個片段所屬的 TDMS 檔案路徑
  
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
    '''「從某個 TDMS 檔裡，讀出你指定的聲道資料。」'''
    tdms = TdmsFile.read(filepath)    # 把這個 TDMS 檔整個讀進來
    target = channel_name.lower()     # 把你指定的聲道名稱先轉成小寫
    
    # 把 TDMS 裡所有 group、所有 channel 一個一個翻出來看
    for group in tdms.groups():
        for channel in group.channels():
            # 如果目前這個 channel 的名字，轉成小寫後，剛好等於你要找的 target，就表示找到了
            if channel.name.lower() == target:
                return np.asarray(channel[:], dtype=np.float64)    # 找到後就把這個 channel 的資料整段取出來
            
    # 如果前面整個找完都沒找到指定聲道，這裡就先把目前檔案裡所有可用的聲道名稱列出來
    available = [channel.name for group in tdms.groups() for channel in group.channels()]
    
    raise ValueError(f"在 {filepath} 找不到聲道 {channel_name}，可用聲道有: {available}")


# ============================================================
# 對多個片段做平均 PSD
# ============================================================
def averaged_psd(records: list[SliceRecord],    # 很多個片段清單
                 channel_name: str,             # 要讀哪個聲道
                 fs: float,                     # 取樣率
                 nperseg: int):                 # Welch method 的分段長度
    # 如果 records 是空的，代表根本沒有可用片段，那就直接報錯。
    if not records:
        raise ValueError("沒有可用片段可計算 PSD")

    psd_list = []    # 等等用來存每個片段算出來的 PSD
    freqs = None     # 等等存頻率軸

    # 逐個處理每一個片段，注意這裡的 record 是一個一個被標成 tap 或 background 的小片段。
    for record in records:
        '''
        這份 disturbance 資料中，一個 record 雖然是「片段紀錄」，但它對應的 tdms_path 本身通常就是一個已經切好的小 TDMS 檔。
        所以這裡直接讀整個 TDMS 聲道來算 PSD 是合理的，不需要再另外根據 t_start / t_end 做二次切片。
        '''
        # 先去讀這個片段所屬的 TDMS 檔案，把指定聲道的原始波形拿出來
        signal = read_tdms_channel(record.tdms_path, channel_name)

        # 對這個 signal 算 Welch PSD，「把訊號切成多段，分別算頻譜，再平均，讓結果更穩定。」
        cur_freqs, cur_psd = welch(
            signal,   # 輸入訊號
            fs=fs,    # 取樣率
            nperseg=min(nperseg, len(signal)),       # 如果你設定的 nperseg 比訊號還長，就不要硬用，改成用訊號本身長度。
            noverlap=min(nperseg, len(signal)) // 2, 
        )
        
        # 如果這是第一個片段，就把它的頻率軸記下來
        if freqs is None:
            freqs = cur_freqs

        # 把這個片段算出的 PSD 加進 psd_list
        psd_list.append(cur_psd)

    # 把很多條 PSD 疊成一個 2D 陣列，每一列 = 一個片段的 PSD，每一欄 = 某個頻率點的功率值
    psd_stack = np.vstack(psd_list)          # np.vstack 是 NumPy 的函式,全名是 "vertical stack"(垂直堆疊),用來把多個陣列沿著垂直方向(也就是「往下疊」、增加列數)接在一起,
    return freqs, psd_stack.mean(axis=0)     # 最後沿著列方向做平均。也就是：同一個頻率點，把所有片段的 PSD 值平均起來


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
        if f_lo <= freq_hz <= f_hi:
            return band_name, f_lo, f_hi, band_type
    # 如果整個 ALL_BANDS 都找完了，還是沒有任何一段包含這個頻率，那就回傳："off_band"：不在已定義頻帶內 , np.nan, np.nan：沒有頻帶上下界 , "其他"：類型歸到其他
    return "off_band", np.nan, np.nan, "其他"


# ============================================================
# 把大量峰值整理成：這段頻帶範圍, 裡面有幾個峰, 代表峰在哪, 最大 dB 是多少 = 從一堆峰值點 -> 頻帶層級摘要
# ============================================================
def build_band_summary_table(peaks_df: pd.DataFrame) -> pd.DataFrame:    # 輸入：peaks_df 也就是峰值清單表
    """
    把大量單點峰值整理成幾段可讀性比較高的頻帶摘要表。
    """
    rows = []     # 先準備一個空清單，等等每整理出一段頻帶，就加一列進去。

    # 穩定共振帶：先篩資料，只保留 0713 驗證通過的峰值，不是把所有搜尋到的峰都拿來整理，而是只整理那些「跨天驗證後仍然成立」的峰。
    valid_df = peaks_df[peaks_df["validation_0713_pass"] == True].copy()

    # 開始逐一掃過你定義好的每一段頻帶
    for band_name, f_lo, f_hi, band_type in ALL_BANDS:
        # 從 valid_df 裡挑出：「所有落在這段頻帶範圍內的峰值」，sub 可以理解成：某一段頻帶內的子集合。
        sub = valid_df[
            (valid_df["frequency_Hz"] >= f_lo) &
            (valid_df["frequency_Hz"] <= f_hi)].copy()

        # 如果這段頻帶內一個峰都沒有，那就跳過
        if sub.empty:
            continue

        # rep_row 就是這段頻帶裡最有代表性的峰:在這段頻帶裡，找 tap_vs_bg_dB 最大的那一個峰
        rep_row = sub.loc[sub["tap_vs_bg_dB"].idxmax()]

        # 建立這個頻帶的一列摘要資料
        rows.append(
            {
                "band_name": band_name,    # 這段頻帶名稱，例如 core_01
                "band_type": band_type,    # 這段是 core 還是 secondary
                "band_start_Hz": f_lo,     # 頻帶起點
                "band_end_Hz": f_hi,       # 頻帶終點
                "band_range": f"{int(f_lo)}-{int(f_hi)} Hz",    # 給人讀的文字格式，例如 118500-130000 Hz
                "peak_count": len(sub),    # 這段頻帶裡總共有幾個驗證通過的峰
                "representative_peak_Hz": rep_row["frequency_Hz"],    # 代表峰的頻率，也就是這段裡最強的那個峰
                "max_tap_vs_bg_dB": sub["tap_vs_bg_dB"].max(),        # 搜尋資料中，這段頻帶內最大的 tap_vs_bg_dB
                "max_validation_0713_tap_vs_bg_dB": sub["validation_0713_tap_vs_bg_dB"].max(),    # 在 0713 驗證資料裡，這段頻帶內最大的對應 dB
                "note": band_type,    # 備註，這裡直接寫 band_type
            }
        )

    if not rows:
        return pd.DataFrame()
    
    # 最後把 rows 轉成 DataFrame，再排序(先按 band_type, 再按 band_start_Hz)後回傳。
    return pd.DataFrame(rows).sort_values(["band_type", "band_start_Hz"]).reset_index(drop=True)


# ============================================================
# 繪圖函式
# ============================================================
# 畫前三天搜尋頻譜 vs 0713 驗證頻譜疊圖
def plot_search_vs_validation_overlay(
    search_spectrum_df: pd.DataFrame,     # 前三天搜尋資料的完整頻譜表
    validation_spectrum_df: pd.DataFrame, # 713 驗證資料的完整頻譜表
    output_png: Path,                     # 這張圖最後要存到哪裡
) -> None:
    """
    * 看整條頻譜在搜尋資料和驗證資料之間是否一致 *
    
    畫「前三天搜尋頻譜 vs 0713 驗證頻譜」疊圖，並用陰影標出核心頻帶。
    """
    # 建立一張圖(寬 14、高 6)和一個座標軸。
    fig, ax = plt.subplots(figsize=(14, 6))

    # 先畫第一條線，也就是前三天搜尋頻譜
    ax.plot(
        search_spectrum_df["frequency_Hz"],    # x 軸：frequency_Hz
        search_spectrum_df["tap_vs_bg_dB"],    # y 軸：tap_vs_bg_dB
        label="搜尋頻譜（0624+0702+0707）",
        linewidth=1.5,
        color="tab:blue",
    )

    # 再畫第二條線，也就是 0713 驗證頻譜
    ax.plot(
        validation_spectrum_df["frequency_Hz"],                     # x 軸一樣是 frequency_Hz
        validation_spectrum_df["validation_0713_tap_vs_bg_dB"],     # y 軸換成 validation_0713_tap_vs_bg_dB
        label="驗證頻譜（0713）",
        linewidth=1.2,
        color="tab:orange",
        alpha=0.9,    # 稍微有一點透明度
    )

    # 畫灰色陰影區, f_lo：頻帶起點, f_hi：頻帶終點
    for band_name, f_lo, f_hi, _ in CORE_BANDS:
        ax.axvspan(f_lo, f_hi, color="gray", alpha=0.15)    # 直接把你選的核心頻帶標出來

    # 設定圖標題
    ax.set_title("搜尋頻譜 vs 2026-07-13 驗證頻譜")
    ax.set_xlabel("頻率 (Hz)")
    ax.set_ylabel("tap_vs_bg_dB")
    ax.legend()            # 顯示圖例
    ax.grid(alpha=0.25)    # 加上淡淡的格線

    fig.tight_layout()    # 自動調整版面
    fig.savefig(output_png, dpi=180, bbox_inches="tight")    # bbox_inches="tight"：把多餘空白裁掉一些
    plt.close(fig)


# ============================================================
# 只畫驗證通過的峰，並依照頻帶分組顏色
# ============================================================
def plot_peak_grouped_by_band(
    peaks_df: pd.DataFrame,    # 峰值表
    output_png: Path,          # 輸出圖檔路徑
) -> None:
    """
    * 看驗證通過的峰值是否真的群聚成幾段可定義的頻帶 *
    只畫驗證通過的峰，並依照頻帶分組上色。
    不是畫所有峰，只畫 0713 驗證通過的峰，而且依照它屬於哪一段頻帶來上色
    """
    # 先從 peaks_df 裡，只保留驗證通過的峰值，也就是：只畫那些跨天仍然成立的峰
    plot_df = peaks_df[peaks_df["validation_0713_pass"] == True].copy()
    # 如果一個驗證通過的峰都沒有，就直接不畫圖
    if plot_df.empty:
        return

    # 對每個峰值頻率，呼叫 find_band_for_frequency() 去判斷：這個峰屬於哪個 band、band 上下界是多少、類型是 core 還是 secondary
    band_info = plot_df["frequency_Hz"].apply(find_band_for_frequency)
    # 剛剛回傳的結果拆出來，存成新欄位，band_name：例如 core_01、band_type：例如 core 或 secondary
    plot_df["band_name"] = [x[0] for x in band_info]
    plot_df["band_type"] = [x[3] for x in band_info]

    # 只保留落在你規劃頻帶內的峰：只畫核心/次要，不畫 off_band
    plot_df = plot_df[plot_df["band_name"] != "off_band"].copy()
    # 如果去掉 off_band 之後沒東西了，就也不用畫
    if plot_df.empty:
        return

    # 手動指定每個頻帶的顏色
    color_map = {
        "core_01": "#d62728",
        "core_02": "#1f77b4",
        "core_03": "#2ca02c",
        "core_04": "#ff7f0e",
        "secondary_01": "#9467bd",
        "secondary_02": "#8c564b",
        "secondary_03": "#e377c2",
    }

    # 建立新圖
    fig, ax = plt.subplots(figsize=(14, 6))
    # 把資料依 band_name 分組
    for band_name, sub in plot_df.groupby("band_name"):
        ax.scatter(    # 把每一組畫成散點圖
            sub["frequency_Hz"],    # x 軸：frequency_Hz
            sub["tap_vs_bg_dB"],    # y 軸：tap_vs_bg_dB
            label=band_name,        # 圖例顯示 band 名稱
            s=28,                   # 點的大小
            alpha=0.85,             # 透明度
            color=color_map.get(band_name, "gray"),    # 從 color_map 取顏色，沒有就用灰色
        )

    ax.set_title("驗證通過峰值的頻帶分組圖")
    ax.set_xlabel("頻率 (Hz)")
    ax.set_ylabel("tap_vs_bg_dB")
    ax.legend(ncol=2, fontsize=9)    # ncol=2：圖例分成兩欄
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    

# ============================================================
# 畫各批次用到多少 tap/background 片段
# ============================================================
def plot_data_coverage(
    slice_summary_df: pd.DataFrame,    # 片段總表，也就是 build_summary_df() 整理出來的那張表
    output_png: Path,                  # 圖片儲存路徑
) -> None:
    """
    畫這次搜尋用到的 tap/background 片段分布，讓你確認資料不是只集中在單一天或單一批次。
    因為如果全部幾乎都來自同一天，那你找到的頻帶可能比較像「那天特有的現象」，而不是穩定的管子響應
    """
    # 把 slice_summary_df 依照：batch_name , label 組後，數每組有幾列
    cov = (
        slice_summary_df.groupby(["batch_name", "label"]).size()
        .reset_index(name="count")    # reset_index(name="count") 是把原本 groupby 的結果整理回一般表格格式，並把計數欄命名成 count
    )

    # 把剛剛那張長格式表，轉成適合畫柱狀圖的寬格式表
    pivot = cov.pivot(index="batch_name", columns="label", values="count").fillna(0)

    # 建立一張圖
    fig, ax = plt.subplots(figsize=(10, 5))
    # 把 pivot 畫成長條圖
    pivot.plot(kind="bar", ax=ax, color=["#a6cee3", "#fb9a99"])

    # 設定圖標題和座標軸名稱
    ax.set_title("Slice Coverage by Batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Number of slices")
    ax.grid(axis="y", alpha=0.25)    # 只在 y 軸方向加格線

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 把頻帶摘要表畫成圖，方便貼報告
# ============================================================
def save_band_summary_png(
    band_summary_df: pd.DataFrame,    # 頻帶摘要表
    output_png: Path,                 # 輸出的圖片路徑
) -> None:
    """
    把頻帶摘要表另存成圖片，方便直接貼報告或簡報。
    表頭：藍色
    core：偏藍白
    secondary：偏暖白
    """
    # 如果表格是空的，就直接不畫
    if band_summary_df.empty:
        return
    
    # 先複製一份資料，避免直接改到原始表。
    display_df = band_summary_df.copy()
    
    # 挑出要顯示的欄位
    display_df = display_df[
        [
            "band_name",                 # 頻帶名稱
            "band_type",                 # core / secondary
            "band_range",                # 頻帶範圍
            "peak_count",                # 這段內有幾個峰
            "representative_peak_Hz",    # 代表峰頻率
            "max_tap_vs_bg_dB",          # 搜尋資料內最大強度
            "max_validation_0713_tap_vs_bg_dB",    # 0713 驗證資料內最大強度
        ]
    ]

    # 自動決定圖的高度, 因為表格列數可能不固定, 所以這裡根據列數去估算高度
    fig_height = 0.55 * len(display_df) + 1.5
    fig, ax = plt.subplots(figsize=(12, fig_height))    # 建立圖，寬度固定 12，高度用剛剛算出來的 fig_height
    ax.axis("off")    # 把座標軸關掉，因為這不是一般圖，而是純表格，不需要看到 x/y 軸。

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

    # 開始逐格處理表格樣式，r：第幾列 , c：第幾欄 , cell：這一格本身
    for (r, c), cell in table.get_celld().items():
        if r == 0:    # 如果是第 0 列，也就是表頭：文字加粗, 背景色設成淡藍色
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")
        # 如果不是表頭，而且這一列對應的 band_type 是 core，就把背景設成一種很淡的藍白色。
        elif display_df.iloc[r - 1]["band_type"] == "core":
            cell.set_facecolor("#f7fbff")
        # 否則就表示這列不是 core，通常就是 secondary，背景設成淡淡的米橘色。
        else:
            cell.set_facecolor("#fff8f2")

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 這個函式是把前面掃描、找 annotation、切片標記，包成一個比較乾淨的入口
# ============================================================
def collect_records(
    root: Path,                         # disturbance 的根目錄
    batch_filter: set[str],             # 要分析哪些批次，例如 0624、0702、0707
    position_filter: set[str] | None,   # 如果只想看某些位置，就傳進來；不限制就用 None
    background_guard_sec: float,        # 背景片段要離 tap 多遠才算安全
    event_overlap_sec: float,           # 判斷 tap 片段時，要不要把事件邊界放寬
) -> list[SliceRecord]:
    """
    收集指定批次中的所有 tap/background 片段。
    """
    # 這次有哪些 session 要處理:是 tap 類型資料夾 -> 屬於指定批次 -> 裡面有對應的 manifest / events
    session_dirs = list(iter_session_dirs(root, batch_filter=batch_filter))
    # 先準備一個空清單 all_records。等等每個 session 整理出來的片段，都會往這裡加。
    all_records: list[SliceRecord] = []

    # 開始逐個處理每個 session
    for session_dir in session_dirs:
        # 對這個 session_dir，去找它裡面的兩份標註檔：manifest_path, events_path
        manifest_path, events_path = find_session_annotation_files(session_dir)
    
        # 如果這個 session 缺檔：找不到 manifest, 或找不到 events, 那就直接跳過這個 session。
        if manifest_path is None or events_path is None:
            continue

        # 對當前這個 session，呼叫 build_slice_records(...) 去做真正的片段整理
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

    return all_records


# ============================================================
# 把 ALL_BANDS 輸出成 JSON 的函式
# ============================================================
def save_resonance_band_json(
    output_json: Path,                # 最後要輸出的 JSON 檔路徑
    search_batches: set[str],         # 這次拿來找共振候選峰的批次集合
    validation_batches: set[str],     # 這次拿來做跨天驗證的批次集合
) -> None:
    # 建立 payload(JSON 檔的完整內容)
    payload = {
        "source": "resonance_exploration",                   # 標記：這份 JSON 是哪種流程產生的
        "search_batches": sorted(search_batches),            # 把傳進來的 search_batches 寫進 JSON
        "validation_batches": sorted(validation_batches),    # 把驗證批次也寫進 JSON
        "resonance_bands": [                                 # 共振頻帶
            {   # 把 ALL_BANDS 裡的每一段頻帶，逐一轉成 JSON 可用的 dictionary 格式
                "name": (
                    f"res_core_{int(f_lo/1000)}_{int(f_hi/1000)}k"     # 每一段頻帶產生正式輸出名稱
                    if band_type == "core"
                    else f"res_sec_{int(f_lo/1000)}_{int(f_hi/1000)}k"
                ),
                "f_low": int(f_lo),        # 存下界
                "f_high": int(f_hi),       # 存上界
                "band_type": band_type,    # 存頻帶的類型
            }
            for _, f_lo, f_hi, band_type in ALL_BANDS
        ]
    }
    # 寫出 JSON 檔
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",)
    
    
# ============================================================
# 比對這次驗證通過的峰，跟目前寫死的 CORE_BANDS/SECONDARY_BANDS 有沒有落差
# ============================================================
def audit_band_drift(
    peaks_df: pd.DataFrame,                # 前面找到的候選峰清單,也就是 result_df
    all_bands: list[tuple],                # 目前寫死的 ALL_BANDS
    boundary_margin_hz: float = 500.0,     # 邊界容許值,預設 500 Hz
) -> dict:
    """
    不會修改 CORE_BANDS / SECONDARY_BANDS，也不會影響 resonance_bands.json。
    純粹拿這次結果去檢查兩件事：
    1. 有沒有跨天驗證通過、但完全沒被目前任何頻帶涵蓋到的強峰
    2. 有沒有峰雖然落在頻帶內，但離邊界太近，頻帶範圍可能該微調
    """
    # 只留下跨天驗證通過的峰
    valid_df = peaks_df[peaks_df["validation_0713_pass"] == True].copy()
    # 如果篩完一個都不剩,就直接回傳兩張空表
    if valid_df.empty:
        return {"uncovered_peaks": pd.DataFrame(), "boundary_close_peaks": pd.DataFrame()}

    # 標註每個峰落在哪個頻帶
    # 呼叫前面定義的 find_band_for_frequency()——這個函式會去掃 ALL_BANDS,看這個頻率落在哪一段裡,回傳 (band_name, f_lo, f_hi, band_type)
    band_info = valid_df["frequency_Hz"].apply(find_band_for_frequency)
    valid_df["band_name"] = [x[0] for x in band_info]
    valid_df["band_low"] = [x[1] for x in band_info]
    valid_df["band_high"] = [x[2] for x in band_info]

    # 找出完全沒被涵蓋到的峰(跨天驗證通過、但目前 CORE_BANDS/SECONDARY_BANDS 完全沒涵蓋到)
    uncovered = valid_df[valid_df["band_name"] == "off_band"].copy()
    uncovered = uncovered.sort_values("tap_vs_bg_dB", ascending=False)

    # 找出落在頻帶內、但太靠近邊界的峰
    # 先取出「有被涵蓋到」的峰(covered)。對每一個峰,算它離「所在頻帶下界」的距離、跟離「所在頻帶上界」的距離,兩者取比較小的那個當作 dist_to_edge_Hz——也就是「這個峰離它所在頻帶最近的那條邊,有多近」。
    covered = valid_df[valid_df["band_name"] != "off_band"].copy()
    dist_to_low = (covered["frequency_Hz"] - covered["band_low"]).abs()
    dist_to_high = (covered["frequency_Hz"] - covered["band_high"]).abs()
    covered["dist_to_edge_Hz"] = np.minimum(dist_to_low, dist_to_high)
    
    # 接著篩出 dist_to_edge_Hz <= boundary_margin_hz(預設小於等於 500 Hz)的峰
    boundary_close = covered[covered["dist_to_edge_Hz"] <= boundary_margin_hz].copy()
    boundary_close = boundary_close.sort_values("dist_to_edge_Hz")

    # uncovered_peaks:完全沒被涵蓋到的峰, boundary_close_peaks:落在頻帶內、但太靠近邊界的峰
    return {"uncovered_peaks": uncovered, "boundary_close_peaks": boundary_close}


# ============================================================
# 接收 audit_band_drift 算出來的結果，負責把結果印出來
# ============================================================
def report_band_drift(drift: dict) -> bool:
    """
    印出頻帶落差警告。回傳 True 代表這次有偵測到落差，值得你回去人工檢查。
    """
    # 先把兩個 DataFrame 分別取出來
    uncovered = drift["uncovered_peaks"]             # 完全沒被涵蓋到的峰
    boundary_close = drift["boundary_close_peaks"]   # 落在頻帶內、但太靠近邊界的峰
    has_drift = not uncovered.empty or not boundary_close.empty     # 這個布林值就是整個函式最後要回傳的東西，用來讓呼叫端（main()）知道要不要進一步存成提案檔

    # 如果兩個 DataFrame 都是空的，就印出一個綠色勾勾的訊息安心通知你「這次沒問題」，然後直接回傳 False
    if not has_drift:
        print("\n✅ 頻帶比對：這次驗證通過的峰全部落在現有 CORE_BANDS/SECONDARY_BANDS 內，且離邊界夠遠，沒有偵測到落差。")
        return False

    print("\n" + "!" * 70)
    print("⚠️ 頻帶比對：偵測到跟目前寫死的 CORE_BANDS/SECONDARY_BANDS 有落差，建議人工檢查")
    print("!" * 70)

    # 如果「完全落在頻帶外」的峰不是空的，就印出總數
    if not uncovered.empty:
        print(f"\n有 {len(uncovered)} 個跨天驗證通過的峰，落在目前所有頻帶範圍之外：")
        print(uncovered[["frequency_Hz", "tap_vs_bg_dB", "validation_0713_tap_vs_bg_dB"]].head(10).to_string(index=False))

    # 落在頻帶內、但離邊界太近的峰
    if not boundary_close.empty:
        print(f"\n有 {len(boundary_close)} 個峰落在頻帶內，但離邊界不到容許值，頻帶邊界可能該微調：")
        print(boundary_close[["frequency_Hz", "band_name", "dist_to_edge_Hz", "tap_vs_bg_dB"]].head(10).to_string(index=False))

    print("\n這不會自動修改程式碼裡的 CORE_BANDS / SECONDARY_BANDS，也不會影響這次輸出的 resonance_bands.json。")
    print("如果確認這些落差是真的、且穩定重現，再自行回去修改檔案最上面的 CORE_BANDS / SECONDARY_BANDS。")
    print("!" * 70 + "\n")
    return True
    

# ============================================================
# 主流程
# ============================================================
def main() -> None:
    # 前面設定的參數讀進來，之後像 args.root、args.channel、args.nperseg 都是從這裡拿
    args = parse_args()
    # 位置篩選條件
    position_filter = set(args.position_filter) if args.position_filter else None

    # 決定：哪些批次拿來「找共振候選峰」,哪些批次拿來「做驗證」
    # 如果你手動傳 --batch-filter，就覆蓋預設搜尋批次
    search_batches = set(args.batch_filter) if args.batch_filter else DEFAULT_SEARCH_BATCHES
    validation_batches = DEFAULT_VALIDATION_BATCHES


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
    # 2. 用前三天資料計算平均頻譜，找共振候選峰
    # ============================================================
    # tap_records 算出 psd_tap , bg_records 算出 psd_bg
    freqs, psd_tap = averaged_psd(tap_records, args.channel, args.fs, args.nperseg)
    _, psd_bg = averaged_psd(bg_records, args.channel, args.fs, args.nperseg)

    # 計算 tap 相對於 background 的 dB 差值
    # 可以觀察哪些頻率敲擊時特別明顯，而且比背景高很多
    eps = 1e-20     # 避免分母或分子剛好接近 0 時出錯
    diff_db = 10 * np.log10((psd_tap + eps) / (psd_bg + eps))

    # 在 diff_db 這條曲線上找峰值
    peak_idx, peak_props = find_peaks(
        diff_db,
        height=args.peak_db_threshold,    # 峰值高度至少要超過 args.peak_db_threshold
        distance=5,                       # 峰與峰之間至少隔 5 個頻率點
        prominence=3.0,                   # 這個峰比它左右兩側的局部谷底高出多少，先設定 3dB
    )

    # 剛剛找到的峰整理成表格, 依強度由大到小排序
    result_df = pd.DataFrame(
        {
            "frequency_Hz": freqs[peak_idx],    # 峰值頻率
            "tap_vs_bg_dB": peak_props["peak_heights"],    # 這個峰有多高
        }).sort_values("tap_vs_bg_dB", ascending=False)

    # 先把 0713 驗證欄位建立好，這樣即使 0713 沒有成功驗證，輸出欄位也會固定一致
    result_df["validation_0713_tap_vs_bg_dB"] = np.nan
    result_df["validation_0713_pass"] = False


    # ============================================================
    # 3. 用 0713 做驗證，不重新找峰，只檢查前三天找到的候選峰，在 0713 是否仍然明顯
    # ============================================================
    # 收集驗證批次中的所有 tap/background 片段
    validation_records = collect_records(
        root=args.root,
        batch_filter=validation_batches,
        position_filter=position_filter,
        background_guard_sec=args.background_guard_sec,
        event_overlap_sec=args.event_overlap_sec,
    )

    # 把 0713 的資料也拆成：驗證用 tap , 驗證用 background
    val_tap_records = [r for r in validation_records if r.label == "tap"]
    val_bg_records = [r for r in validation_records if r.label == "background"]

    # 先建立一個空表, 如果後面 0713 成功算出頻譜，就把它填進去；如果沒有，就維持空表
    validation_spectrum_df = pd.DataFrame()

    # 確認：有驗證資料 + 有驗證 tap + 有驗證 background + 前三天搜尋也真的找到峰，四個都成立才做驗證。
    if validation_records and val_tap_records and val_bg_records and not result_df.empty:
        # 先把 0713 的 tap/background 平均 PSD 算出來
        val_freqs, val_psd_tap = averaged_psd(val_tap_records, args.channel, args.fs, args.nperseg)
        _, val_psd_bg = averaged_psd(val_bg_records, args.channel, args.fs, args.nperseg)

        # 再算出 0713 自己的 tap_vs_bg_dB 曲線
        val_diff_db = 10 * np.log10((val_psd_tap + eps) / (val_psd_bg + eps))

        # 把前三天找到的候選頻率，對到 0713 的頻譜上
        result_df["validation_0713_tap_vs_bg_dB"] = np.interp(
            result_df["frequency_Hz"].values, val_freqs, val_diff_db,)

        # 用同樣門檻判斷 0713 是否也支持這個峰, 如果有，就標成 True
        result_df["validation_0713_pass"] = (
            result_df["validation_0713_tap_vs_bg_dB"] >= args.peak_db_threshold
        )

        # 把 0713 的完整驗證頻譜整理成一張表，後面可以輸出 CSV，也可以拿來畫疊圖
        validation_spectrum_df = pd.DataFrame(
            {
                "frequency_Hz": val_freqs,
                "validation_0713_tap_PSD": val_psd_tap,
                "validation_0713_background_PSD": val_psd_bg,
                "validation_0713_tap_vs_bg_dB": val_diff_db,
            }
        )
    else:
        print("0713 驗證資料不足，這次先只輸出前三天的共振候選峰")

    # ============================================================
    # 4. 整理輸出資料夾結構
    # ============================================================
    # 先整理前三天完整頻譜表與片段摘要表
    spectrum_df = pd.DataFrame(
        {
            "frequency_Hz": freqs,
            "tap_PSD": psd_tap,
            "background_PSD": psd_bg,
            "tap_vs_bg_dB": diff_db,
        }
    )
    # 把用到的片段清單整理成一張追查表
    summary_df = build_summary_df(all_records)
    
    # 決定輸出資料夾結構
    output_anchor = args.output_csv
    run_name = output_anchor.stem
    output_root = output_anchor.parent / run_name

    tables_dir = output_root / "tables"       # 表格
    spectra_dir = output_root / "spectra"     # 完整頻譜
    figures_dir = output_root / "figures"     # 圖檔
    configs_dir = output_root / "configs"     # 輸出共振 json 檔

    # 如果這些資料夾不存在，就建立起來
    for d in [output_root, tables_dir, spectra_dir, figures_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 表格類
    output_csv = tables_dir / f"{run_name}.csv"
    slice_summary_csv = tables_dir / f"{run_name}_slice_summary.csv"
    band_summary_csv = tables_dir / f"{run_name}_band_summary.csv"
    resonance_json = configs_dir / f"{run_name}_resonance_bands.json"

    # 頻譜類
    full_spectrum_csv = spectra_dir / f"{run_name}_full_spectrum.csv"
    validation_csv = spectra_dir / f"{run_name}_validation_0713_spectrum.csv"

    # 圖檔類
    band_summary_png = figures_dir / f"{run_name}_band_summary.png"
    overlay_png = figures_dir / f"{run_name}_search_vs_validation_overlay.png"
    peak_group_png = figures_dir / f"{run_name}_peak_grouped_by_band.png"
    coverage_png = figures_dir / f"{run_name}_data_coverage.png"

    # ============================================================
    # 5. 輸出 CSV
    # ============================================================
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    spectrum_df.to_csv(full_spectrum_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(slice_summary_csv, index=False, encoding="utf-8-sig")

    if not validation_spectrum_df.empty:
        validation_spectrum_df.to_csv(validation_csv, index=False, encoding="utf-8-sig")

    # ============================================================
    # 6. 輸出摘要表與圖
    # ============================================================
    band_summary_df = build_band_summary_table(result_df)

    # 如果頻帶摘要表不是空的：存成 CSV, 再存成 PNG 表格圖
    if not band_summary_df.empty:
        band_summary_df.to_csv(band_summary_csv, index=False, encoding="utf-8-sig")
        save_band_summary_png(
            band_summary_df=band_summary_df,
            output_png=band_summary_png,
        )
        
    # 頻帶落差稽核：自動比對現有頻帶跟這次結果有沒有落差
    band_drift = audit_band_drift(result_df, ALL_BANDS, boundary_margin_hz=args.band_boundary_margin_hz)
    # 接收 audit_band_drift 算出來的結果，負責把結果印出來
    has_drift = report_band_drift(band_drift)
    
    band_drift_json = None
    # 如果有落差,就把細節存成 JSON
    if has_drift:
        # 組出輸出路徑
        band_drift_json = configs_dir / f"{run_name}_band_drift_proposal.json"
        
        # 準備寫進 JSON 的內容, 選出幾個真正需要的欄位作為「人工判讀落差」最關鍵的欄位
        drift_payload = {
            "uncovered_peaks": band_drift["uncovered_peaks"][
                ["frequency_Hz", "tap_vs_bg_dB", "validation_0713_tap_vs_bg_dB"]].to_dict(orient="records"),
            "boundary_close_peaks": band_drift["boundary_close_peaks"][
                ["frequency_Hz", "band_name", "dist_to_edge_Hz", "tap_vs_bg_dB"]].to_dict(orient="records"),
        }
        # 把整個 drift_payload 用 json.dumps() 轉成文字寫進檔案
        band_drift_json.write_text( 
            json.dumps(drift_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    
    # 如果有 0713 驗證頻譜，就畫疊圖
    if not validation_spectrum_df.empty:
        plot_search_vs_validation_overlay(
            search_spectrum_df=spectrum_df,
            validation_spectrum_df=validation_spectrum_df,
            output_png=overlay_png,
        )
    # 畫峰值分組圖
    plot_peak_grouped_by_band(
        peaks_df=result_df,
        output_png=peak_group_png,
    )
    # 畫資料覆蓋圖，看每個 batch 用了多少 tap/background
    plot_data_coverage(
        slice_summary_df=summary_df,
        output_png=coverage_png,
    )
    
    # 共振頻段存成json
    save_resonance_band_json(
        output_json=resonance_json,
        search_batches=search_batches,
        validation_batches=validation_batches,
    )

    # ============================================================
    # 7. 終端列印摘要
    # ============================================================
    print()
    print(f"輸出資料夾          : {output_root}")
    print(f"共振峰清單          : {output_csv}")
    print(f"片段清單            : {slice_summary_csv}")
    print(f"頻帶摘要表 CSV      : {band_summary_csv}")
    print(f"搜尋頻譜            : {full_spectrum_csv}")
    print(f"共振頻帶 JSON       : {resonance_json}")
    
    if band_drift_json is not None:
       print(f"頻帶落差提案 JSON  : {band_drift_json}")

    if not validation_spectrum_df.empty:
        print(f"0713 驗證頻譜       : {validation_csv}")

    if not band_summary_df.empty:
        print(f"頻帶摘要表 PNG      : {band_summary_png}")

    if not validation_spectrum_df.empty:
        print(f"搜尋/驗證疊圖       : {overlay_png}")

    print(f"峰值分組圖          : {peak_group_png}")
    print(f"資料覆蓋圖          : {coverage_png}")

    if not result_df.empty:
        print()
        print("前 20 個共振候選頻率：")
        print(result_df.head(20).to_string(index=False))
            

        
if __name__ == "__main__":
    configure_matplotlib_fonts()
    main()