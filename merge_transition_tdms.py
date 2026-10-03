#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把每個 transition 子資料夾內的正常段 / 洩漏段 TDMS 合併成單一連續 record。
1.找 input_root 底下所有 transition* 資料夾
    例如 transition1, transition2, transition10。
2.每個資料夾裡抓所有 .tdms
3.用檔名分成兩類
    normal_files: 檔名含 leak0 或 normal
    leak_files: 檔名含 leak1
4.固定合併順序
5.逐檔讀出 ai0、ai1
    優先抓 channel name = ai0 / ai1
    找不到就退回前兩個 channel
    每段會先把兩通道長度對齊
6.把每段訊號直接串接
7.用「正常段總長度」決定切換邊界
8.再根據 transition_pad_sec 設灰區
    你現在設定：
    transition_pad_sec = 0.05
    所以左右各留一半，也就是 0.025s
9.最後輸出一個新的 TDMS
10.同時寫一份 transition_meta.csv
    裡面記錄：
    record_id
    output_tdms
    leak_quantity
    distance_cm
    pressure_bar
    normal_end_sec
    leak_start_sec
    onset_gt_sec
    total_duration_sec

"""
# ======================================
# 匯入模組
# ======================================
from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from pathlib import Path

import numpy as np
from nptdms import ChannelObject, TdmsFile, TdmsWriter

'''
from two_sensor_feature_engineering_micro_30ml_v1 import (
    load_tdms_data,
    compute_diff_signal,
    sliding_window_records,
    extract_window_features_core_bands,
    LOW_BANDS,
    CORE_BANDS,
    AUX_BANDS,
)
'''

# ======================================
# 全域常數/頻段設定
# ======================================
DEFAULT_FS = 1_000_000.0     # 沒有採樣率時，預設 1 MHz
EPS = 1e-12                  # 避免除以 0 的超小值


# ======================================
# 解析命令列參數
# ======================================
def parse_args() -> argparse.Namespace:
    # 建立參數解析器
    parser = argparse.ArgumentParser(
        description="Merge transition folders into single TDMS records."
    )
    # 輸入資料夾根目錄，例如 /Users/paul/Desktop/validation_data
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="根目錄，裡面要有 transition1 / transition2 / ... 子資料夾",
    )
    # 輸出資料夾，例如 merged_output
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="合併後的 TDMS 與 transition_meta.csv 輸出位置",
    )
    # 指定要讀哪些檔案，預設抓所有 .tdms
    parser.add_argument(
        "--pattern",
        default="*.tdms",
        help="要讀取的檔案 pattern，預設為 *.tdms",
    )
    # 如果 TDMS 裡沒寫採樣率，就用這個值。
    parser.add_argument(
        "--fs-fallback",
        type=float,
        default=DEFAULT_FS,
        help="當 TDMS 沒有 wf_increment 時使用的備援採樣率",
    )
    # 控制 transition 灰區寬度
    parser.add_argument(
        "--transition-pad-sec",
        type=float,
        default=0.0,
        help=(
            "在 normal / leak 邊界左右各保留一半模糊區。"
            "例如 0.2 表示 normal_end_sec=boundary-0.1, leak_start_sec=boundary+0.1"
        ),
    )
    # 若有更深層子資料夾，就遞迴搜尋 TDMS
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="若 transition 子資料夾內還有更深層資料夾，改用遞迴搜尋",
    )
    return parser.parse_args()


# ======================================
# 根據檔名解析條件
# ======================================
def parse_condition_from_filename(filename: str):
    '''從檔名抓出洩漏量 ml,距離 cm,壓力 bar'''
    # 檔名轉小寫，避免大小寫問題
    name = filename.lower()

    # 先把三個欄位設成空
    leak_quantity = None
    distance_cm = None
    pressure_bar = None

    # 找像 30ml 這種格式, 找到就取出數字 30
    ml_match = re.search(r"(\d+)\s*ml", name)
    if ml_match:
        leak_quantity = int(ml_match.group(1))

    cm_match = re.search(r"(\d+)\s*cm", name)
    if cm_match:
        distance_cm = int(cm_match.group(1))

    bar_match = re.search(r"(\d+)\s*bar", name)
    if bar_match:
        pressure_bar = int(bar_match.group(1))

    return leak_quantity, distance_cm, pressure_bar


# ======================================
# 判斷 normal 檔
# ======================================
def is_normal_file(path: Path) -> bool:
    name = path.name.lower()
    # 如果檔名含 leak0 或 normal，就視為正常檔
    return ("leak0" in name) or ("normal" in name)


# ======================================
# 判斷 leak 檔
# ======================================
def is_leak_file(path: Path) -> bool:
    name = path.name.lower()
    # 如果檔名含 leak1，就視為洩漏檔
    return "leak1" in name


# ======================================
# 找資料夾裡的 TDMS
# ======================================
def get_tdms_files(folder: Path, pattern: str, recursive: bool) -> list[Path]:
    # 從資料夾抓出 TDMS 檔案,如果 recursive=True，就遞迴搜尋,排序後回傳。
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    return [p for p in files if p.is_file()]


# ======================================
# 找所有 transitionX 資料夾：一層
# ======================================
def find_transition_dirs(input_root: Path) -> list[Path]:
    # 在 input_root 底下找所有名稱以 transition 開頭的子資料夾
    dirs = [p for p in sorted(input_root.iterdir()) if p.is_dir() and p.name.lower().startswith("transition")]
    return dirs


# ======================================
# 找所有 transitionX 資料夾：二層
# ======================================
def find_transition_leaf_dirs(input_root: Path) -> list[Path]:
    leaf_dirs = []

    for interval_dir in sorted(input_root.iterdir()):
        if not interval_dir.is_dir():
            continue
        if not interval_dir.name.lower().startswith("transition_"):
            continue

        for sub_dir in sorted(interval_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            if not sub_dir.name.lower().startswith("transition"):
                continue

            leaf_dirs.append(sub_dir)

    return leaf_dirs


# ======================================
# 讀單一 TDMS 的雙通道資料
# ======================================
def read_two_channel_tdms(file_path: Path, fs_fallback: float) -> tuple[np.ndarray, np.ndarray, float]:
    # 讀檔後檢查有沒有 group
    tdms = TdmsFile.read(file_path)
    groups = tdms.groups()
    if not groups:
        raise ValueError(f"{file_path} 沒有任何 group")
    # 取第一個 group,確認至少有兩個 channel。
    group = groups[0]
    ch_list = list(group.channels())
    if len(ch_list) < 2:
        raise ValueError(f"{file_path} 不足兩個通道")
    
    # 優先用 channel 名稱 ai0 / ai1,找不到就退回用前兩個通道
    channels = {ch.name: ch for ch in ch_list}
    if "ai0" in channels and "ai1" in channels:
        ch_ai0 = channels["ai0"]
        ch_ai1 = channels["ai1"]
    else:
        warnings.warn(f"{file_path.name}: 找不到 ai0 / ai1，改用前兩個通道")
        ch_ai0, ch_ai1 = ch_list[:2]

    # 把整條波形讀出來
    ai0 = np.asarray(ch_ai0[:], dtype=np.float64)
    ai1 = np.asarray(ch_ai1[:], dtype=np.float64)

    # 對齊兩通道長度,如果有一條是空的就報錯。
    n = min(len(ai0), len(ai1))
    if n == 0:
        raise ValueError(f"{file_path} 至少有一個通道是空的")

    # 截成一樣長
    ai0 = ai0[:n]
    ai1 = ai1[:n]

    # 從 wf_increment 算採樣率,wf_increment 是每個 sample 的時間間隔
    fs = None
    inc = getattr(ch_ai0, "properties", {}).get("wf_increment")
    if inc:
        fs = 1.0 / float(inc)
        
    # 如果採樣率無效，就用備援值
    if fs is None or not math.isfinite(fs) or fs <= 0:
        fs = fs_fallback

    return ai0, ai1, float(fs)


# ======================================
# 儲存合併後的 TDMS
# ======================================
def save_merged_tdms(
    output_path: Path,
    ai0: np.ndarray,
    ai1: np.ndarray,
    fs: float,
    source_files: list[Path],
    transition_folder: str,
) -> None:
    # 確保輸出資料夾存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 設定寫進 TDMS channel 的 properties：採樣間隔, 起始 offset, 合併自幾個原檔, 來自哪個 transition folder
    props = {
        "wf_increment": 1.0 / fs,
        "wf_start_offset": 0.0,
        "merged_from_count": len(source_files),
        "transition_folder": transition_folder,
    }

    # 建立 TDMS writer，寫入兩個 channel：signals/ai0 , signals/ai1
    with TdmsWriter(output_path) as writer:
        writer.write_segment(
            [
                ChannelObject("signals", "ai0", ai0.astype(np.float64), properties=props),
                ChannelObject("signals", "ai1", ai1.astype(np.float64), properties=props),
            ]
        )


# ======================================
# 合併單一 transition 資料夾
# ======================================
def merge_transition_folder(
    folder: Path,
    output_root: Path,
    pattern: str,
    recursive: bool,
    fs_fallback: float,
    transition_pad_sec: float,
) -> dict:
    # 抓出所有 TDMS 檔, 沒有就報錯
    files = get_tdms_files(folder, pattern=pattern, recursive=recursive)
    if not files:
        raise ValueError(f"{folder} 找不到任何 TDMS")

    # 分成正常檔和洩漏檔
    normal_files = [p for p in files if is_normal_file(p)]
    leak_files = [p for p in files if is_leak_file(p)]

    # 檢查有沒有檔名無法判斷類型,有的話直接報錯，要求你先改檔名
    unknown_files = [p for p in files if p not in normal_files and p not in leak_files]
    if unknown_files:
        names = ", ".join(p.name for p in unknown_files[:5])
        raise ValueError(
            f"{folder.name} 有無法判斷 normal / leak 的檔案: {names}。"
            "請先把檔名改成含 leak0 或 leak1。"
        )

    # 至少要有正常段和洩漏段
    if not normal_files:
        raise ValueError(f"{folder.name} 沒有 normal(leak0) 檔案")
    if not leak_files:
        raise ValueError(f"{folder.name} 沒有 leak(leak1) 檔案")

    # 合併順序固定是：所有 normal 在前, 所有 leak 在後
    ordered_files = normal_files + leak_files
    
    # 解析所有 leak 檔名中的條件
    leak_conditions = [parse_condition_from_filename(p.name) for p in leak_files]
    leak_condition_set = set(leak_conditions)
    
    # 如果同一個 transition 資料夾裡的 leak 檔條件不一致，就報錯
    if len(leak_condition_set) != 1:
        raise ValueError(
            f"{folder.name} 的 leak 檔案條件不一致: {leak_conditions}"
        )
    
    # 取 leak 條件
    leak_quantity, distance_cm, pressure_bar = leak_conditions[0]
    # 如果有欄位解析不到，也報錯
    if leak_quantity is None or distance_cm is None or pressure_bar is None:
        raise ValueError(
            f"{folder.name} 無法從 leak 檔名解析出 leak_quantity / distance_cm / pressure_bar"
        )
    
    # 解析所有 normal 檔的距離與壓力
    normal_conditions = [parse_condition_from_filename(p.name) for p in normal_files]
    normal_distance_pressure = {
        (dist_cm, p_bar)
        for _, dist_cm, p_bar in normal_conditions
    }
    
    # normal 檔的距離/壓力也必須一致
    if len(normal_distance_pressure) != 1:
        raise ValueError(
            f"{folder.name} 的 normal 檔案距離/壓力不一致: {normal_conditions}"
        )
    
    # 拿出唯一的一組 normal 條件
    normal_dist_cm, normal_p_bar = next(iter(normal_distance_pressure))
    
    # normal 和 leak 的距離/壓力必須一致,否則代表把不該拼在一起的資料放進同一個 transition 資料夾
    if normal_dist_cm != distance_cm or normal_p_bar != pressure_bar:
        raise ValueError(
            f"{folder.name} 的 normal 與 leak 距離/壓力不一致: "
            f"normal=({normal_dist_cm}cm, {normal_p_bar}bar), "
            f"leak=({distance_cm}cm, {pressure_bar}bar)"
        )

    # 累積波形與採樣率
    ai0_parts: list[np.ndarray] = []
    ai1_parts: list[np.ndarray] = []
    fs_values: list[float] = []
    normal_samples = 0

    # 逐檔讀 TDMS -> 把每一段 ai0 / ai1 加到清單 -> 收集採樣率 -> 如果這段屬於 normal，累加 sample 數
    for idx, file_path in enumerate(ordered_files):
        ai0, ai1, fs = read_two_channel_tdms(file_path, fs_fallback=fs_fallback)
        ai0_parts.append(ai0)
        ai1_parts.append(ai1)
        fs_values.append(fs)
        if idx < len(normal_files):
            normal_samples += len(ai0)

    # 檢查所有小檔採樣率一致
    fs_ref = fs_values[0]
    for fs in fs_values[1:]:
        if not np.isclose(fs, fs_ref, rtol=0, atol=1e-6):
            raise ValueError(
                f"{folder.name} 採樣率不一致: {fs_values}"
            )

    # 把所有 normal+leak 段串成一條大訊號
    merged_ai0 = np.concatenate(ai0_parts)
    merged_ai1 = np.concatenate(ai1_parts)

    # 切換邊界時間點(normal 段樣本總數 / 採樣率),如果設定transition 灰區：normal_end_sec 提前一點 + leak_start_sec 延後一點
    boundary_sec = normal_samples / (fs_ref + EPS)
    half_pad = max(transition_pad_sec, 0.0) / 2.0
    normal_end_sec = max(0.0, boundary_sec - half_pad)
    leak_start_sec = boundary_sec + half_pad
    onset_gt_sec = boundary_sec
    
    # 合併後整條訊號總長度
    total_duration_sec = len(merged_ai0) / (fs_ref + EPS)

    # 產生輸出檔名
    interval_name = folder.parent.name
    transition_name = folder.name
    
    output_filename = (
        f"{interval_name}__{transition_name}_"
        f"{leak_quantity}ml_{distance_cm}cm_{pressure_bar}bar_tran.tdms"
    )
    record_id = output_filename.replace(".tdms", "")    # 去掉副檔名後當作 record_id
    output_tdms = output_root / output_filename         # 輸出完整路徑
    
    # 寫出合併後 TDMS
    save_merged_tdms(
        output_path=output_tdms,
        ai0=merged_ai0,
        ai1=merged_ai1,
        fs=fs_ref,
        source_files=ordered_files,
        transition_folder=folder.name,
    )

    return {
        "record_id": record_id,
        "interval_name": interval_name,
        "transition_folder": transition_name,
        "transition_path": f"{interval_name}/{transition_name}",
        "output_tdms": output_tdms.name,
        "leak_quantity": leak_quantity,
        "distance_cm": distance_cm,
        "pressure_bar": pressure_bar,
        "n_normal_files": len(normal_files),
        "n_leak_files": len(leak_files),
        "fs": fs_ref,
        "normal_end_sec": normal_end_sec,
        "leak_start_sec": leak_start_sec,
        "onset_gt_sec": onset_gt_sec,
        "total_duration_sec": total_duration_sec,
        "notes": f"boundary from concatenation; normal_files={len(normal_files)}, leak_files={len(leak_files)}",
    }


# ======================================
# 把每個 transition 的 metadata 寫成 CSV
# ======================================
def write_transition_meta(rows: list[dict], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "transition_meta.csv"
    fieldnames = [
        "record_id",
        "transition_folder",
        "interval_name",
        "transition_path",
        "output_tdms",
        "n_normal_files",
        "n_leak_files",
        "fs",
        "normal_end_sec",
        "leak_start_sec",
        "onset_gt_sec",
        "total_duration_sec",
        "notes",
        "leak_quantity",
        "distance_cm",
        "pressure_bar",

    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ======================================
# 主流程
# ======================================
def main() -> None:
    # 設定輸入與輸出路徑
    input_root = Path("/Users/paul/Desktop/Final/transition_test_0615_0616_0617_0706")
    merged_base = input_root / "merged_output"
    output_root = merged_base / "transition_merged_tdms"
    
    # 設定其他參數：只抓 .tdms, 不遞迴, 採樣率備援值 1MHz，transition 前後保留 0.05 秒緩衝區
    pattern = "*.tdms"
    recursive = False
    fs_fallback = 1_000_000.0
    transition_pad_sec = 0.05
    
    # 找所有 transitionX 資料夾,找不到就報錯
    transition_dirs = find_transition_dirs(input_root)
    if not transition_dirs:
        raise ValueError(f"{input_root} 底下找不到 transition* 子資料夾")

    print("=" * 60)
    print("Merge Transition TDMS")
    print("=" * 60)
    print(f"input_root  = {input_root}")
    print(f"output_root = {output_root}")
    print(f"transition folders = {len(transition_dirs)}")

    meta_rows = []    # 成功合併的 metadata
    failed = []       # 失敗的資料夾紀錄

    # 對每個 transitionX 資料夾做合併:成功就加入 meta_rows, 失敗就記到 failed
    for folder in transition_dirs:
        print(f"\n[{folder.parent.name}/{folder.name}]")
        try:
            meta = merge_transition_folder(
                folder=folder,
                output_root=output_root,
                pattern=pattern,
                recursive=recursive,
                fs_fallback=fs_fallback,
                transition_pad_sec=transition_pad_sec,
            )
            meta_rows.append(meta)
            print(
                f"  merged -> {meta['output_tdms']} | "
                f"normal_end={meta['normal_end_sec']:.3f}s | "
                f"leak_start={meta['leak_start_sec']:.3f}s | "
                f"duration={meta['total_duration_sec']:.3f}s"
            )
        except Exception as exc:
            failed.append((folder.name, str(exc)))
            print(f"  failed: {exc}")
            
    # 如果至少有一筆成功，就寫出 transition_meta.csv
    if meta_rows:
        csv_path = write_transition_meta(meta_rows, output_root=merged_base)
        print(f"\n✅ transition_meta 已輸出: {csv_path}")

    # 列出失敗的 transition 資料夾
    if failed:
        print("\n⚠️ 以下資料夾合併失敗:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")

    print(f"\n完成: 成功 {len(meta_rows)} 個, 失敗 {len(failed)} 個")
    if not meta_rows:
        raise ValueError("沒有任何 transition merge 成功，停止後續特徵工程")


if __name__ == "__main__":
    main()
