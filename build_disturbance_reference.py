# -*- coding: utf-8 -*-
"""
build_disturbance_reference.py — 用正常資料建立「疑似干擾」偵測的參考檔

用哪些資料：和 site baseline 同一個 normal 資料夾（現場：安裝好、確認沒有洩漏、沒有人在旁邊動的正常資料）
輸出：disturbance_reference.json（API 和離線評估共用）

用法（Spyder）：
    %runfile /Users/paul/Desktop/Final/build_disturbance_reference.py --args "--normal-dir '<normal 資料夾>' --capture-sec 2.0 --output /Users/paul/Desktop/Final/lab1111_run/test_result/disturbance_reference.json"

--capture-sec 一定要和 API 每批擷取秒數相同（postprocess_config.json 的 sampling.capture_sec）。
"""
import argparse
import json
from pathlib import Path

from disturbance_flag_utils import (DEFAULT_GROUPS_KHZ, build_reference_from_energy,
                                    compute_group_energy_db, iter_session_batches, iter_tdms_sessions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-dir", required=True, nargs="+", help="正常資料夾（可多個）")
    ap.add_argument("--capture-sec", type=float, required=True, help="每批秒數（= API 每批擷取秒數）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fs", type=int, default=1_000_000)
    ap.add_argument("--window-sec", type=float, default=0.2)
    ap.add_argument("--burst-margin-db", type=float, default=3.0, help="突升門檻 = 正常最大晃動 + 這個餘裕（dB）")
    a = ap.parse_args()

    E_batches, paths = [], []
    for d in a.normal_dir:
        for sid, folder, files, _ in iter_tdms_sessions(d):
            paths += [str(f) for f in files]
            for _, x0, x1 in iter_session_batches(files, a.fs, a.capture_sec):
                E_batches.append(compute_group_energy_db(x0, x1, a.fs, a.window_sec, DEFAULT_GROUPS_KHZ))
    ref = build_reference_from_energy(
        E_batches, fs=a.fs, window_sec=a.window_sec, groups_khz=DEFAULT_GROUPS_KHZ,
        capture_sec=a.capture_sec, burst_margin_db=a.burst_margin_db,
        source={"normal_dirs": [str(Path(d)) for d in a.normal_dir], "n_tdms": len(paths),
                "first_tdms": paths[0] if paths else None, "last_tdms": paths[-1] if paths else None})
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    c = ref["calibration"]
    print(f"✅ 干擾參考檔：{out}")
    print(f"   正常資料 {c['n_batches']} 批（每批 {a.capture_sec} 秒、{c['n_windows']} 窗）")
    print(f"   正常批被標「突升」比例：{c['normal_burst_batch_rate']:.1%}（同一份資料，應該是 0；要看別的正常資料才準）")
    print(f"   正常批被標「背景偏高」比例：{c['normal_background_batch_rate']:.1%}")
    print("   突升門檻（dB，列 = ai0 / ai1，欄 = 頻帶）：")
    for ch, row, en in zip(ref["channels"], ref["burst_threshold_db"], ref["burst_enabled"]):
        print("    ", ch, " ".join(f"{g[0]}-{g[1]}k:{v:.1f}" if e else f"{g[0]}-{g[1]}k:不看"
                                   for g, v, e in zip(ref["groups_khz"], row, en)))


if __name__ == "__main__":
    main()
