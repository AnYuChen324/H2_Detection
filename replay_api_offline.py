#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_api_offline.py — 不接 DAQ，用錄好的 TDMS 照 API 的流程跑一遍（11/16）

做的事和 two_sensor_api_final2.inference_worker 一模一樣（直接呼叫 API 裡的函式）：
  每 interval_sec（5 s）取 capture_sec（2 s）一批
  → 引擎 reset → 串流特徵 + 模型 → 告警規則 + 批內暖機 → M/N 批確認
  → 疑似干擾（能量突升）→ combine_status 三色
輸出：
  1. 每段錄音的「燈號時間線」（綠 = 正常、橘 = 疑似干擾 / 待確認、紅 = 洩漏）
  2. 每批計算時間（模型那段 / 干擾偵測 / 合計），和 5 秒週期比

用法（Spyder 主控台；API 不用開）：
  %runfile /Users/paul/Desktop/Final/replay_api_offline.py --args "'資料夾1' '資料夾2' ..."
  資料夾底下有 TDMS 就可以（會依 session_id 分段，缺檔處自動切開）
  可加 --max-sec 60：每段只跑前 60 秒（省時間）

API 讀哪份模型 / baseline / 干擾參考檔，就用哪份（two_sensor_api_final2.py 第 261–263 行與 RESULT_DIR）
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import two_sensor_api_final2 as A                     # 只 import，不會啟動伺服器
from disturbance_flag_utils import (iter_tdms_sessions, iter_session_batches,
                                    detect_disturbance_in_batch, combine_status)
from alarm_rule_utils_v2 import BatchConfirmer

COLOR = {"normal": "綠", "leak": "紅"}               # 其他狀態都算橘

# 直接按 Spyder 的 Run（沒給 --args）時跑這幾段：都沒進過訓練（holdout / dev）
R = "/Users/paul/Desktop/AE Data/run_1115"
DEFAULT_ROOTS = [
    f"{R}/disturbance/tap_pipe/d1111_det1",            # 敲管
    f"{R}/disturbance/other/d1112_det1",               # 聲音
    f"{R}/disturbance/adjacent_vibration/d1112_det1",  # 走動
    f"{R}/disturbance/cable_pull/d1111_det3",          # 拉線
    f"{R}/step2/leak0/normal2",                        # 正常
    f"{R}/transition/d1113_t1",                        # 300 轉換（11/13 重裝）
    f"{R}/transition/d1113_t6",                        # 500 轉換（11/13 重裝）
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", help="要重播的資料夾（可多個；不給就用 DEFAULT_ROOTS）")
    ap.add_argument("--max-sec", type=float, default=None, help="每段只跑前 N 秒")
    ap.add_argument("--result-dir", default=None,
                    help="改用另一組設定（例如 lab1116_live/test_result）；模型 ID 沿用 API 第 262 行")
    args = ap.parse_args()
    if args.result_dir:
        import os
        rd = str(Path(args.result_dir))
        md = os.path.join(rd, "saved_models_cusum_v2", "xgb", A.MODEL_ID)
        A.RESULT_DIR, A.MODEL_DIR = rd, md
        A.BAND_CONFIG_PATH = os.path.join(rd, "band_config_step2.json")
        A.FEATURE_CONFIG_PATH = os.path.join(rd, "feature_config_micro.json")
        A.SITE_BASELINE_PATH = os.path.join(rd, "site_baseline.json")
        A.FEATURE_EXTRACTION_CONFIG_PATH = os.path.join(rd, "feature_extraction_config.json")
        A.DISTURBANCE_REFERENCE_PATH = os.path.join(rd, "disturbance_reference.json")
        A.POSTPROCESS_CONFIG_PATH = os.path.join(md, "postprocess_config.json")
        A.RUNTIME_FEATURE_SCHEMA_PATH = os.path.join(md, "runtime_feature_schema.json")
        A.MODEL_METADATA_JSON = os.path.join(md, "metadata.json")
        print(f"使用設定：{rd}（模型 {A.MODEL_ID}）")
    if not args.roots:
        args.roots = DEFAULT_ROOTS
        print("沒有指定資料夾，使用 DEFAULT_ROOTS：")
        for r in args.roots:
            print("  ", r)

    # ---------- 和 API 啟動時一樣載入 ----------
    A.load_runtime()
    st = A.state
    if not st.ready:
        raise RuntimeError("API 設定載入失敗（看上面的 ❌ 訊息）")
    S = st.sampling
    cap, itv = float(S["capture_sec"]), float(S["interval_sec"])
    fs = int(st.runtime_cfg["fs"])
    if abs(cap - round(cap)) > 1e-9 or abs(itv - round(itv)) > 1e-9:
        raise ValueError("這支工具假設 capture_sec / interval_sec 是整數秒")
    cap_i, itv_i = int(round(cap)), int(round(itv))
    print(f"\n重播設定：每 {itv:g}s 取 {cap:g}s；{S['confirm_n']} 批中 {S['confirm_m']} 批確認；fs={fs}")

    t_model, t_dist, t_total = [], [], []
    for root in args.roots:
        for sid, folder, files, _ in iter_tdms_sessions(root, allow_gaps=True):
            # 以 1 秒為單位讀，湊成「每 5 秒的前 2 秒」= API 的一批
            blocks = []
            confirmer = BatchConfirmer(S["confirm_m"], S["confirm_n"])
            line, notes = [], []
            for t1, a0, a1 in iter_session_batches(files, fs, 1.0):
                if args.max_sec is not None and t1 >= args.max_sec:
                    break
                k = int(round(t1))
                if k % itv_i >= cap_i:
                    continue
                blocks.append((a0, a1))
                if k % itv_i != cap_i - 1:
                    continue
                ai0 = np.concatenate([b[0] for b in blocks]); ai1 = np.concatenate([b[1] for b in blocks])
                blocks = []
                t_batch = k - cap_i + 1

                # ===== 以下照 inference_worker =====
                t0 = time.perf_counter()
                st.streaming_engine.reset()
                result = st.streaming_engine.process_chunk(ai0=ai0, ai1=ai1)
                alarm = A._run_alarm_postprocess(result["df_predictions"], st.runtime_cfg, st.model_bundle)
                batch_alarm = alarm["batch_has_alarm"]
                confirmed = confirmer.update(batch_alarm)
                t1c = time.perf_counter()
                dist = detect_disturbance_in_batch(ai0, ai1, st.disturbance_ref)
                t2c = time.perf_counter()
                status, reason = combine_status(confirmed=confirmed, batch_alarm=batch_alarm,
                                                burst=bool(dist and dist["burst"]),
                                                any_veto=int(alarm["n_veto_windows"]) > 0)
                t_model.append(t1c - t0); t_dist.append(t2c - t1c); t_total.append(t2c - t0)

                c = COLOR.get(status, "橘")
                line.append(c)
                if c != "綠":
                    bands = ",".join(str(b) for b in (dist or {}).get("burst_bands", []))
                    notes.append(f"{t_batch:>4d}s {c} {reason}"
                                 f"（模型最高 {alarm['score']:.2f}"
                                 + (f"，突升 {bands} {dist['max_burst_db']:.1f} dB" if dist and dist["burst"] else "")
                                 + "）")

            if not line:
                continue
            n = len(line)
            print(f"\n=== {Path(folder).name} / {sid}：{n} 批（{n * itv_i} 秒）"
                  f"  綠 {line.count('綠')}、橘 {line.count('橘')}、紅 {line.count('紅')}")
            print("   " + "".join(line))
            for s in notes[:40]:
                print("   " + s)
            if len(notes) > 40:
                print(f"   …還有 {len(notes) - 40} 批非綠燈")

    if t_total:
        tm, td, tt = (np.array(x) for x in (t_model, t_dist, t_total))
        print(f"\n===== 每批計算時間（{len(tt)} 批，每批 {cap:g}s 資料）=====")
        print(f"模型那段（特徵 + 模型 + 規則）：平均 {tm.mean():.2f}s、最大 {tm.max():.2f}s")
        print(f"疑似干擾偵測              ：平均 {td.mean():.2f}s、最大 {td.max():.2f}s")
        print(f"合計                      ：平均 {tt.mean():.2f}s、最大 {tt.max():.2f}s（週期 {itv:g}s）")
        print("✅ 小於週期，即時不會塞車" if tt.max() < itv else "❌ 超過週期，會越積越慢")


if __name__ == "__main__":
    main()
