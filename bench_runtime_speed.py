#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_runtime_speed.py — 不接 DAQ，量「每批 2 秒資料」在這台電腦要算多久（11/16）

用 bench_data/ 裡 8 支 11/16 正常 TDMS（4 秒）切成 2 秒一批，照 API 的流程跑：
  引擎 reset → 特徵（濾波 / 包絡 / 頻譜）→ 模型
每個平行數各跑 REPEAT 次，印平均與最大；最後印單執行緒時各步驟佔多少時間。

用法（Anaconda Prompt / 終端機，在這支程式所在的資料夾）：
  python bench_runtime_speed.py               ← 試 1、2、4、6、8、12 條
  python bench_runtime_speed.py 4 6 8         ← 只試指定的平行數
"""
import os
import sys
import time
import cProfile
import pstats
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import runtime_inference_v2 as RI
from runtime_streaming import StreamingRuntimeEngine
from disturbance_flag_utils import iter_tdms_sessions, iter_session_batches

R = HERE / "lab1115_run" / "test_result"
M = R / "saved_models_cusum_v2" / "xgb" / "xgb_20261116_105324"
REPEAT = 4

cfg = RI.load_runtime_configs(
    band_config_path=R / "band_config_step2.json",
    feature_extraction_config_path=R / "feature_extraction_config.json",
    feature_config_path=R / "feature_config_micro.json",
    postprocess_config_path=M / "postprocess_config.json",
    runtime_feature_schema_path=M / "runtime_feature_schema.json",
    site_baseline_path=R / "site_baseline.json",
)
bundle = RI.load_model_bundle(M / "metadata.json")
fs = int(cfg["fs"])
sid, folder, files, _ = iter_tdms_sessions(HERE / "bench_data")[0]
batches = [(a0, a1) for _, a0, a1 in iter_session_batches(files, fs, 2.0)]
print(f"\n電腦：{os.cpu_count()} 個邏輯處理器；測試資料 {len(batches)} 批 × 2 秒")

njobs_list = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 6, 8, 12]
for nj in njobs_list:
    eng = StreamingRuntimeEngine(runtime_cfg=cfg, model_bundle=bundle, warmup_sec=0.0, n_jobs=nj)
    eng.reset(); eng.process_chunk(*batches[0])           # 暖機（第一次會慢）
    ts = []
    for k in range(REPEAT):
        a0, a1 = batches[k % len(batches)]
        eng.reset()
        t = time.perf_counter()
        eng.process_chunk(a0, a1)
        ts.append(time.perf_counter() - t)
    print(f"平行數 {nj:>2d}：每批平均 {np.mean(ts):.2f} 秒、最大 {np.max(ts):.2f} 秒")

# 單執行緒時，時間花在哪
eng = StreamingRuntimeEngine(runtime_cfg=cfg, model_bundle=bundle, warmup_sec=0.0, n_jobs=1)
eng.reset(); eng.process_chunk(*batches[0])
pr = cProfile.Profile(); eng.reset(); pr.enable(); eng.process_chunk(*batches[0]); pr.disable()
st = pstats.Stats(pr); total = st.total_tt
groups = {"濾波 sosfilt": ["sosfilt"], "包絡 hilbert（FFT）": ["c2c", "hilbert"],
          "頻譜 welch": ["_spectral_helper", "r2c", "_fft_helper", "detrend"]}
print(f"\n單執行緒一批 {total:.2f} 秒，其中：")
for name, keys in groups.items():
    tt = sum(v[2] for k, v in st.stats.items() if any(x in k[2] for x in keys))
    print(f"  {name:<20s} {tt:.2f} 秒（{100 * tt / total:.0f}%）")
