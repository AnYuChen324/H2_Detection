#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime 串流 vs 離線：週期取樣版逐批對齊驗收
==========================================

API 現場：每 interval_sec 擷取 capture_sec → 引擎 reset → 模型 → 告警規則 → 批內暖機 → BatchConfirmer(M/N)
離線評估：整段特徵 → simulate_periodic_capture → 告警規則 → 批內暖機 → BatchConfirmer(M/N)

用同一段連續 TDMS，兩邊各跑一次，逐批比對：
  1. 每批窗數、窗起點一致
  2. 模型機率最大差（應 < 1e-6）
  3. 每窗告警（套規則 + 暖機後）完全一致
  4. 每批有沒有報警、M/N 第一次確認在第幾批，完全一致
  5. 每批（capture_sec 秒資料）的處理時間，對照週期 interval_sec

只改最上面的路徑；取樣參數一律讀 postprocess_config.json 的 sampling（和 API 相同）。
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

# ============================================================
# 路徑（改這裡）
# ============================================================
RESULT_DIR = Path("/Users/paul/Desktop/Final/sim_e2e_v2/test_result")
EVAL_CSV = Path("/Users/paul/Desktop/Final/sim_e2e_v2/transition_fe/transition_rule_features_eval.csv")
SESSION_ROOT = Path("/Users/paul/Desktop/Final/sim_e2e_v2/data/transition")
SPLIT = "holdout"

import runtime_inference_v2 as RI
from runtime_streaming import StreamingRuntimeEngine
from shared_feature_utils import build_continuous_segments, _read_tdms_ai0_ai1_raw
from alarm_rule_utils_v2 import (apply_alarm_method_by_record, simulate_periodic_capture,
                                 apply_batch_warmup, BatchConfirmer)

models = sorted((RESULT_DIR / "saved_models_cusum_v2" / "xgb").glob("xgb_*"))
if len(models) != 1:
    raise RuntimeError(f"模型資料夾應該剛好 1 個：{[m.name for m in models]}")
M = models[0]

cfg = RI.load_runtime_configs(
    band_config_path=RESULT_DIR / "band_config_step2.json",
    feature_extraction_config_path=RESULT_DIR / "feature_extraction_config.json",
    feature_config_path=RESULT_DIR / "feature_config_micro.json",
    postprocess_config_path=M / "postprocess_config.json",
    runtime_feature_schema_path=M / "runtime_feature_schema.json",
    site_baseline_path=RESULT_DIR / "site_baseline.json",
)
bundle = RI.load_model_bundle(M / "metadata.json")
feat_names = list(bundle["feature_names"])
fs = float(cfg["fs"])
wt = cfg["window_threshold"]
win = float(cfg["window_sec"])

pp = json.loads((M / "postprocess_config.json").read_text(encoding="utf-8"))
offline_rule = pp["baseline_rule_settings"]
S = pp.get("sampling")
if not S or S.get("mode") != "periodic":
    raise ValueError("postprocess_config.json 沒有 periodic sampling，請重跑干擾推論 select / evaluate")
cap, itv, wu, cm, cn = S["capture_sec"], S["interval_sec"], S["warmup_sec"], S["confirm_m"], S["confirm_n"]
print(f"取樣：每 {itv}s 擷取 {cap}s、暖機 {wu}s、{cn} 批中 {cm} 批確認；offsets {S['offsets_sec']}")

# API 的規則設定（和 two_sensor_api_final2._run_alarm_postprocess 同一組鍵）
api_rule = {k: cfg[k] for k in ("alarm_threshold", "smooth_window", "consecutive", "n_hits", "m_window",
                                "high_threshold", "low_threshold", "min_leak_duration_sec", "use_veto",
                                "veto_column", "veto_threshold", "veto_rules")}
api_rule["method"] = cfg["alarm_method"]

seg_index = {}
for mf in SESSION_ROOT.rglob("*manifest*.csv"):
    segs, _ = build_continuous_segments(mf, mf.parent, fs)
    for s in segs:
        seg_index[s["record_id"]] = s


def first_confirm(flags):
    conf = BatchConfirmer(cm, cn)
    for k, f in flags:
        if conf.update(f):
            return k
    return None


df_eval = pd.read_csv(EVAL_CSV)
rows, calc_times = [], []
for rid in sorted(df_eval.loc[df_eval["split"] == SPLIT, "record_id"].unique()):
    off = df_eval[df_eval["record_id"] == rid].sort_values("t_start").reset_index(drop=True)
    off = off.assign(pred_proba=bundle["model"].predict_proba(off[feat_names].to_numpy(float))[:, 1])
    seg = seg_index[rid]
    raws = [_read_tdms_ai0_ai1_raw(fp, int(cfg["max_points"]), fs) for fp in seg["filepaths"]]
    a0 = np.concatenate([r[0] for r in raws]); a1 = np.concatenate([r[1] for r in raws])
    t0_off = float(off["t_start"].min())
    eng = StreamingRuntimeEngine(runtime_cfg=cfg, model_bundle=bundle, warmup_sec=0.0, record_id=rid)

    for offset in S["offsets_sec"]:
        # ---------- 離線 ----------
        df_b = simulate_periodic_capture(off, cap, itv, offset, win)
        r_off = apply_batch_warmup(apply_alarm_method_by_record(df_b, offline_rule, wt), wu)

        # ---------- runtime（照 API：每批 reset） ----------
        n_ok_win = n_ok_alarm = n_batches = 0
        p_diff = 0.0
        flags_off, flags_rt = [], []
        for k, b_off in r_off.groupby("batch_index"):
            b_off = b_off.sort_values("t_start")
            s = int(round((t0_off + k * itv + offset) * fs))
            e = s + int(round(cap * fs))
            if e > len(a0):
                continue                                      # 段尾不滿一批：API 不會出現，跳過
            eng.reset()
            t_c = time.perf_counter()
            out = eng.process_chunk(a0[s:e], a1[s:e])
            calc_times.append(time.perf_counter() - t_c)
            rt = out["df_features"][["t_start", "t_ref"]].copy()
            rt["pred_proba"] = out["df_predictions"][cfg["score_col"]].to_numpy(float)
            for c in api_rule.get("veto_rules") or []:
                rt[c["col"]] = out["df_features"][c["col"]].to_numpy(float)
            rt["record_id"] = f"{rid}__b{k}"
            r_rt = apply_batch_warmup(apply_alarm_method_by_record(rt, api_rule, wt), wu)

            n_batches += 1
            same_win = (len(r_rt) == len(b_off) and
                        np.allclose(r_rt["t_start"].to_numpy(float) + (t0_off + k * itv + offset),
                                    b_off["t_start"].to_numpy(float), atol=1e-6))
            n_ok_win += int(same_win)
            if same_win:
                p_diff = max(p_diff, float(np.max(np.abs(r_rt["pred_proba"].to_numpy(float)
                                                          - b_off["pred_proba"].to_numpy(float)))))
                n_ok_alarm += int(np.array_equal(r_rt["detected_rule"].to_numpy(int),
                                                 b_off["detected_rule"].to_numpy(int)))
            flags_off.append((k, bool((b_off["detected_rule"] == 1).any())))
            flags_rt.append((k, bool((r_rt["detected_rule"] == 1).any())))

        c_off, c_rt = first_confirm(flags_off), first_confirm(flags_rt)
        rows.append(dict(record_id=rid[-20:], offset_s=offset, 批數=n_batches,
                         窗一致批數=n_ok_win, 告警一致批數=n_ok_alarm, 機率最大差=p_diff,
                         批報警一致=[f for _, f in flags_off] == [f for _, f in flags_rt],
                         確認批_離線=c_off, 確認批_runtime=c_rt))

res = pd.DataFrame(rows)
print("\n===== 週期模式 runtime vs 離線 =====")
print(res.to_string(index=False))
ok = bool((res["窗一致批數"] == res["批數"]).all() and (res["告警一致批數"] == res["批數"]).all()
          and (res["機率最大差"] < 1e-6).all() and res["批報警一致"].all()
          and (res["確認批_離線"].astype(str) == res["確認批_runtime"].astype(str)).all())
print(f"\n{'✅ 全部一致' if ok else '❌ 有不一致，請把上表貼給 Claude'}")
ct = np.array(calc_times[1:]) if len(calc_times) > 1 else np.array(calc_times)
print(f"ℹ️ 每批 {cap}s 資料處理：平均 {ct.mean():.2f}s、最大 {ct.max():.2f}s（週期 {itv}s；小於週期就不會塞車）")
