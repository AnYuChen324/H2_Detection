#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_bands_with_transitions.py — Step2a 候選頻帶的「轉換資料檢查」

為什麼需要：
    Step2a 用「純正常 vs 純洩漏」排名頻帶。純資料是不同時間錄的，只要背景隨時間飄移
    （暖機、耦合劑、設備狀態），Step2a 就會把「時段差異」當成「洩漏頻帶」選進來
    （11/11 實測：選出的 8 個頻帶都是 ai0 100–310 kHz 隨時間一路下降約 1 dB，和洩漏無關）。
    轉換資料是同一段錄製裡「開閥前 vs 開閥後」，背景、安裝、時段都相同，只差有沒有洩漏，
    所以拿它來確認候選頻帶是不是真的跟著洩漏變。

做什麼：
    1. 讀 band_config_step2_candidate.json 的候選洩漏頻帶（leak_focused_bands）
    2. 讀轉換資料夾（v22 連續擷取：每個子資料夾 = 一段錄製，有 transition_dual_events_*.csv）
    3. 每個 0.2 秒窗去直流後做 FFT，算每個頻帶 × 通道（ai0 / ai1 / diff）的能量（和帶通後的能量相同，Parseval）
    4. 每段錄製：洩漏後（onset + gray_after 之後）中位數 − 洩漏前（onset − gray_before 之前）中位數，單位 dB
       對照：洩漏前自己的後半 − 前半（沒洩漏時一段錄製本身會飄多少）
    5. 判斷（每個頻帶 × 通道）：
       - 方向：由目標洩漏量各段差異的中位數決定（變大或變小都可以，但每段要同方向）
       - 一致性：目標洩漏量的每一段，差異都和方向一致、而且超過對照門檻（所有轉換對照 |差異| 的 95 百分位，至少 min_db）
       - 劑量：洩漏量越大，差異中位數越大（只提示，不擋）
    6. 輸出報表 CSV，並（可選）輸出只保留通過頻帶的 band config 草稿（schema_status 仍是 candidate，要人工確認）

限制：
    - 用 FFT 能量代表頻帶能量，不含 Step2a 的 Shape / Envelope 類特徵
    - 干擾（敲管、週期噪音）會讓差異變大或變小：預設用簡單的能量門檻排除干擾窗
    - 只檢查「能量有沒有跟著洩漏變」，不保證模型一定用得到

用法（Spyder）：
    %runfile /Users/paul/Desktop/Final/check_bands_with_transitions.py --args "--band-config <...>/band_config_step2_candidate.json --transition-root '<...>/transition' --levels 150 --output-dir <...>/band_check"
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy import fft as sfft

FS_DEFAULT = 1_000_000
WINDOW_SEC = 0.2
CHANNELS = ("ai0", "ai1", "diff")


# ============================================================
# 讀轉換資料（v22 連續擷取）
# ============================================================
def find_transition_sessions(root: Path):
    """每個有 transition_dual_events_*.csv 和 TDMS 的子資料夾 = 一段錄製。"""
    sessions = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        ev = sorted(d.glob("transition_dual_events_*.csv"))
        fs = sorted(d.glob("*.tdms"))
        if ev and fs:
            sessions.append((d, ev[0], fs))
    if not sessions:
        raise FileNotFoundError(f"{root} 底下找不到轉換資料（需要 transition_dual_events_*.csv 和 TDMS）")
    return sessions


def read_onset_sec(event_csv: Path, first_file_props: dict) -> float:
    """開閥時間（相對第一支檔開始，秒）= 事件 elapsed − 第一支檔 capture_start_elapsed。一段只允許一次 normal→leak。"""
    rows = list(csv.DictReader(open(event_csv, encoding="utf-8-sig")))
    leak = [float(r["event_elapsed_sec"]) for r in rows if r.get("to_state") == "leak"]
    if len(leak) != 1:
        raise ValueError(f"{event_csv}：normal→leak 事件數 = {len(leak)}，必須剛好 1 次")
    start = first_file_props.get("capture_start_elapsed_sec")
    if start is None:
        raise ValueError(f"{event_csv.parent}：第一支檔沒有 capture_start_elapsed_sec，無法對齊開閥時間")
    return leak[0] - float(start)


def check_continuous(files, props0, fs):
    """
    v22 連續擷取檢查，回傳「依取樣序號排序、去掉重複」的檔案清單。
    - 不是 continuous 模式（舊程式）→ 停止
    - 缺檔（序號差是整數支）→ 提示後繼續：時間軸用每支檔自己的 first_sample_index，缺的地方只是少幾個窗
    - 重複序號 → 只留第一支並提示
    - 序號差不是整數支（資料錯亂）→ 停止
    """
    if props0.get("acquisition_mode") != "continuous":
        raise ValueError(f"{files[0].parent}：不是連續擷取（acquisition_mode={props0.get('acquisition_mode')}），"
                         "可能是舊版收資料程式，時間軸會有空檔")
    items = []
    for f in files:
        i = TdmsFile.read_metadata(f).properties.get("first_sample_index")
        if i is None:
            raise ValueError(f"{f}：沒有 first_sample_index")
        items.append((int(i), f))
    items.sort(key=lambda x: x[0])
    n = int(props0.get("sample_count") or props0.get("sample_num"))
    uniq, n_dup = [], 0
    for i, f in items:
        if uniq and i == uniq[-1][0]:
            n_dup += 1
            continue
        uniq.append((i, f))
    d = np.diff([i for i, _ in uniq])
    if len(d) and np.any(d % n != 0):
        raise ValueError(f"{files[0].parent}：取樣序號差不是整數支（{n}），資料錯亂")
    n_missing = int((d // n - 1).sum()) if len(d) else 0
    if n_missing or n_dup:
        print(f"⚠️ {files[0].parent.name}：缺 {n_missing} 支、重複 {n_dup} 支（缺的地方略過，重複只留一支）")
    return [f for _, f in uniq]


def read_level(session_dir, p0):
    """洩漏量優先用 manifest 的 matched_volume_ml（人工修正過的副本以 manifest 為準），沒有 manifest 才用 TDMS 屬性。"""
    mf = sorted(session_dir.glob("transition_dual_manifest_*.csv"))
    if mf:
        v = pd.to_numeric(pd.read_csv(mf[0], encoding="utf-8-sig")["matched_volume_ml"], errors="coerce").dropna().unique()
        if len(v) != 1:
            raise ValueError(f"{mf[0]}：matched_volume_ml 不只一個值 {v}")
        return float(v[0])
    return float(p0.get("matched_volume_ml") or p0.get("volume_ml") or np.nan)


# ============================================================
# 能量計算
# ============================================================
def band_energy_windows(files, bands, fs):
    """回傳 (窗起點秒數, 能量 dB: 窗 × 通道 × 頻帶)。每支檔切成不重疊的 0.2 秒窗。"""
    w = int(round(WINDOW_SEC * fs))
    freqs = np.fft.rfftfreq(w, 1 / fs)
    idx = [(np.searchsorted(freqs, lo), np.searchsorted(freqs, hi)) for lo, hi in bands]
    t_all, e_all = [], []
    for f in files:
        t = TdmsFile.read(f)
        g = t["AE_Signals"]
        p = t.properties
        a0 = np.asarray(g["ai0"][:], dtype=np.float64)
        a1 = np.asarray(g["ai1"][:], dtype=np.float64)
        if len(a0) != len(a1):
            raise ValueError(f"{f.name}：ai0 / ai1 長度不同")
        t0 = p["first_sample_index"] / fs
        for k in range(len(a0) // w):
            s = slice(k * w, (k + 1) * w)
            x0 = a0[s] - a0[s].mean()
            x1 = a1[s] - a1[s].mean()
            row = []
            for x in (x0, x1, x1 - x0):
                P = np.abs(sfft.rfft(x)) ** 2
                c = np.concatenate([[0.0], np.cumsum(P)])
                row.append([c[b] - c[a] for a, b in idx])
            e_all.append(row)
            t_all.append(t0 + k * WINDOW_SEC)
    E = 10 * np.log10(np.asarray(e_all) + 1e-30)
    return np.asarray(t_all), E


def disturbance_mask(E_groups, rel_t, gray_before, gray_after, jump_db):
    """簡單干擾排除：任何通道的寬頻能量比這段自己的中位數高 jump_db 以上的窗（開閥附近不判斷，本來就排除）。"""
    med = np.median(E_groups, axis=0)
    bad = (E_groups > med + jump_db).any(axis=1)
    near_onset = (rel_t > -gray_before) & (rel_t < gray_after)
    return bad & ~near_onset


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Step2a 候選頻帶的轉換資料檢查")
    ap.add_argument("--band-config", required=True, help="band_config_step2_candidate.json")
    ap.add_argument("--transition-root", required=True, help="轉換資料夾（每個子資料夾一段錄製）")
    ap.add_argument("--levels", type=float, nargs="+", required=True, help="要求每段都要通過的洩漏量（例如 150）")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gray-before-sec", type=float, default=3.0, help="開閥前多少秒不算『洩漏前』")
    ap.add_argument("--gray-after-sec", type=float, default=20.0, help="開閥後多少秒不算『洩漏後』（閥門聲約 10–20 秒）")
    ap.add_argument("--min-db", type=float, default=0.1, help="對照門檻的下限（dB）")
    ap.add_argument("--null-percentile", type=float, default=95.0)
    ap.add_argument("--disturbance-jump-db", type=float, default=3.0, help="寬頻能量高出多少 dB 視為干擾窗")
    ap.add_argument("--fs", type=int, default=FS_DEFAULT)
    ap.add_argument("--write-filtered-config", action="store_true",
                    help="另存只保留通過頻帶的 band config 草稿（schema_status 仍是 candidate）")
    a = ap.parse_args()

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(a.band_config, encoding="utf-8"))
    cand = cfg.get("leak_focused_bands") or []
    if not cand:
        raise ValueError("band config 沒有 leak_focused_bands")
    names = [b["name"] for b in cand]
    bands = [(float(b["f_low"]), float(b["f_high"])) for b in cand]
    nyq = 0.5 * a.fs
    if any(not (0 < lo < hi < nyq) for lo, hi in bands):
        raise ValueError("候選頻帶有不合法的上下界（需 0 < f_low < f_high < Nyquist）")
    # 干擾判斷用的寬頻群組（和候選頻帶分開算）
    broad = [(1e3, 10e3), (10e3, 50e3), (50e3, 200e3), (200e3, 480e3)]

    rows = []
    for d, ev, files in find_transition_sessions(Path(a.transition_root)):
        p0 = TdmsFile.read_metadata(files[0]).properties
        files = check_continuous(files, p0, a.fs)                  # 依序號排序、去重複；缺檔只提示
        p0 = TdmsFile.read_metadata(files[0]).properties           # 開閥時間要用序號最前面那支檔
        level = read_level(d, p0)
        onset = read_onset_sec(ev, p0)
        t, E = band_energy_windows(files, bands + broad, a.fs)
        Eb, Eg = E[:, :, :len(bands)], E[:, :, len(bands):]
        rel = t - t[0] - onset
        dist = disturbance_mask(Eg.reshape(len(t), -1), rel, a.gray_before_sec, a.gray_after_sec, a.disturbance_jump_db)
        pre = (rel < -a.gray_before_sec) & ~dist
        post = (rel > a.gray_after_sec) & ~dist
        if pre.sum() < 20 or post.sum() < 20:
            print(f"⚠️ {d.name}：排除干擾後洩漏前 {pre.sum()} 窗、洩漏後 {post.sum()} 窗，太少，略過")
            continue
        pre_idx = np.where(pre)[0]; h = len(pre_idx) // 2
        delta = np.median(Eb[post], 0) - np.median(Eb[pre], 0)                       # 通道 × 頻帶
        null = np.median(Eb[pre_idx[h:]], 0) - np.median(Eb[pre_idx[:h]], 0)
        for ci, ch in enumerate(CHANNELS):
            for bi_, n in enumerate(names):
                rows.append(dict(session=d.name, level_ml=level, onset_sec=round(onset, 2), band=n, channel=ch,
                                 delta_db=delta[ci, bi_], null_db=null[ci, bi_],
                                 n_pre=int(pre.sum()), n_post=int(post.sum()), n_disturb=int(dist.sum())))
        print(f"✅ {d.name}：{level:g} ml，開閥在第 {onset:.1f} 秒，干擾窗 {dist.sum()}，洩漏前 / 後 {pre.sum()} / {post.sum()} 窗")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("沒有可用的轉換段")
    df.to_csv(out / "band_check_per_session.csv", index=False, encoding="utf-8-sig")

    # 對照門檻：每個頻帶 × 通道，所有轉換段對照 |差異| 的百分位
    thr = (df.groupby(["band", "channel"])["null_db"]
             .apply(lambda s: max(np.percentile(np.abs(s), a.null_percentile), a.min_db)).rename("threshold_db"))
    df = df.join(thr, on=["band", "channel"])
    # 方向：每個頻帶 × 通道，用目標洩漏量各段差異的中位數決定（正 = 洩漏變大）
    tgt_med = df[df["level_ml"].isin(a.levels)].groupby(["band", "channel"])["delta_db"].median()
    sign = np.sign(tgt_med).replace(0, 1.0).rename("sign")
    df = df.join(sign, on=["band", "channel"])
    df["signed_delta"] = df["delta_db"] * df["sign"]
    df["pass"] = df["signed_delta"] > df["threshold_db"]

    levels = sorted(df["level_ml"].dropna().unique())
    summ = []
    for (band, ch), g in df.groupby(["band", "channel"], sort=False):
        r = dict(band=band, channel=ch, direction=("增加" if g["sign"].iloc[0] > 0 else "減少"),
                 threshold_db=round(g["threshold_db"].iloc[0], 3))
        for lv in levels:
            gl = g[g["level_ml"] == lv]
            r[f"{lv:g}ml_median_db"] = round(gl["signed_delta"].median(), 3) if len(gl) else np.nan
            r[f"{lv:g}ml_pass"] = f"{int(gl['pass'].sum())}/{len(gl)}"
        tg = g[g["level_ml"].isin(a.levels)]
        r["target_all_pass"] = bool(len(tg) > 0 and tg["pass"].all())
        meds = [r[f"{lv:g}ml_median_db"] for lv in levels]
        r["dose_ok"] = bool(len(levels) < 2 or all(x <= y + 1e-9 for x, y in zip(meds, meds[1:])))
        summ.append(r)
    S = pd.DataFrame(summ)
    S["band_pass"] = S.groupby("band")["target_all_pass"].transform("any")
    S.to_csv(out / "band_check_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== 候選頻帶 × 通道：轉換資料檢查（dB，已依方向換號：正 = 和目標洩漏量的變化方向一致）===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(S.to_string(index=False))
    ok = S.loc[S["target_all_pass"], ["band", "channel"]].values.tolist()
    bad = sorted(set(names) - {b for b, _ in ok})
    print(f"\n✅ 通過（目標洩漏量 {a.levels} 每段都超過對照）：{ok if ok else '無'}")
    print(f"❌ 沒通過（推測是時段 / 背景差異，不是洩漏）：{bad if bad else '無'}")
    warn = S[S["target_all_pass"] & ~S["dose_ok"]]
    if len(warn):
        print(f"⚠️ 通過但洩漏量越大差異沒有越大（再確認）：{warn[['band','channel']].values.tolist()}")

    if a.write_filtered_config:
        keep = {b for b, _ in ok}
        new = dict(cfg); new["leak_focused_bands"] = [b for b in cand if b["name"] in keep]
        new["schema_status"] = "candidate"
        new.setdefault("notes", []); new["notes"] = list(new["notes"]) + [
            f"check_bands_with_transitions：保留 {sorted(keep)}，移除 {bad}（轉換資料沒有一致變化）"]
        fp = out / "band_config_step2_candidate_transition_checked.json"
        json.dump(new, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"📝 已另存草稿：{fp}（schema_status 仍是 candidate，確認後再改 final）")
    print(f"\n報表：{out/'band_check_summary.csv'}、{out/'band_check_per_session.csv'}")


if __name__ == "__main__":
    main()
