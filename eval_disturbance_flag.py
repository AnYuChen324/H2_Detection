# -*- coding: utf-8 -*-
"""
eval_disturbance_flag.py — 離線驗收「疑似干擾」偵測（和 API 用同一個函式、同一份參考檔）

驗收三件事：
    1. 正常資料（--normal-dir）：被標「突升」的批要很少（目標 ≤ 1–2%）
    2. 干擾資料（--disturbance-root，需要 disturbance_events_*.csv）：每個干擾事件至少有一批被標到
    3. 轉換資料（--transition-root，需要 transition_dual_events_*.csv）：
       - 開閥前：和正常一樣要很少
       - 洩漏穩定後（開閥 + gray 秒之後）：不能一直被標成干擾（只要不是持續標，就不會誤導）
       - 開閥那一批：記錄有沒有被標（大洩漏可能會，屬於已知限制）

批的切法：每段錄製從頭開始、背對背切成每批 capture_sec 秒（= 參考檔的 capture_sec）。
實際 API 是每 interval 秒擷取一次，干擾可能剛好落在沒錄的空檔；這裡評估的是「有錄到時抓不抓得到」。

用法（Spyder）：
    %runfile /Users/paul/Desktop/Final/eval_disturbance_flag.py --args "--reference <...>/disturbance_reference.json --normal-dir '<...>/normal2' '<...>/normal3' --disturbance-root '<...>/run/disturbance' --transition-root '<...>/transition' --output-dir <...>/disturbance_flag_eval"
"""
import argparse
import csv
from pathlib import Path

import pandas as pd
from nptdms import TdmsFile

from disturbance_flag_utils import (detect_disturbance_in_batch, iter_session_batches, iter_tdms_sessions,
                                    load_disturbance_reference)


def run_batches(files, ref):
    """缺檔切段後，太短（不到一批）的段會回傳空表，呼叫端略過。"""
    rows = []
    for t0, x0, x1 in iter_session_batches(files, ref["fs"], ref["capture_sec"]):
        r = detect_disturbance_in_batch(x0, x1, ref)
        rows.append({"t0": t0, "t1": t0 + ref["capture_sec"], "burst": r["burst"],
                     "background_high": r["background_high"], "n_burst_windows": r["n_burst_windows"],
                     "max_burst_db": r["max_burst_db"], "max_background_db": r["max_background_db"],
                     "burst_bands": "; ".join(r["burst_bands"]), "background_bands": "; ".join(r["background_bands"])})
    return pd.DataFrame(rows)


def capture_start_elapsed(folder, files):
    p = TdmsFile.read_metadata(files[0]).properties
    v = p.get("capture_start_elapsed_sec")
    if v is None:
        m = sorted(folder.glob("dual_session_manifest_*.csv"))
        if not m:
            raise ValueError(f"{folder}：找不到 capture_start_elapsed_sec，無法對齊事件時間")
        v = list(csv.DictReader(open(m[0], encoding="utf-8-sig")))[0]["capture_start_elapsed_sec"]
    return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--normal-dir", nargs="*", default=[])
    ap.add_argument("--disturbance-root")
    ap.add_argument("--transition-root")
    ap.add_argument("--transition-gray-after-sec", type=float, default=20.0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--allow-gaps", action="store_true", help="有缺檔時切段處理（不加：缺檔就停）")
    a = ap.parse_args()
    ref = load_disturbance_reference(a.reference)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], []

    # 1. 正常
    for d in a.normal_dir:
        for sid, folder, files, _ in iter_tdms_sessions(d, allow_gaps=a.allow_gaps):
            df = run_batches(files, ref)
            if df.empty:
                continue
            df["kind"] = "normal"; df["session"] = sid; df["folder"] = str(folder)
            all_rows.append(df)
            summary.append({"kind": "normal", "session": sid, "folder": folder.name, "n_batches": len(df),
                            "burst_rate": df["burst"].mean(), "background_rate": df["background_high"].mean()})

    # 2. 干擾
    if a.disturbance_root:
        for sid, folder, files, _ in iter_tdms_sessions(a.disturbance_root, allow_gaps=a.allow_gaps):
            df = run_batches(files, ref)
            if df.empty:
                continue
            df["kind"] = "disturbance"; df["session"] = sid; df["folder"] = str(folder)
            ev_files = sorted(folder.glob("*disturbance_events_*.csv"))
            if not ev_files:
                raise ValueError(f"{folder}：干擾資料沒有 disturbance_events_*.csv（或 transition_dual_disturbance_events_*.csv）")
            start = capture_start_elapsed(folder, files)
            evs = [r for r in csv.DictReader(open(ev_files[0], encoding="utf-8-sig")) if r.get("session_id", sid) == sid]
            df["in_event"] = False
            for e in evs:
                s = float(e["start_elapsed_sec"]) - start; t = float(e["end_elapsed_sec"]) - start
                hit_mask = (df["t1"] > s - 0.5) & (df["t0"] < t + 0.5)     # 事件前後各 0.5 秒容許時間誤差
                df.loc[hit_mask, "in_event"] = True
                summary.append({"kind": "disturbance_event", "session": sid, "folder": f"{folder.parent.name}/{folder.name}",
                                "type": e.get("disturbance_type"), "event_sec": f"{s:.1f}–{t:.1f}",
                                "n_batches": int(hit_mask.sum()), "hit": bool(df.loc[hit_mask, "burst"].any()),
                                "max_burst_db": df.loc[hit_mask, "max_burst_db"].max() if hit_mask.any() else None})
            outside = ~df["in_event"]
            summary.append({"kind": "disturbance_outside_event", "session": sid, "folder": f"{folder.parent.name}/{folder.name}",
                            "n_batches": int(outside.sum()), "burst_rate": df.loc[outside, "burst"].mean() if outside.any() else None})
            all_rows.append(df)

    # 3. 轉換
    if a.transition_root:
        for sid, folder, files, _ in iter_tdms_sessions(a.transition_root, allow_gaps=a.allow_gaps):
            ev = sorted(folder.glob("transition_dual_events_*.csv"))
            if not ev:
                continue
            rows = list(csv.DictReader(open(ev[0], encoding="utf-8-sig")))
            leak = [float(r["event_elapsed_sec"]) for r in rows if r.get("to_state") == "leak"]
            if len(leak) != 1:
                raise ValueError(f"{folder}：normal→leak 事件數 = {len(leak)}，必須剛好 1 次")
            onset = leak[0] - capture_start_elapsed(folder, files)
            df = run_batches(files, ref)
            if df.empty:
                continue
            df["kind"] = "transition"; df["session"] = sid; df["folder"] = str(folder)
            pre = df["t1"] <= onset; at = (df["t0"] <= onset) & (df["t1"] > onset)
            post = df["t0"] >= onset + a.transition_gray_after_sec
            summary.append({"kind": "transition", "session": sid, "folder": folder.name, "onset_sec": round(onset, 2),
                            "pre_burst_rate": df.loc[pre, "burst"].mean(), "onset_batch_burst": bool(df.loc[at, "burst"].any()),
                            "post_burst_rate": df.loc[post, "burst"].mean(),
                            "post_background_rate": df.loc[post, "background_high"].mean(), "n_batches": len(df)})
            all_rows.append(df)

    pd.concat(all_rows, ignore_index=True).to_csv(out / "disturbance_flag_batches.csv", index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summary); sm.to_csv(out / "disturbance_flag_summary.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n=== 1. 正常資料：被標「突升」的批比例（越低越好）===")
    n = sm[sm.kind == "normal"]
    if len(n): print(n[["folder", "n_batches", "burst_rate", "background_rate"]].to_string(index=False))
    e = sm[sm.kind == "disturbance_event"].copy()
    if len(e): e["hit"] = e["hit"].astype(bool)
    if len(e):
        print(f"\n=== 2. 干擾事件：抓到 {int(e['hit'].sum())} / {len(e)} ===")
        print(e.groupby("type")["hit"].agg(["sum", "count"]).rename(columns={"sum": "抓到", "count": "事件數"}).to_string())
        miss = e[~e["hit"]]
        if len(miss): print("沒抓到：\n" + miss[["folder", "type", "event_sec", "max_burst_db"]].to_string(index=False))
        o = sm[sm.kind == "disturbance_outside_event"]
        print(f"干擾資料中「事件以外」的批被標比例（中位數）：{o['burst_rate'].median():.1%}")
    t = sm[sm.kind == "transition"]
    if len(t):
        print("\n=== 3. 轉換資料（開閥前 / 開閥那批 / 洩漏穩定後）===")
        print(t[["folder", "pre_burst_rate", "onset_batch_burst", "post_burst_rate", "post_background_rate"]].to_string(index=False))
    print(f"\n✅ 明細：{out / 'disturbance_flag_batches.csv'}\n✅ 摘要：{out / 'disturbance_flag_summary.csv'}")


if __name__ == "__main__":
    main()
