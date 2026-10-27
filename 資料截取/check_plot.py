# -*- coding: utf-8 -*-
"""
試錄檢查：時間標注有沒有對上訊號、通道有沒有接反、訊號正不正常。

用法（在 Spyder 改下面 FOLDER 後直接執行）：
  FOLDER 填一次輸出的資料夾（裡面有 manifest 和 events CSV），例如
      /Users/paul/Desktop/模擬測試/transition1
  執行後會：
    1. 畫一張圖（同時存成 PNG 放在該資料夾）：
       上：ai0 能量   中：ai1 能量   下：ai1 − ai0（dB）
       紅色實線 = 標記 leak、藍色虛線 = 標記 normal、灰色區塊 = 標記的干擾區間
    2. 印出每個標記和「訊號真正變化的時間」差多少秒
    3. 檢查削峰、兩通道是否一模一樣、兩通道振幅比

時間軸：manifest 的 capture_start_elapsed_sec 和標記的 elapsed_sec
都從同一個 session 起點算起，所以可以直接疊在一起比。

也可以檢查舊版（v21）資料，但要注意：
  - 舊版檔案之間有 0.55～0.70 秒空檔，圖上會斷線；標記剛好落在空檔裡時，
    只能抓到下一支檔案的開頭，落差會多算最多約 0.7 秒
  - 舊版 capture_start_elapsed_sec 比真正開始取樣早數十～數百毫秒，
    落差會整體偏大一點
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from nptdms import TdmsFile

# ===================== 設定 =====================
# 干擾
#FOLDER = r"/Users/paul/Desktop/Final/disturbance_test_0624_0702_0707/adjacent_vibration/detection10"

# 轉換
FOLDER = r"/Users/paul/Desktop/Final/transition_test_0615_0616_0617_0706/transition188"
WINDOW_SEC = 0.05          # 能量計算的時間解析度（50 ms 一點）
BAND_HZ = (20_000, 450_000)  # 只看 AE 頻段，去掉低頻流體/電源雜訊；設 None 不濾波
FULL_SCALE_V = 10.0        # DAQ 量程 ±10 V
SEARCH_SEC = 5.0           # 在標記前後幾秒內找訊號變化
# ================================================

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang TC", "Heiti TC",
                                   "Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------- 讀檔 ----------
def find_newest(folder, patterns):
    hits = []
    for pat in patterns:
        hits += list(Path(folder).rglob(pat))
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def load_manifest(folder):
    path = find_newest(folder, ["dual_session_manifest_*.csv", "transition_dual_manifest_*.csv"])
    if path is None:
        raise FileNotFoundError(f"{folder} 裡找不到 manifest，請確認已按過輸出")
    df = pd.read_csv(path, encoding="utf-8-sig")
    file_col = next(c for c in ("dual_file", "merged_file", "ai0_file") if c in df.columns)
    df["data_file"] = [resolve(p, path.parent) for p in df[file_col]]
    df = df.sort_values("capture_start_elapsed_sec").reset_index(drop=True)
    print(f"manifest：{path.name}（{len(df)} 支）")
    return df, path


def resolve(p, base):
    """manifest 裡的路徑可能是錄製那台電腦（Windows）的路徑；找不到就只取檔名，到 manifest 所在資料夾找"""
    p_str = str(p)
    p = Path(p_str)
    if p.exists():
        return p
    name = p_str.replace("\\", "/").split("/")[-1]   # Windows 的 \ 和 Mac 的 / 都當成分隔符號
    hits = list(Path(base).rglob(name))
    return hits[0] if hits else p


def load_events(folder):
    """回傳 (transition 標記 DataFrame, 干擾區間 DataFrame)；沒有就是空的"""
    folder = Path(folder)
    trans_path = find_newest(folder, ["transition_dual_events_*.csv", "transition_events_*.csv"])
    dist_path = find_newest(folder, ["*disturbance_events_*.csv"])
    trans = pd.read_csv(trans_path, encoding="utf-8-sig") if trans_path else pd.DataFrame()
    dist = pd.read_csv(dist_path, encoding="utf-8-sig") if dist_path else pd.DataFrame()
    # 只保留真的有「切換」的標記（normal → normal 這種重複按的不算）
    if not trans.empty:
        trans = trans[trans["from_state"] != trans["to_state"]].reset_index(drop=True)
    print(f"transition 標記：{len(trans)} 個；干擾區間：{len(dist)} 個")
    return trans, dist


def read_pair(path):
    tdms = TdmsFile.read(str(path))
    out = {}
    for group in tdms.groups():
        for ch in group.channels():
            if ch.name in ("ai0", "ai1"):
                out[ch.name] = np.asarray(ch[:], dtype=np.float64)
    if set(out) != {"ai0", "ai1"}:
        raise ValueError(f"{path.name} 缺少 ai0 / ai1")
    return out["ai0"], out["ai1"]


# ---------- 每支檔案 → 每 50 ms 一點的能量 ----------
def compute_timeline(df):
    fs = float(df["sample_rate"].iloc[0])
    win = int(round(WINDOW_SEC * fs))
    sos = None
    if BAND_HZ:
        sos = butter(4, [BAND_HZ[0], min(BAND_HZ[1], 0.49 * fs)], btype="bandpass", fs=fs, output="sos")

    t_all, rms0_all, rms1_all = [], [], []
    clip = {"ai0": 0, "ai1": 0}
    peak = {"ai0": 0.0, "ai1": 0.0}
    corr_list = []
    n_total = 0

    for _, row in df.iterrows():
        x0, x1 = read_pair(row["data_file"])
        n_total += len(x0)
        for name, x in (("ai0", x0), ("ai1", x1)):
            clip[name] += int(np.sum(np.abs(x) >= 0.98 * FULL_SCALE_V))
            peak[name] = max(peak[name], float(np.max(np.abs(x))))
        corr_list.append(float(np.corrcoef(x0, x1)[0, 1]))

        if sos is not None:
            x0 = sosfiltfilt(sos, x0)
            x1 = sosfiltfilt(sos, x1)
        n_win = len(x0) // win
        r0 = np.sqrt(np.mean(x0[: n_win * win].reshape(n_win, win) ** 2, axis=1))
        r1 = np.sqrt(np.mean(x1[: n_win * win].reshape(n_win, win) ** 2, axis=1))
        t0 = float(row["capture_start_elapsed_sec"])
        t_all.append(t0 + (np.arange(n_win) + 0.5) * WINDOW_SEC)   # 窗中心時間
        rms0_all.append(r0)
        rms1_all.append(r1)

    tl = pd.DataFrame({
        "t": np.concatenate(t_all),
        "ai0_db": 20 * np.log10(np.concatenate(rms0_all) + 1e-12),
        "ai1_db": 20 * np.log10(np.concatenate(rms1_all) + 1e-12),
    })
    tl["diff_db"] = tl["ai1_db"] - tl["ai0_db"]

    # 舊版（v21）資料檔案之間有空檔：在空檔處插入 NaN，畫圖時線才會斷開，不會把空檔畫成連續
    gaps = np.flatnonzero(np.diff(tl["t"].to_numpy()) > 1.5 * WINDOW_SEC)
    if len(gaps):
        breaks = pd.DataFrame({"t": tl["t"].to_numpy()[gaps] + WINDOW_SEC})
        tl = pd.concat([tl, breaks], ignore_index=True).sort_values("t").reset_index(drop=True)
        print(f"⚠️ 檔案之間有 {len(gaps)} 個空檔（舊版 v21 資料會這樣），圖上會顯示成斷線")
    quality = {"clip": clip, "peak": peak, "n_total": n_total,
               "corr_median": float(np.median(corr_list))}
    return tl, quality


# ---------- 找訊號真正變化的時間 ----------
def change_time_step(tl, col, t_event):
    """
    transition 標記用：標記前（−SEARCH ~ −1 秒）和標記後（+1 ~ +SEARCH 秒）的中位數，
    訊號第一次越過兩者中點的時間，就是訊號真正變化的時間。
    回傳 (變化時間, 前中位數, 後中位數)；前後差不到 1 dB 視為「看不出變化」。
    """
    pre = tl.loc[(tl.t >= t_event - SEARCH_SEC) & (tl.t <= t_event - 1.0), col].dropna()
    post = tl.loc[(tl.t >= t_event + 1.0) & (tl.t <= t_event + SEARCH_SEC), col].dropna()
    if len(pre) < 5 or len(post) < 5:
        return np.nan, np.nan, np.nan
    a, b = float(pre.median()), float(post.median())
    if abs(b - a) < 1.0:
        return np.nan, a, b
    mid = (a + b) / 2
    seg = tl.loc[(tl.t >= t_event - SEARCH_SEC) & (tl.t <= t_event + SEARCH_SEC)].dropna(subset=[col])
    crossed = seg[(seg[col] - mid) * np.sign(b - a) > 0]
    # 取「越過之後至少連續 0.2 秒都維持」的第一點，避免被單點雜訊騙
    need = int(round(0.2 / WINDOW_SEC))
    flags = ((seg[col] - mid) * np.sign(b - a) > 0).to_numpy()
    for i in range(len(flags) - need + 1):
        if flags[i:i + need].all():
            return float(seg["t"].iloc[i]), a, b
    return (float(crossed["t"].iloc[0]) if not crossed.empty else np.nan), a, b


def burst_span(tl, col, t_start, t_end):
    """
    干擾區間用：標記開始前 SEARCH 秒當基準（中位數 + 6 × MAD），
    在 [開始 − SEARCH, 結束 + SEARCH] 內找第一個和最後一個超過基準的時間。
    """
    base = tl.loc[(tl.t >= t_start - SEARCH_SEC) & (tl.t < t_start - 0.5), col].dropna()
    if len(base) < 5:
        return np.nan, np.nan, np.nan
    med = float(base.median())
    mad = float(np.median(np.abs(base - med))) * 1.4826 + 1e-6
    thr = med + max(6 * mad, 3.0)            # 至少比背景高 3 dB 才算
    seg = tl.loc[(tl.t >= t_start - SEARCH_SEC) & (tl.t <= t_end + SEARCH_SEC)].dropna(subset=[col])
    hit = seg[seg[col] > thr]
    if hit.empty:
        return np.nan, np.nan, thr
    return float(hit["t"].iloc[0]), float(hit["t"].iloc[-1]), thr


# ---------- 報告 ----------
def report_quality(q):
    print("=" * 70)
    print("訊號品質")
    for ch in ("ai0", "ai1"):
        ratio = q["clip"][ch] / q["n_total"]
        flag = "✅" if ratio < 1e-6 else ("⚠️ 偶爾削峰" if ratio < 1e-3 else "❌ 削峰嚴重，請調小增益或改大量程")
        print(f"  {ch}：最大 {q['peak'][ch]:.3f} V，接近滿刻度的點 {q['clip'][ch]} 個（{ratio:.2e}）{flag}")
    c = q["corr_median"]
    print(f"  ai0 與 ai1 原始波形相關係數（中位數）：{c:.3f}  "
          f"{'❌ 幾乎一模一樣，可能兩個通道接到同一顆感測器' if c > 0.99 else '✅'}")


def report_transition(tl, trans):
    if trans.empty:
        return
    print("=" * 70)
    print("transition 標記 vs 訊號變化（落差 = 訊號變化時間 − 標記時間；負的 = 訊號比標記早）")
    for _, ev in trans.iterrows():
        t = float(ev["event_elapsed_sec"])
        print(f"  #{int(ev['event_index'])} {ev['from_state']} → {ev['to_state']} @ {t:.2f} 秒")
        for col, name in (("ai0_db", "ai0"), ("ai1_db", "ai1")):
            tc, a, b = change_time_step(tl, col, t)
            if np.isnan(a):
                print(f"     {name}：標記前後資料不足 {SEARCH_SEC:.0f} 秒，無法判斷")
            elif np.isnan(tc):
                print(f"     {name}：前 {a:.1f} dB → 後 {b:.1f} dB，變化不到 1 dB，看不出切換")
            else:
                lag = tc - t
                flag = "✅" if abs(lag) <= 1.0 else ("⚠️" if abs(lag) <= 2.0 else "❌ 落差太大")
                print(f"     {name}：前 {a:.1f} dB → 後 {b:.1f} dB（{b - a:+.1f} dB），"
                      f"訊號在 {tc:.2f} 秒變化，落差 {lag:+.2f} 秒 {flag}")


def report_disturbance(tl, dist):
    if dist.empty:
        return
    print("=" * 70)
    print("干擾標記 vs 訊號突增（落差 = 訊號時間 − 標記時間）")
    for _, ev in dist.iterrows():
        ts, te = float(ev["start_elapsed_sec"]), float(ev["end_elapsed_sec"])
        print(f"  #{int(ev['event_index'])} {ev['disturbance_type']}（{ev['disturbance_strength']}）"
              f"標記 {ts:.2f} ~ {te:.2f} 秒")
        amps = {}
        for col, name in (("ai0_db", "ai0"), ("ai1_db", "ai1")):
            s, e, thr = burst_span(tl, col, ts, te)
            if np.isnan(thr):
                print(f"     {name}：標記前資料不足，無法判斷")
                continue
            if np.isnan(s):
                print(f"     {name}：沒有明顯突增（門檻 {thr:.1f} dB）")
                continue
            ok = abs(s - ts) <= 1.0
            print(f"     {name}：訊號 {s:.2f} ~ {e:.2f} 秒；開始落差 {s - ts:+.2f}、結束落差 {e - te:+.2f} 秒 "
                  f"{'✅' if ok else '❌ 開始時間差超過 1 秒'}")
            seg = tl.loc[(tl.t >= s) & (tl.t <= e), col].dropna()
            amps[name] = float(seg.max())
        if len(amps) == 2:
            d = amps["ai0"] - amps["ai1"]
            print(f"     峰值 ai0 − ai1 = {d:+.1f} dB → 這次干擾比較靠近 {'ai0' if d > 0 else 'ai1'}")


def plot(tl, trans, dist, title, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, col, label in zip(axes, ("ai0_db", "ai1_db", "diff_db"),
                              ("ai0 能量 (dB)", "ai1 能量 (dB)", "ai1 − ai0 (dB)")):
        ax.plot(tl["t"], tl[col], lw=0.8, color="black")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        for _, ev in dist.iterrows():
            ax.axvspan(float(ev["start_elapsed_sec"]), float(ev["end_elapsed_sec"]), color="gray", alpha=0.25)
        for _, ev in trans.iterrows():
            leak = ev["to_state"] == "leak"
            ax.axvline(float(ev["event_elapsed_sec"]), color="red" if leak else "blue",
                       ls="-" if leak else "--", lw=1.2)
    axes[0].set_title(f"{title}\n紅實線 = 標記 leak，藍虛線 = 標記 normal，灰區 = 標記干擾")
    axes[-1].set_xlabel("session 時間（秒）")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"圖已存：{save_path}")
    plt.show()


if __name__ == "__main__":
    folder = Path(FOLDER).expanduser()
    if folder.is_file():              # 填的是某一支 tdms 檔 → 改用它所在的資料夾
        folder = folder.parent
    df, manifest_path = load_manifest(folder)
    trans, dist = load_events(manifest_path.parent)   # 標記檔和 manifest 放在同一個資料夾
    tl, quality = compute_timeline(df)
    report_quality(quality)
    report_transition(tl, trans)
    report_disturbance(tl, dist)
    print("=" * 70)
    plot(tl, trans, dist, f"{folder.name}（{manifest_path.name}）", folder / f"pilot_check_{folder.name}.png")