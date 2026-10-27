# -*- coding: utf-8 -*-
"""
檢查長時間收集的輸出是否「首尾相接、沒有空檔」。

用法（在 Spyder 改下面 MANIFEST_CSV 後直接執行）：
  1. 錄完、按輸出後，會產生 dual_session_manifest_*.csv（或 transition_dual_manifest_*.csv）
  2. MANIFEST_CSV 填「那個 CSV 的路徑」，或直接填「v22 的存檔資料夾」
     （填資料夾時，會自動找裡面最新的一份 manifest）
  3. 執行，看最後的「結論」

檢查三件事：
  A. 取樣序號：相鄰檔案的 first_sample_index 是否都差「每支點數」（例如 500000）
  B. 時間：相鄰檔案的 capture_wall_time 是否都差「每支秒數」（例如 0.5）
  C. 波形（選用，需要接一個穩定訊號，例如正弦波）：
     檔案交界處的跳動，是否和檔案內部相鄰兩點的跳動一樣小。
     有空檔的話，交界處相位會突然跳掉，跳動會大很多。
     A、B 是程式自己算的，只能證明「程式認為」沒漏；C 是用真實訊號獨立驗證。
"""
from pathlib import Path
import numpy as np
import pandas as pd
from nptdms import TdmsFile

MANIFEST_CSV = r"/Users/paul/Desktop/模擬測試/transition2"   # ← manifest 的路徑，或 v22 的存檔資料夾
CHECK_WAVEFORM = False      # 有接正弦波才有意義；頻率請用不整齊的數字（例如 1234.5 Hz），只有雜訊時改成 False
CHANNEL = "ai0"             # 用哪個通道檢查波形


def find_manifest(path):
    """給檔案就直接用；給資料夾就找裡面（含子資料夾）最新的一份 manifest"""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"找不到：{path}")
    candidates = list(path.rglob("dual_session_manifest_*.csv")) + list(path.rglob("transition_dual_manifest_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"{path} 裡找不到 dual_session_manifest_*.csv，請確認已經按過「輸出」")
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"使用 manifest：{newest}")
    return newest


def load_manifest(path):
    path = find_manifest(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 一般 dual 輸出用 dual_file；transition dual 輸出用 merged_file
    file_col = next((c for c in ("dual_file", "merged_file", "ai0_file") if c in df.columns), None)
    if file_col is None:
        raise ValueError("manifest 找不到資料檔欄位（dual_file / merged_file / ai0_file）")
    df["data_file"] = [resolve_data_file(p, path.parent) for p in df[file_col]]
    need = ["first_sample_index", "capture_wall_time", "sample_clock_start_wall", "sample_count", "sample_rate"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"manifest 缺少欄位 {missing}：這份可能是舊版 v21 輸出的")
    if df["first_sample_index"].isna().all() or (df["first_sample_index"].astype(str).str.strip() == "").all():
        raise ValueError("first_sample_index 全是空的：這次沒有走連續擷取（請確認雙通道 + 間隔 0）")
    return df.sort_values(["sample_clock_start_wall", "first_sample_index"]).reset_index(drop=True)


def check_index_and_time(df):
    """
    同一個暫存 session 裡可能包含好幾次「長時間收集」（例如按了兩次開始）。
    每一次都是一個獨立的 DAQ 任務，取樣序號都從 0 開始，
    所以用 sample_clock_start_wall（任務啟動時間）分組，每一次各自檢查。
    """
    n = int(df["sample_count"].iloc[0])
    fs = float(df["sample_rate"].iloc[0])
    chunk_sec = n / fs
    all_ok = True

    runs = list(df.groupby("sample_clock_start_wall", sort=True))
    print("=" * 70)
    print(f"共 {len(df)} 支檔案，每支 {n} 點、{chunk_sec:.3f} 秒；分屬 {len(runs)} 次長時間收集")

    first_start = float(runs[0][0])
    for k, (start_wall, run) in enumerate(runs, start=1):
        run = run.sort_values("first_sample_index")
        idx = run["first_sample_index"].astype(np.int64).to_numpy()
        t = run["capture_wall_time"].astype(float).to_numpy()
        idx_diff = np.diff(idx)
        t_diff = np.diff(t)

        print("-" * 70)
        print(f"第 {k} 次：第 {float(start_wall) - first_start:.1f} 秒開始，"
              f"{len(run)} 支、共 {len(run) * chunk_sec:.1f} 秒")

        ok_first = idx[0] == 0
        print(f"  第一支的取樣序號：{idx[0]}  {'✅' if ok_first else '❌ 應該是 0（前面有檔案不見了）'}")

        bad_idx = int(np.sum(idx_diff != n))
        vals = sorted(set(idx_diff.tolist())) if len(idx_diff) else []
        print(f"  A. 取樣序號差（應該全是 {n}）：出現 {vals}  "
              f"{'✅' if bad_idx == 0 else f'❌ {bad_idx} 處不對'}")

        bad_t = int(np.sum(np.abs(t_diff - chunk_sec) > 1e-3))
        if len(t_diff):
            print(f"  B. 開始時間差（應該全是 {chunk_sec:.3f} 秒）：最小 {t_diff.min():.4f}、最大 {t_diff.max():.4f}  "
                  f"{'✅' if bad_t == 0 else f'❌ {bad_t} 處不對'}")
        all_ok = all_ok and ok_first and bad_idx == 0 and bad_t == 0

    return all_ok


def resolve_data_file(p, manifest_dir):
    """manifest 裡的路徑可能是錄製那台電腦（Windows）的路徑；找不到就只取檔名，到 manifest 所在資料夾找"""
    p_str = str(p)
    p = Path(p_str)
    if p.exists():
        return p
    name = p_str.replace("\\", "/").split("/")[-1]
    hits = list(manifest_dir.rglob(name))
    return hits[0] if hits else p


def read_channel(path, channel):
    tdms = TdmsFile.read(str(path))
    for group in tdms.groups():
        for ch in group.channels():
            if ch.name == channel:
                return np.asarray(ch[:], dtype=np.float64)
    raise ValueError(f"{path} 找不到通道 {channel}")


def check_waveform(df, channel):
    """
    交界跳動 = |下一支第一點 − 這一支最後一點|
    內部跳動 = 檔案內部相鄰兩點差的 99.9 百分位
    比值接近 1 → 接得順；比值很大（> 5）→ 交界處波形斷掉，代表有空檔
    """
    ratios = []
    for _, run in df.groupby("sample_clock_start_wall", sort=True):
        prev_tail = None                       # 每一次收集各自檢查，不跨次比較
        for _, row in run.sort_values("first_sample_index").iterrows():
            x = read_channel(row["data_file"], channel)
            inner = np.quantile(np.abs(np.diff(x)), 0.999)
            if prev_tail is not None:
                step = abs(x[0] - prev_tail)
                ratios.append(step / max(inner, 1e-12))
            prev_tail = x[-1]
    ratios = np.array(ratios)
    print(f"C. 波形交界（通道 {channel}）：交界跳動 ÷ 內部跳動，應該都 ≲ 1")
    print(f"   中位數 {np.median(ratios):.2f}、最大 {ratios.max():.2f}")
    bad = np.flatnonzero(ratios > 5)
    print(f"   {'✅ 全部接得順' if len(bad) == 0 else f'❌ {len(bad)} 處斷掉，位置 {bad[:10].tolist()}'}")
    return len(bad) == 0


if __name__ == "__main__":
    df = load_manifest(MANIFEST_CSV)
    ok = check_index_and_time(df)
    if CHECK_WAVEFORM:
        ok = check_waveform(df, CHANNEL) and ok
    print("=" * 70)
    print("結論：", "✅ 檔案首尾相接，沒有空檔" if ok else "❌ 有問題，請把上面的輸出貼給 Claude")