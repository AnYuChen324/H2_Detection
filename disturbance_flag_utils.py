# -*- coding: utf-8 -*-
"""
disturbance_flag_utils.py — 即時「疑似干擾」偵測（能量支線，和模型無關）
=========================================================================

為什麼需要：
    模型只分「正常 / 洩漏」，干擾在訓練時被標成正常。模型判對時（敲管 → 機率低）
    後處理看不到任何東西，畫面維持綠燈，現場看不出「剛剛有干擾」。
    這支直接看訊號：寬頻能量突然跳升 → 疑似干擾。不需要洩漏頻帶，只需要正常參考。

偵測邏輯（每一批資料，例如每 5 秒擷取 2 秒）：
    1. 把這批切成不重疊的 0.2 秒窗，每窗去直流後 FFT，算 ai0 / ai1 在 9 個寬頻帶的能量（dB）
       頻帶：1–5k、5–10k、10–20k、20–30k、30–50k、50–100k、100–200k、200–300k、300–480k
    2. 突升（burst）：某窗某頻帶 − 這批中位數 > 門檻 → 這窗是干擾窗
       （預設不看 1–10 kHz：那裡正常就常因說話、環境聲跳 5～14 dB）
       → 「批內少數窗突出」= 短暫干擾（敲擊、碰線、腳步）；整批一起高不算 burst
    3. 背景偏高（background_high）：這批中位數 − 正常參考中位數 > 門檻
       → 持續偏高（持續噪音、背景漂移、大洩漏都有可能），只記錄，不當成干擾
    4. 門檻來自正常參考資料（和 site baseline 同一個 normal 資料夾）：
       - burst：每格（通道 × 頻帶）門檻 = max(最低 dB, 正常資料「窗 − 批中位數」的最大值 + 餘裕 3 dB)
         （正常資料要選乾淨的；裡面如果有干擾，門檻會被拉高、靈敏度變差）
       - background：每格門檻 = max(最低 dB, 正常資料「批中位數 − 參考中位數」的 99 百分位 + 餘裕)

離線 / 即時共用：
    - 參考檔由 build_disturbance_reference.py 產生（json），API 和離線評估讀同一份
    - 即時：API 每批呼叫 detect_disturbance_in_batch(ai0, ai1, ref)
    - 離線：eval_disturbance_flag.py 把連續資料切成同樣長度的批，呼叫同一個函式

限制：
    - 只判斷「訊號有沒有突然變大」，不判斷干擾種類
    - 洩漏剛開始的那一批，如果洩漏量大到能量明顯跳升，也可能被標 burst 一次（離線評估會量這個比例）
    - 這支只負責顯示 / 記錄，不改變洩漏告警
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import fft as sfft

SCHEMA_NAME = "disturbance_reference"
SCHEMA_VERSION = 1

# 寬頻帶（kHz）：涵蓋 1–480 kHz，和 11/11 干擾分析相同
DEFAULT_GROUPS_KHZ = [(1, 5), (5, 10), (10, 20), (20, 30), (30, 50),
                      (50, 100), (100, 200), (200, 300), (300, 480)]
CHANNELS = ("ai0", "ai1")

# 最低門檻（dB）：1–10 kHz 本來就晃比較大（環境低頻、電源），給 3 dB；其他 1.5 dB
DEFAULT_MIN_BURST_DB = {(1, 5): 3.0, (5, 10): 3.0}
DEFAULT_MIN_BURST_DB_OTHER = 1.5
DEFAULT_MIN_BACKGROUND_DB = 3.0
MIN_WINDOWS_PER_BATCH = 3      # 一批至少要 3 窗，中位數才有意義
# 突升判斷預設不看 1–10 kHz：11/11 實測正常資料、轉換資料在這兩個頻帶本來就常跳 5～14 dB（說話、環境聲），
# 看了會讓正常批 3% 以上、轉換資料 10～25% 被標干擾；拿掉後 30 個干擾事件仍然全部抓到
DEFAULT_BURST_EXCLUDE_GROUPS_KHZ = [(1, 5), (5, 10)]


def _group_label(ch, lo, hi):
    return f"{ch} {lo:g}–{hi:g}k"


# ============================================================
# 1. 每窗、每通道、每頻帶的能量（dB）
# ============================================================
def compute_group_energy_db(ai0, ai1, fs, window_sec, groups_khz):
    """
    把一段訊號切成不重疊的窗，回傳能量 E[窗, 通道, 頻帶]（dB）。
    - 每窗先去直流（和特徵工程一致）
    - FFT 功率在頻帶內加總（Parseval：等同帶通後能量，只差常數）
    - 結尾不足一窗的部分丟掉
    """
    ai0 = np.asarray(ai0, dtype=np.float64)
    ai1 = np.asarray(ai1, dtype=np.float64)
    if ai0.ndim != 1 or ai1.ndim != 1:
        raise ValueError("ai0 / ai1 必須是一維陣列")
    if len(ai0) != len(ai1):
        raise ValueError(f"ai0 / ai1 長度不同：{len(ai0)} vs {len(ai1)}")
    if not (np.all(np.isfinite(ai0)) and np.all(np.isfinite(ai1))):
        raise ValueError("訊號有 NaN / Inf，無法計算干擾能量")
    w = int(round(window_sec * fs))
    if abs(w - window_sec * fs) > 1e-6 or w <= 0:
        raise ValueError(f"window_sec × fs 必須是正整數：{window_sec} × {fs}")
    nyq = fs / 2.0
    for lo, hi in groups_khz:
        if not (0 <= lo * 1000 < hi * 1000 <= nyq):
            raise ValueError(f"頻帶 {lo}–{hi} kHz 不合法（Nyquist = {nyq / 1000:g} kHz）")
    n_win = len(ai0) // w
    freqs = np.fft.rfftfreq(w, 1.0 / fs)
    idx = [(int(np.searchsorted(freqs, lo * 1000)), int(np.searchsorted(freqs, hi * 1000)))
           for lo, hi in groups_khz]
    E = np.empty((n_win, len(CHANNELS), len(groups_khz)), dtype=np.float64)
    for k in range(n_win):
        s = slice(k * w, (k + 1) * w)
        for c, x in enumerate((ai0[s], ai1[s])):
            x = x - x.mean()
            P = np.abs(sfft.rfft(x)) ** 2
            cs = np.concatenate([[0.0], np.cumsum(P)])
            E[k, c, :] = [cs[b] - cs[a] for a, b in idx]
    return 10.0 * np.log10(E + 1e-30)


# ============================================================
# 2. 一批的判斷（離線、即時共用）
# ============================================================
def _batch_scores(E, ref):
    """回傳 (burst 超出量[窗,通道,頻帶], 背景超出量[通道,頻帶])，單位 dB，> 0 代表超過門檻。"""
    med = np.median(E, axis=0)                                        # 這批中位數
    burst_thr = np.asarray(ref["burst_threshold_db"], dtype=np.float64)
    bg_thr = np.asarray(ref["background_threshold_db"], dtype=np.float64)
    ref_med = np.asarray(ref["reference_median_db"], dtype=np.float64)
    burst_excess = (E - med[None, :, :]) - burst_thr[None, :, :]
    enabled = np.asarray(ref["burst_enabled"], dtype=bool)              # 不看的頻帶（低頻環境聲）不算突升
    burst_excess = np.where(enabled[None, :, :], burst_excess, -np.inf)
    bg_excess = (med - ref_med) - bg_thr
    return burst_excess, bg_excess, med


def detect_disturbance_in_batch(ai0, ai1, ref):
    """
    判斷一批資料有沒有疑似干擾。ref = load_disturbance_reference() 讀進來的參考。
    回傳 dict：
        burst                 : bool，批內有窗突升（短暫干擾）
        background_high       : bool，整批比正常參考高（只記錄）
        n_windows             : 這批幾窗
        n_burst_windows       : 幾窗突升
        burst_window_index    : 突升窗的編號（0 起算，窗長 = ref["window_sec"]）
        burst_bands           : 突升最多的頻帶（由大到小，最多 3 個），例如 ["ai1 30–50k", ...]
        max_burst_db          : 最大突升量（窗 − 批中位數，dB）
        background_bands      : 背景偏高的頻帶（最多 3 個）
        max_background_db     : 批中位數 − 參考中位數 最大值（dB）
    """
    E = compute_group_energy_db(ai0, ai1, ref["fs"], ref["window_sec"],
                                [tuple(g) for g in ref["groups_khz"]])
    n = E.shape[0]
    if n < MIN_WINDOWS_PER_BATCH:
        raise ValueError(f"這批只有 {n} 窗（< {MIN_WINDOWS_PER_BATCH}），無法判斷干擾；"
                         f"批長至少 {MIN_WINDOWS_PER_BATCH * ref['window_sec']:g} 秒")
    burst_excess, bg_excess, med = _batch_scores(E, ref)
    groups = [tuple(g) for g in ref["groups_khz"]]

    win_hit = (burst_excess > 0).any(axis=(1, 2))
    # 每格最大突升量，用來列出最明顯的頻帶
    cell_max = burst_excess.max(axis=0)
    burst_bands = []
    if win_hit.any():
        order = np.argsort(cell_max, axis=None)[::-1]
        for flat in order[:3]:
            c, g = np.unravel_index(flat, cell_max.shape)
            if cell_max[c, g] > 0:
                burst_bands.append(_group_label(CHANNELS[c], *groups[g]))
    rise = np.where(np.asarray(ref["burst_enabled"], dtype=bool)[None, :, :], E - med[None, :, :], -np.inf)
    bg_rise = med - np.asarray(ref["reference_median_db"], dtype=np.float64)
    bg_bands = []
    if (bg_excess > 0).any():
        order = np.argsort(bg_excess, axis=None)[::-1]
        for flat in order[:3]:
            c, g = np.unravel_index(flat, bg_excess.shape)
            if bg_excess[c, g] > 0:
                bg_bands.append(_group_label(CHANNELS[c], *groups[g]))
    return {
        "burst": bool(win_hit.any()),
        "background_high": bool((bg_excess > 0).any()),
        "n_windows": int(n),
        "n_burst_windows": int(win_hit.sum()),
        "burst_window_index": np.flatnonzero(win_hit).astype(int).tolist(),
        "burst_bands": burst_bands,
        "max_burst_db": round(float(rise.max()), 2),
        "background_bands": bg_bands,
        "max_background_db": round(float(bg_rise.max()), 2),
    }


# ============================================================
# 3. 參考檔：建立 / 讀取
# ============================================================
def build_reference_from_energy(E_batches, *, fs, window_sec, groups_khz, capture_sec,
                                burst_margin_db=3.0, background_margin_db=1.0,
                                min_burst_db=None, min_background_db=DEFAULT_MIN_BACKGROUND_DB,
                                burst_exclude_groups_khz=DEFAULT_BURST_EXCLUDE_GROUPS_KHZ,
                                source=None):
    """
    E_batches：list，每個元素是一批的 E[窗, 通道, 頻帶]（正常資料）。
    回傳參考 dict（可直接 json 存檔）。
    """
    if len(E_batches) < 20:
        raise ValueError(f"正常資料只有 {len(E_batches)} 批（< 20），門檻估不穩；請用更長的正常資料")
    groups = [tuple(g) for g in groups_khz]
    if min_burst_db is None:
        min_burst_db = {g: DEFAULT_MIN_BURST_DB.get(g, DEFAULT_MIN_BURST_DB_OTHER) for g in groups}
    min_b = np.array([[min_burst_db[g] for g in groups] for _ in CHANNELS])

    allE = np.concatenate(E_batches, axis=0)
    ref_med = np.median(allE, axis=0)
    # 窗 − 批中位數：正常資料自己的批內晃動
    dev = np.concatenate([b - np.median(b, axis=0)[None] for b in E_batches], axis=0)
    # 突升門檻（每格）= max(最低門檻, 正常資料批內最大晃動 + 餘裕)
    #   餘裕預設 3 dB：11/12 下午實驗室較吵，1 dB 時正常批被標 5～8%，3 dB 降到 1%，干擾事件一個都沒少抓
    #   11/11 實測：50 kHz 以上正常晃動最大只有 0.1–0.2 dB（干擾 +15～68 dB）；
    #   1–10 kHz 正常就會跳 5～14 dB（環境低頻），所以低頻門檻會自動變高
    burst_thr = np.maximum(min_b, dev.max(axis=0) + burst_margin_db)

    # 背景門檻：批中位數 − 參考中位數 的 99 百分位 + 餘裕
    bmed = np.stack([np.median(b, axis=0) for b in E_batches])
    bg_q = np.quantile(bmed - ref_med[None], 0.99, axis=0)
    bg_thr = np.maximum(min_background_db, bg_q + background_margin_db)

    ref = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fs": int(fs),
        "window_sec": float(window_sec),
        "capture_sec": float(capture_sec),
        "channels": list(CHANNELS),
        "groups_khz": [list(g) for g in groups],
        "reference_median_db": ref_med.round(4).tolist(),
        "burst_threshold_db": burst_thr.round(4).tolist(),
        "burst_enabled": [[tuple(g) not in {tuple(x) for x in burst_exclude_groups_khz} for g in groups]
                          for _ in CHANNELS],
        "background_threshold_db": bg_thr.round(4).tolist(),
        "calibration": {
            "n_batches": int(len(E_batches)),
            "n_windows": int(allE.shape[0]),
            "burst_rule": "max(min_burst_db, 正常批內最大晃動 + burst_margin_db)",
            "burst_margin_db": float(burst_margin_db),
            "min_burst_db": [[min_burst_db[g] for g in groups] for _ in CHANNELS],
            "min_background_db": float(min_background_db),
            "background_margin_db": float(background_margin_db),
        },
        "source": source or {},
    }
    # 自我檢查：用同一套判斷跑一次正常資料，記下誤標率
    n_burst = sum(1 for b in E_batches if (_batch_scores(b, ref)[0] > 0).any())
    n_bg = sum(1 for b in E_batches if (_batch_scores(b, ref)[1] > 0).any())
    ref["calibration"]["normal_burst_batch_rate"] = round(n_burst / len(E_batches), 4)
    ref["calibration"]["normal_background_batch_rate"] = round(n_bg / len(E_batches), 4)
    return ref


def load_disturbance_reference(path, *, fs=None, capture_sec=None):
    """讀參考檔並檢查和目前設定一致（fs、批長）。不一致直接停止，不自動調整。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到干擾參考檔：{path}（請先跑 build_disturbance_reference.py）")
    ref = json.loads(path.read_text(encoding="utf-8"))
    if ref.get("schema_name") != SCHEMA_NAME or ref.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} 不是干擾參考檔（schema {ref.get('schema_name')} v{ref.get('schema_version')}）")
    shape = (len(ref["channels"]), len(ref["groups_khz"]))
    for key in ("reference_median_db", "burst_threshold_db", "background_threshold_db"):
        a = np.asarray(ref[key], dtype=np.float64)
        if a.shape != shape or not np.all(np.isfinite(a)):
            raise ValueError(f"{path} 的 {key} 形狀 {a.shape} ≠ {shape} 或有 NaN")
    en = np.asarray(ref.get("burst_enabled"), dtype=object)
    if en.shape != shape or not all(isinstance(v, bool) for v in en.ravel()):
        raise ValueError(f"{path} 的 burst_enabled 缺少或形狀錯誤（應為 {shape} 的 true/false）")
    if not any(en.ravel()):
        raise ValueError(f"{path} 的 burst_enabled 全部是 false，等於沒有偵測")
    if fs is not None and int(fs) != int(ref["fs"]):
        raise ValueError(f"干擾參考檔 fs={ref['fs']}，目前 fs={fs}，必須重建參考檔")
    if capture_sec is not None and abs(float(capture_sec) - float(ref["capture_sec"])) > 1e-9:
        raise ValueError(f"干擾參考檔是用每批 {ref['capture_sec']} 秒校正的，目前每批 {capture_sec} 秒；"
                         "批長不同門檻會不準，請用目前批長重建參考檔")
    return ref


# ============================================================
# 4. 和模型結果合併成畫面狀態（三色＋原因）
# ============================================================
REASON_TEXT = {
    "leak_confirmed": "洩漏（連續多批確認）",
    "suspect_with_disturbance": "疑似干擾：衝擊／寬頻（模型短暫判洩漏，未確認）",
    "disturbance_rejected": "偵測到干擾，已排除",
    "pending_confirm": "異常待確認，觀察下一批",
    "normal": "正常",
}


def combine_status(*, confirmed, batch_alarm, burst, any_veto):
    """
    三色狀態（leak_status 值不變：normal / disturbance_suspect / leak）＋原因代碼。
    洩漏告警只由 confirmed（BatchConfirmer）決定，這裡不改變它。
    """
    disturbance = bool(burst or any_veto)
    if confirmed:
        return "leak", "leak_confirmed"
    if batch_alarm:
        return "disturbance_suspect", ("suspect_with_disturbance" if disturbance else "pending_confirm")
    if disturbance:
        return "disturbance_suspect", "disturbance_rejected"
    return "normal", "normal"


# ============================================================
# 5. 離線工具：讀 v22 連續資料、切成和即時一樣長的批
# ============================================================
def iter_tdms_sessions(root, allow_gaps=False):
    """
    找出 root 底下所有 TDMS，依 session_id（檔案屬性）分組、依 first_sample_index 排序。
    要求 v22 連續擷取（acquisition_mode=continuous），舊程式資料直接停。
    allow_gaps=False：取樣序號必須連續，有缺檔就停
    allow_gaps=True ：缺檔處切成多段（每段各自連續），重複序號的檔只留一支；回傳的 session 名稱加 #段號
    回傳 list：(session 名稱, 資料夾, [檔案路徑...], 第一支檔的 first_sample_index)
    """
    from nptdms import TdmsFile
    root = Path(root)
    files = sorted(root.rglob("*.tdms"))
    if not files:
        raise FileNotFoundError(f"{root} 底下沒有 TDMS")
    by = {}
    for f in files:
        p = TdmsFile.read_metadata(f).properties
        sid = p.get("session_id") or f.parent.name
        by.setdefault(sid, []).append((int(p["first_sample_index"]), f, p))
    out = []
    for sid, items in sorted(by.items()):
        items.sort(key=lambda x: x[0])
        p0 = items[0][2]
        if p0.get("acquisition_mode") != "continuous":
            raise ValueError(f"{items[0][1].parent}：不是 v22 連續擷取（acquisition_mode={p0.get('acquisition_mode')}）")
        n = int(p0.get("sample_count") or p0.get("sample_num"))
        if not allow_gaps:
            d = np.diff([x[0] for x in items])
            if len(d) and not np.all(d == n):
                raise ValueError(f"session {sid}：檔案之間取樣序號不連續（應該每支差 {n}）；"
                                 "缺檔的資料請用 allow_gaps=True（評估工具加 --allow-gaps）")
            out.append((sid, items[0][1].parent, [x[1] for x in items], items[0][0]))
            continue
        uniq = []
        for it in items:                       # 重複序號只留第一支
            if not uniq or it[0] != uniq[-1][0]:
                uniq.append(it)
        runs, cur = [], [uniq[0]]
        for it in uniq[1:]:
            if it[0] - cur[-1][0] == n:
                cur.append(it)
            else:
                runs.append(cur); cur = [it]
        runs.append(cur)
        for k, run in enumerate(runs):
            name = sid if len(runs) == 1 else f"{sid}#{k}"
            out.append((name, run[0][1].parent, [x[1] for x in run], run[0][0]))
    return out


def iter_session_batches(files, fs, capture_sec):
    """把一段連續錄製接起來，切成背對背、每批 capture_sec 秒的批。回傳 (批起點秒, ai0, ai1)。"""
    from nptdms import TdmsFile
    n_batch = int(round(capture_sec * fs))
    buf0, buf1, t0 = [], [], 0.0
    have = 0
    for f in files:
        g = TdmsFile.read(f)["AE_Signals"]
        a0 = np.asarray(g["ai0"][:], dtype=np.float64)
        a1 = np.asarray(g["ai1"][:], dtype=np.float64)
        if len(a0) != len(a1):
            raise ValueError(f"{f.name}：ai0 / ai1 長度不同")
        buf0.append(a0); buf1.append(a1); have += len(a0)
        while have >= n_batch:
            x0 = np.concatenate(buf0); x1 = np.concatenate(buf1)
            yield t0, x0[:n_batch], x1[:n_batch]
            buf0, buf1 = [x0[n_batch:]], [x1[n_batch:]]
            have -= n_batch
            t0 += capture_sec
