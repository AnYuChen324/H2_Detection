#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
是「共用告警規則引擎」：接收每個 window 的模型機率、正式告警設定與 veto 規則，輸出套用規則前後的告警結果

模型機率 pred_proba
→ 建立 veto_mask
→ 機率平滑
→ threshold/consecutive/n-of-m/hysteresis
→ 必要時執行 two-stage
→ 移除持續時間太短的警報
→ 輸出 raw/final/veto 結果
"""

from collections import deque     # 載入雙端佇列

import numpy as np                # 建立陣列與布林遮罩
import pandas as pd               # 資料表、數值轉換、rolling mean


# ============================================================
# 負責解讀單條 veto 規則
# ============================================================
def _apply_single_veto_rule(df,       # 包含特徵欄位的 DataFrame
                            rule):    # 單條規則設定
    # 取得這條規則要使用的欄位名稱
    col = rule["col"]
    # 如果資料沒有正式規則需要的欄位，直接停止
    if col not in df.columns:
        raise ValueError(f"veto rule 指定的欄位不存在: {col}")

    # 將欄位轉成數值
    s = pd.to_numeric(df[col], errors="coerce")
    op = rule.get("op", "ge")    # 取得比較運算子,如果規則沒有提供 op，預設使用：ge(大於等於)

    if op == "ge":    # >= threshold
        return s >= rule["threshold"]
    if op == "gt":    # > threshold
        return s > rule["threshold"]
    if op == "le":    # <= threshold
        return s <= rule["threshold"]
    if op == "lt":    # < threshold
        return s < rule["threshold"]

    raise ValueError(f"未知 veto op: {op}")
    
    
# ============================================================
# 根據規則設定去產生一條布林遮罩：True 代表這個 window 被 veto , False 代表這個 window 可以正常進 alarm rule
# ============================================================
def build_veto_mask_for_record(sub_df, cfg):
    '''建立整筆資料的 veto mask：針對一筆 record_id 的所有 windows 建立 veto mask'''
    # 如果：use_veto=False，直接回傳全是 False 的陣列。
    if not cfg.get("use_veto", False):
        return np.zeros(len(sub_df), dtype=bool)
    
    # 表示目前還沒有找到可使用的 veto 規則
    veto_series = None

    # 如果設定有單一 veto_column，優先使用它, 如果欄位不存在，直接報錯
    if cfg.get("veto_column"):
        col = cfg["veto_column"]
        if col not in sub_df.columns:
            raise ValueError(f"指定的 veto_column 不存在: {col}")
        # 如果欄位本來就是布林值：True / False，就直接當成 veto 結果。
        raw = sub_df[col]
        if pd.api.types.is_bool_dtype(raw):
            veto_series = raw.fillna(False).astype(bool)
        else:
            # 如果欄位是數值型，就比較：數值 >= veto_threshold, 不能轉成數值或缺值的資料會補成 0。
            thr = cfg.get("veto_threshold", 0.5)
            veto_series = pd.to_numeric(raw, errors="coerce").fillna(0) >= thr
    
    # 如果沒有單一 veto_column，但有多條 veto_rules，就逐條執行
    elif cfg.get("veto_rules"):
        # 每一條規則產生自己的布林遮罩
        masks = [_apply_single_veto_rule(sub_df, rule).fillna(False) for rule in cfg["veto_rules"]]
        veto_series = masks[0].copy()
        for mask in masks[1:]:
            veto_series |= mask    # 將規則用 OR 合併:規則 A 成立 OR 規則 B 成立→ veto=True
    
    # use_veto=True，卻沒有指定 veto_column 或 veto_rules 時直接停止（不自己猜欄位）
    if veto_series is None:
        raise ValueError("use_veto=True，但規則沒有指定 veto_column 或 veto_rules")

    return veto_series.astype(bool).values


# ============================================================
# alarm rule 警報規則核心
# ============================================================
def detect_leak_events(
    probs,                   # 每個 window 的洩漏機率
    method="consecutive",    # 警報方法
    alarm_threshold=0.5,     # 機率門檻
    smooth_window=3,         # 平滑窗數 smooth_window=3：第 3 個點的分數會參考前 3 個 window 的平均
    consecutive=2,           # 連續命中數
    n_hits=3,                # M 選 N
    m_window=5,              #
    high_threshold=0.8,      # 遲滯門檻
    low_threshold=0.4,       #
    veto_mask=None,          # 哪些 window 不准報警
):
    '''
    模型已經給你每個 window 一個 leak 機率了，接下來要怎麼把這串機率轉成真正的警報訊號
    什麼時候開始算「有警報」, 要不要平滑, 要不要連續幾次才算, 要不要避免一下一下跳動
    整體流程
    1.模型先對每個 window 輸出 pred_proba
    2.這些 pred_proba 先做平滑
    3.再套不同 alarm rule
    4.看哪種 rule 在干擾資料上最不容易誤報
    '''
        
    # 1. 先統一視窗參數型別
    smooth_window = max(int(smooth_window), 1)
    consecutive = max(int(consecutive), 1)

    if n_hits is not None:
        n_hits = int(n_hits)

    if m_window is not None:
        m_window = int(m_window)

    # n_of_m 必須同時具備兩個參數
    if method == "n_of_m":
        if n_hits is None or m_window is None:
            raise ValueError(
                "method=n_of_m 時必須提供 "
                "n_hits 與 m_window"
            )

        if n_hits <= 0 or m_window <= 0:
            raise ValueError(
                "n_hits 與 m_window 必須大於 0"
            )

        if n_hits > m_window:
            raise ValueError(
                "n_hits 不可大於 m_window"
            )

    # 2. 使用轉成整數後的 smooth_window 做平滑
    # 先平滑：做 rolling mean 平滑, 降低單點噪聲, 避免某一個 window 突然飆高就亂報警
    s = pd.Series(probs).rolling(window=smooth_window, min_periods=1).mean().values
    
    # 3. 整理 veto mask
    # 如果沒有傳入 veto，就視為沒有任何 window 被阻擋
    veto_mask = np.zeros(len(s), dtype=bool) if veto_mask is None else np.asarray(veto_mask, dtype=bool)
    
    # 4. 平滑完成、mask 轉成陣列後檢查長度
    if len(veto_mask) != len(s):
        raise ValueError(
            "veto_mask 長度必須與 probs 相同，"
            f"veto_mask={len(veto_mask)}，"
            f"probs={len(s)}"
        )
    
    # 5. 接原本的告警判斷：先做一個和 s 一樣長度的 0/1 陣列，預設全部沒警報, 後面哪個時間點觸發警報，就把對應位置改成 1
    detected = np.zeros_like(s, dtype=int)
    
    # -- 1.threshold --
    #    平滑後分數 >= alarm_threshold AND 沒有被 veto,就直接報警
    #    最簡單, 容易理解, 但很容易抖動, 分數剛好上下浮動時，警報會忽開忽關
    if method == "threshold":
        detected = ((s >= alarm_threshold) & (~veto_mask)).astype(int)

    # -- 2.consecutive --
    #    要連續幾個 windows 都高於 threshold, 才真的報警
    #    比單純 threshold 穩很多, 能過濾偶發尖峰, 還是可能提早報, 而且只會把「達標那一刻」標 1，之前不會回補
    elif method == "consecutive":
        count = 0     # 建立連續命中計數器
        # 逐一檢查每個 window
        for i, p in enumerate(s):     # 遇到 veto：- 連續命中歸零 - 當前 window 不警報 - 跳到下一個 window，所以 veto 會直接切斷連續警報
            if veto_mask[i]:
                count = 0
                continue
            # 達標就累加，未達標就歸零
            if p >= alarm_threshold:
                count += 1
            else:
                count = 0
            # 達到要求次數後，當前 window 才標記為 1
            if count >= consecutive:
                detected[i] = 1

    # -- 3. n_of_m --
    #    不要求一定「連續」, 只要求最近 m_window 個裡面，有至少 n_hits 個超過 threshold
    #    對局部波動比較寬容, 比連續規則更不容易被單一低點打斷、有時候也會更容易提早報, 因為它不要求連在一起
    elif method == "n_of_m":
        # 建立最多保存最近 M 個結果的佇列
        buf = deque(maxlen=m_window)
        for i, p in enumerate(s):
            # 遇到 veto 時清空最近的命中歷史
            if veto_mask[i]:
                buf.clear()
                continue
            # 達標放 1，未達標放 0
            buf.append(int(p >= alarm_threshold))
            # 最近 M 個 window 中至少有 N 個達標，當前 window 就警報
            if sum(buf) >= n_hits:
                detected[i] = 1

    # -- 4. hysteresis --
    #    要開警報，門檻高一點：high_threshold, 已經開了以後，要關警報，門檻低一點：low_threshold
    #    非常適合避免警報抖動, 很像真實系統會用的 alarm latch 概念
    elif method == "hysteresis":
        alarm_on = False     # 一開始警報為關閉狀態
        for i, p in enumerate(s):
            # 遇到 veto 時立即解除警報
            if veto_mask[i]:
                alarm_on = False
                detected[i] = 0
                continue
            # 尚未警報時，必須達到較高門檻才能啟動
            if not alarm_on and p >= high_threshold:
                alarm_on = True
            # 警報啟動後，必須掉到較低門檻以下才解除
            elif alarm_on and p < low_threshold:
                alarm_on = False
            # 將目前警報狀態寫入輸出
            detected[i] = int(alarm_on)

    else:
        raise ValueError(f"未知 method: {method}")

    return s, detected      # 回傳：s平滑後的分數序列 , detected 最後的 0/1 警報序列


# ============================================================
# 尋找連續警報區段:找出一段 0/1 序列中連續為 1 的區間
# ============================================================
def find_positive_runs(binary_arr):
    runs = []       # 建立結果清單
    start = None    # 初始化：表示目前不在正例區段

    for i, v in enumerate(binary_arr):
        # 第一次遇到 1，記錄開始位置
        if v == 1 and start is None:
            start = i
        # 遇到 0 時，結束目前區段並保存
        elif v == 0 and start is not None:
            runs.append((start, i - 1))
            start = None

    # 若序列最後仍是 1，補上最後一段
    if start is not None:
        runs.append((start, len(binary_arr) - 1))

    return runs


# ============================================================
# 把一條 record 依「真正的時間空檔」切成幾段連續區間
# ============================================================
def split_continuous_segments(sub, max_gap_sec=None):
    """
    相鄰兩窗的 t_start 差超過 max_gap_sec，就視為中間有空檔，切成不同段。
    每段各自平滑、計數、算持續時間，空檔前後不會被當成相鄰。

    max_gap_sec 預設 = 窗長 + 步長：
      - 同一檔內相鄰窗差 1 個步長（0.05 秒）
      - 檔案首尾相接時，交界處差 1 個窗長（0.20 秒），仍視為連續
      - 超過這個值，才是真正沒錄到的空檔（例如 Disturbance 的 0.5 秒空檔）
    回傳每段的 index 清單（依時間排序）
    """
    t = sub["t_start"].to_numpy(dtype=float)
    if len(t) <= 1:
        return [sub.index]

    if max_gap_sec is None:
        hop_sec = float(np.median(np.diff(t)))     # 步長（大多數相鄰窗的時間差）
        if "t_ref" in sub.columns:
            window_sec = float(np.median((sub["t_ref"] - sub["t_start"]).to_numpy(dtype=float)) * 2.0)
        else:
            window_sec = hop_sec
        max_gap_sec = window_sec + hop_sec

    # 每段的第一窗標 True，累加後就得到段編號
    new_segment = np.r_[True, np.diff(t) > max_gap_sec + 1e-9]
    segment_id = np.cumsum(new_segment)
    return [sub.index[segment_id == k] for k in np.unique(segment_id)]


# ============================================================
# two-stage 規則：短事件先用 stage1，長事件再用 stage2 複核
# ============================================================
def detect_two_stage_events(probs, stage1_cfg, stage2_cfg, window_threshold, veto_mask=None):
    # ---  stage 1：先用短干擾最佳規則  ---
    smoothed1, detected1 = detect_leak_events(
        probs,
        method=stage1_cfg["method"],
        alarm_threshold=stage1_cfg.get("alarm_threshold", window_threshold),
        smooth_window=stage1_cfg.get("smooth_window", 3),
        consecutive=stage1_cfg.get("consecutive", 2),
        n_hits=stage1_cfg.get("n_hits", 3),
        m_window=stage1_cfg.get("m_window", 5),
        high_threshold=stage1_cfg.get("high_threshold", 0.8),
        low_threshold=stage1_cfg.get("low_threshold", 0.4),
        veto_mask=None,     # Stage 1 完全不套 veto
    )

    # 先假設 Stage 1 結果就是最終結果
    final_detected = detected1.copy()
    # 建立欄位記錄哪些 window 進過 Stage 2
    stage2_applied = np.zeros(len(probs), dtype=int)

    # 先找 stage1 報出的連續 alarm 段
    runs = find_positive_runs(detected1)
    # 預設 Stage 1 警報至少連續四個 window，才進 Stage 2
    trigger_len = stage2_cfg.get("trigger_run_length", 4)

    # 逐段計算長度
    for start, end in runs:
        run_len = end - start + 1

        # 只有夠長的事件才進 stage 2, 太短就跳過，因此短區段會直接保留 Stage 1 結果，不經過 Stage 2，也不套 veto
        if run_len < trigger_len:
            continue
        # 取出目前區段的：模型機率, 對應 veto mask
        seg_probs = probs[start:end + 1]
        seg_veto = None if veto_mask is None else veto_mask[start:end + 1]

        _, seg_detected2 = detect_leak_events(
            seg_probs,
            method=stage2_cfg["method"],
            alarm_threshold=stage2_cfg.get("alarm_threshold", window_threshold),
            smooth_window=stage2_cfg.get("smooth_window", 3),
            consecutive=stage2_cfg.get("consecutive", 2),
            n_hits=stage2_cfg.get("n_hits", 3),
            m_window=stage2_cfg.get("m_window", 5),
            high_threshold=stage2_cfg.get("high_threshold", 0.8),
            low_threshold=stage2_cfg.get("low_threshold", 0.4),
            veto_mask=seg_veto,     # Stage 2 會正式套用 veto
        )
        
        # 用 Stage 2 結果覆蓋原本 Stage 1 的該段結果
        final_detected[start:end + 1] = seg_detected2
        stage2_applied[start:end + 1] = 1     # 將這個區段標記成：1

    return smoothed1, detected1, final_detected, stage2_applied


# ============================================================
# 每個 session / record 分開套規則
# ============================================================
def apply_alarm_method_by_record(
        df_pred,             # 推論後的 DataFrame,至少要有：record_id, t_start, pred_proba
        cfg,                 # alarm rule 的設定字典
        window_threshold,
        max_gap_sec=None):   # 相鄰窗超過這個秒數視為空檔；None = 窗長 + 步長
    '''
    把 alarm rule 真正套到每一條 record 上
    '''
    # 先複製一份資料，並依照：record_id, t_start 排序。(因為 alarm rule 是沿著時間走的,如果時間順序亂掉，rolling / consecutive 就全錯)
    df_out = df_pred.copy().sort_values(["record_id", "t_start"]).copy()
    
    # 新增 5 個欄位
    df_out["smoothed_proba_rule"] = np.nan     # 存平滑後的機率序列
    df_out["veto_flag"] = 0                    # 是否被 veto
    df_out["detected_rule_raw"] = 0            # 未套 veto 的警報
    df_out["detected_rule"] = 0                # 套完 veto 的正式警報
    df_out["stage2_applied"] = 0               # 是否經過 Stage 2
    
    # --  最短持續時間（即時版）：一段警報要持續滿 min_leak_duration_sec 才開始報 --
    def suppress_short_alarm_runs(sub, detected, min_leak_duration_sec):
        """
        即時系統在警報剛出現時，還不知道它之後會不會持續夠久，
        所以每段連續警報都要等到「已持續 min_leak_duration_sec 秒」那一窗才開始報警，前面的窗一律不報。
        太短的警報從頭到尾都達不到門檻，整段自然不會報。
        （舊版是整段結束後才判斷，夠長就連前段一起保留，等於偷看未來，會低估偵測延遲。）
        """
        # 沒有設定最短時間時，直接回傳原結果
        if min_leak_duration_sec is None or pd.isna(min_leak_duration_sec):
            return np.asarray(detected, dtype=int)
    
        # 複製 detected，避免直接修改輸入陣列
        detected = np.asarray(detected, dtype=int).copy()
        if detected.sum() == 0:
            return detected
    
        # 每個 window 的參考時間：有 t_ref（窗中心）就用它，沒有就退回 t_start
        time_col = "t_ref" if "t_ref" in sub.columns else "t_start"
        time_vals = sub[time_col].to_numpy(dtype=float)
        
        # 窗長：t_ref 是窗中心，所以 (t_ref − t_start) × 2 就是窗長
        if {"t_ref", "t_start"}.issubset(sub.columns):
            window_sec = float(np.median((sub["t_ref"] - sub["t_start"]).to_numpy(dtype=float)) * 2.0)
        elif len(time_vals) >= 2:
            window_sec = float(np.median(np.diff(time_vals)))
        else:
            window_sec = 0.0
    
        min_sec = float(min_leak_duration_sec)
        for start, end in find_positive_runs(detected):
            # 這段警報到每一窗為止「已經持續多久」（第一窗就算一個窗長）
            elapsed = (time_vals[start:end + 1] - time_vals[start]) + window_sec
            # 還沒持續滿 min_sec 的窗不報
            seg = detected[start:end + 1]
            seg[elapsed < min_sec - 1e-9] = 0
            detected[start:end + 1] = seg
    
        return detected.astype(int)
        
    # 逐 record 處理；同一條 record 中間若有空檔，再切成幾段，每段獨立跑 alarm rule
    min_leak_duration_sec = cfg.get("min_leak_duration_sec", None)

    for rid, rec in df_out.groupby("record_id"):
        rec = rec.sort_values("t_start")

        for seg_index in split_continuous_segments(rec, max_gap_sec):
            sub = df_out.loc[seg_index].sort_values("t_start")   # 這一段連續的 windows
            probs = sub["pred_proba"].values                     # 這段時間序列的 leak 機率

            # 根據正式設定建立這一段的 veto mask
            veto_mask = build_veto_mask_for_record(sub, cfg)

            # ---  兩階段規則  ---
            if cfg["method"] == "two_stage":
                smoothed, detected_raw, detected, stage2_applied = detect_two_stage_events(
                    probs=probs,
                    stage1_cfg=cfg["stage1"],
                    stage2_cfg=cfg["stage2"],
                    window_threshold=window_threshold,
                    veto_mask=veto_mask,
                )
                df_out.loc[sub.index, "stage2_applied"] = stage2_applied.astype(int)

            # ---  單一規則：跑兩次，一次不套 veto（raw），一次套 veto（正式）  ---
            else:
                rule_kwargs = dict(
                    method=cfg["method"],
                    alarm_threshold=cfg.get("alarm_threshold", window_threshold),
                    smooth_window=cfg.get("smooth_window", 3),
                    consecutive=cfg.get("consecutive", 2),
                    n_hits=cfg.get("n_hits", 3),
                    m_window=cfg.get("m_window", 5),
                    high_threshold=cfg.get("high_threshold", 0.8),
                    low_threshold=cfg.get("low_threshold", 0.4),
                )
                smoothed, detected_raw = detect_leak_events(probs, veto_mask=None, **rule_kwargs)
                _, detected = detect_leak_events(probs, veto_mask=veto_mask, **rule_kwargs)

            # 最短持續時間（即時版）：raw 與正式結果都套
            detected_raw = suppress_short_alarm_runs(sub, detected_raw, min_leak_duration_sec)
            detected = suppress_short_alarm_runs(sub, detected, min_leak_duration_sec)

            # 寫回輸出結果：把這一段算好的結果，填回整張表對應的那些列
            df_out.loc[sub.index, "smoothed_proba_rule"] = smoothed         # 平滑機率
            df_out.loc[sub.index, "veto_flag"] = veto_mask.astype(int)      # veto 遮罩
            df_out.loc[sub.index, "detected_rule_raw"] = detected_raw       # 未套 veto
            df_out.loc[sub.index, "detected_rule"] = detected               # 套完 veto
    
    # 統一將結果轉成整數
    df_out["veto_flag"] = df_out["veto_flag"].astype(int)
    df_out["detected_rule_raw"] = df_out["detected_rule_raw"].astype(int)
    df_out["detected_rule"] = df_out["detected_rule"].astype(int)
    df_out["stage2_applied"] = df_out["stage2_applied"].astype(int)
    
    return df_out



