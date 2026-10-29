#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
整體流程是這樣：

1. 讀入 disturbance 的 model / rule（可選 full）三種特徵表
2. 以 model csv 為基準，檢查至少要有：
   session_id, disturbance_group, t_start
3. 先把每個 session_id 彙整成一列 summary
4. 依 disturbance_group 分層，每組打亂後：
   holdout_per_group 筆 → holdout、dev_per_group 筆 → dev、其餘 → train
   同一個 session 整個放在同一份，不會被切開
5. model / rule / full 三個版本都加上同一個 split 欄
6. 輸出：
   disturbance_features_train.csv       （model 版，只有 train）→ 模型訓練
   disturbance_rule_features_eval.csv   （rule 版，dev + holdout）→ 干擾推論
   disturbance_full_features_all.csv    （full 版，全部，可選）→ 除錯
   disturbance_session_split.csv        （每個 session 分到哪一份）

三份的用途：
1. train  ：訓練模型（模型看過，不能拿來選規則或評估）
2. dev    ：選規則（干擾推論 mode="select"）
3. holdout：規則定案後量一次成績（干擾推論 mode="evaluate"）

每個 disturbance_group 至少要有 holdout_per_group + dev_per_group + 1 個 session。
'''


from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 先把原始資料用 session_id 分組，做出每筆 session 的摘要表
# ============================================================
def build_session_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("session_id", as_index=False)
        .agg(
            n_windows=("session_id", "size"),
            t_start_min=("t_start", "min"),
            t_start_max=("t_start", "max"),
            disturbance_group=("disturbance_group", "first"),
            disturbance_id=("disturbance_id", "first"),
        )
        .copy()
    )
    summary["group_key"] = summary["disturbance_group"]
    return summary.sort_values(["disturbance_group", "session_id"]).reset_index(drop=True)


# ============================================================
# 依 disturbance_group 分層，把 session 分成 train / dev / holdout 三份
# ============================================================
def stratified_session_split(
    session_summary: pd.DataFrame,
    holdout_per_group: int,
    dev_per_group: int,
    seed: int,
) -> pd.DataFrame:
    """
    三份的用途：
      train   ：訓練模型（模型看過，不能拿來選規則或評估）
      dev     ：選規則（干擾推論 mode="select" 用）
      holdout ：規則定案後量一次成績（干擾推論 mode="evaluate" 用）

    每個 disturbance_group 各自打亂後：前 holdout_per_group 筆 → holdout，
    接著 dev_per_group 筆 → dev，其餘 → train（至少要留 1 筆給 train）。
    同一個 session 整個放在同一份，不會被切開。
    """
    rng = np.random.default_rng(seed)
    split_map: dict[str, str] = {}

    for group_key, sub in session_summary.groupby("group_key", sort=True):
        session_ids = sorted(sub["session_id"].tolist())
        need = holdout_per_group + dev_per_group + 1          # +1：至少 1 筆給 train
        if len(session_ids) < need:
            raise ValueError(
                f"{group_key} 這組只有 {len(session_ids)} 筆 session，"
                f"不夠分成 holdout {holdout_per_group} + dev {dev_per_group} + train 至少 1。"
            )
        shuffled = rng.permutation(session_ids).tolist()
        n_h, n_d = holdout_per_group, dev_per_group
        for sid in shuffled[:n_h]:
            split_map[sid] = "holdout"
        for sid in shuffled[n_h:n_h + n_d]:
            split_map[sid] = "dev"
        for sid in shuffled[n_h + n_d:]:
            split_map[sid] = "train"

    split_summary = session_summary.copy()
    split_summary["split"] = split_summary["session_id"].map(split_map)
    return split_summary


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="把 disturbance features CSV 依 session_id 切成 train / dev / holdout。"
    )

    # 輸入 3 個
    parser.add_argument(
        "--model-csv",
        type=Path,
        default=Path("/Users/paul/Desktop/Final/disturbance_feature_engineering/inference_features_disturbance_model.csv"),
        help="模型版 disturbance features CSV",
    )
    
    parser.add_argument(
        "--rule-csv",
        type=Path,
        default=Path("/Users/paul/Desktop/Final/disturbance_feature_engineering/inference_features_disturbance.csv"),
        help="規則版 disturbance features CSV（含 offband）",
    )
    
    parser.add_argument(
        "--full-csv",
        type=Path,
        default=Path("/Users/paul/Desktop/Final/disturbance_feature_engineering/inference_features_disturbance_full.csv"),
        help="完整 full disturbance features CSV（可選）",
    )

    # 輸出
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="輸出資料夾，預設輸入 CSV 所在資料夾",
    )   
    
    parser.add_argument(
        "--dev-per-group",
        type=int,
        default=1,
        help="每個 disturbance_group 留幾筆 session 到 dev（選規則用），預設 1",
    )

    parser.add_argument(
        "--holdout-per-group",
        type=int,
        default=1,
        help="每個 disturbance_group 留幾筆 session 到 holdout，預設 1",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機切分 seed，預設 42",
    )

    args = parser.parse_args()

    model_csv = args.model_csv
    rule_csv = args.rule_csv
    full_csv = args.full_csv
    
    output_dir = args.output_dir or model_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_model = pd.read_csv(model_csv)
    df_rule = pd.read_csv(rule_csv)
    df_full = pd.read_csv(full_csv) if full_csv is not None and full_csv.exists() else None

    required_cols = {"session_id", "disturbance_group", "t_start"}
    missing_model = sorted(required_cols - set(df_model.columns))
    if missing_model:
        raise ValueError(f"model csv 缺少必要欄位: {missing_model}")
    
    missing_rule = sorted(required_cols - set(df_rule.columns))
    if missing_rule:
        raise ValueError(f"rule csv 缺少必要欄位: {missing_rule}")
    
    if df_full is not None:
        missing_full = sorted(required_cols - set(df_full.columns))
        if missing_full:
            raise ValueError(f"full csv 缺少必要欄位: {missing_full}")
    
    session_summary = build_session_summary(df_model)

    split_summary = stratified_session_split(
        session_summary=session_summary,
        holdout_per_group=args.holdout_per_group,
        dev_per_group=args.dev_per_group,
        seed=args.seed,
    )
    split_map = dict(zip(split_summary["session_id"], split_summary["split"]))
    
    # 三個版本都加上 split 欄，後面的程式一律看這一欄，不再各自重新切
    for df in (df_model, df_rule) + ((df_full,) if df_full is not None else ()):
        df["split"] = df["session_id"].map(split_map)
        if df["split"].isna().any():
            missing = sorted(df.loc[df["split"].isna(), "session_id"].unique().tolist())
            raise ValueError(f"有 session 在 model csv 裡找不到，無法分配：{missing[:10]}")
    
    # 1. 模型訓練用：只有 train
    model_train_path = output_dir / "disturbance_features_train.csv"
    df_model[df_model["split"] == "train"].to_csv(model_train_path, index=False, encoding="utf-8-sig")
    
    # 訓練程式最後評估模型用（model 版）；選規則、驗收仍用 disturbance_rule_features_eval.csv
    model_holdout_path = output_dir / "disturbance_features_holdout.csv"
    df_model[df_model["split"] == "holdout"].to_csv(model_holdout_path, index=False, encoding="utf-8-sig")
    
    # 2. 干擾推論用：dev + holdout 放同一份，靠 split 欄區分
    #    （干擾推論 select 模式只讀 dev；evaluate 模式只在 holdout 上量定案規則）
    rule_eval_path = output_dir / "disturbance_rule_features_eval.csv"
    df_rule[df_rule["split"] != "train"].to_csv(rule_eval_path, index=False, encoding="utf-8-sig")
    
    # 3. 除錯用：全部資料都留，含 split 欄
    if df_full is not None:
        full_path = output_dir / "disturbance_full_features_all.csv"
        df_full.to_csv(full_path, index=False, encoding="utf-8-sig")
    
    split_path = output_dir / "disturbance_session_split.csv"
    split_summary.to_csv(split_path, index=False, encoding="utf-8-sig")
    
    counts = split_summary["split"].value_counts()
    print("=" * 80)
    print("Disturbance train / dev / holdout Split 完成")
    print("=" * 80)
    print(f"holdout_per_group : {args.holdout_per_group}")
    print(f"dev_per_group     : {args.dev_per_group}")
    print(f"seed              : {args.seed}")
    print(f"session 數        ：train {counts.get('train', 0)}、dev {counts.get('dev', 0)}、holdout {counts.get('holdout', 0)}")
    print()
    print("各 disturbance_group 的分配：")
    print(split_summary.groupby(["disturbance_group", "split"]).size().unstack(fill_value=0))
    print()
    print(f"已輸出 模型訓練用   : {model_train_path}")
    print(f"已輸出 干擾推論用   : {rule_eval_path}")
    if df_full is not None:
        print(f"已輸出 除錯用 full  : {full_path}")
    print(f"已輸出 split 表     : {split_path}")



if __name__ == "__main__":
    main()