#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
整體流程是這樣：

1. 讀入 disturbance 的 model / rule（可選 full）三種特徵表
2. 以 model csv 為基準，檢查至少要有：
   session_id, disturbance_group, t_start
3. 先把每個 session_id 彙整成一列 summary，算出：
   n_windows
   t_start_min / t_start_max
   disturbance_group
   disturbance_id
4. 依 disturbance_group 分組，從每組隨機抽出 holdout_per_group 筆 session 當 holdout，其餘進 dev
5. 用同一組 session_id，同步切：
   - model csv
   - rule csv
   - full csv（可選）
6. 輸出：
   disturbance_features_dev.csv
   disturbance_features_holdout.csv
   disturbance_rule_features_dev.csv
   disturbance_rule_features_holdout.csv
   disturbance_full_features_dev.csv（可選）
   disturbance_full_features_holdout.csv（可選）
   disturbance_session_split.csv

用途區分：
1. model 版：
   只保留模型訓練需要的欄位，給模型訓練與 transition 合併使用
2. rule 版：
   保留模型欄位 + offband 規則欄位，給干擾推論與規則比較使用
3. full 版：
   保留全部欄位，供除錯與核對使用
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
# 定義切分函式，輸出 train, holdout, split summary
# ============================================================
def stratified_session_split(
    session_summary: pd.DataFrame,
    holdout_per_group: int,
    seed: int,
) -> tuple[list[str], list[str], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    holdout_ids: list[str] = []

    # 按 disturbance_group 分組，逐組處理
    for group_key, sub in session_summary.groupby("group_key", sort=True):
        session_ids = sorted(sub["session_id"].tolist())

        if len(session_ids) <= holdout_per_group:
            raise ValueError(
                f"{group_key} 這組只有 {len(session_ids)} 筆 session，"
                f"無法留 {holdout_per_group} 筆給 holdout。"
            )

        chosen = sorted(rng.choice(session_ids, size=holdout_per_group, replace=False).tolist())
        holdout_ids.extend(chosen)

    holdout_ids = sorted(holdout_ids)
    train_ids = sorted(set(session_summary["session_id"]) - set(holdout_ids))

    split_summary = session_summary.copy()
    split_summary["split"] = np.where(
        split_summary["session_id"].isin(holdout_ids),
        "holdout",
        "dev",
    )

    return train_ids, holdout_ids, split_summary


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="把 disturbance features CSV 依 session_id 切成 dev / holdout。"
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

    dev_ids, holdout_ids, split_summary = stratified_session_split(
        session_summary=session_summary,
        holdout_per_group=args.holdout_per_group,
        seed=args.seed,
    )

    df_model_dev = df_model[df_model["session_id"].isin(dev_ids)].copy()
    df_model_holdout = df_model[df_model["session_id"].isin(holdout_ids)].copy()
    
    df_rule_dev = df_rule[df_rule["session_id"].isin(dev_ids)].copy()
    df_rule_holdout = df_rule[df_rule["session_id"].isin(holdout_ids)].copy()
    
    if df_full is not None:
        df_full_dev = df_full[df_full["session_id"].isin(dev_ids)].copy()
        df_full_holdout = df_full[df_full["session_id"].isin(holdout_ids)].copy()

    model_dev_path = output_dir / "disturbance_features_dev.csv"
    model_holdout_path = output_dir / "disturbance_features_holdout.csv"
    
    rule_dev_path = output_dir / "disturbance_rule_features_dev.csv"
    rule_holdout_path = output_dir / "disturbance_rule_features_holdout.csv"
    
    split_path = output_dir / "disturbance_session_split.csv"
    
    df_model_dev.to_csv(model_dev_path, index=False, encoding="utf-8-sig")
    df_model_holdout.to_csv(model_holdout_path, index=False, encoding="utf-8-sig")
    
    df_rule_dev.to_csv(rule_dev_path, index=False, encoding="utf-8-sig")
    df_rule_holdout.to_csv(rule_holdout_path, index=False, encoding="utf-8-sig")
    
    if df_full is not None:
        full_dev_path = output_dir / "disturbance_full_features_dev.csv"
        full_holdout_path = output_dir / "disturbance_full_features_holdout.csv"
        df_full_dev.to_csv(full_dev_path, index=False, encoding="utf-8-sig")
        df_full_holdout.to_csv(full_holdout_path, index=False, encoding="utf-8-sig")
    
    split_summary.to_csv(split_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Disturbance Dev / Holdout Split 完成")
    print("=" * 80)
    print(f"model_csv            : {model_csv}")
    print(f"rule_csv             : {rule_csv}")
    print(f"full_csv             : {full_csv if df_full is not None else 'None'}")
    print(f"output_dir           : {output_dir}")
    print(f"holdout_per_group    : {args.holdout_per_group}")
    print(f"seed                 : {args.seed}")
    print(f"n_total_sessions     : {session_summary['session_id'].nunique()}")
    print(f"n_dev_sessions       : {len(dev_ids)}")
    print(f"n_holdout_sessions   : {len(holdout_ids)}")
    print(f"n_model_dev_windows  : {len(df_model_dev)}")
    print(f"n_model_holdout_win  : {len(df_model_holdout)}")
    print(f"n_rule_dev_windows   : {len(df_rule_dev)}")
    print(f"n_rule_holdout_win   : {len(df_rule_holdout)}")
    print()
    
    print(f"已輸出 model dev     : {model_dev_path}")
    print(f"已輸出 model holdout : {model_holdout_path}")
    print(f"已輸出 rule dev      : {rule_dev_path}")
    print(f"已輸出 rule holdout  : {rule_holdout_path}")
    if df_full is not None:
        print(f"已輸出 full dev      : {full_dev_path}")
        print(f"已輸出 full holdout  : {full_holdout_path}")
    print(f"已輸出 split 表      : {split_path}")



if __name__ == "__main__":
    main()