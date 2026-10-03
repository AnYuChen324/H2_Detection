#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
整體流程是這樣：

1.讀入 CSV，並檢查至少要有 record_id 和 t_start 兩個欄位。
2.先把每個 record_id 彙整成一列 summary，算出：
    n_windows：這個 record 有幾筆 window
    t_start_min / t_start_max：時間範圍
    distance_cm：從 record_id 字串裡解析出的距離
3.依 distance_cm 進行分層切分，80% record 作為 dev，20% 作為 holdout。
4.再用這些 record_id 回頭切原始資料列，輸出：
    transition_features_dev.csv
    transition_features_holdout.csv
    transition_record_split.csv
'''


from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 從 record_id 裡抓出像 _30cm_ 這種格式中的數字
# ============================================================
def parse_distance_cm(record_id: str) -> int:
    match = re.search(r"_(\d+)cm_", str(record_id))    # 用正規表示式在 record_id 中尋找像 _30cm_ 這種片段
    if not match:    # 如果沒有找到符合格式的內容，進入錯誤處理
        raise ValueError(f"無法從 record_id 解析 distance_cm: {record_id}")
    # 把正規表示式抓到的第一個括號群組，也就是數字部分取出並轉成整數回傳
    return int(match.group(1))   


# ============================================================
# 先把原始資料用 record_id 分組，做出每筆 record 的摘要表
# ============================================================
def build_record_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("record_id", as_index=False)    # 依 record_id 分組，as_index=False 表示分組鍵保留成普通欄位，而不是變成索引
        .agg(                                      # 對每個 record_id 做聚合
            n_windows=("record_id", "size"),       # 該 record 有幾列資料
            t_start_min=("t_start", "min"),        # 最小 t_start
            t_start_max=("t_start", "max"),        # 最大 t_start
        )
        .copy()
    )
    # 對 summary["record_id"] 每一列套用 parse_distance_cm，產生新欄位 distance_cm
    summary["distance_cm"] = summary["record_id"].map(parse_distance_cm)
    # 把距離整數轉成字串，再加上 "cm"，例如 30 變成 "30cm"
    summary["group_key"] = summary["distance_cm"].astype(str) + "cm"
    return summary.sort_values(["distance_cm", "record_id"]).reset_index(drop=True)


# ============================================================
# 定義切分函式,輸出dev,holdout,split summary
# ============================================================
def stratified_record_split(
    record_summary: pd.DataFrame,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    以 record_id 為單位，依 distance_cm 分層切分。

    Dev：80%
    Holdout：20%
    """

    if not 0 < holdout_ratio < 1:
        raise ValueError(
            "holdout_ratio 必須介於 0 與 1 之間，"
            f"目前收到：{holdout_ratio}"
        )

    dev_summary, holdout_summary = (
        train_test_split(
            record_summary,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=(
                record_summary[
                    "distance_cm"
                ]
            ),
        )
    )

    dev_ids = sorted(
        dev_summary[
            "record_id"
        ].tolist()
    )

    holdout_ids = sorted(
        holdout_summary[
            "record_id"
        ].tolist()
    )

    overlap_ids = (
        set(dev_ids) &
        set(holdout_ids)
    )

    if overlap_ids:
        raise ValueError(
            "Dev 與 Holdout 出現重複 "
            "record_id："
            f"{sorted(overlap_ids)[:20]}"
        )

    split_summary = (
        record_summary.copy()
    )

    split_summary["split"] = np.where(
        split_summary[
            "record_id"
        ].isin(holdout_ids),
        "holdout",
        "dev",
    )

    return (
        dev_ids,
        holdout_ids,
        split_summary,
    )



# ============================================================
# 主流程
# ============================================================
def main():
    # 建立命令列參數解析器，並設定這支工具的說明文字
    parser = argparse.ArgumentParser(
        description="把 transition features CSV 依 record_id 切成 dev / holdout。"
    )
    # 加入 --input-csv 參數
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/Users/paul/Desktop/Final/transition_feature_engineering/transition_30ml_test/inference_features_transition_model.csv"),
        help="輸入的 transition features CSV",
    )
    # 加入 --output-dir 參數:可以指定輸出資料夾, 若不指定，後面會用輸入 CSV 的資料夾
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="輸出資料夾，預設輸入 CSV 所在資料夾",
    )
    # 加入 --holdout-per-group，控制每個距離組抽幾筆去 holdout
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.20,
        help="Holdout 比例，預設為 0.20",
    )
    # 加入 --seed 參數，控制隨機抽樣結果
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機切分 seed，預設 42",
    )
    # 加入 --input-csv 參數：針對含有規則欄位的轉換資料
    parser.add_argument(
        "--rule-csv",
        type=Path,
        default=Path(
            "/Users/paul/Desktop/Final/transition_feature_engineering/"
            "transition_30ml_test/inference_features_transition_rule.csv"
        ),
        help="輸入的 transition rule features CSV，含 offband / 後處理規則欄位",
    )
    
    # 真正解析命令列輸入，把結果放進 args
    args = parser.parse_args()

    input_csv = args.input_csv       # 取出輸入 CSV 路徑
    output_dir = args.output_dir or input_csv.parent     # 如果使用者有指定 output_dir 就用它，否則用輸入檔所在資料夾
    output_dir.mkdir(parents=True, exist_ok=True)        # 建立輸出資料夾

   # ============================================================
    # 1. 讀取 model 版 transition features
    # ============================================================
    df = pd.read_csv(input_csv)
    required_cols = {"record_id", "t_start"}             # 定義必要欄位集合
    missing = sorted(required_cols - set(df.columns))    # 計算哪些必要欄位缺失
    if missing:
        raise ValueError(f"輸入 CSV 缺少必要欄位: {missing}")

    # 根據 model 版資料建立 record 層級摘要
    record_summary = build_record_summary(df)
    
    # 呼叫分層切分函式，只在這裡產生一次 dev / holdout record_id
    dev_ids, holdout_ids, split_summary = stratified_record_split(
        record_summary=record_summary,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    
    # 用同一組 ids 切 model 版資料
    df_dev = df[df["record_id"].isin(dev_ids)].copy()
    df_holdout = df[df["record_id"].isin(holdout_ids)].copy()
    
    dev_path = output_dir / "transition_features_dev.csv"
    holdout_path = output_dir / "transition_features_holdout.csv"
    split_path = output_dir / "transition_record_split.csv"

    df_dev.to_csv(dev_path, index=False, encoding="utf-8-sig")
    df_holdout.to_csv(holdout_path, index=False, encoding="utf-8-sig")
    split_summary.to_csv(split_path, index=False, encoding="utf-8-sig")
    
    
    # ============================================================
    # 2. 讀取 rule 版 transition features
    # ============================================================
    # 確認含有規則特徵資料集是否存在
    if not args.rule_csv.exists():
        raise FileNotFoundError(
            f"找不到 rule CSV，請先重跑 transition_feature_engineering_final2.py：{args.rule_csv}"
        )
    # 從含有規則欄位的資料中，根據 record_id 在 dev_ids 裡的列，形成開發集
    df_rule = pd.read_csv(args.rule_csv)

    missing_rule = sorted(required_cols - set(df_rule.columns))
    if missing_rule:
        raise ValueError(f"rule CSV 缺少必要欄位: {missing_rule}")
    
    # 確認 model 版與 rule 版是同一批 record
    model_record_ids = set(df["record_id"].unique())
    rule_record_ids = set(df_rule["record_id"].unique())
    
    if model_record_ids != rule_record_ids:
        only_model = sorted(model_record_ids - rule_record_ids)[:10]
        only_rule = sorted(rule_record_ids - model_record_ids)[:10]
        raise ValueError(
            "model CSV 和 rule CSV 的 record_id 不一致，不能共用 split。\n"
            f"只在 model 裡的前幾筆: {only_model}\n"
            f"只在 rule 裡的前幾筆: {only_rule}"
        )
        
    # 理論上 model/rule 是同一批 window，只是欄位不同，所以 row 數應相同
    if len(df_rule) != len(df):
        print(
            "⚠️ model CSV 和 rule CSV 的 row 數不同："
            f"model={len(df)}, rule={len(df_rule)}。"
            "如果 rule 版是從同一份 transition feature engineering 輸出，理論上應該相同。"
        )
    
    # 用同一組 ids 切 rule 版資料
    df_rule_dev = df_rule[df_rule["record_id"].isin(dev_ids)].copy()
    df_rule_holdout = df_rule[df_rule["record_id"].isin(holdout_ids)].copy()

    # 輸出路徑
    rule_dev_path = output_dir / "transition_rule_features_dev.csv"
    rule_holdout_path = output_dir / "transition_rule_features_holdout.csv"
    
    # 輸出檔案
    df_rule_dev.to_csv(rule_dev_path, index=False, encoding="utf-8-sig")
    df_rule_holdout.to_csv(rule_holdout_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Transition  Dev / Holdout Split 完成")
    print("=" * 80)
    print(f"input_csv           : {input_csv}")
    print(f"output_dir          : {output_dir}")
    print(
        f"dev_ratio           : "
        f"{1.0 - args.holdout_ratio:.2f}"
    )
    
    print(
        f"holdout_ratio       : "
        f"{args.holdout_ratio:.2f}"
    )
    print(f"seed                : {args.seed}")
    print(f"n_total_records     : {record_summary['record_id'].nunique()}")
    print(f"n_dev_records       : {len(dev_ids)}")
    print(f"n_holdout_records   : {len(holdout_ids)}")
    print(f"n_dev_windows       : {len(df_dev)}")
    print(f"n_holdout_windows   : {len(df_holdout)}")
    print()
    print("Holdout record_ids:")
    for rid in holdout_ids:
        print(f"  - {rid}")
    print()
    print("Split summary by distance:")
    print(
        split_summary.groupby(["distance_cm", "split"])["record_id"]
        .count()
        .unstack(fill_value=0)
    )
    print()
    print(f"已輸出 dev      : {dev_path}")
    print(f"已輸出 holdout  : {holdout_path}")
    print(f"已輸出 split 表 : {split_path}")
    print(f"已輸出 rule dev      : {rule_dev_path}")
    print(f"已輸出 rule holdout  : {rule_holdout_path}")

if __name__ == "__main__":
    main()
