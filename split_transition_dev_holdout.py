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
# 先把原始資料依「切分單位」分組，做出每個單位的摘要表
# ============================================================
def build_record_summary(df: pd.DataFrame, unit_col: str = "record_id") -> pd.DataFrame:
    """
    切分單位：
      - 有 session_id 就用 session_id（同一個 session 的所有 record 放同一份，最不會洩漏）
      - 沒有就用 record_id（舊資料：一筆 transition record 就是一次連續錄製）
    分層依據：distance_cm（從 record_id 解析）；有 leak_quantity 欄位時再加上洩漏量
    """
    summary = (
        df.groupby(unit_col, as_index=False)
        .agg(
            n_windows=(unit_col, "size"),
            t_start_min=("t_start", "min"),
            t_start_max=("t_start", "max"),
            record_id=("record_id", "first"),
        )
        .copy()
    )
    summary["distance_cm"] = summary["record_id"].map(parse_distance_cm)
    summary["group_key"] = summary["distance_cm"].astype(str) + "cm"
    if "leak_quantity" in df.columns:
        qty = df.groupby(unit_col)["leak_quantity"].max()
        summary["leak_quantity"] = summary[unit_col].map(qty)
        summary["group_key"] = summary["group_key"] + "_" + summary["leak_quantity"].astype(str) + "ml"
    return summary.sort_values(["group_key", unit_col]).reset_index(drop=True)


# ============================================================
# 依 group_key 分層，把單位分成 train / dev / holdout 三份
# ============================================================
def stratified_record_split(
    record_summary: pd.DataFrame,
    holdout_ratio: float,
    dev_ratio: float,
    seed: int,
    unit_col: str = "record_id",
) -> pd.DataFrame:
    """
    三份的用途（和干擾資料一致）：
      train   ：訓練模型
      dev     ：選規則（轉換推論在 dev 上量偵測率，和干擾 dev 的誤報率一起看）
      holdout ：規則定案後量一次成績

    每個 group_key 各自打亂後，依比例分配，每份至少 1 個單位（至少要有 3 個單位才分得開）。
    """
    if not (0 < holdout_ratio < 1 and 0 < dev_ratio < 1 and holdout_ratio + dev_ratio < 1):
        raise ValueError(f"比例不合理：holdout={holdout_ratio}, dev={dev_ratio}")

    rng = np.random.default_rng(seed)
    split_map: dict[str, str] = {}
    for group_key, sub in record_summary.groupby("group_key", sort=True):
        ids = sorted(sub[unit_col].tolist())
        n = len(ids)
        n_h = max(1, int(round(n * holdout_ratio)))
        n_d = max(1, int(round(n * dev_ratio)))
        if n_h + n_d >= n:
            raise ValueError(
                f"{group_key} 只有 {n} 個 {unit_col}，不夠分成 holdout {n_h} + dev {n_d} + train 至少 1"
            )
        shuffled = rng.permutation(ids).tolist()
        for u in shuffled[:n_h]:
            split_map[u] = "holdout"
        for u in shuffled[n_h:n_h + n_d]:
            split_map[u] = "dev"
        for u in shuffled[n_h + n_d:]:
            split_map[u] = "train"

    split_summary = record_summary.copy()
    split_summary["split"] = split_summary[unit_col].map(split_map)
    return split_summary


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
    
    # 加入 --dev-ratio，控制 dev（選規則用）的比例
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.20,
        help="Dev（選規則用）比例，預設為 0.20",
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

        # 切分單位：有 session_id 用 session_id，沒有就用 record_id
    unit_col = "session_id" if "session_id" in df.columns else "record_id"
    record_summary = build_record_summary(df, unit_col=unit_col)

    # 只在這裡產生一次 train / dev / holdout
    split_summary = stratified_record_split(
        record_summary=record_summary,
        holdout_ratio=args.holdout_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        unit_col=unit_col,
    )
    split_map = dict(zip(split_summary[unit_col], split_summary["split"]))

    # ============================================================
    # 2. 讀取 rule 版 transition features，確認和 model 版是同一批
    # ============================================================
    if not args.rule_csv.exists():
        raise FileNotFoundError(
            f"找不到 rule CSV，請先重跑 transition_feature_engineering_final2.py：{args.rule_csv}"
        )
    df_rule = pd.read_csv(args.rule_csv)
    missing_rule = sorted(required_cols - set(df_rule.columns))
    if missing_rule:
        raise ValueError(f"rule CSV 缺少必要欄位: {missing_rule}")
    if unit_col not in df_rule.columns:
        raise ValueError(f"rule CSV 沒有 {unit_col} 欄，無法和 model 版用同一種單位切分")

    model_units = set(df[unit_col].unique())
    rule_units = set(df_rule[unit_col].unique())
    if model_units != rule_units:
        raise ValueError(
            f"model CSV 和 rule CSV 的 {unit_col} 不一致，不能共用 split。\n"
            f"只在 model 裡的前幾筆: {sorted(model_units - rule_units)[:10]}\n"
            f"只在 rule 裡的前幾筆: {sorted(rule_units - model_units)[:10]}"
        )
    if len(df_rule) != len(df):
        print(f"⚠️ model CSV 和 rule CSV 的 row 數不同：model={len(df)}, rule={len(df_rule)}")

    # 兩個版本都加上 split 欄，後面的程式一律看這一欄
    df["split"] = df[unit_col].map(split_map)
    df_rule["split"] = df_rule[unit_col].map(split_map)

    # 1. 模型訓練用：只有 train
    train_path = output_dir / "transition_features_train.csv"
    df[df["split"] == "train"].to_csv(train_path, index=False, encoding="utf-8-sig")

    # 2. 轉換推論用：dev + holdout 放同一份，靠 split 欄區分
    rule_eval_path = output_dir / "transition_rule_features_eval.csv"
    df_rule[df_rule["split"] != "train"].to_csv(rule_eval_path, index=False, encoding="utf-8-sig")

    split_path = output_dir / "transition_record_split.csv"
    split_summary.to_csv(split_path, index=False, encoding="utf-8-sig")

    counts = split_summary["split"].value_counts()
    print("=" * 80)
    print("Transition train / dev / holdout Split 完成")
    print("=" * 80)
    print(f"input_csv     : {input_csv}")
    print(f"切分單位      : {unit_col}")
    print(f"比例          : train {1 - args.holdout_ratio - args.dev_ratio:.2f}、"
          f"dev {args.dev_ratio:.2f}、holdout {args.holdout_ratio:.2f}")
    print(f"seed          : {args.seed}")
    print(f"{unit_col} 數 ：train {counts.get('train', 0)}、dev {counts.get('dev', 0)}、holdout {counts.get('holdout', 0)}")
    print()
    print("各組分配：")
    print(split_summary.groupby(["group_key", "split"]).size().unstack(fill_value=0))
    print()
    print(f"已輸出 模型訓練用 : {train_path}")
    print(f"已輸出 轉換推論用 : {rule_eval_path}")
    print(f"已輸出 split 表   : {split_path}")

if __name__ == "__main__":
    main()
