#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
整體流程：

1. 讀入切分後的 transition train 與 disturbance train 模型版特徵表（變數名和輸出檔名 _dev 是沿用舊名，內容是 train 份）
2. 讀取 feature_config_micro.json 內的 selected_feature_names
3. 檢查兩份 dev 資料是否都包含模型訓練必要欄位：
   - record_id
   - label
   - selected_feature_names
   - session_id
4. 只保留模型訓練需要的欄位後做合併
5. 額外保留一份含 metadata 的合併版本，供分析與除錯使用
6. 輸出：
   - transition_disturbance_dev.csv
   - transition_disturbance_dev_meta.csv

用途：
1. transition_disturbance_dev.csv
   給模型訓練程式使用
2. transition_disturbance_dev_meta.csv
   給資料分析、來源追蹤、除錯使用
'''


from pathlib import Path
import pandas as pd
import json  

# 兩支切分程式的 train 份（模型訓練用）；dev / holdout 留給推論選規則和評估，不能拿來訓練
transition_dev_csv = Path("/Users/paul/Desktop/Final/transition_feature_engineering/transition_30ml_test/transition_features_train.csv")
disturbance_dev_csv = Path("/Users/paul/Desktop/Final/disturbance_feature_engineering/disturbance_features_train.csv")
selected_json = Path("/Users/paul/Desktop/Final/test_result/feature_config_micro.json")

output_dir = Path("/Users/paul/Desktop/Final/test_result")
output_dir.mkdir(parents=True, exist_ok=True)

output_dev_csv = output_dir / "transition_disturbance_dev.csv"
output_dev_meta_csv = output_dir / "transition_disturbance_dev_meta.csv"

# 讀資料
df_transition_dev = pd.read_csv(transition_dev_csv)
df_disturbance_dev = pd.read_csv(disturbance_dev_csv)

# 讀 selected features
feature_config = json.loads(selected_json.read_text(encoding="utf-8"))
selected_features = feature_config.get("selected_feature_names", [])

if not selected_features:
    raise ValueError("feature_config_micro.json 裡的 selected_feature_names 是空的")

# 來源欄位
df_transition_dev["source_dataset"] = "transition_train"
df_disturbance_dev["source_dataset"] = "disturbance_train"


# 防呆 1：只能用 train 份訓練（dev / holdout 是推論選規則和評估用的，模型不能看過）
# 沒有 split 欄、有空值、或不是 train，一律停止
for name, df_ in (("transition", df_transition_dev), ("disturbance", df_disturbance_dev)):
    if "split" not in df_.columns:
        raise ValueError(f"{name} 沒有 split 欄，無法確認是不是 train 份；請用目前的切分程式重新輸出")
    n_missing = int(df_["split"].isna().sum())
    if n_missing:
        raise ValueError(f"{name} 的 split 欄有 {n_missing} 個空值，無法確認是不是 train 份")
    wrong = sorted(set(df_["split"].astype(str)) - {"train"})
    if wrong:
        raise ValueError(f"{name} 訓練資料裡出現 split = {wrong}，只能用 train 份訓練，請確認輸入檔路徑")

# 防呆 2：灰色地帶（label 是空值）不拿來訓練
n_gray = int(df_transition_dev["label"].isna().sum()) if "label" in df_transition_dev.columns else 0
if n_gray:
    print(f"transition 有 {n_gray} 窗 label 是空值（灰色地帶），不拿來訓練，已排除")
    df_transition_dev = df_transition_dev[df_transition_dev["label"].notna()].copy()

# 干擾資料全部是 non-leak：先確認原本的 label 沒有非 0 的值（有的話代表讀錯檔），再統一設成 0
if "label" in df_disturbance_dev.columns:
    bad_label = df_disturbance_dev["label"].notna() & (df_disturbance_dev["label"] != 0)
    if bad_label.any():
        raise ValueError(f"disturbance 訓練資料有 {int(bad_label.sum())} 窗 label 不是 0，請確認輸入檔路徑")
df_disturbance_dev["label"] = 0

# 如果 disturbance 的 record_id 不完整，就用 session_id 補
if "record_id" not in df_disturbance_dev.columns and "session_id" in df_disturbance_dev.columns:
    df_disturbance_dev["record_id"] = df_disturbance_dev["session_id"]

# session_id：訓練程式內部切 validation / CV 的分組單位（和 split 程式一致，同一個 session 的段不能分到兩邊）
train_cols = ["record_id", "session_id", "label"] + selected_features

# 檢查欄位
for name, df_ in {
    "transition_dev": df_transition_dev,
    "disturbance_dev": df_disturbance_dev,
}.items():
    missing = [c for c in train_cols if c not in df_.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要欄位: {missing}")
    
    if df_["session_id"].isna().any():
        raise ValueError(f"{name} 的 session_id 有空值，無法依 session 分組")

# 給訓練用：只留訓練欄位
df_transition_train = df_transition_dev[train_cols].copy()
df_disturbance_train = df_disturbance_dev[train_cols].copy()

# 合併訓練資料
df_train_merged = pd.concat(
    [df_transition_train, df_disturbance_train],
    axis=0,
    ignore_index=True,
).sample(frac=1.0, random_state=42).reset_index(drop=True)


# 額外保留一份 meta 版方便分析
meta_cols = [
    "source_dataset",
    "session_id",
    "disturbance_group",
    "disturbance_id",
    "disturbance_type",
    "eval_class",
    "t_start",
    "onset_gt_sec",
    "leak_quantity",
    "distance_cm",
    "pressure_bar",
]

for col in meta_cols:
    if col not in df_transition_dev.columns:
        df_transition_dev[col] = pd.NA
    if col not in df_disturbance_dev.columns:
        df_disturbance_dev[col] = pd.NA

meta_keep_cols = ["record_id", "label"] + meta_cols + selected_features

df_transition_meta = df_transition_dev[meta_keep_cols].copy()
df_disturbance_meta = df_disturbance_dev[meta_keep_cols].copy()

df_train_merged_meta = pd.concat(
    [df_transition_meta, df_disturbance_meta],
    axis=0,
    ignore_index=True,
).sample(frac=1.0, random_state=42).reset_index(drop=True)

# 輸出
df_train_merged.to_csv(output_dev_csv, index=False, encoding="utf-8-sig")
df_train_merged_meta.to_csv(output_dev_meta_csv, index=False, encoding="utf-8-sig")

print("=" * 80)
print("Merged dev dataset 完成")
print("=" * 80)
print(f"transition_dev rows    : {len(df_transition_dev)}")
print(f"disturbance_dev rows   : {len(df_disturbance_dev)}")
print(f"merged train rows      : {len(df_train_merged)}")
print()
print("label 分布：")
print(df_train_merged['label'].value_counts(dropna=False))
print()
print(f"已輸出 dev       : {output_dev_csv}")
print(f"已輸出 dev+meta  : {output_dev_meta_csv}")