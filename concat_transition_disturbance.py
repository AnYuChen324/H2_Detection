#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
整體流程：

1. 讀入切分後的 transition dev 與 disturbance dev 模型版特徵表
2. 讀取 feature_config_micro.json 內的 selected_feature_names
3. 檢查兩份 dev 資料是否都包含模型訓練必要欄位：
   - record_id
   - label
   - selected_feature_names
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
import json  # 記得在檔案最上面加這行 import

transition_dev_csv = Path("/Users/paul/Desktop/Final/transition_feature_engineering/transition_30ml_test/transition_features_dev.csv")
disturbance_dev_csv = Path("/Users/paul/Desktop/Final/disturbance_feature_engineering/disturbance_features_dev.csv")
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
df_transition_dev["source_dataset"] = "transition_dev"
df_disturbance_dev["source_dataset"] = "disturbance_dev"

# 干擾資料全部設成 non-leak
df_disturbance_dev["label"] = 0

# 如果 disturbance 的 record_id 不完整，就用 session_id 補
if "record_id" not in df_disturbance_dev.columns and "session_id" in df_disturbance_dev.columns:
    df_disturbance_dev["record_id"] = df_disturbance_dev["session_id"]

# 模型訓練必要欄位
train_cols = ["record_id", "label"] + selected_features

# 檢查欄位
for name, df_ in {
    "transition_dev": df_transition_dev,
    "disturbance_dev": df_disturbance_dev,
}.items():
    missing = [c for c in train_cols if c not in df_.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要欄位: {missing}")

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