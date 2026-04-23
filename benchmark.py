import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

print("=" * 50)
print("LightGBM Benchmark - Credit Card Fraud Detection")
print("=" * 50)

# 1. Load data
print("\n[1/4] Loading data...")
t0 = time.time()
df = pd.read_csv("~/ml-benchmark/creditcard.csv")
load_time = time.time() - t0
print(f"    Loaded {len(df):,} rows in {load_time:.2f}s")

# 2. Prepare
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train
print("\n[2/4] Training LightGBM...")
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "class_weight": "balanced",
    "n_jobs": -1,
    "verbose": -1,
}
t1 = time.time()
model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)
train_time = time.time() - t1
print(f"    Training done in {train_time:.2f}s | Best iteration: {model.best_iteration_}")

# 4. Evaluate
print("\n[3/4] Evaluating...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc       = roc_auc_score(y_test, y_pred_proba)
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

# 5. Inference latency
print("\n[4/4] Measuring inference latency...")
single_row = X_test.iloc[[0]]
runs = 100
t2 = time.time()
for _ in range(runs):
    model.predict_proba(single_row)
latency_1row_ms = (time.time() - t2) / runs * 1000

batch_1000 = X_test.iloc[:1000]
t3 = time.time()
model.predict_proba(batch_1000)
throughput_1000_ms = (time.time() - t3) * 1000

# 6. Save results
results = {
    "dataset": "creditcard-fraud-284807-rows",
    "model": "LightGBM GBDT",
    "machine_type": "n2-standard-8",
    "load_data_sec": round(load_time, 3),
    "training_time_sec": round(train_time, 3),
    "best_iteration": model.best_iteration_,
    "auc_roc": round(auc, 6),
    "accuracy": round(accuracy, 6),
    "f1_score": round(f1, 6),
    "precision": round(precision, 6),
    "recall": round(recall, 6),
    "inference_latency_1row_ms": round(latency_1row_ms, 3),
    "inference_throughput_1000rows_ms": round(throughput_1000_ms, 3),
}

with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
for k, v in results.items():
    print(f"  {k:<40} {v}")

print("\n✓ Saved to benchmark_result.json")
