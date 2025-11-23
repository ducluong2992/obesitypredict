import pandas as pd
from c45 import build_tree,predict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dữ liệu encode
df = pd.read_csv('ObesityDataSet_encoded.csv')
target_col = 'NObeyesdad_Encoded'
features = [col for col in df.columns if col != target_col]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []

for i, (train_idx, test_idx) in enumerate(kf.split(df), 1):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    tree = build_tree(train_df, target_col, features)

    y_true = test_df[target_col].tolist()
    y_pred = [predict(tree, test_df.iloc[j].to_dict()) for j in range(len(test_df))]

    acc = round(accuracy_score(y_true, y_pred)*100, 2)
    report = classification_report(y_true, y_pred, zero_division=0)

    folds.append({
        "number": i,
        "accuracy": acc,
        "report": report
    })

mean_accuracy = round(sum([f['accuracy'] for f in folds])/len(folds), 2)

results = {
    "folds": folds,
    "mean_accuracy": mean_accuracy
}

# Lưu kết quả
joblib.dump(results, "evaluation_results.pkl")
print("Đã lưu kết quả đánh giá vào 'evaluation_results.pkl'")
