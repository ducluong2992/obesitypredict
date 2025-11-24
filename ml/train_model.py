from ml.c45 import build_tree, predict
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter


df = pd.read_csv('ObesityDataSet_encoded.csv')
target_col = 'NObeyesdad_Encoded'
features = [col for col in df.columns if col != target_col]

X = df[features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df[target_col] = y_train

print("Training C4.5 tree...")
tree = build_tree(train_df, target_col, list(X_train.columns))

with open('obesity_c45_model.pkl', 'wb') as f:
    pickle.dump(tree, f)

print("Đã lưu model vào 'obesity_c45_model.pkl'")
