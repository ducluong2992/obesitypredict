from ml.c45 import build_tree
import pandas as pd
import pickle

df = pd.read_csv('ObesityDataSet_encoded.csv')
target_col = 'NObeyesdad_Encoded'
features = [col for col in df.columns if col != target_col]

print("Training C4.5 tree...")
tree = build_tree(df, target_col, features)

with open('obesity_c45_model.pkl', 'wb') as f:
    pickle.dump(tree, f)

print("Đã lưu model vào 'obesity_c45_model.pkl'")
