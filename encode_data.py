import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

encoders = {}

for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

df.to_csv("obesity_encoded.csv", index=False)

joblib.dump(encoders, "encoders.pkl")

print("Đã encode xong! Tạo file: obesity_encoded.csv và encoders.pkl")
