import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Obesity_cleaned.csv')

#Mã hóa dữ liệu

#category có thứ tự sang số
caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC_Encoded'] = df['CAEC'].map(caec_mapping)

calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3} 
df['CALC_Encoded'] = df['CALC'].map(calc_mapping)

# mã hóa nhãn NObeyesdad
nobeyesdad_mapping = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
df['NObeyesdad_Encoded'] = df['NObeyesdad'].map(nobeyesdad_mapping)

#category không có thứ tự

nominal_cols = [ 'Gender', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'MTRANS']
le = LabelEncoder()  #encode bằng LabelEncoder
for col in nominal_cols:
    df[f'{col}_Encoded'] = le.fit_transform(df[col])
    print(f"Mã hóa cho cột {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

#chuẩn bị dữ liệu sau mã hóa
features_encoded = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'FAF', 'TUE', 'CH2O', 
    'Gender_Encoded', 'family_history_with_overweight_Encoded', 'FAVC_Encoded',
    'CAEC_Encoded', 'CALC_Encoded', 'SCC_Encoded', 'SMOKE_Encoded', 'MTRANS_Encoded'
]
df_encoded = df[features_encoded + ['NObeyesdad_Encoded']].copy() 

#kết quả
df_encoded.to_csv('ObesityDataSet_encoded.csv', index=False)
print("\nĐã lưu dữ liệu đã mã hóa vào file: ObesityDataSet_encoded.csv")