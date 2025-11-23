import pandas as pd
from sklearn.preprocessing import LabelEncoder

#1. Tải dữ liệu
try:
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'ObesityDataSet_raw_and_data_sinthetic.csv'. Vui lòng kiểm tra đường dẫn.")
    exit()


# 2. Mã hóa Dữ liệu Danh mục Có Thứ tự (Ordinal Label Encoding)
# CAEC (Ăn vặt giữa bữa)
caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC_Encoded'] = df['CAEC'].map(caec_mapping)
# CALC (Uống rượu)
calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3} # Lưu ý: Dữ liệu mẫu chỉ có 'no', 'Sometimes', 'Frequently'
df['CALC_Encoded'] = df['CALC'].map(calc_mapping)

# NObeyesdad (Nhãn Mục tiêu) - Đây là bước quan trọng
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

#3. Mã hóa Dữ liệu Danh mục Không Thứ tự (Nominal Label Encoding)

# Các cột cần dùng LabelEncoder (Binary và Nominal)
nominal_cols = [
    'Gender', 
    'FAVC', 
    'SCC', 
    'SMOKE', 
    'family_history_with_overweight', 
    'MTRANS'
]

le = LabelEncoder()

for col in nominal_cols:
    df[f'{col}_Encoded'] = le.fit_transform(df[col])
    print(f"Mã hóa cho cột {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 4. Chuẩn bị Dữ liệu Cuối cùng

# Danh sách các cột đã được mã hóa (Features)
features_encoded = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'FAF', 'TUE', 'CH2O', # Số
    'Gender_Encoded', 'family_history_with_overweight_Encoded', 'FAVC_Encoded',
    'CAEC_Encoded', 'CALC_Encoded', 'SCC_Encoded', 'SMOKE_Encoded', 'MTRANS_Encoded'
]

# Tạo DataFrame chỉ chứa các cột đã được xử lý
df_encoded = df[features_encoded + ['NObeyesdad_Encoded']].copy()

# ----------------------------------------------------------------------
## 5. In kết quả và Lưu file (Tùy chọn)
# ----------------------------------------------------------------------

print("\n--- 5 Hàng Dữ liệu sau khi Mã hóa ---")
print(df_encoded.head())

# Lưu file đã mã hóa để sử dụng cho việc huấn luyện mô hình
df_encoded.to_csv('ObesityDataSet_encoded.csv', index=False)
print("\nĐã lưu dữ liệu đã mã hóa vào file: ObesityDataSet_encoded.csv")