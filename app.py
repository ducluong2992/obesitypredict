from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib
from ml import c45 

app = Flask(__name__)

with open("obesity_c45_model.pkl", "rb") as f:
    tree = pickle.load(f)

# Các cột số
numeric_cols = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']

# Thứ tự cột theo dữ liệu đã train (cột encode)
columns_order = [
    'Age',
    'Gender_Encoded',
    'Height',
    'Weight',
    'CALC_Encoded',
    'FAVC_Encoded',
    'FCVC',
    'NCP',
    'SCC_Encoded',
    'SMOKE_Encoded',
    'CH2O',
    'family_history_with_overweight_Encoded',
    'FAF',
    'TUE',
    'CAEC_Encoded',
    'MTRANS_Encoded'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def show_predict_form():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # 1. Convert numeric
        for col in numeric_cols:
            if col in data:
                data[col] = float(data[col])

        # 2. Map các cột ordinal
        caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

        # 3. Map các cột binary / categorical
        gender_map = {'Male': 1, 'Female': 0}
        yes_no_map = {'yes': 1, 'no': 0}
        mtrans_map = {'Walking':0, 'Bike':1, 'Motorbike':2, 'Public_Transport':3, 'Car':4}

        # 4. Tạo các cột encode
        encoded_data = {
            'Age': data['Age'],
            'Gender_Encoded': gender_map[data['Gender']],
            'Height': data['Height'],
            'Weight': data['Weight'],
            'CALC_Encoded': calc_mapping[data['CALC']],
            'FAVC_Encoded': yes_no_map[data['FAVC']],
            'FCVC': data['FCVC'],
            'NCP': data['NCP'],
            'SCC_Encoded': yes_no_map[data['SCC']],
            'SMOKE_Encoded': yes_no_map[data['SMOKE']],
            'CH2O': data['CH2O'],
            'family_history_with_overweight_Encoded': yes_no_map[data['family_history_with_overweight']],
            'FAF': data['FAF'],
            'TUE': data['TUE'],
            'CAEC_Encoded': caec_mapping[data['CAEC']],
            'MTRANS_Encoded': mtrans_map[data['MTRANS']]
        }

        # 5. Chuẩn bị DataFrame đúng thứ tự cột
        df = pd.DataFrame([encoded_data])
        df = df[columns_order]

        # 6. Dự đoán bằng c45
        sample = df.iloc[0].to_dict()
        pred_numeric = c45.predict(tree, sample)

        # 7. Map sang nhãn tiếng Việt
        obesity_labels_vi = {
            0: 'Thiếu cân',
            1: 'Bình thường',
            2: 'Thừa cân cấp I',
            3: 'Thừa cân cấp II',
            4: 'Béo phì cấp I',
            5: 'Béo phì cấp II',
            6: 'Béo phì cấp III'
        }
        result = obesity_labels_vi.get(pred_numeric, "Unknown")

        return render_template("predict.html", result=result)

    except Exception as e:
        return render_template("predict.html", result="Lỗi: " + str(e))


@app.route("/evaluate")
def evaluation():
    results = joblib.load("evaluation_results.pkl")
    folds = results.get("folds", [])
    mean_accuracy = results.get("mean_accuracy", 0)

    return render_template("evaluate.html", folds=folds, mean_accuracy=mean_accuracy)


if __name__ == "__main__":
    app.run(debug=True)
