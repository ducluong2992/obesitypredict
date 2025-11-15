from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model đã train
model = joblib.load("obesity_model.pkl")
eval_metrics = joblib.load("eval_metrics.pkl")


# mapping categorical
gender_map = {"Male": 1, "Female": 0}
family_map = {"yes": 1, "no": 0}
favc_map = {"yes": 1, "no": 0}
caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2}
smoke_map = {"yes": 1, "no": 0}
scc_map = {"yes": 1, "no": 0}
calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2}
mtrans_map = {"Walking": 0, "Bike": 1, "Motorbike": 2, "Public_Transport": 3, "Car": 4}

columns_order = ['Age','Gender','Height','Weight','CALC','FAVC','FCVC','NCP','SCC','SMOKE',
                 'CH2O','family_history_with_overweight','FAF','TUE','CAEC','MTRANS']

# chuyển nhãn sang tiếng Việt
obesity_labels = {
    0: 'Thiếu cân',
    1: 'Bình thường',
    2: 'Thừa cân cấp I',
    3: 'Thừa cân cấp II',
    4: 'Béo phì cấp I',
    5: 'Béo phì cấp II',
    6: 'Béo phì cấp III'
}


@app.route("/")
def index():
    return render_template("index.html", eval_metrics=eval_metrics)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        numeric_cols = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
        for col in numeric_cols:
            data[col] = float(data[col])

        data['Gender'] = gender_map[data['Gender']]
        data['family_history_with_overweight'] = family_map[data['family_history_with_overweight']]
        data['FAVC'] = favc_map[data['FAVC']]
        data['CAEC'] = caec_map[data['CAEC']]
        data['SMOKE'] = smoke_map[data['SMOKE']]
        data['SCC'] = scc_map[data['SCC']]
        data['CALC'] = calc_map[data['CALC']]
        data['MTRANS'] = mtrans_map[data['MTRANS']]
        df = pd.DataFrame([data])
        df = df[columns_order]

        pred = model.predict(df)[0]
        result = obesity_labels.get(pred, "Unknown")

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result="Lỗi: " + str(e))


if __name__ == "__main__":
    app.run(debug=True)
