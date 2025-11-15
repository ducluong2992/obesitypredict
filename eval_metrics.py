from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

# Load dữ liệu đã encode
df = pd.read_csv("obesity_encoded.csv")
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = joblib.load("obesity_model.pkl")

# Dự đoán trên test
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)


report = classification_report(y_test, y_pred)
eval_metrics = {
    "accuracy": accuracy,
    "report": report
}
joblib.dump(eval_metrics, "eval_metrics.pkl")
