from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
x = dataset[:,0:8]  # 8 đầu vào
y = dataset[:,8]    # 1 đầu ra (true/false)

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=200)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Voting Classifier
voting_model = VotingClassifier(
    estimators=[
        ('logistic', logistic_model),
        ('random_forest', random_forest_model)
    ],
    voting='soft'  # soft voting (dựa trên xác suất dự đoán của các mô hình)
)

voting_model.fit(x_train, y_train)

pkl_filename = 'myModel.pkl'
joblib.dump(voting_model, pkl_filename)

y_pred = voting_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test data: {:.2f}%".format(accuracy * 100))

# Predict new data
x_new = x_test[10].reshape(1, -1)  # chọn phần tử thứ 10 trong test set
y_new = y_test[10]

loaded_model = joblib.load(pkl_filename)
y_predict_loaded = loaded_model.predict(x_new)

result_loaded = "Có tiểu đường (1)" if y_predict_loaded[0] == 1 else "Không tiểu đường (0)"
print("Dự đoán với mô hình đã lưu =", y_predict_loaded[0], "|", result_loaded)
print("Giá trị thực =", y_new)

