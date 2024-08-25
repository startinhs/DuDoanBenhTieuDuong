from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('myModel.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    
    prediction = model.predict(input_data)
    result = 'Có tiểu đường (1)' if prediction[0] == 1 else 'Không tiểu đường (0)'
    
    return jsonify({'prediction': int(prediction[0]), 'result': result})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
