from flask import Flask, request, jsonify
import numpy as np
from keras.src.saving import load_model

app = Flask(__name__)
model = load_model('myModel.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_data)
    result = 'Co tieu duong (1)' if prediction[0][0] > 0.5 else 'Khong tieu duong (0)'
    return jsonify({'prediction': prediction[0][0].tolist(), 'result': result})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
