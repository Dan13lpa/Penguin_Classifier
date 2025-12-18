# Servidor Flask
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Función para cargar y predecir
def predict_penguin(model_name, data):
    with open(f'models/{model_name}_model.pkl', 'rb') as f:
        package = pickle.load(f)
    
    # Transformar los datos de entrada
    data_df = pd.DataFrame([data])
    data_dict = data_df.to_dict(orient='records')
    X_enc = package['dv'].transform(data_dict)
    X_scaled = package['scaler'].transform(X_enc)
    
    # Predecir
    prediction = package['model'].predict(X_scaled)
    species = package['le'].inverse_transform(prediction)
    return species[0]

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    data = request.get_json()
    result = predict_penguin(model_type, data)
    return jsonify({'model': model_type, 'prediction': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)