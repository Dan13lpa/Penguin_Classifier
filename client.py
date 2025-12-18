
import requests
import json

# URL servidor Flask
URL_BASE = "http://127.0.0.1:5000/predict/"

# Casos de prueba 
test_cases = [
    # 1. Regresión Logística: Pingüino grande y pesado (Probable Gentoo)
    {"model": "logistic_regression", "data": {
        "island": "Biscoe", "bill_length_mm": 50.5, "bill_depth_mm": 15.2, 
        "flipper_length_mm": 230.0, "body_mass_g": 5500.0, "sex": "Male"
    }, "desc": "Macho grande de Biscoe (Perfil Gentoo)"},

    # 2. Regresión Logística: Cría o ejemplar pequeño
    {"model": "logistic_regression", "data": {
        "island": "Torgersen", "bill_length_mm": 36.0, "bill_depth_mm": 17.0, 
        "flipper_length_mm": 180.0, "body_mass_g": 3100.0, "sex": "Female"
    }, "desc": "Hembra pequeña de Torgersen (Perfil Adélie)"},

    # 3. SVM: Pico largo pero delgado
    {"model": "svm", "data": {
        "island": "Dream", "bill_length_mm": 52.0, "bill_depth_mm": 19.0, 
        "flipper_length_mm": 195.0, "body_mass_g": 3800.0, "sex": "Male"
    }, "desc": "Macho con pico largo en Dream (Perfil Chinstrap)"},

    # 4. SVM: Dimensiones medias en Biscoe
    {"model": "svm", "data": {
        "island": "Biscoe", "bill_length_mm": 45.0, "bill_depth_mm": 14.5, 
        "flipper_length_mm": 210.0, "body_mass_g": 4600.0, "sex": "Female"
    }, "desc": "Hembra de Biscoe con aletas largas"},

    # 5. Decision Tree: Extremadamente ligero
    {"model": "decision_tree", "data": {
        "island": "Torgersen", "bill_length_mm": 34.5, "bill_depth_mm": 18.5, 
        "flipper_length_mm": 175.0, "body_mass_g": 2800.0, "sex": "Female"
    }, "desc": "Ejemplar muy ligero (Límite inferior Adélie)"},

    # 6. Decision Tree: Proporciones de pico cuadradas
    {"model": "decision_tree", "data": {
        "island": "Dream", "bill_length_mm": 40.0, "bill_depth_mm": 20.0, 
        "flipper_length_mm": 190.0, "body_mass_g": 4000.0, "sex": "Male"
    }, "desc": "Pico corto y grueso en Dream"},

    # 7. KNN: El "vecino" de un Gentoo estándar
    {"model": "knn", "data": {
        "island": "Biscoe", "bill_length_mm": 48.0, "bill_depth_mm": 15.0, 
        "flipper_length_mm": 220.0, "body_mass_g": 5100.0, "sex": "Male"
    }, "desc": "Macho robusto en Biscoe"},

    # 8. KNN: El "vecino" de un Chinstrap
    {"model": "knn", "data": {
        "island": "Dream", "bill_length_mm": 50.0, "bill_depth_mm": 18.5, 
        "flipper_length_mm": 200.0, "body_mass_g": 3700.0, "sex": "Female"
    }, "desc": "Hembra con pico estilizado en Dream"}
]

print("--- INICIANDO PRUEBAS DEL CLIENTE ---")
for i, test in enumerate(test_cases, 1):
    url = URL_BASE + test['model']
    response = requests.post(url, json=test['data'])
    
    if response.status_code == 200:
        res_json = response.json()
        print(f"Petición #{i} | Modelo: {test['model']} | Caso: {test['desc']}")
        print(f"   PREDICCIÓN: {res_json['prediction']}")
    else:
        print(f"Error en petición #{i}")
    print("-" * 50)