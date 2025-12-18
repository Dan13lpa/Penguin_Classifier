
import pandas as pd
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Carga y limpieza de datos (Elimina filas con NAs)
df = sns.load_dataset("penguins")
df = df.dropna()  

# Separa variable objetivo y características
X = df.drop('species', axis=1)
y = df['species']

# Codificar la variable objetivo (especies) a números
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Entrenamiento/Prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

# (DictVectorizer para One-Hot + numéricas)
train_dicts = X_train.to_dict(orient='records')
test_dicts = X_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train_encoded = dv.fit_transform(train_dicts)
X_test_encoded = dv.transform(test_dicts)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Definición de los 4 modelos
models = {
    "logistic_regression": LogisticRegression(),
    "svm": SVC(probability=True),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier()
}

# Entrenamiento y Serialización
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    # Modelo, el vectorizador, el escalador y el encoder
    with open(f'models/{name}_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'dv': dv,
            'scaler': scaler,
            'le': le
        }, f)
    print(f"Modelo {name} entrenado y guardado.")