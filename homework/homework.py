# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import os
import json
import gzip
import pickle

# Paso 1: Cargar y limpiar los datos
train_data = pd.read_csv('files/input/train_data.csv.zip')
test_data = pd.read_csv('files/input/test_data.csv.zip')

def clean_data(df):
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    df = df.loc[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
    return df

train_data = clean_data(train_data)
test_data = clean_data(test_data)

# Paso 2: Dividir los datasets en x_train, y_train, x_test, y_test
X_train = train_data.drop(columns=["default"])
y_train = train_data["default"]
X_test = test_data.drop(columns=["default"])
y_test = test_data["default"]

# Paso 3: Crear el pipeline
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features = [col for col in X_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ('selectkbest', SelectKBest(score_func=f_classif)),
        ('pca', PCA()),
        ('mlp', MLPClassifier(max_iter=1000, random_state=420))
    ]
)

# Paso 4: Optimizar hiperparámetros
param_grid = {
    'pca__n_components': [20],
    'selectkbest__k': [20],
    'mlp__hidden_layer_sizes': [(50,30,40,60)],
    'mlp__alpha': [0.256],
    'mlp__learning_rate': ['adaptive'],
    'mlp__activation': ['relu'],
    'mlp__solver': ['adam'],
    'mlp__learning_rate_init': [0.001]
}


model = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True
)

model.fit(X_train, y_train)

# Paso 5: Guardar el modelo entrenado
models_dir = 'files/models'
os.makedirs(models_dir, exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as file:
    pickle.dump(model, file)

# Paso 6: Calcular métricas
def calculate_metrics(model, x, y, dataset_type):
    y_pred = model.predict(x)
    metrics = {
        "type": "metrics",
        'dataset': dataset_type,
        'precision': precision_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred)
    }
    return metrics

train_metrics = calculate_metrics(model, X_train, y_train, 'train')
test_metrics = calculate_metrics(model, X_test, y_test, 'test')

# Paso 7: Calcular matrices de confusión
def calculate_confusion_matrix(model, x, y, dataset_type):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        'type': 'cm_matrix',
        'dataset': dataset_type,
        'true_0': {'predicted_0': int(cm[0, 0]), 'predicted_1': int(cm[0, 1])},
        'true_1': {'predicted_0': int(cm[1, 0]), 'predicted_1': int(cm[1, 1])}
    }
    return cm_dict

train_cm = calculate_confusion_matrix(model, X_train, y_train, 'train')
test_cm = calculate_confusion_matrix(model, X_test, y_test, 'test')

# Paso 8: Guardar métricas y matrices de confusión
output_file = "files/output/metrics.json"
os.makedirs("files/output", exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(train_metrics, f, ensure_ascii=False)
    f.write('\n')
    json.dump(test_metrics, f, ensure_ascii=False)
    f.write('\n')

with open(output_file, 'a', encoding='utf-8') as f:
    json.dump(train_cm, f, ensure_ascii=False)
    f.write('\n')
    json.dump(test_cm, f, ensure_ascii=False)
    f.write('\n')