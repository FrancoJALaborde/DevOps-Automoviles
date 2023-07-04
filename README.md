<h1 align="center">Proyecto de Aprendizaje Automático para Clasificación y Regresión de Precios de Automóviles</h1>

<h2>Descripción del Proyecto</h2>

Este proyecto tiene como objetivo aplicar técnicas de aprendizaje automático para clasificar y predecir los precios de los automóviles. Se utiliza un conjunto de datos de automóviles que incluye diversas características de los vehículos, como el tamaño del motor, la potencia, el tipo de carrocería, entre otros.

<h2>Código y Funcionalidades</h2>

El código proporcionado realiza las siguientes tareas:

python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Cargar datos
data = pd.read_csv('M6\Proyecto Integrador\Propuesta 2\ML_cars.csv')

# Limpieza de datos
data = data.drop(['car_ID', 'symboling', 'CarName'], axis=1) # Eliminar columnas innecesarias
data = pd.get_dummies(data, drop_first=True) # Codificar variables categóricas

# Crear variable objetivo para la clasificación (alta o baja)
data['price_class'] = data['price'].apply(lambda x: 'alta' if x > data['price'].median() else 'baja')

# Separar datos en conjuntos de entrenamiento y prueba
X = data.drop(['price', 'price_class'], axis=1)
y_class = data['price_class']
y_reg = data['price']
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(X, y_class, y_reg, test_size=0.2, random_state=0)

# Entrenar modelo de clasificación
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_class_train)
y_class_pred = clf.predict(X_test)

# Evaluar modelo de clasificación
accuracy = accuracy_score(y_class_test, y_class_pred)
print(f'Accuracy: {accuracy}')

# Entrenar modelo de regresión
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train, y_reg_train)
y_reg_pred = reg.predict(X_test)

# Evaluar modelo de regresión
mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f'MSE: {mse}')

# Resto del código de visualización de resultados y gráficos
El código se encarga de cargar y preprocesar los datos, crear una variable objetivo para la clasificación, separar los datos en conjuntos de entrenamiento y prueba, entrenar modelos de clasificación y regresión utilizando árboles de decisión, evaluar los modelos y visualizar los resultados mediante gráficos y métricas.

<h2>Requisitos</h2>
Para ejecutar este código, se requiere tener instaladas las siguientes bibliotecas de Python:

pandas
matplotlib
seaborn
scikit-learn

<h2>Instrucciones de Uso</h2>
Descarga el archivo CSV con los datos de los automóviles y asegúrate de que esté en la misma ubicación que el archivo de código.
Ejecuta el código en un entorno de Python.
Observa los resultados de los modelos de clasificación y regresión, así como los gráficos generados.
