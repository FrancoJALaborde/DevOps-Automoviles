import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Lectura del archivo CSV
data = pd.read_csv("M6\Proyecto Integrador\Propuesta 2\ML_cars.csv")

# Preprocesamiento de los datos
data = data.drop(["car_ID", "symboling", "CarName"], axis=1)  # Eliminar columnas innecesarias
data = pd.get_dummies(data, columns=["fueltype", "aspiration", "doornumber", "carbody", "drivewheel",
                                     "enginelocation", "enginetype", "cylindernumber", "fuelsystem"], drop_first=True)  # Codificar variables categóricas
median_price = data["price"].median()  # Obtener la mediana de los precios

# Modelo de clasificación
data["price_category"] = pd.Series(data["price"] > median_price, dtype=int)  # Crear la variable de categoría
X = data.drop(["price", "price_category"], axis=1)
y = data["price_category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Separar datos de entrenamiento y prueba
clf = DecisionTreeClassifier(max_depth=3)  # Crear modelo de clasificación
clf.fit(X_train, y_train)  # Entrenar modelo
y_pred = clf.predict(X_test)  # Realizar predicciones
print(confusion_matrix(y_test, y_pred))  # Mostrar matriz de confusión
print(classification_report(y_test, y_pred))  # Mostrar reporte de clasificación

# Modelo de regresión
data.drop(["price_category"], axis=1, inplace=True)  # Eliminar la variable de categoría creada previamente
X = data.drop(["price"], axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Separar datos de entrenamiento y prueba
reg = LinearRegression()  # Crear modelo de regresión
reg.fit(X_train, y_train)  # Entrenar modelo
y_pred = reg.predict(X_test)  # Realizar predicciones
print("Precisión del modelo de regresión: ", reg.score(X_test, y_test))  # Mostrar precisión del modelo de regresión


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=data, x="price", bins=20, kde=True)
plt.axvline(x=median_price, color="r", linestyle="--", label="Mediana")
plt.legend()
plt.show()