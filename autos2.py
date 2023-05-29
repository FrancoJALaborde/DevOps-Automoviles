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

# Guardar predicciones en archivos de texto plano
with open('predicciones_clasificacion.txt', 'w') as file:
    for pred in y_class_pred:
        file.write(f'{pred}\n')

with open('predicciones_regresion.txt', 'w') as file:
    for pred in y_reg_pred:
        file.write(f'{pred}\n')


# Visualizar resultados del modelo de clasificación
sns.heatmap(confusion_matrix(y_class_test, y_class_pred), annot=True, cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Visualizar resultados del modelo de regresión
plt.scatter(X_test.iloc[:, 2], y_reg_test, color='black')
plt.plot(X_test.iloc[:, 2], y_reg_pred, color='blue', linewidth=3)
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.show()

# Graficar la distribución de precios de los vehículos
sns.histplot(data=data, x='price', bins=20, kde=True)
plt.title('Distribución de precios de los vehículos')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.show()

# Graficar la relación entre el precio y la potencia del motor
sns.scatterplot(data=data, x='horsepower', y='price')
plt.title('Relación entre el precio y la potencia del motor')
plt.xlabel('Potencia del motor')
plt.ylabel('Precio')
plt.show()

# Graficar la relación entre el precio y el tipo de carrocería
sns.boxplot(data=data, x='carbody_hardtop', y='price')
plt.title('Relación entre el precio y el tipo de carrocería')
plt.xlabel('Tipo de carrocería')
plt.ylabel('Precio')
plt.show()

# Graficar la matriz de correlación entre las variables
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de correlación entre las variables')
plt.show()

# Graficar el árbol de decisión para la clasificación de precios
from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['baja', 'alta'])
plt.title('Árbol de decisión para la clasificación de precios')
plt.show()

# Graficar la comparación entre los precios reales y los precios predichos
plt.scatter(y_reg_test, y_reg_pred)
plt.title('Comparación entre los precios reales y los precios predichos')
plt.xlabel('Precios reales')
plt.ylabel('Precios predichos')
plt.show()


