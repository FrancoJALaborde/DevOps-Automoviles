
<h1 align="center">Machine Learning Project for Classification and Regression of Car Prices</h1>
<h2>Project Description</h2>
This project aims to apply machine learning techniques to classify and predict car prices. It utilizes a dataset of cars that includes various vehicle features such as engine size, horsepower, body type, among others.

<h2>Code and Functionality</h2>
The provided code performs the following tasks:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Load data
data = pd.read_csv('M6\Proyecto Integrador\Propuesta 2\ML_cars.csv')

# Data cleaning
data = data.drop(['car_ID', 'symboling', 'CarName'], axis=1) # Remove unnecessary columns
data = pd.get_dummies(data, drop_first=True) # Encode categorical variables

# Create target variable for classification (high or low)
data['price_class'] = data['price'].apply(lambda x: 'high' if x > data['price'].median() else 'low')

# Split data into training and test sets
X = data.drop(['price', 'price_class'], axis=1)
y_class = data['price_class']
y_reg = data['price']
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(X, y_class, y_reg, test_size=0.2, random_state=0)

# Train classification model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_class_train)
y_class_pred = clf.predict(X_test)

# Evaluate classification model
accuracy = accuracy_score(y_class_test, y_class_pred)
print(f'Accuracy: {accuracy}')

# Train regression model
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train, y_reg_train)
y_reg_pred = reg.predict(X_test)

# Evaluate regression model
mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f'MSE: {mse}')

# Remaining code for result visualization and plots
The code handles data loading and preprocessing, creates a target variable for classification, splits the data into training and test sets, trains classification and regression models using decision trees, evaluates the models, and visualizes the results through plots and metrics.

<h2>Requirements</h2>
To execute this code, the following Python libraries need to be installed:
pandas
matplotlib
seaborn
scikit-learn
<h2>Usage Instructions</h2>
1. Download the CSV file with the car data and ensure it is in the same location as the code file.
2. Run the code in a Python environment.
3. Observe the results of the classification and regression models, as well as the generated plots.
