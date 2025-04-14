import pandas as pd
import numpy as np
import seaborn as sns

# KIDNEY DATA
kidney_data = pd.read_csv(r"D:\Capestone3\venv\kidney_disease - kidney_disease.csv")
print(kidney_data)
print(kidney_data.isnull().sum())

kidney_dataset = kidney_data.dropna()
print(kidney_dataset.isnull().sum())
print(type(kidney_dataset))

from sklearn.preprocessing import LabelEncoder as le

print(kidney_dataset)
print(kidney_dataset.drop(columns=['id'], inplace = True))
print(kidney_dataset.head())
print(kidney_dataset.info())

for s in ['pcv', 'wc', 'rc']:
    kidney_dataset[s] = pd.to_numeric(kidney_dataset[s])
    
print(kidney_dataset.info())

num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                  'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 
                    'appet', 'pe', 'ane', 'classification']

le = le()

for i in cat_cols:
    kidney_dataset[i] = le.fit_transform(kidney_dataset[i])
    
print(kidney_dataset.head())

from sklearn.model_selection import train_test_split as tnts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

plt.figure(figsize = (20, 15))
sns.heatmap(kidney_dataset.corr(), annot = True, fmt=".1f", linewidths=0.3)

# Features and Target
X = kidney_dataset.drop(columns = ['classification', 'cad', 'rbc', 'pc', 'pcc', 'ba'])  #features
Y = kidney_dataset['classification']   #target

# Train-test split
X_train, X_test, Y_train, Y_test = tnts(X, Y, test_size = 0.2, random_state = 42)

# Logistic Regression 
lr = LogisticRegression()
lr.fit(X_train, Y_train)

# Predict on test set
Y_pred = lr.predict(X_test)

# Accuracy Calculation
accuracy = accuracy_score(Y_test, Y_pred) * 100
print(f"Accuracy Score: {accuracy:.1f}%")
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Mean Absolute Error (MAE)
M = mean_absolute_error(Y_test, Y_pred)
print(f"MAE: {M:.2f}")

# Root Mean Squared Error (RMSE)
R = np.sqrt(root_mean_squared_error(Y_test, Y_pred))
print(f"RMSE: {R:.2f}")

import pickle

pickle.dump(lr, open("kidney_model.pkl", 'wb'))
# kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
print("Model saved successfully!")

