from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split as tnts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import xgboost as xgb 

# Load dataset
liver_patient_data = pd.read_csv(r"D:\Capestone3\venv\indian_liver_patient - indian_liver_patient.csv")

# Drop missing values
liver_data = liver_patient_data.dropna()

# Encode categorical column
Le = LabelEncoder()
liver_data['Gender'] = Le.fit_transform(liver_data['Gender'])
liver_data['Dataset'] = liver_data['Dataset'].map({1: 1, 2: 0})  # 1 - Liver Disease, 2 - No Disease

# Features and Target
X = liver_data.drop(columns=['Dataset'])  # Features (Keeping Gender)
Y = liver_data['Dataset']  # Target

# Apply MinMax Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Check class distribution
print("Class distribution before SMOTE:")
print(Y.value_counts())

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

# Check class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(Y_resampled).value_counts())

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = tnts(X_resampled, Y_resampled, test_size=0.2, random_state=42)

# Initialize XGBoost with tuned hyperparameters
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, 
    colsample_bytree=0.8, objective='binary:logistic', random_state=42
)

# Train the model
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Evaluate Metrics
accuracy = accuracy_score(Y_test, Y_pred) * 100
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print(f"\nAccuracy Score after SMOTE: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix after SMOTE")
plt.show()

import pickle

pickle.dump(model, open("Liver_Disease_Model.pkl", 'wb'))
# liver_model = pickle.load(open('Liver_Disease_Model.pkl', 'rb'))
print("Model saved successfully!")