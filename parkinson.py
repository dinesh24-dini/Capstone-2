import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

parkinsons_data = pd.read_csv(r"D:\Capestone3\venv\parkinsons - parkinsons.csv")
print(parkinsons_data.head())
print(parkinsons_data.info())
print(parkinsons_data.isnull().sum()) 

print(parkinsons_data['status'].head())

# Features and Target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)  # Features (all except target)
y = parkinsons_data['status']  # Target (1 = Parkinson’s, 0 = Healthy)

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Handling Class Imbalance using SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

from sklearn.svm import SVC

# Initialize SVM model with RBF kernel
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)

# Perform cross-validation
svm_cv_scores = cross_val_score(svm_model, X_resampled, y_resampled, cv=5, scoring='accuracy')

# Train on full dataset
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluate SVM
svm_accuracy = accuracy_score(y_test, y_pred) * 100
print(f"SVM Cross-Validation Accuracy: {svm_cv_scores.mean() * 100:.2f}%")
print(f"SVM Test Accuracy: {svm_accuracy:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Parkinson's Prediction")
plt.show()

# Save the trained model
pickle.dump(svm_model, open("parkinsons_model.pkl", 'wb'))
# parks_model = pickle.load(open('parkinsons_model.pkl', 'rb'))
print("Model saved successfully!")
