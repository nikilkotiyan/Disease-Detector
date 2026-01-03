import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle

# Load datasets
diabetes_data = pd.read_csv('diabetes.csv')
heart_data = pd.read_csv('heart.csv')
parkinsons_data = pd.read_csv('parkinsons.csv')

# Diabetes Model with XGBoost
X_diabetes = diabetes_data.drop(columns='Outcome', axis=1)
Y_diabetes = diabetes_data['Outcome']

# Feature Engineering for Diabetes
X_diabetes['BMI_Age'] = X_diabetes['BMI'] * X_diabetes['Age']
X_diabetes['Glucose_Insulin'] = X_diabetes['Glucose'] * X_diabetes['Insulin']

X_train_d, X_test_d, Y_train_d, Y_test_d = train_test_split(X_diabetes, Y_diabetes, test_size=0.2, stratify=Y_diabetes, random_state=2)

diabetes_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)
diabetes_model.fit(X_train_d, Y_train_d)

# Evaluate
Y_pred_d = diabetes_model.predict(X_test_d)
print(f"Diabetes Model Accuracy: {accuracy_score(Y_test_d, Y_pred_d)}")

pickle.dump(diabetes_model, open('diabetes_model_advanced.sav', 'wb'))

# Heart Disease Model with XGBoost
X_heart = heart_data.drop(columns='target', axis=1)
Y_heart = heart_data['target']

# Feature Engineering for Heart
X_heart['Age_Chol'] = X_heart['age'] * X_heart['chol']
X_heart['Thalach_Oldpeak'] = X_heart['thalach'] * X_heart['oldpeak']

X_train_h, X_test_h, Y_train_h, Y_test_h = train_test_split(X_heart, Y_heart, test_size=0.2, stratify=Y_heart, random_state=2)

heart_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)
heart_model.fit(X_train_h, Y_train_h)

# Evaluate
Y_pred_h = heart_model.predict(X_test_h)
print(f"Heart Disease Model Accuracy: {accuracy_score(Y_test_h, Y_pred_h)}")

pickle.dump(heart_model, open('heart_disease_model_advanced.sav', 'wb'))

# Parkinson's Model with XGBoost
X_parkinsons = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y_parkinsons = parkinsons_data['status']

# Feature Engineering for Parkinson's
X_parkinsons['jitter_ratio'] = X_parkinsons['MDVP:Jitter(%)'] / (X_parkinsons['MDVP:Jitter(Abs)'] + 1e-6)
X_parkinsons['vocal_instability'] = X_parkinsons['Jitter:DDP'] + X_parkinsons['Shimmer:DDA']
X_parkinsons['log_PPE'] = np.log(X_parkinsons['PPE'] + 1e-6)

X_train_p, X_test_p, Y_train_p, Y_test_p = train_test_split(X_parkinsons, Y_parkinsons, test_size=0.2, random_state=2)

parkinsons_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)
parkinsons_model.fit(X_train_p, Y_train_p)

# Evaluate
Y_pred_p = parkinsons_model.predict(X_test_p)
print(f"Parkinson's Model Accuracy: {accuracy_score(Y_test_p, Y_pred_p)}")

pickle.dump(parkinsons_model, open('parkinsons_model_advanced.sav', 'wb'))

print("Advanced models trained and saved.")
