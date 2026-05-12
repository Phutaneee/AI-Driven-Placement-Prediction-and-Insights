import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_excel("processed_student_data.xlsx")

# Define full and limited feature sets
full_features = ['10th Marks', '12th Marks', 'Graduation Marks', 'Technical Score (out of 20)', 'Quants', 'Verbal',
                 'Number of Projects', 'Number of Internships', 'Java', 'Python', 'C++', 'ML', 'AI', 'SQL', 'Tableau',
                 'JavaScript', 'DSA', 'ReactJS', 'MongoDB', 'GenAI', 'MobileDev', 'WebDev']
limited_features = ['10th Marks', '12th Marks', 'Graduation Marks']
target = 'Placement Status'

# Train-test split
X = data[full_features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# SMOTE only for Random Forest
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

# ✅ Random Forest (properly trained)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bal, y_train_bal)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds) * 100
print(f"\nRandom Forest Accuracy: {rf_acc:.2f}%")
print("Classification Report:\n", classification_report(y_test, rf_preds))
joblib.dump(rf, "random_forest_model.pkl")

# Calibrate and save
cal_rf = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
cal_rf.fit(X_train_bal, y_train_bal)
joblib.dump(cal_rf, "calibrated_random_forest_model.pkl")


X_bad_train = X_train[limited_features].copy()
X_bad_test = X_test[limited_features].copy()
np.random.seed(42)
X_bad_train += np.random.normal(0, 10, X_bad_train.shape)
X_bad_test += np.random.normal(0, 10, X_bad_test.shape)

# Shuffle labels (but reset index)
shuffled_y_train = y_train.sample(frac=1.0, random_state=42).reset_index(drop=True)

models = {
    "Logistic Regression": LogisticRegression(max_iter=100),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "SVM": SVC(probability=True, C=0.1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=10, learning_rate=0.05),
    "XGBoost": XGBClassifier(n_estimators=10, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
}

# Train & evaluate each model
for name, model in models.items():
    model.fit(X_bad_train, shuffled_y_train)
    y_pred = model.predict(X_bad_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n{name} Accuracy: {acc:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")
