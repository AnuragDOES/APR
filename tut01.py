# %% Import libraries
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %% Download dataset
path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")
print("Path to dataset files:", path)

# %% Load CSV file (dataset has 'heart_cleveland_upload.csv')
csv_path = path + "/heart_cleveland_upload.csv"
data = pd.read_csv(csv_path)

# %% Features and target
X = data.drop('condition', axis=1)   # 'condition' is target in this dataset
y = data['condition']

# %% Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% Logistic Regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# %% Predictions
y_pred = model.predict(X_test)

# %% Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


