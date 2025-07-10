import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load your CSV file
df = pd.read_csv("heart.csv")

# Features & target
X = df.drop("target", axis=1)
y = df["target"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
