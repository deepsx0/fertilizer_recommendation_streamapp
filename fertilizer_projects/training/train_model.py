import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# === STEP 1: Load Dataset ===
data_path = "training\data_core.csv"  # Ensure this file is in the same directory
df = pd.read_csv(data_path)

# === STEP 2: Encode Categorical Columns ===
label_encoders = {}
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === STEP 3: Prepare Features & Labels ===
X = df.drop(columns=['Fertilizer Name'])
y = df['Fertilizer Name']

# === STEP 4: Scale Numerical Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === STEP 5: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === STEP 6: Train XGBoost Model ===
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# === STEP 7: Evaluate ===
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# === STEP 8: Save Model and Preprocessing Tools ===
os.makedirs("models", exist_ok=True)

# Save model
model.save_model("models/fertilizer_model.json")

# Save encoders
with open("models/encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model, encoders, and scaler saved in 'models/' directory.")
