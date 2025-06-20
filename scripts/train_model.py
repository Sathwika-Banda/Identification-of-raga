import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Load features.csv
csv_path = 'data/processed/features.csv'
df = pd.read_csv(csv_path)

print("Columns in dataset:", df.columns.tolist())

# Strip spaces and lowercase column names for safety
df.columns = df.columns.str.strip().str.lower()

# Ensure 'label' exists
if 'label' not in df.columns:
    raise ValueError(f"'label' column not found in features.csv. Found columns: {df.columns.tolist()}")

# Split features and target
X = df.drop('label', axis=1)
y = df['label']

print("Number of samples:", len(X))
print("Number of features used for training:", X.shape[1])

# Split dataset (80% train, 20% test)
if len(X) < 5:
    raise ValueError("🚫 Not enough data to split. Add more samples to features.csv.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
with open('models/raga_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model trained and saved as 'models/raga_classifier.pkl'")
