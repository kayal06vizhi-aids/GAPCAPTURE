import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# --- Step 1: Load dataset ---
df = pd.read_csv("training_dataset.csv")

# --- Step 2: Prepare features and labels ---
X = df.drop(columns=["Job Role"])
y = df["Job Role"]

# --- Step 3: Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train Logistic Regression model ---
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# --- Step 5: Evaluate model ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# --- Step 6: Save model ---
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save columns (skills) for later use in app
with open("skills_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Model and skill columns saved for Streamlit app")
