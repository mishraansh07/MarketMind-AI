import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your marketing CSV
df = pd.read_csv("train.csv")
df = pd.read_csv("text.csv")
df = pd.read_csv("bigdata.csv")

# Feature engineering
df["CTR (%)"] = (df["Clicks"] / df["Impressions"]) * 100
df["CPC (₹)"] = df["Spend (₹)"] / df["Clicks"]
df["CPA (₹)"] = df["Spend (₹)"] / df["Conversions"]

features = ["Impressions", "Clicks", "Spend (₹)", "CTR (%)", "CPC (₹)", "CPA (₹)"]
target = "Conversions"

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
