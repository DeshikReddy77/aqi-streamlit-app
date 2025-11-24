import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. LOAD DATA
data = pd.read_csv(r"C:\Users\Asus\Downloads\AI POWERED AIR QUALITY PREDICTION SYSTEM\shimoga_college_aqi_6months.csv")

print("File loaded successfully!")
print(data.head())

# 2. SELECT FEATURES
features = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
target = "aqi"

X = data[features]
y = data[target]

# 3. SCALE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 6. TEST MODEL
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# 7. SAVE MODEL + SCALER
joblib.dump(model, "aqi_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved as aqi_model.pkl")
print("Scaler saved as scaler.pkl")
