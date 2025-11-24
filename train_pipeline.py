# train_pipeline.py
import pandas as pd, joblib, datetime, json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

CSV_PATH = "shimoga_college_aqi_6months.csv" # your feature CSV
FEATURES = ["co","no","no2","o3","so2","pm2_5","pm10","nh3"]
TARGET = "aqi"
OUT_MODEL = "models/aqi_pipeline_v1.joblib"
META = "models/metadata_v1.json"

df = pd.read_csv(CSV_PATH).dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]; y = df[TARGET]

# time-wise split (no shuffle) — important for time-series
train_frac = 0.8
split_idx = int(len(df)*train_frac)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

pre = ColumnTransformer([("num", StandardScaler(), FEATURES)], remainder="drop")
pipeline = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])

pipeline.fit(X_train, y_train)

# quick eval
from sklearn.metrics import mean_absolute_error, r2_score
pred = pipeline.predict(X_test)
print("Train rows:", len(X_train), "Test rows:", len(X_test))
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# save model + metadata
import os
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, OUT_MODEL)
meta = {"version":"v1", "created": datetime.datetime.utcnow().isoformat()+"Z", "features": FEATURES}
with open(META,"w") as f: json.dump(meta,f,indent=2)
print("Saved pipeline:", OUT_MODEL)
