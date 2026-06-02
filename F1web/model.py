import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("正在讀取資料...")
df = pd.read_csv("test_engineered.csv")
df = df.fillna(0)

# ⚠️ 這裡的特徵名稱和順序，必須跟 app.py 裡面定義的完全一模一樣
features = ['TyreLife', 'Stint', 'LapNumber', 'Position', 'Laps_Since_LastPit']
X = df[features]
y = df['PitStop']

print("正在訓練模型...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl", compress=3)