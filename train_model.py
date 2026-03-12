import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset/stroke_cleaned.csv")

FEATURES = [
    "age",
    "hypertension",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]

X = df[FEATURES]
y = df["stroke"]

print("Original class distribution:")
print(y.value_counts())

# ---------------- APPLY SMOTE ----------------
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE class distribution:")
print(pd.Series(y_resampled).value_counts())

# ---------------- TRAIN RANDOM FOREST ----------------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=5,
    class_weight={0: 1, 1: 4},
    random_state=42,
    n_jobs=-1
)

model.fit(X_resampled, y_resampled)

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "stroke_model.pkl")

print("\n✅ Random Forest Stroke Model trained & saved")