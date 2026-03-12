import pandas as pd

# Load dataset
df = pd.read_csv("dataset/healthcare-dataset-stroke-data.csv")

print("Original shape:", df.shape)

# ---------------- Drop unnecessary columns ----------------
df.drop(columns=[
    "id",
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "heart_disease"
], inplace=True)

# ---------------- Handle missing BMI ----------------
df["bmi"].fillna(df["bmi"].median(), inplace=True)

# ---------------- Encode smoking_status ----------------
df["smoking_status"] = df["smoking_status"].map({
    "never smoked": 0,
    "formerly smoked": 1,
    "smokes": 2,
    "Unknown": 0
})

# ---------------- Ensure correct types ----------------
df["hypertension"] = df["hypertension"].astype(int)
df["stroke"] = df["stroke"].astype(int)

# ---------------- Save cleaned dataset ----------------
df.to_csv("dataset/stroke_cleaned.csv", index=False)

print("✅ Cleaned dataset saved as dataset/stroke_cleaned.csv")
print("Final shape:", df.shape)
print(df.head())