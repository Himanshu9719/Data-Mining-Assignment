import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("ChronicKidneyDisease.csv")

# Step 1: Handle Missing Values
# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].str.lower().str.strip()   # normalize text
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numerical missing values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Step 2: Handle Outliers with IQR
def cap_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)

for col in numeric_cols:
    df[col] = cap_outliers_iqr(df[col])

# Step 3: Standardize Inconsistent Values
# Example: RBC, PC, PCC, BA must be limited categories
valid_map = {
    "rbc": {"normal": "normal", "abnormal": "abnormal"},
    "pc": {"normal": "normal", "abnormal": "abnormal"},
    "pcc": {"present": "present", "notpresent": "notpresent"},
    "ba": {"present": "present", "notpresent": "notpresent"}
}

for col, mapping in valid_map.items():
    df[col] = df[col].map(mapping).fillna(df[col].mode()[0])

# Step 4: Apply Validation Rules
validation_rules = {
    "age": (0, 100),
    "bp": (50, 180),
    "sg": (1.005, 1.025),
    "al": (0, 5),
    "su": (0, 5)
}

for col, (low, high) in validation_rules.items():
    if col in df.columns:
        df[col] = df[col].clip(lower=low, upper=high)

# Step 5: Save Cleaned Dataset
df.to_csv("ChronicKidneyDisease_Cleaned.csv", index=False)

print("✅ Data Cleaning Completed! Cleaned file saved as ChronicKidneyDisease_Cleaned.csv")
