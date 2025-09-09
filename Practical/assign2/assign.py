import pandas as pd
import numpy as np

# Load dataset with correct separator
data = pd.read_csv("winequality-red.csv", sep=";")

# Remove extra spaces in column names
data.columns = data.columns.str.strip()

# 1. Standardization (Z-score) → fixed acidity
data["fixed acidity (std)"] = (data["fixed acidity"] - data["fixed acidity"].mean()) / data["fixed acidity"].std()

# 2. Normalization (Min-Max) → volatile acidity
data["volatile acidity (norm)"] = (data["volatile acidity"] - data["volatile acidity"].min()) / (data["volatile acidity"].max() - data["volatile acidity"].min())

# 3. Log Transformation → residual sugar
data["residual sugar (log)"] = np.log1p(data["residual sugar"])   # log(1+x) to avoid log(0)

# 4. Aggregation → combine free + total sulfur dioxide
data["SO2 aggregate"] = data["free sulfur dioxide"] + data["total sulfur dioxide"]

# 5. Discretization (binning) → pH into 3 categories
data["pH (binned)"] = pd.cut(data["pH"], bins=3, labels=["Low","Medium","High"])

# 6. Binarization → alcohol (threshold = 10)
data["alcohol (binary)"] = np.where(data["alcohol"] > 10, 1, 0)

# 7. Sampling → take random 100 rows
sampled_data = data.sample(n=100, random_state=42)

# Save processed dataset
data.to_csv("winequality_processed.csv", index=False)
sampled_data.to_csv("winequality_sampled.csv", index=False)

print("✅ Processed dataset saved as 'winequality_processed.csv'")
print("✅ Sampled dataset saved as 'winequality_sampled.csv'")

# Show first 10 rows of processed data
print("\nPreview of processed dataset:")
print(data.head(10)[[
    "fixed acidity","fixed acidity (std)",
    "volatile acidity","volatile acidity (norm)",
    "residual sugar","residual sugar (log)",
    "SO2 aggregate","pH","pH (binned)",
    "alcohol","alcohol (binary)","quality"
]])
