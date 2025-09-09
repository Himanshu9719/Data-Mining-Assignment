import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------
# Step 1: Load CSV
# -----------------------
df = pd.read_csv("train.csv")  # Replace with your file

# -----------------------
# Step 2: Select numeric columns
# -----------------------
X = df.select_dtypes(include=['int64', 'float64'])
X = X.fillna(X.mean())  # Fill missing values
X_scaled = StandardScaler().fit_transform(X)

# -----------------------
# Step 3: KMeans clustering and MSE tracking
# -----------------------
k = 3
max_iter = 10
mse_list = []

kmeans = KMeans(n_clusters=k, n_init=1, max_iter=1, random_state=42)

for i in range(max_iter):
    kmeans.max_iter = i + 1
    kmeans.fit(X_scaled)
    mse_list.append(kmeans.inertia_)

# -----------------------
# Step 4: Plot MSE vs iteration
# -----------------------
plt.plot(range(1, max_iter+1), mse_list, marker='o')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title(f"K-Means Convergence (k={k})")
plt.show()

# -----------------------
# Step 5: Compare MSE for different k
# -----------------------
for k_val in [2,3,4]:
    model = KMeans(n_clusters=k_val, random_state=42).fit(X_scaled)
    print(f"k={k_val}, Final MSE={model.inertia_:.2f}")
