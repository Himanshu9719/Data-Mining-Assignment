import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# -------------------------------
# Function to build and save tree
# -------------------------------
def build_tree(X, y, features, dataset_name, criterion="gini", max_depth=None):
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    clf.fit(X, y)

    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=features, class_names=[str(c) for c in set(y)],
              filled=True, rounded=True, fontsize=9)
    plt.title(f"{dataset_name} Decision Tree ({criterion})")
    filename = f"{dataset_name.lower()}_{criterion}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {dataset_name} tree ({criterion}) as {filename}")

# -------------------------------
# 1. IRIS Dataset
# -------------------------------
iris = load_iris(as_frame=True)
X_iris = iris.data
y_iris = iris.target
features_iris = iris.feature_names

build_tree(X_iris, y_iris, features_iris, "Iris", "gini")
build_tree(X_iris, y_iris, features_iris, "Iris", "entropy")

# -------------------------------
# 2. Borrowers Dataset
# -------------------------------
data_borrowers = {
    "Age": [25, 40, 50, 23, 30, 35, 28, 60],
    "AnnualIncome": [50000, 80000, 60000, 40000, 75000, 120000, 30000, 90000],
    "HomeOwner": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No"],
    "MaritalStatus": ["Single", "Married", "Married", "Single", "Single", "Married", "Single", "Married"],
    "DefaultedBorrower": ["No", "No", "Yes", "Yes", "No", "No", "Yes", "Yes"]
}
df_borrowers = pd.DataFrame(data_borrowers)

# Encode categorical variables
le = LabelEncoder()
df_borrowers["HomeOwner"] = le.fit_transform(df_borrowers["HomeOwner"])
df_borrowers["MaritalStatus"] = le.fit_transform(df_borrowers["MaritalStatus"])
y_borrowers = df_borrowers["DefaultedBorrower"]
X_borrowers = df_borrowers.drop("DefaultedBorrower", axis=1)
features_borrowers = X_borrowers.columns

build_tree(X_borrowers, y_borrowers, features_borrowers, "Borrowers", "gini")
build_tree(X_borrowers, y_borrowers, features_borrowers, "Borrowers", "entropy")

# -------------------------------
# 3. Mall Customers Dataset
# -------------------------------
data_mall = {
    "Age": [19, 35, 26, 27, 19, 27, 40, 23],
    "AnnualIncome": [15, 35, 75, 50, 20, 80, 60, 45],
    "SpendingScore": [39, 81, 6, 77, 40, 40, 20, 60],
    "CustomerType": ["Low", "High", "Low", "High", "Low", "High", "Low", "High"]
}
df_mall = pd.DataFrame(data_mall)

y_mall = df_mall["CustomerType"]
X_mall = df_mall.drop("CustomerType", axis=1)
features_mall = X_mall.columns

# Limit depth for readability
build_tree(X_mall, y_mall, features_mall, "MallCustomers", "gini", max_depth=3)
build_tree(X_mall, y_mall, features_mall, "MallCustomers", "entropy", max_depth=3)
