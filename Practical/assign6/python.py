# -----------------------------
# Decision Tree Classification
# -----------------------------

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_iris

# --- Dataset 1: Email ---
data1 = pd.read_csv("email.csv")

# Remove missing rows (fixes NaN issue)
data1 = data1.dropna(subset=['Category', 'Message'])

# Encode target
data1['Category'] = data1['Category'].map({'spam': 1, 'ham': 0})

# Drop rows that didn’t match spam/ham (if any)
data1 = data1.dropna(subset=['Category'])

# Convert text to numeric features
tf = TfidfVectorizer(stop_words='english', max_features=1000)
X1 = tf.fit_transform(data1['Message'])
y1 = data1['Category']

# --- Dataset 2: Iris ---
iris = load_iris()
X2 = iris.data
y2 = iris.target

# --- Model ---
model = DecisionTreeClassifier(criterion='entropy', random_state=0)

# ---------- Holdout ----------
print("=== HOLDOUT METHOD ===")

# 80-20 split
for name, X, y, ts in [('Email', X1, y1, 0.2), ('Iris', X2, y2, 0.2)]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=0)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    print(f"{name} (80-20):",
          round(accuracy_score(yte, yp), 3),
          round(precision_score(yte, yp, average='weighted'), 3),
          round(recall_score(yte, yp, average='weighted'), 3),
          round(f1_score(yte, yp, average='weighted'), 3))

# 66.6-33.3 split
for name, X, y, ts in [('Email', X1, y1, 0.333), ('Iris', X2, y2, 0.333)]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=0)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    print(f"{name} (66.6-33.3):",
          round(accuracy_score(yte, yp), 3),
          round(precision_score(yte, yp, average='weighted'), 3),
          round(recall_score(yte, yp, average='weighted'), 3),
          round(f1_score(yte, yp, average='weighted'), 3))

# ---------- Cross Validation ----------
print("\n=== CROSS VALIDATION ===")

for name, X, y in [('Email', X1, y1), ('Iris', X2, y2)]:
    for k in [10, 5]:
        acc = cross_val_score(model, X, y, cv=k, scoring='accuracy').mean()
        print(f"{name} ({k}-Fold):", round(acc, 3))
