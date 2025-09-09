import pandas as pd

# Load dataset
df = pd.read_csv('data.csv')

# Step 1: Handle missing values (fill numerical columns with median)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# Step 2: Bin Age and Salary into categorical ranges
df['Age_group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100],
                         labels=['Age_0-30','Age_31-40','Age_41-50','Age_51-60','Age_61+'])
df['Salary_group'] = pd.cut(df['Salary'], bins=[0, 50000, 60000, 70000, 80000, 100000],
                            labels=['Salary_0-50k','Salary_51-60k','Salary_61-70k','Salary_71-80k','Salary_81k+'])

# Step 3: Convert Purchased column to categorical item
df['Purchased_item'] = df['Purchased'].apply(lambda x: 'Purchased_Yes' if x=='Yes' else 'Purchased_No')

# Step 4: Create transactions (list of items for each row)
transactions = df.apply(lambda row: [f"Country_{row['Country']}", row['Age_group'], row['Salary_group'], row['Purchased_item']], axis=1)

# Step 5: Save transactions to CSV (each transaction as comma-separated string)
transactions_df = pd.DataFrame(transactions, columns=['Transaction'])
transactions_df['Transaction'] = transactions_df['Transaction'].apply(lambda x: ','.join(x))
transactions_df.to_csv('transactional_data.csv', index=False)

print("Transactional data saved as transactional_data.csv")
