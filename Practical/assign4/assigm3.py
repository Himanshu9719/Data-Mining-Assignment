import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori(transactions, support, confidence, output_file):
    te = TransactionEncoder()
    trans_df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    frequent_itemsets = apriori(trans_df, min_support=support, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        rules.to_csv(output_file, index=False)
        print(f"Rules saved to {output_file}")
    else:
        print(f"No frequent itemsets found for {output_file}")

# Dataset 1: transactional_data.csv
df1 = pd.read_csv('transactional_data.csv')
transactions1 = df1['Transaction'].apply(lambda x: x.split(',')).tolist()
run_apriori(transactions1, support=0.2, confidence=0.6, output_file='rules_data_set1a.csv')
run_apriori(transactions1, support=0.3, confidence=0.5, output_file='rules_data_set1b.csv')

# Dataset 2: Groceries_dataset.csv
df2 = pd.read_csv('Groceries_dataset.csv')
transactions2 = df2.groupby('Member_number')['itemDescription'].apply(list).tolist()
run_apriori(transactions2, support=0.1, confidence=0.6, output_file='rules_groceries_set2a.csv')
run_apriori(transactions2, support=0.15, confidence=0.5, output_file='rules_groceries_set2b.csv')
