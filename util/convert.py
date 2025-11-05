import pandas as pd
import json

# load json file into dataframe
with open('1mil_pw_structured.json', 'r') as f:
    data = json.load(f)

# convert to dataframe
df = pd.DataFrame(list(data.items()), columns=['hash', 'password'])

print(df.head)

# randomly sample 1000 entries for eval
eval_df = df.sample(n=1000, random_state=42)

# remove those entries from the original dataframe
train_df = df.drop(eval_df.index)

# save both as tsv
eval_df.to_csv('1K_eval.tsv', sep='\t', index=False)
train_df.to_csv('1M_train.tsv', sep='\t', index=False)


