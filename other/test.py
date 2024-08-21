import pandas as pd

df = pd.read_csv("copy.txt", sep='\t', header=None)
df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
df["label"] = df["label"].astype(int)

df['onset'] = df['onset'].apply(lambda x: f"{x:.6f}")
df['offset'] = df['offset'].apply(lambda x: f"{x:.6f}")

df.to_csv("welp.txt", index=False, sep='\t', header=None)
print(df)