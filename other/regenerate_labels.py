import pandas as pd
import numpy as np

df_zetian = pd.read_csv('data/Bl122.txt', sep='\t', header=None)
df_dorian = pd.read_csv('data/Bl122_piezo_short.txt', sep='\t', header=None)
df_zetian.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
df_dorian.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)

df_dorian['label'] = 2
print(df_zetian)
print(df_dorian)

temp = np.ones(len(df_dorian))
temp = temp*2
print(temp)

for index, row in df_zetian.iterrows():
    start = row['onset']
    end = row['offset']
    label = row['label']
    print(f"\t({index}): {start}:{end}")

    if label == 1:
        for index, item in df_dorian.iterrows():
            start_d = item['onset']
            end_d = item['offset']

            if start_d >= start and end_d <= end:
                temp[index] = int(1)
                print(temp[index])

                #print(item['label'])
                #print(temp)
                #input('stope ')
    else:
        pass
df_dorian['onset'] = df_dorian['onset'].apply(lambda x: f"{x:.6f}")
df_dorian['offset'] = df_dorian['offset'].apply(lambda x: f"{x:.6f}")
df_dorian['label2'] = temp

columns_to_keep = ['onset', 'offset', 'label2']
df_dorian = df_dorian.loc[:, columns_to_keep]
df_dorian.to_csv("test.txt", index=False, sep='\t')

    