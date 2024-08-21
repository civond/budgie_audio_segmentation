import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Read the DataFrame from CSV
df = pd.read_csv('csv/df_ch3.csv')
#df2 = pd.read_csv('csv/df_ch4.csv')
print("Original length:", len(df))

corr_cutoff = 0.5

sns.histplot(df['corrcoeff'], 
             bins=50, 
             kde=True,
             color='c',
             label="Ch3")
"""sns.histplot(df2['corrcoeff'], 
             bins=50, 
             kde=True,
             color='g',
             label='Ch4')"""

plt.axvline(x=corr_cutoff, linestyle='--', color='r')
plt.title("Correlation Distribution (KDE)")
plt.ylabel("Count")
plt.xlabel("Pearson's Corr. Val.")
plt.xlim(-1, 1)
plt.legend() 
plt.tight_layout()
plt.show()

#df = df[df['corrcoeff'] >= corr_cutoff]

df.loc[df['corrcoeff'] < corr_cutoff, 'label'] = 2


print(df)
df['corrcoeff'] = np.round(df['corrcoeff'],2)
df.reset_index(drop=True, inplace=False)

columns_to_keep = ['onset', 'offset', 'label']
df['onset'] = df['onset'].apply(lambda x: f"{x:.6f}")
df['offset'] = df['offset'].apply(lambda x: f"{x:.6f}")

df['onset'] = np.round(df['onset'],6)
df['offset'] = np.round(df['offset'],6)
df = df.loc[:, columns_to_keep]
df.to_csv("labels/bl122.txt", 
          header=False, 
          index=False,
          sep='\t',
          float_format='%.2f')
print(f"After drop length: {len(df)}")
