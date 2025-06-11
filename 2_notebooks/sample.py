import pandas as pd
df = pd.read_csv(r"1_data/NFHS_5_Factsheets_Data.csv", nrows=0)  # only header
print(df.columns.tolist())
