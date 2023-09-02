import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = 'gemm_b16.csv'
save_file = 'select_metrices_median_b16.csv'
#csv_file = 'gemm_b32.csv'
#save_file = 'select_metrices_median_b32.csv'

df = pd.read_csv(csv_file, thousands=',')

with open('select_cols.txt', 'r') as f:
    cols = f.read()

cols = cols.split(',\n')[:-1]

def udf_split_data(x):
    ret = x.split('(')[0]
    ret = ret.split(',')
    return int(''.join(ret))

df_select = df[cols]
df_select['inst_executed [inst]'] = df_select['inst_executed [inst]'].apply(udf_split_data)

for col in cols:
    if 'smsp' in col:
        df_select[col] = df_select[col].apply(udf_split_data)

df_median = df_select.groupby(by=['Function Name', 'Grid Size', 'Block Size']).median()
df_median.to_csv(save_file)

