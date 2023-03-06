import pandas as pd 
import os
def read_data(PATH):
    filename,ext=os.path.splitext(PATH)
    if ext=='.csv':
        df=pd.read_csv(PATH)
        return df
    if ext=='.parquet':
        df=pd.read_parquet(PATH)
        return df
    if ext=='.xlsx':
        df=pd.read_excel(PATH)
