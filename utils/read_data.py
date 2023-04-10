import pandas as pd 
import os
def read_data(PATH,seperator,decimal):
    filename,ext=os.path.splitext(PATH)
    if ext=='.csv':
        df=pd.read_csv(PATH,delimiter=seperator,decimal=decimal)
        return df
    if ext=='.parquet':
        df=pd.read_parquet(PATH)
        return df
    if ext=='.xlsx':
        df=pd.read_excel(PATH,decimal=decimal,seperator=seperator)
