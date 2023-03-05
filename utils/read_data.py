import pandas as pd 

def read_data(PATH):
    try:
        df=pd.read_csv(PATH)
        return df
    except:
        return Exception

import plotly.express as px
def px_data():
    df = px.data.stocks()
    return df