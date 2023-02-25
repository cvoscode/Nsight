import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')
import os
from dash import Input,State,Output,dcc,html,ctx,dash_table,Dash
import dash_bootstrap_components as dbc 
import plotly.express as px
import plotly.offline as offline
import pandas as pd 
from flask import Flask
import base64
from utils.styling import style_app
from utils.read_data import read_data
dirname=os.path.dirname(__file__)

#-------------------Styling---------------------#
external_stylesheets, figure_template,colors,min_style=style_app()
image_path=os.path.join(dirname,os.path.normpath('utils/images/logo.png'))
image=base64.b64encode(open(image_path,'rb').read())



#app=Dash(__name__, external_stylesheets=external_stylesheets)

def create_table(df):
    return html.Div(dash_table.DataTable(
        id='data_table',
        columns=[{"name": i, "id": i, "deletable": True} for i in df.columns],
        data=df.to_dict("records"),
        page_size=3,
        editable=True,
        row_deletable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        selected_rows=[],
        selected_columns=[],
        page_current=0,
        style_table={"overflowX": "auto"},
        row_selectable="multi"),className='dbc-row-selectable')


Tab1=dcc.Tab(label='Parallel Coordinates',id='PC-tab',children=[html.H1('Test1')])
Tab2=dcc.Tab(label='Scatter Plot',id='SC-tab',children=[html.H1('Test2')])
Tab3=dcc.Tab(label='3d Scatter Plot',id='SC3D-tab',children=[html.H1('Test3')])



app=Dash(__name__,external_stylesheets=[dbc.themes.SKETCHY])
app.layout = dbc.Container([
                    #header
                    dbc.Row([
                            dbc.Col(html.H1(id='Header',children='Christophs Rapid Viz',className='Header')),html.Img(src='data_image/png;base64,{}'.format(image.decode()),style={'height':'100px','width':'100px'}),html.Hr()
                            ]),
                    #Table and GlobalSettings
                    dbc.Row([
                            dcc.Tabs(id='Table_Settings',children=[
                                    #input and casting
                                    dcc.Tab(label='Load Data',children=[dcc.Input(id='Path',type='text',placeholder='Path to data (supportes *.xlsx,*.parquet,*.csv)',debounce=True,style=min_style),html.Button('Load Data',id='Load-Data-button',n_clicks=0,style=min_style),dcc.Checklist(['Automatically convert datatypes'],['Automatically convert datatypes'],id='change_dtypes'),html.Div(id='loading_info')]),
                                    # richtige App
                                    dcc.Tab(label='Data_Exploration',id='Data-exp',children=[])
                                    
                            ])
                            ]),
                    dcc.Store(id='Store',storage_type='session'),
                            ],fluid=True)
@app.callback(
    [Output('Store','data'),
    Output('loading_info','children'),],
    Input('Path','value'),
    Input('Load-Data-button','n_clicks'), 
    Input('change_dtypes','value'),prevent_initial_call=True)
def load_data(Path,n_clicks,change_dtypes):
    if n_clicks and ctx.triggered_id=='Load-Data-button':
        if Path:
            try:
                df=read_data(Path)
            except:
                return [{},html.H3(children='The data was not loaded sucessfully! It seems the format you provided is not supported, the data is corrupt, or the path is not valid!',style={'color':f'{colors["Error"]}'})]    
            #check box
            if change_dtypes=='Automatically convert datatypes':
                df=df.convert_dtypes()
            return [df.to_dict('records'),html.H3(children='Data Loaded Sucessfully!',style={'color':f'{colors["Sucess"]}'})]
        else:
            return [{},html.H3(children='The data was not laoded sucessfully! You must specify a valid Path',style={'color':f'{colors["Error"]}'})]

@app.callback(Output('Data-exp','children'),
    Input('Store','data'),prevent_initial_call=True)
def update_table(data):
    if data:
        df=pd.DataFrame.from_records(data)
        return dbc.Row(create_table(df)),dbc.Row(dcc.Tabs(id='graphs',children=[Tab1,Tab2,Tab3]))


        
    

            
         



if __name__ == "__main__":
    app.run(debug=True)