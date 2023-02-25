import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')
import os
from dash import Input,State,Output,dcc,html,ctx,dash_table,Dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc 
import plotly.express as px
import plotly.offline as offline
import pandas as pd 
from flask import Flask
import base64
import numpy as np
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
def cerate_Numeric(id,placeholder):
    return dbc.Input(id=id,type='Number',placeholder=placeholder,debounce=True)

def save_plot(fig,name,save_path):
    if save_path:
        path=os.path.join(save_path,name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        path=name
    offline.plot(fig,filename=path,auto_open=True,include_plotlyjs='cdn')

def create_Tab1(df):
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    return dcc.Tab(label='Parallel Coordinates',id='PC-tab',children=[
    dbc.Row(dcc.Loading(id='PC-Loading',children=[dcc.Graph(id='PC-Graph',figure={})])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Color'),dcc.Dropdown(options=num_columns,id='PC-color-dropdown',),]),
    dbc.Row([html.H5('Lower and Upper Bound'),cerate_Numeric('PC-Lower-Bound',placeholder='Lower Bound'),cerate_Numeric('PC-Upper-Bound',placeholder='Upper Bound')]),
    dbc.Row([dcc.Input(id='PC-name',type='text',placeholder='Plot Title and Save Name',debounce=True),html.Button('Save Parallel Coordinates Plot',id='PC-save-plot')])
        	])
Tab2=dcc.Tab(label='Scatter Plot',id='SC-tab',children=[html.H1('Test2')])
Tab3=dcc.Tab(label='3d Scatter Plot',id='SC3D-tab',children=[html.H1('Test3')])



app=Dash(__name__,external_stylesheets=[dbc.themes.SKETCHY],suppress_callback_exceptions=True)
app.layout = dbc.Container([
                    #header
                    dbc.Row([
                            dbc.Col(html.H1(id='Header',children='Christophs Rapid Viz',className='Header')),html.Img(src='data_image/png;base64,{}'.format(image.decode()),style={'height':'100px','width':'100px'}),html.Hr()
                            ]),
                    #Table and GlobalSettings
                    dbc.Row([
                            dcc.Tabs(id='Table_Settings',children=[
                                    #input and casting
                                    dcc.Tab(label='Load Data',children=[dcc.Input(id='Path',type='text',placeholder='Path to data (supportes *.xlsx,*.parquet,*.csv)',debounce=True,style=min_style),dcc.Input(id='Save_Path',type='text',placeholder='Path to where the plots shall be saved',debounce=True,style=min_style),html.Button('Load Data',id='Load-Data-button',n_clicks=0,style=min_style),dcc.Checklist(['Automatically convert datatypes'],['Automatically convert datatypes'],id='change_dtypes'),html.Div(id='loading_info')]),
                                    # richtige App
                                    dcc.Tab(label='Data_Exploration',id='Data-exp',children=[])
                                    
                            ])
                            ]),
                    dcc.Store(id='Store',storage_type='session'),
                            ],fluid=True)
@app.callback(
    [Output('Store','data'),
    Output('loading_info','children')],
    Input('Path','value'),
    Input('Load-Data-button','n_clicks'), 
    Input('change_dtypes','value'))
def load_data(Path,n_clicks,change_dtypes):
    if Path is None:
        return[{},'Welcome to my Rapid Viz Tool! To start provide a Link to your Data']
    if n_clicks is None:
        raise PreventUpdate
    if n_clicks and ctx.triggered_id=='Load-Data-button':
        if Path:
            try:
                df=read_data(Path)
            except:
                return [{},html.H3(children='The data was not loaded sucessfully! It seems the format you provided is not supported, the data is corrupt, or the path is not valid!',style={'color':f'{colors["Error"]}'})]    
            #check box
            if change_dtypes=='Automatically convert datatypes':
                df=df.convert_dtypes()
                print(df.info())
            return [df.to_dict('records'),html.H3(children='Data Loaded Sucessfully!',style={'color':f'{colors["Sucess"]}'})]
        else:
            return [{},html.H3(children='The data was not laoded sucessfully! You must specify a valid Path',style={'color':f'{colors["Error"]}'})]

@app.callback(Output('Data-exp','children'),
    Input('Store','data'),prevent_initial_call=True)
def update_table(data):
    if data:
        df=pd.DataFrame.from_records(data)
        return dbc.Row(create_table(df)),dbc.Row(dcc.Tabs(id='graphs',children=[create_Tab1(df),Tab2,Tab3]))
    

@app.callback(
    Output('PC-Graph','figure'),
    Input('Store','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('PC-color-dropdown','value'),
    Input('PC-Lower-Bound','value'),
    Input('PC-Upper-Bound','value'),
    Input('PC-name','value'),
    Input('Save_Path','value'),
    Input('PC-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_PC_graph(data,rows,derived_virtual_selected_rows,color_column,up,low,title,save_path,save):
    df=pd.DataFrame.from_records(data)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows=[]
    dff=df if rows is None else pd.DataFrame(rows)
    fig=px.parallel_coordinates(dff,color=color_column)
    if title:
        fig.update_layout(title=title)
    if ctx.triggered_id=='PC-save-plot':
        save_plot(fig,name=f'{title}.html',save_path=save_path)
    return fig



        
    

            
         



if __name__ == "__main__":
    app.run(debug=True)