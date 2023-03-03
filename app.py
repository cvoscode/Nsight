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
from sklearn import preprocessing
from utils.styling import style_app
from utils.read_data import read_data,px_data
dirname=os.path.dirname(__file__)

#-------------------Styling---------------------#
external_stylesheets, figure_template,colors,min_style=style_app()
image_path=os.path.join(dirname,os.path.normpath('utils/images/logo.png'))
image=base64.b64encode(open(image_path,'rb').read())



#app=Dash(__name__, external_stylesheets=external_stylesheets)

def create_table(df,id,renameable,pagesize=3):
    return html.Div(dash_table.DataTable(
        id=id,
        columns=[{"name": i, "id": i, "deletable": True,'renamable': renameable} for i in df.columns],
        data=df.to_dict("records"),
        page_size=pagesize,
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
    print('save plot')
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
def create_Tab2(df):
    columns=df.columns.to_list()
    return dcc.Tab(label='Histogram and barplot',id='Col-tab',children=[
    dbc.Row(dcc.Loading(id='Col-Loading',children=[dcc.Graph(id='Col-Graph',figure={})])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Color'),dcc.Dropdown(options=columns,id='Col-x-dropdown',),]),
    dbc.Row([html.H5('Color and Pattern'),dcc.Dropdown(options=columns,id='Col-color-dropdown',),dcc.Dropdown(options=columns,id='Col-pattern-dropdown',)]),
    dbc.Row([dcc.Input(id='Col-name',type='text',placeholder='Plot Title and Save Name',debounce=True),html.Button('Save Histogram or Bar Plot',id='Col-save-plot')])
        	])
def create_Tab3(df):
    dff=df.describe(include='all')
    dff.insert(0,'statistical values',dff.index)
    print(dff)
    return dcc.Tab(label='Statistics',id='Col-tab',children=[dbc.Row(create_table(dff,'stats-table',False,pagesize=12)),
                                                             #TODO callbakc for export stats
                                                             dbc.Row(html.Button('Export Statistics',id='export-stats'))])

Tab3=dcc.Tab(label='Stats Tabelle',id='Stats-tab',children=[html.H1('Test3')])
Tab4=dcc.Tab(label='Scatter Plot 2D',id='SC-tab',children=[html.H1('Test2')])
Tab5=dcc.Tab(label='3d Scatter Plot',id='SC3D-tab',children=[html.H1('Test3')])
Tab6=dcc.Tab(label='Ridgid Plot',id='Rigid-tab',children=[html.H1('Test3')])
Tab7=dcc.Tab(label='Pareto Analysis ABC Analyse',id='Pareto-tab',children=[html.H1('Test3')])
Tab8=dcc.Tab(label='Correlation Heatmap',id='Corr-tab',children=[html.H1('Test3')])
Tab9=dcc.Tab(label='Dot',id='Dot-tab',children=[html.H1('Test3')])



app=Dash(__name__,external_stylesheets=[dbc.themes.SKETCHY],suppress_callback_exceptions=True)
app.layout = dbc.Container([
                    #header
                    dbc.Row([
                            dbc.Col(html.H1(id='Header',children='Christophs Rapid Viz',className='Header')),html.Img(src='data_image/png;base64,{}'.format(image.decode()),style={'height':'100px','width':'100px'}),html.Hr()
                            ]),
                    #Table and GlobalSettings
                    dbc.Row([
                            dcc.Tabs(id='Table_Settings',children=[
                                    #input and casting #TODO Scaling, column, renaming, label encoding 
                                    dcc.Tab(label='Load Data and gernal setting',children=[dcc.Input(id='Path',type='text',placeholder='Path to data (supportes *.xlsx,*.parquet,*.csv)',value=r'C:\Python\Christophs_Rapid_Viz\test_data.csv',debounce=True,style=min_style),dcc.Input(id='Save_Path',type='text',placeholder='Path to where the plots shall be saved',debounce=True,style=min_style),html.Button('Load Data',id='Load-Data-button',n_clicks=0,style=min_style),dcc.Checklist(['Automatically convert datatypes'],['Automatically convert datatypes'],id='change_dtypes'),html.Div(id='loading_info')]),
                                    # richtige App
                                    dcc.Tab(label='Data Transformation',id='Data-trans',children=[]),
                                    dcc.Tab(label='Data_Exploration',id='Data-exp',children=[])
                                    
                            ])
                            ]),
                    dcc.Store(id='store',storage_type='session'),
                            ],fluid=True)
@app.callback(
    [Output('store','data'),
    Output('loading_info','children')],
    State('Path','value'),
    Input('Load-Data-button','n_clicks'), 
    State('change_dtypes','value'),prevent_initial_call=True)
def load_data(Path,n_clicks,change_dtypes):
    if Path is None:
        return[{},'Welcome to my Rapid Viz Tool! To start provide a Link to your Data']
    if n_clicks and ctx.triggered_id=='Load-Data-button':
        if Path:
            try:
                df=px_data()
                #df=read_data(Path)
            except:
                return [{},html.H3(children='The data was not loaded sucessfully! It seems the format you provided is not supported, the data is corrupt, or the path is not valid!',style={'color':f'{colors["Error"]}'})]    
            #check box
            if change_dtypes=='Automatically convert datatypes':
                df=df.convert_dtypes()
                print(df.info())
            return [df.to_dict('records'),html.H3(children='Data Loaded Sucessfully!',style={'color':f'{colors["Sucess"]}'})]
        else:
            return [{},html.H3(children='The data was not laoded sucessfully! You must specify a valid Path',style={'color':f'{colors["Error"]}'})]
    


# callbacks for Data Transformation Layout
@app.callback(Output('Data-trans','children'),
    Input('store','data'))
def update_trans_layout(data):
    if ctx.triggered_id==('store'):
        df=pd.DataFrame.from_records(data)                  
        return  [dbc.Row(create_table(df,id='trans_table',renameable=True)),
                 dbc.Row([dbc.Col([html.H4('Transform Columns'),dcc.Dropdown(options=df.columns,id='trans-dropdown'),html.Button('Label Encode Column',id='label-encode-button'),html.Button('Scale Column Min/Max',id='scale-min/max-button'),html.Button('Standardize Column',id='standardize-button'),dcc.Checklist(['Scale all columns Min/Max','Standardize all columns'],[],id='scale-checklist',inline=True,),html.Button('Confirm Transformation',id='confirm-trans-button')])])]
    #update layout based on table
               
@app.callback(
        Output('trans_table','data'),
        State('trans_table','data'),
        State('trans-dropdown','value'),
        Input('label-encode-button','n_clicks'),
        Input('standardize-button','n_clicks'),
        Input('scale-min/max-button','n_clicks'),
        State('scale-checklist','value'),
        Input('confirm-trans-button','n_clicks')
        ,prevent_initial_call=True)
def transform_data(data,column,label,standard,scale,checklist,confirm):
    df=pd.DataFrame.from_records(data)
    if ctx.triggered_id=='label-encode-button':
        df[column]=preprocessing.LabelEncoder().fit_transform(df[column])
        return df.to_dict("records")
    if ctx.triggered_id=='standardize-button':
        df[column]=preprocessing.StandardScaler().fit_transform(df[column].values.reshape(-1, 1))
        return df.to_dict("records")
    if ctx.triggered_id=='scale-min/max-button':
        df[column]=preprocessing.MinMaxScaler().fit_transform(df[column].values.reshape(-1, 1))
        return df.to_dict("records")
    if ctx.triggered_id=='confirm-trans-button':
        if "Scale all columns Min/Max" in checklist: 
            dff=pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df),columns=df.columns)
            return dff.to_dict("records")
        if "Standardize all columns" in checklist: 
            dff=pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df),columns=df.columns)
            return dff.to_dict("records")
        else: return df.to_dict("records")

# Muss umgeschriben werden auf Button fertig von Data Transformation
@app.callback(Output('Data-exp','children'),
    State('trans_table','data'),
    Input('confirm-trans-button','n_clicks'))
def update_table(data,confirm):
    if data:
        df=pd.DataFrame.from_records(data)
        return dbc.Row(create_table(df,id='data_table',renameable=False)),dbc.Row(dcc.Tabs(id='graphs',children=[create_Tab1(df),create_Tab2(df),create_Tab3(df),Tab4,Tab5,Tab6,Tab7,Tab8,Tab9])),
    
#--------------------------Graph---------callbacks-------------
@app.callback(
    Output('PC-Graph','figure'),
    State('trans_table','data'),
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
    #TODO upper value and lower value are not in use right now
    fig=px.parallel_coordinates(dff,color=color_column)
    if title:
        fig.update_layout(title=title)
    if ctx.triggered_id=='PC-save-plot':
        save_plot(fig,name=f'{title}.html',save_path=save_path)
    return fig

@app.callback(
    Output('Col-Graph','figure'),
    State('trans_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('Col-color-dropdown','value'),
    Input('Col-x-dropdown','value'),
    Input('Col-pattern-dropdown','value'),
    Input('Col-name','value'),
    Input('Save_Path','value'),
    Input('Col-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_PC_graph(data,rows,derived_virtual_selected_rows,color_column,x,pattern,title,save_path,save):
    df=pd.DataFrame.from_records(data)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows=[]
    dff=df if rows is None else pd.DataFrame(rows)
    fig=px.histogram(dff,x=x,color=color_column,marginal='box',pattern_shape=pattern)
    if title:
        fig.update_layout(title=title)
    if ctx.triggered_id=='Col-save-plot':
        save_plot(fig,name=f'{title}.html',save_path=save_path)
    return fig


        
    

            
         



if __name__ == "__main__":
    app.run(debug=True)