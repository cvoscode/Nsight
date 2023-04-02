""" This is Nsight. A web based tool to create visualizations of data.
I made this in my freetime and hope you can enjoy.
02.04.2023 cvoscode"""

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')
import os
from dash import Input,State,Output,dcc,html,ctx,dash_table,Dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc 
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go
import plotly.colors as plcolor
import plotly.io as pio
from ridgeplot import ridgeplot
import plotly.figure_factory as ff
import pandas as pd 
import ppscore as pps
from flask import Flask
from waitress import serve
import base64
import numpy as np
from sklearn import preprocessing
import flask
from utils.styling import style_app
from utils.plotting import RidgePlotFigureFactory_Custom
from utils.read_data import read_data
dirname=os.path.dirname(__file__)

#-------------------Styling---------------------#
external_style,colors,min_style,discrete_color_scale,color_scale,figure_temp=style_app()
image_path=os.path.join(dirname,os.path.normpath('utils/images/logo.png'))
image=base64.b64encode(open(image_path,'rb').read())
figure_template=load_figure_template(figure_temp)
pio.templates.default = f"{figure_temp}+watermark"

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
        row_selectable="multi"),className='dbc-row-selectable',style=min_style)
def cerate_Numeric(id,placeholder):
    return dbc.Input(id=id,type='Number',placeholder=placeholder,debounce=True,style=min_style)

def save_plot(fig,name,save_path):
    if save_path:
        path=os.path.join(save_path,name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        path=name
    offline.plot(fig,filename=path,auto_open=True,include_plotlyjs='cdn')

def create_Tab1(df):
    dff=df.describe(include='all')
    dff.insert(0,'statistical values',dff.index)
    return dcc.Tab(label='Statistics',id='Col-tab',children=[dbc.Row(create_table(dff,'stats-table',False,pagesize=12),style=min_style),
                                                             dbc.Row(dbc.Input(id='stats-name',type='text',placeholder='Name of the export',debounce=True,style=min_style),style=min_style),
                                                             dbc.Row(dbc.Button('Export Statistics as csv',id='export-stats',style=min_style),style=min_style),
                                                             dbc.Row(html.Div(id='stat-export',style=min_style))])    
def create_Tab2(df):
    columns=df.columns.to_list()
    return dcc.Tab(label='Histogram',id='Col-tab',children=[
    dbc.Row(dcc.Loading(id='Col-Loading',children=[])),
    dbc.Row([html.H4('Plot Settings'),html.Hr(),html.H5('Column')]),
    dbc.Row([dbc.Col(dcc.Dropdown(options=columns,id='Col-x-dropdown',placeholder='Select Column for Histogram',style=min_style),width=8),dbc.Col(dbc.ButtonGroup([dbc.Button("Previous",id='Col-previous-button'), dbc.Button("Next",id='Col-next-button')]),width=1),dbc.Col(dbc.ButtonGroup([dbc.Button('Save Histogram Plot',id='Col-save-plot'),dbc.Button('Open Plot on fullscreen',id='Col-popup')]),width=3)],style=min_style),
    dbc.Row([html.H5('Color and Pattern')]),
    dbc.Row([dbc.Col(dcc.Dropdown(options=columns,id='Col-color-dropdown',placeholder='Select Color Column',style=min_style)),dbc.Col(dcc.Dropdown(options=columns,id='Col-pattern-dropdown',placeholder='Select Pattern Column',style=min_style)),dbc.Col(dbc.Input(id='Col-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True))],style=min_style),
        	])

def create_Tab3(df):
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    return dcc.Tab(label='Parallel Coordinates',id='PC-tab',children=[
    dbc.Row(dcc.Loading(id='PC-Loading',children=[])),
    dbc.Row([html.H4('Plot Settings'),html.Hr(),html.H5('Color')]),
    dbc.Row([dbc.Col(dcc.Dropdown(options=num_columns,id='PC-color-dropdown',placeholder='Select Color Column',style=min_style),width=8),dbc.Col(dbc.ButtonGroup([dbc.Button('Save Parallel Coordinates Plot',id='PC-save-plot'),dbc.Button('Open Plot on fullscreen',id='PC-popup')]),width=3)],style=min_style),
    dbc.Row([html.H5('Lower and Upper Bound')]),
    dbc.Row([dbc.Col(cerate_Numeric('PC-Lower-Bound',placeholder='Lower Bound (without function)')),dbc.Col(cerate_Numeric('PC-Upper-Bound',placeholder='Upper Bound (without function)')),dbc.Col(dbc.Input(id='PC-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style))],style=min_style),
        	])

def create_Tab4(df):
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    columns=df.columns
    return dcc.Tab(label='Scatterplot 2D',id='SC-tab',children=[
    dbc.Row(dcc.Loading(id='SC-Loading',children=[])),
    dbc.Row([html.H4('Plot Settings'),html.Hr(),html.H5('Columns')]),

    dbc.Row([dbc.Col(dcc.Dropdown(options=num_columns,id='SC-x-dropdown',placeholder='Select the x-Column',style=min_style),width=11),dbc.Col(dbc.ButtonGroup([dbc.Button("Previous",id='SC-x-previous-button'), dbc.Button("Next",id='SC-x-next-button')]),width=1)],style=min_style),
    dbc.Row([dbc.Col(dcc.Dropdown(options=num_columns,id='SC-y-dropdown',placeholder='Select the y-Column',style=min_style),width=11),dbc.Col(dbc.ButtonGroup([dbc.Button("Previous",id='SC-y-previous-button'), dbc.Button("Next",id='SC-y-next-button')]),width=1)],style=min_style),    
    dbc.Row([html.H5('Color and Size')]),
    dbc.Row([dbc.Col(dcc.Dropdown(options=columns,id='SC-color-dropdown',placeholder='Select Color Column',style=min_style)),dbc.Col(dcc.Dropdown(options=num_columns,id='SC-size-dropdown',placeholder='Select Size Column (will not work if column values are negativ)',style=min_style)),dbc.Col(dbc.Input(id='SC-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style)),dbc.Col(dbc.ButtonGroup([dbc.Button('Save Scatter Plot',id='SC-save-plot'), dbc.Button('Open Plot on fullscreen',id='SC-popup',)]))],style=min_style),
        	])




def create_Tab5(df):
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    columns=df.columns
    return dcc.Tab(label='Scatterplot 3D',id='SC3D-tab',children=[
    dbc.Row(dcc.Loading(id='SC3D-Loading',children=[])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([html.H5('Columns'),]),
    dbc.Row([dbc.Col(dcc.Dropdown(options=num_columns,id='SC3D-x-dropdown',placeholder='Select the x-Column',style=min_style),width=11),dbc.Col(dbc.ButtonGroup([dbc.Button("Previous",id='SC3D-x-previous-button'), dbc.Button("Next",id='SC3D-x-next-button')]),width=1)],style=min_style),
    dbc.Row([dbc.Col(dcc.Dropdown(options=num_columns,id='SC3D-y-dropdown',placeholder='Select the y-Column',style=min_style),width=11),dbc.Col(dbc.ButtonGroup([dbc.Button("Previous",id='SC3D-y-previous-button'), dbc.Button("Next",id='SC3D-y-next-button')]),width=1)],style=min_style),    
    dbc.Row([dbc.Col(dcc.Dropdown(options=num_columns,id='SC3D-z-dropdown',placeholder='Select the z-Column',style=min_style),width=11),dbc.Col(dbc.ButtonGroup([dbc.Button("Previous",id='SC3D-z-previous-button'), dbc.Button("Next",id='SC3D-z-next-button')]),width=1)],style=min_style),    
    dbc.Row([html.H5('Color and Size')]),
    dbc.Row([dbc.Col(dcc.Dropdown(options=columns,id='SC3D-color-dropdown',placeholder='Select Color Column',style=min_style)),dbc.Col(dcc.Dropdown(options=num_columns,id='SC3D-size-dropdown',placeholder='Select Size Column (will not work if column values are negativ)',style=min_style)),dbc.Col(dbc.Input(id='SC3D-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style)),dbc.Col(dbc.ButtonGroup([dbc.Button('Save 3D Scatter Plot',id='SC3D-save-plot'), dbc.Button('Open Plot on fullscreen',id='SC3D-popup',)]))],style=min_style),
    ])


def create_Tab6(df):
    return dcc.Tab(label='Ridge',id='Ridge-tab',children=[
    dbc.Row(dcc.Loading(id='Ridge-Loading',children=[])),
    dbc.Row([html.H4('Plot Settings'),html.Hr(),html.H5('Distance')]),
    dbc.Row([dbc.Col(dbc.Input(id='Ridge-space',type='number',placeholder='Space between the Columns (in times the highest distribution)',value=2,style=min_style))],style=min_style),
    dbc.Row([dbc.Col(dbc.Input(id='Ridge-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True),width=9),dbc.Col(dbc.ButtonGroup([dbc.Button('Save Ridge Plot',id='Ridge-save-plot',),dbc.Button('Open Plot on fullscreen',id='Ridge-popup')]))],style=min_style),

    ])

#Tab7=dcc.Tab(label='Pareto Analysis ABC Analyse',id='Pareto-tab',children=[html.H1('Test3')])

def create_Tab8(df):
    return dcc.Tab(label='Correlations',id='Corr-tab',children=[
    dbc.Row(dcc.Loading(id='Corr-Loading',children=[])),
    dbc.Row([html.H4('Plot Settings'),html.Hr()]),
    dbc.Row([dbc.Col(dcc.Dropdown(options=['Over all', 'Just with Target column'],id='Corr-scope',placeholder='Select Correltation Scope',style=min_style)),dbc.Col(dcc.Dropdown(id='Corr-columns',placeholder='Select Target Column',style=min_style))],style=min_style),
    dbc.Row([html.H5('Correlation Type')]),
    dbc.Row([dbc.Col(dcc.Dropdown(options=['pearson','spearman','kendall','Power Predictive Score'],id='Corr-type-dropdown',placeholder='Select Correlation Type',style=min_style)),dbc.Col(dbc.Input(id='Corr-name',type='text',placeholder='Input Plot Title (This is also the file name when saving)',debounce=True,style=min_style)),dbc.Col(dbc.Col(dbc.ButtonGroup([dbc.Button('Save Correlations Plot',id='Corr-save-plot',),dbc.Button('Open Plot on fullscreen',id='Corr-popup')])))],style=min_style),

    ])
def create_Export(): 
    return dcc.Tab(label='Export Data',id='Export-tab',children=[dbc.Row([dbc.Input(id='Export-name',type='text',placeholder='Name of the Data Export (including the extension - possible extensions are: .csv, .xlsx, .parquet)',debounce=True,style=min_style),dbc.Button('Export Data',id='Export Data',style=min_style),html.Div(id='Export-div')],style=min_style)])
#----------------------------------------------------------------------------
server = flask.Flask(__name__)
app=Dash(__name__,external_stylesheets=[external_style],suppress_callback_exceptions=True,server=server)


app.layout = dbc.Container([
                    #header
                    dbc.Row([
                            dbc.Col(html.H1(id='Header',children=' Nsight',className='Header')),html.Img(src=app.get_asset_url('logo.png'),style={'height':'60px','width':'105px'}),html.Hr()
                            ]),
                    #Table and GlobalSettings
                    dbc.Row([
                            dcc.Tabs(id='Table_Settings',children=[
                                    #TODO displaying data types
                                    dcc.Tab(label='Load Data and Gernal settings',children=[
                                            dbc.Row([dbc.Col([dbc.Row(dbc.Input(id='Path',type='text',placeholder='Path to data (supportes *.xlsx,*.parquet,*.csv)',debounce=True,style=min_style),style=min_style),dbc.Row(dbc.Input(id='Save_Path',type='text',placeholder='Path to where the plots shall be saved',debounce=True,style=min_style),style=min_style),dbc.Row(dbc.Button('Load Data',id='Load-Data-button',n_clicks=0,style=min_style),style=min_style),dbc.Row(dcc.Checklist(['Automatically convert datatypes'],['Automatically convert datatypes'],id='change_dtypes',style=min_style),style=min_style),dbc.Row(html.Div(id='loading_info',style=min_style),style=min_style)]),
                                                    dbc.Col([dbc.Row(children=[dcc.Markdown('Welcome to Nsight, a web based tool to visualize your Data! \n\n To start please insert the path of data you want to visualize and click the Button Load Data! \n\n PS: If you want to clear a dropdown, just use Backspace or Del',style={'text-align':'center'})]),
                                                            dbc.Row(html.Img(src=app.get_asset_url('pexels-anna-nekrashevich-6802049.jpg'),style={'height':'80%','width':'80%','display':'block','margin-left':'auto','margin-right':'auto',})),]
                                                            ),]),]),
                                    # richtige App
                                    dcc.Tab(label='Data Transformation',id='Data-trans',children=[]),
                                    dcc.Tab(label='Data Exploration',id='Data-exp',children=[])
                                    
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
        return[{},'Welcome to my Nsight! To start provide a Link to your Data']
    if n_clicks and ctx.triggered_id=='Load-Data-button':
        if Path:
            try:
                #df=px_data()
                Path=Path.strip('\"')
                df=read_data(os.path.normpath(Path))
            except:
                return [{},html.H6(children='The data was not loaded sucessfully! It seems the format you provided is not supported, the data is corrupt, or the path is not valid!',style={'color':f'{colors["Error"]}'})]    
            #check box
            if change_dtypes=='Automatically convert datatypes':
                df=df.convert_dtypes()
            return [df.to_dict('records'),html.H6(children='Data Loaded Sucessfully!',style={'color':f'{colors["Sucess"]}'})]
        else:
            return [{},html.H6(children='The data was not laoded sucessfully! You must specify a valid Path',style={'color':f'{colors["Error"]}'})]
    


# callbacks for Data Transformation Layout
#TODO build trasnrom Columns on trans_table
@app.callback(Output('Data-trans','children'),
    Input('store','data'),prevent_initial_call=True)
def update_trans_layout(data):
    if ctx.triggered_id==('store'):
        if data:
            df=pd.DataFrame.from_records(data)                 
            return  [dbc.Row(create_table(df,id='trans_table',renameable=True)),
                 dbc.Row([html.H4('Transform Columns'),html.Hr(),
                        dbc.Col([ dbc.Row([dcc.Dropdown(options=df.columns,id='trans-dropdown',placeholder='Select Column to transform',style=min_style)],style=min_style),dbc.Row([dbc.Col(dbc.Button('Label Encode Column',id='label-encode-button',style=min_style)),dbc.Col(dbc.Button('Scale Column Min/Max',id='scale-min/max-button',style=min_style)),dbc.Col(dbc.Button('Standardize Column',id='standardize-button',style=min_style))]),dbc.Row([dcc.Dropdown(options=['object','int64','float64','datetime64[ns]','bool'],id='dtypes-dropdown',placeholder='Select Column to transform',style=min_style),dbc.Button('Change Data Type of the selected column',id='change-dtype-button',style=min_style),html.Div(id='dtype-div')],style=min_style)]),
                        dbc.Col([ dbc.Row([dbc.Input(id='varianz-value',type='Number',min=0.000001,max=0.9,step=0.000001,placeholder='Input a variance treshold for the variance filter',debounce=True,style=min_style),dbc.Button('Filter columns with a low variance (only on numeric columns)',id='filter-varianz-button',style=min_style),dbc.Button('Drop Rows with missing values',id='dropna-button',style=min_style)],style=min_style),
                                    dbc.Row([dbc.Button('Scale all numerical columns Min/Max',id='all-minmax-button',style=min_style),dbc.Button('Standardize all numerical columns',id='all-standard-button',style=min_style),dbc.Button('Label Encode all categorical columns',id='all-label-button',style=min_style)],style={'margin':'8px 2px 8px'})]),
                        dbc.Row(dbc.Button('Confirm Transformation',id='confirm-trans-button',style=min_style),style=min_style)])]
               
@app.callback(
        Output('trans_table','data'),
        Output('dtype-div','children'),
        State('trans_table','data'),
        State('trans-dropdown','value'),
        Input('label-encode-button','n_clicks'),
        Input('standardize-button','n_clicks'),
        Input('scale-min/max-button','n_clicks'),
        Input('all-minmax-button','n_clicks'),
        Input('all-standard-button','n_clicks'),
        Input('all-label-button','n_clicks'),
        Input('confirm-trans-button','n_clicks'),
        Input('change-dtype-button','n_clicks'),
        State('dtypes-dropdown','value'),
        State('varianz-value','value'),
        Input('filter-varianz-button','n_clicks'),
        Input('dropna-button','n_clicks')
        ,prevent_initial_call=True)
def transform_data(data,column,label,standard,scale,all_minmax,all_standard,all_label, confirm,change_dtypes_button,dtype,varianz,filter_var,dropna):
    df=pd.DataFrame.from_records(data)
    num_columns=df.select_dtypes(include=np.number).columns.to_list()
    if ctx.triggered_id=='filter-varianz-button':
        if not varianz:
            return df.to_dict("records"),html.H6(children=f'No columns were droped! You must provide a variance threshold fot the filter to work!',style={'color':f'{colors["Error"]}'})
        else:
            variance=df[num_columns].var()
            drop_cols=[]
            for i,col in enumerate(num_columns):
                if variance[i]<=float(varianz):
                    drop_cols.append(col)
            if drop_cols:
                dff=df.drop(columns=drop_cols)
                return dff.to_dict("records"),html.H6(children=f'The following columns were dropped: {drop_cols}, since the variance is lower than the varianz threshold!',style={'color':f'{colors["Sucess"]}'})
            else:
                return df.to_dict("records"),html.H6(children=f'No columns were droped! There are no columns with a lower variance than the threshold.',style={'color':f'{colors["Info"]}'})
    if ctx.triggered_id=='confirm-trans-button':
        return df.to_dict("records"),html.H6(children=f'The Data was transformed sucessfully! You can now proceed to the Data Exploration Tab',style={'color':f'{colors["Sucess"]}'})
    if ctx.triggered_id=='all-minmax-button': 
        df[num_columns]=preprocessing.MinMaxScaler().fit_transform(df[num_columns])
        num_columns=', '.join(num_columns)
        return df.to_dict("records"),html.H6(children=f'The column(s) "{num_columns}"  was/were scaled sucessfully to Min/Max!',style={'color':f'{colors["Sucess"]}'})
    if ctx.triggered_id=='all-standard-button': 
        df[num_columns]=preprocessing.StandardScaler().fit_transform(df[num_columns])
        num_columns=', '.join(num_columns)
        return df.to_dict("records"),html.H6(children=f'The column(s) "{num_columns}" was/were standardized sucessfully!',style={'color':f'{colors["Sucess"]}'})
    if ctx.triggered_id=='all-label-button':
        cat_cols=df.select_dtypes(exclude=np.number).columns.to_list()
        for col in cat_cols:
            df[col]=preprocessing.LabelEncoder().fit_transform(df[col])
        cat_cols=', '.join(cat_cols)
        return df.to_dict("records"),html.H6(children=f'The column(s) "{cat_cols}" was/were Label Encoded sucessfully!',style={'color':f'{colors["Sucess"]}'})
    if ctx.triggered_id=='dropna-button':
        dff=df.dropna()
        dropped=len(df)-len(dff)
        return dff.to_dict("records"),html.H6(children=f'{dropped} rows with missing values were dropped!',style={'color':f'{colors["Sucess"]}'})
    if column:
        if ctx.triggered_id=='change-dtype-button':
            try:
                df[column]=df[column].astype(dtype)
                return df.to_dict("records"),html.H6(children=f'Changing the data type of "{column}" to "{dtype}" was scessfully!',style={'color':f'{colors["Sucess"]}'})
            except: return df.to_dict("records"), html.H6(children=f'Changing the data type to "{dtype}" was NOT scessfully! It seems the conversation to "{dtype}" for the column "{column}" is not possible',style={'color':f'{colors["Error"]}'})
        try:
            if ctx.triggered_id=='label-encode-button':
                df[column]=preprocessing.LabelEncoder().fit_transform(df[column])
                return df.to_dict("records"),html.H6(children=f'The Column "{column}" was Label Encoded sucessfully!',style={'color':f'{colors["Sucess"]}'})
            if ctx.triggered_id=='standardize-button':
                df[column]=preprocessing.StandardScaler().fit_transform(df[column].values.reshape(-1, 1))
                return df.to_dict("records"),html.H6(children=f'The Column "{column}" was Standardized sucessfully!',style={'color':f'{colors["Sucess"]}'})
            if ctx.triggered_id=='scale-min/max-button':
                df[column]=preprocessing.MinMaxScaler().fit_transform(df[column].values.reshape(-1, 1))
                return df.to_dict("records"),html.H6(children=f'The Column "{column}" was Scaled sucessfully!',style={'color':f'{colors["Sucess"]}'})
        except:
            return df.to_dict("records"),html.H6(children=f'Something went wrong!!! Maybe you tried to scale/standardize a non numeric column.',style={'color':f'{colors["Error"]}'})
    else: raise PreventUpdate
    

@app.callback(Output('Data-exp','children'),
    State('trans_table','data'),
    Input('confirm-trans-button','n_clicks'))
def update_table(data,confirm):
    if data:
        df=pd.DataFrame.from_records(data)
        return dbc.Row(create_table(df,id='data_table',renameable=False)),dbc.Row(dcc.Tabs(id='graphs',children=[create_Tab1(df),create_Tab2(df),create_Tab3(df),create_Tab4(df),create_Tab5(df),create_Tab6(df),create_Tab8(df),create_Export()])),
    
@app.callback(Output('Corr-columns','options'),
              Output('Corr-columns','disabled'),
              State('data_table','data'),
              Input('Corr-scope','value'),)
def update_corr_columns(data,scope):
    if scope=='Just with Target column':
        df=pd.DataFrame.from_records(data)
        return df.select_dtypes(include=np.number).columns,False
    else:
        return [],True

#--------------------------Graph---------callbacks-------------
@app.callback(Output('stats-table','data'),
              State('data_table','data'),
              Input('data_table','derived_virtual_data'),
            Input('data_table','derived_virtual_selected_rows'))
def update_stats(data,rows,derived_virtual_selected_rows):
    df=pd.DataFrame.from_records(data)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows=[]
    dff=df if rows is None else pd.DataFrame(rows)
    dfff=dff.describe(include='all')
    dfff.insert(0,'statistical values',dfff.index)
    return dfff.to_dict('records')

@app.callback(
        Output('stat-export','children'),
        Input('export-stats','n_clicks'),
        State('stats-name','value'),
        State('stats-table','data'),
        State('Save_Path','value'),
)
def export_Stats(n_clicks,name,data,save_path):
    if ctx.triggered_id=='export-stats':
        df=pd.DataFrame.from_records(data)
        if not name:
            name='stats'
        if save_path:
            path=os.path.join(save_path,f'{name}.csv')
        else:
            path=f'{name}.csv'
        df.to_csv(path)
        return html.H5(f"Statistics are saved sucessfully under '{path}'",style={'color':f'{colors["Sucess"]}'})


@app.callback(
    Output('PC-Loading','children'),
    State('data_table','data'),
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
    dimensions = list([dict(range = [dff[col].min(),dff[col].max()],
         label = col, values = dff[col],multiselect = True,) for col in dff.select_dtypes(include=np.number).columns])
    if not title:
            color_column_=f'_{color_column}' if color_column else ''
            title_=f'PC{color_column_}'
            layout=go.Layout(title={'text':title_})
    else:
        layout=go.Layout(title={'text':title})
    if len(dimensions)<12:
        labelangle=0
    else:
        labelangle=-45
    if color_column:
        fig=go.Figure(data=go.Parcoords(name=figure_temp,dimensions=dimensions,labelangle=labelangle,labelside='bottom',line = dict(color = dff[color_column],colorscale = color_scale,showscale = True, colorbar = {'title': color_column}),unselected=dict(line={'opacity':0.1})),layout=layout)
    else:
        fig=go.Figure(data=go.Parcoords(name=figure_temp,dimensions=dimensions,labelangle=labelangle,labelside='bottom',unselected=dict(line={'opacity':0.1})),layout=layout)
    if ctx.triggered_id=='PC-save-plot':
        save_plot(fig,name=f'{title}.html',save_path=save_path)
    return [dbc.Modal(id='PC-Modal',children=[
                    dbc.ModalHeader(dbc.ModalTitle('Parallel Coordinates')),
                    dbc.ModalBody(dcc.Graph(id='PC-Graph',figure=fig,style={'height':'100%','width':'100%'})),
                ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='PC-Graph',figure=fig)]  
 
@app.callback(Output('PC-Modal','is_open'),
              Input('PC-popup','n_clicks'),
              prevent_initial_call=True,)
def open_modal(popup):
    if ctx.triggered_id=='PC-popup':
        return True

@app.callback( Output('Col-x-dropdown','value'),
               State('Col-x-dropdown','value'),
               State('Col-x-dropdown','options'),
               Input('Col-previous-button','n_clicks'),
               Input('Col-next-button','n_clicks'))
def update_dopdown(value,options,previous,next_element):
    if value:
        index=options.index(value)
        if ctx.triggered_id=='Col-next-button':
            index=index+1
            if index>=len(options):
                index=0
            return options[index]
        if ctx.triggered_id=='Col-previous-button':
            index=index-1
            if index<0:
                index=len(options)-1
            return options[index]
        
@app.callback(Output('data_table','style_header_conditional'),
              Input('Col-x-dropdown','value'),
              Input('SC-x-dropdown','value'),
              Input('SC-y-dropdown','value'),
              Input('SC3D-x-dropdown','value'),
              Input('SC3D-y-dropdown','value'),
              Input('SC3D-z-dropdown','value'),)
   
def update_table_color(value,x,y,x_,y_,z_):
    if ctx.triggered_id=='Col-x-dropdown':
        return [{'if': {'column_id': value},
                'backgroundColor': colors['Selected'],
                'color': colors['Selected_text']}]
    if ctx.triggered_id=='SC-x-dropdown' or ctx.triggered_id=='SC-y-dropdown':
        if x and y:
            return [{'if': {'column_id': x},
                    'backgroundColor': colors['Selected'],
                    'color': colors['Selected_text']},
                    {'if': {'column_id': y},
                    'backgroundColor': colors['Selected'],
                    'color': colors['Selected_text']},
                    ]
        if x:
            return [{'if': {'column_id': x},
                    'backgroundColor': colors['Selected'],
                    'color': colors['Selected_text']}]   
        if y:
            return [{'if': {'column_id': y},
                    'backgroundColor': colors['Selected'],
                    'color': colors['Selected_text']}]
    if ctx.triggered_id=='SC3D-x-dropdown' or ctx.triggered_id=='SC3D-y-dropdown' or ctx.triggered_id=='SC3D-z-dropdown': 
        if x_:
            return [{'if': {'column_id': x_},
                    'backgroundColor': colors['Selected'],
                    'color': colors['Selected_text']}]   
        if y_:
            return [{'if': {'column_id': y_},
                    'backgroundColor': colors['Selected'],
                    'color': colors['Selected_text']}]  
        if z_:
            return [{'if': {'column_id': z_},
                    'backgroundColor': colors['Selected'],
                    'color': colors['Selected_text']}]    





@app.callback(
    Output('Col-Loading','children'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('Col-color-dropdown','value'),
    Input('Col-x-dropdown','value'),
    Input('Col-pattern-dropdown','value'),
    Input('Col-name','value'),
    Input('Save_Path','value'),
    Input('Col-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_Col_graph(data,rows,derived_virtual_selected_rows,color_column,x,pattern,title,save_path,save):
    if x:
        df=pd.DataFrame.from_records(data)
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows=[]
        dff=df if rows is None else pd.DataFrame(rows)
        if color_column:
            n_colors=len(dff[color_column].unique())
            color_values=plcolor.sample_colorscale(discrete_color_scale,[n/(n_colors -1) for n in range(n_colors)])
            fig=px.histogram(dff,x=x,color=color_column,marginal='box',pattern_shape=pattern,template=figure_template,color_discrete_sequence=color_values)
        else:
            fig=px.histogram(dff,x=x,marginal='box',pattern_shape=pattern,template=figure_template)
        if not title:
            color_column=f'_{color_column}' if color_column else ''
            pattern=f'_{pattern}' if pattern else ''
            title=f'Hist_{x}{color_column}{pattern}'
        fig.update_layout(title=title)
        if fig and ctx.triggered_id=='Col-save-plot':
            save_plot(fig,name=f'{title}.html',save_path=save_path)
        return [dbc.Modal(id='Col-Modal',children=[
                    dbc.ModalHeader(dbc.ModalTitle('Histogramm')),
                    dbc.ModalBody(dcc.Graph(id='Col-Graph',figure=fig,style={'height':'100%','width':'100%'})),
                ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='Col-Graph',figure=fig)]  
    else: return [dbc.Modal(id='Col-Modal',children=[
                    dbc.ModalHeader(dbc.ModalTitle('Histogramm')),
                    dbc.ModalBody(dcc.Graph(id='Col-Graph',figure={})),
                ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='Col-Graph',figure={}),html.H5(children=['The plot will be displayed here. To display the plot you must first choose any settings'],style={'color':f'{colors["Info"]}'})]

@app.callback(Output('Col-Modal','is_open'),
              Input('Col-popup','n_clicks'),
              prevent_initial_call=True,)
def open_modal(popup):
    if ctx.triggered_id=='Col-popup':
        return True
    
@app.callback(
    Output('SC-Loading','children'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('SC-color-dropdown','value'),
    Input('SC-x-dropdown','value'),
    Input('SC-y-dropdown','value'),
    Input('SC-size-dropdown','value'),
    Input('SC-name','value'),
    Input('Save_Path','value'),
    Input('SC-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_SC_graph(data,rows,derived_virtual_selected_rows,color_column,x,y,size,title,save_path,save):
    if x or y:
        df=pd.DataFrame.from_records(data)
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows=[]
        dff=df if rows is None else pd.DataFrame(rows)
        if size and dff[size].min()<0:
            size=None
        if color_column:
            if color_column not in df.select_dtypes(include=np.number).columns:
                n_colors=len(dff[color_column].unique())
                color_values=plcolor.sample_colorscale(discrete_color_scale,[n/(n_colors -1) for n in range(n_colors)])
                fig=px.scatter(dff,x=x,y=y,color=color_column,size=size,template=figure_template,color_discrete_sequence=color_values,trendline='ols')
            else:
                a_,b_,c_,d_,color_scale,template=style_app()
                fig=px.scatter(dff,x=x,y=y,color=color_column,trendline='ols',size=size,marginal_x='box',marginal_y='box',template=figure_template,color_continuous_scale=color_scale)
        else:
            fig=px.scatter(dff,x=x,y=y,trendline='ols',size=size,marginal_x='box',marginal_y='box',template=figure_template)
        if not title:
            color_column=f'_{color_column}' if color_column else ''
            size=f'_{size}' if size else ''
            x=f'_{x}' if x else ''
            y=f'_{y}' if y else ''
            title=f'SC{x}{y}{color_column}{size}'
        fig.update_layout(title=title,autosize=True)
        if ctx.triggered_id=='SC-save-plot':
            save_plot(fig,name=f'{title}.html',save_path=save_path)
        return [dbc.Modal(id='SC-Modal',children=[
                        dbc.ModalHeader(dbc.ModalTitle('2D Scatterplot')),
                        dbc.ModalBody(dcc.Graph(id='SC-Graph',figure=fig,style={'height':'100%','width':'100%'})),
                    ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='SC-Graph',figure=fig)]
      
    else: return [dbc.Modal(id='SC-Modal',children=[
                    dbc.ModalHeader(dbc.ModalTitle('2D Scatterplot')),
                    dbc.ModalBody(dcc.Graph(id='SC-Graph',figure={})),
                ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='SC-Graph',figure={}),html.H5(children=['The plot will be displayed here. To display the plot you must first choose any settings'],style={'color':f'{colors["Info"]}'})]

@app.callback(Output('SC-Modal','is_open'),
              Input('SC-popup','n_clicks'),
              prevent_initial_call=True,)
def open_modal(popup):
    if ctx.triggered_id=='SC-popup':
        return True
    

@app.callback( Output('SC-x-dropdown','value'),
               State('SC-x-dropdown','value'),
               State('SC-x-dropdown','options'),
               Input('SC-x-previous-button','n_clicks'),
               Input('SC-x-next-button','n_clicks'))
def update_dopdown(value,options,previous,next_element):
    if value:
        index=options.index(value)
        if ctx.triggered_id=='SC-x-next-button':
            index=index+1
            if index>=len(options):
                index=0
            return options[index]
        if ctx.triggered_id=='SC-x-previous-button':
            index=index-1
            if index<0:
                index=len(options)-1
            return options[index]
@app.callback( Output('SC-y-dropdown','value'),
               State('SC-y-dropdown','value'),
               State('SC-y-dropdown','options'),
               Input('SC-y-previous-button','n_clicks'),
               Input('SC-y-next-button','n_clicks'))
def update_dopdown(value,options,previous,next_element):
    if value:
        index=options.index(value)
        if ctx.triggered_id=='SC-y-next-button':
            index=index+1
            if index>=len(options):
                index=0
            return options[index]
        if ctx.triggered_id=='SC-y-previous-button':
            index=index-1
            if index<0:
                index=len(options)-1
            return options[index]





@app.callback(
    Output('SC3D-Loading','children'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('SC3D-color-dropdown','value'),
    Input('SC3D-x-dropdown','value'),
    Input('SC3D-y-dropdown','value'),
    Input('SC3D-z-dropdown','value'),
    Input('SC3D-size-dropdown','value'),
    Input('SC3D-name','value'),
    Input('Save_Path','value'),
    Input('SC3D-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_SC3D_graph(data,rows,derived_virtual_selected_rows,color_column,x,y,z,size,title,save_path,save):
    if x and y and z:
        df=pd.DataFrame.from_records(data)
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows=[]
        dff=df if rows is None else pd.DataFrame(rows)
        if color_column:
            if color_column not in df.select_dtypes(include=np.number).columns:
                n_colors=len(dff[color_column].unique())
                color_values=plcolor.sample_colorscale(discrete_color_scale,[n/(n_colors -1) for n in range(n_colors)])
                fig=px.scatter_3d(dff,x=x,y=y,z=z,color=color_column,template=figure_template,color_discrete_sequence=color_values)
            else:
                a_,b_,c_,d_,color_scale,template=style_app()
                fig=px.scatter_3d(dff,x=x,y=y,z=z,color=color_column,size=size,template=figure_template,color_continuous_scale=color_scale)
        else:
            fig=px.scatter_3d(dff,x=x,y=y,z=z,size=size,template=figure_template)
        if not title:
            color_column=f'_{color_column}' if color_column else ''
            size=f'_{size}' if size else ''
            title=f'SC3D_{x}_{y}_{z}{color_column}{size}'
            fig.update_layout(title=title)
        if ctx.triggered_id=='SC3D-save-plot':
            save_plot(fig,name=f'{title}.html',save_path=save_path)
        return [dbc.Modal(id='SC3D-Modal',children=[
                        dbc.ModalHeader(dbc.ModalTitle('3D Scatterplot')),
                        dbc.ModalBody(dcc.Graph(id='SC3D-Graph',figure=fig,style={'height':'100%','width':'100%'})),
                    ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='SC3D-Graph',figure=fig)]
    else: return [dbc.Modal(id='SC3D-Modal',children=[
                    dbc.ModalHeader(dbc.ModalTitle('3D Scatterplot')),
                    dbc.ModalBody(dcc.Graph(id='SC3D-Graph',figure={})),
                ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='SC3D-Graph',figure={}),html.H5(children=['The plot will be displayed here. To display the plot you must first choose any settings'],style={'color':f'{colors["Info"]}'})]

@app.callback(Output('SC3D-Modal','is_open'),
              Input('SC3D-popup','n_clicks'),
              prevent_initial_call=True,)
def open_modal(popup):
    if ctx.triggered_id=='SC3D-popup':
        return True

@app.callback( Output('SC3D-x-dropdown','value'),
               State('SC3D-x-dropdown','value'),
               State('SC3D-x-dropdown','options'),
               Input('SC3D-x-previous-button','n_clicks'),
               Input('SC3D-x-next-button','n_clicks'))
def update_dopdown(value,options,previous,next_element):
    if value:
        index=options.index(value)
        if ctx.triggered_id=='SC3D-x-next-button':
            index=index+1
            if index>=len(options):
                index=0
            return options[index]
        if ctx.triggered_id=='SC3D-x-previous-button':
            index=index-1
            if index<0:
                index=len(options)-1
            return options[index]
@app.callback( Output('SC3D-y-dropdown','value'),
               State('SC3D-y-dropdown','value'),
               State('SC3D-y-dropdown','options'),
               Input('SC3D-y-previous-button','n_clicks'),
               Input('SC3D-y-next-button','n_clicks'))
def update_dopdown(value,options,previous,next_element):
    if value:
        index=options.index(value)
        if ctx.triggered_id=='SC3D-y-next-button':
            index=index+1
            if index>=len(options):
                index=0
            return options[index]
        if ctx.triggered_id=='SC3D-y-previous-button':
            index=index-1
            if index<0:
                index=len(options)-1
            return options[index]

@app.callback( Output('SC3D-z-dropdown','value'),
               State('SC3D-z-dropdown','value'),
               State('SC3D-z-dropdown','options'),
               Input('SC3D-z-previous-button','n_clicks'),
               Input('SC3D-z-next-button','n_clicks'))
def update_dopdown(value,options,previous,next_element):
    if value:
        index=options.index(value)
        if ctx.triggered_id=='SC3D-z-next-button':
            index=index+1
            if index>=len(options):
                index=0
            return options[index]
        if ctx.triggered_id=='SC3D-z-previous-button':
            index=index-1
            if index<0:
                index=len(options)-1
            return options[index]

@app.callback(
    Output('Ridge-Loading','children'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('Ridge-space','value'),
    Input('Ridge-name','value'),
    Input('Save_Path','value'),
    Input('Ridge-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_Ridge_graph(data,rows,derived_virtual_selected_rows,spacing,title,save_path,save):
    df=pd.DataFrame.from_records(data)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows=[]
    dff=df if rows is None else pd.DataFrame(rows)
    #maybe use https://github.com/tpvasconcelos/ridgeplot#
    dff=dff.select_dtypes(include=np.number)
    n_colors=len(dff.columns)
    if not spacing:
        spacing =1
    color_values=plcolor.sample_colorscale(discrete_color_scale,[n/(n_colors -1) for n in range(n_colors)])
    fig=RidgePlotFigureFactory_Custom(samples=dff.values.T,spacing=spacing,labels=dff.columns,linewidth= 1.1,colors=color_values).make_figure()
    if not title:
        title='Ridgeplot'
    fig.update_layout(title=title)
    if ctx.triggered_id=='Ridge-save-plot':
        save_plot(fig,name=f'{title}.html',save_path=save_path)
    return [dbc.Modal(id='Ridge-Modal',children=[
                        dbc.ModalHeader(dbc.ModalTitle('Ridgeplot')),
                        dbc.ModalBody(dcc.Graph(id='Ridge-Graph',figure=fig,style={'height':'100%','width':'100%'})),
                    ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='Ridge-Graph',figure=fig)]  

@app.callback(Output('Ridge-Modal','is_open'),
              Input('Ridge-popup','n_clicks'),
              prevent_initial_call=True,)
def open_modal(popup):
    if ctx.triggered_id=='Ridge-popup':
        return True

@app.callback(
    Output('Corr-Loading','children'),
    Input('Corr-scope','value'),
    Input('Corr-columns','value'),
    State('data_table','data'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    Input('Corr-type-dropdown','value'),
    Input('Corr-name','value'),
    Input('Save_Path','value'),
    Input('Corr-save-plot','n_clicks'),prevent_initial_call=True,
)
def update_Corr_graph(scope,colum,data,rows,derived_virtual_selected_rows,corr_type,title,save_path,save):
    df=pd.DataFrame.from_records(data)
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows=[]
    dff=df if rows is None else pd.DataFrame(rows) 
    if corr_type:
        if corr_type=='Power Predictive Score':
            if scope=='Over all':
                matrix_df = pps.matrix(dff,random_seed=42,)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
                matrix_df.columns=dff.columns
                matrix_df.index=dff.columns
                fig=px.imshow(matrix_df,text_auto=True,template=figure_template,color_continuous_scale=color_scale,labels={'x':'','y':''})
                if not title:
                    scope_=f'_scope' if scope else ''
                    title=f'{corr_type}{scope_}'
                fig.update_layout(title=title)
                if ctx.triggered_id=='Corr-save-plot':
                    save_plot(fig,name=f'{title}.html',save_path=save_path)
            else:
                if colum:
                    matrix_df = pps.predictors(dff,colum,random_seed=42,)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
                    matrix_df.columns=dff.drop(columns=[colum]).columns
                    #matrix_df.index=dff.columns
                    fig=px.imshow(matrix_df,text_auto=True,template=figure_template,color_continuous_scale=color_scale,labels={'x':'','y':''})
                    if not title:
                        scope_=f'_scope' if scope else ''
                        title=f'{corr_type}{scope_}'
                    fig.update_layout(title=title)
                    if ctx.triggered_id=='Corr-save-plot':
                        save_plot(fig,name=f'{title}.html',save_path=save_path)
                else: return  [dbc.Modal(id='Corr-Modal',children=[
                    dbc.ModalHeader(dbc.ModalTitle('Feature dependencies')),
                    dbc.ModalBody(dcc.Graph(id='Corr-Graph',figure={})),
                ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='Corr-Graph',figure={}),html.H5(children=['Please provide a target column'],style={'color':f'{colors["Error"]}'})]
        else:
            cor=dff.corr(corr_type)
            if scope=='Over all':
                fig=px.imshow(abs(cor),text_auto=True,template=figure_template,color_continuous_scale=color_scale)
                if title:
                    fig.update_layout(title=title)
                if ctx.triggered_id=='Corr-save-plot':
                    save_plot(fig,name=f'{title}.html',save_path=save_path)
            else:
                if colum:
                    fig=px.imshow(abs(cor[[colum]].transpose()),template=figure_template,color_continuous_scale=color_scale,text_auto=True)
                    if not title:
                        scope_=f'_scope' if scope else ''
                        title=f'{corr_type}{scope_}'
                    fig.update_layout(title=title)
                    if ctx.triggered_id=='Corr-save-plot':
                        save_plot(fig,name=f'{title}.html',save_path=save_path)  
                else: return  [dbc.Modal(id='Corr-Modal',children=[
                    dbc.ModalHeader(dbc.ModalTitle('Feature dependencies')),
                    dbc.ModalBody(dcc.Graph(id='Corr-Graph',figure={})),
                ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='Corr-Graph',figure={}),html.H5(children=['Please provide a target column'],style={'color':f'{colors["Error"]}'})]
        return [dbc.Modal(id='Corr-Modal',children=[
                        dbc.ModalHeader(dbc.ModalTitle('Feature dependencies')),
                        dbc.ModalBody(dcc.Graph(id='Corr-Graph',figure=fig,style={'height':'100%','width':'100%'})),
                    ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='Corr-Graph',figure=fig)]  
    else: return [dbc.Modal(id='Corr-Modal',children=[
                    dbc.ModalHeader(dbc.ModalTitle('Feature dependencies')),
                    dbc.ModalBody(dcc.Graph(id='Corr-Graph',figure={})),
                ], is_open=False,fullscreen=True,scrollable=True),dcc.Graph(id='Corr-Graph',figure={}),html.H5(children=['The plot will be displayed here. To display the plot you must first choose any settings'],style={'color':f'{colors["Info"]}'})]

@app.callback(Output('Corr-Modal','is_open'),
              Input('Corr-popup','n_clicks'),
              prevent_initial_call=True,)
def open_modal(popup):
    if ctx.triggered_id=='Corr-popup':
        return True    

@app.callback(
        Output('Export-div','children'),
    Input('Export Data','n_clicks'),
    State('Export-name','value'),
    Input('data_table','derived_virtual_data'),
    Input('data_table','derived_virtual_selected_rows'),
    State('data_table','data'),
    State('Save_Path','value'),
)
def export_data(export,name,rows,derived_virtual_selected_rows,data,save_path):
    if ctx.triggered_id=='Export Data':
        if name:
            df=pd.DataFrame.from_records(data)
            if derived_virtual_selected_rows is None:
                derived_virtual_selected_rows=[]
            dff=df if rows is None else pd.DataFrame(rows)
            filename,ext=os.path.splitext(name)
            if save_path:
                path=os.path.join(save_path,name)
            else:
                path=name
            if ext:
                try:
                    if ext=='.csv':
                        dff.to_csv(path)
                    elif ext=='.parquet':
                        dff.to_parquet(path)
                    elif ext=='.xlsx':
                        dff.to_excel(path)
                    return html.H6(children=f'The Data was exportet to {ext} sucessfully! You can find the export under "{path}"',style={'color':f'{colors["Sucess"]}'})
                except:
                    return html.H6(children=f'The Export failed! Please provide a file name with the targeted extension. (Supportet are: *.xlsx,*.parquet,*.csv) ',style={'color':f'{colors["Error"]}'})
            else: html.H6(children=f'Please provide a file name with the targeted extension. (Supportet are: *.xlsx,*.parquet,*.csv)',style={'color':f'{colors["Error"]}'})
        else:
            return html.H6(children=f'Please provide a file name with the targeted extension. (Supportet are: *.xlsx,*.parquet,*.csv)',style={'color':f'{colors["Error"]}'})

if __name__ == "__main__":
    app.title="Nsight"
    print('running')
    serve(app.server, host="127.0.0.1", port=8050)
    