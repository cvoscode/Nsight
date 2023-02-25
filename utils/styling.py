import dash_bootstrap_components as dbc 
from dash_bootstrap_templates import load_figure_template



def style_app():
    """styling of the app
    """
    external_style=dbc.themes.SKETCHY
    figure_template=load_figure_template("sketchy")
    colors={'Sucess':'Green','Error':'Red'}
    min_style={'margin':'5px'}
    return external_style, figure_template,colors,min_style