import dash_bootstrap_components as dbc 
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.templates["draft"] = go.layout.Template(
    layout_annotations=[
        dict(
            name="draft watermark",
            #Here you can input a text for a watermark
            text="Leck Mich es is spät",
            textangle=-30,
            opacity=0.1,
            font=dict(color="black", size=100),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    ],
)


def style_app():
    """styling of the app
    """
    external_style=dbc.themes.SKETCHY
    colors={'Sucess':'Green','Error':'Red'}
    color_scale=['rgb(0, 0, 0)','rgb(0, 255, 255)']
    min_style={'margin':'2px'}
    return external_style,colors,min_style,color_scale

