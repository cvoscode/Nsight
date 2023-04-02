import dash_bootstrap_components as dbc 
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
#Make the water marks for the graphs
pio.templates["watermark"] = go.layout.Template(
    layout_annotations=[
        dict(
            name="watermark",
            #Here you can input a text for a watermark
            text="Nsight",
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
    """Here you can style the app
    With the external Style sheets you can select dbc Themes.
    In colors you can Change the colors used for the Texts.
    color_scale is a continouse colorscale, which you can choose by putting in rgb values or plotly.colorscales
    dicrete_color_scale is a continouse colorscale provided by plotly which is discretesized depending on the values it shoul have
    min_style is a CSS dict of the minimum Style.

    Returns:
       external_style,colors,min_style,discrete_color_scale,color_scale
    """
    external_style=dbc.themes.SKETCHY
    figure_template="sketchy"
    colors={'Sucess':'Green','Error':'Red','Info':'Grey','Selected':'black','Selected_text':'white'}
    color_scale=['rgb(0, 0, 0)','rgb(0, 255, 255)','rgb(255,255,0)']
    discrete_color_scale='rainbow'
    min_style={'margin':'2px'}
    return external_style,colors,min_style,discrete_color_scale,color_scale,figure_template

