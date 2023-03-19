from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from plotly import graph_objects as go

from ridgeplot._colors import (
    apply_alpha,
    get_color,
    get_colorscale,
    validate_colorscale,
)
from ridgeplot._types import ColorScaleType, NestedNumericSequence
from ridgeplot._utils import get_xy_extrema, normalise_min_max
from .styling import style_app
from ridgeplot._kde import get_densities

external_style,colors,min_style,discrete_color_scale,color_scale,figure_template=style_app()

class RidgePlotFigureFactory_Custom():
    """Refer to :func:`~ridgeplot.ridgeplot()`."""

    def __init__(
        self,
        colors,
        samples=None,
        densities: Optional[Iterable[NestedNumericSequence]] = None,
        kernel: str = "gau",
        bandwidth="normal_reference",
        kde_points=500,
        colormode: str = "mean-means",
        coloralpha: Optional[float] = None,
        labels=None,
        linewidth: float = 1.4,
        spacing: float = 0.5,
        show_annotations: bool = True,
        xpad: float = 0.05,
    ) -> None:
        # ==============================================================
        # ---  Get clean and validated input arguments
        # ==============================================================
        has_samples = samples is not None
        has_densities = densities is not None
        if has_samples and has_densities:
            raise ValueError("You may not specify both `samples` and `densities` arguments!")
        elif not has_samples and not has_densities:
            raise ValueError("You have to specify one of: `samples` or `densities`")
        elif not has_densities:
            densities = get_densities(samples, points=kde_points, kernel=kernel, bandwidth=bandwidth)
        # Check whether all density arrays have shape (2, N)
        new_densities: List[np.ndarray] = []
        for array in densities:
            array = np.asarray(array)
            if array.ndim != 2 or array.shape[0] != 2:
                raise ValueError(
                    "Each density array must have shape (2, N), "
                    f"but got array with shape {array.shape}"
                )
            new_densities.append(array)

        n_traces = len(new_densities)

        if colormode not in self.colormode_maps.keys():
            raise ValueError(
                f"The colormode argument should be one of "
                f"{tuple(self.colormode_maps.keys())}, got {colormode} instead."
            )

        if coloralpha is not None:
            coloralpha = float(coloralpha)

        if labels is not None:
            n_labels = len(labels)
            if n_labels != n_traces:
                raise ValueError(f"Expected {n_traces} labels, got {n_labels}.")
            labels = list(map(str, labels))
        else:
            labels = [f"Trace {i + 1}" for i in range(n_traces)]

        self.densities: List[np.ndarray] = new_densities
        self.coloralpha: Optional[float] = coloralpha
        self.colormode = str(colormode)
        self.labels: list = labels
        self.linewidth: float = float(linewidth)
        self.spacing: float = float(spacing)
        self.show_annotations: bool = bool(show_annotations)
        self.xpad: float = float(xpad)

        # ==============================================================
        # ---  Other instance variables
        # ==============================================================
        self.n_traces: int = n_traces
        self.x_min, self.x_max, _, self.y_max = get_xy_extrema(arrays=self.densities)
        self.fig: go.Figure = go.Figure()
        self.colors=colors

    @property
    def colormode_maps(self) -> Dict[str, Callable[[], List[float]]]:
        return {
            "index": self._compute_midpoints_index,
            "mean-minmax": self._compute_midpoints_mean_minmax,
            "mean-means": self._compute_midpoints_mean_means,
        }

    def draw_base(self, x, y_shifted) -> None:
        """Draw the base for a density trace.
        Adds an invisible trace at constant y that will serve as the fill-limit
        for the corresponding density trace.
        """
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=[y_shifted] * len(x),
                # make trace 'invisible'
                # Note: visible=False does not work with fill="tonexty"
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False,
            )
        )

    def draw_density_trace(self, x, y, label, color) -> None:
        """Draw a density trace.
        Adds a density 'trace' to the Figure. The ``fill="tonexty"`` option
        fills the trace until the previously drawn trace (see
        :meth:`draw_base`). This is why the base trace must be drawn first.
        """
        line_color = "rgba(0,0,0,0.6)" if color is not None else None
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fillcolor=color,
                name=label,
                fill="tonexty",
                mode="lines",
                line=dict(color=line_color, width=self.linewidth),
            ),
        )

    def update_layout(self, y_ticks: list) -> None:
        """Update figure's layout."""
        self.fig.update_layout(
            hovermode=False,
            legend=dict(traceorder="normal"),
        )
        axes_common = dict(
            zeroline=False,
            showgrid=True,
        )
        self.fig.update_yaxes(
            showticklabels=self.show_annotations,
            tickvals=y_ticks,
            ticktext=self.labels,
            **axes_common,
        )
        x_padding = self.xpad * (self.x_max - self.x_min)
        self.fig.update_xaxes(
            range=[self.x_min - x_padding, self.x_max + x_padding],
            showticklabels=True,
            **axes_common,
        )

    def _compute_midpoints_index(self) -> List[float]:
        return [i / (self.n_traces - 1) for i in reversed(range(self.n_traces))]

    def _compute_midpoints_mean_minmax(self) -> List[float]:
        means = [np.sum(x * y) / np.sum(y) for x, y in self.densities]
        return [normalise_min_max(mean, min_=self.x_min, max_=self.x_max) for mean in means]

    def _compute_midpoints_mean_means(self) -> List[float]:
        means = [np.sum(x * y) / np.sum(y) for x, y in self.densities]
        return [normalise_min_max(mean, min_=min(means), max_=max(means)) for mean in means]

    # def pre_compute_colors(self) -> List[str]:
    #     midpoints = self.colormode_maps[self.colormode]()
    #     colors = []
    #     for midpoint in midpoints:
    #         color = get_color(self.colorscale, midpoint=midpoint)
    #         if self.coloralpha is not None:
    #             color = apply_alpha(color, alpha=self.coloralpha)
    #         colors.append(color)
    #     return colors

    def make_figure(self) -> go.Figure:
        y_ticks = []
        for i, ((x, y), label, color) in enumerate(zip(self.densities, self.labels, self.colors)):
            # y_shifted is the y-origin for the new trace
            y_shifted = -i * (self.y_max * self.spacing)
            self.draw_base(x=x, y_shifted=y_shifted)
            self.draw_density_trace(x=x, y=y + y_shifted, label=label, color=color)
            y_ticks.append(y_shifted)
        self.update_layout(y_ticks=y_ticks)
        return self.fig
    
import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc


# class GraphPopout(ExplainerComponent):
#     """Provides a way to open a modal popup with the content of a graph figure."""

#     def __init__(
#         self,
#         name: str,
#         graph_id: str,
#         title: str = "Popout",
#         description: str = None,
#         button_text: str = "Popout",
#         button_size: str = "sm",
#         button_outline: bool = True,
#     ):
#         """
#         Args:
#             name (str): name id for this GraphPopout. Should be unique.
#             graph_id (str): id of of the dcc.Graph component that gets included
#                 in the modal.
#             title (str): Title above the modal. Defaults to Popout.
#             description (str): description of the graph to be include in the footer.
#             button_text (str, optional): Text on the Button. Defaults to "Popout".
#             button_size (str, optiona). Size of the button.Defaults to "sm" or small.
#             button_outline (bool, optional). Show outline of button instead with fill color.
#                 Defaults to True.
#         """

#         self.title = title
#         self.name = name
#         self.graph_id = graph_id
#         self.description = description
#         self.button_text, self.button_size, self.button_outline = (
#             button_text,
#             button_size,
#             button_outline,
#         )

#     def layout(self):
#         return html.Div(
#             [
#                 dbc.Button(
#                     self.button_text,
#                     id=self.name + "modal-open",
#                     size=self.button_size,
#                     color="secondary",
#                     outline=self.button_outline,
#                 ),
#                 dbc.Modal(
#                     [
#                         # ToDo the X on the top right is not rendered properly, disabling
#                         dbc.ModalHeader(dbc.ModalTitle(self.title), close_button=True),
#                         dbc.ModalBody(
#                             dcc.Graph(
#                                 id=self.name + "-modal-graph",
#                                 style={"max-height": "none", "height": "80%"},
#                             )
#                         ),
#                         dbc.ModalFooter(
#                             [
#                                 html.Div(
#                                     [
#                                         html.Div(
#                                             [
#                                                 html.Div(
#                                                     [
#                                                         dbc.Button(
#                                                             html.Small("Description"),
#                                                             id=self.name
#                                                             + "-show-description",
#                                                             color="link",
#                                                             className="text-muted ml-auto",
#                                                         ),
#                                                         dbc.Fade(
#                                                             [
#                                                                 html.Small(
#                                                                     self.description,
#                                                                     className="text-muted",
#                                                                 )
#                                                             ],
#                                                             id=self.name + "-fade",
#                                                             is_in=True,
#                                                             appear=True,
#                                                         ),
#                                                     ],
#                                                     style=dict(
#                                                         display="none"
#                                                         if not self.description
#                                                         else None
#                                                     ),
#                                                 )
#                                             ],
#                                             className="text-left",
#                                         ),
#                                         html.Div(
#                                             [
#                                                 dbc.Button(
#                                                     "Close",
#                                                     id=self.name + "-modal-close",
#                                                     className="mr-auto",
#                                                 )
#                                             ],
#                                             className="text-right",
#                                             style=dict(float="right"),
#                                         ),
#                                     ],
#                                     style={"display": "flex"},
#                                 ),
#                             ],
#                             className="justify-content-between",
#                         ),
#                     ],
#                     id=self.name + "-modal",
#                     size="xl",
#                 ),
#             ],
#             style={"display": "flex", "justify-content": "flex-end"},
#         )

#     def component_callbacks(self, app):
#         @app.callback(
#             [
#                 Output(self.name + "-modal", "is_open"),
#                 Output(self.name + "-modal-graph", "figure"),
#             ],
#             [
#                 Input(self.name + "modal-open", "n_clicks"),
#                 Input(self.name + "-modal-close", "n_clicks"),
#             ],
#             [State(self.name + "-modal", "is_open"), State(self.graph_id, "figure")],
#         )
#         def toggle_modal(open_modal, close_modal, modal_is_open, fig):
#             if open_modal or close_modal:
#                 ctx = dash.callback_context
#                 button_id = ctx.triggered[0]["prop_id"].split(".")[0]
#                 if button_id == self.name + "modal-open":
#                     return (not modal_is_open, fig)
#                 else:
#                     return (not modal_is_open, dash.no_update)
#             return (modal_is_open, dash.no_update)

#         if self.description is not None:

#             @app.callback(
#                 Output(self.name + "-fade", "is_in"),
#                 [Input(self.name + "-show-description", "n_clicks")],
#                 [State(self.name + "-fade", "is_in")],
#             )
#             def toggle_fade(n_clicks, is_in):
#                 if not n_clicks:
#                     # Button has never been clicked
#                     return False
#                 return not is_in
