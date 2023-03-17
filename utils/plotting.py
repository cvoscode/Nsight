import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go
from typing import Tuple




import numpy as np
import statsmodels.api as sm


def evaluate_density(samples, points, kernel, bandwidth) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a density function at a set of points.
    For a given set of samples, computes the kernel densities at the given
    points. Both the original points and density arrays are returned.
    """
    # By default, we'll use a 'hard' KDE span. That is, we'll
    # evaluate the densities and N equally spaced points
    # over the range [min(samples), max(samples)]
    if isinstance(points, int):
        points = np.linspace(np.min(samples), np.max(samples), points)

    # Unless a specific range is specified...
    else:
        points = np.asarray(points)

    if points.ndim > 1:
        raise ValueError(
            f"The 'points' at which KDE is computed should be represented by a "
            f"one-dimensional array, got an array of shape {points.shape} instead."
        )

    # I decided to use statsmodels' KDEUnivariate for KDE. There are many
    # other supported alternatives in the python scientific computing
    # ecosystem. See, for instance, scipy's alternative - on which
    # statsmodels relies - `from scipy.stats import gaussian_kde`
    dens = sm.nonparametric.KDEUnivariate(samples)

    # I'm hard-coding the `fft=self.kernel == "gau"` for convenience here.
    # This avoids the need to expose yet another __init__ argument (fft)
    # to this class. The drawback is that, if and when statsmodels
    # implements another kernel with fft, this will fall back to
    # using the unoptimised version (with fft = False).
    dens.fit(kernel=kernel, fft=kernel == "gau", bw=bandwidth)
    densities = dens.evaluate(points)

    # I haven't investigated the root of this issue yet
    # but statsmodels' KDEUnivariate implementation
    # can return a nan float if something goes
    # wrong internally. As to avoid confusion
    # further down the pipeline, I decided
    # to check whether the correct object
    # (and shape) are being returned.
    if not isinstance(densities, np.ndarray) or densities.shape != points.shape:
        raise RuntimeError(
            f"Could now evaluate densities using the {kernel!r} kernel! "
            f"Try using kernel='gau' (default)."
        )

    return points, densities


def get_densities(samples, points, kernel, bandwidth) -> np.ndarray:
    return np.asarray(
        [
            evaluate_density(
                samples=s,
                points=points,
                kernel=kernel,
                bandwidth=bandwidth,
            )
            for s in samples
        ]
    )








def draw_base(fig,x, y_shifted) -> None:
        """Draw the base for a density trace.
        Adds an invisible trace at constant y that will serve as the fill-limit
        for the corresponding density trace.
        """
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[y_shifted] * len(x),
                # make trace 'invisible'
                # Note: visible=False does not work with fill="tonexty"
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False,
            )
        )

def draw_density_trace(fig, x, y, label, color,linewidth) -> None:
        """Draw a density trace.
        Adds a density 'trace' to the Figure. The ``fill="tonexty"`` option
        fills the trace until the previously drawn trace (see
        :meth:`draw_base`). This is why the base trace must be drawn first.
        """
        line_color = "rgba(0,0,0,0.6)" if color is not None else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fillcolor=color,
                name=label,
                fill="tonexty",
                mode="lines",
                line=dict(color=line_color, width=linewidth),
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

def make_figure(labels,colors,samples, points, kernel, bandwidth) -> go.Figure:
        y_ticks = []
        densities=get_densities()
        for i, ((x, y), label, color) in enumerate(zip(densities, labels, colors)):
            # y_shifted is the y-origin for the new trace
            y_shifted = -i * (self.y_max * self.spacing)
            draw_base(x=x, y_shifted=y_shifted)
            draw_density_trace(x=x, y=y + y_shifted, label=label, color=color)
            y_ticks.append(y_shifted)
        fig.update_layout(y_ticks=y_ticks)
        return fig