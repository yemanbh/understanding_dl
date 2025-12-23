import torch
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as ex
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

class View3D:
    """
    Utility class for visualizing 3D geometric objects using Matplotlib.

    This class provides convenience wrappers around common 3D plotting
    primitives such as:
        - surfaces
        - scatter points
        - vector fields (quiver arrows)

    All data is assumed to be in the form of tensors with shape:
        (3, N)        for point clouds / vectors
        (3, H, W)     for surfaces (meshgrids)

    Coordinate convention:
        xyz[0] → x-coordinates
        xyz[1] → y-coordinates
        xyz[2] → z-coordinates
    """

    def __init__(self, xyz, figsize=(10., 10.)):
        """
        Initialize a 3D figure and axes.

        Parameters
        ----------
        xyz : torch.Tensor or np.ndarray
            Tensor containing 3D data, typically of shape (3, H, W)
            for surfaces or (3, N) for point clouds.

        figsize : tuple of float, optional
            Size of the Matplotlib figure in inches.
        """
        self.xyz = xyz

        # Create figure and 3D axis
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

    def surface(self, label='surf'):
        """
        Plot a 3D surface.

        Assumes self.xyz has shape (3, H, W) corresponding to a meshgrid.
        """
        surf = self.ax.plot_surface(
            self.xyz[0, :, :],   # X coordinates
            self.xyz[1, :, :],   # Y coordinates
            self.xyz[2, :, :],   # Z values
            cmap="viridis",
            alpha=0.8,
            linewidth=0
        )

        if label is not None:
            surf._legend_label = label
            self._has_label = True

    def contour(self):
        """
        Plot a 3D contours.

        Assumes self.xyz has shape (3, H, W) corresponding to a meshgrid.
        """
        self.ax.contour(
            self.xyz[0, :, :],   # X coordinates
            self.xyz[1, :, :],   # Y coordinates
            self.xyz[2, :, :],   # Z values
            cmap="viridis",
            zdir='z',
            offset=self.xyz[2, :, :].min(),
            alpha=0.8,
            levels=50
        )
    def quiver(self, xyz: torch.Tensor, uvw: torch.Tensor, normalize: bool = False, color: str = 'c'):
        """
        Plot 3D vectors (quiver plot).

        Parameters
        ----------
        xyz : torch.Tensor
            Starting points of vectors, shape (3, N).

        uvw : torch.Tensor
            Vector directions / magnitudes, shape (3, N).

        normalize : bool, optional
            If True, all vectors are normalized to unit length.
            Direction is preserved, magnitude is discarded.
        """
        self.ax.quiver(
            xyz[0, :],  # vector origins (x)
            xyz[1, :],  # vector origins (y)
            xyz[2, :],  # vector origins (z)
            uvw[0, :],  # x-component
            uvw[1, :],  # y-component
            uvw[2, :],  # z-component
            normalize=normalize,
            linewidth=1,
            arrow_length_ratio=0.4,
            color=color,
        )

    def scatter(self, xyz: torch.Tensor, z_as_color: bool = True):
        """
        Plot a 3D scatter plot.

        Parameters
        ----------
        xyz : torch.Tensor
            Point coordinates, shape (3, N).

        z_as_color : bool, optional
            If True, use z-values to color points.
        """
        self.ax.scatter(
            xyz[0, :],  # x positions
            xyz[1, :],  # y positions
            xyz[2, :],  # z positions
            c=xyz[2, :] if z_as_color else None,
            s=10,
            alpha=0.8,
            cmap='magma',
            label='scatter'
        )

    def quiver_x_component(self, xyz: torch.Tensor, uvw: torch.Tensor, normalize: bool = False, color: str = 'b'):
        """
        Draw vectors showing only the x-component of a vector field.

        Useful for visualizing directional decomposition.

        Parameters
        ----------
        xyz : torch.Tensor
            Vector origins, shape (3, N).

        uvw : torch.Tensor
            Full vector field, shape (3, N).

        normalize : bool, optional
            Normalize vector lengths.
        """
        self.ax.quiver(
            xyz[0, :],
            xyz[1, :],
            xyz[2, :],
            uvw[0, :],                       # x-component only
            torch.zeros_like(uvw[1, :]),
            torch.zeros_like(uvw[2, :]),
            color=color,
            normalize=normalize
        )

    def quiver_y_component(self, xyz: torch.Tensor, uvw: torch.Tensor, normalize: bool = False, color: str = 'g'):
        """
        Draw vectors showing only the y-component of a vector field.
        """
        self.ax.quiver(
            xyz[0, :],
            xyz[1, :],
            xyz[2, :],
            torch.zeros_like(uvw[0, :]),
            uvw[1, :],                       # y-component only
            torch.zeros_like(uvw[2, :]),
            color=color,
            normalize=normalize
        )

    def quiver_z_component(self, xyz: torch.Tensor, uvw: torch.Tensor, normalize: bool = False, color: str = 'b'):
        """
        Draw vectors showing only the z-component of a vector field.
        """
        self.ax.quiver(
            xyz[0, :],
            xyz[1, :],
            xyz[2, :],
            torch.zeros_like(uvw[0, :]),
            torch.zeros_like(uvw[1, :]),
            uvw[2, :],                       # z-component only
            color=color,
            normalize=normalize
        )

    def set_xlabel(self, label: str):
        """Set x-axis label."""
        self.ax.set_xlabel(label)

    def set_ylabel(self, label: str):
        """Set y-axis label."""
        self.ax.set_ylabel(label)

    def set_zlabel(self, label: str):
        """Set z-axis label."""
        self.ax.set_zlabel(label)

    def set_xlim(self, min_: float, max_: float):
        """Set x-axis limits."""
        self.ax.set_xlim(min_, max_)

    def set_ylim(self, min_: float, max_: float):
        """Set y-axis limits."""
        self.ax.set_ylim(min_, max_)

    def set_zlim(self, min_: float, max_: float):
        """Set z-axis limits."""
        self.ax.set_zlim(min_, max_)
    
    def legend(self, **kwargs):
        """ Show legend if labeled componets exist
        """

        self.ax.legend(**kwargs)

    def show(self):
        """Render the figure."""
        self.legend()
        
        plt.show()

class PlotlibViewer:
    def __init__(self, 
                 mode, 
                 width=800, 
                 height=600,
                 renderer=None):
        
        self.mode = mode
        self.width = width
        self.height = height
        self.renderer = renderer

        # create figure
        self.fig = go.Figure()

    
    def surface(self, x, y, z, cmap="Viridis", opacity=0.9):
        assert self.mode == '3d', f"Specified mode={self.mode}, but ploting 3D surface"
        self.fig.add_trace(
            go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=cmap,
            opacity=opacity
            )
        )
    
    def scatter3d(self, x, y, z, color=None, cmap="Reds", name="scatter 3d", s=5):
        assert self.mode == '3d', f"Specified mode={self.mode}, but ploting 3D scatter plot"
        self.fig.add_trace(
            go.Scatter3d(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                mode='markers', # markers only
                marker=dict(
                    size=5,
                    color= color, # z.flatten(),          # color by z value
                    colorscale=cmap,
                    opacity=0.5
                ),
                name=name
            )
        )
    
    def contour(self, x, y, z):
        
        assert self.mode == '2d', f"Specified mode={self.mode}, but ploting 2D contour"
        self.fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=z,
                colorscale="Viridis",
                contours=dict(showlabels=True)
            )
        )

    def quiver3d(self, x, y, z, u, v, w):

        x1 = x + u
        y1 = y + v
        z1 = z + w

        # fig = go.Figure()

        # 1️⃣ Vector shafts (lines)
        for i in range(len(x)):
            self.fig.add_trace(
                go.Scatter3d(
                    x=[x[i], x1[i]],
                    y=[y[i], y1[i]],
                    z=[z[i], z1[i]],
                    mode="lines",
                    line=dict(width=2),
                    showlegend=False
                )
            )

            # 2️Arrowheads (cones)
            self.fig.add_trace(
                go.Cone(
                    x=x1,
                    y=y1,
                    z=z1,
                    u=u,
                    v=v,
                    w=w,
                    anchor="tail",
                    sizemode="absolute",
                    sizeref=0.25,
                    showscale=False
                )
            )

    def quiver2d(self, x, y, u, v, name='grad_dir'):
        # Create quiver
        quiver_fig = ff.create_quiver(x, y, u, v, arrow_scale=0.1, name=name)

        # Create a new figure and add the quiver trace

        for trace in quiver_fig.data:
            self.fig.add_trace(trace)

    def scatter2d(self, x, y, name="scatter 2d", size=5, color=None, symbol="circle", cmap='Reds'):
        assert self.mode == '2d', f"Specified mode={self.mode}, but ploting 2D scatter plot"
        # scatter points ON TOP
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=size,
                    colorscale=cmap,
                    color=color,
                    symbol=symbol,
                    line=dict(width=0, color="black")
                ),
                name=name
            )
        )
    
    def update_layout_2d(self, xlable, ylabel, title):

        self.fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),                
                    xaxis_title=xlable,
                    yaxis_title=ylabel,
                    title=title,
                    height=self.height,
                    width=self.width
                )
        
    def update_layout_3d(self, xlable, ylabel,zlabel, title):

        self.fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),   
                    scene=dict(             
                    xaxis_title=xlable,
                    yaxis_title=ylabel,
                    zaxis_title=zlabel,
                    ),
                    title=title,
                    height=self.height,
                    width=self.width
                )
        
        
    def show(self, xlable='x', ylabel='y', zlabel='z', title=None):

        # update layout
        if self.mode == "2d":
            self.update_layout_2d(xlable, ylabel, title)
        else:
            self.update_layout_3d(xlable, ylabel, zlabel, title)
        
        print(f'renderer is {self.renderer}')
        self.fig.show(renderer=self.renderer)
    
    def save(self, file_path: Path):
        if not file_path.is_file:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        ext = file_path.suffix
        if ext not in ['.png', '.html']:
            raise ValueError('file name should have html or png extension')
        if ext == '.html':
            self.fig.write_html(file_path)
        else:
            self.fig.write_image(file_path)

        