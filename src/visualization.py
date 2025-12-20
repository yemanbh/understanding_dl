import torch
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go

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