import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def synthesize_view(adata, angle_x=0, angle_y=0, angle_z=0, color_key='leiden', palette=None, slice_offset=0, slice_thickness=10, volume_visualization_sample_size=100000, save_plot=False, save_path='3d_plot.html'):
    """
    The synthesize_view function is designed to visualize three-dimensional (3D) data from the SpatialZ-reconstructed dense atlas, 

    Input Parameters:

        adata (AnnData): The AnnData object (the SpatialZ-reconstructed dense atlas).
        angle_x, angle_y, angle_z (float): Angles to rotate the data around the x, y, and z axes, respectively. 
        color_key (str): The key (Annotation information) in adata.obs used for coloring the data points in the plot. 
                         Typically, this could be a clustering label or any categorical annotation.
        palette (dict or None): A dictionary mapping categorical labels to colors. If None, a default colormap (viridis) is used.
        slice_offset (float): The offset along the z-axis to position the slice plane within the volume. 
                              The offset is initial located from the mid-point of the z-axis range.
        slice_thickness (float): The thickness of the slice.
        volume_visualization_sample_size (int): The maximum number of points to include in the 3D scatter plot.
        save_plot (bool): If True, the generated plot will be saved as an HTML file.
        save_path (str): The path where the plot HTML file should be saved if save_plot is True
    
    Returns:
    
    An AnnData object (A synthesize view). 
    
    """
    # Retrieve coordinates and color information
    coords = adata.obs[['x', 'y', 'z']].values
    colors = adata.obs[color_key].cat.codes.values

    # Remove rows with NaN values in coordinates if any are detected
    if np.isnan(coords).any():
        print("NaN values detected in coordinates. Removing rows with NaN values.")
        valid_mask = ~np.isnan(coords).any(axis=1)
        coords = coords[valid_mask]
        colors = colors[valid_mask]

    # Calculate the center of the data
    center = np.mean(coords, axis=0)

    # Create rotation matrices
    theta_x = np.radians(-angle_x)
    theta_y = np.radians(angle_y)
    theta_z = np.radians(angle_z)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx  # Combine rotation matrices

    # Compute the data range after rotation
    rotated_coords = (coords - center) @ R.T
    
    # Determine the coordinate ranges after rotation
    x_range = [rotated_coords[:, 0].min(), rotated_coords[:, 0].max()]
    y_range = [rotated_coords[:, 1].min(), rotated_coords[:, 1].max()]
    z_range = [rotated_coords[:, 2].min(), rotated_coords[:, 2].max()]

    # Determine the position of the slice
    slice_position = (z_range[0] + z_range[1]) / 2 + slice_offset

    # Obtain slice data
    slice_mask = np.abs(rotated_coords[:, 2] - slice_position) <= slice_thickness / 2
    slice_coords = rotated_coords[slice_mask]
    slice_colors = colors[slice_mask]

    # Create a new AnnData object to store slice data
    slice_adata = sc.AnnData(adata.X[slice_mask])
    slice_adata.obs = adata.obs.iloc[slice_mask]
    slice_adata.uns = adata.uns
    slice_adata.obsm = {key: val[slice_mask] for key, val in adata.obsm.items()}
    slice_adata.var = adata.var
    slice_adata.obs['x_slice'] = slice_coords[:, 0]
    slice_adata.obs['y_slice'] = slice_coords[:, 1]

    # Create a new numpy array containing x_slice and y_slice values
    spatial_slice = np.vstack((slice_adata.obs['x_slice'], slice_adata.obs['y_slice'])).T

    # Add this new array to obsm
    slice_adata.obsm['spatial_slice'] = spatial_slice

    # Calculate total cell count
    total_cells = slice_adata.n_obs

    # Calculate 10% of the total cells
    sample_size = int(0.13 * total_cells)

    # Randomly select indices for sampling
    np.random.seed(6666)  
    random_indices = np.random.choice(total_cells, size=sample_size, replace=False)

    # Create a new AnnData object from randomly selected indices
    adata_new = slice_adata[random_indices, :].copy()

    adata_new = adata_new[~adata_new.obs[color_key].isnull()]

    # Visualization
    fig, ax = plt.subplots(figsize=(18, 15))  
    palette = palette if palette is not None else 'viridis'
    ax = sc.pl.embedding(adata_new, basis='spatial_slice', color=color_key, size=20, show=False,
                         palette=palette, ax=ax, frameon=False)
    ax.set_aspect('equal')
    plt.show()

    n_points = coords.shape[0]
    if n_points > volume_visualization_sample_size:
        sample_indices = np.sort(np.random.choice(n_points, volume_visualization_sample_size, replace=False))
        sampled_coords = coords[sample_indices]
        sampled_colors = colors[sample_indices]
    else:
        sampled_coords = coords
        sampled_colors = colors

    # Create a 3D scatter plot
    scatter = go.Scatter3d(
        x=sampled_coords[:, 0],
        y=sampled_coords[:, 1],
        z=sampled_coords[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=sampled_colors,
            colorscale='Viridis',
            opacity=0.8,
        )
    )

    # Create a slice plane
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * slice_position

    plane_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    plane_points_rotated = plane_points @ R + center
    X_rotated = plane_points_rotated[:, 0].reshape(X.shape)
    Y_rotated = plane_points_rotated[:, 1].reshape(Y.shape)
    Z_rotated = plane_points_rotated[:, 2].reshape(X.shape)

    slice_plane = go.Surface(
        x=X_rotated, y=Y_rotated, z=Z_rotated,
        colorscale=[[0, 'red'], [1, 'red']],
        opacity=0.3,
        showscale=False
    )

    # Create layout
    layout = go.Layout(
        title=f'The position of synthesize slice in 3d volumn',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectratio=dict(x=1, y=1, z=1.5)
        ),
        width=1000,
        height=800
    )

    # Create the figure
    fig = go.Figure(data=[scatter, slice_plane], layout=layout)

    # Save the figure if requested
    if save_plot:
        fig.write_html(save_path)
        print(f"Plot saved to {save_path}")

    fig.show()
    return adata_new

# Example usage:
# synthesize_slice = synthesize_view(adata, angle_x=0, angle_y=90, angle_z=0, color_key='major_brain_region', 
#                                    palette=brain_region_colors, slice_offset=-850, slice_thickness=100, volume_visualization_sample_size=100000, 
#                                    save_plot=True, save_path='./3d_plot.html')
# Sagittal plane:(angle_x=0, angle_y=90, angle_z=0)
# Horizontal plane:(angle_x=90, angle_y=0, angle_z=0)
# Oblique plane:(angle_x=90, angle_y=0, angle_z=0)