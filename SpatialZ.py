import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import isspmatrix
import ot 
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
from anndata import AnnData
import MENDER
from tqdm import tqdm
import contextlib
import io
from scipy.spatial import cKDTree
import time
#from sklearn.neighbors import KDTree
import os

def Generate_spatialz(adata1, adata2, adata1_id='above', adata2_id='below',
                      alpha=0.5, device='auto', n_cell=None, k_neighbors=1,
                      n_mag=1.0, lr=1e5, nb_iter_max=3000, seed=42,
                      num_projections=80, cell_type_key='cell_type', syn_mode= 'default', k_sam=3, 
                      micro_env_key='mender', Beta = 100, add_obs_list= None, verbose=True):
    """
    The Generate_spatialz function is designed to integrate spatial coordinates and gene expression from two AnnData objects and generate a new AnnData object. 

    Input Parameters:
        adata1, adata2 (AnnData): The two input AnnData objects containing spatial coordinates in obsm['spatial'] and gene expression in X. 
        adata1_id, adata2_id (str): Unique identifiers for each slice. 
        alpha (float): The approximate factor between upper and lower slice.
        device (torch.device or str): Specifies the computation device (e.g., 'cuda:0' for GPU). 
                                      If set to 'auto', the function selects between GPU and CPU based on availability.
        n_cell (int, optional): Specifies the number of cells in the synthesized slice. 
                                If not provided, this will automatically calculate based on the cell number of adjacent upper and lower slices.
        k_neighbors (int): The number of nearest neighbors to consider when integrating cell types and other attributes.
        n_mag (float): A magnification factor, allowing to create a denser synthesized dataset in plane.
        lr (float): The learning rate for gradient optimization using Sliced Wasserstein Distance.
        nb_iter_max (int): The maximum number of iterations.
        seed (int): A random seed.
        num_projections (int): The number of random projections using the Sliced Wasserstein Distance.
        cell_type_key (str): Molecularly defined identifiers for each cell.
        Annotation information transferred from the real slices in obs.
        syn_mode (str): Specifies the mode for synthesizing gene expression data, with options 'default' and 'fast'. 
                        The 'default' mode includes detailed microenvironment measurements, whereas 'fast' focuses on quicker integration from the same cell type.
        k_sam (int): Determines the number of samples to draw when synthesizing gene expression.
        micro_env_key (str): Specifies the key in obsm for storing microenvironment measurements.
        Beta = (float): Weight of microenvironment similarity.
        add_obs_list (list of str, optional): Annotation information transferred from the real slices in obs.
        verbose (bool): Controls the verbosity of the output during the function's execution.
    
    Returns:
        A new AnnData object (a intermediate virtual slices) that combines spatial coordinates and gene expression data. 
        The synthesized data includes: spatial coordinates, gene expression, Cell types and other annotations.

    """

    def print_time(message, start):
        if verbose:
            print(f"{message} time: {time.time() - start:.2f} seconds")
    
    # Device handling
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)

    # Check required fields in AnnData
    if 'spatial' not in adata1.obsm or 'spatial' not in adata2.obsm:
        raise ValueError("Both adata1 and adata2 must have 'spatial' coordinates in obsm.")
    if cell_type_key not in adata1.obs or cell_type_key not in adata2.obs:
        raise ValueError(f"Both adata1 and adata2 must have '{cell_type_key}' information in obs.")
    if syn_mode not in ['default', 'fast']:
        raise ValueError("syn_mode must be either 'default' or 'fast'.")

    # Adjusting indices with unique identifiers
    adata1.obs_names = [f"{name}_{adata1_id}" for name in adata1.obs_names]
    adata2.obs_names = [f"{name}_{adata2_id}" for name in adata2.obs_names]
    
    if verbose:
        print("Begin to generate spatial coordinates......")
    
    start_time = time.time()
    # Convert spatial coordinates to PyTorch tensors
    coor1_torch = torch.tensor(adata1.obsm['spatial'], dtype=torch.float).to(device=device)
    coor2_torch = torch.tensor(adata2.obsm['spatial'], dtype=torch.float).to(device=device)

    # Number of points in the barycenter
    if n_cell is None:
        n_cell = int((alpha * adata1.n_obs + (1 - alpha) * adata2.n_obs) * n_mag)
    
    # Uniform sampling from both coor1_torch and coor2_torch
    n_cell1 = int(n_cell * alpha)
    n_cell2 = n_cell - n_cell1

    sampled_indices1 = np.linspace(0, coor1_torch.shape[0] - 1, n_cell1, dtype=int)
    sampled_indices2 = np.linspace(0, coor2_torch.shape[0] - 1, n_cell2, dtype=int)

    Coor_init1 = coor1_torch[sampled_indices1].cpu().numpy()
    Coor_init2 = coor2_torch[sampled_indices2].cpu().numpy()
    Coor_init = np.concatenate([Coor_init1, Coor_init2], axis=0)   
    
    # # Sampling from the longer of coor1_torch or coor2_torch
    # longer_coor = coor1_torch
    # sampled_indices = np.random.choice(longer_coor.shape[0], size=n_cell, replace=True)
    # Coor_init = longer_coor[sampled_indices].cpu().numpy()
    Coor_torch = torch.tensor(Coor_init, dtype=torch.float32, device=device).requires_grad_(True)
    print_time("coordinate initialization", start_time)

    start_time = time.time()
    # Optimization loop using Sliced Wasserstein Distance
    gen = torch.Generator(device=device).manual_seed(seed)
    for i in range(nb_iter_max):
        loss = (alpha * ot.sliced_wasserstein_distance(Coor_torch, coor1_torch, n_projections=num_projections, seed=gen) +
                (1 - alpha) * ot.sliced_wasserstein_distance(Coor_torch, coor2_torch, n_projections=num_projections, seed=gen))
        loss.backward()

        with torch.no_grad():
            Coor_torch -= Coor_torch.grad * lr
            Coor_torch.grad.zero_()
        # Print loss every 1000 iterations
        if verbose and i % 1000 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")
    print_time("Ot optimization", start_time)

    # Create new AnnData
    Coor_final = Coor_torch.detach().cpu().numpy()
    var_data = pd.DataFrame(index=adata1.var_names)
    adata3 = AnnData(X=np.zeros((Coor_final.shape[0], adata1.n_vars), dtype=np.float32), var=var_data, dtype=np.float32)
    adata3.obsm['spatial'] = Coor_final
    if verbose:
        print("Begin to determine cell types......")
    
    start_time = time.time()
    # Nearest neighbors for cell type integration
    nn_adata1 = NearestNeighbors(n_neighbors=k_neighbors).fit(adata1.obsm['spatial'])
    nn_adata2 = NearestNeighbors(n_neighbors=k_neighbors).fit(adata2.obsm['spatial'])
    distances_1, indices_1 = nn_adata1.kneighbors(Coor_final)
    distances_2, indices_2 = nn_adata2.kneighbors(Coor_final)
    
    epsilon = 0.1  # Or choose an appropriate value based on data scale
    
    sim_celltype = []
    closest_indices = []
    for i in range(Coor_final.shape[0]):
        # Obtain closest cell types
        #print(f"Cell {i} indices in adata1: {indices_1[i]}, types: {adata1.obs.iloc[indices_1[i]][cell_type_key].values}")
        #print(f"Cell {i} indices in adata2: {indices_2[i]}, types: {adata2.obs.iloc[indices_2[i]][cell_type_key].values}")
        types_1 = adata1.obs.iloc[indices_1[i]][cell_type_key].values
        types_2 = adata2.obs.iloc[indices_2[i]][cell_type_key].values

        weights_1 = 1 / (distances_1[i] + epsilon)
        weights_2 = 1 / (distances_2[i] + epsilon)
        all_types = np.concatenate([types_1, types_2])
        all_weights = np.concatenate([weights_1, weights_2])

        # Determine dominant cell type considering weights
        type_weights = pd.Series(all_weights, index=all_types).groupby(level=0).sum()
        dominant_type = type_weights.idxmax() if not type_weights.empty else None
        #print(f"Cell {i} dominant type: {dominant_type}")
        
        # Find the closest index for the dominant type in either dataset
        if dominant_type:
            min_dist_1 = np.min(distances_1[i][types_1 == dominant_type]) if dominant_type in types_1 else np.inf
            min_dist_2 = np.min(distances_2[i][types_2 == dominant_type]) if dominant_type in types_2 else np.inf
            if min_dist_1 <= min_dist_2:
                closest_index = adata1.obs_names[indices_1[i][np.where(types_1 == dominant_type)[0][0]]]
            else:
                closest_index = adata2.obs_names[indices_2[i][np.where(types_2 == dominant_type)[0][0]]]
        else:
            # Select the closest available cell type as the dominant type
            closest_dist_1 = np.argmin(distances_1[i])
            closest_dist_2 = np.argmin(distances_2[i])
            if distances_1[i][closest_dist_1] < distances_2[i][closest_dist_2]:
                dominant_type = types_1[closest_dist_1]
                closest_index = adata1.obs_names[indices_1[i][closest_dist_1]]
            else:
                dominant_type = types_2[closest_dist_2]
                closest_index = adata2.obs_names[indices_2[i][closest_dist_2]]

        sim_celltype.append(dominant_type)        
        closest_indices.append(closest_index)
        
    adata3.obs[cell_type_key] = sim_celltype
    print_time("Cell type determination", start_time)

    # Attribute transfer using modified indices
    if add_obs_list is not None:
        start_time = time.time()
        if verbose:
            print("Begin to transfer the attribute......")
        for obs_key in add_obs_list:
            adata3.obs[obs_key] = [
                adata1.obs[obs_key][index] if adata1_id in index else adata2.obs[obs_key][index]
                for index in closest_indices
            ]
        print_time("Transfer the attribute", start_time)
    
    if syn_mode == 'default':
        if verbose:
            print("Begin to calculate micro-environment measurement......")
        start_time = time.time()
        adata1, adata2, adata3 = perform_microenv_measure(adata1, adata2, adata3, cell_type_key, 
                             scale=6, radius=15, micro_env_key=micro_env_key)    
        print_time("Micro-environment measurement", start_time)   

        if verbose:
            print("Begin to synthesize gene expression......")
        start_time = time.time()
        adata3 = synthesize_gene_expression([adata1, adata2], adata3, cell_type_key, k_sam, micro_env_key, Beta = Beta)
        print_time("Gene expression synthesis", start_time)
    
    elif syn_mode == 'fast':
        if verbose:
            print("Begin to synthesize gene expression......")
        start_time = time.time()
        adata3 = synthesize_gene_expression_fast([adata1, adata2], adata3, cell_type_key, k_sam)    
        print_time("Gene expression synthesis", start_time)
    
    return adata3


def perform_microenv_measure(adata1, adata2, adata3, cell_type_key, 
                             scale = 6, radius = 15, micro_env_key='mender'):
    # input parameters of MENDER

    temp_adata_list = [adata1, adata2, adata3]

    adata1.obs['temp_slice_id'] = 'temp_adata1'
    adata2.obs['temp_slice_id'] = 'temp_adata2'
    adata3.obs['temp_slice_id'] = 'temp_adata3'
    
    temp_adata_all = temp_adata_list[0].concatenate(temp_adata_list[1:])
    
    temp_adata_all.obs['temp_slice_id'] = temp_adata_all.obs['temp_slice_id'].astype('category')
    temp_adata_all.obs[cell_type_key] = temp_adata_all.obs[cell_type_key].astype('category')
    
    # main body of MENDER
    msm = MENDER.MENDER(
        temp_adata_all,
        batch_obs = 'temp_slice_id',
        ct_obs = cell_type_key,
        random_seed = 0
    )

    # set the MENDER parameters
    msm.prepare()
    msm.set_MENDER_para(
        n_scales=scale,
        nn_mode='radius',
        nn_para=radius,
    )
    
    # Suppressing output from msm.run_representation_mp
    with contextlib.redirect_stdout(io.StringIO()):
        msm.run_representation_mp(200)
    
    adata_MENDER = msm.adata_MENDER
    temp_adata_all.obsm[micro_env_key] = adata_MENDER.X

    adata1.obsm[micro_env_key] = temp_adata_all[temp_adata_all.obs['temp_slice_id']=='temp_adata1'].obsm[micro_env_key]
    adata2.obsm[micro_env_key] = temp_adata_all[temp_adata_all.obs['temp_slice_id']=='temp_adata2'].obsm[micro_env_key]
    adata3.obsm[micro_env_key] = temp_adata_all[temp_adata_all.obs['temp_slice_id']=='temp_adata3'].obsm[micro_env_key]
    
    del adata1.obs['temp_slice_id']
    del adata2.obs['temp_slice_id']
    del adata3.obs['temp_slice_id']
    
    return adata1, adata2, adata3


def synthesize_gene_expression(reference_adatas, query_adata, cell_type_key, k_sam, micro_env_key, Beta):
    if micro_env_key not in query_adata.obsm:
        raise ValueError(f"Microenvironment key '{micro_env_key}' is not present in query dataset. Please perform microenvironment measurement calculations first.")
    
    combined_adata = reference_adatas[0].concatenate(reference_adatas[1:], batch_key='batch')
    
    # Create KD-tree for spatial coordinates
    kdtree = cKDTree(combined_adata.obsm['spatial'])
    
    # Pre-compute cosine similarities
    cos_similarities = cosine_similarity(combined_adata.obsm[micro_env_key], query_adata.obsm[micro_env_key])
    
    # Process each cell in query_adata
    for i, cell_type in enumerate(query_adata.obs[cell_type_key]):
        # Filter cells by the same type from combined dataset
        same_type_mask = combined_adata.obs[cell_type_key] == cell_type
        same_type_data = combined_adata[same_type_mask]
        
        if same_type_data.n_obs > 0:
            # Query KD-tree for k nearest neighbors
            #distances, indices = kdtree.query(query_adata.obsm['spatial'][i], k=min(k_sam, same_type_data.n_obs))
            distances, indices = kdtree.query(query_adata.obsm['spatial'][i], k=min(k_sam, same_type_data.n_obs))
            indices = np.atleast_1d(indices)  # Ensure that indices is always an array

            
            # Filter indices to include only cells of the same type
            valid_indices = indices[same_type_mask[indices]]
            
            if len(valid_indices) > 0:
                # Get pre-computed cosine similarities for valid indices
                similarities = cos_similarities[valid_indices, i]
                
                # Adjust similarities for sampling weights
                exp_similarities = np.exp(Beta*similarities)
                weights = exp_similarities / exp_similarities.sum()
                
                # Sample gene expressions
                sampled_expression = np.zeros(query_adata.X[i].shape)
                for gene_idx in range(sampled_expression.shape[0]):
                    chosen_cell_idx = np.random.choice(np.arange(len(valid_indices)), p=weights)
                    sampled_expression[gene_idx] = combined_adata.X[valid_indices[chosen_cell_idx], gene_idx]
                
                query_adata.X[i] = sampled_expression
            else:
                print(f"No valid cells of the same type found for cell index {i}")
        else:
            print(f"No cells of the same type found for cell index {i}")

    # Clear pre-computed cosine similarities to free memory
    del cos_similarities
    
    return query_adata

def synthesize_gene_expression_fast(reference_adatas, query_adata, cell_type_key, k_sam):
    combined_adata = reference_adatas[0].concatenate(reference_adatas[1:], batch_key='batch')

    nn = NearestNeighbors(n_neighbors=10).fit(combined_adata.obsm['spatial'])
    distances, indices = nn.kneighbors(query_adata.obsm['spatial'])
    
    for i in range(query_adata.n_obs):
        query_cell_class = query_adata.obs[cell_type_key][i]
        same_type_indices = [idx for idx in indices[i] if combined_adata.obs[cell_type_key][idx] == query_cell_class]
        
        if len(same_type_indices) > 0:
            k_use = min(len(same_type_indices), k_sam)
            selected_indices = same_type_indices[:k_use]
        else:
            # If no same type in the 10 nearest neighbors, find the nearest of the same type
            all_same_type_indices = np.where(combined_adata.obs[cell_type_key] == query_cell_class)[0]
            distances_to_same_type = np.linalg.norm(combined_adata.obsm['spatial'][all_same_type_indices] - query_adata.obsm['spatial'][i], axis=1)
            selected_indices = [all_same_type_indices[np.argmin(distances_to_same_type)]]
        
        expressions = combined_adata.X[selected_indices]
        if isspmatrix(expressions):
            expressions = expressions.toarray()
        query_adata.X[i] = expressions.mean(axis=0)
    
    return query_adata


def Generate_multiple_spatialz(adata1, adata2, num_sim, adata1_id='above', adata2_id='below',
                               device='auto', n_cell=None, n_mag=1.0, lr=1e5, nb_iter_max=3000, seed=42, num_projections=80,
                               cell_type_key='cell_type',syn_mode= 'default', k_sam=3, micro_env_key = 'mender', Beta = 100, add_obs_list=None, verbose=True,
                               include_raw=True):
    """
    The Generate_multiple_spatialz function extends the capabilities of the Generate_spatialz by generating multiple integrated AnnData objects. 

    Input Parameters
        adata1, adata2 (AnnData): The two input AnnData objects containing spatial coordinates in obsm['spatial'] and gene expression in X. 
        num_sim (int): The number of synthesized slices to generate. 
        adata1_id, adata2_id (str): Unique identifiers for each slice.
        device (torch.device or str): Specifies the computation device (e.g., 'cuda:0' for GPU). 
                                      If set to 'auto', the function selects between GPU and CPU based on availability.
        n_cell (int, optional): Specifies the number of cells in the synthesized slice. 
                                If not provided, this will automatically calculate based on the cell number of adjacent upper and lower slices.
        n_mag (float): A magnification factor, allowing to create a denser synthesized dataset in plane.
        lr (float): The learning rate for gradient optimization using Sliced Wasserstein Distance.
        nb_iter_max (int): The maximum number of iterations.
        seed (int): A random seed.
        num_projections (int): The number of random projections using the Sliced Wasserstein Distance.
        cell_type_key (str): Molecularly defined identifiers for each cell.
        syn_mode (str): Specifies the mode for synthesizing gene expression data, with options 'default' and 'fast'. 
                        The 'default' mode includes detailed microenvironment measurements, whereas 'fast' focuses on quicker integration from the same cell type.
        k_sam (int): Determines the number of samples to draw when synthesizing gene expression.
        micro_env_key (str): Specifies the key in obsm for storing microenvironment measurements.
        Beta = (float): Weight of microenvironment similarity.
        add_obs_list (list of str, optional): Annotation information transferred from the real slices in obs.
        verbose (bool): Controls the verbosity of the output during the function's execution.
        include_raw (bool): Whether to include the original datasets in the final output alongside the synthesized slices. 


    Returns:
        A concatenated AnnData object that includes:  multiple synthesized slices generated by varying the alpha parameter. 
        Optionally includes the original data if include_raw is set to true. 
    """
    sim_adatas = []
    num_sim = num_sim + 1

    # Optionally include raw adata1 at the beginning
    if include_raw:
        adata1.obs['slice_id'] = f"{adata1_id}"
        adata1.obs['data_type'] = 'real'
        sim_adatas.append(adata1.copy())

    #for i in range(1, num_sim):  # Start from 1 to exclude alpha=1 and end at num_sim to exclude alpha=0
    for i in tqdm(range(1, num_sim), desc="Generating simulations"): 
        alpha = 1 - i / num_sim
        #print(alpha)
        sim_adata = Generate_spatialz(adata1, adata2, adata1_id=adata1_id, adata2_id=adata2_id,
                                      alpha=alpha, device=device, n_cell=n_cell, n_mag=n_mag, lr=lr,
                                      nb_iter_max=nb_iter_max, seed=seed, num_projections=num_projections,
                                      cell_type_key=cell_type_key, syn_mode= syn_mode, k_sam=k_sam, micro_env_key = micro_env_key, Beta = Beta, add_obs_list=add_obs_list,verbose=True
                                      )
        # Create slice_id
        #slice_id = f"{adata1_id}-{adata2_id}-{alpha:.2f}"
        slice_id = f"{adata1_id}-{adata2_id}-{i}"
        sim_adata.obs['slice_id'] = slice_id
        sim_adata.obs['data_type'] = 'synthetic'
        sim_adatas.append(sim_adata)
        if verbose:
            print(f"Completed {slice_id} generated!")

    # Optionally include raw adata2 at the end
    if include_raw:
        adata2.obs['slice_id'] = f"{adata2_id}"
        adata2.obs['data_type'] = 'real'
        sim_adatas.append(adata2.copy())

    # Concatenate all generated AnnData objects
    concatenated_adata = AnnData.concatenate(*sim_adatas, batch_key='slice_id', batch_categories=[s.obs['slice_id'][0] for s in sim_adatas])
    return concatenated_adata

# Example usage:
# adatas = Generate_multiple_spatialz(adata1, adata2, num_sim=7, device='cuda:0', include_raw=True)



def Generate_multiple_slices(adata_list, num_sim_list, adatas_id_list,
                             save_path, device='auto', n_cell=None, n_mag=1.0, lr=1e-5, nb_iter_max=3000, seed=42, 
                             num_projections=80, cell_type_key='cell_type', syn_mode= 'default', k_sam=3, micro_env_key='mender', Beta = 100, add_obs_list=None,
                             verbose=True, include_raw=True):
    """
 
    The Generate_multiple_slices function aims to integrate multiple AnnData objects, 
    generating synthesized slices from pairs of data while optionally retaining the original data. 
    
    Input Parameters:
        adata_list (list of AnnData): A list of AnnData objects of consecutive pairs. 
                                      Each AnnData object must have spatial coordinates in obsm['spatial'] and gene expression data in X. 
                                      The function processes consecutive pairs of these datasets.
        num_sim_list (list of int): A list specifying the number of synthesized slices to generate for each pair data. 
        adatas_id_list (list of str): A list of unique identifiers for each AnnData data in adata_list. 
        save_path (str): The directory where the generated synthesized AnnData objects are saved. 
        device (torch.device or str): Specifies the computation device (e.g., 'cuda:0' for GPU). 
                                      If set to 'auto', the function selects between GPU and CPU based on availability.
        n_cell (int, optional): Specifies the number of cells in the synthesized slice. 
                                If not provided, this will automatically calculate based on the cell number of adjacent upper and lower slices.
        n_mag (float): A magnification factor, allowing to create a denser synthesized dataset in plane.
        lr (float): The learning rate for gradient optimization using Sliced Wasserstein Distance.
        nb_iter_max (int): The maximum number of iterations.
        seed (int): A random seed.
        num_projections (int): The number of random projections using the Sliced Wasserstein Distance.
        cell_type_key (str): Molecularly defined identifiers for each cell.
        syn_mode (str): Specifies the mode for synthesizing gene expression data, with options 'default' and 'fast'. 
                        The 'default' mode includes detailed microenvironment measurements, whereas 'fast' focuses on quicker integration from the same cell type.
        k_sam (int): Determines the number of samples to draw when synthesizing gene expression.
        micro_env_key (str): Specifies the key in obsm for storing microenvironment measurements.
        Beta = (float): Weight of microenvironment similarity.
        add_obs_list (list of str, optional): Annotation information transferred from the real slices in obs.
        verbose (bool): Controls the verbosity of the output during the function's execution.
        include_raw (bool): Whether to include the original datasets in the final output alongside the synthesized slices. 

    Returns:
        A concatenated AnnData object that contains: 
        original Datas: If include_raw=True, the function includes copies of the original datasets from adata_list.
        synthesized Datas: Intermediate datasets are generated for each pair of AnnData objects in adata_list, as specified by num_sim_list. 
    """
    concatenated_slices = []

    if include_raw:
        for adata, id in zip(adata_list, adatas_id_list):
            adata.obs['slice_id'] = f"{id}"
            adata.obs['data_type'] = 'real'
            concatenated_slices.append(adata.copy())
            adata.write(os.path.join(save_path, f"{id}_raw.h5ad"))

    for i in tqdm(range(len(adata_list) - 1), desc="Generating simulations for slices"):
        num_sim = num_sim_list[i]
        adata1 = adata_list[i]
        adata2 = adata_list[i + 1]
        adata1_id = adatas_id_list[i]
        adata2_id = adatas_id_list[i + 1]

        for j in range(1, num_sim + 1):  # Varying alpha from 1 to 0 in num_sim steps
            alpha = 1 - j / (num_sim + 1)
            sim_adata = Generate_spatialz(adata1, adata2, adata1_id=adata1_id, adata2_id=adata2_id,
                                          alpha=alpha, device=device, n_cell=n_cell, n_mag=n_mag, lr=lr,
                                          nb_iter_max=nb_iter_max, seed=seed, num_projections=num_projections,
                                          cell_type_key=cell_type_key, syn_mode= syn_mode, k_sam=k_sam, micro_env_key=micro_env_key, Beta = Beta, add_obs_list=add_obs_list, verbose=verbose
                                          )
            slice_id = f"{adata1_id}-{adata2_id}-{j}"
            sim_adata.obs['slice_id'] = slice_id
            sim_adata.obs['data_type'] = 'synthetic'
            concatenated_slices.append(sim_adata)
            sim_adata.write(os.path.join(save_path, f"{slice_id}.h5ad"))
            if verbose:
                print(f"Completed {slice_id} generated and saved!")

    # Concatenate all generated and raw AnnData objects
    adatas = AnnData.concatenate(*concatenated_slices, batch_key='slice_id',
                                 batch_categories=[s.obs['slice_id'][0] for s in concatenated_slices])
    return adatas

# Example usage:
# adatas_combined = Generate_multiple_slices([adata1, adata2, adata3], [10, 5], ['sample1', 'sample2', 'sample3'], device='cuda:0', include_raw=True)
