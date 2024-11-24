import os
import pandas as pd
import multiprocessing
import geopandas as gpd
from tqdm import tqdm, trange


def node2edge(node_path, edges_gdf):
    path_edges_ls = []
    for node_i in range(len(node_path)-1):
        # Origin and destination nodes
        orig_node = node_path[node_i]
        dest_node = node_path[node_i+1]
        # Record edge
        # Use key=0 to find the 1st edge connecting the origin and destination
        path_edges_ls.append(edges_gdf.loc[(orig_node, dest_node, 0)])

    # Concatenate pd.Series to form a dataframe (gdf)
    path_edges_df = pd.concat(path_edges_ls, axis=1).T
    # Project to UTM 17N
    path_edges_gdf = gpd.GeoDataFrame(path_edges_df, crs="EPSG:4326")  # WGS84
    path_edges_gdf = path_edges_gdf.to_crs("EPSG:32617")  # WGS 84 / UTM zone 17N

    return path_edges_gdf

def proc_node2edge_paths(
        proc_i, n_processes, paths_ls, edges_gdf, path_dict
):
    # Determine the processing range for each process
    n_objs = len(paths_ls)
    n_obj_proc = int(n_objs / n_processes)

    start_num = proc_i * n_obj_proc
    if proc_i == (n_processes - 1):
        end_num = n_objs - 1
    else:
        end_num = (proc_i + 1) * n_obj_proc - 1

    if proc_i == n_processes-1:  # progress bar based on the last process
        for obj_j in tqdm(range(start_num, end_num+1)):
            node_path = paths_ls[obj_j]
            path_edges_gdf = node2edge(node_path, edges_gdf)
            path_dict[obj_j] = path_edges_gdf

    else:
        for obj_j in range(start_num, end_num+1):
            node_path = paths_ls[obj_j]
            path_edges_gdf = node2edge(node_path, edges_gdf)
            path_dict[obj_j] = path_edges_gdf


def multiproc_node2edge_paths(paths_ls, edges_gdf):
    n_processes = os.cpu_count()
    print("Number of CPU processes:", n_processes)

    processes = []

    # Manager dictionary
    manager = multiprocessing.Manager()
    path_dict = manager.dict()

    print("path_dict created! Next, filling the dict...")
    for proc_i in range(n_processes):
        p = multiprocessing.Process(
            target=proc_node2edge_paths, args=(
                proc_i, n_processes, paths_ls, edges_gdf, path_dict
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("dict filled! Next create output...")
    edge_paths_ls = []
    for path_count in trange(len(paths_ls)):
        edge_paths_ls.append(path_dict[path_count])

    return edge_paths_ls

