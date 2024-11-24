import os
import multiprocessing
import geopandas as gpd
from tqdm import tqdm, trange


def edge2pt(edge_path_i, path_count, fixed_distance):
    path_i_pts_ls = []
    for idx, row in edge_path_i.iterrows():
        linestring = row.geometry
        n_segments = int(linestring.length / fixed_distance)

        # Capture the edge's attributes
        edge_attributes = {
            'traj_id': path_count,  # trajectory id
            'u': idx[0],
            'v': idx[1],
            'key': idx[2],
            'speed_kph': row.speed_kph
        }

        # Generate points along the linestring at fixed distances
        for count in range(n_segments+1):
            pt_j = linestring.interpolate(count * fixed_distance)
            # Combine the point with the edge's attributes
            pt_data = {'geometry': pt_j, **edge_attributes}
            path_i_pts_ls.append(pt_data)

    # Convert the list of points to a GeoDataFrame
    path_i_pts_gdf = gpd.GeoDataFrame(path_i_pts_ls)

    return path_i_pts_gdf


def proc_edge2pt_paths(
        proc_i, n_processes, edge_paths_ls, fixed_distance, path_dict
):
    # Determine the processing range for each process
    n_objs = len(edge_paths_ls)
    n_obj_proc = int(n_objs / n_processes)

    start_num = proc_i * n_obj_proc
    if proc_i == (n_processes - 1):
        end_num = n_objs - 1
    else:
        end_num = (proc_i + 1) * n_obj_proc - 1

    if proc_i == n_processes-1:  # progress bar based on the last process
        for obj_j in tqdm(range(start_num, end_num+1)):
            edge_path_j = edge_paths_ls[obj_j]
            path_j_pts_gdf = edge2pt(edge_path_j, obj_j, fixed_distance)
            path_dict[obj_j] = path_j_pts_gdf

    else:
        for obj_j in range(start_num, end_num+1):
            edge_path_j = edge_paths_ls[obj_j]
            path_j_pts_gdf = edge2pt(edge_path_j, obj_j, fixed_distance)
            path_dict[obj_j] = path_j_pts_gdf


def multiproc_edge2pt_paths(edge_paths_ls, fixed_distance=5):
    n_processes = os.cpu_count()
    print("Number of CPU processes:", n_processes)

    processes = []

    # Manager dictionary
    manager = multiprocessing.Manager()
    path_dict = manager.dict()

    print("path_dict created! Next, filling the dict...")
    for proc_i in range(n_processes):
        p = multiprocessing.Process(
            target=proc_edge2pt_paths, args=(
                proc_i, n_processes, edge_paths_ls, fixed_distance, path_dict
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("dict filled! Next create output...")
    pt_paths_ls = []
    for path_count in trange(len(edge_paths_ls)):
        pt_paths_ls.append(path_dict[path_count])

    return pt_paths_ls

