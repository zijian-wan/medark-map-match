import os
import numpy as np
import osmnx as ox
import multiprocessing
from tqdm import tqdm, trange


def k_paths(node_ids_arr, k, graph, minimize_time):
    # Randomly choose two nodes
    two_nodes_arr = np.random.choice(node_ids_arr, size=2, replace=False)
    orig_node = two_nodes_arr[0]
    dest_node = two_nodes_arr[1]

    # k-shortest-time paths
    try:
        if minimize_time:  # minimize travel time instead of distance
            path_gen = ox.k_shortest_paths(
                G=graph, orig=orig_node, dest=dest_node, k=k, weight="travel_time"
            )  # returns a generator
        else:
            path_gen = ox.k_shortest_paths(
                G=graph, orig=orig_node, dest=dest_node, k=k
            )  # returns a generator

        k_paths_ls = list(path_gen)  # convert to list

    except:  # no path found between the two nodes
        k_paths_ls = [[-1]]

    return k_paths_ls


def proc_rand_gen_k_paths(
        proc_i, n_processes, node_ids_arr, n_times, k, graph, minimize_time, path_dict
):
    # Determine the processing range for each process
    n_objs = n_times
    n_obj_proc = int(n_objs / n_processes)

    start_num = proc_i * n_obj_proc
    if proc_i == (n_processes - 1):
        end_num = n_objs - 1
    else:
        end_num = (proc_i + 1) * n_obj_proc - 1

    if proc_i == n_processes-1:  # progress bar based on the last process
        for obj_j in tqdm(range(start_num, end_num+1)):
            k_paths_ls = k_paths(node_ids_arr, k, graph, minimize_time)
            # Save to manager dictionary
            path_dict[obj_j] = k_paths_ls

    else:
        for obj_j in range(start_num, end_num+1):
            k_paths_ls = k_paths(node_ids_arr, k, graph, minimize_time)
            # Save to manager dictionary
            path_dict[obj_j] = k_paths_ls


def multiproc_rand_gen_k_paths(
        nodes_gdf, n_times, k, graph, minimize_time=False, n_nodes_filter=None
):
    """
    Generate k-shortest-time paths from two random nodes until the total number of paths reaches n_paths.

    :param
    n_times: int
        number of times to generate a pair of nodes
    k: int
        number of shortest-time paths for each pair of nodes

    :return
    paths_ls: list (of lists)
        list of paths
        each path is represented as a list of node osmid
    """
    n_processes = os.cpu_count()
    print("Number of CPU processes:", n_processes)

    processes = []

    # Manager dictionary
    manager = multiprocessing.Manager()
    path_dict = manager.dict()

    # Node osmid
    node_ids_arr = nodes_gdf.index.to_numpy()

    print("path_dict created! Next, filling the dict...")
    for proc_i in range(n_processes):
        p = multiprocessing.Process(
            target=proc_rand_gen_k_paths, args=(
                proc_i, n_processes, node_ids_arr, n_times, k, graph, minimize_time, path_dict
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("dict filled! Next create output...")
    # Create paths_ls
    paths_ls = []
    count = 0

    for time_i in trange(n_times):
        try:
            k_paths_ls = path_dict[time_i]

            for path in k_paths_ls:
                if n_nodes_filter is not None:  # filter trajectories that are too short
                    if len(path) < n_nodes_filter:
                        continue
                    else:
                        paths_ls.append(path)
                        count += 1

        except:
            continue

    print("Number of paths generated:", count)
    return paths_ls