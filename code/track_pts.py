import os
import math
import multiprocessing
from tqdm import tqdm, trange

import numpy as np
import pandas as pd

from shapely.affinity import translate


def pt_path2track_pts(pt_path_gdf, fixed_dist=5, samp_int_mean=5):
    """
    Generate tracking pts by selecting pts from the pt paths.

    :param pt_path_gdf: gpd.GeoDataFrame
        pt path created by densified the edges with fixed_dist (in m)
    :param fixed_dist: int or float
        the distance used to densify (interpolate) the edge paths
    :param samp_int_mean: int
        sampling interval (s) to mimic
        it will be used as the mean of a normal distribution, from which sampling intervals are drawn

    :return:
        track_pts_gdf: gpd.GeoDataFrame
    """
    n_pts_in_path = len(pt_path_gdf)

    # Mimic sampling time interval
    pt_idx = 0
    samp_time = 0
    pt_indicies_ls = [0]  # the 1st pt is always added
    samp_time_ls = [0]  # record sampling time (relative time)
    speed_ls = []  # record speed in m/s
    while pt_idx < n_pts_in_path:
        # Sampling interval drawn from a normal distribution N(samp_int_mean, sigma)
        samp_int = abs(
            round(np.random.normal(samp_int_mean, samp_int_mean/5, 1)[0], 2)
        )
        if samp_int <= 0:
            samp_int = 0.1
        samp_time += samp_int
        # Speed drawn from a normal distribution N(spd_mean, sigma)
        spd_mean = pt_path_gdf.loc[pt_idx, "speed_kph"] / 3.6  # km/h -> m/s
        spd = round(np.random.normal(spd_mean, spd_mean/2, 1)[0], 3)
        if spd < 0:
            spd = 0.1
        elif spd > 1.5*spd_mean:  # speed is too high
            spd = 1.5*spd_mean
        # Calculate the index jump
        idx_jump = int(spd * samp_int / fixed_dist)
        pt_idx += idx_jump
        # Record in pt_indicies_ls
        if pt_idx < n_pts_in_path:
            if np.random.uniform(0, 1, 1)[0] > 0.9:  # randomly drop 10% of the tracking pts
                continue
            else:
                pt_indicies_ls.append(pt_idx)
                samp_time_ls.append(samp_time)
                speed_ls.append(spd)
        else:  # pt_idx exceeding the range
            if pt_indicies_ls[-1] < n_pts_in_path-1:  # the last pt is not recorded
                pt_indicies_ls.append(n_pts_in_path-1)
                samp_time_ls.append(samp_time)
                speed_ls.append(spd)
    # Assign the speed of the second to last pt to the last pt
    speed_ls.append(speed_ls[-1])

    # Tracking pts
    track_pts_gdf = pt_path_gdf.loc[pt_indicies_ls].reset_index(drop=True)
    track_pts_gdf["time"] = samp_time_ls
    track_pts_gdf["spd_mps"] = speed_ls

    return track_pts_gdf


def gen_pts_from_chosen_edge(line, pt_i_neighbors_ls):
    """
    Generate pts on the given line.

    :param line: LineString
    :param pt_i_neighbors_ls: list
    :return: generated_points_ls: list
    """
    n_gen_pts = len(pt_i_neighbors_ls)
    # Calcualte the interval at which to place the points
    interval = line.length / (n_gen_pts + 1)
    # Generate the points
    generated_points_ls = [line.interpolate(i * interval) for i in range(1, n_gen_pts + 1)]
    return generated_points_ls


def check_add_neighbors(neighbors_ls, unmatched_idx_arr, pt_i_neighbors_ls):
    for neighbor_pt in neighbors_ls:
        if neighbor_pt not in unmatched_idx_arr:
            pt_i_neighbors_ls.append(neighbor_pt)
    return pt_i_neighbors_ls


def assign_match_errors(track_pts_gdf, edges_gdf, error_rate=0.2, unmatched_err_rate=0.5):
    """
    Assign match errors to tracking pts. Wrongly matched points have labels `mat_error`=1 and shifted locations.

    :param track_pts_gdf: gpd.GeoDataFrame
    :return: track_pts_gdf: gpd.GeoDataFrame
    """
    n_pts = len(track_pts_gdf)

    # Number of match errors
    n_errors = int(error_rate * n_pts)
    # Split the array to represent either unmatched errors or wrongly matched errors
    # Number of unmatched errors
    n_unmatched = int(error_rate * unmatched_err_rate * n_pts)
    # Initial of wrongly matched (matched to neighboring edges) errors
    # This number will be later inflated as the neighboring 2 or 4 pts of each wrongly matched pt
    # are also wrongly matched
    init_n_wrongly_matched = int((n_errors - n_unmatched) / 3)

    # Select initial match error pts
    error_pt_idx_arr = np.random.choice(
        np.arange(2, n_pts-2), size=int(n_unmatched+init_n_wrongly_matched)
    )
    np.random.shuffle(error_pt_idx_arr)  # randomly shuffle the error indices, no returns

    # Unmatched error indices
    unmatched_idx_arr = error_pt_idx_arr[:n_unmatched]
    # Wrongly matched error indices
    init_wrongly_matched_idx_arr = error_pt_idx_arr[n_unmatched:]
    wrongly_matched_neighbors_ls = []
    for pt_i in init_wrongly_matched_idx_arr:
        pt_i_neighbors_ls = []
        if np.random.uniform(0, 1, 1)[0] > 0.5:  # add neighboring 2 pts
            pt_i_neighbors_ls = check_add_neighbors([pt_i-1, pt_i+1], unmatched_idx_arr, pt_i_neighbors_ls)
        else: # add neighboring 4 pts
            pt_i_neighbors_ls = check_add_neighbors(
                [pt_i-2, pt_i-1, pt_i+1, pt_i+2], unmatched_idx_arr, pt_i_neighbors_ls
            )
        wrongly_matched_neighbors_ls.append(pt_i_neighbors_ls)

    # Complete index array of wrongly matched pts
    neighbors_flattened_arr = np.array(sum(wrongly_matched_neighbors_ls, []))
    wrongly_matched_idx_arr = np.concatenate([init_wrongly_matched_idx_arr, neighbors_flattened_arr])

    # Assign mat_error label, unmatched: 1, wrongly matched: 2
    track_pts_gdf["mat_error"] = 0
    track_pts_gdf.loc[unmatched_idx_arr, "mat_error"] = 1
    track_pts_gdf.loc[wrongly_matched_idx_arr, "mat_error"] = 2

    # Shift the locations of the unmatched pts
    for pt_i in unmatched_idx_arr:
        # y shift
        shift = np.random.normal(100, 30, 1)[0]  # mostly in range(0, 200)
        if np.random.uniform(0, 1, 1)[0] > 0.5:
            x_shift = shift
        else:
            x_shift = -shift

        # y shift
        shift = np.random.normal(100, 30, 1)[0]
        if np.random.uniform(0, 1, 1)[0] > 0.5:
            y_shift = shift
        else:
            y_shift = -shift

        # Shift the location
        track_pts_gdf.at[pt_i, 'geometry'] = translate(track_pts_gdf.at[pt_i, 'geometry'], xoff=x_shift, yoff=y_shift)

    # Place the wrongly matched pts on the neighboring edges
    # Spatial index
    spatial_idx = edges_gdf.sindex
    for pt_count, pt_i in enumerate(init_wrongly_matched_idx_arr):
        buffer_dist = 50
        possible_matches_idx_ls = []
        while len(possible_matches_idx_ls) <= 0:
            buffer_dist *= 2  # starts with buffer distance 50 m
            pt_i_buffer = track_pts_gdf.loc[pt_i, "geometry"].buffer(buffer_dist)
            possible_matches_idx_ls = list(spatial_idx.intersection(pt_i_buffer.bounds))
        # Randomly choose a neighboring edge
        chosen_edge_idx = edges_gdf.index[
            np.random.choice(np.array(possible_matches_idx_ls), size=1)[0]
        ]
        chosen_edge = edges_gdf.loc[chosen_edge_idx]

        # Generate the new coordinates to shift the location of pt_i and its neighbors
        pt_i_neighbors_ls = wrongly_matched_neighbors_ls[pt_count]
        n_neighbors = len(pt_i_neighbors_ls)
        if n_neighbors == 0:
            pt_i_neighbors_ls = [pt_i]
        else:  # insert pt_i to the middle of the list
            insert_idx = n_neighbors // 2
            pt_i_neighbors_ls.insert(insert_idx, pt_i)

        generated_points_ls = gen_pts_from_chosen_edge(
            line=chosen_edge["geometry"], pt_i_neighbors_ls=pt_i_neighbors_ls
        )

        for pt_count_2, pt_j in enumerate(pt_i_neighbors_ls):
            track_pts_gdf.at[pt_j, 'geometry'] = generated_points_ls[pt_count_2]

    return track_pts_gdf


def cal_hdg(pt1, pt2):
    '''
    Calculate heading.
        Due north is 0.
        Clockwise.
        The calculated heading is the same as OSM bearing.

    Input:
        pt1: [x, y]
        pt2: [x, y]
            pt2 is the subsequent pt of pt1
    '''
    vec_x = pt2[0] - pt1[0]
    vec_y = pt2[1] - pt1[1]

    if vec_x * vec_y == 0:
        if vec_x == 0:
            if vec_y < 0:
                heading = 180.0
            else:
                heading = 0.0
        else:  # vec_x != 0, vec_y == 0
            if vec_x > 0:
                heading = 90.0
            else:
                heading = 270.0
    else:  # neither of vec_x nor vec_y is 0
        if vec_x > 0:
            if vec_y > 0:  # 1st quadrant
                heading = math.degrees(math.atan(round(vec_x/vec_y, 6)))
            else:  # vec_x > 0 & vec_y < 0 -> 4th quadrant
                heading = math.degrees(
                    math.atan(round(-vec_y/vec_x, 6))
                ) + 90
        else:  # vec_x < 0
            if vec_y < 0:  # 3rd quadrant
                heading = math.degrees(math.atan(round(vec_x/vec_y, 6))) + 180
            else:  # 2rd quadrant
                heading = math.degrees(math.atan(round(vec_y/(-vec_x), 6))) + 270

    heading = round(heading, 3)

    return heading


def cal_spdxy_hdg(track_pts_gdf):
    n_pts = len(track_pts_gdf)

    # Time interval of the 1st pt
    track_pts_gdf.loc[0, "time_int"] = 0

    for pt_i in range(n_pts-1):
        pt_i_next = pt_i + 1  # next pt

        # Calculate heading
        pt1 = [track_pts_gdf.loc[pt_i, "geometry"].x, track_pts_gdf.loc[pt_i, "geometry"].y]
        pt2 = [track_pts_gdf.loc[pt_i_next, "geometry"].x, track_pts_gdf.loc[pt_i_next, "geometry"].y]
        pt_i_hdg = cal_hdg(pt1, pt2)

        # Calculate speed in the x,y directions
        disp_x = pt2[0] - pt1[0]  # displacement_x
        disp_y = pt2[1] - pt1[1]  # displacement_y
        t_int = track_pts_gdf.loc[pt_i_next, "time"] - track_pts_gdf.loc[pt_i, "time"]
        spd_x = round(disp_x/t_int, 3)
        spd_y = round(disp_y/t_int, 3)

        # Record in gdf
        track_pts_gdf.loc[pt_i, "heading"] = pt_i_hdg
        track_pts_gdf.loc[pt_i, "speed_x"] = spd_x
        track_pts_gdf.loc[pt_i, "speed_y"] = spd_y
        track_pts_gdf.loc[pt_i_next, "time_int"] = t_int

    # The last pt has the same speed and heading as the previous one
    track_pts_gdf.loc[n_pts-1, "heading"] = track_pts_gdf.loc[n_pts-2, "heading"]
    track_pts_gdf.loc[n_pts-1, "speed_x"] = track_pts_gdf.loc[n_pts-2, "speed_x"]
    track_pts_gdf.loc[n_pts-1, "speed_y"] = track_pts_gdf.loc[n_pts-2, "speed_y"]

    return track_pts_gdf


def pt2track(pt_path_gdf, fixed_dist, edges_gdf, samp_int_mean, error_rate):
    # Select pts to represent tracking pts
    track_pts_gdf = pt_path2track_pts(pt_path_gdf, fixed_dist=fixed_dist, samp_int_mean=samp_int_mean)
    # Select pts that represent wrong matches
    track_pts_err_gdf = assign_match_errors(track_pts_gdf, edges_gdf, error_rate=error_rate, unmatched_err_rate=0.5)
    # Calculate attributes
    track_pts_attr_gdf = cal_spdxy_hdg(track_pts_gdf=track_pts_err_gdf)
    return track_pts_attr_gdf


def proc_pt2track(
        proc_i, n_processes, pt_paths_ls, fixed_dist, edges_gdf, samp_int_mean, error_rate, path_dict
):
    # Determine the processing range for each process
    n_objs = len(pt_paths_ls)
    n_obj_proc = int(n_objs / n_processes)

    start_num = proc_i * n_obj_proc
    if proc_i == (n_processes - 1):
        end_num = n_objs - 1
    else:
        end_num = (proc_i + 1) * n_obj_proc - 1

    if proc_i == n_processes-1:  # progress bar based on the last process
        for obj_j in tqdm(range(start_num, end_num+1)):
            pt_path_gdf = pt_paths_ls[obj_j]
            track_pts_attr_gdf = pt2track(pt_path_gdf, fixed_dist, edges_gdf, samp_int_mean, error_rate)
            path_dict[obj_j] = track_pts_attr_gdf

    else:
        for obj_j in range(start_num, end_num+1):
            pt_path_gdf = pt_paths_ls[obj_j]
            track_pts_attr_gdf = pt2track(pt_path_gdf, fixed_dist, edges_gdf, samp_int_mean, error_rate)
            path_dict[obj_j] = track_pts_attr_gdf


def multiproc_pt2track(pt_paths_ls, fixed_dist, edges_gdf, samp_int_mean, error_rate):
    n_processes = os.cpu_count()
    print("Number of CPU processes:", n_processes)

    processes = []

    # Manager dictionary
    manager = multiprocessing.Manager()
    path_dict = manager.dict()

    print("path_dict created! Next, filling the dict...")
    for proc_i in range(n_processes):
        p = multiprocessing.Process(
            target=proc_pt2track, args=(
                proc_i, n_processes, pt_paths_ls, fixed_dist, edges_gdf, samp_int_mean, error_rate, path_dict
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("dict filled! Next create output...")
    track_pts_attr_ls = []
    for path_count in trange(len(pt_paths_ls)):
        track_pts_attr_ls.append(path_dict[path_count])

    # Concatenate
    track_pts_combined_gdf = pd.concat(track_pts_attr_ls, ignore_index=True)
    #  Set crs
    track_pts_combined_gdf = track_pts_combined_gdf.set_crs('epsg:32617')

    return track_pts_combined_gdf

