import os
import copy
import dill
import pickle
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString


def error_segments(pred_labels_ls):
    """
    Get error segments.
    :param pred_labels_ls:
    :return:
    """
    err_seg_ls = []

    for samp_labels_arr in tqdm(pred_labels_ls):
        samp_err_seg_ls = []

        if sum(samp_labels_arr) > 0:
            # Initialization
            pt_i = 0
            err_seg = []
            while pt_i < len(samp_labels_arr):
                if samp_labels_arr[pt_i] == 0: # pt_i is not an error
                    if len(err_seg) > 0:
                        samp_err_seg_ls.append(err_seg)
                        err_seg = [] # Reset err_seg_ls
                else: # pt_i is an error
                    err_seg.append(pt_i)
                pt_i += 1
            # Final check
            if len(err_seg) > 0:
                samp_err_seg_ls.append(err_seg)

        err_seg_ls.append(samp_err_seg_ls)

    return err_seg_ls


def find_anchor_pts(single_err_seg, coords_df):
    # Find two anchor pts
    # Anchor pt 1
    if single_err_seg[0] > 0: # Not the 1st pt
        anchor_pt1 = single_err_seg[0] - 1
    else:
        anchor_pt1 = single_err_seg[0]
    pt1_coords_arr = coords_df.loc[anchor_pt1].to_numpy()
    # Anchor pt 2
    if single_err_seg[-1] < len(coords_df) - 1: # Not the last pt
        anchor_pt2 = single_err_seg[-1] + 1
    else:
        anchor_pt2 = single_err_seg[-1]
    pt2_coords_arr = coords_df.loc[anchor_pt2].to_numpy()

    return anchor_pt1, anchor_pt2, pt1_coords_arr, pt2_coords_arr


def find_nearest_node(coords_arr, nodes_gdf, node_spatial_idx, buffer_dist=100):
    # shapely Point object
    pt = Point(coords_arr)

    # Sparse search
    possible_nodes_ls = list(node_spatial_idx.intersection(pt.buffer(buffer_dist).bounds))
    possible_nodes_gdf = nodes_gdf.iloc[possible_nodes_ls]
    # Precise search by distance
    node_dist_sorted = possible_nodes_gdf.distance(pt).sort_values().reset_index()
    nearest_node_osmid = node_dist_sorted.loc[0, "osmid"]

    return nearest_node_osmid


def node_path2path_pts_coords(node_path, edges_gdf, samp_traj_gdf):
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
    path_edges_gdf = gpd.GeoDataFrame(path_edges_df, crs=samp_traj_gdf.crs)

    path_pts_coords_df = path_edges_gdf["geometry"].get_coordinates().reset_index(drop=True)

    return path_pts_coords_df


def avg_intp(coords1, coords2, n_pts2intp):
    x_intp_ls = []
    y_intp_ls = []

    # Increments
    x_increment = (coords2[0] - coords1[0]) / (n_pts2intp + 1)
    y_increment = (coords2[1] - coords1[1]) / (n_pts2intp + 1)

    for pt_count in range(n_pts2intp):
        x_intp_ls.append(coords1[0] + (pt_count+1) * x_increment)
        y_intp_ls.append(coords1[1] + (pt_count+1) * y_increment)

    return x_intp_ls, y_intp_ls


def path_pts2rectified_pts(n_err_pts, path_pts_coords_df):
    if len(path_pts_coords_df) == 2: # path_pts_coords_df only contains two pts
        # Interpolation in the middle
        coords1 = path_pts_coords_df.loc[0].to_numpy()
        coords2 = path_pts_coords_df.loc[1].to_numpy()
        x_intp_ls, y_intp_ls = avg_intp(coords1, coords2, n_pts2intp=n_err_pts)
        rectified_pts_df = pd.DataFrame({"x": x_intp_ls, "y": y_intp_ls})

    elif len(path_pts_coords_df) == 3:
        x_ls = []
        y_ls = []
        # Interpolate the 1st segment
        n_pts2intp = n_err_pts // 2
        coords1 = path_pts_coords_df.loc[0].to_numpy()
        coords2 = path_pts_coords_df.loc[1].to_numpy()
        x_intp_ls, y_intp_ls = avg_intp(coords1, coords2, n_pts2intp)
        # Save to x_ls, y_ls
        x_ls.extend(x_intp_ls)
        y_ls.extend(y_intp_ls)
        # Interpolate the 2nd segment
        n_pts2intp = n_err_pts - n_pts2intp
        coords1 = path_pts_coords_df.loc[1].to_numpy()
        coords2 = path_pts_coords_df.loc[2].to_numpy()
        x_intp_ls, y_intp_ls = avg_intp(coords1, coords2, n_pts2intp)
        # Save to x_ls, y_ls
        x_ls.extend(x_intp_ls)
        y_ls.extend(y_intp_ls)
        rectified_pts_df = pd.DataFrame({"x": x_ls, "y": y_ls})

    else: # More than 3 pts
        # Exclude start and end pts
        # Because those pts may overlap with or may be too close to neighboring correctly matched pts
        shortpath_coords_df = path_pts_coords_df.loc[1:len(path_pts_coords_df)-2, :]

        if n_err_pts <= len(shortpath_coords_df):
            rectified_pts_df = shortpath_coords_df.sample(n_err_pts).sort_index()
        else: # Needs interpolation
            n_intp_pts = n_err_pts - len(shortpath_coords_df)
            # Number of segments that can be interpolated
            n_seg2intp = len(shortpath_coords_df) - 1
            # Number of interpolated pts per segment
            n_pts_per_seg = int(np.ceil(n_intp_pts / n_seg2intp))

            rectified_pts_count = 0
            pt_id = shortpath_coords_df.index[0]
            x_ls = []
            y_ls = []
            while rectified_pts_count < n_err_pts:
                x_ls.append(shortpath_coords_df.loc[pt_id, "x"])
                y_ls.append(shortpath_coords_df.loc[pt_id, "y"])
                # Update
                rectified_pts_count += 1
                if rectified_pts_count >= n_err_pts:
                    break
                # Interpolation
                coords1 = shortpath_coords_df.loc[pt_id].to_numpy()
                coords2 = shortpath_coords_df.loc[pt_id+1].to_numpy()
                x_intp_ls, y_intp_ls = avg_intp(coords1, coords2, n_pts2intp=n_pts_per_seg)
                # Save to x_ls, y_ls
                x_ls.extend(x_intp_ls)
                y_ls.extend(y_intp_ls)
                # Update
                rectified_pts_count += len(x_intp_ls)
                pt_id += 1

            rectified_pts_df = pd.DataFrame({"x": x_ls, "y": y_ls})
            if len(rectified_pts_df) > n_err_pts: # Because of np.ceil in interpolation
                rectified_pts_df = rectified_pts_df.sample(n_err_pts).sort_index()

    return rectified_pts_df


def update_pt_coords_in_traj(traj_gdf, idx_ls, new_pts_df):
    coords = traj_gdf.get_coordinates().reset_index(drop=True)
    coords.iloc[idx_ls] = new_pts_df
    traj_gdf.at[traj_gdf.index[0], "geometry"] = LineString(coords)
    return traj_gdf


def rectify_shortest_path_avg_dist(sample_id, err_seg_ls, samp_trajs_gdf, subgraph_ls):
    if len(err_seg_ls) == 0:
        print("No errors to rectify!")
        return

    # Sample with sample_id
    samp_traj_gdf = samp_trajs_gdf.take([sample_id])
    coords_df = samp_traj_gdf.get_coordinates().reset_index(drop=True)
    subgraph = subgraph_ls[sample_id]
    err_seg = err_seg_ls[sample_id]

    # Rectified trajectory
    rect_traj_gdf = copy.deepcopy(samp_traj_gdf)[["sample_id", "geometry"]]

    # Convert the graph to GeoDataFrames
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)
    # Spatial index
    node_spatial_idx = nodes_gdf.sindex

    for single_err_seg in err_seg:
        # Number of erroneous pts
        n_err_pts = len(single_err_seg)

        # Find two anchor pts
        anchor_pt1, anchor_pt2, pt1_coords_arr, pt2_coords_arr = find_anchor_pts(single_err_seg, coords_df)

        # Find two anchor nodes in the graph
        # Nearest nodes
        anchor_node1 = find_nearest_node(pt1_coords_arr, nodes_gdf, node_spatial_idx, buffer_dist=500)
        anchor_node2 = find_nearest_node(pt2_coords_arr, nodes_gdf, node_spatial_idx, buffer_dist=500)

        if anchor_node1 != anchor_node2: # Two different nodes
            # Shortest-time path
            path_gen = ox.k_shortest_paths(
                G=subgraph, orig=anchor_node1, dest=anchor_node2, k=1, weight="travel_time"
            )  # returns a generator
            try:
                node_path = list(path_gen)[0]  # Convert the generator to a list and obtain the path
                # From node_path to path pt coordinates
                path_pts_coords_df = node_path2path_pts_coords(node_path, edges_gdf, samp_traj_gdf)
                # Rectified pts
                rectified_pts_df = path_pts2rectified_pts(n_err_pts, path_pts_coords_df)
            except: # No path
                print(f"Sample id {sample_id}: No path between nodes {anchor_node1} and {anchor_node2}!")
                continue # Do not update rect_traj_gdf since no rectified_pts_df is generated

        else: # Only find one single anchor node
            # Repeat the node n_err_pts times
            x_ls = [nodes_gdf.loc[anchor_node1, "x"] for i in range(n_err_pts)]
            y_ls = [nodes_gdf.loc[anchor_node1, "y"] for i in range(n_err_pts)]
            rectified_pts_df = pd.DataFrame({"x": x_ls, "y": y_ls})

        # Update rect_traj_gdf
        rect_traj_gdf = update_pt_coords_in_traj(rect_traj_gdf, single_err_seg, rectified_pts_df)

    return rect_traj_gdf
