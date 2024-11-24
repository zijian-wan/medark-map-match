import os
import copy
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm, trange

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


def organize_trajs(pts_df, use_traj_id=True):
    if use_traj_id:
        col = "traj_id"
    else:
        col = "Trip"

    n_trajs = pts_df[col].max() + 1

    traj_ptid_ls = []
    for tj_i in trange(n_trajs):
        traj_arr = pts_df.loc[pts_df[col] == tj_i].index.to_numpy()
        traj_ptid_ls.append([tj_i, traj_arr, len(traj_arr)])

    return traj_ptid_ls


def cal_d_hdg(hdg1, hdg2):
    """
    Calculate delta heading (the difference between two headings).

    :param hdg1: float
        current pt, range: [0, 360)
    :param hdg2: float
        previous pt, range: [0, 360)
    :return: d_hdg: float
        range: [-180, 180)
    """
    # Calculate the raw difference
    d_hdg_raw = hdg1 - hdg2

    # Adjust the difference to be within the range [-180, 180)
    d_hdg_adjusted = (d_hdg_raw + 180) % 360 - 180

    # For cases where d_hdg_adjusted is 180, it should be converted to -180
    # because the range is inclusive of 180 at the lower end
    d_hdg = d_hdg_adjusted if d_hdg_adjusted != 180 else -180

    return d_hdg


def normalize(x_arr, orig_range_tp=None, target_range_tp=(0.1, 1)):
    """
    A customized normalization function to transform data from a predefined range to another predefined range.

    :param x_arr: np.array
        1d
    :param orig_range_tp: tuple
        (original_min, originimal_max)
    :param target_range_tp: tuple
        (target_min, target_max)
    :return: x_norm_arr: np.array
        normalized data
    """
    # Original range
    if orig_range_tp is None:
        orig_min = min(x_arr)
        orig_max = max(x_arr)
    else:
        orig_min = orig_range_tp[0]
        orig_max = orig_range_tp[1]

    # Target range
    target_min = target_range_tp[0]
    target_max = target_range_tp[1]

    # Standard deviation
    x_std = (x_arr - orig_min) / (orig_max - orig_min)

    # Normalize
    x_norm_arr = x_std * (target_max - target_min) + target_min

    return x_norm_arr


def create_samples(min_seq_len, max_seq_len, stride, traj_ptid_ls):
    '''
    Create trajectory samples.

    Input:
    ---
        min_seq_len: minimum number of pts in a trajectory sample
        max_seq_len: maximum number of pts in a trajectory sample
        stride
        pts_df
        traj_ptid_ls

    Output:
    ---
        samp_pts_ls: list of samples (pt np.array)
            [pts_arr0, pts_arr1, ...]
    '''
    samp_pts_ls = []

    print(f"Woring on {len(traj_ptid_ls)} trajectories...")

    for tj_i in tqdm(traj_ptid_ls):
        if tj_i[2] < min_seq_len:  # not enough pts in the traj
            continue

        elif tj_i[2] < min([min_seq_len + stride, max_seq_len]):  # no need to split
            samp_pts_ls.append(tj_i[1])

        else:  # needs spliting
            n_remaining_pts = tj_i[2]
            subtj_k = 0

            while n_remaining_pts > min_seq_len:
                start_pt_id = subtj_k*stride

                if n_remaining_pts > max_seq_len:  # cannot add the remaining as a whole
                    samp_pts_ls.append(tj_i[1][start_pt_id:(start_pt_id+max_seq_len)])

                else:  # add all remaining pts
                    samp_pts_ls.append(tj_i[1][start_pt_id:])

                subtj_k += 1
                n_remaining_pts -= stride

    print(f"Number of trajectory samples: {len(samp_pts_ls)}")

    return samp_pts_ls


def calc_pos_weight(Y_ls):
    """
    Calculate the pos_weight for BCEWithLogitsLoss.

    Output:
    ---
    pos_weight: torch.Tensor
    pos_weight = n_negative / n_positive
    """
    zero_count = 0
    one_count = 0

    for sample_i in Y_ls:
        sample_n_pts = len(sample_i)
        sample_one_count = int(sum(sample_i))
        sample_zero_count = sample_n_pts - sample_one_count

        one_count += sample_one_count
        zero_count += sample_zero_count

    weight_for_class_1 = zero_count / one_count

    return weight_for_class_1


def tensorize_input(X_ls, Y_ls):
    X_ts_ls = [torch.tensor(x, dtype=torch.float) for x in X_ls]
    Y_ts_ls = [torch.tensor(y, dtype=torch.float) for y in Y_ls]
    return X_ts_ls, Y_ts_ls


class MatchErrorDataset(Dataset):
    def __init__(self, X_ls, Y_ls):
        self.X_ts_ls, self.Y_ts_ls = tensorize_input(X_ls, Y_ls)

    def __len__(self):
        return len(self.X_ts_ls)

    def __getitem__(self, i):
        return self.X_ts_ls[i], self.Y_ts_ls[i]


def collate_fn(batch):
    X_batch, Y_batch = zip(*batch)

    # Calculate lengths of sequences before padding
    X_lengths = [len(x) for x in X_batch]
    Y_lengths = [len(y) for y in Y_batch]

    # Pad the sequences in the batch
    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=-1)
    Y_padded = pad_sequence(Y_batch, batch_first=True, padding_value=-1)  # Using -1 for label padding

    # Create a mask for the labels
    # Padding mask for Transformer
    padding_mask = (Y_padded == -1)  # positions with the value of True will be ignored
    # Nonpadding mask for loss
    nonpad_mask = ~padding_mask  # ~ negate the tensor

    return X_padded, Y_padded, nonpad_mask, padding_mask, X_lengths, Y_lengths


def train_val_test_set(X_ls, Y_ls, train_pct=0.7):
    # Train, val, test split
    val_test_pct = 1 - train_pct
    test_pct = 0.33  # test/(val+test)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_ls, Y_ls, test_size=val_test_pct
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_test, Y_test, test_size=test_pct
    )

    print("Train, validation, test split done! Next: create PyTorch datasets")
    # PyTorch dataset
    train_set = MatchErrorDataset(X_train, Y_train)
    val_set = MatchErrorDataset(X_val, Y_val)
    test_set = MatchErrorDataset(X_test, Y_test)

    return train_set, val_set, test_set


def calc_eval_metrics(y_pred, y_gt):
    '''
    Calculate evaluation metrics
        Cohen's kappa coefficient

    Input:
        y_pred: torch.Tensor
            prediction (non-padded, flattened)
        y_gt: torch.Tensor
            ground truth (non-padded, flattened)
    '''
    y_pred_arr = y_pred.data.cpu().numpy()
    y_gt_arr = y_gt.data.cpu().numpy()

    kappa = cohen_kappa_score(y_pred_arr, y_gt_arr)
    accuracy = accuracy_score(y_pred_arr, y_gt_arr)

    return kappa, accuracy


def sample_subgraph(graph, samp_trajs_gdf, buffer_dist):
    """
    Get a subgraph within the neighborhood a sample trajectory.

    :param graph: networkx.graph
        Road network graph (obtained from osmnx)
    :param samp_trajs_gdf: geopandas.GeoDataFrame
        GeoDataFrame containing trajectory samples
    :return: subgraph_ls: list of networkx.graph
        List of local road network graph
    """
    # Create buffers of trajectory samples
    print("Creating buffers of sample trajectories created with buffer distance:", buffer_dist)
    for samp_i in trange(len(samp_trajs_gdf)):
        samp_trajs_gdf.loc[samp_i, "buffer"] = samp_trajs_gdf.loc[samp_i, "geometry"].buffer(distance=buffer_dist)

    # Convert the graph to a GeoDataFrame
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
    print("Graph converted to nodes_gdf, edges_gdf!")

    # Spatial index
    spatial_idx = nodes_gdf.sindex
    print("Spatial index created! Next, creating subgraphs...")

    subgraph_ls = []
    # Iterate through each buffered trajectory to find intersecting nodes
    for buffer in tqdm(samp_trajs_gdf["buffer"]):
        # Find nodes within the buffer
        possible_nodes_ls = list(spatial_idx.intersection(buffer.bounds))
        nodes_within_buffer = nodes_gdf.iloc[possible_nodes_ls]

        # Filter nodes by actual intersection with the buffer polygon
        nodes_within_buffer_df = nodes_within_buffer[nodes_within_buffer.intersects(buffer)]

        # Extract subgraph that includes these nodes
        subgraph = graph.subgraph(nodes_within_buffer_df.index)
        subgraph_ls.append(subgraph)

    return subgraph_ls
