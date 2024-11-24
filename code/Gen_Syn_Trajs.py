import os
import pickle
import multiprocessing
from tqdm import tqdm, trange

import pandas as pd

import osmnx as ox
import geopandas as gpd

from node_paths import multiproc_rand_gen_k_paths
from edge_paths import multiproc_node2edge_paths
from pt_paths import multiproc_edge2pt_paths
from track_pts import multiproc_pt2track

if __name__ == "__main__":
    proj_dir = "<YOUR_PROJECT_DIRECTORY>"

    # Load road network graph
    graph_name = "aa_road_graph_drive_service_bbox_time_speed_bearing.graphml"  # Ann Arbor
    # graph_name = "la_road_graph_drive_service_bbox_time_speed_bearing.graphml"  # Los Angeles
    roadnet_graph = ox.load_graphml(
        os.path.join(proj_dir, graph_name)
    )
    print("Road graph loaded!")

    # Graph to gdf
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(roadnet_graph)
    print("nodes_gdf, edges_gdf created!")

    # --------------------Node paths--------------------
    n_times = 10000  # Number of times to generate a pair of nodes
    k = 10  # Number of paths for each pair of nodes in the order from the shortest time to the longest
    n_nodes_filter = 100  # Length filter to remove short trajectories
    print(f"Next, generate node paths, n_times: {n_times}, k: {k}, n_nodes_filter: {n_nodes_filter}")
    # Generate node paths
    node_paths_ls = multiproc_rand_gen_k_paths(
        nodes_gdf, n_times=n_times, k=k, graph=roadnet_graph, minimize_time=True, n_nodes_filter=n_nodes_filter
    )
    print("Number of paths generated:", len(node_paths_ls))

    # Save node paths to pickle
    pk_name = f"node_paths_ntimes{int(n_times/1000)}k_k{k}_nnodesfilter{n_nodes_filter}.pkl"
    with open(os.path.join(proj_dir, pk_name), 'wb') as f:
        pickle.dump(node_paths_ls, f)
    print("node_paths_ls saved to pickle:", pk_name)

    # --------------------Edge paths--------------------
    # Node paths to edge paths
    edge_paths_ls = multiproc_node2edge_paths(node_paths_ls, edges_gdf)

    # Save edge paths to pickle
    pk_name = f"edge_paths_ntimes{int(n_times/1000)}k_k{k}_nnodesfilter{n_nodes_filter}.pkl"
    with open(os.path.join(proj_dir, pk_name), 'wb') as f:
        pickle.dump(edge_paths_ls, f)
    print("edge_paths_ls saved to pickle:", pk_name)

    # --------------------Point paths--------------------
    # Edge paths to pt paths
    fixed_dist = 20  # Distance to generate pts (densify the edges)
    pt_paths_ls = multiproc_edge2pt_paths(edge_paths_ls, fixed_distance=fixed_dist)

    # Save pt paths to pickle
    pk_name = f"pt_paths_ntimes{int(n_times/1000)}k_k{k}_nnodesfilter{n_nodes_filter}_fixeddist{fixed_dist}.pkl"
    with open(os.path.join(proj_dir, pk_name), 'wb') as f:
        pickle.dump(pt_paths_ls, f)
    print("pt_paths_ls saved to pickle:", pk_name)

    # --------------------Point selection and error introduction--------------------
    # Pt paths to tracking pts
    samp_int_mean_ls = [2, 5, 10, 30]  # List of mean sampling intervals
    for samp_int_mean in samp_int_mean_ls:
        print("\nNow working with samp_int_mean:", samp_int_mean)

        error_rate = 0.2
        track_pts_combined_gdf = multiproc_pt2track(
            pt_paths_ls, fixed_dist=fixed_dist, edges_gdf=edges_gdf, samp_int_mean=samp_int_mean, error_rate=error_rate
        )

        # Save track pts to pickle
        pk_name = (
            f"track_pts_tint{samp_int_mean}_ntimes{int(n_times/1000)}k_fixeddist{fixed_dist}"
            + f"_error0{int(error_rate*10)}.pkl"
        )
        with open(os.path.join(proj_dir, pk_name), 'wb') as f:
            pickle.dump(track_pts_combined_gdf, f)
        print("Number of tracking pts:", len(track_pts_combined_gdf))
        print("track_pts_combined_gdf saved to pickle:", pk_name)

        # Save as shapefiles if "save_shp = True", projection espg has to be set
        save_shp = False
        if save_shp:  # save as shapefiles
            print("Saving as shapefiles...")
            # Set crs
            track_pts_combined_gdf = track_pts_combined_gdf.set_crs("<YOUR ESPG CODE>")
            # Save to shapefile
            shp_name = (
                    f"SyntheticTrackPts_Attr_tint{samp_int_mean}_NTimes{int(n_times/1000)}k_FixedDist{fixed_dist}"
                    + f"_Error0{int(error_rate*10)}.shp"
            )
            track_pts_combined_gdf.to_file(os.path.join(proj_dir, shp_name))
            print("Saved to shapefile:", shp_name)

