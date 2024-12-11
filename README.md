# Medark: A map-matching error detection and rectification framework for vehicle trajectories
Medark is a framework that leverages LSTM to detect map-matching errors in vehicle trajectories and subsequently rectifies these errors to produce trajectories accurately aligned with the road network.

![](figures/LSTM_Architecture.png)

**Citation info:**
> Wan, Z., & Dodge, S. (2024). Medark: a map-matching error detection and rectification framework for vehicle trajectories. International Journal of Geographical Information Science, 1–28. https://doi.org/10.1080/13658816.2024.2436482

## Abstract
The widespread use of Global Navigation Satellite System (GNSS) trackers has significantly enhanced the availability of vehicle tracking data, providing researchers with critical insights into human mobility. Map matching, a key preprocessing step in movement analysis, matches vehicle tracking data to road segments but often introduces errors that can affect subsequent analyses. Existing map-matching methods, categorized into classic spatially generalizable methods and region-specific deep-learning-based methods, both have limitations. Region-specific deep learning methods, while more accurate, do not transfer well across different geographical regions. Moreover, the temporal adaptability of both approaches---their ability to handle GNSS signals of varying sampling intervals---has not been thoroughly examined. To overcome these limitations, we introduce Medark, a novel framework for detecting and rectifying errors in classic map-matching methods while preserving spatial generalizability. The proposed model is trained using a transfer-learning approach with synthetic trajectories generated in Ann Arbor and Los Angeles at various sampling intervals and a real vehicle trajectory dataset from Ann Arbor. Our experimental results validate the effectiveness of Medark. This framework can be integrated with any map-matching method to improve accuracy and produce high-quality trajectories for further analysis.

## Descriptions
### Code
1. Road network \
   Run `Get_OSM_RoadNet.ipynb` (with your study area information) to obtain the road network graph for your study area.
2. Synthetic trajectories \
   Synthetic trajectories are generated in this study to facilitate transfer learning of the LSTM-based map-matching error detection model.
   Run `Gen_Syn_Trajs.py` (with your project information) to generate synthetic trajectories. Our synthetic trajectory generation method consists of four stages: route generation, point generation, point
selection, and error introduction in map-matched tracking points, which correspond to functions in `node_paths.py`, `edge_paths.py`, `pt_paths.py`, and `track_pts.py`, respectively.
3. Training samples \
   Run `Create_Training_Samples.ipynb` to create training samples with attributes and the structure detailed in the paper.
4. Map-matching error detection model \
   Run `Error_Detect_Models.ipynb` to train the proposed LSTM-based model from `lstm_error_detect_model.py`. A Transformer-based model `transformer_error_detect_model.py` is also provided for comparison.
5. Error rectification \
   Run `Rectify_Errors.ipynb` to rectify detected errors using a plausible route.

### Data
`labeled184_pts_traj_ptid.pkl` is the pickled data of manually labeled 184 trajectories. Each erroneous point (map-matching error) is labeled 1, while other tracking points are labeled 0. \
It can be loaded as \
`pk_name = "labeled184_pts_traj_ptid.pkl" 
with open( 
    os.path.join(proj_dir, "Data", pk_name), 'rb' 
) as my_file_obj: 
    pts_gdf, traj_ptid_ls = pickle.load(my_file_obj)`
