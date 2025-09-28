conda activate npn

# Prepare geo-split nuScenes dataset
# NOTE: must change info-prefix to sth other than 'nuscenes' to avoid overwriting
python tools/data_converter/nuscenes_converter.py --data-root ./data/nuscenes --info-prefix nuScences_map --geosplit
python project/neural_map_prior/data_utils/nusc_city_infos.py --data-root ./data/nuscenes --info-prefix nuScences_map

# Training the model
bash tools/dist_train.sh project/configs/bevformer_30m_60m.py 8
bash tools/dist_train.sh project/configs/neural_map_prior_bevformer_30m_60m.py 8