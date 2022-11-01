cd data_preprocess

python build_road_graph.py
python build_A.py

downsample_rate=0.125

python data_process.py $downsample_rate
python build_trace_graph.py $downsample_rate
python maproad2grid.py $downsample_rate
python build_grid_road_matrix.py $downsample_rate