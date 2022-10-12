# -> peeling_new_trace_road.py -> build_road_graph.py -> 
# reclean_trace.py -> trace_prepare.py -> build_trace_grpah.py -> utils/maproad2grid.py
# -> utils/build_grid_road_matrix.py
# -> data_process.py

python peeling_new_trace_road.py
python build_road_graph.py
python reclean_trace.py
python trace_prepare.py
python build_trace_graph.py
cd ../utils
python maproad2grid.py
python build_grid_road_matrix.py
cd ../utils_gq
python data_process.py