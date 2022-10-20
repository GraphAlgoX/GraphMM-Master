# -> peeling_new_trace_road.py -> build_road_graph.py -> 
# reclean_trace.py -> trace_prepare.py -> build_trace_grpah.py -> utils/maproad2grid.py
# -> utils/build_grid_road_matrix.py
# -> data_process.py

python peeling_new_trace_road.py
echo "finished -1"
python build_road_graph.py
echo "finished 0"
python reclean_trace.py
echo "finished 1"
python trace_prepare.py
echo "finished 2"
python build_trace_graph.py
echo "finished 3"
cd ../utils
python maproad2grid.py
echo "finished 4"
python build_grid_road_matrix.py
echo "finished 5"
cd ../utils_gq
python rewrite_time.py
echo "finished 6"
python data_process.py
echo "finished 7"