# python peeling_new_trace_road.py
# echo "finished -1"
# python build_road_graph.py
# echo "finished 0"
# python reclean_trace.py
# echo "finished 1"
# python trace_prepare.py
# echo "finished 2"
# python rewrite_time.py

# python data_process.py


# python build_trace_graph.py
# echo "finished 3"
# cd ../utils
# python maproad2grid.py
# echo "finished 4"
# python build_grid_road_matrix.py
# echo "finished 5"


# inductive-test
# python get_downsample_train.py

# python build_train_trace_graph.py

cd ../utils
python maproad2grid_train.py
python build_train_grid_road_matrix.py
