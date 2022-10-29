cd ../utils_gq
for num in {0..99}  
do  
echo $num  
# python road_graph.py $num
# python trace_graph.py $num
# python maproad2grid.py $num
# python build_grid_road_matrix.py $num
python build_A_100.py $num
done


