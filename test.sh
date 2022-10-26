# python -u train_gmm_inductive.py --data_dir gmm-data0.25 --dev_id 0 > inductive_results/0.25_log/no_crf.txt 

# python -u train_gmm_inductive.py --dev_id 0 --eval_bsize 256 --atten_flag --bi --data_dir gmm-data0.25 > inductive_results/0.25_log/no_crf.txt 

python -u test_inductive.py --eval_bsize 256 --atten_flag --bi --data_dir gmm-data0.25 > inductive_results/0.25_log/test_no_crf.txt