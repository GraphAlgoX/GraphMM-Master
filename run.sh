# python -u train_gmm_inductive.py --data_dir gmm-data0.25 --dev_id 0 > inductive_results/0.25_log/no_crf.txt 

python -u train_gmm_inductive.py --dev_id 1 --eval_bsize 32 --atten_flag --bi --use_crf --data_dir gmm-data0.25 > inductive_results/0.25_log/crf.txt 