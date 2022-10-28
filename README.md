# GMM: Graph-based Vehicular Map Matching by Leveraging Trajectory and Road Correlations

The repository is the official Pytorch implementation of experiments in GMM.



## Experiment Configuration

```
Python: 3.7.13
Pytorch: 1.10.1
PyG: 2.0.4
networkx: 2.6.3
nni: 2.9 # https://github.com/microsoft/nni
```



## Run Examples

Training gmm with crf:

```shell
# way 1
python train_gmm.py --dev_id 0 --batch_size 32 --eval_bsize 32 --atten_flag --bi --use_crf --data_dir {your data dir}

# way 2
# first modify your configuration and hyper-parameter search space, then run following command
nnictl create --config exp_local.yml
```

> The batch_size and eval_bsize can be setted based on your GPU memory size!

Training gmm without crf:

```shell
# way 1
python train_gmm.py --dev_id 0 --batch_size 256 --eval_bsize 256 --atten_flag --bi --data_dir {your data dir}

# way 2
nnictl create --config exp_local.yml
```

Evaluate gmm with crf:

```shell
# First set the ckpt_path with your trained model, then run
python test.py --atten_flag --bi --use_crf --data_dir {your data dir} --eval_bsize 32
```

Evaluate gmm without crf:

```shell
python test.py --atten_flag --bi --data_dir {your data dir} --eval_bsize 256
```



## Note

Our crf is modified based on [pytorch-crf](https://pytorch-crf.readthedocs.io/en/stable/).