# GraphMM: Graph-based Vehicular Map Matching by Leveraging Trajectory and Road Correlations

This is a Python implementation of 

> GraphMM: Graph-based Vehicular Map Matching by Leveraging Trajectory and Road Correlations
>
> Yu Liu, Qian Ge, Wei Luo, Qiang Huang, Lei Zou, Haixu Wang, Xin Li, Chang Liu
>
> TKDE 2023 [PDF](https://ieeexplore.ieee.org/document/10158034)


## Full Dataset Available
   The full dataset containing 64k trajectories has now been uploaded and can be accessed and downloaded via the link [trace.txt](https://drive.google.com/file/d/1L8IeRn9GTWVHm6qlwrK_oTsdGq-UC3At/view?usp=sharing). If you want to conduct experiments with this dataset, you need to replace the original `trace.txt` in the `data` directory with this file and re-run the `data_preprocess.sh` script. The `road.txt` does not need to be changed.




## GraphMM Architecture

<p align="center"><img src="./images/GraphMM.jpg"></p>



## Requirements

- `Python: 3.7.13`
- `Pytorch: 1.10.1`
- `PyG: 2.0.4`
- `networkx: 2.6.3`
- `nni: 2.9 # https://github.com/microsoft/nni`



## Run Examples

Data preprocess:
```
bash data_preprocess.sh
```

Training gmm with crf:

>modify the root_path in `config.py` to the absolute path where `road.txt` in

```shell
# way 1
python train_gmm.py --dev_id 0 --batch_size 32 --eval_bsize 32 --atten_flag --bi --use_crf --root_path {your data dir} --downsample_rate {your downsample_rate}

# way 2
# first modify your configuration and hyper-parameter search space, then run following command
nnictl create --config exp_local.yml -p {specified port, default 8080}
```

> The batch_size and eval_bsize can be setted based on your GPU memory size!

Training gmm without crf:

```shell
# way 1
python train_gmm.py --dev_id 0 --batch_size 256 --eval_bsize 256 --atten_flag --bi --root_path {your data dir} --downsample_rate {your downsample_rate}

# way 2
nnictl create --config exp_local.yml -p {specified port, default 8080}
```

Evaluate gmm with crf:

```shell
# First set the ckpt_path with your trained model, then run
python test.py --atten_flag --bi --use_crf --root_path {your data dir} --eval_bsize 32 --downsample_rate {your downsample_rate}
```

Evaluate gmm without crf:

```shell
python test.py --atten_flag --bi --root_path {your data dir} --eval_bsize 256 --downsample_rate {your downsample_rate}
```

The hyper parameters in `config.py` are tuned for Tencent dataset with 64K trajectories, and may not be suitable for other settings.



## Dataset

### Trajectory

The original trajectory data is in the file `./data/trace.txt`. Each trajectory begins with a `# `line, where `#` is followed by the trajectory's ID. Each subsequent line represents a trajectory point.

Meaning of each line (trajectory point):

```
2022/03/03 18:24:23, # datetime
40.07856,116.30941, # longitude and latitude coordinates
744, # matched road segment id
40.07879261945333,116.3096391218959, 0,0.7528351264902587,0.7528351264902587 # negligible, not actually used
```

### Road Graph

The original road network data is in the file `./data/road.txt`, where each line represents a road segment.

Meaning of each line (road segment):

```
4736 # road segment id
13535	1825 # start and end point ids of the road segment
2	3	2 # attributes
116.3414171 40.0795484,116.3420562 40.0801667 # start and end longitude and latitude of the  road segment
| # delimiter, followed by intersecting road segment information, separated by ';'
4725,13535,116.3414171 40.0795484; # Intersecting road segment ID, intersecting endpoints, longitude and latitude
6135,13535,116.3414171 40.0795484;
5642,13535,116.3414171 40.0795484;
8173,13535,116.3414171 40.0795484
```



## Citations

If you find this repo useful and would like to cite it, citing our paper as following will be appropriate:

```
@ARTICLE{10158034,
  author={Liu, Yu and Ge, Qian and Luo, Wei and Huang, Qiang and Zou, Lei and Wang, Haixu and Li, Xin and Liu, Chang},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={GraphMM: Graph-Based Vehicular Map Matching by Leveraging Trajectory and Road Correlations}, 
  year={2024},
  volume={36},
  number={1},
  pages={184-198},
  keywords={Trajectory;Roads;Hidden Markov models;Correlation;Decoding;Training;Predictive models;Conditional model;graph neural network;inductive capability;map matching;trajectory and road correlations},
  doi={10.1109/TKDE.2023.3287739}}
```



## Note

Our crf is modified based on [pytorch-crf](https://pytorch-crf.readthedocs.io/en/stable/).
