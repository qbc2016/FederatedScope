## Configurations
We summarize all the customizable configurations here.

### Data
The configurations related to the data/dataset are defined in `cfg_data.py`.

| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `data.root` | (string) 'data' | <font size=1> The folder where the data file located. `data.root` would be used together with `data.type` to load the dataset. </font> | - |
| `data.type` | (string) 'toy' | <font size=1>Dataset name</font> | CV: 'femnist', 'celeba' ; NLP: 'shakespeare', 'subreddit', 'twitter'; Graph: 'cora', 'citeseer', 'pubmed', 'dblp_conf', 'dblp_org', 'csbm', 'epinions', 'ciao', 'fb15k-237', 'wn18', 'fb15k' , 'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1', 'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI', 'HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP', 'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX', 'graph_multi_domain_mol', 'graph_multi_domain_small', 'graph_multi_domain_mix', 'graph_multi_domain_biochem'; MF: 'vflmovielens1m', 'vflmovielens10m', 'hflmovielens1m', 'hflmovielens10m', 'vflnetflix', 'hflnetflix'; Tabular: 'toy', 'synthetic'; External dataset: 'DNAME@torchvision', 'DNAME@torchtext', 'DNAME@huggingface_datasets', 'DNAME@openml'. |
| `data.args` | (list) [] | <font size=1>Args for the external dataset</font> | Used for external dataset, eg. `[{'download': False}]` |
| `data.save_data` | (bool) False | <font size=1> Whether to save the generated toy data </font> | - |
| `data.splitter` | (string) '' | <font size=1>Splitter name for standalone dataset</font> | Generic splitter: 'lda'; Graph splitter: 'louvain', 'random', 'rel_type', 'graph_type', 'scaffold', 'scaffold_lda', 'rand_chunk' |
| `data.splitter_args` | (list) [] | <font size=1>Args for splitter.</font> | Used for splitter, eg. `[{'alpha': 0.5}]` |
| `data.transform` | (list) [] | <font size=1>Transform for x of data</font> | Used in `get_item` in torch.dataset, eg. `[['ToTensor'], ['Normalize', {'mean': [0.1307], 'std': [0.3081]}]]` |
| `data.target_transform` | (list) [] | <font size=1>Transform for y of data</font> | Use as `data.transform` |
| `data.pre_transform` | (list) [] | <font size=1>Pre_transform for `torch_geometric` dataset</font> | Use as `data.transform` |
| `data.batch_size` | (int) 64 | <font size=1>batch_size for DataLoader</font> | - |
| `data.drop_last` | (bool) False | <font size=1>Whether drop last batch (if the number of last batch is smaller than batch_size) in DataLoader</font> | - |
| `data.sizes` | (list) [10, 5] | <font size=1>Sample size for graph DataLoader</font> | The length of `data.sizes` must meet the layer of GNN models. |
| `data.shuffle` | (bool) True | <font size=1>Shuffle train DataLoader</font> | - |
| `data.server_holds_all` | (bool) False | <font size=1>Only use in global mode, whether the server (workers with idx 0) holds all data, useful in global training/evaluation case</font> | - |
| `data.subsample` | (float) 1.0 | <font size=1> Only used in LEAF datasets, subsample clients from all clients</font> | - |
| `data.splits` | (list) [0.8, 0.1, 0.1] | <font size=1>Train, valid, test splits</font> | - |
| `data.consistent_label_distribution` | (bool) False | <font size=1>Make label distribution of train/val/test set over clients keep consistent during splitting</font> | - |
| `data.cSBM_phi` | (list) [0.5, 0.5, 0.5] | <font size=1>Phi for cSBM graph dataset</font> | - |
| `data.loader` | (string) '' | <font size=1>Graph sample name, used in minibatch trainer</font> | 'graphsaint-rw': use `GraphSAINTRandomWalkSampler` as DataLoader; 'neighbor': use `NeighborSampler` as DataLoader. |
| `data.num_workers` | (int) 0 | <font size=1>num_workers in DataLoader</font> | - |
| `data.graphsaint.walk_length` | (int) 2 | <font size=1>The length of each random walk in graphsaint.</font> | - |
| `data.graphsaint.num_steps` | (int) 30 | <font size=1>The number of iterations per epoch in graphsaint.</font> | - |
| `cfg.data.quadratic.dim` | (int) 1 | <font size=1>Dim of synthetic quadratic  dataset</font> | - |
| `cfg.data.quadratic.min_curv` | (float) 0.02 | <font size=1>Min_curve of synthetic quadratic  dataset</font> | - |
| `cfg.data.quadratic.max_curv` | (float) 12.5 | <font size=1>Max_cur of synthetic quadratic  dataset</font> | - |


### Model

The configurations related to the model are defined in `cfg_model.py`.

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `model.model_num_per_trainer` |     (int) 1     | Number of model per trainer |                                                                 some methods may leverage more                                                                 |
| `model.type` | (string) 'lr' | The model name used in FL | CV: 'convnet2', 'convnet5', 'vgg11', 'lr'; NLP: 'LSTM', 'MODEL@transformers'; Graph:  'gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn';  Tabular: 'mlp', 'lr', 'quadratic'; MF: 'vmfnet', 'hmfnet' |
| `model.use_bias` | (bool) True | Whether use bias in lr model | - |
| `model.task` | (string) 'node' | The task type of model, the default is `Classification` | NLP: 'PreTraining', 'QuestionAnswering', 'SequenceClassification', 'TokenClassification', 'Auto', 'WithLMHead'; Graph: 'NodeClassification', 'NodeRegression', 'LinkClassification', 'LinkRegression', 'GraphClassification', 'GraphRegression', |
| `model.hidden` | (int) 256 | Hidden layer dimension | - |
| `model.dropout` | (float) 0.5 | Dropout ratio | - |
| `model.in_channels` | (int) 0 | Input channels dimension | If 0, model will be built by `data.shape` |
| `model.out_channels` | (int) 1 | Output channels dimension | - |
| `model.layer` | (int) 2 | Model layer | - |
| `model.graph_pooling` | (string) 'mean' | Graph pooling method in graph-level task | 'add', 'mean' or 'max' |
| `model.embed_size` | (int) 8 | `embed_size` in LSTM | - |
| `model.num_item` | (int) 0 | Number of items in MF. | It will be overwritten by the real value of the dataset. |
| `model.num_user` | (int) 0 | Number of users in MF. | It will be overwritten by the real value of the dataset. |

#### Criterion

|            Name            | (Type) Default Value |                   Description                    |                                                                                         Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `criterion.type` |     (string) 'MSELoss'     | Criterion type |                                                                                           Chosen from https://pytorch.org/docs/stable/nn.html#loss-functions , eg. 'CrossEntropyLoss', 'L1Loss', etc.                                                                                            |

#### regularizer

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `regularizer.type` |     (string) ' '     | The type of the regularizer |                                                                 Chosen from [`proximal_regularizer`]                                                                 |
| `regularizer.mu` | (float) 0 | The factor that controls the loss of the regularization term | - |


### Federated Algorithms 
The configurations related to specific federated algorithms, which are 
defined in 
`cfg_fl_algo.py`.

#### `fedopt`: for FedOpt algorithm
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `fedopt.use` | (bool) False | <font size=1> Whether to run FL courses with FedOpt algorithm. </font> | If False, all the related configurations (cfg.fedopt.xxx) would not take effect. |
| `fedopt.optimizer.type` | (string) 'SGD' | <font size=1> The type of optimizer used for FedOpt algorithm. </font> | Currently we support all optimizers build in PyTorch (The modules under torch.optim). |
| `fedopt.optimizer.lr` | (float) 0.1 | <font size=1> The learning rate used in for FedOpt optimizer. </font> | - |
#### `fedprox`: for FedProx algorithm 
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `fedprox.use` | (bool) False | <font size=1> Whether to run FL courses with FedProx algorithm. </font> | If False, all the related configurations (cfg.fedprox.xxx) would not take effect. |
| `fedprox.mu` | (float) 0.0 | <font size=1> The hyper-parameter $\mu$ used in FedProx algorithm. </font> | - |
#### `personalization`: for personalization algorithms
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `personalization.local_param` | (list of str) [] | <font size=1> The client-distinct local param names, e.g., ['pre', 'bn'] </font> | - |
| `personalization.share_non_trainable_para` | (bool) False | <font size=1> Whether transmit non-trainable parameters between FL participants </font> | - |
| `personalization.local_update_steps` | (int) -1 | <font size=1> The local training steps for personalized models </font> | By default, -1 indicates that the local model steps will be set to be the same as the valid `train.local_update_steps` |
| `personalization.regular_weight` | (float) 0.1 | <font size=1> The regularization factor used for model para regularization methods such as Ditto and pFedMe. </font> | The smaller the regular_weight is, the stronger emphasising on personalized model. |
| `personalization.lr` | (float) 0.0 | <font size=1> The personalized learning rate used in personalized FL algorithms. </font> | The default value 0.0 indicates that the value will be set to be the same as `train.optimizer.lr` in case of users have not specify a valid `personalization.lr` |
| `personalization.K` | (int) 5 | <font size=1> The local approximation steps for pFedMe. </font> | - |
| `personalization.beta` | (float) 5 | <font size=1> The average moving parameter for pFedMe. </font> | - |
#### `fedsageplus`: for fedsageplus algorithm
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `fedsageplus.num_pred` | (int) 5 | <font size=1> Number of nodes generated by the generator </font> | - |
| `fedsageplus.gen_hidden` | (int) 128 | <font size=1> Hidden layer dimension of generator </font> | - |
| `fedsageplus.hide_portion` | (float) 0.5 | <font size=1> Hide graph portion </font> | - |
| `fedsageplus.fedgen_epoch` | (int) 200 | <font size=1> Federated training round for generator </font> | - |
| `fedsageplus.loc_epoch` | (int) 1 | <font size=1> Local pre-train round for generator </font> | - |
| `fedsageplus.a` | (float) 1.0 | <font size=1> Coefficient for criterion number of missing node </font> | - |
| `fedsageplus.b` | (float) 1.0 | <font size=1> Coefficient for criterion feature </font> | - |
| `fedsageplus.c` | (float) 1.0 | <font size=1> Coefficient for criterion classification </font> | - |
#### `gcflplus`: for gcflplus algorithm
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `gcflplus.EPS_1` | (float) 0.05 | <font size=1> Bound for mean_norm </font> | - |
| `gcflplus.EPS_2` | (float) 0.1 | <font size=1> Bound for max_norm </font> | - |
| `gcflplus.seq_length` | (int) 5 | <font size=1> Length of the gradient sequence </font> | - |
| `gcflplus.standardize` | (bool) False | <font size=1> Whether standardized dtw_distances </font> | - |
#### `flitplus`: for flitplus algorithm
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `flitplus.tmpFed` | (float) 0.5 | <font size=1>  gamma in focal loss (Eq.4) </font> | - |
| `flitplus.lambdavat` | (float) 0.5 | <font size=1> lambda in phi (Eq.10) </font> | - |
| `flitplus.factor_ema` | (float) 0.8 | <font size=1> beta in omega (Eq.12) </font> | - |
| `flitplus.weightReg` | (float) 1.0 | <font size=1> balance lossLocalLabel and lossLocalVAT </font> | - |


### Federated training
The configurations related to federated training are defined in `cfg_training.py`.
Considering it's infeasible to list all the potential arguments for optimizers and schedulers, we allow the users to add new parameters directly under the corresponding namespace. 
For example, we haven't defined the argument `train.optimizer.weight_decay` in `cfg_training.py`, but the users are allowed directly use it. 
If the optimizer doesn't require the argument named `weight_decay`, an error will be raised. 

#### Local training
The following configurations are related to the local training. 

|            Name            | (Type) Default Value |                   Description                    |                                                                                         Note                                                                                         |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `train.local_update_steps` |       (int) 1        |       The number of local training steps.        |                                                                                          -                                                                                           |
|   `train.batch_or_epoch`   |   (string) 'batch'   |           The type of local training.            |               `train.batch_or_epoch` specifies the unit that `train.local_update_steps` adopts. All new parameters will be used as arguments for the chosen optimizer.               |
|     `train.optimizer`      |          -           |                        -                         |                     You can add new parameters under `train.optimizer` according to the optimizer, e.g., you can set momentum by `cfg.train.optimizer.momentum`.                     |
|   `train.optimizer.type`   |    (string) 'SGD'    |  The type of optimizer used in local training.   |                                               Currently we support all optimizers build in PyTorch (The modules under `torch.optim`).                                                |
| `train.optimizer.lr` |     (float) 0.1      |  The learning rate used in the local training.   |                                                                                          -                                                                                           |
|     `train.scheduler`      |          -           |                        -                         | Similar with `train.optimizer`, you can add new parameters as you need, e.g., `train.scheduler.step_size=10`. All new parameters will be used as arguments for the chosen scheduler. |
| `train.scheduler.type` |     (string) ''      | The type of the scheduler used in local training |                                         Currently we support all schedulers build in PyTorch (The modules under `torch.optim.lr_scheduler`).                                         |

#### Fine tuning
The following configurations are related to the fine tuning.

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `finetune.before_eval` |     (bool) False     |      Indicator of fintune before evaluation      | If `True`, the clients will fine tune its model before each evaluation. Note the fine tuning is only conducted before evaluation and won't influence the upload weights in each round. |
| `finetune.local_update_steps` |       (int) 1        |       The number of local fine tune steps        |                                                                                           -                                                                                            |
| `finetune.batch_or_epoch` |   (string) `batch`   |          The type of local fine tuning.          |                                   Similar with `train.batch_or_epoch`, `finetune.batch_or_epoch` specifies the unit of `finetune.local_update_steps`                                   |
| `finetune.optimizer` |          -           |                        -                         |            You can add new parameters under `finetune.optimizer` according to the type of optimizer. All new parameters will be used as arguments for the chosen optimizer.            |
| `finetune.optimizer.type` |    (string) 'SGD'    |  The type of the optimizer used in fine tuning.  |                                                Currently we support all optimizers build in PyTorch (The modules under `torch.optim`).                                                 |
| `finetune.optimizer.lr` |     (float) 0.1      |   The learning rate used in local fine tuning    |                                                                                           -                                                                                            |
| `finetune.scheduler` |          -           | - |                   Similar with `train.scheduler`, you can add new parameters as you need, and all new parameters will be used as arguments for the chosen scheduler.                   |

#### Grad Clipping
The following configurations are related to the grad clipping.  

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `grad.grad_clip` |     (float) -1.0     | The threshold used in gradient clipping. |                                                                 `grad.grad_clip < 0` means we don't clip the gradient.                                                                 |

#### Early Stop

|                   Name                   | (Type) Default Value |                   Description                    |                                                                                         Note                                                                                          |
|:----------------------------------------:|:--------------------:|:------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          `early_stop.patience`           | (int) 5 |  How long to wait after last time the monitored metric improved. |                        Note that the actual_checking_round = `early_step.patience` * `eval.freq`. To disable the early stop, set the `early_stop.patience` <=0                        |
|            `early_stop.delta`            | (float) 0. |  Minimum change in the monitored metric to indicate a improvement. |                                                                                           -                                                                                           |
|   `early_stop.improve_indicaator_mode`   | (string) 'best' | Early stop when there is no improvement within the last `early_step.patience` rounds, in ['mean', 'best'] |                                                                             Chosen from 'mean' or 'best'                                                                              |
|   `early_step.the_smaller_the_better`    | (bool) True | The optimized direction of the chosen metric |                                                                                           -                                                                                           |

### FL Setting
The configurations related to FL settings are defined in `cfg_fl_setting.py`.
#### `federate`: basic fl setting
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `federate.client_num` | (int) 0 | The number of clients that involves in the FL courses. | It can set to 0 to automatically specify by the partition of dataset. |
| `federate.sample_client_num` | (int) -1 | The number of sampled clients in each training round. | - |
| `federate.sample_client_rate` | (float) -1.0 | The ratio of sampled clients in each training round. | - |
| `federate.unseen_clients_rate` | (float) 0.0 | The ratio of clients served as unseen clients, which would not be used for training and only for evaluation. | - |
| `federate.total_round_num` | (int) 50 | The maximum training round number of the FL course. | - |
| `federate.mode` | (string) 'standalone' </br> Choices: {'standalone', 'distributed'} | The running mode of the FL course. | - |
| `federate.share_local_model` | (bool) False | If `True`, only one model object is created in the FL course and shared among clients for efficient simulation. | - | 
| `federate.data_weighted_aggr` | (bool) False | If `True`, the weight of aggregator is the number of training samples in dataset. | - |
| `federate.online_aggr` | (bool) False | If `True`, an online aggregation mechanism would be applied for efficient simulation. | - | 
| `federate.make_global_eval` | (bool) False | If `True`, the evaluation would be performed on the server's test data, otherwise each client would perform evaluation on local test set and the results would be merged. | - |
| `federate.use_diff` | (bool) False | If `True`, the clients would return the variation in local training (i.e., $\delta$) instead of the updated models to the server for federated aggregation. | - | 
| `federate.merge_test_data` | (bool) False | If `True`, clients' test data would be merged and perform global evaluation for efficient simulation. | - |
| `federate.method` | (string) 'FedAvg' | The method used for federated aggregation. | We support existing federated aggregation algorithms (such as 'FedAvg/FedOpt'), 'global' (centralized training), 'local' (isolated training), personalized algorithms ('Ditto/pFedMe/FedEM'), and allow developer to customize. | 
| `federate.ignore_weight` | (bool) False | If `True`, the model updates would be averaged in federated aggregation. | - |
| `federate.use_ss` | (bool) False | If `True`, additively secret sharing would be applied in the FL course. | Only used in vanilla FedAvg in this version. | 
| `federate.restore_from` | (string) '' | The checkpoint file to restore the model. | - |
| `federate.save_to` | (string) '' | The path to save the model. | - | 
| `federate.join_in_info` | (list of string) [] | The information requirements (from server) for joining in the FL course. | We support 'num_sample/client_resource' and allow user customization.
| `federate.sampler` | (string) 'uniform' </br> Choices: {'uniform', 'group'} | The sample strategy of server used for client selection in a training round. | - |
| `federate.` </br>`resource_info_file` | (string) '' | the device information file to record computation and communication ability | - | 
#### `distribute`: for distribute mode
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `distribute.use` | (bool) False | Whether to run FL courses with distribute mode. | If `False`, all the related configurations (`cfg.distribute.xxx`) would not take effect.  |
| `distribute.server_host` | (string) '0.0.0.0' | The host of server's ip address for communication | - |
| `distribute.server_port` | (string) 50050 | The port of server's ip address for communication | - |
| `distribute.client_host` | (string) '0.0.0.0' | The host of client's ip address for communication | - |
| `distribute.client_port` | (string) 50050 | The port of client's ip address for communication | - |
| `distribute.role` | (string) 'client' </br> Choices: {'server', 'client'} | The role of the worker | - |
| `distribute.data_file` | (string) 'data' | The path to the data dile | - |
| `distribute.data_idx` | (int) -1 | It is used to specify the data index in distributed mode when adopting a centralized dataset for simulation (formatted as {data_idx: data/dataloader}). | `data_idx=-1` means that the entire dataset is owned by the participant. And we randomly sample the index in simulation for other invalid values excepted for -1.
| `distribute.` </br>`grpc_max_send_message_length` | (int) 100 * 1024 * 1024 | The maximum length of sent messages | - |
| `distribute.` </br>`grpc_max_receive_message_length` | (int) 100 * 1024 * 1024 | The maximum length of received messages | - |
| `distribute.`grpc_enable_http_proxy | (bool) False | Whether to enable http proxy | - |
#### `vertical`: for vertical federated learning
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `vertical.use` | (bool) False | Whether to run vertical FL. | If `False`, all the related configurations (`cfg.vertical.xxx`) would not take effect.  |
| `vertical.encryption` | (string) `paillier` | The encryption algorithms used in vertical FL. | - |
| `vertical.dims` | (list of int) [5,10] | The dimensions of the input features for participants. | - |
| `vertical.key_size` | (int) 3072 | The length (bit) of the public keys. | - | 






### Evaluation
The configurations related to monitoring and evaluation, which are
defined in
`cfg_evaluation.py`.

| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `eval.freq` | (int) 1 | <font size=1> The frequency we conduct evaluation. </font> | - |
| `eval.metrics` | (list of str) [] | <font size=1> The names of adopted evaluation metrics. </font> | By default, we calculate the ['loss', 'avg_loss', 'total'], all the supported metric can be find in `core/monitors/metric_calculator.py` |
| `eval.split` | (list of str) ['test', 'val'] | <font size=1> The data splits' names we conduct evaluation. </font> | - |
| `eval.report` | (list of str) ['weighted_avg', 'avg', 'fairness', 'raw'] | <font size=1> The results reported forms to loggers </font> | By default, we report comprehensive results, - `weighted_avg` and `avg` indicate the weighted average and uniform average over all evaluated clients; - `fairness` indicates report fairness-related results such as individual performance and std across all evaluated clients; - `raw` indicates that we save and compress all clients' individual results without summarization, and users can flexibly post-process the saved results further.|
| `eval.best_res_update_round_wise_key` | (str) 'val_loss' | <font size=1> The metric name we used to as the primary key to check the performance improvement at each evaluation round. </font> | - |
| `eval.monitoring` | (list of str) [] | <font size=1> Extended monitoring methods or metric, e.g., 'dissim' for B-local dissimilarity </font> | - |
| `eval.count_flops` | (bool) True | <font size=1> Whether to count the flops during the FL courses. </font> | - |
#### `wandb`: for wandb tracking and visualization
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `wandb.use` | (bool) False | <font size=1> Whether to use wandb to track and visualize the FL dynamics and results. </font> | If `False`, all the related configurations (`wandb.xxx`) would not take effect. |
| `wandb.name_user` | (str) '' | <font size=1> the user name used in wandb management </font> | - |
| `wandb.name_project` | (str) '' | <font size=1> the project name used in wandb management </font> | - |
| `wandb.online_track` | (bool) True | <font size=1> whether to track the results in an online manner, i.e., log results at every evaluation round </font> | - |
| `wandb.client_train_info` | (bool) True | <font size=1> whether to track the training info of clients </font> | - |



#### Fine tuning
The following configurations are related to the fine tuning.

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `finetune.before_eval` |     (bool) False     |      Indicator of fintune before evaluation      | If `True`, the clients will fine tune its model before each evaluation. Note the fine tuning is only conducted before evaluation and won't influence the upload weights in each round. |
| `finetune.local_update_steps` |       (int) 1        |       The number of local fine tune steps        |                                                                                           -                                                                                            |
| `finetune.batch_or_epoch` |   (string) `batch`   |          The type of local fine tuning.          |                                   Similar with `train.batch_or_epoch`, `finetune.batch_or_epoch` specifies the unit of `finetune.local_update_steps`                                   |
| `finetune.optimizer` |          -           |                        -                         |            You can add new parameters under `finetune.optimizer` according to the type of optimizer. All new parameters will be used as arguments for the chosen optimizer.            |
| `finetune.optimizer.type` |    (string) 'SGD'    |  The type of the optimizer used in fine tuning.  |                                                Currently we support all optimizers build in PyTorch (The modules under `torch.optim`).                                                 |
| `finetune.optimizer.lr` |     (float) 0.1      |   The learning rate used in local fine tuning    |                                                                                           -                                                                                            |
| `finetune.scheduler` |          -           | - |                   Similar with `train.scheduler`, you can add new parameters as you need, and all new parameters will be used as arguments for the chosen scheduler.                   |

#### Grad Clipping
The following configurations are related to the grad clipping.

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `grad.grad_clip` |     (float) -1.0     | The threshold used in gradient clipping. |                                                                 `grad.grad_clip < 0` means we don't clip the gradient.                                                                 |

#### Early Stop

|                   Name                   | (Type) Default Value |                   Description                    |                                                                                         Note                                                                                          |
|:----------------------------------------:|:--------------------:|:------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          `early_stop.patience`           | (int) 5 |  How long to wait after last time the monitored metric improved. |                        Note that the actual_checking_round = `early_step.patience` * `eval.freq`. To disable the early stop, set the `early_stop.patience` <=0                        |
|            `early_stop.delta`            | (float) 0. |  Minimum change in the monitored metric to indicate a improvement. |                                                                                           -                                                                                           |
|   `early_stop.improve_indicaator_mode`   | (string) 'best' | Early stop when there is no improvement within the last `early_step.patience` rounds, in ['mean', 'best'] |                                                                             Chosen from 'mean' or 'best'                                                                              |
|   `early_step.the_smaller_the_better`    | (bool) True | The optimized direction of the chosen metric |                                                                                           -                                                                                           |




### Asynchronous Training Strategies
The configurations related to applying asynchronous training strategies in FL are defined in `cfg_asyn.py`.

| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `asyn.use` | (bool) False | Whether to use asynchronous training strategies. | If `False`, all the related configurations (`cfg.asyn.xxx`) would not take effect.  |
| `asyn.time_budget` | (int/float) 0 | The predefined time budget (seconds) for each training round. | `time_budget`<=0 means the time budget is not applied. |
| `asyn.min_received_num` | (int) 2 | The minimal number of received feedback for the server to trigger federated aggregation. | - |
| `asyn.min_received_rate` | (float) -1.0 | The minimal ratio of received feedback w.r.t. the sampled clients for the server to trigger federated aggregation. | - |
| `asyn.staleness_toleration` | (int) 0 | The threshold of the tolerable staleness in federated aggregation. | - | 
| `asyn.` </br>`staleness_discount_factor` | (float) 1.0 | The discount factor for the staled feedback in federated aggregation. | - |
| `asyn.aggregator` | (string) 'goal_achieved' </br> Choices: {'goal_achieved', 'time_up'} | The condition for federated aggregation. | 'goal_achieved': perform aggregation when the defined number of feedback has been received; 'time_up': perform aggregation when the allocated time budget has been run out. |
| `asyn.broadcast_manner` | (string) 'after_aggregating' </br> Choices: {'after_aggregating', 'after_receiving'} | The broadcasting manner of server. | 'after_aggregating': broadcast the up-to-date global model after performing federated aggregation; 'after_receiving': broadcast the up-to-date global model after receiving the model update from clients. |
| `asyn.overselection` | (bool) False | Whether to use the overselection technique | - |

### Differential Privacy

#### NbAFL
The configurations related to NbAFL method. 

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `nbafl.use` |     (bool) False     | The indicator of the NbAFL method.         | - |
| `nbafl.mu` |      (float) 0.      | The argument $\mu$ in NbAFL.               | - | 
| `nbafl.epsilon` |     (float) 100.     | The $\epsilon$-DP guarantee used in NbAFL. | - |
| `nbafl.w_clip` |      (float) 1.      | The threshold used for weight clipping.    | - |
| `nbafl.constant` |     (float) 30. | The constant used in NbAFL.                | - |

#### SGDMF
The configurations related to SGDMF method (only used in matrix factorization tasks).

|      Name       | (Type) Default Value | Description                        | Note                                                    |
|:---------------:|:--------------------:|:-----------------------------------|:--------------------------------------------------------|
|   `sgdmf.use`   |     (bool) False     | The indicator of the SGDMF method. |                                                         |
|    `sgdmf.R`    |      (float) 5.      | The upper bound of rating.         | -                                                       |
| `sgdmf.epsilon` |      (float) 4.      | The $\epsilon$ used in DP.         | -                                                       |
| `sgdmf.delta` |     (float) 0.5      | The $\delta$ used in DP.           | -                                                       |
| `sgdmf.constant` |      (float) 1. | The constant in SGDMF | -                                                       |
| `sgdmf.theta` | (int) -1 | - | -1 means per-rating privacy, otherwise per-user privacy |

### Auto-tuning Components

These arguments are exposed for customizing our provided auto-tuning components.

#### General

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `hpo.working_folder` |     (string) 'hpo'     | Save model checkpoints and search space configurations to this folder.         | Trials in the next stage of an iterative HPO algorithm can restore from the checkpoints of their corresponding last trials. |
| `hpo.ss` |     (string) 'hpo'     | File path of the .yaml that specifying the search space.         | - |
| `hpo.num_workers` |     (int) 0     | The number of threads to concurrently attempt different hyperparameter configurations.         | Multi-threading is banned in current version. |
| `hpo.init_cand_num` |     (int) 16     | The number of initial hyperparameter configurations sampled from the search space.         | - |
| `hpo.larger_better` |     (bool) False     | The indicator of whether the larger metric is better.         | - |
| `hpo.scheduler` |     (string) 'rs' </br> Choices: {'rs', 'sha', 'wrap_sha'}     | Which algorithm to use.         | - |
| `hpo.metric` |     (string) 'client_summarized_weighted_avg.val_loss'     | Metric to be optimized.         | - |

#### Successive Halving Algorithm (SHA)

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `hpo.sha.elim_rate` |     (int) 3     | Reserve only top 1/`hpo.sha.elim_rate` hyperparameter configurations in each state.        | - |
| `hpo.sha.budgets` |     (list of int) []     | Budgets for each SHA stage.        | - |


#### FedEx

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `hpo.fedex.use` |     (bool) False     | Whether to use FedEx.        | - |
| `hpo.fedex.ss` |     (striing) ''     | Path of the .yaml specifying the search space to be explored.        | - |
| `hpo.fedex.flatten_ss` |     (bool) True     | Whether the search space has been flattened.        | - |
| `hpo.fedex.eta0` |     (float) -1.0     | Initial learning rate.        | -1.0 means automatically determine the learning rate based on the size of search space. |
| `hpo.fedex.sched` |     (string) 'auto' </br> Choices: {'auto', 'adaptive', 'aggressive', 'constant', 'scale' } | The strategy to update step sizes    | - |
| `hpo.fedex.cutoff` |     (float) 0.0 | The entropy level below which to stop updating the config.        | - |
| `hpo.fedex.gamma` |     (float) 0.0 | The discount factor; 0.0 is most recent, 1.0 is mean.        | - |
| `hpo.fedex.diff` |     (bool) False | Whether to use the difference of validation losses before and after the local update as the reward signal.        | - |

#### Wrappers for FedEx 

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `hpo.table.eps` |     (float) 0.1 | The probability to make local perturbation.        | Larger values lead to drastically different arms of the bandit FedEx attempts to solve. |
| `hpo.table.num` |     (int) 27 | The number of arms of the bandit FedEx attempts to solve.        | - |
| `hpo.table.idx` |     (int) 0 | The key (i.e., name) of the hyperparameter wrapper considers.        | No need to change this argument. |


### Attack 

The configurations related to the data/dataset are defined in `cfg_attack.py`.

Attack related configuration: 


#### For Privacy Attack
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
`attack.attack_method` | (str) '' | Attack method name | Choices: {'gan_attack', 'GradAscent', 'PassivePIA', 'DLG', 'IG'} |
`attack.target_label_ind` | (int) -1 | The target label to attack | Used in class representative attack (GAN based method) and back-door attack; defult -1 means no label to target|
`attack.attacker_id` | (int) -1 | The id of the attack client | Default -1 means no client as attacker; Used in both privacy attack and back-door attack when client is the attacker |
`attack.reconstruct_lr `| (float) 0.01 | The learning rate of the optimization based training data/label inference attack||
`attack.reconstruct_optim` | (str) 'Adam' | The learning rate of the optimization based training data/label inference attack|Choices: {'Adam', 'SGD', 'LGBFS'}|
`attack.info_diff_type` | (str) 'l2' | The distance to compare the ground-truth info (gradients or model updates) and the info generated by the dummy data. | Options: 'l2', 'l1', 'sim' representing L2, L1 and cosin similarity |
`attack.max_ite` | (int) 400 | The maximum iteration of the optimization based training data/label inference attack ||
`attack.alpha_TV` | (float) 0.001 | The hyperparameter of the total variance term | Used in the mehtod invert gradint |
`attack.inject_round` | (int) 0 | The round to start performing the attack actions | |
`attack.classifier_PIA` | (str) 'randomforest' | The property inference classifier name ||

#### For Back-door Attack
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
`attack.edge_path` |(str) 'edge_data/' | The folder where the ood data used by edge-case backdoor attacks located  ||
`attack.trigger_path` |(str) 'trigger/'|The folder where the trigger pictures used by pixel-wise backdoor attacks located  ||
`attack.setting` | (str) 'fix'| The setting about how to select the attack client. |Choices:{'fix', 'single', and 'all'}, 'single' setting means the attack client can be only selected in the predefined round (cfg.attack.insert_round). 'all' setting means the attack client can be selected in all round. 'fix' setting means that the attack client can be selected every freq round. freq has beed defined in the cfg.attack.freq keyword.|
`attack.freq` | (int) 10 |This keyword is used in the 'fix' setting. The attack client can be selected every freq round.|| 
`attack.insert_round` |(int) 100000 |This keyword is used in the 'single' setting. The attack client can be only selected in the insert_round round.||`
`attack.mean` |(list) [0.1307] |The mean value which is used in the normalization procedure of poisoning data. |Notice: The length of this list must be same as the number of channels of used dataset.|
`attack.std` |(list) [0.3081] |The std value which is used in the normalization procedure of poisoning data.|Notice: The length of this list must be same as the number of channels of used dataset.|
`attack.trigger_type`|(str) 'edge'|This keyword represents the type of used triggers|Choices: {'edge', 'gridTrigger', 'hkTrigger', 'sigTrigger', 'wanetTrigger', 'fourCornerTrigger'}|
`attack.label_type` |(str) 'dirty'| This keyword represents the type of used attack.|It contains 'dirty'-label and 'clean'-label attacks. Now, we only support 'dirty'-label attack. |
`attack.edge_num` |(int) 100 | This keyword represents the number of used good samples for edge-case attack.||
`attack.poison_ratio` |(float) 0.5|This keyword represents the percentage of samples with pixel-wise triggers in the local dataset of attack client||
`attack.scale_poisoning` |(bool) False| This keyword represents whether to use the model scaling attack for attack client. ||
`attack.scale_para` |(float) 1.0 |This keyword represents the value to amplify the model update when conducting the model scaling attack.||
`attack.pgd_poisoning` |(bool) False|This keyword represents whether to use the pgd to train the local model for attack client. ||
`attack.pgd_lr` | (float) 0.1 |This keyword represents learning rate of pgd training for attack client.||
`attack.pgd_eps`|(int) 2 | This keyword represents perturbation budget of pgd training for attack client.||
`attack.self_opt` |(bool) False |This keyword represents whether to use his own training procedure for attack client.||
`attack.self_lr` |(float) 0.05|This keyword represents learning rate of his own training procedure for attack client.||
`attack.self_epoch` |(int) 6 |This keyword represents epoch number of his own training procedure for attack client.||