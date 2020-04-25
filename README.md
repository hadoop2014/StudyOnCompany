# mxnet-tensorflow-pytorch-paddle
多个深度学习框架的组合，有助于理解在不同深度学习框架写模型的差异．

### 环境配置说明如下：
本项目环境可通过如下命令创建：
conda env create -f environment.yaml
请事先安装anaconda ,推荐版本4.7.0

另外，对于paddlepaddle的GPU版本，需要额外安装独立的cuda版本,
* 安装完需要在.bashrc文件中增加（以cuda10.1为例）：export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64/
* 同时在pycharm中的Environment Variables中增加：LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64/

### 部分目录说明如下：

１）logging_directory   用于存放log文件，包括mxnet,tensorflow,paddle,pytorch的可视化组件的输出．
   #### tensorflow
   运行tensorboard --logdir  logging_directory/tensorflow 
   #### pytorch
   运行tensorboard --logdir  logging_directory/pytorch．
   #### paddlepaddle
   运行visualdl --logdir  logging_directory/paddle

２）working_directory 用于存放模型持久化文件，所有变量和模型的持久化文件，如tensorflow的ckeckpoint文件存放于此目录．

３）data_directory用于存放所有的数据文件，从网络下载的数据文件也存放于此．

４）webapp用于存放c/s的应用，采用streamlit框架实现
　　运行streamlit run streamlitApp.py
