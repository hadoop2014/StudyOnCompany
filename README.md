# StudyOnCompany
用于分析公司的财务报表．

### 环境配置说明如下：
本项目环境可通过如下命令创建：
conda env create -f environment.yaml
请事先安装anaconda ,推荐版本4.7.0


### 部分目录说明如下：

１）logging_directory   用于存放log文件．   

２）working_directory 用于存放模型持久化文件．

３）data_directory用于存放所有的数据文件，从网络下载的数据文件也存放于此．

４）webapp用于存放c/s的应用，采用streamlit框架实现
　　运行streamlit run streamlitApp.py
