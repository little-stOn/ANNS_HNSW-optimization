# ANNS_HNSW Optimization

本项目来自于 FDU 2025 Fall 孙未未老师的《数据结构》课程项目，聚焦于向量数据高精度低时延检索算法优化

本项目的主要功能是

项目的 baseline 来自 [HNSW](https://arxiv.org/abs/1603.09320) 算法及其相关实现 [hnswlib](https://github.com/nmslib/hnswlib)，在此基础上添加了 SQ16 量化算法，NSG 启发的 Gamma Search 策略等算法和工程上的优化，在本机上（ Intel(R) Core(TM) Ultra 5 125H , g++ 14.2.0 -O2）和课程提供的评测机（具体信息未知）上都取得了超越 baseline 的性能

由于本项目作为课程项目较为简陋，目前仅支持下载源码后自行编译运行，测试数据集可以参考 
[Datasets for approximate nearest neighbor search](http://corpus-texmex.irisa.fr/)
