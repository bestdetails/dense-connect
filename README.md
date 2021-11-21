# dense-connect
# 2021-11-02
扩展torch.nn.Sequential类，初步实现密集连接  connect_test.py

# 2021-11-06
密集连接测试  test.py

# 2021-11-07
密集连接测试 test2.py
密集连接测试 test3.py

# 2021-11-15
密集连接测试 test4.py

# 2021-11-21
第一轮在cifar10上的训练，密集连接导致的模型复杂度过高，造成了严重过拟合，validation acc仅有0.6，
下一步想分析模型中各个残差连接的激活情况，看看哪些连接更有用。
尽管模型复杂度高，但由于cifar10数据集较小，单张图像仅有32* 32，所以单个TitanV也能很快完成训练，
接下来想在ImageNet上进行训练，ImageNet数据集规模远远超过cifar10，单个GPU无法完成任务，必须考虑多gpu并行
所以接下来的一段时间要学习使用dask，就这样，github网页版没法上传25M以上的文件，所以训练好的模型无法提交到model文件夹
而我又不是很想装gitlfs，所以先这样吧，训练记录提交在log文件夹下
