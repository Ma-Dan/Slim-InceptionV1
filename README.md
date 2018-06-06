# Slim Inception V1

##Example for Slim Inception V1 training and test
1. Save BatchNorm variables;
2. Adjust moving averages decay to fit small samples (30 pictures);

###Training
```shell
python train.py
```

###Test
```shell
python test.py
```

###Distributed Training
On ps and worker 0 (192.168.1.1)
```shell
python train_dist.py --ps_hosts=192.168.1.1:2221 --worker_hosts=192.168.1.1:2223,192.168.1.2:2223 --job_name=ps --task_index=0
python train_dist.py --ps_hosts=192.168.1.1:2221 --worker_hosts=192.168.1.1:2223,192.168.1.2:2223 --job_name=worker --task_index=0
```

On Worker 1 (192.168.1.2)
```shell
python train_dist.py --ps_hosts=192.168.1.1:2221 --worker_hosts=192.168.1.1:2223,192.168.1.2:2223 --job_name=worker --task_index=1
```

###Test pretrained weight
```shell
wget https://appcenter-deeplearning.sh1a.qingstor.com/models/TensorFlow-Slim%20image%20classification/inception_v1_2016_08_28.tar.gz
tar xvf inception_v1_2016_08_28.tar.gz
python test_pretrained.py dog.jpg
```
