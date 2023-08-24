# README-cifar数据集训练

## python=3.9.7

### 本程序用到的第三方库：

```
tensorflow==2.6.0
```



### 运行方式：

在 “cifar/” 文件夹内，运行

```
python cifar_main.py # 训练模型，内含训练部分，保存模型部分，测试部分
python test.py # 测试模型，内含导入模型
```



### 内含文件：

cifar_main.py : 用于训练保存模型，也含有测试部分

test.py ： 用于测试模型

imageOfResult.png ： 部分程序运行结果截图

cifar.h5 ： 已经训练好的模型文件

logs_cifar : tensorboard日志文件夹





### 查看tensorboard日志文件：

在 “cifar/” 文件夹内，打开cmd运行

```
tensorboard --logdir=./logs_cifar
```

在浏览器中，访问 http://localhost:6006/

即可浏览