# MindCon-西安旅游主题图像分类

本赛题任务是对西安的热门景点、美食、特产、民俗、工艺品等图片进行分类，即首先识别出图片中物品的类别（比如大雁塔、肉夹馍等），然后根据图片分类的规则，输出该图片中物品属于景点、美食、特产、民俗和工艺品中的哪一种.

本项目采用 resnet101 模型进行图片分类.



## 数据集

训练数据集：https://developer.huaweicloud.com/develop/aigallery/dataset/detail?id=808abc0b-c760-4687-83d5-59bc5c67062e

测试数据集：https://xihe.mindspore.cn/datasets/drizzlezyk/mindcon_xian_travel

数据集下载后按以下文件目录放置：

```
└─dataset
   ├─img_label  # 解压后的训练数据集, 文件夹改名为img_label
   └─test
      └─images  # 解压后的测试数据集，文件夹改名为images
```



## 环境要求

- 硬件（Ascend910）
- 框架（MindSpore1.8.1）



## 脚本说明

```
├── mindcon_travel
  ├── README.md                           // mindcon_travel相关说明
  ├── dataset                             // 数据集
      ├── img_label						  // 训练数据集
      ├── test                            // 测试数据集
  ├── src
      ├── data                            // 数据集配置
          ├──augment                      // 数据增强
          ├──data_utils                   // modelarts运行时数据集复制函数文件
          ┕──travel_image_dataset.py      // 数据集处理
      ├── args.py                         // 配置文件
      ├── callback.py                     // 自定义回调函数
      ├── utils.py                         
  ├── log                                 // 训练日志
  ├── resnet101_ckpt                      // 模型保存文件夹
  ├── result                              // 推理结果
  ├── train.py                            // 训练文件
  ├── inference.py                        // 推理文件
```



## 训练和推理

- 训练

```shell
python train.py
```

- 推理

```shell
python inference.py
```

推理结果保存在 `result/sorted_result.txt`

