# mxnet-object-detection
学习使用mxnet，复现目标检测模型

目前已复现 vgg16-ssd, resnet18v1-ssd模型，后续会继续添加其他检测模型
数据集为kaggle上的口罩检测数据集，因次数据集样本比例不均匀，难以达到预期效果，目前只能检测出人脸

训练启动脚本在train.py
预测单张图片在predict.py
实时检测启动脚本在webcam_detection.py
