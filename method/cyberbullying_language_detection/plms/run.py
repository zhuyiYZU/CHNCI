# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from models import bert

# 声明argparse对象 可附加说明
parser = argparse.ArgumentParser(description='Chinese Text Classification')
# 模型是必须设置的参数(required=True) 类型是字符串
parser.add_argument('--model', type=str, default="bert", help='choose a model: bert, bert_CNN,'
                                                              'bert_DPCNN,bert_RNN,bert_RCNN,ERNIE')
# 解析参数
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'datas'  # 数据集路径

    model_name = args.model  # bert 设置的模型名称
    x = import_module('models.' + model_name)  # 根据所选模型名字在models包下 获取相应模块
    config = x.Config(dataset)  # 模型文件中的配置类

    # 设置随机种子 确保每次运行的条件(模型参数初始化、数据集的切分或打乱等)是一样的
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # 构建训练集、验证集、测试集
    train_data, dev_data, test_data = build_dataset(config)
    # 构建训练集、验证集、测试集迭代器
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)  # 构建模型对象
    train(config, model, train_iter, dev_iter, test_iter)  # 训练
