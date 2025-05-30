# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer,RobertaTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + r'/cyberbullying/600.csv'                                # 训练集
        self.dev_path = dataset + r'/cyberbullying/dev.csv'                                    # 验证集
        self.test_path = dataset + r'/cyberbullying/test.csv'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/cyberbullying/classes.txt', encoding='utf-8').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 19                                     # epoch数
        self.batch_size = 32                                     # mini-batch大小
        self.dropout = 0.1
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                 # 学习率
        self.bert_path = './model/bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768



class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        _, pooled = self.bert(context, attention_mask=mask, token_type_ids=None, return_dict=False)
        out = self.fc(pooled)
        return out
