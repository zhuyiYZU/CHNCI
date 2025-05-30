# coding: UTF-8
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, Logger
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd

logger = Logger(os.path.join("datas/log", "log.txt"))


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    logger.log()
    logger.log("model_name:", config.model_name)
    logger.log("Device:", config.device)
    logger.log("Epochs:", config.num_epochs)
    logger.log("Batch Size:", config.batch_size)
    logger.log("Learning Rate:", config.learning_rate)
    logger.log("dropout", config.dropout)
    logger.log("Max Sequence Length:", config.pad_size)
    logger.log()

    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_step = len(train_iter) * config.num_epochs
    num_warmup_steps = round(total_step * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_step)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            #print(f'Epoch [{epoch+1}/{config.num_epochs}],Batch [{i+1}/{len(train_iter)}],loss: {loss.item()}')

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                cal, dev_loss = evaluate(config, model, dev_iter)
                dev_acc = cal[0]
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                logger.log(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        dev_acc,dev_loss = evaluate(config,model,dev_iter)
        print(f'Epoch [{epoch+1}/{config.num_epochs}],Valid loss: {dev_loss}, Validd acc:{dev_acc}')

        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    print(acc)
    test_acc = acc[0]
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Precision: {1:>6.2%}, Test Recall: {1:>6.2%}, Test f1score: {1:>6.2%}'
    logger.log(msg.format(test_loss, test_acc,acc[1],acc[2],acc[3]))
    print(msg.format(test_loss, test_acc,acc[1],acc[2],acc[3]))

    print("Precision, Recall and F1-Score...")
    logger.log(test_report)
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    data = pd.DataFrame({'labels': labels_all, 'predict': predict_all})
    data.to_csv('labels.csv', index=False)
    #yudaoduofenleishi chule acc doujia average='macro'
    acc = metrics.accuracy_score(labels_all, predict_all)
    precision = metrics.precision_score(labels_all,predict_all,average='macro')
    recall = metrics.recall_score(labels_all,predict_all,average='macro')
    f1score = metrics.f1_score(labels_all,predict_all,average='macro')
    cal = [acc,precision,recall,f1score]

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return cal, loss_total / len(data_iter), report, confusion
    return cal, loss_total / len(data_iter)
