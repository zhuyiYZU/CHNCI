import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


descript_file = r'/home/ubuntu/zx/double-llm-vote/result/sampled_output_m-agent1.csv'
label_descript_file = r'E:\Study project\double-llm-vote\result\kuaishou-hhh.csv'

with open(descript_file,mode='r',newline='',encoding='utf-8') as infile,\
    open(label_descript_file,mode='a',newline='',encoding='utf-8') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)

    true_labels = []
    pred_labels = []
    for row in csv_reader:
        label_list = [0,0,0,0,0]
        # print(row)
        for i in range(2, len(row), 6):
            index = (i - 2) // 6  # 将i映射到label_list的索引0-4
            for j in range(i, min(i + 6, len(row)-1), 2):#2-7,步长是2
                label_list[index] = label_list[index] + int(float(row[j]))
            if label_list[index] >= 2:#内部投票
                label_list[index] = 1
            else:
                label_list[index] = 0
        label = 0
        for l in label_list:
            label = label + l
        if label >= 3:#外部投票
            label = 1
        else:
            label = 0
        row.append(label)
        csv_writer.writerow(row)

        true_labels.append(row[0])
        pred_labels.append(str(label))

    # 计算准确率
    accuracy = accuracy_score(true_labels, pred_labels)


    # 计算 F1 分数
    f1 = f1_score(true_labels, pred_labels, average='weighted')


    print("Accuracy:", accuracy)
    print("F1 Score:", f1)



