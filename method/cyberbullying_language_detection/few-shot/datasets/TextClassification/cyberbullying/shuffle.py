import pandas as pd
import csv
import random
import os
path = os.getcwd()

with open('/home/zy-4090-1/zx/CN_implicit/datasets/TextClassification/dy/test.csv','r') as file:
    reader = csv.reader(file)
    data = list(reader)
random.shuffle(data)
with open('/home/zy-4090-1/zx/CN_implicit/datasets/TextClassification/dy/test1.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
