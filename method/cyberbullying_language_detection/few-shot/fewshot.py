import tqdm
from openprompt.data_utils.text_classification_dataset import *
import torch
from openprompt.data_utils.utils import InputExample
from sklearn.metrics import *
import argparse
import pandas as pd
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate, PtuningTemplate

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--seed", type=int, default=144)

parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='/home/zy-4090-1/models/bert_chinese')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results_fewshot_manual_kpt.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--kptw_lr", default=0.05, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--learning_rate", default=4e-5, type=str)
parser.add_argument("--batch_size", default=16, type=int)
args = parser.parse_args()

import random

this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed

set_seed(args.seed)

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}



if args.dataset == "ToxiCN2":
    dataset['train'] = Toxi2Processor().get_train_examples("datasets/TextClassification/ToxiCN2/")
    dataset['test'] = Toxi2Processor().get_test_examples("datasets/TextClassification/ToxiCN2/")
    class_labels =Toxi2Processor().get_labels()
    scriptsbase = "TextClassification/ToxiCN2"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "ProsCons":
    dataset['train'] = ProsProcessor().get_train_examples("datasets/TextClassification/ProsCons/")
    dataset['test'] = ProsProcessor().get_test_examples("datasets/TextClassification/ProsCons/")
    class_labels =ProsProcessor().get_labels()
    scriptsbase = "TextClassification/ProsCons"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "ToxiCN2_analysis":
    dataset['train'] = ToxiCN2analysisProcessor().get_train_examples("datasets/TextClassification/ToxiCN2_analysis/")
    dataset['test'] = ToxiCN2analysisProcessor().get_test_examples("datasets/TextClassification/ToxiCN2_analysis/")
    class_labels =ToxiCN2analysisProcessor().get_labels()
    scriptsbase = "TextClassification/ToxiCN2_analysis"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "ToxiCN2_analysis_10to1":
    dataset['train'] = ToxiCN2analysis10to1Processor().get_train_examples("datasets/TextClassification/ToxiCN2_analysis_10to1/")
    dataset['test'] = ToxiCN2analysis10to1Processor().get_test_examples("datasets/TextClassification/ToxiCN2_analysis_10to1/")
    class_labels =ToxiCN2analysis10to1Processor().get_labels()
    scriptsbase = "TextClassification/ToxiCN2_analysis_10to1"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "chinese_implict":
    dataset['train'] = CNIProcessor().get_train_examples("datasets/TextClassification/chinese_implict/")
    dataset['test'] = CNIProcessor().get_test_examples("datasets/TextClassification/chinese_implict/")
    class_labels =CNIProcessor().get_labels()
    scriptsbase = "TextClassification/chinese_implict"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "chinese_implict_content":
    dataset['train'] = CNIcontentProcessor().get_train_examples("datasets/TextClassification/chinese_implict_content/")
    dataset['test'] = CNIcontentProcessor().get_test_examples("datasets/TextClassification/chinese_implict_content/")
    class_labels =CNIcontentProcessor().get_labels()
    scriptsbase = "TextClassification/chinese_implict_content"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "THUCNews":
    dataset['train'] = THUProcessor().get_train_examples("datasets/TextClassification/THUCNews/")
    dataset['test'] = THUProcessor().get_test_examples("datasets/TextClassification/THUCNews/")
    class_labels =THUProcessor().get_labels()
    scriptsbase = "TextClassification/THUCNews"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "ChnSentiCorp":
    dataset['train'] = ChnProcessor().get_train_examples("datasets/TextClassification/ChnSentiCorp/")
    dataset['test'] = ChnProcessor().get_test_examples("datasets/TextClassification/ChnSentiCorp/")
    class_labels =ChnProcessor().get_labels()
    scriptsbase = "TextClassification/ChnSentiCorp"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "dy":
    dataset['train'] = ProsProcessor().get_train_examples("datasets/TextClassification/dy/")
    dataset['test'] = ProsProcessor().get_test_examples("datasets/TextClassification/dy/")
    class_labels = ProsProcessor().get_labels()
    scriptsbase = "TextClassification/dy"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "wb":
    dataset['train'] = ProsProcessor().get_train_examples("datasets/TextClassification/wb/")
    dataset['test'] = ProsProcessor().get_test_examples("datasets/TextClassification/wb/")
    class_labels = ProsProcessor().get_labels()
    scriptsbase = "TextClassification/wb"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "xhs":
    dataset['train'] = ChnProcessor().get_train_examples("datasets/TextClassification/xhs/")
    dataset['test'] = ChnProcessor().get_test_examples("datasets/TextClassification/xhs/")
    class_labels =ChnProcessor().get_labels()
    scriptsbase = "TextClassification/xhs"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
else:
    raise NotImplementedError


#mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(path=f"./scripts/{scriptsbase}/manual_template.txt",choice=args.template_id)

mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/ptuning_template.txt", choice=args.template_id)
#
if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/kpt_translate_verbalizer.{scriptformat}")

elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(
        f"./scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "inter":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/inter_refine_verbalizer.{scriptformat}")
elif args.verbalizer == "ex_re":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/expand_refine_verbalizer.{scriptformat}")
elif args.verbalizer == "ex_re1":#一个中心点的阔删
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/expand_refine1_verbalizer.{scriptformat}")
elif args.verbalizer == "sc":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/Spectral_Clustering_verbalizer.{scriptformat}")
elif args.verbalizer == "reverse":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/Reverse_dict_verbalizer.{scriptformat}")
elif args.verbalizer == "kpt++":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/kpt++2_verbalizer.{scriptformat}")


# (contextual) calibration
if args.verbalizer in ["kpt", "manual",'inter','ex_re','ex_re1','sc','reverse','kpt++']:
    if args.calibration or args.filter != "none":
        from openprompt.data_utils.data_sampler import FewShotSampler

        support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
        dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

        # for example in dataset['support']:
        #     example.label = -1 # remove the labels of support set for clarification
        support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
                                              tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                              decoder_max_length=3,
                                              batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                              predict_eos_token=False,
                                              truncate_method="tail")

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

# HP
# if args.calibration:
if args.verbalizer in ["kpt", "manual",'inter','ex_re','ex_re1','sc','reverse','kpt++']:
    if args.calibration or args.filter != "none":
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
        from openprompt.utils.calibrate import calibrate

        # calculate the calibration logits
        cc_logits = calibrate(prompt_model, support_dataloader)
        print("the calibration logits is", cc_logits)
        print("origial label words num {}".format(org_label_words_num))

    if args.calibration:
        myverbalizer.register_calibrate_logits(cc_logits)
        new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
        print("After filtering, number of label words per class: {}".format(new_label_words_num))

    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
  # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.

#### sannhang henzhongyao！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
################！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
from openprompt.data_utils.data_sampler import FewShotSampler
sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)
######################################################################################！！！！！！！！！！！！！！！！！！！！！！！！！！！

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")
'''
import numpy as np

# 计算混淆矩阵
def compute_confusion_matrix(precited,expected):
    part = precited ^ expected             # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
    pcount = np.bincount(part)             # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
    tp_list = list(precited & expected)    # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
    tp = tp_list.count(1)                  # 统计TP的个数
    fp = fp_list.count(1)                  # 统计FP的个数
    tn = pcount[0] - tp                    # 统计TN的个数
    fn = pcount[1] - fp                    # 统计FN的个数
    return tp, fp, tn, fn

# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    if (tp+tn+fp+fn)!=0 & (tp+fp)!=0 & (tp+fn)!= 0:
        accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
        precision = tp / (tp+fp)               # 精确率
        recall = tp / (tp+fn)                  # 召回率
        F1 = (2*precision*recall) / (precision+recall)    # F1
    else:
        accuracy = 0.1     # 准确率
        precision = 0.1               # 精确率
        recall = 0.1                  # 召回率
        F1 = 0.1    # F1
    return accuracy, precision, recall, F1
'''

def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    pd.DataFrame({"alllabels":alllabels,"allpreds":allpreds}).to_csv('out_label_bili_bl.csv', header=0,index=False)
    # print(alllabels)
    # print(allpreds)

    # allpreds = np.array(allpreds)
    # alllabels = np.array(alllabels)
    # tp, fp, tn, fn = compute_confusion_matrix(allpreds, alllabels)
    #
    # acc, pre, recall, F1score = compute_indexes(tp, fp, tn, fn)



    # acc = sum([int(i != j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    acc = accuracy_score(alllabels, allpreds)
    # pre = precision_score(alllabels, allpreds, average='weighted')
    # recall = recall_score(alllabels, allpreds, average='weighted')
    # F1score = f1_score(alllabels, allpreds, average='weighted')
    pre, recall, F1score, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
    cal_data = [acc, pre, recall, F1score]
    return cal_data


############
#############
###############

from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()




if args.verbalizer == "kpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    # optimizer_grouped_parameters2 = [
    #     {'params': , "lr":1e-1},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)
    # print(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler2 = get_linear_schedule_with_warmup(
    #     optimizer2,
    #     num_warmup_steps=0, num_training_steps=tot_step)
    scheduler2 = None


elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "trpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "inter":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None
elif args.verbalizer == "ex_re":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None
elif args.verbalizer == "ex_re1":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "sc":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None
elif args.verbalizer == "reverse":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "kpt++":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

tot_loss = 0
log_loss = 0
best_val_acc = 0
for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss = tot_loss + loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

    cal_data = evaluate(prompt_model, validation_dataloader, desc="Valid")
    val_acc = cal_data[0]
    if val_acc >= best_val_acc:
        torch.save(prompt_model.state_dict(), f"./ckpts/{this_run_unicode}.ckpt")
        best_val_acc = val_acc
    print("Epoch {}, val_acc {}".format(epoch, val_acc), flush=True)

prompt_model.load_state_dict(torch.load(f"./ckpts/{this_run_unicode}.ckpt"))
prompt_model = prompt_model.cuda()
data_set = evaluate(prompt_model, test_dataloader, desc="Test")
test_acc = data_set[0]
test_pre = data_set[1]
test_recall = data_set[2]
test_F1scall = data_set[3]

content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += f"bt {args.batch_size}\t"
content_write += f"lr {args.learning_rate}\t"

content_write += "\n"

content_write += f"Acc: {test_acc}\t"
content_write += f"Pre: {test_pre}\t"
content_write += f"Rec: {test_recall}\t"
content_write += f"F1s: {test_F1scall}\t"
content_write += "\n\n"

print(content_write)
with open("./results.txt","a",encoding="utf-8") as file:
     file.write(content_write)

import os

os.remove(f"./ckpts/{this_run_unicode}.ckpt")