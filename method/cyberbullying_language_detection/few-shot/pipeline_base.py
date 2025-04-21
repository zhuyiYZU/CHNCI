from torch.utils.data.sampler import RandomSampler
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from collections import defaultdict
from openprompt.utils import round_list, signature
import numpy as np
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch_geometric.nn as gnn
from torch_geometric.nn import GATConv,GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from ltp import LTP
from transformers import BertTokenizer
import spacy
import torch
from torch_geometric.data import Data
from torch.nn import MultiheadAttention

class PromptDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer. 
    
    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper. 
    """
    def __init__(self, 
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_wrapper_class: TokenizerWrapper,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset
        
        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
        prepare_kwargs = {
            "max_seq_length" : max_seq_length,
            "truncate_method" : truncate_method,
            "decoder_max_length" : decoder_max_length,
            "predict_eos_token" : predict_eos_token,
            "tokenizer" : tokenizer,
            **kwargs,
        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
        

        self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)
        
        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"
        
        # processs
        self.wrap()
        self.tokenize()

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset, 
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = InputFeatures.collate_fct
        )
        
    
    
    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        这个方法是DatasetWrapper类中的方法，作用是将数据集中的每个样本进行包装，即使用模板和语言化模块对样本进行处理。
        具体来说，该方法会遍历数据集中的每个样本，然后将其作为参数传递给模板和语言化模块的wrap_one_example方法进行处理，
        得到经过包装的样本，并将其添加到wrapped_dataset列表中。其中，如果语言化模块没有实现wrap_one_example方法，则不进行额外处理。
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List): 
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):

                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                    # print(example)
                wrapped_example = self.template.wrap_one_example(example)

                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError
    
    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer, 
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
           这段代码是一个类中的 tokenize() 方法，用于将 wrapped_dataset 中的文本数据转换为模型可接受的张量数据（tensor dataset）。
           具体地，该方法使用 tokenizer_wrapper 对象中的 tokenize_one_example() 方法，将 wrapped_example 中的文本转换为 token，
           然后使用 InputFeatures() 将 token 和其他特征打包成一个 InputFeatures 对象。接着，调用 to_tensor() 方法将 InputFeatures 对象转换为一个张量，最后将其加入 tensor_dataset 中。
            其中，enumerate() 函数是一个 Python 内置函数，用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标。
            在这里，enumerate() 函数用于对 wrapped_dataset 中的每个文本进行遍历，并为每个文本分配一个索引序号 idx。
            而 tqdm() 函数是一个 Python 进度条库，用于在命令行中显示程序的进度。它的作用是为 enumerate() 函数添加一个进度条，以便用户可以实时查看数据处理的进度。
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
        #     print(wrapped_example)
            '''
            [[{'text': '这是一个', 'loss_ids': 0, 'shortenable_ids': 0}, {'text': '<mask>', 'loss_ids': 1, 'shortenable_ids': 0}, {'text': ' 的新闻:', 'loss_ids': 0, 'shortenable_ids': 0}, 
            {'text': ' 中国驻美大使秦刚会见美国财长耶伦', 'loss_ids': 0, 'shortenable_ids': 1}, 
            {'text': ' 中新社华盛顿12月15日电中国驻美大使秦刚15日在华盛顿会见美国财长耶伦。双方就落实中美元首巴厘岛会晤重要共识以及共同关心的双边和多边经济财金问题深入交换了意见。双方同意继续保持交往，加强宏观经济政协调及在双边经贸问题上的沟通，促进共同应对全球性挑战，推动中美关系健康稳定发展。（完）责任编辑：刘光博', 'loss_ids': 0, 'shortenable_ids': 1}], 
            {'guid': '550', 'label': 1}]
    
    
    [[{'text': ' 这是一条关于', 'loss_ids': 0, 'shortenable_ids': 0}, {'text': '<mask>', 'loss_ids': 1, 'shortenable_ids': 0}, {'text': '的评论：', 'loss_ids': 0, 'shortenable_ids': 0}, {'text': '这个的前提是蓝精灵和黄精灵本质是一样的，但男女不一样', 'loss_ids': 0, 'shortenable_ids': 1}, {'text': '，这shortenable_ids': 0}], {'guid': '2840', 'label': 0}]
           '''
            # print(wrapped_example[0][3]['text'])
            # print(len(wrapped_example[0][3]['text']))
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            # print(inputfeatures)
            #101, 6821, 3221, 671, 702, 103, 4638, 3173, 7319, 131  对应模板
            #101 [cls]  102[sep] 103[mask] 131 :
            # print(inputfeatures['input_ids'][10: 9 +len(wrapped_example[0][3]['text'])])  标题对应的索引
            # print(len(inputfeatures['loss_ids']))  #128
            # print(len(inputfeatures['attention_mask']))  #128
            # print(wrapped_example[0][2]['text'])
            # inputfeatures['title_len'] = len(wrapped_example[0][2]['text'])
            title = wrapped_example[0][3]['text']
            # gat_input = self.title_to_pos_graph(title)
            inputfeatures['title'] = title
            # print(inputfeatures['gat_inputs'])  #Data(x=[7], edge_index=[2, 9])
            self.tensor_dataset.append(inputfeatures)
            '''
            {"input_ids": [101, 6821, 3221, 671, 702, 103, 4638, 3173, 7319, 131, 704, 1925, 2137, 6444, 9707, 8152, 2399, 5307, 3845, 8024, 5307, 3845, 1920, 4689, 1114, 1906, 1962, 749, 1408, 8043, 3341, 3975, 8038, 704, 1744, 5307, 3845, 145
                3, 1149, 6381, 5442, 3330, 3719, 1290, 2101, 2431, 7345, 429, 1266, 776, 2845, 6887, 100, 1059, 7481, 3918, 1265, 3121, 7484, 2458, 3123, 8024, 1920, 1213, 2990, 2920, 2356, 1767, 928, 2552, 100, 8039, 100, 2972, 1220, 5307, 3845, 6
                817, 6121, 3146, 860, 1962, 6760, 8024, 2141, 4385, 6574, 4638, 3300, 3126, 2990, 1285, 1469, 7030, 4638, 1394, 4415, 1872, 7270, 100, 8039, 100, 100, 8110, 3299, 127, 3189, 8024, 704, 1925, 3124, 3780, 2229, 833, 6379, 7025, 3123, 
                1139, 3353, 1071, 2487, 4164, 4638, 928, 1384, 8038, 1939, 102], 
                "inputs_embeds": null, 
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1], 
                "token_type_ids": null, 
                "label": 1, 
                "decoder_input_ids": null, 
                "decoder_inputs_embeds": null, 
                "soft_token_ids": null, 
                "past_key_values": null, 
                "loss_ids": [-100, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100], 
                "guid": "577", "tgt_text": null, "encoded_tgt_text": null, "input_ids_len": null}
            '''


        '''
        这是一个自定义的DataLoader类的实现，用于封装PyTorch内置的DataLoader类，可以简化训练时的数据加载操作。
        该类中实现了__len__和__iter__两个方法，分别返回该数据集的长度和一个迭代器对象。
        在这个实现中，__iter__方法直接调用了PyTorch内置的DataLoader的__iter__方法，因此可以通过迭代该自定义DataLoader对象来获得每个batch的数据。
        '''



    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()



class PromptModel(nn.Module):
    r'''``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``, 
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``



    Args:
        plm (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train. 
    '''
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        self.template = template
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False 
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
    
    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self
        
    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position.



        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.template.post_processing_outputs(outputs)
        return outputs
    
    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures]) -> Dict:
        r"""Will be used in generation
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch

'''GCN+ATT'''
# 以下两个类结合是GCN+Attention  再往下一个是只GCN
# class GCNWithSelfAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
#         super(GCNWithSelfAttention, self).__init__()
#         self.gcn = GCNConv(input_dim, hidden_dim).to('cuda')
#         self.self_attention = MultiheadAttention(hidden_dim, num_heads).to('cuda')
#         self.fc = nn.Linear(hidden_dim, output_dim).to('cuda')
#         # self.gcn = GCNConv(input_dim, hidden_dim)
#         # self.self_attention = MultiheadAttention(hidden_dim, num_heads)
#         # self.fc = nn.Linear(hidden_dim, output_dim)
# 
#     def forward(self, x, edge_index):
#         # GCN layer
#         # x = self.gcn(x, edge_index)
#         x = self.gcn(x=x.to('cuda'), edge_index=torch.as_tensor(edge_index).to('cuda'))
#         # Self-attention layer
#         x = x.transpose(0, 1)  # Shape: (N, num_nodes, hidden_dim)
#         x, _ = self.self_attention(x, x, x) # Self-attention
#         x = x.transpose(0, 1)  # Shape: (num_nodes, N, hidden_dim)
#         x = F.relu(x)
#         # Fully connected layer
#         x = self.fc(x)
#         return x
# 
# 
# class PromptForClassification(nn.Module):
#     r'''``PromptModel`` with a classification head on top. The classification head will map
#     the logits in all position of the sequence (return value of a ``PromptModel``) into the
#     logits of the labels, using a verbalizer.
# 
#     Args:
#         plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
#         template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
#         verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the lables to label words for classification, e.g. ``ManualVerbalizer``.
#         freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
#         plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
#     '''
#     def __init__(self,
#                  plm: PreTrainedModel,
#                  template: Template,
#                  verbalizer: Verbalizer,
#                  freeze_plm: bool = False,
#                  plm_eval_mode: bool=False,
#                  in_channels: int=64,
#                  out_channels: int=128
#                 ):
#         super().__init__()
#         # self.gat = GATConv(in_channels=64, out_channels=out_channels)
# 
#         self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
#         self.verbalizer = verbalizer
# 
# 
#     @property
#     def plm(self):
#         return self.prompt_model.plm
# 
#     @property
#     def template(self):
#         return self.prompt_model.template
# 
#     @property
#     def device(self,):
#         r"""Register the device parameter."""
#         return self.plm.device
# 
#     def extract_at_mask(self,
#                        outputs: torch.Tensor,
#                        batch: Union[Dict, InputFeatures]):
#         r"""Get outputs at all <mask> token
#         E.g., project the logits of shape
#         (``batch_size``, ``max_seq_length``, ``vocab_size``)
#         into logits of shape (if num_mask_token > 1)
#         (``batch_size``, ``num_mask_token``, ``vocab_size``)
#         or into logits of shape (if ``num_mask_token`` = 1)
#         (``batch_size``, ``vocab_size``).
# 
# 
#         Args:
#             outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
#                  `gather_outputs` before) etc. of the whole sequence.
#             batch (:obj:`Union[Dict, InputFeatures]`): The original batch
# 
#         Returns:
#             :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.
# 
#         """
#         outputs = outputs[torch.where(batch['loss_ids']>0)]
# 
#         outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
#         if outputs.shape[1] == 1:
#             outputs = outputs.view(outputs.shape[0], outputs.shape[2])
#         return outputs
#     def title_to_pos_graph(self, title):
#         # print(title)
#         ltp = LTP(pretrained_model_name_or_path='/home/star/anaconda3/envs/syn_prompt/lib/python3.9/site-packages/ltp/ltp_model')
#         #调用ltp
#         output = ltp.pipeline([title], tasks=["cws", "pos", "sdp", "dep"])
#         cws = output.cws  # 分词
#         pos = output.pos  # 词性标注  #{'nt', 'q', 'n', 'a', 'v'}
#         dep = output.dep[0]  # 依存句法结构
#         graph = {'nodes': set(pos[0]), 'edges': []}
#         for i, word in enumerate(cws[0]):
#             head = dep['head'][i] - 1  # 依存句法关系中的头
#             if head >= 0:  # 头存在（注意有些标点符号头为-1）
#                 label = dep['label'][i]  # 依存句法关系中的标签
#                 graph['edges'].append((pos[0][i], label, pos[0][head]))
#         #构建边
#         edge_index = []
#         for edge in graph['edges']:
#             if edge[0] in graph['nodes'] and edge[2] in graph['nodes']:
#                 src = list(graph['nodes']).index(edge[0])
#                 dst = list(graph['nodes']).index(edge[2])
#                 edge_index.append([src, dst])
#         # print(edge_index)  #[[1, 0], [4, 2], [3, 2], [2, 0]]
#         if edge_index == []:
#             edge_index = [[0, 0]]
#         edge_index = torch.tensor(edge_index).t().contiguous()
#         # x = []
#         # for node in graph['nodes']:
#         #     x.append(list(graph['nodes']).index(node))
#         x = torch.tensor(BertTokenizer.from_pretrained('/home/star/anaconda3/envs/syn_prompt/lib/python3.9/site-packages/openprompt/plms/chinese-roberta-wwm-ext').convert_tokens_to_ids(pos[0]), dtype=torch.float)
#         # x = x.to(torch.float32)
#         data = Data(x=x.to('cuda'), edge_index=torch.as_tensor(edge_index).to('cuda'))
#         # data = Data(x, edge_index=torch.as_tensor(edge_index))
#        #####meiyong # data = Data(x=torch.tensor(x).to('cuda'), edge_index=torch.tensor(edge_index).to('cuda'))
#         '''
#         data:
# Data(x=[19], edge_index=[2, 18])
# 
#         '''
#         # print('data:')
#         # print(data)
#         return data
# 
# 
# 
#     def title_to_en_pos_graph(self, title):
#         # Load SpaCy English model
#         nlp = spacy.load("en_core_web_sm")
# 
#         # Process English title using SpaCy
#         doc = nlp(title)
# 
#         # Extract information
#         pos = [token.pos_ for token in doc]
#         dep = [(token.dep_, token.head.text) for token in doc]
# 
#         # Build graph representation
#         graph = {'nodes': set(pos), 'edges': []}
#         for i, token in enumerate(doc):
#             head = token.head.text
#             label = dep[i][0]
#             graph['edges'].append((pos[i], label, head))
# 
#         # Edge indexing
#         edge_index = []
#         for edge in graph['edges']:
#             if edge[0] in graph['nodes'] and edge[2] in graph['nodes']:
#                 src = list(graph['nodes']).index(edge[0])
#                 dst = list(graph['nodes']).index(edge[2])
#                 edge_index.append([src, dst])
# 
#         if edge_index == []:
#             edge_index = [[0, 0]]
#         edge_index = torch.tensor(edge_index).t().contiguous()
# 
#         # Convert node features using BERT tokenizer
#         tokenizer = BertTokenizer.from_pretrained('/home/mjy/wangye/text_classification/model/bert-case')
#         x = torch.tensor(tokenizer.convert_tokens_to_ids([token.text for token in doc]), dtype=torch.float)
# 
#         # Create PyTorch Data object
#         data = Data(x=x.to('cuda'), edge_index=edge_index.to('cuda'))
#         # print(data)
#         return data
# 
#     def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
#         r"""
#         Get the logits of label words.
# 
#         Args:
#             batch (:obj:`Union[Dict, InputFeatures]`): The original batch
# 
#         Returns:
#             :obj:`torch.Tensor`: The logits of the lable words (obtained by the current verbalizer).
#         """
#         # print(batch['gat_inputs'])
#         # print(batch['input_ids'][10: 9 +len(wrapped_example[0][3]['text'])])
#         # merged_dict = {k: v for k, v in zip(batch['title_len'],batch['input_ids'])}
#         # for k, v in merged_dict.items():
#         #     print(v[10: 9 + k])             #标题的vocab索引
#         # print(merged_dict)
#         outputs = self.prompt_model(batch)  #torch.Size([4, 250, 768]) batch, max_seq_l, vector_dim
#         outputs = self.verbalizer.gather_outputs(outputs)
#         # print('outputs:', outputs.shape)  #(batch_size, max_seq_length, vocab_size) outputs: torch.Size([2, 128, 21128])
# 
# 
#         titles = batch['title']
#         gcn_inputs = []
#         for title in titles:
#             # print(title)
#             gcn_input = self.title_to_pos_graph(title)
#             # gcn_input = self.title_to_en_pos_graph(title)
#             gcn_with_self_attention = GCNWithSelfAttention(input_dim=64, hidden_dim=len(gcn_input.x), output_dim=128,
#                                                            num_heads=len(gcn_input.x))
#             # 使用 unsqueeze 将 x 的维度从 1 扩展到 2
#             gcn_input.x = gcn_input.x.unsqueeze(1)
#             # print(gat_input.x.size())  # 输出 torch.Size([9, 1])
# 
#             # 使用 repeat 将 x 沿着新的维度复制 300 次
#             gcn_input.x = gcn_input.x.repeat(1, 64)
#             # print(gat_input.x.shape)
#             # gat_input = gat_input.x.to(torch.float32).to(device)
#             # gat_input = gat_input.x.to(torch.float32).to(device)
#             # print(gat_input.edge_index)
#             gcn_output = gcn_with_self_attention(gcn_input.x.cuda(), gcn_input.edge_index.cuda())
#             # gcn_output = gcn_with_self_attention(gcn_input.x, gcn_input.edge_index)
#             # print('1: ', gat_output.shape)
#             gcn_output = F.elu(gcn_output)
#             # print('2: ', gat_output)
#             gcn_inputs.append(gcn_output)
# 
# 
#         # print(gat_inputs)
#         #gat
# 
#         # gat_outputs = self.gat(gat_inputs.x, gat_inputs.edge_index)
#         # gat_outputs = F.elu(gat_outputs)
#         # print(gat_outputs)
# 
# 
#         #从模型的输出中提取出所有"<mask>"标记的输出  (batch_size, max_seq_length, vocab_size) --> (batch_size, num_mask_token, vocab_size)
#         outputs_at_mask = self.extract_at_mask(outputs, batch) #torch_size([4, 21128]) 只有一个mask (batch_size, vocab_size)
#         # print(outputs_at_mask)
# 
#         # num_vectors 表示需要拼接的向量个数
#         num_vectors = outputs_at_mask.shape[0]
#         compressed_vectors = []
#         for i in range(num_vectors):
#             compressed_vector = torch.mean(gcn_inputs[i], dim=0, keepdim=True)
#             compressed_vectors.append(compressed_vector)
#         concatenated = torch.cat(compressed_vectors, dim=0)
# 
#         outputs_at_mask_pos = torch.cat((outputs_at_mask, concatenated), dim=1)
# 
#         # 从模型的输出中提取出所有"<mask>"标记的输出  (batch_size, max_seq_length, vocab_size) --> (batch_size, num_mask_token, vocab_size)
#         # outputs_at_mask = self.extract_at_mask(outputs, batch)  # torch_size([4, 21128]) 只有一个mask (batch_size, vocab_size)
#         # print(outputs_at_mask_pos.shape)
#         label_words_logits = self.verbalizer.process_outputs(outputs_at_mask_pos, batch=batch)
#         # label_words_logits = self.verbalizer.process_outputs(outputs_at_mask_pos, batch=batch)
#         # print(label_words_logits)
#         return label_words_logits
# 
#     def predict(self):
#         pass
# 
#     def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
#         outputs = self.prompt_model(batch)
#         outputs = self.verbalizer.gather_outputs(outputs)
#         outputs_at_mask = self.extract_at_mask(outputs, batch)
#         return outputs_at_mask
# 
#     @property
#     def tokenizer(self):
#         r'''Utility property, to get the tokenizer more easily.
#         '''
#         return self.verbalizer.tokenizer
# 
#     def state_dict(self, *args, **kwargs):
#         """ Save the model using template, plm and verbalizer's save methods."""
#         _state_dict = {}
#         if not self.prompt_model.freeze_plm:
#             _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
#         _state_dict['template'] = self.template.state_dict(*args, **kwargs)
#         _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
#         return _state_dict
# 
#     def load_state_dict(self, state_dict, *args, **kwargs):
#         """ Load the model using template, plm and verbalizer's load methods."""
#         if 'plm' in state_dict and not self.prompt_model.freeze_plm:
#             self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
#         self.template.load_state_dict(state_dict['template'], *args, **kwargs)
#         self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)
# 
#     def parallelize(self, device_map=None):
#         r"""Parallelize the model across device
#         """
#         if hasattr(self.plm, "parallelize"):
#             self.plm.parallelize(device_map)
#             self.device_map = self.plm.device_map
#             self.template.cuda()
#             self.verbalizer.cuda()
#             # self.template.cpu()
#             # self.verbalizer.cpu()
#         else:
#             raise NotImplementedError("parallelize method was not implemented for this plm.")
# 
#     def deparallelize(self):
#         r"""Deparallelize the model across device
#         """
#         if hasattr(self.plm, "deparallelize"):
#             self.plm.deparallelize()
#             self.device_map = None
#             self.template.cpu()
#             self.verbalizer.cpu()
#         else:
#             raise NotImplementedError("parallelize method was not implemented for this plm.")

'''GAT'''
# GAT
# class PromptForClassification(nn.Module):
#     r'''``PromptModel`` with a classification head on top. The classification head will map
#     the logits in all position of the sequence (return value of a ``PromptModel``) into the
#     logits of the labels, using a verbalizer.
#
#     Args:
#         plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
#         template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
#         verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the lables to label words for classification, e.g. ``ManualVerbalizer``.
#         freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
#         plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
#     '''
#
#     def __init__(self,
#                  plm: PreTrainedModel,
#                  template: Template,
#                  verbalizer: Verbalizer,
#                  freeze_plm: bool = False,
#                  plm_eval_mode: bool = False,
#                  in_channels: int = 64,
#                  out_channels: int = 128
#                  ):
#         super().__init__()
#         self.gat = GATConv(in_channels=64, out_channels=out_channels)
#
#         self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
#         self.verbalizer = verbalizer
#
#     @property
#     def plm(self):
#         return self.prompt_model.plm
#
#     @property
#     def template(self):
#         return self.prompt_model.template
#
#     @property
#     def device(self, ):
#         r"""Register the device parameter."""
#         return self.plm.device
#
#     def extract_at_mask(self,
#                         outputs: torch.Tensor,
#                         batch: Union[Dict, InputFeatures]):
#         r"""Get outputs at all <mask> token
#         E.g., project the logits of shape
#         (``batch_size``, ``max_seq_length``, ``vocab_size``)
#         into logits of shape (if num_mask_token > 1)
#         (``batch_size``, ``num_mask_token``, ``vocab_size``)
#         or into logits of shape (if ``num_mask_token`` = 1)
#         (``batch_size``, ``vocab_size``).
#
#
#         Args:
#             outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
#                  `gather_outputs` before) etc. of the whole sequence.
#             batch (:obj:`Union[Dict, InputFeatures]`): The original batch
#
#         Returns:
#             :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.
#
#         """
#         outputs = outputs[torch.where(batch['loss_ids'] > 0)]
#
#         outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
#         if outputs.shape[1] == 1:
#             outputs = outputs.view(outputs.shape[0], outputs.shape[2])
#         return outputs
#
#     def title_to_pos_graph(self, title):
#         ltp = LTP(
#             pretrained_model_name_or_path='/home/star/anaconda3/envs/syn_prompt/lib/python3.9/site-packages/ltp/ltp_model')
#         # 调用ltp
#         output = ltp.pipeline([title], tasks=["cws", "pos", "sdp", "dep"])
#         cws = output.cws  # 分词
#         pos = output.pos  # 词性标注  #{'nt', 'q', 'n', 'a', 'v'}
#         dep = output.dep[0]  # 依存句法结构
#         graph = {'nodes': set(pos[0]), 'edges': []}
#         for i, word in enumerate(cws[0]):
#             head = dep['head'][i] - 1  # 依存句法关系中的头
#             if head >= 0:  # 头存在（注意有些标点符号头为-1）
#                 label = dep['label'][i]  # 依存句法关系中的标签
#                 graph['edges'].append((pos[0][i], label, pos[0][head]))
#         # 构建边
#         edge_index = []
#         for edge in graph['edges']:
#             if edge[0] in graph['nodes'] and edge[2] in graph['nodes']:
#                 src = list(graph['nodes']).index(edge[0])
#                 dst = list(graph['nodes']).index(edge[2])
#                 edge_index.append([src, dst])
#         # print(edge_index)  #[[1, 0], [4, 2], [3, 2], [2, 0]]
#         if edge_index == []:
#             edge_index = [[0, 0]]
#         edge_index = torch.tensor(edge_index).t().contiguous()
#         # x = []
#         # for node in graph['nodes']:
#         #     x.append(list(graph['nodes']).index(node))
#         x = torch.tensor(BertTokenizer.from_pretrained(
#             '/home/star/anaconda3/envs/syn_prompt/lib/python3.9/site-packages/openprompt/plms/chinese-roberta-wwm-ext').convert_tokens_to_ids(
#             pos[0]), dtype=torch.float)
#         # x = x.to(torch.float32)
#         data = Data(x=x.to('cuda'), edge_index=torch.as_tensor(edge_index).to('cuda'))
#         # data = Data(x=torch.tensor(x).to('cuda'), edge_index=torch.tensor(edge_index).to('cuda'))
#         return data
#
#     def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
#         r"""
#         Get the logits of label words.
#
#         Args:
#             batch (:obj:`Union[Dict, InputFeatures]`): The original batch
#
#         Returns:
#             :obj:`torch.Tensor`: The logits of the lable words (obtained by the current verbalizer).
#         """
#         # print(batch['gat_inputs'])
#         # print(batch['input_ids'][10: 9 +len(wrapped_example[0][3]['text'])])
#         # merged_dict = {k: v for k, v in zip(batch['title_len'],batch['input_ids'])}
#         # for k, v in merged_dict.items():
#         #     print(v[10: 9 + k])             #标题的vocab索引
#         # print(merged_dict)
#         outputs = self.prompt_model(batch)  # torch.Size([4, 250, 768]) batch, max_seq_l, vector_dim
#         outputs = self.verbalizer.gather_outputs(outputs)
#         # print('outputs:', outputs.shape)  #(batch_size, max_seq_length, vocab_size) outputs: torch.Size([2, 128, 21128])
#
#
#         titles = batch['title']
#         gat_inputs = []
#         for title in titles:
#             # print(title)
#             gat_input = self.title_to_pos_graph(title)
#             # 使用 unsqueeze 将 x 的维度从 1 扩展到 2
#             gat_input.x = gat_input.x.unsqueeze(1)
#             # print(gat_input.x.size())  # 输出 torch.Size([9, 1])
#
#             # 使用 repeat 将 x 沿着新的维度复制 300 次
#             gat_input.x = gat_input.x.repeat(1, 64)
#             # print(gat_input.x.shape)
#             # gat_input = gat_input.x.to(torch.float32).to(device)
#             # gat_input = gat_input.x.to(torch.float32).to(device)
#             # print(gat_input.edge_index)
#             gat_output = self.gat(gat_input.x, gat_input.edge_index)
#             # print('1: ', gat_output.shape)
#             gat_output = F.elu(gat_output)
#             # print('2: ', gat_output)
#             gat_inputs.append(gat_output)
#
#
#         # print(gat_inputs)
#         # gat
#
#         # gat_outputs = self.gat(gat_inputs.x, gat_inputs.edge_index)
#         # gat_outputs = F.elu(gat_outputs)
#         # print(gat_outputs)
#
#         # 从模型的输出中提取出所有"<mask>"标记的输出  (batch_size, max_seq_length, vocab_size) --> (batch_size, num_mask_token, vocab_size)
#         outputs_at_mask = self.extract_at_mask(outputs,
#                                                batch)  # torch_size([4, 21128]) 只有一个mask (batch_size, vocab_size)
#         # print(outputs_at_mask)
#
#         # num_vectors 表示需要拼接的向量个数
#         num_vectors = outputs_at_mask.shape[0]
#         compressed_vectors = []
#         for i in range(num_vectors):
#             compressed_vector = torch.mean(gat_inputs[i], dim=0, keepdim=True)
#             compressed_vectors.append(compressed_vector)
#         concatenated = torch.cat(compressed_vectors, dim=0)
#
#         outputs_at_mask_pos = torch.cat((outputs_at_mask, concatenated), dim=1)
#
#         # 从模型的输出中提取出所有"<mask>"标记的输出  (batch_size, max_seq_length, vocab_size) --> (batch_size, num_mask_token, vocab_size)
#         # outputs_at_mask = self.extract_at_mask(outputs, batch)  # torch_size([4, 21128]) 只有一个mask (batch_size, vocab_size)
#         # print(outputs_at_mask_pos.shape)
#         label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
#         # label_words_logits = self.verbalizer.process_outputs(outputs_at_mask_pos, batch=batch)
#         # print(label_words_logits)
#         return label_words_logits
#
#     def predict(self):
#         pass
#
#     def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
#         outputs = self.prompt_model(batch)
#         outputs = self.verbalizer.gather_outputs(outputs)
#         outputs_at_mask = self.extract_at_mask(outputs, batch)
#         return outputs_at_mask
#
#     @property
#     def tokenizer(self):
#         r'''Utility property, to get the tokenizer more easily.
#         '''
#         return self.verbalizer.tokenizer
#
#     def state_dict(self, *args, **kwargs):
#         """ Save the model using template, plm and verbalizer's save methods."""
#         _state_dict = {}
#         if not self.prompt_model.freeze_plm:
#             _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
#         _state_dict['template'] = self.template.state_dict(*args, **kwargs)
#         _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
#         return _state_dict
#
#     def load_state_dict(self, state_dict, *args, **kwargs):
#         """ Load the model using template, plm and verbalizer's load methods."""
#         if 'plm' in state_dict and not self.prompt_model.freeze_plm:
#             self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
#         self.template.load_state_dict(state_dict['template'], *args, **kwargs)
#         self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)
#
#     def parallelize(self, device_map=None):
#         r"""Parallelize the model across device
#         """
#         if hasattr(self.plm, "parallelize"):
#             self.plm.parallelize(device_map)
#             self.device_map = self.plm.device_map
#             self.template.cuda()
#             self.verbalizer.cuda()
#         else:
#             raise NotImplementedError("parallelize method was not implemented for this plm.")
#
#     def deparallelize(self):
#         r"""Deparallelize the model across device
#         """
#         if hasattr(self.plm, "deparallelize"):
#             self.plm.deparallelize()
#             self.device_map = None
#             self.template.cpu()
#             self.verbalizer.cpu()
#         else:
#             raise NotImplementedError("parallelize method was not implemented for this plm.")


'''GCN'''
# 只GCN
# class PromptForClassification(nn.Module):
#     r'''``PromptModel`` with a classification head on top. The classification head will map
#     the logits in all position of the sequence (return value of a ``PromptModel``) into the
#     logits of the labels, using a verbalizer.
#
#     Args:
#         plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
#         template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
#         verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the lables to label words for classification, e.g. ``ManualVerbalizer``.
#         freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
#         plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
#     '''
#     def __init__(self,
#                  plm: PreTrainedModel,
#                  template: Template,
#                  verbalizer: Verbalizer,
#                  freeze_plm: bool = False,
#                  plm_eval_mode: bool=False,
#                  in_channels: int=64,
#                  out_channels: int=128
#                 ):
#         super().__init__()
#         # self.gat = GATConv(in_channels=64, out_channels=out_channels)
#         self.gcn = GCNConv(in_channels=64, out_channels=128)
#         self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
#         self.verbalizer = verbalizer
#
#
#     @property
#     def plm(self):
#         return self.prompt_model.plm
#
#     @property
#     def template(self):
#         return self.prompt_model.template
#
#     @property
#     def device(self,):
#         r"""Register the device parameter."""
#         return self.plm.device
#
#     def extract_at_mask(self,
#                        outputs: torch.Tensor,
#                        batch: Union[Dict, InputFeatures]):
#         r"""Get outputs at all <mask> token
#         E.g., project the logits of shape
#         (``batch_size``, ``max_seq_length``, ``vocab_size``)
#         into logits of shape (if num_mask_token > 1)
#         (``batch_size``, ``num_mask_token``, ``vocab_size``)
#         or into logits of shape (if ``num_mask_token`` = 1)
#         (``batch_size``, ``vocab_size``).
#
#
#         Args:
#             outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
#                  `gather_outputs` before) etc. of the whole sequence.
#             batch (:obj:`Union[Dict, InputFeatures]`): The original batch
#
#         Returns:
#             :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.
#
#         """
#         outputs = outputs[torch.where(batch['loss_ids']>0)]
#
#         outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
#         if outputs.shape[1] == 1:
#             outputs = outputs.view(outputs.shape[0], outputs.shape[2])
#         return outputs
#     def title_to_pos_graph(self, title):
#         ltp = LTP(pretrained_model_name_or_path='/home/star/anaconda3/envs/syn_prompt/lib/python3.9/site-packages/ltp/ltp_model')
#         #调用ltp
#         output = ltp.pipeline([title], tasks=["cws", "pos", "sdp", "dep"])
#         cws = output.cws  # 分词
#         pos = output.pos  # 词性标注  #{'nt', 'q', 'n', 'a', 'v'}
#         dep = output.dep[0]  # 依存句法结构
#         graph = {'nodes': set(pos[0]), 'edges': []}
#         for i, word in enumerate(cws[0]):
#             head = dep['head'][i] - 1  # 依存句法关系中的头
#             if head >= 0:  # 头存在（注意有些标点符号头为-1）
#                 label = dep['label'][i]  # 依存句法关系中的标签
#                 graph['edges'].append((pos[0][i], label, pos[0][head]))
#         #构建边
#         edge_index = []
#         for edge in graph['edges']:
#             if edge[0] in graph['nodes'] and edge[2] in graph['nodes']:
#                 src = list(graph['nodes']).index(edge[0])
#                 dst = list(graph['nodes']).index(edge[2])
#                 edge_index.append([src, dst])
#         # print(edge_index)  #[[1, 0], [4, 2], [3, 2], [2, 0]]
#         if edge_index == []:
#             edge_index = [[0, 0]]
#         edge_index = torch.tensor(edge_index).t().contiguous()
#         # x = []
#         # for node in graph['nodes']:
#         #     x.append(list(graph['nodes']).index(node))
#         x = torch.tensor(BertTokenizer.from_pretrained('/home/star/anaconda3/envs/syn_prompt/lib/python3.9/site-packages/openprompt/plms/chinese-roberta-wwm-ext').convert_tokens_to_ids(pos[0]), dtype=torch.float)
#         # x = x.to(torch.float32)
#         # data = Data(x=x.to('cuda'), edge_index=torch.as_tensor(edge_index).to('cuda'))
#         data = Data(x,edge_index=torch.as_tensor(edge_index))
#         # data = Data(x=torch.tensor(x).to('cuda'), edge_index=torch.tensor(edge_index).to('cuda'))
#         '''
#         data:
# Data(x=[19], edge_index=[2, 18])
#
#         '''
#         # print('data:')
#         # print(data)
#         return data
#
#
#
#     def title_to_en_pos_graph(self, title):
#         # Load SpaCy English model
#         nlp = spacy.load("en_core_web_sm")
#
#         # Process English title using SpaCy
#         doc = nlp(title)
#
#         # Extract information
#         pos = [token.pos_ for token in doc]
#         dep = [(token.dep_, token.head.text) for token in doc]
#
#         # Build graph representation
#         graph = {'nodes': set(pos), 'edges': []}
#         for i, token in enumerate(doc):
#             head = token.head.text
#             label = dep[i][0]
#             graph['edges'].append((pos[i], label, head))
#
#         # Edge indexing
#         edge_index = []
#         for edge in graph['edges']:
#             if edge[0] in graph['nodes'] and edge[2] in graph['nodes']:
#                 src = list(graph['nodes']).index(edge[0])
#                 dst = list(graph['nodes']).index(edge[2])
#                 edge_index.append([src, dst])
#
#         if edge_index == []:
#             edge_index = [[0, 0]]
#         edge_index = torch.tensor(edge_index).t().contiguous()
#
#         # Convert node features using BERT tokenizer
#         tokenizer = BertTokenizer.from_pretrained('/home/mjy/wangye/text_classification/model/bert-case')
#         x = torch.tensor(tokenizer.convert_tokens_to_ids([token.text for token in doc]), dtype=torch.float)
#
#         # Create PyTorch Data object
#         data = Data(x=x.to('cuda'), edge_index=edge_index.to('cuda'))
#         # print(data)
#         return data
#
#     def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
#         r"""
#         Get the logits of label words.
#
#         Args:
#             batch (:obj:`Union[Dict, InputFeatures]`): The original batch
#
#         Returns:
#             :obj:`torch.Tensor`: The logits of the lable words (obtained by the current verbalizer).
#         """
#         # print(batch['gat_inputs'])
#         # print(batch['input_ids'][10: 9 +len(wrapped_example[0][3]['text'])])
#         # merged_dict = {k: v for k, v in zip(batch['title_len'],batch['input_ids'])}
#         # for k, v in merged_dict.items():
#         #     print(v[10: 9 + k])             #标题的vocab索引
#         # print(merged_dict)
#         outputs = self.prompt_model(batch)  #torch.Size([4, 250, 768]) batch, max_seq_l, vector_dim
#         outputs = self.verbalizer.gather_outputs(outputs)
#         # print('outputs:', outputs.shape)  #(batch_size, max_seq_length, vocab_size) outputs: torch.Size([2, 128, 21128])
#
#
#         titles = batch['title']
#         gcn_inputs = []
#         for title in titles:
#             # print(title)
#             gcn_input = self.title_to_pos_graph(title)
#             # gcn_input = self.title_to_en_pos_graph(title)
#
#             # 使用 unsqueeze 将 x 的维度从 1 扩展到 2
#             gcn_input.x = gcn_input.x.unsqueeze(1)
#             # print(gat_input.x.size())  # 输出 torch.Size([9, 1])
#
#             # 使用 repeat 将 x 沿着新的维度复制 300 次
#             gcn_input.x = gcn_input.x.repeat(1, 64)
#             # print(gat_input.x.shape)
#             # gat_input = gat_input.x.to(torch.float32).to(device)
#             # gat_input = gat_input.x.to(torch.float32).to(device)
#             # print(gat_input.edge_index)
#             gcn_output = self.gcn(gcn_input.x.cuda(), gcn_input.edge_index.cuda())
#             # print('1: ', gat_output.shape)
#             gcn_output = F.elu(gcn_output)
#             # print('2: ', gat_output)
#             gcn_inputs.append(gcn_output)
#
#
#         # print(gat_inputs)
#         #gat
#
#         # gat_outputs = self.gat(gat_inputs.x, gat_inputs.edge_index)
#         # gat_outputs = F.elu(gat_outputs)
#         # print(gat_outputs)
#
#
#         #从模型的输出中提取出所有"<mask>"标记的输出  (batch_size, max_seq_length, vocab_size) --> (batch_size, num_mask_token, vocab_size)
#         outputs_at_mask = self.extract_at_mask(outputs, batch) #torch_size([4, 21128]) 只有一个mask (batch_size, vocab_size)
#         # print(outputs_at_mask)
#
#         # num_vectors 表示需要拼接的向量个数
#         num_vectors = outputs_at_mask.shape[0]
#         compressed_vectors = []
#         for i in range(num_vectors):
#             compressed_vector = torch.mean(gcn_inputs[i], dim=0, keepdim=True)
#             compressed_vectors.append(compressed_vector)
#         concatenated = torch.cat(compressed_vectors, dim=0)
#
#         outputs_at_mask_pos = torch.cat((outputs_at_mask, concatenated), dim=1)
#
#         # 从模型的输出中提取出所有"<mask>"标记的输出  (batch_size, max_seq_length, vocab_size) --> (batch_size, num_mask_token, vocab_size)
#         # outputs_at_mask = self.extract_at_mask(outputs, batch)  # torch_size([4, 21128]) 只有一个mask (batch_size, vocab_size)
#         # print(outputs_at_mask_pos.shape)
#         label_words_logits = self.verbalizer.process_outputs(outputs_at_mask_pos, batch=batch)
#         # label_words_logits = self.verbalizer.process_outputs(outputs_at_mask_pos, batch=batch)
#         # print(label_words_logits)
#         return label_words_logits
#
#     def predict(self):
#         pass
#
#     def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
#         outputs = self.prompt_model(batch)
#         outputs = self.verbalizer.gather_outputs(outputs)
#         outputs_at_mask = self.extract_at_mask(outputs, batch)
#         return outputs_at_mask
#
#     @property
#     def tokenizer(self):
#         r'''Utility property, to get the tokenizer more easily.
#         '''
#         return self.verbalizer.tokenizer
#
#     def state_dict(self, *args, **kwargs):
#         """ Save the model using template, plm and verbalizer's save methods."""
#         _state_dict = {}
#         if not self.prompt_model.freeze_plm:
#             _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
#         _state_dict['template'] = self.template.state_dict(*args, **kwargs)
#         _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
#         return _state_dict
#
#     def load_state_dict(self, state_dict, *args, **kwargs):
#         """ Load the model using template, plm and verbalizer's load methods."""
#         if 'plm' in state_dict and not self.prompt_model.freeze_plm:
#             self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
#         self.template.load_state_dict(state_dict['template'], *args, **kwargs)
#         self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)
#
#     def parallelize(self, device_map=None):
#         r"""Parallelize the model across device
#         """
#         if hasattr(self.plm, "parallelize"):
#             self.plm.parallelize(device_map)
#             self.device_map = self.plm.device_map
#             # self.template.cuda()
#             # self.verbalizer.cuda()
#             self.template.cpu()
#             self.verbalizer.cpu()
#         else:
#             raise NotImplementedError("parallelize method was not implemented for this plm.")
#
#     def deparallelize(self):
#         r"""Deparallelize the model across device
#         """
#         if hasattr(self.plm, "deparallelize"):
#             self.plm.deparallelize()
#             self.device_map = None
#             self.template.cpu()
#             self.verbalizer.cpu()
#         else:
#             raise NotImplementedError("parallelize method was not implemented for this plm.")


'''无啥'''
class PromptForClassification(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a ``PromptModel``) into the
    logits of the labels, using a verbalizer.

    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the lables to label words for classification, e.g. ``ManualVerbalizer``.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''

    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False,
                 in_channels: int = 64,
                 out_channels: int = 128
                 ):
        super().__init__()
        self.gat = GATConv(in_channels=64, out_channels=out_channels)

        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self, ):
        r"""Register the device parameter."""
        return self.plm.device

    def extract_at_mask(self,
                        outputs: torch.Tensor,
                        batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).


        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        outputs = outputs[torch.where(batch['loss_ids'] > 0)]

        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def title_to_pos_graph(self, title):
        ltp = LTP(
            pretrained_model_name_or_path='/home/star/anaconda3/envs/syn_prompt/lib/python3.9/site-packages/ltp/ltp_model')
        # 调用ltp
        output = ltp.pipeline([title], tasks=["cws", "pos", "sdp", "dep"])
        cws = output.cws  # 分词
        pos = output.pos  # 词性标注  #{'nt', 'q', 'n', 'a', 'v'}
        dep = output.dep[0]  # 依存句法结构
        graph = {'nodes': set(pos[0]), 'edges': []}
        for i, word in enumerate(cws[0]):
            head = dep['head'][i] - 1  # 依存句法关系中的头
            if head >= 0:  # 头存在（注意有些标点符号头为-1）
                label = dep['label'][i]  # 依存句法关系中的标签
                graph['edges'].append((pos[0][i], label, pos[0][head]))
        # 构建边
        edge_index = []
        for edge in graph['edges']:
            if edge[0] in graph['nodes'] and edge[2] in graph['nodes']:
                src = list(graph['nodes']).index(edge[0])
                dst = list(graph['nodes']).index(edge[2])
                edge_index.append([src, dst])
        # print(edge_index)  #[[1, 0], [4, 2], [3, 2], [2, 0]]
        if edge_index == []:
            edge_index = [[0, 0]]
        edge_index = torch.tensor(edge_index).t().contiguous()
        # x = []
        # for node in graph['nodes']:
        #     x.append(list(graph['nodes']).index(node))
        x = torch.tensor(BertTokenizer.from_pretrained(
            '/home/star/anaconda3/envs/syn_prompt/lib/python3.9/site-packages/openprompt/plms/chinese-roberta-wwm-ext').convert_tokens_to_ids(
            pos[0]), dtype=torch.float)
        # x = x.to(torch.float32)
        data = Data(x=x.to('cuda'), edge_index=torch.as_tensor(edge_index).to('cuda'))
        # data = Data(x=torch.tensor(x).to('cuda'), edge_index=torch.tensor(edge_index).to('cuda'))
        return data

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the lable words (obtained by the current verbalizer).
        """
        # print(batch['gat_inputs'])
        # print(batch['input_ids'][10: 9 +len(wrapped_example[0][3]['text'])])
        # merged_dict = {k: v for k, v in zip(batch['title_len'],batch['input_ids'])}
        # for k, v in merged_dict.items():
        #     print(v[10: 9 + k])             #标题的vocab索引
        # print(merged_dict)
        outputs = self.prompt_model(batch)  # torch.Size([4, 250, 768]) batch, max_seq_l, vector_dim
        outputs = self.verbalizer.gather_outputs(outputs)
        # print('outputs:', outputs.shape)  #(batch_size, max_seq_length, vocab_size) outputs: torch.Size([2, 128, 21128])

        '''
        titles = batch['title']
        gat_inputs = []
        for title in titles:
            # print(title)
            gat_input = self.title_to_pos_graph(title)
            # 使用 unsqueeze 将 x 的维度从 1 扩展到 2
            gat_input.x = gat_input.x.unsqueeze(1)
            # print(gat_input.x.size())  # 输出 torch.Size([9, 1])

            # 使用 repeat 将 x 沿着新的维度复制 300 次
            gat_input.x = gat_input.x.repeat(1, 64)
            # print(gat_input.x.shape)
            # gat_input = gat_input.x.to(torch.float32).to(device)
            # gat_input = gat_input.x.to(torch.float32).to(device)
            # print(gat_input.edge_index)
            gat_output = self.gat(gat_input.x, gat_input.edge_index)
            # print('1: ', gat_output.shape)
            gat_output = F.elu(gat_output)
            # print('2: ', gat_output)
            gat_inputs.append(gat_output)
         '''


        # print(gat_inputs)
        # gat

        # gat_outputs = self.gat(gat_inputs.x, gat_inputs.edge_index)
        # gat_outputs = F.elu(gat_outputs)
        # print(gat_outputs)

        # 从模型的输出中提取出所有"<mask>"标记的输出  (batch_size, max_seq_length, vocab_size) --> (batch_size, num_mask_token, vocab_size)
        outputs_at_mask = self.extract_at_mask(outputs,
                                               batch)  # torch_size([4, 21128]) 只有一个mask (batch_size, vocab_size)
        # print(outputs_at_mask)

        # num_vectors 表示需要拼接的向量个数
        # num_vectors = outputs_at_mask.shape[0]
        # compressed_vectors = []
        # for i in range(num_vectors):
        #     compressed_vector = torch.mean(gat_inputs[i], dim=0, keepdim=True)
        #     compressed_vectors.append(compressed_vector)
        # concatenated = torch.cat(compressed_vectors, dim=0)
        #
        # outputs_at_mask_pos = torch.cat((outputs_at_mask, concatenated), dim=1)

        # 从模型的输出中提取出所有"<mask>"标记的输出  (batch_size, max_seq_length, vocab_size) --> (batch_size, num_mask_token, vocab_size)
        # outputs_at_mask = self.extract_at_mask(outputs, batch)  # torch_size([4, 21128]) 只有一个mask (batch_size, vocab_size)
        # print(outputs_at_mask_pos.shape)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        # label_words_logits = self.verbalizer.process_outputs(outputs_at_mask_pos, batch=batch)
        # print(label_words_logits)
        return label_words_logits

    def predict(self):
        pass

    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self, *args, **kwargs):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
        _state_dict['template'] = self.template.state_dict(*args, **kwargs)
        _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
        return _state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
        self.template.load_state_dict(state_dict['template'], *args, **kwargs)
        self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.verbalizer.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.verbalizer.cpu()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")


class PromptForGeneration(nn.Module, GenerationMixin):
    r'''``PromptModel`` with generation loss caculation and generation utils integrated.


    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for generation, e.g. GPT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``PrefixTemplate``.
        tokenizer (:obj:`Tokenizer`): A ``Tokenizer`` of the current model.
        gen_config (:obj:`CfgNode`): The generation configs to pass into `GenerationMixin.generate <https://huggingface.co/transformers/_modules/transformers/generation_utils.html#GenerationMixin.generate>`_
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train. 
    '''

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False,
                 gen_config: Optional[CfgNode] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                ):
        super().__init__()
        self.freeze_plm = freeze_plm
        if tokenizer is None:
            assert template.tokenizer is not None, "Tokenizer can't be set from input args or template"
            self.tokenizer = template.tokenizer
        else:
            self.tokenizer = tokenizer
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.config = plm.config
        if gen_config:
            for key in gen_config:
                setattr(self.config, key, gen_config[key])
        self.in_generation_function = False

    @property
    def plm(self):
        return self.prompt_model.plm
    
    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self):
        return self.plm.device
    

    def shift_logits_and_labels(self, 
                                logits, 
                                loss_ids, 
                                reference_ids):

        r"""
        Left shift the label, and make label of the positions that are
        not loss position to -100, which is the ignore index in pytorch's
        loss function.

        Args:
            logits (:obj:`torch.Tensor`):
            batch (:obj:`InputFeatures`): The input features of batchified data sequences.
        
        Returns:
            shift_logits (:obj:`torch.Tensor`):
            shift_input_ids (:obj:`List[int]`):

        """
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_loss_ids = loss_ids[..., 1:].contiguous()
        shift_input_ids = reference_ids[..., 1:].contiguous()
        shift_input_ids = torch.where(shift_loss_ids>0, shift_input_ids, -100)
        return shift_logits, shift_input_ids

    def forward(self, *args, **kwargs):
        r"""In generation process, it will use the plm's forward function.
        This is because, in the first step we will directly call the process_batch function to 
        generate initial input with the template, after that the all template
        have been processed into the past_key_value,
        then we can use the normal generation function. 
        In learning process, the forward is linked to ``_forward`` functions.
        in which the loss will be calculated for all the positions in the same time. 
        """
        if self.in_generation_function:
            return self.plm.forward(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" 
        This is the forward method of the training of generation in prompt-learning framework. 
        
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        
        Returns:
            loss(:obj:torch.Tensor): The loss of the current generation procedure.
        """
        if self.config.is_encoder_decoder:
            reference_ids = batch['decoder_input_ids']
        else:
            reference_ids = batch['input_ids']  # in case in some template, these field is dropped
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        logits, labels = self.shift_logits_and_labels(logits, batch['loss_ids'], reference_ids)
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(batch_size, -1).sum(dim=-1) #TODO support more objectives
        loss = loss.mean()
        return loss
    
    
    def generate(self, batch: Union[Dict, InputFeatures], verbose: Optional[bool]=False, **generation_kwargs):
        r""" This function wraps the generate() methods in parent class ``GenerationMixin``.
        Forward uses the ``PretrainedModel``'s forward method. 
        generation_kwargs include all the parameters that are passed in to 
        ``transformers.generation_util.GenerationMixin.generate``
    
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
            verbose (:obj:`Optional[bool]`): Set to true to verbose the generated sentence. 
        
        Returns:
            output_sequences (:obj:`List[torch.Tensor]`): The raw sequences generated by the generation model.
            generated_sentences (:obj:`List[torch.Tensor]`): The generated sentences that have been post-processed.
        """
        input_generation_kwargs = {key: value for key,value in generation_kwargs.items() if key in signature(GenerationMixin.generate).args}
        if self.config.is_encoder_decoder:
            loss_ids_start = batch['loss_ids'].argmax(dim=-1)
            assert loss_ids_start.min() == loss_ids_start.max(), "The generation start from different position in a batch."
            batch['decoder_input_ids'] = batch['decoder_input_ids'][:, :loss_ids_start.min()+1]
            input_length = batch['decoder_input_ids'].size(1)
            batch_size = batch['decoder_input_ids'].size(0)

            self.generate_ith_token = 0
            self.in_generation_function = True
            output_sequences = super().generate(**batch, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            self.in_generation_function = False
            output_sequences = output_sequences.cpu().tolist()
            generated_sentences = self.post_processing(output_sequences=output_sequences, input_lengths=input_length)
        else:
            input_length = batch['input_ids'].size(1)
            batch_size = batch['input_ids'].size(0)
            
            # Currently huggingface transformers only support single sample generation, or padding to the left (instead of the right).
            # because it will only extract the last position of the output 
            # generate one_by_one
            if 'input_ids_len' in batch:
                input_real_lens = batch['input_ids_len']
            else:
                input_real_lens = torch.sum((batch['input_ids'] != self.tokenizer.pad_token_id).to(torch.int), dim=-1)
            output_sequences = []
            for instance_id in range(batch_size):  
                # remove the pad token 
                instance = {key: batch[key][instance_id:instance_id+1][:,:input_real_lens[instance_id]] for key in batch if isinstance(batch[key], torch.Tensor) and batch[key].shape[:2]==torch.Size([batch_size, input_length])}
                self.generate_ith_token = 0
                self.in_generation_function = True
                output_sequence = super().generate(**instance, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                self.in_generation_function = False
                output_sequences.extend(output_sequence.cpu().tolist()) # TODO: to support generate multiple sentence
            generated_sentences = self.post_processing(output_sequences=output_sequences, input_lengths=input_real_lens.cpu().tolist())
        if verbose:
            logger.info(f"Generated:{generated_sentences}")
        return output_sequences, generated_sentences
    


    def post_processing(self, output_sequences, input_lengths):
        r"""
            Post-process the sequences generated by the generation model.

            Args:
                output_sequences (:obj:`torch.Tensor`): The raw sequences generated by the generation model.
                input_lengths (:obj:`int` or `list`): The length(s) of the input sequence.
            
            Returns:
                :obj:`List`: The generated sentences that have been post-processed.
        """
        generated_sentences = []
        if type(input_lengths)==int:
            input_lengths = [input_lengths] * len(output_sequences)
        for sent_id, seq in enumerate(output_sequences):
            seq = seq[input_lengths[sent_id]:]
            text_output = self.tokenizer.decode(seq, clean_up_tokenization_spaces=True)
            idx = text_output.find(self.tokenizer.eos_token)
            if idx >= 0:
                text_output = text_output[:idx]
            text_output = text_output.strip()
            generated_sentences.append(text_output)
        return generated_sentences


    
    def prepare_inputs_for_generation(self, input_ids: Optional[torch.Tensor] = None,
                                         **model_kwargs):
        r"""This function wraps the ``prepare_inputs_for_generation`` function in the huggingface transformers.

        When the `past` not in model_kwargs, we prepare the input from scratch. 
        When `past` is in model_kwargs, we don't need to prepare the template wrapped input,
        instead we use the inner pretrain_models' function to prepare the next step's input.
        `model_kwargs` includes all the argument passed in the `batch`: InputFeatures, except ``input_ids``
        , as long as they do not conflict with keywords in ``generation_kwargs``.    if 'past' not in model_kwargs: # the past_key_value not in model_kwargs, then we need to prepare input from scrath
        , as long as they do not conflict with keywords in ``generation_kwargs``.

        Args:
            input_ids(:obj:`torch.Tensor`): Indices of input sequence tokens in the vocabulary.
        """
        if self.generate_ith_token == 0 and 'encoder_outputs' not in model_kwargs: # generating the first token in decoder only setting.

            batch = InputFeatures(input_ids=input_ids, **model_kwargs)
            model_inputs = self.prompt_model.prepare_model_inputs(batch)
            # check the competibility for more models. Having checked gpt2, T5
        else: # generating the subsequence generation can use the default setting
            model_inputs = self.plm.prepare_inputs_for_generation(input_ids, **model_kwargs)
        self.last_model_inputs = model_inputs  # to update the model_kwargs in _update_model_kwargs_for_generation, in-place operation.
        return model_inputs
    
    
    def _update_model_kwargs_for_generation(self,
        outputs, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        r""" The parents class's ``_update_model_kwargs_for_generation`` method will
        add ``past_key_values`` to model_kwargs, and update ``token_type_ids``, and ``attention_mask_ids``.

        In case some of the model_kwargs are modified in the prepare_inputs_for_generation function
        and should be used as the subsequent model_kwargs, we upate these kwargs before the parent class
        call. 

        Other updates should be added here after the parent's function call.

        Args:
            outputs (:obj:`torch.Tensor`): 
            is_encoder_decoder (:obj:`bool`, defaults to False): 
        """
        if self.generate_ith_token == 0:
            for key in self.last_model_inputs:
                if key in model_kwargs:
                    model_kwargs[key] = self.last_model_inputs[key]
        model_kwargs = super(PromptForGeneration, PromptForGeneration)._update_model_kwargs_for_generation(outputs=outputs, model_kwargs=model_kwargs, is_encoder_decoder=is_encoder_decoder)
        self.generate_ith_token += 1
        return model_kwargs


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        r""" This function resemble the function in GeneraionMix
        
        Args:
            input_ids (:obj:`torch.LongTensor`) The input ids for 
        """
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.plm.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            batch = {"input_ids":input_ids, **encoder_kwargs}
            model_inputs = self.prompt_model.prepare_model_inputs(batch) # This line differs from the orinigal code base, we should process the input
            # with our template, then pass it into the model.
            # some of the arguments may have been changed by the template,
            # e.g. the attention mask. Here we update the model_kwargs
            for key in model_kwargs:
                if key in model_inputs:
                    model_kwargs[key] = model_inputs[key]
            model_kwargs["encoder_outputs"] = encoder(return_dict=True, **model_inputs)
        return model_kwargs
    
    def state_dict(self, *args, **kwargs):
        """ Save the model using template and plm's save methods. """
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
        _state_dict['template'] = self.template.state_dict(*args, **kwargs)
        return _state_dict
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """ Load the model using template and plm's load methods. """
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
        self.template.load_state_dict(state_dict['template'], *args, **kwargs)
    
    def _reorder_cache(self, past, beam_idx):
        r"""Use the plm's default _reorder_cache function
        """
        return self.plm._reorder_cache(past, beam_idx)

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")
