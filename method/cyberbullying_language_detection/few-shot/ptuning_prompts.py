
from openprompt.data_utils import InputFeatures
import os
import torch
from torch import nn
from typing import *
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.prompts import ManualTemplate, ManualVerbalizer

class PtuningTemplate(ManualTemplate):
    """
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        prompt_encoder_type (:obj:`str`): head above the embedding layer of new tokens. Can be ``lstm`` or ``mlp``.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        mask_token (:obj:`str`, optional): The special token that is masked and need to be predicted by the model. Default to ``<mask>``
        soft_token (:obj:`str`, optional): The special token for new token. Default to ``<soft>``
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """

    registered_inputflag_names = ["soft_token_ids", "loss_ids", 'shortenable_ids']

    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 prompt_encoder_type: str = "bilstm",
                 text:  Optional[List[str]] = None,
                 # mask_token: str = '<mask>',
                 soft_token: str = '<soft>',
                 placeholder_mapping: dict = {'<text_a>':'text_a', '<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer, 
                         # mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.prompt_encoder_type = prompt_encoder_type
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.soft_token = soft_token
        self.text = text

    def get_default_soft_token_ids(self) -> List[int]:
        r"""get the new token indices for the template
        e.g. when self.text is ['<text_a>', '<soft>', '<soft>', '<mask>', '.'],
        output is [0, 1, 2, 0, 0]
        """
        # TODO ptuing supervised use same new token for each <soft> ?
        idx = []
        num_soft_token = 0
        # print(self.text)
        # [{'add_prefix_space': '', 'soft': '<soft>'}, {'add_prefix_space': ' ', 'mask': '<soft>'},
        #  {'add_prefix_space': '', 'soft': '<soft>'}, {'add_prefix_space': ' ', 'placeholder': 'text_a'}]
        tokens = []
        for item in self.text:
            for key, value in item.items():
                if key != 'add_prefix_space':
                    tokens.append(value)
                    break
        # self.text = tokens
        for token in tokens:

            if token == self.soft_token:
                num_soft_token += 1
                idx.append(num_soft_token)
            else:
                idx.append(0)
        # print(idx)
        return idx

    def on_text_set(self):
        r"""
        when template text was set, generate parameters needed in p-tuning input embedding phrase
        """
        print(self.text)
        self.text = self.parse_text(self.text)
        print(self.text)
        tokens = []
        for item in self.text:
            for key, value in item.items():
                if key != 'add_prefix_space':
                    tokens.append(value)
                    break
        self.num_soft_token = sum([token == self.soft_token for token in tokens])
        print(self.num_soft_token)
        self.generate_parameters()

    def generate_parameters(self) -> None:
        r"""
        generate parameters needed for new tokens' embedding in P-tuning
        """
        if self.num_soft_token == 0:
            return
        self.new_embedding = nn.Embedding(self.num_soft_token, self.embedding_size)
        self.new_ids = nn.Parameter(torch.LongTensor(list(range(self.num_soft_token))), requires_grad = False)
        if self.prompt_encoder_type == "bilstm":
            self.new_bilstm_head = nn.LSTM(
                input_size = self.embedding_size,
                hidden_size = self.embedding_size, # TODO P-tuning different in LAMA & FewGLUE
                # TODO dropout different in LAMA and FewGLUE
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            self.new_mlp_head = nn.Sequential(
                nn.Linear(2*self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.prompt_encoder_type == "mlp":
            self.new_mlp_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.prompt_encoder_type == "lstm":
            self.new_lstm_head = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.embedding_size,
                num_layers=2,
                bidirectional=False,
                batch_first=True
            )
            self.new_mlp_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),  # 调整线性层的输出维度为单向 LSTM 的输出维度
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.prompt_encoder_type == "cnn":
            kernel_size = 3  # 定义卷积核大小

            self.new_cnn_head = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.embedding_size,
                                          kernel_size=kernel_size, padding=1)
            self.new_mlp_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.prompt_encoder_type == "transformer":
            num_layers = 2  # 定义 Transformer 编码器的层数
            self.new_transformer_head = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=2)
            self.transformer_encoder = nn.TransformerEncoder(self.new_transformer_head, num_layers=num_layers)
            self.new_mlp_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        else:
            raise ValueError("unknown prompt_enocder_type")
            
    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        r"""
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for new tokens, use a brand new embedding layer, with MLP or LSTM head
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        # print(inputs_embeds.shape)
        # print(self.num_soft_token)   #0
        if self.num_soft_token != 0:
            new_embeds = self.new_embedding(self.new_ids).unsqueeze(0)
            # print('new_embeds')
            # print(self.prompt_encoder_type)
            if self.prompt_encoder_type == "lstm":
                lstm_output, _ = self.new_lstm_head(new_embeds)

                new_embeds = lstm_output
                new_embeds = self.new_mlp_head(new_embeds)
                # print(new_embeds)
            elif self.prompt_encoder_type == "bilstm":
                new_embeds, _ = self.new_bilstm_head(new_embeds)  # 使用双向 LSTM 处理新标记嵌入
                # new_embeds = self.new_mlp_head(new_embeds)
            elif self.prompt_encoder_type == "cnn":
                # print('cnn---')
                new_embeds = self.new_cnn_head(new_embeds.permute(0, 2, 1)).permute(0, 2, 1)  # 处理新标记嵌入
                # new_embeds = self.new_cnn_head(new_embeds)  # 使用 CNN 头部处理新标记嵌入
                # print(new_embeds)
            elif self.prompt_encoder_type == "transformer":
                new_embeds = self.transformer_encoder(new_embeds)  # 使用 Transformer Encoder 头部处理新标记嵌入
                # print(new_embeds)
            new_embeds = self.new_mlp_head(new_embeds)

            replace_idxs = torch.nonzero(batch['soft_token_ids']>0).view(-1, self.num_soft_token, 2)
            for b in range(replace_idxs.shape[0]):
                for i in range(self.num_soft_token):
                    inputs_embeds[b][replace_idxs[b][i][1]] = new_embeds[0][i]

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch
