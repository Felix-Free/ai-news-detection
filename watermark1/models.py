import math
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import torch


class BitStringMapper(nn.Module):
    def __init__(self):
        super(BitStringMapper, self).__init__()

        #self.device = "cuda:0" if torch.cuda.is_available() else torch.device("cpu")

        # 创建参数并注册
        self.fc1_weight = nn.Parameter(torch.randn(512, 768))
        self.fc1_bias = nn.Parameter(torch.randn(512))

        # 注意力机制的参数
        self.attention_q = nn.Parameter(torch.randn(512, 512))
        self.attention_k = nn.Parameter(torch.randn(512, 512))
        self.attention_v = nn.Parameter(torch.randn(512, 512))

        self.fc2_weight = nn.Parameter(torch.randn(256, 512))
        self.fc2_bias = nn.Parameter(torch.randn(256))

        self.fc3_weight = nn.Parameter(torch.randn(1024, 256))
        self.fc3_bias = nn.Parameter(torch.randn(1024))

        # Layer Norm 参数
        self.norm_weight = nn.Parameter(torch.ones(512))
        self.norm_bias = nn.Parameter(torch.zeros(512))

        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x

        # FC1
        x = F.linear(x, self.fc1_weight, self.fc1_bias)

        # Layer Norm
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        x = x * self.norm_weight + self.norm_bias

        # 多头注意力
        batch_size = x.size(0)
        # 计算Q, K, V
        q = torch.matmul(x, self.attention_q)
        k = torch.matmul(x, self.attention_k)
        v = torch.matmul(x, self.attention_v)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(512)
        attention_weights = F.softmax(scores, dim=-1)
        x = torch.matmul(attention_weights, v)

        x = self.dropout(x)

        # FC2
        x = F.linear(x, self.fc2_weight, self.fc2_bias)
        x = self.dropout(x)

        # FC3
        x = F.linear(x, self.fc3_weight, self.fc3_bias)
        x = self.sigmoid(x)

        return x

    def get_parameter_count(self):
        """打印参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")
        for name, param in self.named_parameters():
            print(f"{name}: {param.numel()} parameters")


class ContrastiveTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        # args = parse_args()
        #self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "princeton-nlp/sup-simcse-bert-base-uncased")  # BertTokenizer.from_pretrained(model_name)#
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.batch_size = 2

        # 添加一个线性层来改变输出维度
        self.output_projection = nn.Linear(768, 768)# 768 是BERT的默认输出维度

        # self.watermark_encoder = WatermarkEncoder(args.feature_dim, args.wm_dim)
        self.wmMatrix = BitStringMapper()

        self.models = nn.ModuleDict({
            'encoder': self.model,
            'output_projection': self.output_projection,  # 添加到ModuleDict中
            'watermark_encoder': self.wmMatrix
        })

    def forward(self, texts,watermark=None):
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encodings = {key: val for key, val in encodings.items()}

        # 获取BERT的输出
        bert_output = self.model(**encodings, output_hidden_states=True, return_dict=True).pooler_output

        # 通过投影层改变维度
        outputs = self.output_projection(bert_output)

        if watermark is not None:
            watermark_output = self.wmMatrix(watermark)
            return outputs, watermark, watermark_output
        else:
            return outputs

    def get_optimizer(self, now_epoch):

        if now_epoch < 4:

            encoder_lr = 5e-5
            watermark_encoder_lr = 1e-5
        else:

            encoder_lr = 1e-7
            watermark_encoder_lr = 1e-5

        optimizer_grouped_parameters = [
            {
                'params': self.models['encoder'].parameters(),
                'lr': encoder_lr
            },
            {
                'params': self.models['watermark_encoder'].parameters(),
                'lr': watermark_encoder_lr
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        return optimizer


def create_optimizer(model):
    optimizer_params = [
        {'params': model.model.parameters(), 'lr': 2e-5},
        {'params': model.wmMatrix.parameters(), 'lr': 5e-4}
    ]
    optimizer = AdamW(optimizer_params, weight_decay=1e-5)

    return optimizer  # , scheduler


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
