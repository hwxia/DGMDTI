from torch.nn.utils.weight_norm import weight_norm
from MCA_utils import FC, MLP, LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch, math


class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = (self.linear_v(v).
             view(n_batches, -1, self.args.num_heads, self.args.head_dim).transpose(1, 2))

        k = (self.linear_k(k).
             view(n_batches, -1, self.args.num_heads, self.args.head_dim).transpose(1, 2))

        q = (self.linear_q(q).
             view(n_batches, -1, self.args.num_heads, self.args.head_dim).transpose(1, 2))

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.args.hidden_size)

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        # self.mlp = KAN([args.hidden_size, args.hidden_size * 2, args.hidden_size])

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.hidden_size * 2,
            out_size=args.hidden_size,
            dropout_r=args.dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, x, x_mask):
        b, t, d = x.shape
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            # 这里修改了代码!!!!!!
            # self.ffn(x)
            self.ffn(x.reshape(-1, x.shape[-1])).reshape(b, t, d)
        ))

        return x


class SGA(nn.Module):
    def __init__(self, args):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout)
        self.norm2 = LayerNorm(args.hidden_size)

        self.dropout3 = nn.Dropout(args.dropout)
        self.norm3 = LayerNorm(args.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        # x是靶点，y是药物
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        b, t, d = x.shape  # 获取输入张量 x 的批次大小、序列长度和特征维度
        x = self.norm3(x + self.dropout3(
            self.ffn(x.reshape(-1, x.shape[-1])).reshape(b, t, d)
        ))

        return x


class MCA_ED(nn.Module):
    def __init__(self, args):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(args) for _ in range(args.n_layers)])
        self.dec_list = nn.ModuleList([SGA(args) for _ in range(args.n_layers)])

        self.attFlat = AttFlat(args)

    def forward(self, x, y, x_mask, y_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        drug_emd = self.attFlat(x, None)
        pro_emd = self.attFlat(y, None)
        res = torch.cat((drug_emd, pro_emd), dim=1)

        return res


class AttFlat(nn.Module):
    def __init__(self, args):
        super(AttFlat, self).__init__()
        self.args = args

        self.mlp = MLP(
            in_size=args.hidden_size,     # 256
            mid_size=args.FLAT_MLP_SIZE,  # 256
            out_size=args.FLAT_GLIMPSES,  # 1
            dropout_r=args.dropout,     # 0.1
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            args.hidden_size * args.FLAT_GLIMPSES,
            args.FLAT_OUT_SIZE  # 256
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        # att = att.masked_fill(
        #     x_mask.squeeze(1).squeeze(1).unsqueeze(2),
        #     -1e9
        # )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.args.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class FCNet(nn.Module):

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)