from MCA import *
import torch.nn.functional as F


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss



class DGMDTI(nn.Module):
    def __init__(self, hyperparam_dict, args):
        super(DGMDTI, self).__init__()

        self.pro_max_nodes = hyperparam_dict['PROTEIN_MAX_NODES']

        mlp_in_dim = hyperparam_dict['DECODER_IN_DIM']  # 1024
        mlp_hidden_dim = hyperparam_dict['DECODER_HIDDEN_DIM']  # 512
        mlp_out_dim = hyperparam_dict['DECODER_OUT_DIM']  # 128
        out_binary = hyperparam_dict['DECODER_BINARY']  # 1


        self.drug_seq_extractor1 = Dynamic_conv1d(in_planes=768, out_planes=512, kernel_size=3, ratio=0.25, padding=1)
        self.drug_seq_extractor2 = Dynamic_conv1d(in_planes=512, out_planes=512, kernel_size=3, ratio=0.25, padding=1)
        self.gru_drug = nn.GRU(input_size=512, hidden_size=128, batch_first=True, bidirectional=True)

        self.protein_seq_extractor1 = Dynamic_conv1d(in_planes=1280, out_planes=512, kernel_size=5, ratio=0.25, padding=2)
        self.protein_seq_extractor2 = Dynamic_conv1d(in_planes=512, out_planes=512, kernel_size=3, ratio=0.25, padding=1)
        self.gru_protein = nn.GRU(input_size=512, hidden_size=128, batch_first=True, bidirectional=True)

        self.mca = MCA_ED(args)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)


    def forward(self, mol_emd, esm2_emd, mode="train"):
        mol_emd = mol_emd.permute(0, 2, 1)
        v_d_seq = self.drug_seq_extractor1(mol_emd)  # v_d_seq大小为(64,256,290)
        v_d_seq = self.drug_seq_extractor2(v_d_seq)  # (64,256,290)
        v_d_seq = v_d_seq.permute(0, 2, 1)  # 调整维度以适应GRU
        v_d_seq, _ = self.gru_drug(v_d_seq)
        v_d = v_d_seq

        esm2_emd = esm2_emd.permute(0, 2, 1)
        v_p_seq = self.protein_seq_extractor1(esm2_emd)  # v_p_seq大小为(64,256,600)
        v_p_seq = self.protein_seq_extractor2(v_p_seq)
        v_p_seq = v_p_seq.permute(0, 2, 1)  # 调整维度以适应GRU
        v_p_seq, _ = self.gru_protein(v_p_seq)
        v_p = v_p_seq

        f = self.mca(v_d, v_p, None, None)  # f的大小为(64,1024)
        score = self.mlp_classifier(f)  # score大小为(64,1)

        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score



class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class attention1d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).reshape(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height = x.size()
        x = x.reshape(1, -1, height)
        weight = self.weight.reshape(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).reshape(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size,)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).reshape(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.reshape(batch_size, self.out_planes, output.size(-1))
        return output



