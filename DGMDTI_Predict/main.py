from model import DGMDTI
from time import time
from utils import graph_collate_func
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import pandas as pd

def main(model_hyperparam_dict, device, args):
    # 1.定义数据的地址
    train_path = '../datasets/biosnap/random/train.csv'
    val_path = '../datasets/biosnap/random/val.csv'
    test_path = '../datasets/biosnap/random/test.csv'

    # 2.加载数据
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)

    # 3.创建数据加载器
    params = {'batch_size': 64, 'shuffle': False, 'num_workers': 0,
              'drop_last': False, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    # 4.定义模型
    model = DGMDTI(model_hyperparam_dict, args).to(device)

    # 5.定义优化器
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    torch.backends.cudnn.benchmark = True  # 用来优化运行性能

    # 6.运行模型
    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator,
                      model_hyperparam_dict)
    trainer.train()




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_hyperparam_dict = {

        'DRUG_MAX_NODES': 290,    # 药物的长度为290
        'PROTEIN_MAX_NODES': 600, # 蛋白质的序列长度为600

        # 解码器的参数
        'DECODER_IN_DIM': 512,
        'DECODER_HIDDEN_DIM': 512,
        'DECODER_OUT_DIM': 128,
        'DECODER_BINARY': 1,

        'MAX_EPOCH' : 100
    }

    # 下面是MCAN模块的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of hidden')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # Node: head_dim × num_heads == hidden_size
    parser.add_argument('--num_heads', type=int, default=4, help='the number of heads')
    parser.add_argument('--head_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2, help='层数')

    # AttFlat参数
    parser.add_argument('--FLAT_MLP_SIZE', type=int, default=256)
    parser.add_argument('--FLAT_GLIMPSES', type=int, default=1)
    parser.add_argument('--FLAT_OUT_SIZE', type=int, default=256)
    args = parser.parse_args()

    s = time()
    main(model_hyperparam_dict, device, args)
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
