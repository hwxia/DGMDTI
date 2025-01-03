import numpy as np
import torch

def graph_collate_func(x):
    d, p, y = zip(*x)
    return d, p, torch.tensor(y)


def loadESM2andMol(train_num_batches, val_num_batches, test_num_batches):
    esm2_train_list = []
    esm2_val_list = []
    esm2_test_list = []

    mol_train_list = []
    mol_val_list = []
    mol_test_list = []
    for index in range(train_num_batches):
        tmp1 = np.load(f'./ESM2_embedding/bioSNAP/train/train_pro_emd_{index + 1}.npy')
        tmp2 = np.load(f'./MolFormer_embedding/bioSNAP/train/train_drug_emd_{index + 1}.npy')
        esm2_train_list.append(tmp1)
        mol_train_list.append(tmp2)

    for index in range(val_num_batches):
        tmp1 = np.load(f'./ESM2_embedding/bioSNAP/val/val_pro_emd_{index + 1}.npy')
        tmp2 = np.load(f'./MolFormer_embedding/bioSNAP/val/val_drug_emd_{index + 1}.npy')
        esm2_val_list.append(tmp1)
        mol_val_list.append(tmp2)

    for index in range(test_num_batches):
        tmp1 = np.load(f'./ESM2_embedding/bioSNAP/test/test_pro_emd_{index + 1}.npy')
        tmp2 = np.load(f'./MolFormer_embedding/bioSNAP/test/test_drug_emd_{index + 1}.npy')
        esm2_test_list.append(tmp1)
        mol_test_list.append(tmp2)

    return esm2_train_list, esm2_val_list, esm2_test_list, mol_train_list, mol_val_list, mol_test_list



