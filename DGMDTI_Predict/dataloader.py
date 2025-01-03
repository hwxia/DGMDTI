import torch.utils.data as data

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']

        # 下面是处理蛋白质和标签部分
        v_p = self.df.iloc[index]['Protein']

        # 下面处理标签部分
        y = self.df.iloc[index]["Y"]
        return v_d, v_p, y
