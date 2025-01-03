import pandas as pd
import torch as torch
import numpy as np
import esm
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = (esm_model).to(device)
batch_converter = alphabet.get_batch_converter()


def trans_data_esm(data):
    results = []
    for i in tqdm(range(len(data))):
        protein_item = [data[i]]

        # Process batches
        batch_labels, batch_strs, batch_tokens = batch_converter(protein_item)
        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            res = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = res["representations"][33]
        sequence_representations = []
        for i, (_, seq) in enumerate(protein_item):
            temp_tensor = token_representations[i, 1: len(seq) + 1]
            sequence_representations.append(temp_tensor.detach().cpu().numpy())

        result = np.array(sequence_representations)
        result = np.squeeze(result, axis=0)

        results.append(result)
        torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    # 读取 Excel 文件
    df = pd.read_excel("./bioSNAP/test_output.xlsx")
    # 将 DataFrame 转换为列表
    protein_data = [(row["name"], row["protein_sequence"]) for index, row in df.iterrows()]

    res = trans_data_esm(protein_data)
    res = np.array(res, dtype=np.float32)

    batch_size = 64
    index = 0
    for i in range(0, res.shape[0], batch_size):
        index += 1
        start_index = i
        end_index = min(i + batch_size, res.shape[0])

        # 获取当前批次的数据
        batch_array = res[start_index:end_index]
        # 将分批次的数组保存到文件中
        np.save(f'../DGMDTI_Predict/ESM2_embedding/bioSNAP/test/test_pro_emd_{index}.npy', batch_array)
