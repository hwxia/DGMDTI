from argparse import Namespace
import yaml
from tokenizer.tokenizer import MolTranBertTokenizer
from train_pubchem_light import LightningModule
import torch
from fast_transformers.masking import LengthMask as LM
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm

def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

def pad_or_truncate(tensor, max_length, padding_value=0):
    """Pad or truncate the tensor to the specified length."""
    length = tensor.size(1)
    if length > max_length:
        return tensor[:, :max_length]
    elif length < max_length:
        pad_size = max_length - length
        padding = torch.full((tensor.size(0), pad_size, tensor.size(2)), padding_value)
        return torch.cat([tensor, padding], dim=1)
    return tensor


def embed(model, smiles, tokenizer, batch_size=64):
    model.eval()
    index = 1
    for batch in tqdm(batch_split(smiles, batch_size=batch_size)):
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
        with torch.no_grad():
            token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))

        # Pad or truncate token embeddings to the max_length
        # tensor(64,290,768)
        token_embeddings = pad_or_truncate(token_embeddings, 290)
        arr = token_embeddings.numpy()

        np.save(f'../DGMDTI_Predict/MolFormer_embedding/bioSNAP/train/train_drug_emd_{index}.npy', arr)
        # np.save(f'../DGMDTI_Predict/MolFormer_embedding/bioSNAP/val/val_drug_emd_{index}.npy', arr)
        # np.save(f'../DGMDTI_Predict/MolFormer_embedding/bioSNAP/test/test_drug_emd_{index}.npy', arr)

        index += 1


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    with open('./Pretrained MoLFormer/hparams.yaml', 'r') as f:
        config = Namespace(**yaml.safe_load(f))

    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    tokenizer.to(device)

    ckpt = './Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
    lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)

    def canonicalize(s):
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

    df = pd.read_csv('./bioSNAP/random/train.csv')
    # df = pd.read_csv('./bioSNAP/random/val.csv')
    # df = pd.read_csv('./bioSNAP/random/test.csv')

    smiles = df.SMILES.apply(canonicalize)
    embed(lm, smiles, tokenizer)


