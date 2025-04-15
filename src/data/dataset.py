import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
from .data_processing import load_data
class PretrainDataset(Dataset):

    def __init__(self,
                 data_path:str,
                 tokenizer:AutoTokenizer,
                 max_len:int=2048,
                 ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = load_data(data_path)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        sample = self.samples[idx]
        text = f"{self.tokenizer.bos_token}{sample['text']}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ) # 对文本进行编码
        input_ids = encoding.input_ids.squeeze() # 获取input_ids
        loss_mask = (input_ids != self.tokenizer.pad_token_id) # 获取loss_mask
        
        # X = torch.tensor(input_ids[:-1],dtype=torch.long)
        X = input_ids[:-1].clone().detach().to(torch.long)
        # Y = torch.tensor(input_ids[1:],dtype=torch.long)
        Y = input_ids[1:].clone().detach().to(torch.long)
        # loss_mask = torch.tensor(loss_mask[1:],dtype=torch.float)
        loss_mask = loss_mask[1:].clone().detach().to(torch.float)
        return X,Y,loss_mask


if __name__ == "__main__":
    tokenizer_path = "/Users/leon_zheng/PycharmProjects/LLM/ReQwen/tokenizer_output"
    data_path = "/Users/leon_zheng/PycharmProjects/LLM/ReQwen/data_sample/wikipedia_zh_sample_data.json"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = PretrainDataset(data_path=data_path,tokenizer=tokenizer)
    print(len(dataset))
    print(dataset[0])


        
