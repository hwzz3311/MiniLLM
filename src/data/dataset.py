import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from src.data.data_processing import load_data
from concurrent.futures import ThreadPoolExecutor
import threading
class PretrainDataset(Dataset):

    def __init__(self,
                 data_path:str,
                 tokenizer:AutoTokenizer,
                 max_len:int=2048,
                 chunk_size:int=10000,  # 每次处理的样本数
                 num_workers:int=min(os.cpu_count(), 16),     # token化时的线程数
                 ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        
        # 生成缓存文件路径
        cache_dir = os.path.dirname(data_path)
        cache_filename = f"tokenized_{os.path.basename(data_path)}"
        self.cache_path = os.path.join(cache_dir, cache_filename)
        
        # 检查缓存文件是否存在
        if os.path.exists(self.cache_path):
            print(f"Loading tokenized data from cache: {self.cache_path}")
            self._init_from_cache()
        else:
            print(f"Tokenizing data and creating cache: {self.cache_path}")
            self.samples = load_data(data_path)
            self._tokenize_and_cache()
            self._init_from_cache()
        
        # 计算总长度和样本数
        print("计算总长度和样本数")
        self.total_length = len(self.all_input_ids)
        self.total_samples = (self.total_length // self.max_len) * self.max_len
        print(f"total_length: {self.total_length}")
        print(f"total_samples: {self.total_samples}")
    def _tokenize_chunk(self, chunk):
        """处理单个数据块的token化"""
        chunk_input_ids = []
        for sample in tqdm(chunk, desc="Tokenizing chunk"):
            # 确保添加特殊token
            text = sample['text']
            if self.tokenizer.bos_token and not text.startswith(self.tokenizer.bos_token):
                text = self.tokenizer.bos_token + text
            if self.tokenizer.eos_token and not text.endswith(self.tokenizer.eos_token):
                text = text + self.tokenizer.eos_token
            encoding = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )
            # 获取ids，然后再将ids反推回去，看看是否能还原
            # ids = encoding['input_ids']
            # text_ = self.tokenizer.decode(ids)
            # print(f"text_: {text_}")
            # print(f"text: {text}")
            # print(f"text == text_: {text == text_}")

            # 打印一些样本的token看看
            if len(chunk_input_ids) == 0:  # 只打印第一个样本
                print(f"First sample tokens: {encoding['input_ids']}")
                print(f"Contains bos_token_id: {self.tokenizer.bos_token_id in encoding['input_ids']}")
                print(f"Contains eos_token_id: {self.tokenizer.eos_token_id in encoding['input_ids']}")
                
            chunk_input_ids.extend(encoding['input_ids'])
        return chunk_input_ids

    def _tokenize_and_cache(self):
        """分块对数据进行token化并缓存"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor
        
        # 创建parquet writer
        writer = None
        
        # 准备数据块
        chunks = [self.samples[i:i + self.chunk_size] 
                 for i in range(0, len(self.samples), self.chunk_size)]
        
        # 使用线程池处理数据块
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_chunk = {
                executor.submit(self._tokenize_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # 处理完成的任务
            for future in tqdm(future_to_chunk, desc="Tokenizing data"):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_input_ids = future.result()
                    # 创建当前块的数据
                    table = pa.Table.from_pydict({'input_ids': chunk_input_ids})
                    # 写入parquet文件
                    if writer is None:
                        writer = pq.ParquetWriter(
                            self.cache_path,
                            table.schema,
                            compression='snappy'
                        )
                    writer.write_table(table)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
        
        if writer:
            writer.close()

    def _init_from_cache(self):
        """从缓存文件初始化"""
        import pyarrow.parquet as pq
        
        # 获取parquet文件的元数据
        self.parquet_file = pq.ParquetFile(self.cache_path)
        self.num_rows = self.parquet_file.num_row_groups
        print(f"Number of row groups: {self.num_rows}")
        
        # 使用更高效的方式读取数据
        # 方案1：使用pandas直接读取（更快）
        df = pd.read_parquet(self.cache_path)
        self.all_input_ids = df['input_ids'].tolist()
        
        # 或者方案2：使用pyarrow的批处理读取
        # self.all_input_ids = []
        # for i in tqdm(range(self.num_rows), desc="Loading data from cache"):
        #     row_group = self.parquet_file.read_row_group(i)
        #     self.all_input_ids.extend(row_group['input_ids'].to_pylist())
        
        print(f"Total tokens loaded: {len(self.all_input_ids)}")

    def _get_input_ids(self, start_idx, end_idx):
        """获取指定范围的input_ids"""
        return self.all_input_ids[start_idx:end_idx]

    def __len__(self):
        return self.total_samples // self.max_len
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_len
        end_idx = start_idx + self.max_len
        
        input_ids = self._get_input_ids(start_idx, end_idx)
        # 尝试将input_ids 反推回去，看看是否能还原
        # text_ = self.tokenizer.decode(input_ids)
        # print(f"text: {text_}")
        # 添加长度检查
        if len(input_ids) == 0:
            raise ValueError(f"Empty input_ids at index {idx}")
        
        input_ids = torch.tensor(input_ids)
        
        # 确保长度正确
        if len(input_ids) < self.max_len:
            raise ValueError(f"Input sequence too short at index {idx}")
        
        # 创建loss_mask（所有位置都参与计算）
        loss_mask = torch.ones_like(input_ids, dtype=torch.float)
        
        # 准备输入和标签
        X = input_ids[:-1].clone().detach().to(torch.long)
        Y = input_ids[1:].clone().detach().to(torch.long)
        loss_mask = loss_mask[1:].clone().detach().to(torch.float)
        
        return X, Y, loss_mask


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(current_dir,"../../assets/tokenizer_output")
    # data_path = os.path.join(current_dir,"../../data_sample/baidubaike_wikipedia_sample_data.parquet")
    # data_path = "/mnt/d/pretrain/minimind/pretrain_hq.parquet"
    data_path = "/mnt/d/pretrain/merge_data/baidubaike_wikipedia_sample_data_20min_4096max.parquet"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 检查tokenizer配置
    print(f"bos_token: {tokenizer.bos_token}")
    print(f"eos_token: {tokenizer.eos_token}")
    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"add_special_tokens: {tokenizer.add_special_tokens}")
    dataset = PretrainDataset(data_path=data_path,
                              tokenizer=tokenizer,
                              max_len=512,num_workers=1)
    print("tokenizer.vocab_size: ",tokenizer.vocab_size)
    print(len(dataset))
    for i in range(0,len(dataset),5* 512):
        print(dataset[i][0].shape)
        # 统计 dataset[i][0] 中 eos_token 的数量
        eos_token_count = (dataset[i][0] == tokenizer.eos_token_id).sum().item()
        print(f"eos_token_count: {eos_token_count}")
    


        
