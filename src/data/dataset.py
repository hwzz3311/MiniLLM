import numpy
import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
import queue
from queue import Queue



from src.data.data_processing import load_data


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
            self.samples = [] # 清空样本
        
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
        """分块对数据进行token化并缓存，引入队列进行线程控制，可以有效解决内存暴涨的问题"""
        writer = None
        
        # 准备数据块
        chunks = [self.samples[i:i + self.chunk_size] 
                 for i in range(0, len(self.samples), self.chunk_size)]
        
        # 使用线程池处理数据块
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 使用队列来管理任务
            result_queue = Queue()
            
            def process_chunk(chunk, chunk_idx):
                try:
                    chunk_input_ids = self._tokenize_chunk(chunk)
                    # 立即将结果放入队列
                    result_queue.put((chunk_idx, chunk_input_ids))
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                finally:
                    # 确保清理线程局部数据
                    del chunk_input_ids
                    gc.collect()
            
            # 提交所有任务
            for i, chunk in enumerate(chunks):
                executor.submit(process_chunk, chunk, i)
                del chunk  # 立即清理原始数据
            
            # 处理完成的任务
            processed_chunks = 0
            while processed_chunks < len(chunks):
                try:
                    chunk_idx, chunk_input_ids = result_queue.get(timeout=30)  # 设置超时
                    # 使用 ListArray 直接存储嵌套列表
                    chunk_input_ids = pa.array(chunk_input_ids, type=pa.list_(pa.int64()))
                    # 创建表
                    table = pa.Table.from_pydict({'input_ids': chunk_input_ids})
                    # 写入parquet文件
                    if writer is None:
                        writer = pq.ParquetWriter(
                            self.cache_path,
                            table.schema,
                            compression='snappy'
                        )
                    writer.write_table(table)
                    
                    # 立即清理数据
                    del chunk_input_ids, table
                    processed_chunks += 1
                    
                    # 定期进行垃圾回收
                    if processed_chunks % 5 == 0:
                        gc.collect()
                except queue.Empty:
                    print("等待结果超时")
                    continue
                except Exception as e:
                    print(f"Error writing chunk {chunk_idx}: {e}")
                    continue
        
        if writer:
            writer.close()
        
        # 清理不再需要的数据
        del self.samples
        gc.collect()

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

class SFT_Dataset(Dataset):
    def __init__(self,
                 data_path:str,
                 tokenizer:AutoTokenizer,
                 max_len:int=1024,
                 chunk_size:int=10000,  # 每次处理的样本数
                 num_workers:int=min(os.cpu_count(), 16),     # token化时的线程数
                 ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.num_workers = num_workers

        self.sft_bos_token = "<|im_start|>assistant\n" # 系统提示词 结束提示词应该根据 tokenizer 的 chat_template 来确定
        self.sft_eos_token = "<|im_end|>\n" # 结束提示词
        # 生成 sft阶段的 bos 和 eos token id
        self.sft_bos_token_id:list = tokenizer(self.sft_bos_token,add_special_tokens=False).input_ids
        self.sft_eos_token_id:list = tokenizer(self.sft_eos_token,add_special_tokens=False).input_ids

        # 生成缓存文件路径
        cache_dir = os.path.dirname(data_path)
        file_name = os.path.basename(data_path)
        cache_filename = f"tokenized_sft_{file_name}.parquet"
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
        print(f"total_samples: {len(self.all_tokenized_samples)}")
    def _create_chat_prompt(self,conversations):
        """创建chat prompt"""
        messages = []
        for i,conversation in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role":role,"content":conversation["content"]})
        # 返回 tokenizer 的 chat_template 的格式
        prompt = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=False)
        return prompt
    def _tokenize_chunk(self, chunk):
        """处理单个数据块的token化"""
        chunk_input_ids = []
        for sample in tqdm(chunk, desc="Tokenizing SFT data chunk"):
            assert "conversations" in sample, "conversations 不存在"
            conversations = sample['conversations']
            prompt = self._create_chat_prompt(conversations)
            encoding = self.tokenizer(prompt)
            chunk_input_ids.append(encoding.input_ids)
            # 及时清理不再需要的数据
            del sample, conversations, prompt, encoding
        return chunk_input_ids

    def _tokenize_and_cache(self):
        """分块对数据进行token化并缓存"""
        writer = None
        
        # 准备数据块
        chunks = [self.samples[i:i + self.chunk_size] 
                 for i in range(0, len(self.samples), self.chunk_size)]
        # chunks = chunks[:1] # 临时只处理一个，看看效果
        
        # 使用线程池处理数据块
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 使用队列来管理任务
            result_queue = Queue()
            
            def process_chunk(chunk, chunk_idx):
                try:
                    chunk_input_ids = self._tokenize_chunk(chunk)
                    # 立即将结果放入队列
                    result_queue.put((chunk_idx, chunk_input_ids))
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                finally:
                    # 确保清理线程局部数据
                    del chunk_input_ids
                    gc.collect()
            
            # 提交所有任务
            for i, chunk in enumerate(chunks):
                executor.submit(process_chunk, chunk, i)
                del chunk  # 立即清理原始数据
            
            # 处理完成的任务
            processed_chunks = 0
            while processed_chunks < len(chunks):
                try:
                    chunk_idx, chunk_input_ids = result_queue.get(timeout=30)  # 设置超时
                    
                    # 使用 ListArray 直接存储嵌套列表
                    chunk_input_ids = pa.array(chunk_input_ids, type=pa.list_(pa.int64()))
                    # 创建表
                    table = pa.Table.from_pydict({'input_ids': chunk_input_ids})
                    
                    # 写入parquet文件
                    if writer is None:
                        writer = pq.ParquetWriter(
                            self.cache_path,
                            table.schema,
                            compression='snappy'
                        )
                    writer.write_table(table)
                    
                    # 立即清理数据
                    del chunk_input_ids, table
                    processed_chunks += 1
                    
                    # 定期进行垃圾回收
                    if processed_chunks % 5 == 0:
                        gc.collect()
                        
                except queue.Empty:
                    print("等待结果超时...")
                    continue
                except Exception as e:
                    print(f"Error writing chunk {chunk_idx}: {e}")
                    continue
        
        if writer:
            writer.close()
        
        # 清理不再需要的数据
        del self.samples
        gc.collect()

    def _init_from_cache(self):
        """从缓存文件初始化"""  
        self.parquet_file = pq.ParquetFile(self.cache_path)
        self.num_rows = self.parquet_file.num_row_groups
        print(f"Number of row groups: {self.num_rows}")
        # 使用更高效的方式读取数据
        df = pd.read_parquet(self.cache_path)
        self.all_tokenized_samples = df['input_ids'].tolist()
        print(f"Total tokens loaded: {len(self.all_tokenized_samples)}")
    def __len__(self):
        return len(self.all_tokenized_samples)

    def _generate_loss_mask(self, input_ids):
        """
        只对 模型生成的 内容 进行损失掩码，其他部分为输入，不需要参与loss计算。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.sft_bos_token_id)] == self.sft_bos_token_id:
                start = i + len(self.sft_bos_token_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.sft_eos_token_id)] == self.sft_eos_token_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.sft_eos_token_id) + 1, len(input_ids))):
                    loss_mask[j] = 1
                i = end + len(self.sft_eos_token_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    def __getitem__(self, idx):
        input_ids:numpy.ndarray = self.all_tokenized_samples[idx]
        input_ids:list = input_ids.tolist()
        # 生成损失掩码
        loss_mask = self._generate_loss_mask(input_ids)
        
        # 确保长度正确
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            loss_mask = loss_mask[:self.max_len]
        elif len(input_ids) < self.max_len:
            # 填充到max_len
            pad_length = self.max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            loss_mask = loss_mask + [0] * pad_length
        
        # 准备输入和标签
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.float)
        
        return X, Y, loss_mask

class PretrainVLDataset(PretrainDataset):
    def __init__(self,
                 data_path:str,
                 tokenizer:AutoTokenizer,
                 max_len:int=2048,
                 chunk_size:int=10000,  # 每次处理的样本数
                 num_workers:int=min(os.cpu_count(), 16),     # token化时的线程数
                ):
        super().__init__(data_path,tokenizer,max_len,chunk_size,num_workers)
        pass

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    minillm_tokenizer_path = os.path.join(current_dir,"../../assets/minillm_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(minillm_tokenizer_path)
    data_path = os.path.join(current_dir,"../../data_sample/baidubaike_wikipedia_sample_data.parquet")
    data_path = "/mnt/d/pretrain/minimind/sft_mini_512.parquet"

    sft_dataset = SFT_Dataset(data_path=data_path,
                              tokenizer=tokenizer,
                              max_len=1024,
                              num_workers=8)
    print(len(sft_dataset))
    for i in range(len(sft_dataset)):
        X,Y,loss_mask = sft_dataset[i]
        print(X.shape,Y.shape,loss_mask.shape)
        print(X)
        print(Y)
        print(loss_mask)
        break
