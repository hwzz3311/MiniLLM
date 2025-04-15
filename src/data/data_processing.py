import json
import pandas as pd
import gzip
from tqdm import tqdm
from pathlib import Path
import os


def load_data(data_path: str, iterative: bool = False):
    file_ext = Path(data_path).suffix.lower()

    if file_ext == '.json':
        with open(data_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            if iterative:
                for line in tqdm(json_data, desc="Loading JSON"):
                    try:
                        yield line
                    except json.JSONDecodeError:
                        continue
            else:
                samples = []
                for line in tqdm(json_data, desc="Loading JSON"):
                    try:
                        samples.append(line)
                    except json.JSONDecodeError:
                        continue
                return samples
    elif file_ext == '.jsonl' or file_ext == '.jsonl.gz':
        with open(data_path, "r", encoding="utf-8") as f:
            if iterative:
                for line in tqdm(f, desc="Loading JSONL"):
                    try:
                        yield json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
            else:
                samples = []
                for line in tqdm(f, desc="Loading JSONL"):
                    try:
                        samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
                return samples
    elif file_ext == '.parquet':
        df = pd.read_parquet(data_path)
        if iterative:
            for _, row in tqdm(df.iterrows(), desc="Loading Parquet", total=len(df)):
                yield row.to_dict()
        else:
            samples = []
            for _, row in tqdm(df.iterrows(), desc="Loading Parquet", total=len(df)):
                samples.append(row.to_dict())
            return samples
    elif file_ext == ".txt":
        with open(data_path, "r", encoding="utf-8") as f:
            if iterative:
                for line in tqdm(f, desc="Loading Text"):
                    yield line.strip()
            else:
                samples = []
                for line in tqdm(f, desc="Loading Text"):
                    samples.append(line.strip())
                return samples
    else:
        raise ValueError(f"不支持的文件格式: {data_path}, 推荐使用 .parquet .json, .jsonl, .txt, 格式")


def save_data(data: list, data_path: str):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if data_path.endswith(".json"):
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    elif data_path.endswith(".jsonl"):
        with open(data_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif data_path.endswith(".parquet"):
        data = pd.DataFrame(data)
        data.to_parquet(data_path, engine="fastparquet")
    else:
        raise ValueError(f"不支持的文件格式: {data_path}, 推荐使用 .parquet .json, .jsonl, 格式")


def merge_data(data_dict: dict, split_key: str = "", min_length: int = 100):
    result_data = []
    for file_path, main_keys in tqdm(data_dict.items(), total=len(data_dict), desc="合并数据"):
        data = load_data(file_path)
        for data_item in data:
            this_text_list = []
            for main_key in main_keys:
                this_text_list.append(data_item[main_key])
            this_text = split_key.join(this_text_list)
            if len(this_text) < min_length:
                continue
            other_info = {k: v for k, v in data_item.items() if k not in main_keys}
            result_data.append({"text": this_text, "other_info": other_info})
    return result_data


if __name__ == "__main__":
    baidu_pretrain_data_path = "/mnt/d/pretrain/clean_step3/baidubaike/"
    wikipedia_pretrain_data_path = "/mnt/d/pretrain/clean_step3/wikipedia_zh.parquet"
    data_dict = {
        wikipedia_pretrain_data_path: ["completion"]
    }
    for file_name in os.listdir(baidu_pretrain_data_path):
        data_dict[os.path.join(baidu_pretrain_data_path, file_name)] = ["text"]
    result_data = merge_data(data_dict, split_key="\n")
    out_path = "/mnt/d/pretrain/merge_data/baidubaike_wikipedia_sample_data.parquet"
    save_data(result_data, out_path)
