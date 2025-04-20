from typing import List, Dict
from collections import Counter

def compute_tokenizer_metrics(texts: List[str], tokenizer) -> Dict[str, float]:
    """
    计算 tokenizer 的常见评估指标，包括压缩比、平均分词长度、UNK率等。

    Args:
        texts (List[str]): 原始文本列表
        tokenizer: HuggingFace 或兼容 tokenizer，需支持 encode 和 decode

    Returns:
        Dict[str, float]: 指标结果字典
    """
    total_chars = 0
    total_tokens = 0
    total_unk_tokens = 0
    subword_lengths = []

    for text in texts:
        total_chars += len(text)
        # 使用 tokenizer 分词并记录 token id
        encoding = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoding)
        total_tokens += len(tokens)

        # UNK 检查（依赖 tokenizer 具体 unk_token）
        unk_token = tokenizer.unk_token or "[UNK]"
        total_unk_tokens += sum(1 for token in tokens if token == unk_token)

        # 子词判断：是否是 wordpiece/sentencepiece 的 continuation
        if hasattr(tokenizer, "is_fast") and tokenizer.is_fast:
            encoding_fast = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            word_ids = encoding_fast.word_ids()
            subword_lengths.extend([
                len(tokens[i]) for i in range(len(tokens))
                if word_ids[i] is None or (i > 0 and word_ids[i] == word_ids[i - 1])
            ])
        else:
            # 简化判断：token 中包含 continuation 符号（如 '##'）
            subword_lengths.extend([
                len(token) for token in tokens
                if token.startswith("##") or (token.startswith("▁") == False and token != unk_token)
            ])

    avg_token_len = total_tokens / len(texts) if texts else 0
    compression_ratio = total_chars / total_tokens if total_tokens else 0
    subword_ratio = len(subword_lengths) / total_tokens if total_tokens else 0
    unk_rate = total_unk_tokens / total_tokens if total_tokens else 0

    return {
        "avg_token_per_sentence": avg_token_len,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "subword_ratio": subword_ratio,
        "unk_rate": unk_rate
    }