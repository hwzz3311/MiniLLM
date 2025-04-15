import json
import os

from tokenizers import models, Tokenizer, decoders, trainers, processors, pre_tokenizers
from transformers import PretrainedConfig
from src.data.data_processing import load_data

current_dir = os.path.dirname(os.path.abspath(__file__))


def train_tokenizer(pre_train_data_path, output_dir):
    def iterative_load_data(data_path):
        training_corpus_list = load_data(data_path, iterative=True)
        for item in training_corpus_list:
            yield item["text"]

    training_corpus = iterative_load_data(pre_train_data_path)
    # 添加调试信息
    sample_count = 0
    char_count = 0
    for text in training_corpus:
        sample_count += 1
        char_count += len(text)
        if sample_count % 10000 == 0:
            print(f"已处理 {sample_count} 个样本，总字符数: {char_count}")

    training_corpus = iterative_load_data(pre_train_data_path)

    # 定义特殊token列表
    special_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<tool_call>",
        "</tool_call>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>"
    ]
    # 创建tokenizer
    tokenizer = Tokenizer(models.BPE())
    # 设置预分词器
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()
    # 从数据文件加载训练数据
    # 配置BPE训练器
    trainer = trainers.BpeTrainer(
        vocab_size=6400,  # 词汇表大小
        min_frequency=2,  # 最小词频
        special_tokens=special_tokens,  # 特殊标记列表
        show_progress=True,  # 显示进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 初始字母表
    )
    # 训练tokenizer
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    # 设置后处理器
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 保存tokenizer和配置
    os.makedirs(output_dir, exist_ok=True)
    # 创建并配置tokenizer的基础配置
    tokenizer.save(f"{output_dir}/tokenizer.json")

    # 创建基础配置
    config = PretrainedConfig(
        tokenizer_class="Qwen2Tokenizer",
        model_max_length=131072,
        pad_token="<|endoftext|>",  # 用于将序列填充到固定长度的标记
        eos_token="<|endoftext|>",  # 用于表示序列结束的标记  在很多场景下，文本的结束和填充在语义上是等价的
        add_bos_token=False,  # 是否添加开始标记
        add_prefix_space=False,  # 是否在单词前添加空格
        bos_token=None,  # 开始标记
        clean_up_tokenization_spaces=False,  # 是否清理分词结果中的多余空格
        split_special_tokens=False,  # 是否分割特殊标记
        unk_token=None,  #
        chat_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
    )

    # 添加additional_special_tokens
    additional_special_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>"
    ]

    # 将配置转换为字典并添加额外的special tokens
    config_dict = config.to_dict()
    config_dict["additional_special_tokens"] = additional_special_tokens

    # 保存配置文件
    with open(f"{output_dir}/tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    save_tokenizer_files(tokenizer, output_dir)


def eval_tokenizer(output_dir):
    from transformers import AutoTokenizer  # 导入AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_dir)  # 从指定路径加载tokenizer

    messages = [  # 定义消息列表
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(  # 应用聊天模板
        messages,
        tokenize=False  # 不进行tokenize
    )
    print('new_prompt:', new_prompt)  # 打印新的提示

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)  # 获取tokenizer的词汇表长度
    print('tokenizer实际词表长度：', actual_vocab_size)  # 打印词汇表长度

    model_inputs = tokenizer(new_prompt)  # 对新提示进行tokenize
    print('encoder长度：', len(model_inputs['input_ids']))  # 打印编码器长度

    input_ids = model_inputs['input_ids']  # 获取输入ID
    response = tokenizer.decode(input_ids, skip_special_tokens=False)  # 解码输入ID
    print('decoder和原始文本是否一致：', response == new_prompt)  # 检查解码结果是否与原始文本一致
    print('response:', response)


def save_tokenizer_files(tokenizer, output_dir):
    # 保存tokenizer.json
    tokenizer.save(f"{output_dir}/tokenizer.json")

    # 获取词表并保存为vocab.json
    vocab = tokenizer.get_vocab()
    with open(f"{output_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 保存成txt格式（每行一个token）
    with open(f"{output_dir}/vocab.txt", "w", encoding="utf-8") as f:
        for token, _ in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")

    # 保存merges文件（BPE合并规则）
    try:
        # 尝试获取merges
        if hasattr(tokenizer.model, "merge_map"):
            # 从merge_map构建merges列表
            merges = list(tokenizer.model.merge_map.keys())
            with open(f"{output_dir}/merges.txt", "w", encoding="utf-8") as f:
                for merge in merges:
                    # merge可能是元组形式，需要转换成适当的格式
                    if isinstance(merge, tuple):
                        f.write(f"{merge[0]} {merge[1]}\n")
                    else:
                        f.write(f"{merge}\n")
    except Exception as e:
        print(f"警告：无法保存merges文件: {str(e)}")
        # 如果无法获取merges，我们可以跳过这一步，因为tokenizer.json已经包含了所有必要的信息


if __name__ == '__main__':
    # pre_train_data_path = os.path.join(current_dir, "./data_sample/baidubaike_sample_data.json")
    pre_train_data_path = "/mnt/d/pretrain/merge_data/baidubaike_wikipedia_sample_data.parquet"
    output_dir = os.path.join(current_dir, "./assets/tokenizer_output")

    train_tokenizer(pre_train_data_path, output_dir)
    eval_tokenizer(output_dir)
