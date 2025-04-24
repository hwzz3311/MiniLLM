import json
import os
import psutil
import gc

from tokenizers import models, Tokenizer, decoders, trainers, processors, pre_tokenizers
from transformers import PretrainedConfig, AutoTokenizer
from src.data.data_processing import load_data
from src.data.utils import compute_tokenizer_metrics

current_dir = os.path.dirname(os.path.abspath(__file__))


def train_tokenizer(pre_train_data_path, output_dir):
    def iterative_load_data(data_path, batch_size=100 * 100 * 50): # 10w条数据
        """使用生成器逐批加载数据"""
        training_corpus_list = load_data(data_path, iterative=True)
        batch = []
        for item in training_corpus_list:
            batch.append(item["text"])
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # 处理剩余的样本
            yield batch

    # 添加内存监控
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    # 创建tokenizer
    tokenizer = Tokenizer(models.BPE())
    # 设置预分词器
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 先统计词频
    word_freq = {}
    for batch in iterative_load_data(pre_train_data_path):
        for text in batch:
            # 统计词频
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
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
    # 配置BPE训练器
    trainer = trainers.BpeTrainer(
        vocab_size=32000, # 32000个token
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        word_frequency=word_freq
    )

    # 分批训练tokenizer
    for batch_idx, batch in enumerate(iterative_load_data(pre_train_data_path)):
        print(f"处理第 {batch_idx} 批数据，当前内存使用: {get_memory_usage():.2f} MB")
        tokenizer.train_from_iterator(batch, trainer=trainer)
        # 每处理完一批数据后清理内存
        gc.collect()

    # 设置后处理器
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 保存tokenizer和配置
    os.makedirs(output_dir, exist_ok=True)
    # 创建并配置tokenizer的基础配置
    tokenizer.save(f"{output_dir}/tokenizer.json")

    # 创建基础配置
    config = PretrainedConfig(
        tokenizer_class="PreTrainedTokenizerFast",
        model_max_length=131072, # 参考qwen2
        pad_token="<|endoftext|>",  # 用于将序列填充到固定长度的标记
        eos_token="<|endoftext|>", # 用于表示序列结束的标记  在很多场景下，文本的结束和填充在语义上是等价的
        add_bos_token=False,  #,  # 是否自动添加开始标记，不使用自动添加，方便控制
        add_eos_token=False,
        add_prefix_space=False,  # 是否在单词前添加空格
        # bos_token=None,  # 开始标记，预训练的开始标记，由于是gpt-style风格的处理方式，因此只需要知道结束标记即可，无需开始标记。
        clean_up_tokenization_spaces=False,  # 是否清理分词结果中的多余空格
        split_special_tokens=False,  # 是否分割特殊标记
        unk_token="<|unk|>",  #
        chat_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
    )
    """
    提问：chat_template为什么要用<|im_end|>而不用<|endoftext|>进行区分？
    	•	<|endoftext|>：是通用的文本终止符（End Of Sequence, EOS），在模型预训练阶段就存在，用于标识一段文本的结束。
	    •	<|im_end|>：是对话模板专用的终止标识符，专门用于标识一条对话消息的结束。它只在对话格式化时用，比如 system/user/assistant 的消息转换时。
	    1. 作用域不同：
            •	<|endoftext|> 是“一整段对话的终止”；
            •	<|im_end|> 是“一条消息的结束”（例如一条 user 消息、一条 assistant 回复）。
        在 chat_template 中，我们需要结构化地构建多轮对话，不能在每一条消息后就用 <|endoftext|>，否则模型会误认为对话已经结束，不会继续生成。
        2. 防止生成提前终止：
        模型如果看到 <|endoftext|>，有极高概率会停止生成（EOS Token）。但 <|im_end|> 不具有这种终止“硬中断”特性，而是作为软性的边界符使用。
    """
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
    # pre_train_data_path = "/mnt/d/pretrain/merge_data/baidubaike_wikipedia_sample_data.parquet"
    pre_train_data_path = "/mnt/d/pretrain/minimind/pretrain_hq.parquet"
    output_dir = os.path.join(current_dir, "./assets/minillm_tokenizer")

    # train_tokenizer(pre_train_data_path, output_dir)
    eval_tokenizer(output_dir)
    # 计算tokenizer的压缩率等指标
    texts = ["我爱自然语言处理。", "ChatGPT 是一个大型语言模型。"]
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    print(len(tokenizer))
    print(f"{len(tokenizer.get_vocab())=}")
    print(f"vocab size : {tokenizer.vocab_size}")
    # res = compute_tokenizer_metrics(texts, tokenizer)
    # print(res)
    # 将Qwen2Tokenizer 下载到本地
    tokenizer_path = os.path.join(current_dir, "./assets/qwen_tokenizer/")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("qwen2:\n")
    print(len(tokenizer))
    print(f"{len(tokenizer.get_vocab())=}")
    print(f"vocab size : {tokenizer.vocab_size}")
    minimind_tokenizer_path = os.path.join(current_dir, "./assets/minimind_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(minimind_tokenizer_path, trust_remote_code=True)
    print("minimind:\n")
    print(len(tokenizer))
    print(f"{len(tokenizer.get_vocab())=}")
    print(f"vocab size : {tokenizer.vocab_size}")

    # s = {'text': '<s>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。</s> <s>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？</s> <s>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。</s> <s>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。</s> <s>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。</s> <s>帮我想一个有趣的标题。这个挺有趣的："如何成为一名成功的魔术师" 调皮的标题往往会吸引读者的注意力。</s> <s>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。</s> <s>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本："今天是我的生日！"这个文本的语气是喜悦。</s>'}
    # print(s["text"])
    # text_list = s["text"].replace("</s>", "").split("<s>")
    # for text in text_list:
    #     print(text)
    #     print(tokenizer.encode(text))
    #     print(tokenizer.decode(tokenizer.encode(text)))
    #     print("-"*100)
    # # 获取tokenizer中的 bos_token 和 eos_token，pad_token，unk_token
    # print("bos_token:", tokenizer.bos_token)
    # print("eos_token:", tokenizer.eos_token)
    # print("pad_token:", tokenizer.pad_token)
    # print("unk_token:", tokenizer.unk_token)

    # # 判断下<s>和</s>是否为tokenizer的特殊token
    # print(tokenizer.special_tokens_map)
    # print("tokenizer size:", tokenizer.vocab_size)
    
    # tokenizer.save_pretrained(tokenizer_path)

