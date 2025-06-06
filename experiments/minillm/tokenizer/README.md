## tokenizer 训练

### 训练数据

- 百度百科数据
- 维基百科数据

### 训练命令

```bash
python train_tokenizer.py
```


### 训练结果

- 机器内存：64G
- 词表大小：6400
- 训练数据量：约361W条数据，共计2.1亿字符，约 3.75GB
- 内存不足，训练失败 😭


### 其他方案

- 使用Qwen2的tokenizer，自己训练tokenizer的价值不大


## 失败现象

### 现象表述

- 训练过程中loss一直为明显下降，且多次出现nan的情况
- 分析排查可能和qewn2的tokenizer的词表太大有关，qwen2 词表约15w，而minimind 词表为6400，相差23倍。

### 解决方案

- 自己训练tokenizer，词表大小为6400，但是内存不足
- 使用迭代式训练的方式进行训练，每次迭代50w条数据，最终占用内存为26G


## 实验记录1
- 自己训练tokenizer时设定的vocab为6400，但是加载后vocab就变为了6402，排查分析时由于add_bos_token=True,add_prefix_space=False,导致增加了两个。
- 提问：qwen2的chat_template为什么要用<|im_end|>而不用<|endoftext|>进行区分？
  1. <|endoftext|>：是通用的文本终止符（End Of Sequence, EOS），在模型预训练阶段就存在，用于标识一段文本的结束。
  2. <|im_end|>：是对话模板专用的终止标识符，专门用于标识一条对话消息的结束。它只在对话格式化时用，比如 system/user/assistant 的消息转换时。
  3. 作用域不同：
     1. <|endoftext|> 是“一整段对话的终止”；
     2. <|im_end|> 是“一条消息的结束”（例如一条 user 消息、一条 assistant 回复）。
                在 chat_template 中，我们需要结构化地构建多轮对话，不能在每一条消息后就用 <|endoftext|>，否则模型会误认为对话已经结束，不会继续生成。
  4. 防止生成提前终止： 
     1. 模型如果看到 <|endoftext|>，有极高概率会停止生成（EOS Token）。但 <|im_end|> 不具有这种终止“硬中断”特性，而是作为软性的边界符使用。
