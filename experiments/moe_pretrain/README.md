## MOE 预训练

### MOE学习细节

[2025-05-06_10-17-moe模型学习记录-cursor.md](../../.specstory/history/2025-05-06_10-17-moe%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95-cursor.md)   

### 数据准备

数据部分使用和MiniLM一致的语料，使用`src/data/pretrain_dataset.py`中的`PretrainDataset`类进行处理。


### 训练记录

1. 训练时将参数num_experts_per_tok（每个token的专家数）设置为2，n_routed_experts（路由专家数）设置为4，n_shared_experts（共享专家数）设置为1，aux_loss_alpha（辅助损失系数）设置为0.01，seq_aux（序列辅助损失）设置为True，norm_topk_prob（归一化topk概率）设置为True。最终 模型参数量增大至108.16M，增加约4倍。

2. 整体loss会比未使用moe的模型要高很多，未使用moe的模型loss初始在7-8左右，训练后可以下降至2-3左右。而使用moe的模型loss初始在26左右，训练时下降速度缓慢，经过约3个epoch后loss下降至23左右，但验证可以正常进行续写。需要增大学习率继续训练，看下loss是否可以继续下降。

3. 将学习率从5e-4增大到9e-4，loss下降依然很缓慢，且验证时模型续写效果变差。切换预训练数据集试试。

4. 更换 baike+wiki的数据集训练，loss依然很大。分析发现这和moe中的专家选择的loss有很大关系，由于专家选择的分散导致辅助损失变大。这就导致最终的loss必然很大。这也证明moe模型更加难以训练。