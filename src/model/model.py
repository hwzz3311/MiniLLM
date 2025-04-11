from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn
import torch
from typing import Optional, Tuple,List
import torch.nn.functional as F
import math
from src.model.config import MiniLLMConfig
import numpy as np


class RMSNorm(nn.Module):
    """
    RMSNorm 是一种层归一化方法，是 LayerNorm 的简化版本。主要特点：
    计算更简单：只需计算均方根，不需要计算均值
    训练更稳定：去除了均值计算，减少了方差的影响
    计算效率更高：比传统 LayerNorm 少了约 20% 的计算量
    """

    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps  # 设置eps 防止除零的小常数
        self.dim = dim  # 设置维度
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化权重参数，可学习的缩放参数，初始化为全1向量
    
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # 计算 RMSNorm 的前向传播
        return self.weight * self._norm(x).type_as(x)  # 保持数据类型一致

    def _test_normalization(self):
        import matplotlib.pyplot as plt

        # 设置随机种子以确保结果可复现
        torch.manual_seed(42)

        # 创建测试数据
        batch_size = 2
        seq_len = 4
        dim = 8
        x = torch.randn(batch_size, seq_len, dim)
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-5
        # 初始化归一化层
        layer_norm = nn.LayerNorm(dim, eps=1e-5)

        # 进行归一化
        x_rms = self.forward(x)
        x_layer = layer_norm(x)

        # 打印结果
        print("原始数据形状:", x.shape)
        print("\n原始数据示例:")
        print(x[0, 0])  # 打印第一个样本的第一个位置的数据

        print("\nRMSNorm 归一化后:")
        print(x_rms[0, 0])
        print("\nRMSNorm 统计信息:")
        print(f"均值: {x_rms.mean().item():.6f}")
        print(f"标准差: {x_rms.std().item():.6f}")

        print("\nLayerNorm 归一化后:")
        print(x_layer[0, 0])
        print("\nLayerNorm 统计信息:")
        print(f"均值: {x_layer.mean().item():.6f}")
        print(f"标准差: {x_layer.std().item():.6f}")

        # 可视化对比
        plt.figure(figsize=(12, 4))

        # 绘制原始数据分布
        plt.subplot(131)
        plt.hist(x.detach().numpy().flatten(), bins=50, alpha=0.7, label='原始数据')
        plt.title('原始数据分布')
        plt.legend()

        # 绘制 RMSNorm 结果分布
        plt.subplot(132)
        plt.hist(x_rms.detach().numpy().flatten(), bins=50, alpha=0.7, label='RMSNorm')
        plt.title('RMSNorm 分布')
        plt.legend()

        # 绘制 LayerNorm 结果分布
        plt.subplot(133)
        plt.hist(x_layer.detach().numpy().flatten(), bins=50, alpha=0.7, label='LayerNorm')
        plt.title('LayerNorm 分布')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # 计算两种方法的差异
        diff = torch.abs(x_rms - x_layer)
        print(f"\nRMSNorm 和 LayerNorm 的最大差异: {diff.max().item():.6f}")
        print(f"平均差异: {diff.mean().item():.6f}")


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 计算频率
    """
    •	torch.arange(0, dim, 2)：生成从 0 到 dim-1，步长为 2 的整数序列，形状 (dim//2,)
    •	[: (dim // 2)]：确保取 dim//2 个数，形状仍是 (dim//2,)
    •	.float() / dim：将整数转换为浮点数，并归一化到 [0, 1] 之间
    •	theta ** (...)：按指数缩放
    •	1.0 / (...)：取倒数，得到最终的 freqs
    最终 freqs 形状： (dim//2,)
    例如 dim=8，则 freqs.shape == (4,)
    """
    t = torch.arange(end, device=freqs.device)  # 生成时间序列 torch.arange(end)：生成 0 到 end-1 的整数序列
    freqs = torch.outer(t, freqs).float()  # 计算外积
    """
    torch.outer(t, freqs)：计算外积，每个 t[i] 都会与 freqs 的每个元素相乘 
    t.shape == (32 * 1024,)
    freqs.shape == (dim//2,)
    结果 freqs.shape == (32768, dim//2)
    """
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 生成复数位置编码
    """
    torch.ones_like(freqs)：创建与 freqs 形状相同的全 1 张量（模长）
    torch.polar(模长, 角度)：将模长和角度转换为复数
    freqs.shape == (32768, dim//2)
    pos_cis.shape == (32768, dim//2)
    """
    return pos_cis  # 返回位置编码


def apply_rotary_emb(xq, xk, pos_cis):
    """
    应用旋转嵌入
    """

    def unite_shape(pos_cis: torch.Tensor, x: torch.Tensor):
        """
        计算 pos_cis 需要广播的形状，以适配 x 的维度
        """
        ndim = x.ndim  # 获取x的维度
        assert 1 < ndim  # 断言 x 的维度大于等于 2，确保 x 的维度满足要求
        assert pos_cis.shape == (x.shape[1], x.shape[-1])  # 断言检查，seq_len和dim需要匹配
        shape = []
        for i, d in enumerate(x.shape):
            if i == 1 or i == ndim - 1:  # 如果 i 是 1 或 ndim-1，则保持维度不变
                shape.append(d)
            else:
                shape.append(1)  # 其他维度保持为1
        return pos_cis.view(*shape)  # 返回变形后的 pos_cis
        """
        # 创建一个查询张量 xq
        xq = torch.randn(2, 8, 64)  # (batch_size, seq_len, dim)
        pos_cis = torch.randn(8, 64)  # (seq_len, dim)

        # 计算广播形状
        shape = [1, 8, 1, 64]  # 只在 seq_len 和 dim 上匹配，其他维度为 1
        pos_cis_broadcasted = pos_cis.view(*shape)
        print(pos_cis_broadcasted.shape)  # torch.Size([1, 8, 1, 64])
        """

    # 将 xq 和 xk 形状变换为 (batch_size, seq_len, dim//2, 2) 后，每两个维度当作复数的实部和虚部，转换为复数形式。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # 将xq转换为复数 xq.shape: bsz,seq_len,head_dim
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # 将xk转换为复数
    pos_cis = unite_shape(pos_cis, xq_)  # 调整 pos_cis 的形状，使其与 xq_ 匹配

    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)  # 将复数 xq_ 乘以 pos_cis 后，再展平
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)  # 将复数 xk_ 乘以 pos_cis 后，再展平

    return xq_out.type_as(xq), xk_out.type_as(xk)  # 保持数据类型一致，返回最终的旋转嵌入后的结果

    """
    # 创建查询 (query) 和键 (key) 张量
    xq = torch.randn(2, 8, 64)  # (batch_size, seq_len, dim)
    xk = torch.randn(2, 8, 64)

    # 创建位置编码 pos_cis (旋转嵌入矩阵)
    theta = np.pi / 8  # 每个位置旋转的角度
    pos_cis_real = torch.cos(theta * torch.arange(64))  # 实部
    pos_cis_imag = torch.sin(theta * torch.arange(64))  # 虚部
    pos_cis = torch.stack([pos_cis_real, pos_cis_imag], dim=-1).view(8, 64)  

    # 应用旋转嵌入
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, pos_cis)

    print(xq_rot.shape)  # torch.Size([2, 8, 64])
    print(xk_rot.shape)  # torch.Size([2, 8, 64])
    """


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """

    :param x:输入的键或值张量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)。
    :param n_rep:复制的次数，即 query 头和 key/value 头的比例 (n_heads // n_kv_heads)。
    :return:
    """
    bsz, seq_len, n_kv_heads, head_dim = x.shape  # 获取batch大小、序列长度、kv头数和头维度
    if n_rep == 1:
        return x
    # return x.repeat_interleave(n_rep, dim=2)
    return (
        x[:, :, :, None, :]  # 扩展维度1，增加一个维度，形状为 (bsz, seq_len, 1, head_dim)
        .expand(bsz, seq_len, n_kv_heads, n_rep,
                head_dim)  # 扩展维度2，增加n_rep个维度，形状为 (bsz, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)  # 展平维度2和3，形状为 (bsz, seq_len, n_kv_heads * n_rep, head_dim)
    )
    """
    从attention的机制中分析为什么要进行repeat_kv？
    答：
    1. 在self-attention中，每个query需要与所有的key进行点积运算，以计算注意力权重。
    2. 如果key的数量较少，可能会导致注意力权重分布不均匀，某些重要的信息可能会被忽略。
    3. 通过重复key，可以增加key的数量，从而提高注意力权重的分布均匀性，使得模型能够更好地捕捉到长距离依赖关系。
    4. 重复key还可以减少计算量，提高计算效率。
    重复 KV 的主要原因：
        内存效率：
        KV 缓存（cache）占用的内存减少了 n_rep 倍
        特别是在推理时，这种设计可以显著减少内存使用
        计算效率：
        减少了 K、V 的参数量和计算量
        每个 Q 头仍然可以访问完整的 K、V 信息
        注意力计算的需求：
        注意力计算需要 Q 和 K 的头数相同才能进行矩阵乘法
        通过重复 KV，确保维度匹配：
        实际效果：
        研究表明，共享 KV 头不会显著影响模型性能
        在某些情况下甚至可能起到正则化的作用
    """


class Attention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads  # 设置kv头数
        self.n_local_heads = config.n_heads  # 设置本地头数
        assert self.n_local_heads % self.n_kv_heads == 0  # 确保本地头数是kv头数的倍数
        self.n_local_kv_heads = self.n_kv_heads  # 设置本地kv头数
        self.n_rep = self.n_local_heads // self.n_kv_heads  # 设置重复次数
        self.head_dim = config.hidden_size // config.n_heads  # 设置头维度
        self.wq = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)  # 设置查询权重
        self.wk = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)  # 设置键权重
        self.wv = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)  # 设置值权重
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False)  # 设置输出权重

        self.attn_dropout = nn.Dropout(config.dropout)  # 设置注意力dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # 设置残差dropout

        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn  # 设置flash注意力
        mask = torch.full(size=(1, 1, config.max_seq_len, config.max_seq_len), fill_value=float('-inf'))  # 设置mask
        mask = torch.triu(mask, diagonal=1)  # 设置上三角mask
        self.register_buffer("mask", mask, persistent=False)  # 注册掩码为缓冲区

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, dim = x.shape  # 获取batch大小、序列长度和隐藏维度
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # 计算查询、键和值
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # 重塑查询
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 重塑键
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 重塑值

        # 应用旋转嵌入
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 连接过去的键
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 连接过去的值
        past_kv = (xk, xv) if use_cache else None  # 如果use_cache为True，则保存当前的键和值，更新kv缓存，否则为None

        # 计算注意力权重
        xq = xq.transpose(1, 2)  # 交换维度1和2，将batch和head维度交换 # [B, n_heads, L, D]
        xk = repeat_kv(xk, self.n_rep)  # 重复键 # [B, L, n_heads, D]
        xk = xk.transpose(1, 2)  # 交换维度1和2，将batch和head维度交换 # [B, n_heads, L, D]
        xv = repeat_kv(xv, self.n_rep)  # 重复值 # [B, L, n_heads, D]
        xv = xv.transpose(1, 2)  # 交换维度1和2，将batch和head维度交换 # [B, n_heads, L, D]

        if self.flash and seq_len != 1:
            # 使用flash注意力
            dropout_p = self.dropout if self.training else 0.0  # 设置dropout概率,训练时为dropout，推理时为0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,  #
                dropout_p=dropout_p,
                is_causal=True  # 是否考虑掩码，当 is_causal=True 时，函数会自动生成一个上三角掩码
            )  # shape: [B, n_heads, L, D]
        else:
            socres = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 计算注意力权重 # shape: [B, n_heads, L, L]
            scores += self.mask[:, :, :seq_len, :seq_len]  # 设置掩码 # scores.shape: [1, 1, L, L]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # 计算softmax # scores.shape: [B, n_heads, L, L]
            scores = self.attn_dropout(scores)  # 设置注意力dropout
            output = scores @ xv  # 计算输出 # output.shape: [B, n_heads, L, D]

        output = output.transpose(1, 2)  # 交换维度1和2，将n_heads和seq_len维度交换 # output.shape: [B, L, n_heads, D]
        output = output.reshape(bsz, seq_len, -1)  # 展平维度，恢复到原始形状 # output.shape: [B, L, D]
        output = self.resid_dropout(self.wo(output))  # 设置残差dropout, 并进行线性变换 # output.shape: [B, L, D]
        return output, past_kv  # 返回输出和kv缓存


class FeedForward(nn.Module):

    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        if config.intermediate_size is None:
            hidden_dim = 4 * config.hidden_size  # 计算隐藏维度
            hidden_dim = int(2 * hidden_dim / 3)  # 计算隐藏维度
            config.intermediate_size = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 设置第一个线性层
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 设置第二个线性层
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 设置第三个线性层

        self.dropout = nn.Dropout(config.dropout)  # 设置dropout

    def forward(self, x: torch.Tensor):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MOEFeedForward(nn.Module):
    """
    TODO: 需要实现MOEFeedForward
    """

    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        pass


class MiniLLMBlock(nn.Module):
    def __init__(self, layer_idx: int, config: MiniLLMConfig):
        super().__init__()
        self.n_heads = config.n_heads  # 设置头数
        self.hidden_size = config.hidden_size  # 设置隐藏维度
        self.head_dim = config.hidden_size // config.n_heads  # 设置头维度
        self.attention = Attention(config)  # 初始化注意力层

        self.layer_idx = layer_idx  # 设置层索引
        self.attention_norm = RMSNorm(dim=self.hidden_size, eps=config.rms_norm_eps)  # 设置注意力归一化
        self.ffn_norm = RMSNorm(dim=self.hidden_size, eps=config.rms_norm_eps)  # 设置前馈归一化
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # 创建前馈层

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),  # 归一化输入
            pos_cis,  # 位置编码
            past_key_value=past_key_value,  # 过去键值
            use_cache=use_cache  # 是否使用缓存
        )  # past_kv: 保存当前层的键和值，用于下一个时间步的计算
        h = x + h_attn  # 自注意力层后的残差连接，帮助网络保留原始输入信息，防止信息在注意力处理过程中丢失
        out = h + self.feed_forward(self.ffn_norm(h))  # 残差连接，同样帮助保留信息，防止在前馈网络处理过程中丢失
        return out, past_kv


class MiniLLM(PreTrainedModel):
    config_class = MiniLLMConfig

    def __init__(self, config: MiniLLMConfig = None):
        self.config = config or MiniLLMConfig()
        super().__init__(config)
        self.vocab_size = config.vocab_size  # 设置词汇表大小
        self.n_layers = config.n_layers  # 设置层数
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)  # 创建词嵌入层
        self.dropout = nn.Dropout(config.dropout)  # 设置dropout

        self.layers = nn.ModuleList([MiniLLMBlock(i, config) for i in range(self.n_layers)])  # 创建层列表

        self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)  # 设置归一化
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 设置输出层
        self.tok_embeddings.weight = self.output.weight  # 将词嵌入层的权重与输出层的权重共享

        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=config.hidden_size // config.n_heads, theta=config.rope_theta),
                             persistent=False) # 注册位置编码，persistent=False表示不保存位置编码

        self.OUT = CausalLMOutputWithPast() # 创建输出对象
    
    def forward(self,
            input_ids:Optional[torch.Tensor]=None,
            pask_key_values:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
            use_cache:bool=False,
            **args)->CausalLMOutputWithPast:
        past_key_values = pask_key_values or [None] * self.n_layers # 初始化past_key_values
        start_pos = args.get("start_pos",0) # 获取开始位置
        h = self.dropout(self.tok_embeddings(input_ids)) # 计算词嵌入并应用dropout
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.shape[1]] # 获取位置编码
        past_kvs = [] # 初始化过去的键值对列表
        for l,layer in enumerate(self.layers):
            h,past_kv = layer(
                h, # 输入
                pos_cis, # 位置编码
                past_key_value=past_key_values[l], # 过去键值
                use_cache = use_cache # 是否使用缓存
            )
            past_kvs.append(past_kv) # 保存当前层的键值对
        h = self.norm(h) # 归一化
        logits = self.output(h) # 计算输出
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward)) # 计算辅助损失
        self.OUT.__setitem__("logits",logits) # 设置logits
        self.OUT.__setitem__("aux_loss",aux_loss) # 设置辅助损失
        self.OUT.__setitem__("past_key_values",past_kvs) # 设置past_key_values
        return self.OUT
    
    def generate(self,input_ids:torch.Tensor,eos_token_id:int,
                 max_new_tokens:int=1024,temperature:float=0.75,
                 top_p:float=0.90,stream:bool=False,
                 repetition_penalty:float=1.0,use_cache:bool=True,pad_token_id:int=0,
                 num_return_sequences:int=1,**kwargs)->torch.Tensor:
        """

        Args:
            input_ids: 输入的ids
            eos_token_id: 结束符
            max_new_tokens: 最大新词数
            temperature: 温度
            top_p: 
            stream: 是否流式生成
            repetition_penalty: 重复惩罚
            use_cache: 是否使用缓存
            pad_token_id: 填充符
            num_return_sequences: 返回序列数
            **kwargs:

        Returns:

        """
        if stream:
            pass


    def _stream_generate(self,input_ids:torch.Tensor,eos_token_id:int,
                         max_new_tokens:int=1024,temperature:float=0.75,
                         top_p:float=0.90,
                         repetition_penalty:float=1.0,use_cache:bool=True,**kwargs)->torch.Tensor:
        """

        Args:
            eos_token_id: 结束符
            max_new_tokens: 最大新词数
            temperature: 温度
            top_p: 
            repetition_penalty: 
            use_cache: 是否使用缓存
            **kwargs:

        Returns:

        """
                         
        start,first_seq,past_kvs = input_ids.shape[1],True,None # 初始化开始位置、是否是第一个序列、是否使用缓存
        while input_ids.shape[1] < max_new_tokens -1: # 当序列长度小于最大新词数时
            if first_seq or not use_cache: # 如果是第一个序列或者不使用缓存
                out = self.forward(input_ids, past_key_values=past_kvs,use_cache=use_cache,**kwargs) # 计算输出
                first_seq = False # 设置为False
            else:
                # 如果不是第一个序列，则使用缓存,输入的input_ids是最后一个token，start_pos是最后一个token的位置（因为前面的序列已经计算过了）
                out = self.forward(input_ids[:,-1], pask_key_values=past_kvs,
                                   use_cache=use_cache,start_pos=input_ids.shape[1]- 1,**kwargs) 
                
            logits,past_kvs = out.logits[:,-1,:],out.past_key_values # 获取最后一个token的logits和past_kvs
            # 对当前已生成的 token（即在 input_ids 中出现过的 token）对应的 logits 值进行“惩罚”，除以一个 repetition_penalty 值。
            # 这样做可以降低重复 token 的生成概率，从而减少重复。
            logits[:,list(set(input_ids.tolist()))] /= repetition_penalty 
            # 使用 temperature 参数来控制生成结果的多样性。
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                # 对 logits 进行排序，并计算累积概率
                sorted_logits,sorted_indices = torch.sort(logits,descending=True,dim=-1)
                
                sorted_probs = F.softmax(sorted_logits,dim=-1)
                # torch.cumsum()函数用于对输入张量进行累加和操作，返回一个新的张量，其中每个元素都是原张量中对应位置及之前所有元素的累加和。
                cumulative_probs = torch.cumsum(sorted_probs,dim=-1)
                # 如果累积概率大于 top_p，则将该位置的索引标记为需要移除的索引。
                sorted_indices_to_remove = cumulative_probs > top_p 
                # 将需要移除的索引向右移动一位，确保索引 0 不会被移除。
                sorted_indices_to_remove[:,1:] = sorted_indices_to_remove[:,:-1].clone()
                sorted_indices_to_remove[:,0] = False # 确保索引 0 不会被移除
                # TODO 理解scatter函数
                indices_to_remove = sorted_indices_to_remove.scatter(1,sorted_indices,sorted_indices_to_remove)
                # 将需要移除的索引设置为 -inf，这样在后续的 softmax 操作中，这些位置的概率将变为 0。
                sorted_logits[sorted_indices_to_remove] = -float('inf')
            # 使用 torch.multinomial 函数从概率分布中随机采样一个 token。
            input_ids_next = torch.multinomial(F.softmax(sorted_logits,dim=-1),num_samples=1)
            input_ids = torch.cat([input_ids,input_ids_next],dim=1)
            yield input_ids[:,start:] # 返回当前生成的序列
            if input_ids_next == eos_token_id: # 如果生成的token是结束符，则停止生成
                break
            

if __name__ == "__main__":
    rms_norm = RMSNorm(dim=10, eps=1e-5)
    rms_norm.test_normalization()
