import os.path

from transformers import PretrainedConfig, Qwen2Config
from typing import List


class MiniLLMConfig(PretrainedConfig):
    model_type = "mini_llm"

    def __init__(self, hidden_size: int = 512,
                 n_layers: int = 8,
                 n_heads: int = 8,
                 n_kv_heads: int = 2,
                 vocab_size: int = 6400,
                 intermediate_size: int = None,
                 multiple_of: int = 64,
                 rms_norm_eps: float = 1e-5,
                 max_seq_len: int = 8192,
                 rope_theta: float = 1e6,
                 dropout: float = 0.0,
                 flash_attn: bool = True,
                 ########################################
                 # moe 模式 如果use moe 为False，则以下参数无效
                 ########################################
                 use_moe: bool = False,
                 num_experts_per_tok: int = 2,
                 n_routed_experts: int = 4,
                 n_shared_experts: int = 1,
                 scoring_func: str = "softmax",
                 aux_loss_alpha: float = 0.01,
                 seq_aux: bool = True,
                 norm_topk_prob: bool = True,
                 **kwargs):
        self.hidden_size = hidden_size # 隐层维度
        self.n_layers = n_layers # 层数
        self.n_heads = n_heads # 注意力头数
        self.n_kv_heads = n_kv_heads # key-value 头数
        self.vocab_size = vocab_size # 词表大小
        self.intermediate_size = intermediate_size # 前馈中间层维度
        self.multiple_of = multiple_of # 中间层维度必须是 multiple_of 的倍数
        self.rms_norm_eps = rms_norm_eps # 归一化 eps
        self.max_seq_len = max_seq_len # 最大序列长度
        self.rope_theta = rope_theta # rope 的 theta
        self.dropout = dropout # 丢弃率
        self.flash_attn = flash_attn # 是否使用 flash attention
        self.use_moe = use_moe # 是否使用 MoE
        self.num_experts_per_tok = num_experts_per_tok # 每个 token 的专家数
        self.n_routed_experts = n_routed_experts # 路由专家数
        self.n_shared_experts = n_shared_experts # 共享专家数
        self.scoring_func = scoring_func # 评分函数
        self.aux_loss_alpha = aux_loss_alpha # 辅助损失 alpha
        self.seq_aux = seq_aux # 序列辅助
        self.norm_topk_prob = norm_topk_prob # 归一化 topk 概率 
        super().__init__(**kwargs)



class MiniLLM_VLConfig(MiniLLMConfig):
    model_type = "mini_llm_vl"
    base_path = os.path.abspath(__file__)
    this_clip_model_path = os.path.abspath(os.path.join(base_path,"../assets/clip-vit-base-patch16"))

    def __init__(self,
                 clip_model_path: str=this_clip_model_path,
                 image_special_token: str='<|image_pad|>' * 196,
                 image_token_id: List=[12] * 196,
                 **kwargs
                ):
        self.clip_model_path = clip_model_path
        self.image_special_token = image_special_token
        self.image_token_id = image_token_id
        super().__init__(**kwargs)
        # 判断 clip_model_path 是否和 默认的this_clip_model_path 一致，注意两者的类型可能不一样，需要先统一转换一下
        self.this_clip_model_path_abs = os.path.abspath(self.this_clip_model_path)
        self.clip_model_path_abs = os.path.abspath(self.clip_model_path)
        if self.clip_model_path_abs == self.this_clip_model_path_abs and not os.path.exists(self.clip_model_path_abs):
            
            output_info = """
            # 下载clip模型到 ./assets/ 目录下
            # git clone https://huggingface.co/openai/clip-vit-base-patch16
            # or
            # git clone https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch16
            """
            # 将上述的信息已报错的形式输出
            raise ValueError(f"clip_model_path 路径不存在: {self.clip_model_path} , 请参考以下信息进行下载 : \n{output_info}")
        





if __name__ == "__main__":
    # 先看下 Qwen/Qwen2.5-0.5B的config
    config = Qwen2Config.from_pretrained("Qwen/Qwen2.5-0.5B")
    print(config.to_dict())
    {'vocab_size': 151936, 'max_position_embeddings': 32768, 'hidden_size': 896, 'intermediate_size': 4864,
     'num_hidden_layers': 24, 'num_attention_heads': 14, 'use_sliding_window': False, 'sliding_window': 32768,
     'max_window_layers': 24, 'num_key_value_heads': 2, 'hidden_act': 'silu', 'initializer_range': 0.02,
     'rms_norm_eps': 1e-06, 'use_cache': True, 'rope_theta': 1000000.0, 'rope_scaling': None, 'attention_dropout': 0.0,
     'return_dict': True, 'output_hidden_states': False, 'output_attentions': False, 'torchscript': False,
     'torch_dtype': 'bfloat16', 'use_bfloat16': False, 'tf_legacy_loss': False, 'pruned_heads': {},
     'tie_word_embeddings': True, 'chunk_size_feed_forward': 0, 'is_encoder_decoder': False, 'is_decoder': False,
     'cross_attention_hidden_size': None, 'add_cross_attention': False, 'tie_encoder_decoder': False, 'max_length': 20,
     'min_length': 0, 'do_sample': False, 'early_stopping': False, 'num_beams': 1, 'num_beam_groups': 1,
     'diversity_penalty': 0.0, 'temperature': 1.0, 'top_k': 50, 'top_p': 1.0, 'typical_p': 1.0,
     'repetition_penalty': 1.0, 'length_penalty': 1.0, 'no_repeat_ngram_size': 0, 'encoder_no_repeat_ngram_size': 0,
     'bad_words_ids': None, 'num_return_sequences': 1, 'output_scores': False, 'return_dict_in_generate': False,
     'forced_bos_token_id': None, 'forced_eos_token_id': None, 'remove_invalid_values': False,
     'exponential_decay_length_penalty': None, 'suppress_tokens': None, 'begin_suppress_tokens': None,
     'architectures': ['Qwen2ForCausalLM'], 'finetuning_task': None, 'id2label': {0: 'LABEL_0', 1: 'LABEL_1'},
     'label2id': {'LABEL_0': 0, 'LABEL_1': 1}, 'tokenizer_class': None, 'prefix': None, 'bos_token_id': 151643,
     'pad_token_id': None, 'eos_token_id': 151643, 'sep_token_id': None, 'decoder_start_token_id': None,
     'task_specific_params': None, 'problem_type': None, '_name_or_path': '', '_attn_implementation_autoset': False,
     'transformers_version': '4.50.0', 'model_type': 'qwen2', 'use_mrope': False}
