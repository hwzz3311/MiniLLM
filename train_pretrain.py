import os
import time
import argparse
import torch
import math
import torch.distributed as dist
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DistributedSampler, DataLoader
import torch.optim as optim
from tqdm import tqdm

from src.model.model import MiniLLM
from src.model.model_vl import MiniLLM_VL
from src.data.dataset import PretrainDataset, PretrainVLDataset, SFT_Dataset
from src.model.config import MiniLLMConfig, MiniLLM_VLConfig


def Log(message: str):
    # 处理分布式训练的日志
    print(message)


def init_distributed_mode():
    if not ddp: return  # 如果不是分布式训练，则直接返回
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")  # 初始化分布式训练，使用 nccl 后端
    ddp_rank = int(os.environ["RANK"])  # 当前进程的 rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 当前进程的本地 rank
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    dist.barrier()  # 同步所有进程
    Log(f"Distributed training: {ddp_rank} / {ddp_local_rank} on {DEVICE}")


def init_model(lm_config: MiniLLMConfig, tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # from src.model.minimind_model import MiniMindLM as MiniLLM

    # 添加词表大小检查
    Log(f"Tokenizer vocab size: {len(tokenizer)}")
    Log(f"Model vocab size: {lm_config.vocab_size}")
    if len(tokenizer) != lm_config.vocab_size:
        Log(f"词表大小不匹配，请检查词表大小")
        Log(f"强制将 模型词表大小 设置为 tokenizer 词表大小，{lm_config.vocab_size} -> {len(tokenizer)}")
        lm_config.vocab_size = len(tokenizer)

    model = MiniLLM(lm_config)
    # 添加更详细的初始化检查
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            Log(f"警告：参数 {name} 在初始化时包含 NaN 值")
            Log(f"  - 形状: {param.shape}")
            Log(f"  - 数据类型: {param.dtype}")
            Log(f"  - 统计信息:")
            Log(f"    - 最大值: {param.max().item()}")
            Log(f"    - 最小值: {param.min().item()}")
            Log(f"    - 平均值: {param.mean().item()}")
            Log(f"    - 标准差: {param.std().item()}")

    # 打印总的模型参数，单位为百万
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    Log(f"Total parameters: {total_params:.2f}M (million)")
    # 打印可以训练的模型参数，单位为百万，并打印可以训练的模型参数占总参数的比例
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Log(f"Trainable parameters: {trainable_params:.2f}M (million)")
    Log(f"Trainable parameters ratio: {trainable_params / total_params:.2f}")
    return model, tokenizer


def init_vl_model(lm_config: MiniLLM_VLConfig,
                  tokenizer_path: str,
                  args: argparse.Namespace
                  ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 添加词表大小检查
    Log(f"Tokenizer vocab size: {len(tokenizer)}")
    Log(f"Model vocab size: {lm_config.vocab_size}")
    if len(tokenizer) != lm_config.vocab_size:
        Log(f"词表大小不匹配，请检查词表大小")
        Log(f"强制将 模型词表大小 设置为 tokenizer 词表大小，{lm_config.vocab_size} -> {len(tokenizer)}")
        lm_config.vocab_size = len(tokenizer)
    llm_cpk = args.llm_checkpoint_path
    if llm_cpk is None:
        Log(f"llm_checkpoint_path 为空，请检查 llm_checkpoint_path")
        raise ValueError(f"llm_checkpoint_path 为空，请检查 llm_checkpoint_path")
    state_dict = torch.load(llm_cpk, map_location=args.device)
    model = MiniLLM_VL(lm_config)
    model.load_state_dict(state_dict, strict=False)  # 设置 strict 为 False，忽略不匹配的参数，因为 MiniLLM_VL 的参数比 MiniLLM 多

    # 冻结 vision_proj 外的所有参数，只训练 视觉的投影层的参数
    # for name, param in model.named_parameters():
    #     if "vision_proj" not in name:
    #         param.requires_grad = False
    # # 设置可训练的层
    # if hasattr(model, "layers"):
    #     # 只训练最后两层，因为最后两层是视觉的投影层
    #     last_two_layers = model.layers[-1:]
    #     for layer in last_two_layers:
    #         for param in layer.parameters():
    #             param.requires_grad = True

    # 添加更详细的初始化检查
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            Log(f"警告：参数 {name} 在初始化时包含 NaN 值")
            Log(f"  - 形状: {param.shape}")
            Log(f"  - 数据类型: {param.dtype}")
            Log(f"  - 统计信息:")
            Log(f"    - 最大值: {param.max().item()}")
            Log(f"    - 最小值: {param.min().item()}")
            Log(f"    - 平均值: {param.mean().item()}")
            Log(f"    - 标准差: {param.std().item()}")

    # 打印总的模型参数，单位为百万
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    Log(f"Total parameters: {total_params:.2f}M (million)")
    # 打印可以训练的模型参数，单位为百万，并打印可以训练的模型参数占总参数的比例
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Log(f"Trainable parameters: {trainable_params:.2f}M (million)")
    Log(f"Trainable parameters ratio: {trainable_params / total_params:.2f}")
    _, preprocess = MiniLLM_VL.load_vision_model(lm_config.clip_model_path)
    return model, tokenizer, preprocess


def get_lr(current_step, total_steps, lr, warmup_iters):
    """
    改进的学习率调度策略：
    1. 添加预热阶段
    2. 使用余弦退火
    3. 设置最小学习率
    """
    # 预热步数
    warmup_steps = min(int(total_steps * 0.1), warmup_iters)  # 预热步数不超过500
    min_lr = lr * 0.2  # 最小学习率为初始学习率的0.1倍

    if current_step < warmup_steps:
        # 使用更激进的预热
        return lr * (current_step / warmup_steps) ** 0.5
    else:
        # 余弦退火
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_dynamic_batch_size(seq_lengths, max_tokens_per_batch):
    """根据序列长度动态调整批次大小"""
    batch_size = 0
    total_tokens = 0
    for length in seq_lengths:
        if total_tokens + length > max_tokens_per_batch:
            break
        total_tokens += length
        batch_size += 1
    return batch_size


def collate_fn(batch):
    """自定义的collate函数，处理不同长度的序列"""
    # 按序列长度排序，便于后续处理
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    # 获取当前批次中所有序列的长度
    seq_lengths = [len(item[0]) for item in batch]
    
    # 使用动态批处理大小
    if args.use_dynamic_length:
        batch_size = get_dynamic_batch_size(seq_lengths, args.max_tokens_per_batch)
        batch = batch[:batch_size]
    
    # 获取当前批次中最长序列的长度
    max_len = len(batch[0][0])
    
    # 准备批次数据
    X_batch = []
    Y_batch = []
    loss_mask_batch = []
    
    for X, Y, loss_mask in batch:
        # 填充到当前批次的最大长度
        pad_length = max_len - len(X)
        if pad_length > 0:
            X = torch.cat([X, torch.full((pad_length,), args.tokenizer.pad_token_id, dtype=X.dtype)])
            Y = torch.cat([Y, torch.full((pad_length,), args.tokenizer.pad_token_id, dtype=Y.dtype)])
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_length, dtype=loss_mask.dtype)])
        
        X_batch.append(X)
        Y_batch.append(Y)
        loss_mask_batch.append(loss_mask)
    
    return {
        'X': torch.stack(X_batch),
        'Y': torch.stack(Y_batch),
        'loss_mask': torch.stack(loss_mask_batch)
    }


def train_one_epoch(model, train_loader, optimizer, scaler, epoch, wandb, args):
    model.train()
    iter_per_epoch = len(train_loader)
    loss_fn = CrossEntropyLoss(reduction="none")
    start_time = time.time()

    # 计算总步数
    total_steps = args.epochs * iter_per_epoch

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training of epoch {epoch}", total=iter_per_epoch)):
        if args.train_model == "llm-vl":
            X, Y, loss_mask, pixel_tensors = batch
            pixel_tensors = pixel_tensors.to(args.device)
        else:
            X, Y, loss_mask = batch['X'], batch['Y'], batch['loss_mask']
            
        # 获取当前批次的实际长度
        batch_size, seq_len = X.shape
        
        # 将数据移动到设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 计算当前步数
        current_step = epoch * iter_per_epoch + step

        # 获取动态学习率
        lr = get_lr(current_step,
                    total_steps,
                    args.learning_rate,
                    args.warmup_iters)

        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            if args.train_model == "llm-vl":
                res = model(X, pixel_tensors=pixel_tensors)
            else:
                res = model(X)

            # 计算损失时考虑实际序列长度
            loss = loss_fn(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            
            # 使用 loss_mask 和实际长度计算损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        # 使用 Flash Attention 时，建议使用混合精度训练
        scaler.scale(loss).backward()
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), min(args.grad_clip, 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iter_per_epoch - 1:
            spend_time = time.time() - start_time
            Log(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                log_dict = {
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                }
                wandb.log(log_dict)
            if args.train_model != "llm-vl":
                input_text = tokenizer.batch_decode(X)[0]
                predict_text = tokenizer.batch_decode(torch.argmax(res.logits,dim=-1))[0]
                Log("*"*50)
                Log(f"input text: {input_text}")
                Log("=" * 50)
                Log(f"predict_text: {predict_text}")
                Log("*" * 50)

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0) or step == iter_per_epoch - 1:
            model.eval()
            moe_path = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/dim_{args.dim}/n_layers_{args.n_layers}/epoch_{epoch}_step_{step}{moe_path}.pth"
            os.makedirs(os.path.dirname(ckp), exist_ok=True)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            Log(f"Save checkpoint to {ckp}")
            model.train()

        # # 添加数值稳定性检查
        # if torch.isnan(logits).any():
        #     # 检查每一层的输出
        #     for name, module in model.named_modules():
        #         if hasattr(module, 'output'):
        #             output = module.output
        #             if torch.isnan(output).any():
        #                 Log(f"层 {name} 的输出包含 NaN 值")
        #                 Log(f"  - 形状: {output.shape}")
        #                 Log(f"  - 统计信息:")
        #                 Log(f"    - 最大值: {output.max().item()}")
        #                 Log(f"    - 最小值: {output.min().item()}")
        #                 Log(f"    - 平均值: {output.mean().item()}")
        #                 Log(f"    - 标准差: {output.std().item()}")


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = "bfloat16" if device == "cuda" else "float32"  # cuda 使用 bfloat16 精度，mps 使用 float32 精度
    minillm_tokenizer_path = os.path.join(this_dir, "./assets/minillm_tokenizer")
    qwen_tokenizer_path = os.path.join(this_dir, "./assets/qwen_tokenizer")
    minimind_tokenizer_path = os.path.join(this_dir, "./assets/minimind_tokenizer")
    tokenizer_path = minillm_tokenizer_path
    # default_data_path = os.path.join(this_dir,"data_sample/baidubaike_wikipedia_sample_data.parquet")
    # default_data_path = "/mnt/d/pretrain/minimind/pretrain_hq.parquet" # 更换为minimind数据集测试效果
    default_data_path = "/mnt/d/pretrain/merge_data/baidubaike_wikipedia_sample_data_100min_512max.parquet"  # 1.2G
    default_data_path = "/mnt/d/pretrain/minimind/pretrain_hq.parquet"
    sft_data_path = "/mnt/d/pretrain/minimind/sft_mini_512.jsonl"
    default_vl_data_path = "/mnt/d/pretrain/minimind-v_dataset/sft_vlm_data.jsonl"
    default_data_path = sft_data_path
    default_image_base_dir = "/mnt/d/pretrain/minimind-v_dataset/sft_images"

    use_moe = False
    train_model = "llm"
    sft = True
    pretrain = False
    mode = "sft"
    model_output_dir = os.path.join(this_dir, f"./assets/mini{train_model}_output")
    model_output_dir = os.path.join(model_output_dir, mode)

    model_dir = os.path.join(model_output_dir, "dim_512/n_layers_8/")
    llm_model_check_point_path = ""
    if os.path.exists(model_dir):
        # 获取模型目录下所有文件，按照创建时间进行倒序。
        model_files = os.listdir(model_dir)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(model_dir, x)), reverse=True)
        if len(model_files) > 0:
            llm_model_check_point_path = os.path.join(model_dir, model_files[0])
            print(f"llm 使用模型: {llm_model_check_point_path}")
    # llm_model_check_point_path = "/mnt/d/linux/LLM/MiniLLM/assets/minillm_output/sft/dim_512/minillm_sft_v1.0_build20250425.pth"
    vl_model_check_point_path = ""
    if os.path.exists(model_dir):
        # 获取模型目录下所有文件，按照创建时间进行倒序。
        model_files = os.listdir(model_dir)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(model_dir, x)), reverse=True)
        if len(model_files) > 0:
            vl_model_check_point_path = os.path.join(model_dir, model_files[0])
            print(f"llm-vl 使用模型: {vl_model_check_point_path}")
    # vl_model_check_point_path = ""
    

    parser = argparse.ArgumentParser(description="Train pretrain model")
    parser.add_argument("--out_dir", type=str, default=model_output_dir, help="The output directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mode", type=str, default=mode)
    parser.add_argument("--train_model", type=str, default=train_model)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=9e-4)
    parser.add_argument("--llm_checkpoint_path", default=llm_model_check_point_path)
    parser.add_argument("--vl_checkpoint_path", default=vl_model_check_point_path)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--dtype", type=str, default=dtype)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="MiniLLM")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)  # 分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=8)  # 梯度累计
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪 暂时关闭
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热步数
    parser.add_argument("--log_interval", type=int, default=50)  # 日志间隔
    parser.add_argument("--save_interval", type=int, default=500)  # 保存间隔
    parser.add_argument("--dim", type=int, default=512)  # 隐层维度
    parser.add_argument("--n_layers", type=int, default=8)  # 层数
    parser.add_argument("--max_seq_len", type=int, default=512)  # 最大序列长度
    parser.add_argument("--use_moe", default=use_moe, type=bool)  # 是否使用 MoE
    parser.add_argument("--data_path", type=str, default=default_data_path)  # 数据路径
    parser.add_argument("--image_base_dir", type=str, default=default_image_base_dir)  # 图片路径
    parser.add_argument("--max_tokens_per_batch", type=int, default=8192, 
                        help="每个批次的最大token数")
    parser.add_argument("--use_dynamic_length", action="store_true", 
                        help="是否使用动态长度训练")

    args = parser.parse_args()
    args.max_tokens_per_batch = max(args.max_seq_len * args.batch_size, args.max_tokens_per_batch)
    Log("start training")
    print(args)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    if train_model == "llm":
        lm_config = MiniLLMConfig(
            hidden_size=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe,
            flash_attn=True  # 确保启用 Flash Attention
        )
    elif train_model == "llm-vl":
        lm_config = MiniLLM_VLConfig(
            hidden_size=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe,
            flash_attn=True  # 确保启用 Flash Attention
        )
    else:
        raise ValueError(f"train_model 参数错误，请检查 train_model 参数")
    Log(f"lm_config: {lm_config}")
    tokens_per_iter = args.batch_size * lm_config.max_seq_len  # 每个迭代步的token数

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    device_type = args.device

    args.wandb_run_name = f"MiniLLM_{args.max_seq_len}seq_{args.batch_size}B_{args.epochs}epochs_{args.dim}dim_{args.use_moe}moe_{args.n_layers}layers"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()  # 上下文管理器

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
    Log("loading model")
    if train_model == "llm":
        model, tokenizer = init_model(lm_config, tokenizer_path)
        if args.llm_checkpoint_path:
            model.load_state_dict(torch.load(args.llm_checkpoint_path))
            Log(f"load check_point {args.llm_checkpoint_path} success!")
    elif train_model == "llm-vl":
        model, tokenizer, preprocess = init_vl_model(lm_config, tokenizer_path, args)
        if args.vl_checkpoint_path:
            model.load_state_dict(torch.load(args.vl_checkpoint_path))
            Log(f"load check_point {args.vl_checkpoint_path} success!")
    args.tokenizer = tokenizer
    model.to(args.device)
    Log("loading data")
    if train_model == "llm":
        if args.mode == "pretrain":
            train_ds = PretrainDataset(args.data_path,
                                   tokenizer,
                                   args.max_seq_len,
                                   num_workers=16)
        elif args.mode == "sft":
            train_ds = SFT_Dataset(args.data_path,
                                       tokenizer,
                                       args.max_seq_len,
                                       num_workers=16)
    elif train_model == "llm-vl":
        assert args.data_path is not None, "data_path 不能为空"
        assert args.image_base_dir is not None, "image_base_dir 不能为空"
        train_ds = PretrainVLDataset(data_path=args.data_path,
                                           tokenizer=tokenizer,
                                           preprocess=preprocess,
                                           image_base_dir=args.image_base_dir,
                                           max_len=args.max_seq_len,
                                           chunk_size=10000,
                                           num_workers=32)
    else:
        raise ValueError(f"train_model 参数错误，请检查 train_model 参数")
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 分布式训练的采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
        sampler=train_sampler,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    Log("loading optimizer")
    scaler = torch.amp.GradScaler(enabled=(args.dtype in ["bfloat16", "float16"]))  # 梯度缩放，用于防止梯度爆炸
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.learning_rate)  # 使用 AdamW 优化器
    if ddp:
        Log("model distributed")
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 忽略分布式训练的参数
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])  # 分布式训练
    Log("training")
    # 等待数据和模型都初始化后，再初始化 wandb
    if args.use_wandb and (not ddp or ddp_local_rank == 0):  # 如果使用 wandb 且不是分布式训练或当前进程是主进程，则初始化 wandb
        # wandb = None

        import wandb
        # pretrain
        # wandb.init(project=args.wandb_project,
        #            name=args.wandb_run_name,
        #            config=vars(args),
        #            id="5v6d9l8e",
        #            resume="must"
        #            )
        # sft
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name,
                   config=vars(args),
                   id="dfyg2h9t",
                   resume="must"
                   )

    else:
        wandb = None
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, scaler, epoch, wandb, args)

    Log("done")
