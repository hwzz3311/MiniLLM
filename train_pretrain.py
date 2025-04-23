import os
import time
import argparse
import torch
import math
from torch.distributed import init_process_group
import torch.distributed as dist
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DistributedSampler,DataLoader
import torch.optim as optim
from tqdm import tqdm

from src.model.model import MiniLLM
from src.data.dataset import PretrainDataset
from src.model.config import MiniLLMConfig


def Log(message:str):
    # 处理分布式训练的日志
    print(message)

def init_distributed_mode():
    if not ddp:return # 如果不是分布式训练，则直接返回
    global ddp_local_rank,DEVICE
    dist.init_process_group(backend="nccl") # 初始化分布式训练，使用 nccl 后端
    ddp_rank = int(os.environ["RANK"]) # 当前进程的 rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # 当前进程的本地 rank
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    dist.barrier() # 同步所有进程
    Log(f"Distributed training: {ddp_rank} / {ddp_local_rank} on {DEVICE}")
    
def init_model(lm_config:MiniLLMConfig,tokenizer_path:str):
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
    Log(f"Trainable parameters ratio: {trainable_params/total_params:.2f}")
    return model,tokenizer

def get_lr(current_step, total_steps, lr,warmup_iters):
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

def train_one_epoch(model,train_loader,optimizer,scaler,epoch,wandb):
    model.train()
    iter_per_epoch = len(train_loader)
    loss_fn = CrossEntropyLoss(reduction="none")
    start_time = time.time()
    # 添加梯度检查
    def check_gradients():
        # 统计梯度范数的平均值
        grad_norms = []
        avg_grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm > 1000:  # 严重警告
                    Log(f"严重警告：参数 {name} 的梯度范数过大: {grad_norm}")
                    torch.nn.utils.clip_grad_norm_(param, 20.0) #
                elif grad_norm > 100:  # 一般警告
                    Log(f"警告：参数 {name} 的梯度范数较大: {grad_norm}")
                elif grad_norm < 0.01:  # 梯度消失警告
                    Log(f"警告：参数 {name} 的梯度范数过小: {grad_norm}")
        # 计算梯度范数的平均值
        if len(grad_norms) > 0:
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
        return avg_grad_norm
    # 计算总步数
    total_steps = args.epochs * iter_per_epoch
    
    for step,batch in enumerate(tqdm(train_loader,desc=f"Training of epoch {epoch}",total=iter_per_epoch)):
        X,Y,loss_mask = batch
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
        # lr = get_lr(current_step, args.epochs * iter_per_epoch, args.learning_rate)

        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            res = model(X)
            # logits = res.logits
            
            # # 检查 logits 是否包含 NaN
            # if torch.isnan(logits).any():
            #     Log(f"发现 NaN 值！")
            #     # 检查每一层的输出
            #     for name, module in model.named_modules():
            #         if isinstance(module, torch.nn.Linear):
            #             if hasattr(module, 'output'):
            #                 output = module.output
            #                 if torch.isnan(output).any():
            #                     Log(f"层 {name} 的输出包含 NaN 值")
            #                     Log(f"  - 形状: {output.shape}")
            #                     Log(f"  - 统计信息:")
            #                     Log(f"    - 最大值: {output.max().item()}")
            #                     Log(f"    - 最小值: {output.min().item()}")
            #                     Log(f"    - 平均值: {output.mean().item()}")
            #                     Log(f"    - 标准差: {output.std().item()}")
                
            #     # 如果发现 NaN，跳过这个 batch
            #     continue
            
            loss = loss_fn(
                res.logits.view(-1,res.logits.size(-1)), 
                Y.view(-1)
                ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps
            
        scaler.scale(loss).backward()
        if (step +1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
                log_dict = {"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           }
                wandb.log(log_dict)

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() ==0) or step == iter_per_epoch - 1:
            model.eval()
            moe_path = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/dim_{args.dim}/n_layers_{args.n_layers}/epoch_{epoch}_step_{step}{moe_path}.pth"
            os.makedirs(os.path.dirname(ckp),exist_ok=True)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict,ckp)
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
    dtype = "bfloat16" if device == "cuda" else "float32" # cuda 使用 bfloat16 精度，mps 使用 float32 精度
    minillm_tokenizer_path = os.path.join(this_dir,"./assets/minillm_tokenizer")
    qwen_tokenizer_path = os.path.join(this_dir,"./assets/qwen_tokenizer")
    minimind_tokenizer_path = os.path.join(this_dir,"./assets/minimind_tokenizer")
    tokenizer_path = minillm_tokenizer_path
    # default_data_path = os.path.join(this_dir,"data_sample/baidubaike_wikipedia_sample_data.parquet")
    # default_data_path = "/mnt/d/pretrain/minimind/pretrain_hq.parquet" # 更换为minimind数据集测试效果
    default_data_path = "/mnt/d/pretrain/merge_data/baidubaike_wikipedia_sample_data_100min_512max.parquet" # 1.2G
    out_dir = os.path.join(this_dir,"./assets/minillm_output")
    model_dir = os.path.join(out_dir,"dim_512/n_layers_8")
    model_check_point_path = ""
    if os.path.exists(model_dir):
        # 获取模型目录下所有文件,安装创建时间进行倒序，
        model_files = os.listdir(model_dir)
        model_files.sort(key=lambda x: os.path.getctime(os.path.join(model_dir, x)), reverse=True)
        if len(model_files) > 0:
            model_check_point_path = os.path.join(model_dir, model_files[0])
            print(f"使用模型: {model_check_point_path}")

    parser = argparse.ArgumentParser(description="Train  pretrain model")
    parser.add_argument("--out_dir",type=str,default=out_dir,help="The output directory")
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--learning_rate",type=float,default=5e-4)
    parser.add_argument("--checkpoint_path",default=model_check_point_path)
    parser.add_argument("--device",type=str,default=device)
    parser.add_argument("--dtype",type=str,default=dtype)
    parser.add_argument("--use_wandb",action="store_true",default=True)
    parser.add_argument("--wandb_project",type=str,default="MiniLLM")
    parser.add_argument("--num_workers",type=int,default=1)
    parser.add_argument("--ddp",action="store_true")
    parser.add_argument("--local_rank",type=int,default=-1) # 分布式训练
    parser.add_argument("--accumulation_steps",type=int,default=8) # 梯度累计
    parser.add_argument("--grad_clip",type=float,default=1.0) # 梯度裁剪 暂时关闭
    parser.add_argument("--warmup_iters",type=int,default=0) # 预热步数
    parser.add_argument("--log_interval",type=int,default=100) # 日志间隔
    parser.add_argument("--save_interval",type=int,default=1000) # 保存间隔
    parser.add_argument("--dim",type=int,default=512) # 隐层维度
    parser.add_argument("--n_layers",type=int,default=8) # 层数
    parser.add_argument("--max_seq_len",type=int,default=512) # 最大序列长度
    parser.add_argument("--use_moe",default=False,type=bool) # 是否使用 MoE
    parser.add_argument("--data_path",type=str,default=default_data_path) # 数据路径
    
    args = parser.parse_args()
    Log("start training")
    print(args)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir,exist_ok=True)
    lm_config = MiniLLMConfig(
        hidden_size=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe
    )
    Log(f"lm_config: {lm_config}")
    tokens_per_iter = args.batch_size * lm_config.max_seq_len # 每个迭代步的token数

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    device_type = args.device

    args.wandb_run_name = f"MiniLLM_{args.max_seq_len}seq_{args.batch_size}B_{args.epochs}epochs_{args.dim}dim_{args.use_moe}moe_{args.n_layers}layers"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast() # 上下文管理器

    ddp = int(os.environ.get("RANK",-1)) != -1
    ddp_local_rank,DEVICE = 0,"cuda"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
    Log("loading model")
    model,tokenizer = init_model(lm_config,tokenizer_path)
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path))
        Log(f"load check_point {args.checkpoint_path} success!")
    model.to(args.device)
    Log("loading data")
    train_ds = PretrainDataset(args.data_path,
                               tokenizer,
                               args.max_seq_len,
                               num_workers=16)
    train_sampler = DistributedSampler(train_ds) if ddp else None # 分布式训练的采样器
    train_loader = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        pin_memory=True, # 是否使用内存
        drop_last=True, # 是否丢弃最后一个批次
        shuffle=False, # 是否打乱数据
        num_workers=args.num_workers, # 使用的线程数
        sampler=train_sampler # 采样器
    )
    Log("loading optimizer")
    scaler = torch.amp.GradScaler(enabled=(args.dtype in ["bfloat16","float16"])) # 梯度缩放，用于防止梯度爆炸
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.learning_rate) # 使用 AdamW 优化器
    if ddp:
        Log("model distributed")
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"} # 忽略分布式训练的参数
        model = DistributedDataParallel(model,device_ids = [ddp_local_rank]) # 分布式训练
    Log("training")
    # 等待数据和模型都初始化后，再初始化 wandb
    if args.use_wandb and (not ddp or ddp_local_rank == 0): # 如果使用 wandb 且不是分布式训练或当前进程是主进程，则初始化 wandb
        import wandb
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name,
                   config=vars(args),
                   id="9ov46iyz",
                   resume="must"
                   )
    else:
        wandb = None
    for epoch in range(args.epochs):
        train_one_epoch(model,train_loader,optimizer,scaler,epoch,wandb)
        # train_epoch_old(model,train_loader,optimizer,scaler,epoch,wandb)

    Log("done")
