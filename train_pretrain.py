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
    model = MiniLLM(lm_config)
    # 打印总的模型参数，单位为百万
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    Log(f"Total parameters: {total_params:.2f}M (million)")
    # 打印可以训练的模型参数，单位为百万，并打印可以训练的模型参数占总参数的比例
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Log(f"Trainable parameters: {trainable_params:.2f}M (million)")
    Log(f"Trainable parameters ratio: {trainable_params/total_params:.2f}")
    return model,tokenizer

def get_lr(current_iter:int,total_iters:int,base_lr:float):
    """
    获取当前学习率，使用余弦退火策略，学习率缓慢下降，最小值为：base_lr / 10
    """
    return base_lr / 10 + 0.5 * base_lr * (1 + math.cos(math.pi * current_iter / total_iters))
    
def train_one_epoch(model,train_loader,optimizer,scaler,epoch,wandb):
    model.train()
    iter_per_epoch = len(train_loader) # 每个 epoch 的迭代次数
    loss_fn = CrossEntropyLoss(reduction="none") # 交叉熵损失函数,reduction="none" 表示不进行平均
    start_time = time.time()
    for step,batch in enumerate(tqdm(train_loader,desc=f"Training of epoch {epoch}",total=iter_per_epoch)):
        X,Y,loss_mask = batch
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)

        for param_group in optimizer.param_groups: # 更新学习率
            param_group["lr"] = lr


        with ctx: # 上下文管理器，用于混合精度训练
            res = model(X)
            """
            logits 是模型在输出层没有经过 softmax 的原始输出值。
            它代表模型对每个类别的“原始打分”，还没有被归一化成概率。比如对于语言模型来说：
            •	如果你有一个词表（vocab）大小是 50000，模型每一步都会输出一个大小为 [batch_size, seq_len, vocab_size] 的 tensor；
            •	这个输出就是对每个位置的 token，它属于每个词的“信心打分”——这个就是 logits。
            logits 是 softmax 前的值，形状是 [B, T, V]，分别表示 batch、序列长度、词表大小。
            """
            # res.logits.shape： (batch_size, seq_len, vocab_size) 
            # res.logits.view(-1,res.logits.size(-1)) -> (batch_size * seq_len, vocab_size)
            # Y.view(-1) -> (batch_size * seq_len)
            # CrossEntropyLoss 要求输入 logits 的 shape 是 (N, C) ，N 是样本数量，C 是类别数量;同时 target 的 shape 是 (N,)，其中每个值是类别的索引（即 token 的 ID）
            loss = loss_fn(res.logits.view(-1,res.logits.size(-1)), Y.view(-1)).view(Y.size()) # 计算损失，最终整理成 (batch_size, seq_len) 的形状
            loss = (loss * loss_mask).sum() / loss_mask.sum() # 计算损失的平均值 
            loss += res.aux_loss # 添加辅助损失
            loss = loss / args.accumulation_steps # 除以梯度累积次数
        scaler.scale(loss).backward() # 反向传播
        if (step +1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer) # 取消缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip) # 梯度裁剪
            scaler.step(optimizer) # 更新参数
            scaler.update() # 更新缩放因子
            optimizer.zero_grad(set_to_none=True) # 清空梯度
        
        if step % args.log_interval == 0 or step == iter_per_epoch - 1:
            spend_time = time.time() - start_time
            Log(f"Epoch: [{epoch}/{args.epochs}] Step: [{step}/{iter_per_epoch}] loss: {loss.item() * args.accumulation_steps:.4f} lr: {optimizer.param_groups[-1]['lr']:.6f} spend_time: {spend_time / (step+1) * iter_per_epoch / 60 - spend_time // 60:.4f}min")
        
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

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


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32 # cuda 使用 bfloat16 精度，mps 使用 float32 精度
    tokenizer_path = os.path.join(this_dir,"tokenizer_output")
    default_data_path = os.path.join(this_dir,"data_sample/wikipedia_zh_sample_data.json")
    parser = argparse.ArgumentParser(description="Train  pretrain model")
    parser.add_argument("--out_dir",type=str,default="output",help="The output directory")
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--learning_rate",type=float,default=5e-4)
    parser.add_argument("--device",type=str,default=device)
    parser.add_argument("--dtype",type=str,default=dtype)
    parser.add_argument("--use_wandb",action="store_true",default=True)
    parser.add_argument("--wandb_project",type=str,default="MiniLLM")
    parser.add_argument("--num_workers",type=int,default=1)
    parser.add_argument("--ddp",action="store_true")
    parser.add_argument("--local_rank",type=int,default=-1) # 分布式训练
    parser.add_argument("--accumulation_steps",type=int,default=1) # 梯度累计
    parser.add_argument("--grad_clip",type=float,default=1.0) # 梯度裁剪
    parser.add_argument("--warmup_iters",type=int,default=100) # 预热步数
    parser.add_argument("--log_interval",type=int,default=10) # 日志间隔
    parser.add_argument("--save_interval",type=int,default=100) # 保存间隔
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
        n_heads=8,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe
    )
    Log(f"lm_config: {lm_config}")
    tokens_per_iter = args.batch_size * lm_config.max_seq_len # 每个迭代步的token数
    torch.manual_seed(2025) # 设置随机种子
    device_type = args.device

    args.wandb_run_name = f"MiniLLM_{args.max_seq_len}B_{args.batch_size}B_{args.epochs}epochs_{args.dim}dim_{args.use_moe}moe_{args.n_layers}layers"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast() # 上下文管理器

    ddp = int(os.environ.get("RANK",-1)) != -1
    ddp_local_rank,DEVICE = 0,"cuda"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
    if args.use_wandb and (not ddp or ddp_local_rank == 0): # 如果使用 wandb 且不是分布式训练或当前进程是主进程，则初始化 wandb
        import wandb 
        wandb.init(project=args.wandb_project,name=args.wandb_run_name,config=vars(args))
    else:
        wandb = None
    Log("loading model")
    model,tokenizer = init_model(lm_config,tokenizer_path)
    model.to(args.device)
    Log("loading data")
    train_ds = PretrainDataset(args.data_path,
                               tokenizer,
                               args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None # 分布式训练的采样器
    train_loader = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        pin_memory=True, # 是否使用内存
        drop_last=True, # 是否丢弃最后一个批次
        shuffle=False if ddp else True, # 是否打乱数据
        num_workers=args.num_workers, # 使用的线程数
        sampler=train_sampler # 采样器
    )
    Log("loading optimizer")
    scaler = torch.amp.GradScaler(enabled=(args.dtype in ["bfloat16","float16"])) # 梯度缩放，用于防止梯度爆炸
    optimizer = optim.AdamW(model.parameters(),lr=args.learning_rate) # 使用 AdamW 优化器
    if ddp:
        Log("model distributed")
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"} # 忽略分布式训练的参数
        model = DistributedDataParallel(model,device_ids = [ddp_local_rank]) # 分布式训练
    Log("training")
    for epoch in range(args.epochs):
        train_one_epoch(model,train_loader,optimizer,scaler,epoch,wandb)

    Log("done")
