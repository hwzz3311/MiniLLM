import torch
from torch import nn

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank:int=4, alpha:float=1.0, dropout:float=0.0):
        """
        LoRA 是一种用于神经网络的低秩适应方法，通过在权重矩阵中添加低秩矩阵来实现参数的增量更新。
        本质上相当于在原来的全连接层，添加一个低秩矩阵的旁路，使得梯度更新时，不再是直接更新权重矩阵，而是更新低秩矩阵。
        
        Args:
            in_features (int): 输入特征的维度
            out_features (int): 输出特征的维度
            rank (int): LoRA 的秩，可以控制低秩矩阵的大小
            alpha (float): 缩放因子
            dropout (float): 随机失活的比例
        """
        super().__init__()
        self.rank = rank # LoRA 的秩，可以控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False) # 低秩矩阵 A
        self.B = nn.Linear(rank, out_features, bias=False) # 低秩矩阵 B
        self.alpha = alpha # 缩放因子
        self.scaling = alpha / rank # 缩放因子
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # 随机失活 or 恒等映射
        # 初始化权重
    def forward(self, x):
        x = self.dropout(x) # 随机失活
        # 通过低秩矩阵 A 进行降维，将输入特征从 in_features 降到 rank 维
        x = self.A(x)
        # 通过低秩矩阵 B 进行升维，将 rank 维特征恢复到 out_features 维
        x = self.B(x)
        # 应用缩放因子，控制 LoRA 更新的强度
        x = x * self.scaling
        return x
    

def apply_lora(model:nn.Module, rank:int=4,target_layers:list[str]=["wq","wv","wo"], alpha:float=1.0, dropout:float=0.0):
    """
    在模型中应用 LoRA 方法。
    Args:
        model (nn.Module): 需要应用 LoRA 的模型
        rank (int): LoRA 的秩
        target_layers (list[str]): 需要应用 LoRA 的层
        alpha (float): 缩放因子
        dropout (float): 随机失活的比例
    """
    # 遍历模型中的所有模块
    for name,module in model.named_modules():
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 检查模块的名称是否在 target_layers 中，自己对目标层进行 LoRA 操作
            if any(name.endswith(suffix) for suffix in target_layers):
                lora = LoRA(module.in_features,module.out_features,rank=rank,alpha=alpha,dropout=dropout)
                # 为其添加一个 lora 属性
                lora.to(module.weight.device)
                setattr(module,"lora",lora)
                # 保存原始的 forward 方法
                original_forward = module.forward 
                # 重写前向传播的方法
                def forward_with_lora(x,
                                    layer1=original_forward,
                                    layer2=lora.forward):
                    return layer1(x) + layer2(x)
                # 将重写后的 forward 方法赋值给 module 的 forward 属性
                module.forward = forward_with_lora

def load_lora(model:nn.Module,path:str):
    """
    加载 LoRA 模型
    """
    state_dict = torch.load(path, map_location=model.device)
    for name,module in model.named_modules():
        if hasattr(module, "lora"):
            # 获取 LoRA 的权重
            lora_state_dict = {k.replace(f"{name}.lora.",""):v for k,v in state_dict.items() if f"{name}.lora." in k}
            module.lora.load_state_dict(lora_state_dict)

    print(f"{path} LoRA 模型已加载到 {model.__class__.__name__}")


def save_lora(model:nn.Module,path:str):
    """
    保存 LoRA 模型
    """
    state_dict = {}
    for name,module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state_dict = {f"{name}.lora.{k}":v for k,v in module.lora.state_dict().items()}
            state_dict.update(lora_state_dict)
    torch.save(state_dict,path)
    print(f"LoRA 模型已保存到 {path}")



