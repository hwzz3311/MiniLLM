import torch
import torch.nn as nn
from .model import MiniLLM,MOEFeedForward
from .config import MiniLLM_VLConfig
from transformers import CLIPModel,CLIPProcessor
from typing import Optional,List,Tuple

# 将语言模型增强至多模态模型：CLIP+LLM(Vicuna，LLaMA结构)

class VisionProj(nn.Module):
    def __init__(self, vision_dim=768, embed_dim=512):
        super().__init__()
        # 输入维度
        self.vision_dim = vision_dim
        # 输出维度
        self.embed_dim = embed_dim

        # 线性层
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_dim,self.embed_dim),
        )

    def forward(self,image_encoders):
        vision_features = self.vision_proj(image_encoders)
        return vision_features


class MiniLLM_VL(MiniLLM):
    def __init__(self,config:MiniLLM_VLConfig=None):
        config = config or MiniLLM_VLConfig()
        super().__init__(config)
        self.config = config
        self.vision_encoder, self.processor = self.load_vision_model(config.clip_model_path)
        self.vision_proj = VisionProj(embed_dim=config.hidden_size)

    @staticmethod
    def load_vision_model(model_path):
        # 加载CLIP模型
        clip_model = CLIPModel.from_pretrained(model_path)
        # 加载CLIP模型
        clip_processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结CLIP模型参数
        for param in clip_model.parameters():
            param.requires_grad = False
        return clip_model.eval(), clip_processor

    @staticmethod
    def image2tensor(image, processor):
        # 将PIL.Image.Image转换为tensor
        if image.mode in ["LA","RGBA"]:
            image = image.convert("RGB")
        image_inputs = processor(image,return_tensors="pt")["pixel_values"]
        return image_inputs
    
    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        # 获取图像特征
        with torch.no_grad():
            image_features = vision_model.vision_model(pixel_values=image_tensors)
        img_emb = image_features.last_hidden_state[:,1:,:].squeeze() # 去除CLIP的[CLS]，然后展平
        return img_emb
    
    def count_vision_proj(self,tokens,h,vision_tensors=None,seq_len=512):
        # 计算图像投影层
        def find_indices(tokens, images_ids):
            # 找到图像token的索引
            image_ids_tensor = torch.tensor(images_ids).to(tokens.device)
            len_image_ids = len(images_ids)
            # 如果图像token的索引大于tokens的长度，则说明没有图像token
            if len_image_ids >  tokens.size(1):
                return None
            # 将tokens展平；unfold(dim,size,step)：在dim维度上，以size为窗口大小，步长为step，将tokens展平
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            # 判断tokens_view是否与image_ids_tensor相等;
            # all(dim=2)：判断tokens_view的每一行是否与image_ids_tensor的每一行相等
            matches = (tokens_view == image_ids_tensor).all(dim=2) # matches 是一个bool类型的张量
            res = {}
            for batch_idx in range(tokens.size(0)): # 遍历batch_size
                if matches[batch_idx].any(): # 如果batch_idx中存在图像token
                    y_list = []
                    # 遍历matches[batch_idx]中为True的索引；既图像token的索引
                    for idx in matches[batch_idx].nonzero(as_tuple=True)[0]:
                        y_list.append((
                            idx.item(),  # 图像token的索引
                            idx.item() + len_image_ids -1 # 图像token的结束索引
                            ))
                    res[batch_idx] = y_list
            return res or None
        image_indices = find_indices(tokens, self.config.image_token_id)
        if vision_tensors is not None and image_indices:
            # 计算图像投影层
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                # 如果vision_proj的形状为3，则说明有多个图像，需要将vision_proj的形状变为4
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            # 遍历batch_size
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    # 遍历图像token的索引
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            # 将图像投影层与文本投影层拼接
                            h_i = torch.cat(
                                (
                                    h_i[:start_idx],
                                    vision_proj[i][img_idx],
                                    h_i[end_idx+1:]
                                ), dim=0
                            )[:seq_len]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            h = torch.stack(new_h,dim=0)
        return h

    
    def forward(self,
                input_ids:Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None,
                use_cache:bool=False,
                **kwargs):
        start_pos =  kwargs.get("start_pos",0) # 获取开始位置
        pixel_tensors = kwargs.get("pixel_tensors", None) # 获取图像张量
        h = self.tok_embeddings(input_ids) # 获取输入的token嵌入 (文本内容)

        if pixel_tensors is not None and start_pos == 0: # 如果图像张量存在且开始位置为0
            if len(pixel_tensors.shape) == 6: # 如果图像张量形状为6？TODO 这里的6是什么？答：
                pixel_tensors = pixel_tensors.squeeze(2) # 压缩维度
            bs, num, c, im_h, im_w = pixel_tensors.shape # 获取图像张量形状
            stack_dim = 1 if bs > 1 else 0 # 如果batch_size大于1，则堆叠维度为1，否则为0
            vision_tensors = torch.stack([
                MiniLLM_VL.get_image_embeddings(
                    pixel_tensors[:,i,:,:,:],
                    self.vision_encoder
                ) for i in range(num)
            ],dim=stack_dim) # 堆叠图像张量
            h = self.count_vision_proj(
                tokens=input_ids,
                h=h,
                vision_tensors=vision_tensors,
                seq_len=input_ids.shape[1]
            )
        pos_cis = self.pos_cis[start_pos: start_pos + input_ids.shape[1]]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h,
                pos_cis,
                past_key_value=  past_key_values[l] if past_key_values else None,
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))
        # 将辅助损失添加到输出中
        aux_loss = sum(
            l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward)
        )
        
        self.OUT.__setitem__("logits",logits)
        self.OUT.__setitem__("aux_loss",aux_loss)
        self.OUT.__setitem__("past_key_values",past_kvs)

        return self.OUT
            
            
