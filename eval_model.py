import argparse
import os
from collections import defaultdict

import torch
import random
import numpy as np
from src.model.model import MiniLLM, MiniLLMConfig
from src.model.model_vl import MiniLLM_VL, MiniLLM_VLConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import json
base_dir = os.path.dirname(os.path.abspath(__file__))


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.load == 0:
        model = MiniLLM(MiniLLMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe,
            vocab_size=len(tokenizer)
        ))
        state_dict = torch.load(args.model_path, map_location=args.device)
        # 删除所有以 'mask' 开头的键，是为了避免加载lora权重时，出现key不匹配的问题
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)
        if args.lora_name != 'None':
            # TODO 加载lora
            pass
    else:
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def init_vl_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.load == 0:
        model = MiniLLM_VL(MiniLLM_VLConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe,
            vocab_size=len(tokenizer)
        ))
        # 
        state_dict = torch.load(args.model_path, map_location=args.device)
        print(f"success load model : {args.model_path}")
        # 删除所有以 'mask' 开头的键，是为了避免加载lora权重时，出现key不匹配的问题
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)
        if args.lora_name != 'None':
            # TODO 加载lora
            pass
    else:
        transformers_model_path = './MiniLLM-VL'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    print(f'MiniMind-VL模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    vision_model, processor = MiniLLM_VL.load_vision_model(model.config.clip_model_path)
    vl_llm = model.eval().to(args.device)
    return vl_llm, tokenizer, vision_model, processor

# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '花园水库是',
            '幻影神奇宝贝的主谋者',
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解“大语言模型”这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas

def get_prompt_datas_for_vl(args, preprocess_fn, model:MiniLLM_VL):
    # 单图片的识别
    single_image_dir = os.path.join(base_dir, "./assets/eval_images")
    single_image_files = os.listdir(single_image_dir)
    single_image_files = [os.path.join(single_image_dir, file) for file in single_image_files]
    # 多图片的识别
    multi_image_dir = os.path.join(base_dir, "./assets/eval_multi_images")
    multi_image_files = {}
    for sub_dir in os.listdir(multi_image_dir):
        sub_dir_path = os.path.join(multi_image_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            multi_image_files[sub_dir] = []
            for file in os.listdir(sub_dir_path):
                multi_image_files[sub_dir].append(os.path.join(sub_dir_path, file))
    # 分别为 单个图片和多个图片构建不同的prompt
    prompt_datas = []
    for single_image_file in single_image_files:
        prompt = f"{model.config.image_special_token}\n描述一下这个图像的内容。"
        image = Image.open(single_image_file).convert("RGB")
        pixel_tensors = MiniLLM_VL.image2tensor(image, preprocess_fn).to(args.device).unsqueeze(0)

        prompt_datas.append({
            "image_path": single_image_file,
            "prompt": prompt,
            "pixel_tensors": pixel_tensors
        })
    for sub_dir, multi_image_file_list in multi_image_files.items():
        prompt = (f"{model.config.image_special_token}\n"
                  f"{model.config.image_special_token}\n"
                  f"比较一下两张图像的异同点。")
        pixel_tensors_multi = []
        for multi_image_file in multi_image_file_list:
            image = Image.open(multi_image_file).convert("RGB")
            pixel_tensors = MiniLLM_VL.image2tensor(image, preprocess_fn).to(args.device).unsqueeze(0)
            pixel_tensors_multi.append(pixel_tensors)
        # 将多张图片的像素张量拼接起来
        pixel_tensors = torch.cat(pixel_tensors_multi, dim=0).to(args.device).unsqueeze(0)

        prompt_datas.append({
            "image_path": json.dumps(multi_image_file_list,ensure_ascii=False,indent=4),
            "prompt": prompt,
            "pixel_tensors": pixel_tensors
        })
    prompt_datas = []
    img_path = "/mnt/d/pretrain/minimind-v_dataset/sft_images/train-00058-of-00059_image_2671_0.jpg"
    prompt = f"{model.config.image_special_token}\n图片中的直升机是什么颜色？"
    image = Image.open(img_path).convert("RGB")
    pixel_tensors = MiniLLM_VL.image2tensor(image, preprocess_fn).to(args.device).unsqueeze(0)
    prompt_datas.append({
        "image_path": img_path,
        "prompt": prompt,
        "pixel_tensors": pixel_tensors
    })

    return prompt_datas

                

def chat_with_model(model, tokenizer, args, prompt, eos_token_id, pixel_tensors=None):
    answer = prompt
    with torch.no_grad():
        # 将prompt转换为token
        x = torch.tensor(tokenizer(prompt)["input_ids"], device=args.device,).unsqueeze(0)
        # x 的形状为 [1, seq_len]，既【batch_size, seq_len】
        outputs = model.generate(
            x,
            eos_token_id=eos_token_id,
            max_new_tokens=args.max_seq_len,
            temperature=args.temperature,
            stream=args.stream,
            pad_token_id=tokenizer.pad_token_id,
            **({"pixel_tensors":pixel_tensors} if isinstance(model, MiniLLM_VL) else {})
        )
        print('🤖️: ', end='')
        try:
            if not args.stream:
                # 非流式模式下，直接打印出答案
                answer = tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True)
                print(answer, end='')
            else:
                # 流式模式下，逐字符打印出答案
                history_idx = 0
                for y in outputs:
                    # y 的形状为 [1, seq_len]，既【batch_size, seq_len】
                    # 因此需要使用 y[0] 来获取第一个样本的生成结果，既【seq_len】
                    answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                    # if (answer and answer[-1] == '�') or not answer:
                    # continue
                    print(answer[history_idx:], end='', flush=True)
                    history_idx = len(answer)

        except StopIteration:
            print("No answer")
        print("\n")
    return answer

def main():
    # 初始化参数 
    this_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32  # cuda 使用 bfloat16 精度，mps 使用 float32 精度

    minillm_tokenizer_path = os.path.join(this_dir,"./assets/minillm_tokenizer")
    qwen_tokenizer_path = os.path.join(this_dir,"./assets/qwen_tokenizer")
    minimind_tokenizer_path = os.path.join(this_dir,"./assets/minimind_tokenizer")
    tokenizer_path = minillm_tokenizer_path

    default_data_path = os.path.join(this_dir, "./assets/data_sample/wikipedia_zh_sample_data.json")
    # model_output_dir = os.path.join(this_dir,"./minillm_output")
    use_moe = False
    sft = False
    pretrain = True
    model_type = "llm-vl"
    model_output_dir = os.path.join(this_dir,f"./assets/mini{model_type}_output/")
    if use_moe:
        model_output_dir = os.path.join(model_output_dir,"moe")
    if sft:
        model_output_dir = os.path.join(model_output_dir,"sft")
    if pretrain:
        model_output_dir = os.path.join(model_output_dir, "pretrain")

    model_dir = os.path.join(model_output_dir,"dim_512/n_layers_8")
    # model_dir = model_output_dir
    # 获取模型目录下所有文件,安装创建时间进行倒序，
    model_files = os.listdir(model_dir)
    model_files.sort(key=lambda x: os.path.getctime(os.path.join(model_dir, x)), reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    print(f"使用模型: {model_path}")
    parser = argparse.ArgumentParser(description="eval mini-llm model")
    parser.add_argument("--model_type", type=str, default=model_type, help="Support llm, llm-vl")
    parser.add_argument("--lora_name", type=str, default="None", help="The lora name")
    parser.add_argument("--model_path", type=str, default=model_path, help="The model checkpoint path")
    parser.add_argument("--tokenizer_path", type=str, default=tokenizer_path, help="The tokenizer path")
    parser.add_argument("--device", type=str, default=device, help="The device")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="The top p")

    parser.add_argument("--dim", type=int, default=512, help="The dimension of the model")
    parser.add_argument("--n_layers", type=int, default=8, help="The number of layers of the model")
    parser.add_argument("--max_seq_len", type=int, default=8192, help="The max sequence length of the model")
    parser.add_argument("--use_moe", type=bool, default=use_moe, help="Whether to use moe")

    # 携带历史对话上下文条数
    # history_cnt需要设为偶数，即【用户问题, 模型回答】为1组；设置为0时，即当前query不携带历史上文
    # 模型未经过外推微调时，在更长的上下文的chat_template时难免出现性能的明显退化，因此需要注意此处设置
    parser.add_argument("--history_cnt", type=int, default=0, help="The number of history context")
    parser.add_argument("--stream", type=bool, default=True, help="Whether to stream")
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=0, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型，4: RLAIF-Chat模型")

    args = parser.parse_args()
    if args.model_type == "llm":
        model, tokenizer = init_model(args)
        prompt_datas = get_prompt_datas(args)
    elif args.model_type == "llm-vl":
        model, tokenizer, vision_model, processor = init_vl_model(args)
        model:MiniLLM_VL
        prompt_datas = get_prompt_datas_for_vl(args, processor, model)
    print(f"success init model for {args.model_type}")

    
    test_model = int(input("[0] 自动测试\n[1] 手动测试\n"))
    messages = []
    eos_token_id = tokenizer.eos_token_id
    if args.model_type == "llm":
        if args.model_mode == 1: # SFT-Chat模型
            eos_token_id = tokenizer("<|im_end|>").input_ids[0]
            print(f"SFT-Chat模型 eos_token_id: {eos_token_id}")
        for idx, prompt in enumerate(prompt_datas if test_model == 0 else iter(lambda: input('👶: '), '')):
            setup_seed(random.randint(0, 2048))  # 每次随机种子
            if test_model == 0:  # 自动测试模式下，需要打印出自动测试的prompt
                print(f'👶: {prompt}')
            messages = messages[-args.history_cnt:] if args.history_cnt else []
            messages.append({'role': 'user', 'content': prompt})
            new_prompt = tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
            # 预训练模型 下仅使用 bos_token + prompt，其他模型则使用 sft_chat_template
            new_prompt = new_prompt[-args.max_seq_len - 1:] if args.model_mode != 0 else (
                tokenizer.bos_token + prompt if tokenizer.bos_token else prompt)
            answer = chat_with_model(model, tokenizer, args, new_prompt, eos_token_id)
            messages.append({"role": "assistant", "content": answer})
    elif args.model_type == "llm-vl":
        # if args.model_mode == 1: # SFT-Chat模型
        eos_token_id = tokenizer("<|im_end|>").input_ids[0]
        print(f"VL 模型 eos_token_id: {eos_token_id}")
        # 定义一个自定义输入的函数，先接收一个图片路径，然后接收一个用户问题
        def custom_input():
            while True:
                # 提示输入一个图片路径
                image_path = input('👶: 请输入一个图片路径: ')
                # 提示输入一个用户问题
                user_question = input('👶: 请输入指令: ')
                image = Image.open(image_path).convert("RGB")
                pixel_tensors = MiniLLM_VL.image2tensor(image, processor).to(args.device).unsqueeze(0)
                prompt = f"{model.config.image_special_token}\n{user_question}"
                yield {
                    "image_path": image_path,
                    "prompt": prompt,
                    "pixel_tensors": pixel_tensors
                }
        for idx, prompt_dict in enumerate(prompt_datas if test_model == 0 else iter(lambda: custom_input)):
            setup_seed(random.randint(0, 2048))  # 每次随机种子
            prompt = prompt_dict["prompt"]
            image_path = prompt_dict["image_path"]
            pixel_tensors = prompt_dict["pixel_tensors"]

            prin_prompt = prompt.replace(model.config.image_special_token,"")
            prin_prompt = prin_prompt.strip()
            print(f'👶: {prin_prompt}')
            print(f"[image_path] : {image_path}")
            messages = messages[-args.history_cnt:] if args.history_cnt else []
            messages.append({'role': 'user', 'content': prompt})
            new_prompt = tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
            # vl 模型下均使用 sft格式
            new_prompt = new_prompt[-args.max_seq_len - 1:]
            answer = chat_with_model(model, tokenizer, args, new_prompt, eos_token_id, pixel_tensors)
            messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

