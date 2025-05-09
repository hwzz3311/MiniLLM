# 一个LLM训练复现项目

## 项目简介
本项目旨在复现大模型的训练过程，

⭐️特点：注释和日志极其详细，适合初学者学习。


## 开发日志

2025.05.09 完善moe的练记录。

下一步计划：

1. 让基础的Dense模型更加智能，解决模型推理时重复和过长的问题。
   
2. 对齐R1格式的Reasoning Model。

3. 实现多模态模型，MiniLLM-VL。
4. 

## 环境要求
- Python 3.10+
- PyTorch 2.6+

## 项目结构
[项目结构说明](./project_structure.txt)

## 快速开始
1. 环境配置
   推荐使用uv安装依赖，uv 是python的包管理工具，类似于pip，但是比pip更强大，安装速度更快。
   新建虚拟环境
   ```bash
   uv venv --python 3.10
   ```
   安装依赖
   ```bash
   uv pip install -r requirements.txt
   ```
2. 数据准备
   1. tokenizer数据准备
        1. tips: 理论上训练tokenizer应该使用和预训练一样的数据集，但是实际训练中，由于本人机器内存有限，无法加载训练。因此选择基于minimind[https://github.com/jingyaogong/minimind]中的预训练数据进行修改，去除原数据中的 <s> 和 </s> 标签。

   2. 预训练数据准备
   3. SFT数据准备
3. 训练步骤
4. 评估方法

## 训练细节
- 模型架构
- 训练策略
- 优化方法
