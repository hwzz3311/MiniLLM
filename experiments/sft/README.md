## 全量SFT训练

### 训练loss 曲线

![loss曲线](./sft_loss.png)

### 训练结果

1. 训练了两个epoch，loss并未完整收敛，依然偏大，但是经过测试，模型可以完成基本的对话。


### SFT-Chat 训练 tips

1. 只使用对话数据中的大模型生成内容计算loss，其他内容不计算loss。
