

# Motivation

在强化学习训练过程中，随着模型性能的持续提升，其输出的 response 序列也随之不断变长 —— 尤其在慢思考（slow thinking）模式下，序列长度可达到数十K tokens（即数万 tokens），这导致推理阶段的耗时占比持续上升。
同时，我们对 response 输出长度的分布进行了统计（见下图），发现其存在显著的长尾效应：少数样本虽占比极低，但序列长度远超出平均水平。


![img.png](img.png)



# Method

我们可采用提前截断（early truncation）的方式，解决推理阶段的长尾问题并提升推理性能。具体思路如下图所示：当样本的 response 长度过长时，对其进行提前截断处理，并将截断后的剩余部分纳入下一次推理流程，从而减少无效等待耗时。

![img_1.png](img_1.png)

# Proposed Design



基于上述思路，我们设计并实现了完整的 partial rollout流程。下文在DAPO算法的基础上将该流程拆分为训练流程（Training Workflow）和推理流程（Inference Workflow）进行详细说明。

在训练流程中，我们首先在数据属性中新增 age 和 raw_response_ids 两个字段：age 用于记录数据样本的老化轮次，raw_response_ids 用于存储上一轮推理未完成的response。同时，我们引入 AggregatorActor 组件，其核心职责是汇总各个 DP（数据并行）组上已完成推理的样本；当已完成推理的样本累计数量达到预设阈值时，AggregatorActor 会向所有 worker 节点发送推理完成信号。收到信号后，各 worker 立即终止当前推理流程，并将未完成的response保存至 raw_response_ids 字段，该部分响应将在后续轮次中继续完成推理。


我们先从所有样本中筛选出推理未完成的部分，组成 partial_batch这部分样本的未完成response会拼接到原prompt后，用于后续轮次的续跑推理（即 partial rollout 机制）。
对于已完成推理的样本（记为 staged_out），由于存在1个样本推理n次的机制，因此，我们会统计已完成 staged_out 样本处理的组数：若完成组数超过 train_batchsize 的设定值，则终止当前推理流程，进入训练更新阶段（update）；否则，将新样本与 partial_batch 合并，启动下一轮推理。


![img_2.png](img_2.png)

在推理流程中，首先对待推理 prompt 对应的样本按 age 属性排序：age 值越大的样本排序越靠前，优先送入推理引擎执行推理。推理启动后，遍历推理引擎的输出结果：若样本推理完成，则将其存入 output_list，同时将已完成推理的样本计数累加到 AggregatorActor 组件，并检查累计完成数是否达到预设阈值。若达到阈值，AggregatorActor 将向所有 worker 节点发送推理终止信号，随即终止本次推理流程，最终返回已完成推理的样本与未完成推理的样本。


![img_6.png](img_6.png)




参考了该 PR  [#1826](https://github.com/volcengine/verl/pull/1826/files) 的部分实现逻辑及相关论文https://arxiv.org/pdf/2509.18521。