---
title: "BERT 系列：辅助任务解析"
date: 2019-05-16T11:36:04+08:00
math: "true"
---

## 简介

本文简要解析 BERT [^1]  的预训练辅助任务（Auxiliary Task）。



## 预训练辅助任务

BERT 通过两个辅助任务训练语言模型：Masked LM（MLM）与 Next Sentence Prediction（NSP）。

*   MLM：随机 mask 15% 的输入（token），模型需要通过 context 信息还原被 masked 的输入。
*   NSP：随机生成句子对，模型需要判断句子对是否连续（next sentence）。

在训练过程中，MLM 与 NSP 的 loss 是同时计算的，属于多任务学习（MLT）。



## MLM

在 BERT 之前，LM 通常是**单向**的，常见做法是分别训练正向与反向的 LM，然后再做一个 ensemble 得到上下文相关表征（context dependent representation）。这种做法会有信息缺失与标注偏差的问题 [^4] 。MLM 的意义在于，可以使 BERT 作为**单模型**学习到上下文相关的表征，并能更充分地利用**双向**的信息。

论文里强调了设计 MLM 任务需要注意的问题：

>   The first is that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token is never seen during fine-tuning

为了解决这个问题，MLM 采取以下策略 mask 15% 的输入（token）：

*   80% 的概率，把输入替换为 `[MASK]`。
*   10% 的概率，把输入替换为随机的 token。
*   10% 的概率，维持输入不变。

这篇博文 [^2] 提供了一个解释：

*   如果把 100% 的输入替换为 `[MASK]`：模型会偏向为 `[MASK]` 输入建模，而不会学习到 non-masked 输入的表征。
*   如果把 90% 的输入替换为 `[MASK]`、10% 的输入替换为随机 token：模型会偏向认为 non-masked 输入是错的。
*   如果把 90% 的输入替换为 `[MASK]`、维持 10% 的输入不变：模型会偏向直接复制 non-masked 输入的上下文无关表征。
*   所以，为了使模型可以学习到相对有效的上下文相关表征，需要以 1:1 的比例使用两种策略处理 non-masked 输入。论文提及，随机替换的输入只占整体的 1.5%，似乎不会对最终效果有影响（模型有足够的容错余量）。



## NSP

句子级别表征（sentence-level representation）对于某些下游任务是很有用的。NSP 使 BERT 可以从大规模语料中学习句子级别表征、句子关系的知识。NSP 的做法是：

1.  从语料中提取两个句子 `A` 与 `B` ，50% 的概率 `B` 是 `A` 的下一个句子，50% 的概率 `B` 是一个随机选取的句子，以此为标注训练分类器。
2.  将 `A` 与 `B`  打包成一个序列（sequence）：`[CLS] A [SEP] B [SEP]` 。
3.  生成区间标识（segment labels），标识序列中  `A` 与 `B` 的位置。`[CLS] A [SEP]` 的区域设为 `0`，`B [SEP]` 的区域设为 `1`：`0, 0..., 0, 1..., 1`。
4.  将序列与区间标识输入到模型，取 `[CLS]` 的表征训练 NSP 分类器。



## 总结

更多细节见 数据预处理源码解析 TODO 一文。



[^1]: Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805*(2018).
[^2]: BERT Explained: State of the art language model for NLP.  https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
[^4]: Discussion on Reddit.  https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/

