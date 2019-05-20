---
title: "BERT 系列：ERNIE 中文训练数据预处理"
date: 2019-05-20T11:36:04+08:00
math: "true"
---



## 简介

本文简要解析 ERNIE [^1] 的数据预处理代码 [^2]。本文假设读者已经阅读过 BERT 与 ERNIE 的论文，以及 [BERT 系列：数据预处理源码解析]({{< relref path="bert-pretrain-data.md" >}})。



## ERNIE Masking Strategy

ERNIE 针对无切分（unsegmented）的语言（如中文），提出了一种 MLM 的数据增强（data augmentation）策略：

>   We take a phrase or a entity as one unit, which is usually composed of several words. **All of the words in the same unit are masked during word representation training, instead of only one word or character being masked**. In this way, the prior knowledge of phrases and entities are implicitly learned during the training procedure.

简单概括，ERNIE 会**利用分词的信息**，在词的粒度（phrase-level）做 masking，通过这种策略可以使模型隐式地学习到词粒度的知识。与之相对的，BERT 的 basic-level masking 策略是基于基础语言单元粒度的（basic language unit，在中文里对应输入的每一个字）：

>   It treat a sentence as a sequence of **basic language unit**, for **English**, the basic language unit is **word**, and for **Chinese**, the basic language unit is **Chinese Character**.
>
>   ...
>
>   **Phrase** is a small group of words or characters together acting as a conceptual unit … and use some language dependent **segmentation tools** to get the word/phrase information in other language such as **Chinese**.



## 源码解析

ERNIE 与 BERT 在模型结构上没有明显的差别。通过简单的转换逻辑，可以直接把 ERNIE 发布的预训练模型（PaddlePaddle 实现） [^3] 转换为 huggingface/pytorch-pretrained-BERT （Pytorch 实现） [^4] 接受的格式。

主要的差别在 MLM 的数据预处理逻辑部分：

*   BERT 的数据预处理的输入是若干个 txt 文本，输入是带 NSP、MLM 标注的序列数据集，详见 [BERT 系列：数据预处理源码解析]({{< relref path="bert-pretrain-data.md" >}}) 一文。
*   ERNIE 的数据预处理分为两个阶段：
    *   训练前的数据准备：构建序列与标记分词边界信息。
    *   训练过程中的数据准备：在每个 epoch，基于分词边界信息动态构建 MLM 标注。



### 构建序列与标记分词边界信息

这一步的输出是一个 txt 文件：

>   每个样本由5个 '`;`' 分隔的字段组成，数据格式: `token_ids; sentence_type_ids; position_ids; seg_labels; next_sentence_label`；其中 `seg_labels` 表示分词边界信息: 0表示词首、1表示非词首、-1为占位符, 其对应的词为 `CLS`或者 `SEP`；

ERNIE 没有开源这一步的代码。个人建议，可以复用 BERT 的预处理代码，然后利用百度的词法分析 API 加入分词边界信息。



### 动态构建 MLM 标注

论文的 *3.2 Knowledge Integration* 小节里提到：

>   Instead of adding the knowledge embedding directly, we proposed a **multi-stage knowledge masking strategy** to integrate phrase and entity level knowledge into the Language representation.

通过阅读源码，其实这个所谓的 *multi-stage knowledge masking strategy* 的逻辑是：在每个 epoch，以某个概率（这个值没给）**随机**选择 `mask_word` 或者 `mask_char` 模式 [^6] 。`mask_word` 模式后续会走 phrase-level masking 逻辑，会利用分词边界信息；`mask_char` 模式后续会走 basic-level masking 的逻辑，不会利用分词边界信息。

论文的 *3.2.2 Phrase-Level Masking* 里提及：

>   In phrase-level mask stage, we also use basic language units as training input, unlike random basic units mask, this time we **randomly select a few phrases in the sentence, mask and predict all the basic units in the same phrase**.

Phrase-level masking 的实现见 [^5]：

1.  与 BERT 相对的，ERNIE 这一步是在每个 epoch 动态生成的，且Masking 的数目没有上限（没有 `max_predictions_per_seq` 的限制）。
2.  基于分词信息，**以词为单位**决定是否触发 masking 逻辑。
3.  对于每一需要 mask 的**字符串（一个词），遍历其中每个输入（字符）**：
    1.  80% 的概率，把输入替换为 `[MASK]`。
    2.  10% 的概率，把输入替换为随机的 token。
    3.  10% 的概率，维持输入不变。



## 总结

本文简要解析了 ERNIE 利用中文分析信息的 masking 逻辑。



[^1]: ERNIE: Enhanced Representation through Knowledge Integration Yu. <https://arxiv.org/abs/1904.09223>
[^2]: https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE
[^3]: https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz
[^4]: https://github.com/huggingface/pytorch-pretrained-BERT
[^5]: https://github.com/PaddlePaddle/LARK/blob/9b55f0b9bc6dbd0320207cf466f5461dd108405b/ERNIE/batching.py#L49-L86
[^6]: https://github.com/PaddlePaddle/LARK/blob/9b55f0b9bc6dbd0320207cf466f5461dd108405b/ERNIE/reader/pretraining.py#L251-L267

