---
title: "BERT 系列：数据预处理源码解析"
date: 2019-05-16T16:31:27+08:00
math: "true"
---



## 简介

本文简要解析 BERT [^1]  预训练（Pre-training）的数据预处理源码 [^2]。本文假设读者已经阅读过 BERT 论文以及 [BERT 系列：辅助任务解析]({{< relref path="bert-auxiliary-tasks.md" >}})。



## 预处理流程

官方实现 [^2] 的 *Pre-training with BERT* 小节简述了预处理的执行方式。通过阅读源码，可以将预处理逻辑划分为以下阶段：

1.  数据准备

2.  读取句子与 Tokenization
3.  构建序列
4.  Masking
5.  构建 Instance 与导出



### 数据准备

1.  准备一个或多个 txt 文本，满足以下格式（示例 [^3] ）：
    1.  每一行，如果非空，存储一个句子。每个句子使用 whitespace 切词。
    2.  使用空行表示文档结尾。空行的目的是禁止生成跨文档的序列。

2.  准备 WordPiece 词典 [^4]。通常情况下，可以直接复用预训练模型的词典。



### 读取句子与 Tokenization

实现见 `create_training_instances` 函数 [^5] 。

逻辑：

1.  读取一个文档的所有句子，whitespace 切分后再基于词典做 WordPiece 切分 [^6]。
2.  基于 (1) 构建嵌套 list  `all_documents`，结构是 `[doc_num, sent_num, token_num]`。
3.  Shuffle `all_documents`。



### 构建序列

"序列（Sequence）"是 BERT 论文定义的概念，包含一个句子对（sentence pair）。基于论文描述与代码逻辑，我认为使用“句子集合对"描述更加合适：

>   To generate each training input sequence, we sample two spans of text from the corpus, **which we refer to as “sentences” even though they are typically much longer than single sentences (but can be shorter also)**. The first sentence receives the A embedding and the second receives the B embedding. 50% of the time B is the actual next sentence that follows A and 50% of the time it is a random sentence, which is done for the “next sentence prediction” task. They are sampled such that **the combined length is ≤ 512 tokens**.

为了防止误解，后续我将使用 "**句子集合 A**" 与 "**句子集合 B**" 描述序列中这两个区域。

逻辑：

1.  `create_instances_from_document` 函数 [^7] 包含了构建序列的所有逻辑。序列的构建是以文档为单位的，脚本仅会对每篇文档执行一次此函数，从中生成若干序列。
2.  序列长度 `target_seq_length` 由 `max_seq_length [default: 128]` 与 `short_seq_prob [default: 0.1]` 决定：
    1.  设 `max_num_tokens = max_seq_length - 3` （考虑特殊 token  `[CLS]`, 2 * `[SEP]`）。
    2.  以 `1 - short_seq_prob` 的概率，将序列长度设为 `max_num_tokens`。
    3.  以 `short_seq_prob` 的概率，随机选取 `[2, max_num_tokens]`  为序列长度，目的是降低预训练与 fine-tuning 阶段序列长度不一致的问题 [^8] 。
    4.  需要注意：一是这个序列长度的生效区域是单个文档；二是为了加速收敛 [^9]，训练初期可能会选取较小的序列长度（如 `128`），训练后期再选取较大的序列长度（如 `512`），这种场景需要生成多批训练数据。
3.  基于（2）的序列长度，生成序列：
    1.  维护一个全局 index `i` 标识当前文档中尚未处理的句子 [^21]。
    2.  从 `i` 开始收集若干句子加入 `current_chunk` ，直到 tokens 的数目大于等于序列长度，或已收集到最后一个句子。
    3.  随机选取 `current_chunk` 的前 `[1, len(current_chunk) - 1]` 个句子作为**句子集合 A** [^12]。 
    4.  选取 **句子集合 B** [^11]：
        1.  以 50% 的概率，选取**非连续**（non-next sentence）的句子集合 B。选取方式：随机选择一个除当前文档以外的文档，然后从中选取若干连续的句子，使 A 与 B 的长度恰好超过 `target_seq_length`。
        2.  以 50% 的概率，选取**连续**（non-next sentence）的句子集合 B。选取方式：直接选取 `current_chunk` 的剩余部分。
        3.  `is_random_next`  标记连续、非连续的随机结果，用于训练 NSP。
        4.  特殊情况：`current_chunk` 仅包含一个句子，这种情况强制选取非连续句子集合。
    5.  使用 `truncate_seq_pair` [^10] 裁剪句子集合 A 与 B，保证最终序列长度小于等于 `max_num_tokens`。
    6.  合并句子集合 A 与 B 构建序列 `[CLS] A [SEP] B [SEP]` ，同时生成区间标识（segment id）`0, 0..., 0, 1..., 1` [^13]。



### Masking

在构建完序列之后，我们需要随机 mask 输入用于训练 MLM。

逻辑：

1.  `create_masked_lm_predictions` 函数 [^14] 包含了 Masking 的所有逻辑。Masking 是以序列为单位的，每个序列只会被 masked 一次。
2.  Masking 的数目 `num_to_predict` 由 `masked_lm_prob [default: 0.15]`  与 `max_predictions_per_seq [default: 20]` 决定 [^20]，即随机选取 `masked_lm_prob` 占比的输入 mask，如果输入超过 `max_predictions_per_seq` 则只 mask `max_predictions_per_seq` 个 tokens。
3.  对于每个需要 mask 的输入：
    1.  80% 的概率，把输入替换为 `[MASK]` [^16] 。
    2.  10% 的概率，把输入替换为随机的 token [^17]。
    3.  10% 的概率，维持输入不变 [^18] 。



### 构建 Instance 与导出

最后将上述处理结果打包，为每一个序列生成一个 Instance [^19]，并导出到文件 [^20] 。



## 总结

本文简要解析了 BERT 官方实现的预处理源码。对于中文场景，可以阅读 [BERT 系列：ERNIE 数据预处理源码解析 TODO]({{}}) 一文了解基于分词信息的预处理流程。



[^1]: Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805*(2018).
[^2]: https://github.com/google-research/bert
[^3]: https://github.com/google-research/bert/blob/master/sample_text.txt
[^4]: https://github.com/google-research/bert#learning-a-new-wordpiece-vocabulary
[^5]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L179-L200
[^6]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/tokenization.py#L172-L173
[^7]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L219
[^8]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L228-L237
[^9]: https://github.com/google-research/bert#pre-training-tips-and-caveats
[^10]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L391
[^11]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L264-L294
[^12]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L256-L262
[^13]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L300-L315
[^14]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L338
[^15]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L352-L353
[^16]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L367
[^17]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L374
[^18]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L371
[^19]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L320-L326
[^20]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L92-L162
[^21]: https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/create_pretraining_data.py#L247

