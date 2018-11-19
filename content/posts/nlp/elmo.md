---
title: "Deep Dive The Implementation Of EMLo"
date: 2018-11-12T13:01:19+08:00
draft: true
math: "true"
---

## Introduction

Intended audience: people who want to train ELMo [Peters et al. (2018)][] from scratch and understand the details of official implementation.

Training & using ELMo roughly consists of the following steps:

1. Train a biLM on a *large corpus*. (self-supervised)
2. Fine-tune the pre-trained model on a *target corpus*. (self-supervised)
3. Build & train a *new model on top of the pre-trained ELMo model*.  (supervised)

Related implementations:

*   [bilm-tf](https://github.com/allenai/bilm-tf): The original tensorflow implementation of ELMo, supporting all steps mentioned above.
*   [AllenNLP](https://allennlp.org/elmo): a powerful pytorch based Deep NLP framework, supporting only *the step (3)*, but with a better user experience.

## bilm-tf

The project structure of [bilm-tf](https://github.com/allenai/bilm-tf):

```
├── bilm                  // ELMo is implemented under this folder.
│   ├── __init__.py
│   ├── data.py             // Load & batch sentences.
│   ├── elmo.py             // ``weight_layers`` for step (3).
│   ├── model.py            // ``BidirectionalLanguageModel`` for step (3)
│   └── training.py         // For training & fine-tuning (step (1) and (2)). 
│
├── bin                   // For shell execution.
│   ├── dump_weights.py     // Dump weight file for AllenNLP.
│   ├── restart.py          // For step (2).
│   ├── run_test.py         // Check the perplexity on the heldout dataset.
│   └── train_elmo.py       // For step (1).
│
...
├── usage_cached.py         // usage_* are for step (3).
├── usage_character.py
└── usage_token.py
```

Note that `bilm/training.py` does not depend on `bilm/elmo.py`,  `bilm/model.py` or `usage_*`. You don't need to look into these files if you only want to use *bilm-tf* for training and fine-tuning. Using *bilm-tf* for step (3) will not be covered in this article since I think using *AllenNLP* for step (3) would be much easier.

Hence, only these files are needed to be studied:

```
├── bilm                  // ELMo is implemented under this folder.
│   ├── __init__.py
│   ├── data.py             // Load & batch sentences.
│   └── training.py         // For training & fine-tuning (step (1) and (2)). 
│
├── bin                   // For shell execution.
│   ├── dump_weights.py     // Dump weight file for AllenNLP.
│   ├── restart.py          // For step (2).
│   ├── run_test.py         // Check the perplexity on the heldout dataset.
│   └── train_elmo.py       // For step (1).
```

TODO(In this section, ...)

### Data Loading & Batch Generation

`bilm/data.py` offers the ability to generate unidirectional/bidirectional $\times$ word-level/char-level ids:

- [BidirectionalLMDataset](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/data.py#L435), load data file & generate batch for training.
  - [LMDataset](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/data.py#L315). unidirectioanl (forworad or backward) processing. *BidirectionalLMDataset* simply consists of two *LMDataset* instances for bidirectional processing.

- [UnicodeCharsVocabulary](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/data.py#L98), char-level ids, referenced by *LMDataset* and *BidirectionalLMDataset*.
  - [Vocabulary](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/data.py#L10), word-level ids.

- [_get_batch](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/data.py#L264), generates batches for **the truncated BPTT with 20 steps** during training, following [Jozefowicz et al. (2016)][]. *bilm-tf* provides related explanation [here](https://github.com/allenai/bilm-tf#can-you-provide-some-more-details-about-how-the-model-was-trained). Similar case for testing, but with the size of windows set to 1, as specified [here](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L965).:
    1. *padding sentences with `<S>` and `</S>`* ,
    2. *then packing tokens from one or more sentences into each row to fill completely fill each batch*, in which the batch row is set *with a fixed size window of 20 tokens*. For a sentence longer than 20 words, here's what will happen:
        1. pop the first 20 words from the sentences and fills to **the $n^{\text{th}}$ row** of the $m^{\text{th}}$ batch.
        2. continue to fill **exactly the $n^{\text{th}}$ row** of the $(m+1)^{\text{th}}, (m+2)^{\text{th}}, \ldots$ batches until the sentence is empty.

### Model Definition

[LanguageModel](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L30) defines the neural network structure of ELMo, which mainly consists of *Context-independent token representation* and *$L\text{-layer}$ biLM*.

#### Context-independent token representation

*bilm-tf* supports training both *CNN over characters* and *token embeddings*, which is controlled by the presence of `char_cnn`  in the config, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L338-L341).

[Peters et al. (2018)][] adopts *CNN over characters* representation, following the settings of [Jozefowicz et al. (2016)][] which closely follows the architecture from  [Kim et al. (2016)][]. The implementation is defined in function [_build_word_char_embeddings](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L105-L323):

1.  Character embeddings, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L158-L162). Note that `n_characters` (the size of char-id vocabulary) should be hardcoded to `261` due to the implementation of *UnicodeCharsVocabulary*, see [here](https://github.com/allenai/bilm-tf#whats-the-deal-with-n_characters-and-padding) and [here](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/data.py#L163).
2.  Convolutional filter & max pooling, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L176-L225). The idea is to capture n-gram features.
3.  Highway network, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L261-L300).
4.  Miscellaneous stuffs like fianal projection, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L303-L311) and reshaping, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L313-L318).

Meanwhile, the *token embedding* implementation is defined in function [_build_word_embeddings](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L74-L103). Note that *AllenNLP* only supports  *CNN over characters* representation, hence you need to find another way out if you want to use *token embedding* representation.

#### $L\text{-layer}$ biLM

The multi-layer LSTM is defined [here](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L343-L428), with some common practices like

*   Cell/projection clip, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L356-L357).
*   Skip (residual) connection, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L378-L386).
*   Variational dropout, [ref](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L390-L391).

Note that the `final_lstm_state` is for *the truncated BPTT*.

#### Loss & Evaluation

TODO

Evaluation: Perplexity & Bits Per Character (BPC)

https://stats.stackexchange.com/questions/129352/how-to-find-the-perplexity-of-a-corpus

https://stats.stackexchange.com/questions/211858/how-to-compute-bits-per-character-bpc

https://datascience.stackexchange.com/questions/17514/word-based-perplexity-from-char-rnn-model

https://github.com/allenai/bilm-tf/issues/22

#### Multi-GPU Training

TODO

### CLI Scripts & Config

TODO

By default, ELMo use *BidirectionalLMDataset*  & *UnicodeCharsVocabulary* to generate **bidirectional char-level ids**, as specified in [load_vocab(args.vocab_file, 50)](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bin/train_elmo.py#L12), [load_vocab](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L1058-L1060) and [BidirectionalLMDataset](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bin/train_elmo.py#L58-L59).

## AllenNLP ELMo

[The tutorial of AllenNLP ELMo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).

Classe references:

- [batch_to_ids](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L228), converts word to char ids in preprocessing.
    - [ELMoTokenCharactersIndexer(TokenIndexer)](https://github.com/allenai/allennlp/blob/7df8275e7f70013185f1afeaa2779c2abce4492d/allennlp/data/token_indexers/elmo_indexer.py#L78), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L243), a *TokenIndexer* (type: `elmo_characters`).
        - [ELMoCharacterMapper](https://github.com/allenai/allennlp/blob/7df8275e7f70013185f1afeaa2779c2abce4492d/allennlp/data/token_indexers/elmo_indexer.py#L25), the actual char id mapper.

- [Elmo(torch.nn.Module)](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L34), high-level class for combining the pre-trained & frozen biLM and trainable softmax-normalized weights $s^{task}$ and scalar parameter $r^{task}$. Look into [ElmoTokenEmbedder(TokenEmbedder)](https://github.com/allenai/allennlp/blob/15e36458b1f6a7452492bd107cd631fede3617a8/allennlp/modules/token_embedders/elmo_token_embedder.py#L12), a *TokenEmbedder* (type: `elmo_token_embedder`), and [this example](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#using-elmo-with-existing-allennlp-models) if you want to add ELMo to your AllenNLP config file directly.
    - [_ElmoBiLm(torch.nn.Module)](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L492), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L105-L108), handles word-level representation and multi-layers BiLSTM.
        - [_ElmoCharacterEncoder(torch.nn.Module)](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L257), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L522), generate word-level representation by applying CNN over characters. (requries `options_file` and `weight_file`)
        - [ElmoLstm(_EncoderBase)](https://github.com/allenai/allennlp/blob/aa1b774ed8de31ec04bebf9f054200bc2507e0c5/allennlp/modules/elmo_lstm.py#L20), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L545-L552), represents the multi-layers BiLSTM. (requries `options_file` and `weight_file`)
    - [ScalarMix(torch.nn.Module)](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/scalar_mix.py#L8), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L112-L120), handles the *task specific combination of the intermediate layer representations in the biLM*  (four trainable parameters $s^{task}$ and $r^{task}$). `num_output_representations` controls the number of combinations to generate.

TODO(what if we don't use char_cnn? AFAIK, AllenNLP doesn't provide impl for that.)


[Jozefowicz et al. (2016)]: https://arxiv.org/pdf/1602.02410.pdf	"Exploring the Limits of Language Modeling"
[Peters et al. (2018)]: https://arxiv.org/abs/1802.05365	"Deep contextualized word representations"
[Kim et al. (2016)]: https://arxiv.org/abs/1508.06615	"Character-Aware Neural Language Models"
