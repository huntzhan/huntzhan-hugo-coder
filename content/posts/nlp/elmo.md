---
title: "Understanding The Implementation Of EMLo"
date: 2018-11-12T13:01:19+08:00
draft: true
---

## Introduction

TODO(what is ELMo)

Training & using ELMo roughly consists of the following steps:

1. Train a biLM on a *large corpus*. (self-supervised)
2. Fine-tune the pre-trained model on a *target corpus*. (self-supervised)
3. Build & train a *new model on top of the pre-trained ELMo model*.  (supervised)

[bilm-tf](https://github.com/allenai/bilm-tf): The original tensorflow implementation of ELMo, supporting all steps mentioned above.

[AllenNLP](https://allennlp.org/elmo): supports only *the step (3)*, but with a better user experience.

## bilm-tf

The project structure of [bilm-tf](https://github.com/allenai/bilm-tf):

```
├── bilm                  // ELMo is implemented under this folder.
│   ├── __init__.py
│   ├── data.py             // Load & batch sentences.
│   ├── elmo.py             // ``weight_layers`` for step (3).
│   ├── model.py            // ``BidirectionalLanguageModel`` for step (3)
│   └── training.py         // For training & fine-tuning (step (1) and (2)). 
├── bin                   // For shell execution.
│   ├── dump_weights.py     // Dump weight file for AllenNLP.
│   ├── restart.py          // For step (2).
│   ├── run_test.py         // Check the perplexity on the heldout dataset.
│   └── train_elmo.py       // For step (1).
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
├── bin                   // For shell execution.
│   ├── dump_weights.py     // Dump weight file for AllenNLP.
│   ├── restart.py          // For step (2).
│   ├── run_test.py         // Check the perplexity on the heldout dataset.
│   └── train_elmo.py       // For step (1).
```

TODO(In this section, ...)

### `bilm/data.py`

`BidirectionalLMDataset`, `LMDataset`: Load data file & generate batch for training.

`UnicodeCharsVocabulary`,  `Vocabulary`: Generate word-level & char-level ids.

By default, ELMo use `BidirectionalLMDataset` (bidirectional) & `UnicodeCharsVocabulary` (char-level ids) for training, as specified in

- [load_vocab(args.vocab_file, 50)](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bin/train_elmo.py#L12), [load_vocab](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/training.py#L1058-L1060).
- [BidirectionalLMDataset](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bin/train_elmo.py#L58-L59)



### `bilm/training.py`



## AllenNLP ELMo

[The tutorial of AllenNLP ELMo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md).

Classe relationships:

- [batch_to_ids](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L228), converts word to char ids in preprocessing.
  - [ELMoTokenCharactersIndexer(TokenIndexer\[List\[int\]])](https://github.com/allenai/allennlp/blob/7df8275e7f70013185f1afeaa2779c2abce4492d/allennlp/data/token_indexers/elmo_indexer.py#L78), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L243), a *TokenIndexer* (type: `elmo_characters`). Look into [ElmoTokenEmbedder(TokenEmbedder)](https://github.com/allenai/allennlp/blob/15e36458b1f6a7452492bd107cd631fede3617a8/allennlp/modules/token_embedders/elmo_token_embedder.py#L12), a *TokenEmbedder* (type: `elmo_token_embedder`) and [this example](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#using-elmo-with-existing-allennlp-models) if you want to inject ELMo to your AllenNLP config file directly.
    - [ELMoCharacterMapper](https://github.com/allenai/allennlp/blob/7df8275e7f70013185f1afeaa2779c2abce4492d/allennlp/data/token_indexers/elmo_indexer.py#L25), the actual char id mapper.

- [Elmo(torch.nn.Module)](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L34), high-level class for combining the pre-trained & frozen biLM and trainable softmax-normalized weights $s^{task}$ and scalar parameter $r^{task}$.
  - [_ElmoBiLm(torch.nn.Module)](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L492), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L105-L108), handles word-level representation and multi-layers BiLSTM.
    - [_ElmoCharacterEncoder(torch.nn.Module)](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L257), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L522), generate word-level representation by applying CNN over characters. (requries `options_file` and `weight_file`)
    - [ElmoLstm(_EncoderBase)](https://github.com/allenai/allennlp/blob/aa1b774ed8de31ec04bebf9f054200bc2507e0c5/allennlp/modules/elmo_lstm.py#L20), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L545-L552), represents the multi-layers BiLSTM. (requries `options_file` and `weight_file`)
  - [ScalarMix(torch.nn.Module)](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/scalar_mix.py#L8), [ref](https://github.com/allenai/allennlp/blob/43243acf4e91ba471923624bd48c9c9ec72332bf/allennlp/modules/elmo.py#L112-L120), represents the trainable weights of layers. `num_output_representations` controls the number of weight combinations.



TODO(what if we don't use char_cnn? AFAIK, AllenNLP doesn't provide impl for that.)



### Evaluation: Perplexity & Bits Per Character (BPC)

https://stats.stackexchange.com/questions/129352/how-to-find-the-perplexity-of-a-corpus

https://stats.stackexchange.com/questions/211858/how-to-compute-bits-per-character-bpc

https://datascience.stackexchange.com/questions/17514/word-based-perplexity-from-char-rnn-model

https://github.com/allenai/bilm-tf/issues/22