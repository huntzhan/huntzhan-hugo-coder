---
title: "Understanding The Implementation Of AllenNLP EMLo"
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

[AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md): supports only *the step (3)*, but with a better user experience.

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

### `bilm/training.py`









### Evaluation: Perplexity & Bits Per Character (BPC)

https://stats.stackexchange.com/questions/129352/how-to-find-the-perplexity-of-a-corpus

https://stats.stackexchange.com/questions/211858/how-to-compute-bits-per-character-bpc

https://datascience.stackexchange.com/questions/17514/word-based-perplexity-from-char-rnn-model

https://github.com/allenai/bilm-tf/issues/22