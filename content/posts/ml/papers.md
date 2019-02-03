+++
date = 2019-01-22T18:02:01+08:00
title = "Papers Reading Roadmap"
math = "true"

+++

## Introduction

This page tracks my  reading roadmap of deep learning papers. I will update this page occasionally (probably every 3 - 5 days) according to my progress.

Notation:

*   ✔️: Done.
*   ❤️: Impressive!
*   🤔: Probably this work is controversial.
*   🏷️: Added to todo list but haven't started yet.
*   🚧: Ongoing.

## Deep Learning Theory And Practice

*   General
    *   Highway connection that is inspired by LSTM forget gate to ease the training of very deep networks. [^12] ✔️ [^13] 
    *   ResNet (deep residual nets). [^32] 
    *   Interpretations of highway and residual networks. [^30] 
    *   Transfer Learning / Multi-Task Learning
        *   MTL survey. [^35] 🏷️
        *   MT-DNN. [^36] 🏷️
        *   Gradually adding new capabilities to a system. [^34] 🏷️
*   LSTM
    *       The original formulation. [^1] 
    *       BPTT. [^21] [^20] [^22] ✔️
    *       Forget gate. [^2] 
    *       Peephole connections. [^4] 
    *       The vanilla LSTM achieved by incorporating [^2] [^4] and full BPTT into [^1], which is the most popular LSTM architecture nowadays. [^5] 🚧
    *       Gradient norm clipping. [^6] 
    *       DropConnect dropout. [^7] ✔️
    *       Variational dropout. [^8] 
    *       Recurrent projection. [^10] 
    *       Residual connections. [^9] 
    *       GRU. [^11] 
    *       An empirical study on the LSTM architecture. This paper shows that none of the variants is significantly better than the vanilla LSTM. [^3] 🚧❤️
    *       RHN. [^31]
*   Transformer
    *   The first paper and the annotated version. [^16] [^17] 🏷️
    *   Universal transformer. [^19] 
    *   Transformer-XL. [^18] 
*   CNN
    *   TCN. [^15] ✔️
*   Optimizer
    *   AMSGRAD. [^14] 🚧



[^1]: Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." *Neural computation* 9.8 (1997): 1735-1780.APA
[^2]: Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to forget: Continual prediction with LSTM." (1999): 850-855.
[^3]: Greff, Klaus, et al. "LSTM: A search space odyssey." *IEEE transactions on neural networks and learning systems* 28.10 (2017): 2222-2232.
[^4]:Gers, Felix A., and Jürgen Schmidhuber. "Recurrent nets that time and count." *Proceedings of the IEEE-INNS-ENNS International Joint Conference on Neural Networks. IJCNN 2000. Neural Computing: New Challenges and Perspectives for the New Millennium*. Vol. 3. IEEE, 2000.
[^5]: Graves, Alex, and Jürgen Schmidhuber. "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." *Neural Networks* 18.5-6 (2005): 602-610.

[^6]:Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty of training recurrent neural networks." *International Conference on Machine Learning*. 2013.
[^7]: Wan, Li, et al. "Regularization of neural networks using dropconnect." *International Conference on Machine Learning*. 2013.
[^8]: Kingma, Durk P., Tim Salimans, and Max Welling. "Variational dropout and the local reparameterization trick." *Advances in Neural Information Processing Systems*. 2015.
[^9]: Kim, Jaeyoung, Mostafa El-Khamy, and Jungwon Lee. "Residual LSTM: Design of a deep recurrent architecture for distant speech recognition." *arXiv preprint arXiv:1701.03360* (2017).
[^10]: Sak, Haşim, Andrew Senior, and Françoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." *Fifteenth annual conference of the international speech communication association*. 2014.
[^11]: Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." *arXiv preprint arXiv:1406.1078* (2014).
[^12]: Srivastava, Rupesh Kumar, Klaus Greff, and Jürgen Schmidhuber. "Highway networks." *arXiv preprint arXiv:1505.00387* (2015).
[^13]: Srivastava, Rupesh K., Klaus Greff, and Jürgen Schmidhuber. "Training very deep networks." *Advances in neural information processing systems*. 2015.
[^14]: Reddi, Sashank J., Satyen Kale, and Sanjiv Kumar. "On the convergence of adam and beyond." (2018).
[^15]: Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling." *arXiv preprint arXiv:1803.01271* (2018).
[^16]: Vaswani, Ashish, et al. "Attention is all you need." *Advances in Neural Information Processing Systems*. 2017.
[^17]: http://nlp.seas.harvard.edu/2018/04/03/attention.html
[^18]: Dai, Zihang, et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." *arXiv preprint arXiv:1901.02860* (2019).
[^19]: Dehghani, Mostafa, et al. "Universal transformers." *arXiv preprint arXiv:1807.03819* (2018).
[^20]: Mozer, Michael C. "A focused backpropagation algorithm for temporal." *Backpropagation: Theory, architectures, and applications* (1995): 137.
[^21]: Williams, Ronald J., and David Zipser. "Gradient-based learning algorithms for recurrent networks and their computational complexity." *Backpropagation: Theory, architectures, and applications* 1 (1995): 433-486.
[^22]: Guo, Jiang. "Backpropagation through time." *Unpubl. ms., Harbin Institute of Technology* (2013).
[^30]: Greff, Klaus, Rupesh K. Srivastava, and Jürgen Schmidhuber. "Highway and residual networks learn unrolled iterative estimation." *arXiv preprint arXiv:1612.07771* (2016).
[^31]: Zilly, Julian Georg, et al. "Recurrent highway networks." *Proceedings of the 34th International Conference on Machine Learning-Volume 70*. JMLR. org, 2017.

[^32]: He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.
[^34]: Li, Zhizhong, and Derek Hoiem. "Learning without forgetting." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 40.12 (2018): 2935-2947.
[^35]: Zhang, Yu, and Qiang Yang. "A survey on multi-task learning." *arXiv preprint arXiv:1707.08114* (2017).
[^36]: https://arxiv.org/abs/1901.11504



## NLP

*   Language Model
    *   LSTM-based
        *   ELMo and the related important references. [^23] [^24] [^25]  ✔️
        *   Understanding the representation of biLM. [^26] 🚧❤️
        *   A simple sentence representation baseline. [^27] ✔️
        *   Doubts about the "pretraining and freezing" pattern. [^28] 🏷️
    *   Transformer-based
        *   BERT. [^29] 🏷
*   Embedding
    *   TODO
*   Chinese Word Segmentation
    *   TODO
*   Named Entity Recognition
    *   TODO
*   Semantic Textual Similarity / Natural Language Inference
    *   TODO
*   Natural Language Understanding
    *   TODO
*   Question Answering
    *   TODO
*   Text Classification
    *   SGM. [^33] 



[^23]: Peters, Matthew E., et al. "Deep contextualized word representations." *arXiv preprint arXiv:1802.05365* (2018).
[^24]: Jozefowicz, Rafal, et al. "Exploring the limits of language modeling." *arXiv preprint arXiv:1602.02410* (2016).
[^25]: Kim, Yoon, et al. "Character-Aware Neural Language Models." *AAAI*. 2016.
[^26]: Peters, Matthew E., et al. "Dissecting contextual word embeddings: Architecture and representation." *arXiv preprint arXiv:1808.08949*(2018).
[^27]: Perone, Christian S., Roberto Silveira, and Thomas S. Paula. "Evaluation of sentence embeddings in downstream and linguistic probing tasks." *arXiv preprint arXiv:1806.06259* (2018).
[^28]: Bowman, Samuel R., et al. "Looking for ELMo's friends: Sentence-Level Pretraining Beyond Language Modeling." *arXiv preprint arXiv:1812.10860* (2018).
[^29]: Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805*(2018).

[^33]: Yang, Pengcheng, et al. "SGM: sequence generation model for multi-label classification." *Proceedings of the 27th International Conference on Computational Linguistics*. 2018.

