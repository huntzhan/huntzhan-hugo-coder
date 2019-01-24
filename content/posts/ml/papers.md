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
    *   Highway connections. [^12] [^13] 
*   LSTM
    *       The original formulation. [^1] 🏷️
    *       BPTT. [^21] [^20] [^22] 🚧
    *       Forget gate. [^2] 🏷️
    *       Peephole connections. [^4] 🏷️
    *       The vanilla LSTM achieved by incorporating [^2], [^4] and full BPTT into [^1], which is the most popular LSTM architecture nowadays. [^5] 🚧
    *       Gradient norm clipping. [^6] 🏷️
    *       DropConnect dropout. [^7] ✔️
    *       Variational dropout. [^8] 
    *       Recurrent projection. [^10] 
    *       Residual connections. [^9] 
    *       GRU. [^11] 
    *       An empirical study on the LSTM architecture. This paper shows that none of the variants is significantly better than the vanilla LSTM. [^3] 🚧❤️
*   Transformer
    *   The original paper [^16] and the annotated version [^17]. 🏷️
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
[^21]: Werbos, Paul J. "Backpropagation through time: what it does and how to do it." *Proceedings of the IEEE* 78.10 (1990): 1550-1560.
[^22]: Guo, Jiang. "Backpropagation through time." *Unpubl. ms., Harbin Institute of Technology* (2013).



## NLP

Language Model:

*   TODO
