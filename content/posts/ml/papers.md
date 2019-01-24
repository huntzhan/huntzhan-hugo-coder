+++
date = 2019-01-22T18:02:01+08:00
title = "Papers Reading Roadmap"
math = "true"

+++

## Introduction

This page tracks my  reading roadmap of deep learning papers. I will update this page occasionally (probably every 3 - 5 days) according to my progress.

Notation:

*   ❤️: Impressive!
*   🤔: Probably this work is controversial.
*   🏷️: Haven't started yet.
*   🚧: Ongoing.

## Deep Learning Theory And Practice

*   LSTM
    *       The original formulation. [^1] 🏷️
    *       The *forget gate*. [^2] 🏷️
    *       The *peephole connections*. [^4] 🏷️
    *       The *vanilla LSTM* achieved by incorporating [^2], [^4] and full BPTT into [^1], which is the *most popular* LSTM architecture nowadays. [^5] 🚧
    *       Gradient norm clipping. [^6] 🏷️
    *       DropConnect dropout. [^7]
    *       Variational dropout. [^8] 🏷️
    *       Residual connections. [^9] 🏷️
    *       Recurrent projection. [^10] 🏷️
    *       An empirical study on the LSTM architecture. This paper shows that *none of the variants* is significantly better than the vanilla LSTM. [^3] 🚧❤️
*   Transformer
    *   TODO
*   CNN
    *   TODO
*   Optimizer
    *   TODO



Note:

*   Tensorflow's LSTM implementation is based on [^2].
*   Pytorch's LSTM imlemntation is based on [^5]. The "peephole connections" is omitted.





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



## NLP

Language Model:

*   TODO
