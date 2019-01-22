+++
date = 2019-01-22T18:02:01+08:00
title = "Papers Reading Roadmap"
math = "true"

+++

## Introduction

This page tracks my  reading roadmap of deep learning papers.



## Deep Learning Theory And Practice

*   LSTM
    *        The original formulation. [^1]
    *        Introduce the *forget gate*. [^2]
    *        Introduce the *peephole connections*. [^4]
    *        The *vanilla LSTM* by incorporating [^2] and [^4] into [^1], which is the *most popular* LSTM architecture. [^5]
    *        A empirical study on the LSTM architecture. This paper shows that *none of the variants is better than the vanilla LSTM significantly*. [^3]
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



## NLP

Language Model:

*   TODO
