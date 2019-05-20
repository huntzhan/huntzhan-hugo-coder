+++
date = 2019-01-22T18:02:01+08:00
title = "Papers Reading Roadmap"
math = "true"

+++



## Introduction

This page tracks my reading roadmap of deep learning papers. I will update this page occasionally (probably every 3 - 5 days) according to my progress.

Notation:

*   ✔️: Done.
*   🐦: Just have a glance.
*   ❤️: Love it!
*   🤔: Probably something is not right, but I'm not sure.



## Deep Learning Theory And Practice

*   Network Architecture 
    *   General
        *   Highway connection that is inspired by LSTM forget gate to ease the training of very deep networks. [^12] ✔️ [^13] 
        *   ResNet (deep residual nets). [^32] 
        *   Interpretations of highway and residual networks. [^30] 
        *   Maxout. [^43]
        *   Layer normalization. [^123]
        *   GELU. [^125]
        *   Survey of dropout methods. [^134]
    *   CNN
        -   A guide with excellent visualization. [^58] ✔️❤️
        -   1x1 Convolutions. [^95]
        -   TCN. [^15] ✔️
    *   LSTM
        *       The original formulation. [^1]
        *       BPTT. [^21] [^20] [^22] ✔️
        *       Enhance TBPTT by auxiliary losses. [^64] 
        *       Forget gate. [^2] 
        *       Peephole connections. [^4] 
        *       The vanilla LSTM achieved by incorporating [^2] [^4] and full BPTT into [^1], which is the most popular LSTM architecture nowadays. [^5] 
        *       Gradient norm clipping. [^6] 
        *       DropConnect dropout. [^7] ✔️
        *       Variational dropout. [^8] 
        *       Recurrent projection. [^10] 
        *       Residual connections. [^9] 
        *       GRU. [^11] 
        *       An empirical study on the LSTM architecture. This paper shows that none of the variants is significantly better than the vanilla LSTM. [^3] 🐦❤️
        *       RHN. [^31]
    *   Transformer
        *   Attention mechanisms.  [^118] ✔️❤️ [^113] ✔️ [^114] ✔️
        *   The first proposed paper and the annotated version. [^16] ✔️❤️ [^17] ✔️
        *   Universal transformer. [^19] 
        *   Transformer-XL. [^18] 
*   Optimizer
    *   AMSGRAD. [^14] 
    *   CLR. [^99]
    *   SGDR. [^100]
*   Transfer Learning / Multi-Task Learning
    -   MTL surveys. [^35] [^44] 
    -   MT-DNN, achieves the SOTA of GLUE by applying MTL to the fine-tuning stage of BERT and formulating the QNLI as a pairwise ranking task which is inspired by DSSM. [^36] ✔️❤️
    -   Gradually adding new capabilities to a system. [^34] 
*   Pruning / Quantization:
    *   Pruning CNN. [^55] [^56]
    *   Deep Compression. [^57]
    *   FP16 computation and mixed-precision training. [^130] ✔️ [^128] [^131]



## Classical Machine Learning

-   CRF. [^50]



## NLP

*   Embedding (Context-Independent)
    *   Word-Level
        *   The groundbreaking papers: Word2Vec [^81] and negative sampling [^82] .
        *   A good tutorial on CBOW, SG, hierarchical softmax, and negative sampling. [^75] ✔️ ❤️
        *   The gains of hyperparameter optimization. [^88] 
        *   Glove. [^79] ✔️
        *   FastText. [^80] ✔️
        *   Wang2Vec. [^77] ✔️
        *   Sense2Vec. [^78] ✔️
        *   "Discourse atoms" for polysemy. [^90] 
        *   PMI-based. [^89] 
        *   A theoretical understanding of the dimensionality. [^63] ❤️
        *   Hybrid CBOW-CMOW. [^83] ✔️
        *   DSG. [^70] ✔️
        *   Ngram2vec. [^102]
        *   Segmentation-free embedding [^76] ✔️ [^74] ✔️ [^104] 
        *   Evaluation on linguistic properties. [^84] [^87]
    *   Above Word-Level
        *   CNN baselines for sentence-level classification. [^101]
        *   Sentence embedding by introducing self-attention. [^116] 🐦
        *   Skip-Thought. [^91] ✔️
        *   InferSent. [^92] ✔️
        *   Quick-Thought. [^93] ✔️
        *   Sent2Vec. [^94] ✔️
*   Language Model
    *   General
        *   Understanding the representation of biLMs (BiLSTM, Transformer, Gated CNN). [^26] ✔️❤️
        *   Doubts about the "pretraining and freezing" pattern. [^28] 
        *   Effects on sentiment. [^61] ✔️
        *   Effects on commonsense reasoning. [^60] ✔️❤️
    *   LSTM-based
        *   ELMo and the related important references. [^23] ✔️ [^24] ✔️ [^25]  ✔️
        *   A simple sentence representation baseline. [^27] ✔️
    *   Transformer-based
        *   BERT [^29] ✔️❤️ and the discussion on Reddit about the cost of pre-training BERT. [^129] ✔️
        *   ERNIE. [^124] ✔️
        *   GPT-2. [^54] 
        *   Adding additional LSTM layers to BERT/GPT to capture the strong word-level context. [^133] ✔️
*   Semantic Textual Similarity / Natural Language Inference
    -   DSSM [^38] 
    -   CDSSM (a.k.a. CLSM) [^53] ✔️❤️ [^39] ✔️
    -   ESIM. [^68] 
    -   BiMPM. [^41] ✔️
    -   pt-DecAtt. [^40] ✔️
    -   DIIN. [^42] 
    -   SAN, applies attention mechanism and GRU to formulate the multi-step inference. [^37] ✔️🤔
    -   An analysis of NN designs for sentence pair modeling. [^47] 
    -   MwAN. [^59] 
    -   COTA, Uber's practice. [^69]
    -   LCQMC dataset. [^112] ✔️❤️
*   Text Classification
    -   An empirical exploration of using char-level CNN for text classification. The idea of using thesaurus for data augmentation should help. [^127] ✔️
    -   SGM. [^33] 
    -   Seq2Set. [^46] 
    -   HN-ATT with multi-level (word, sentence) attention. [^119] 🐦
    -   ABSA with attention. [^120] 🐦
    -   Converting (T)ABSA to sentence-pair classification problems and utilizing the advantage of BERT-pair.  [^122] ✔️
    -   BERT post-training on domain knowledge and task-awareness knowledge with MTL, then fine-tune on RRC, AE, ASC. [^126] ✔️❤️
    -   AEN with label smoothing regularization for TSA. [^132]✔️
*   Chinese Word Segmentation
    *   Wang2vec + stacking bi-LSTM, simple yet effective baseline [^96] ✔️
    *   With multi-criteria leanring (MTL). [^97]  ✔️
    *   WEB-CWS. [^98]
    *   With Lattice LSTM. [^103]
*   Named Entity Recognition
*   Natural Language Understanding
*   Question Answering
    *   Danqi Chen's dissertation. [^67] 
    *   ComQA dataset with paraphrase clusters, showing that the current QA systems can not handle the complex (compositional, temporal, comparison) questions. This dataset could be transformed to evaluate STS/NLI/IR system.  [^111] ✔️❤️
*   Dialogue
    *   DSTC overview. [^65] 
    *   PyDial. [^66]
*   Sequence-to-Sequence Modeling
    -   First proposed papers. [^48] [^49] ✔️
    -   Beam search optimization. [^45] [^51] 
    -   Copy Mechanism. [^85]
*   Information Retrieval
    *   A theoretical understanding of IDF. [^73]
    *   WMD. [^72] 🐦
    *   Incorporating with word embedding. [^71] ✔️
    *   Keyphrase generation in seq2seq pattern. [^86]
*   Knowledge Graph
    -   Survey papers. [^109] [^110]
    -   A systematic study on the geometry of various KGE. [^62]
    -   KEQA. [^108]
*   Grammatical Error Correction
    *   Nested Attention (top-down hierarchical attention). [^115] 🐦
*   Others
    *   PyText. [^52] ✔️
    *   Advice for using GPUs [^117] ✔️❤️ and other hardware [^121] .



## CV

*   Document Layout Analysis
    *   Scholarly figure/table: PDFFigures [^105] and PDFFigures2 [^106] .
    *   TableBank. [^107]



[^1]: Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." *Neural computation* 9.8 (1997): 1735-1780.APA
[^2]: Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to forget: Continual prediction with LSTM." (1999): 850-855.
[^3]: Greff, Klaus, et al. "LSTM: A search space odyssey." *IEEE transactions on neural networks and learning systems* 28.10 (2017): 2222-2232.
[^4]: Gers, Felix A., and Jürgen Schmidhuber. "Recurrent nets that time and count." *Proceedings of the IEEE-INNS-ENNS International Joint Conference on Neural Networks. IJCNN 2000. Neural Computing: New Challenges and Perspectives for the New Millennium*. Vol. 3. IEEE, 2000.
[^5]: Graves, Alex, and Jürgen Schmidhuber. "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." *Neural Networks* 18.5-6 (2005): 602-610.
[^6]: Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty of training recurrent neural networks." *International Conference on Machine Learning*. 2013.
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
[^36]: Liu, Xiaodong, et al. "Multi-Task Deep Neural Networks for Natural Language Understanding." *arXiv preprint arXiv:1901.11504* (2019).
[^23]: Peters, Matthew E., et al. "Deep contextualized word representations." *arXiv preprint arXiv:1802.05365* (2018).
[^24]: Jozefowicz, Rafal, et al. "Exploring the limits of language modeling." *arXiv preprint arXiv:1602.02410* (2016).
[^25]: Kim, Yoon, et al. "Character-Aware Neural Language Models." *AAAI*. 2016.
[^26]: Peters, Matthew E., et al. "Dissecting contextual word embeddings: Architecture and representation." *arXiv preprint arXiv:1808.08949*(2018).
[^27]: Perone, Christian S., Roberto Silveira, and Thomas S. Paula. "Evaluation of sentence embeddings in downstream and linguistic probing tasks." *arXiv preprint arXiv:1806.06259* (2018).
[^28]: Bowman, Samuel R., et al. "Looking for ELMo's friends: Sentence-Level Pretraining Beyond Language Modeling." *arXiv preprint arXiv:1812.10860* (2018).
[^29]: Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805*(2018).

[^33]: Yang, Pengcheng, et al. "SGM: sequence generation model for multi-label classification." *Proceedings of the 27th International Conference on Computational Linguistics*. 2018.
[^37]: Liu, Xiaodong, Kevin Duh, and Jianfeng Gao. "Stochastic Answer Networks for Natural Language Inference." *arXiv preprint arXiv:1804.07888* (2018).
[^38]: Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data." *Proceedings of the 22nd ACM international conference on Conference on information & knowledge management*. ACM, 2013.
[^39]: Shen, Yelong, et al. "Learning semantic representations using convolutional neural networks for web search." *Proceedings of the 23rd International Conference on World Wide Web*. ACM, 2014.
[^40]: Tomar, Gaurav Singh, et al. "Neural paraphrase identification of questions with noisy pretraining." *arXiv preprint arXiv:1704.04565* (2017).
[^41]: Wang, Zhiguo, Wael Hamza, and Radu Florian. "Bilateral multi-perspective matching for natural language sentences." *arXiv preprint arXiv:1702.03814* (2017).
[^42]: Gong, Yichen, Heng Luo, and Jian Zhang. "Natural language inference over interaction space." *arXiv preprint arXiv:1709.04348* (2017).
[^43]: Goodfellow, Ian J., et al. "Maxout networks." *arXiv preprint arXiv:1302.4389* (2013).

[^44]: http://ruder.io/multi-task/
[^45]: Wiseman, Sam, and Alexander M. Rush. "Sequence-to-sequence learning as beam-search optimization." *arXiv preprint arXiv:1606.02960*(2016).
[^46]: Yang, Pengcheng, et al. "A Deep Reinforced Sequence-to-Set Model for Multi-Label Text Classification." *arXiv preprint arXiv:1809.03118*(2018).
[^47]: Lan, Wuwei, and Wei Xu. "Neural Network Models for Paraphrase Identification, Semantic Textual Similarity, Natural Language Inference, and Question Answering." *arXiv preprint arXiv:1806.04330* (2018).
[^48]: Sutskever, Ilya, James Martens, and Geoffrey E. Hinton. "Generating text with recurrent neural networks." *Proceedings of the 28th International Conference on Machine Learning (ICML-11)*. 2011.
[^49]: Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." *Advances in neural information processing systems*. 2014.
[^50]: Lafferty, John, Andrew McCallum, and Fernando CN Pereira. "Conditional random fields: Probabilistic models for segmenting and labeling sequence data." (2001).
[^51]: Learning as Search Optimization: Approximate Large Margin Methods for Structured Prediction

[^52]: Aly, Ahmed, et al. "PyText: A Seamless Path from NLP research to production." *arXiv preprint arXiv:1812.08729* (2018).
[^53]: Shen, Yelong, et al. "A latent semantic model with convolutional-pooling structure for information retrieval." *Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management*. ACM, 2014.
[^54]: https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
[^55]: https://jacobgil.github.io/deeplearning/pruning-deep-learning
[^56]: Molchanov, Pavlo, et al. "Pruning convolutional neural networks for resource efficient inference." *arXiv preprint arXiv:1611.06440* (2016).
[^57]: Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." *arXiv preprint arXiv:1510.00149* (2015).

[^58]: Dumoulin, Vincent, and Francesco Visin. "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285* (2016).  Github: https://github.com/vdumoulin/conv_arithmetic
[^59]: Tan, Chuanqi, et al. "Multiway Attention Networks for Modeling Sentence Pairs." *IJCAI*. 2018.

[^60]: Trinh, Trieu H., and Quoc V. Le. "A simple method for commonsense reasoning." *arXiv preprint arXiv:1806.02847*(2018).
[^61]: Radford, Alec, Rafal Jozefowicz, and Ilya Sutskever. "Learning to generate reviews and discovering sentiment." *arXiv preprint arXiv:1704.01444* (2017).
[^62]: Sharma, Aditya, and Partha Talukdar. "Towards understanding the geometry of knowledge graph embeddings." *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Vol. 1. 2018.
[^63]: Yin, Zi, and Yuanyuan Shen. "On the dimensionality of word embedding." *Advances in Neural Information Processing Systems*. 2018.

[^64]: Trinh, Trieu H., et al. "Learning longer-term dependencies in rnns with auxiliary losses." *arXiv preprint arXiv:1803.00144* (2018).

[^65]: Williams, Jason, Antoine Raux, and Matthew Henderson. "The dialog state tracking challenge series: A review." *Dialogue & Discourse* 7.3 (2016): 4-33.
[^66]: Ultes, Stefan, et al. "Pydial: A multi-domain statistical dialogue system toolkit." *Proceedings of ACL 2017, System Demonstrations* (2017): 73-78.
[^67]: Chen, Danqi. *Neural Reading Comprehension and Beyond*. Diss. Stanford University, 2018.  https://stacks.stanford.edu/file/druid:gd576xb1833/thesis-augmented.pdf
[^68]: Chen, Qian, et al. "Enhanced lstm for natural language inference." *arXiv preprint arXiv:1609.06038* (2016).

[^69]: Molino, Piero, Huaixiu Zheng, and Yi-Chia Wang. "COTA: Improving the Speed and Accuracy of Customer Support through Ranking and Deep Networks." *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM, 2018.
[^70]: Song, Yan, et al. "Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings." *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)*. Vol. 2. 2018.  https://ai.tencent.com/ailab/nlp/embedding.html
[^71]: Galke, Lukas, Ahmed Saleh, and Ansgar Scherp. "Word embeddings for practical information retrieval." *INFORMATIK 2017* (2017).
[^72]: Kusner, Matt, et al. "From word embeddings to document distances." *International Conference on Machine Learning*. 2015.

[^73]: Robertson, Stephen. "Understanding inverse document frequency: on theoretical arguments for IDF." *Journal of documentation* 60.5 (2004): 503-520.

[^74]: Kim, Geewook, Kazuki Fukui, and Hidetoshi Shimodaira. "Segmentation-free compositional $ n $-gram embedding." *arXiv preprint arXiv:1809.00918* (2018).
[^75]: Rong, Xin. "word2vec parameter learning explained." *arXiv preprint arXiv:1411.2738* (2014).
[^76]: Oshikiri, Takamasa. "Segmentation-free word embedding for unsegmented languages." *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. 2017.
[^77]: Ling, Wang, et al. "Two/too simple adaptations of word2vec for syntax problems." *Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. 2015.
[^78]: Trask, Andrew, Phil Michalak, and John Liu. "sense2vec-a fast and accurate method for word sense disambiguation in neural word embeddings." *arXiv preprint arXiv:1511.06388* (2015).
[^79]: Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*. 2014.
[^80]: Bojanowski, Piotr, et al. "Enriching word vectors with subword information." *Transactions of the Association for Computational Linguistics* 5 (2017): 135-146.APA
[^81]: Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." *arXiv preprint arXiv:1301.3781* (2013).
[^82]: Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." *Advances in neural information processing systems*. 2013.
[^83]: Mai, Florian, Lukas Galke, and Ansgar Scherp. "CBOW Is Not All You Need: Combining CBOW with the Compositional Matrix Space Model." *arXiv preprint arXiv:1902.06423* (2019).  Review: https://openreview.net/forum?id=H1MgjoR9tQ
[^84]: Conneau, Alexis, et al. "What you can cram into a single vector: Probing sentence embeddings for linguistic properties." *arXiv preprint arXiv:1805.01070* (2018).

[^85]: Gu, Jiatao, et al. "Incorporating copying mechanism in sequence-to-sequence learning." *arXiv preprint arXiv:1603.06393* (2016).APA
[^86]: Meng, Rui, et al. "Deep keyphrase generation." *arXiv preprint arXiv:1704.06879* (2017).
[^87]: Schnabel, Tobias, et al. "Evaluation methods for unsupervised word embeddings." *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*. 2015.APA
[^88]: Levy, Omer, Yoav Goldberg, and Ido Dagan. "Improving distributional similarity with lessons learned from word embeddings." *Transactions of the Association for Computational Linguistics* 3 (2015): 211-225.
[^89]: Arora, Sanjeev, et al. "A latent variable model approach to pmi-based word embeddings." *Transactions of the Association for Computational Linguistics* 4 (2016): 385-399.
[^90]: Arora, Sanjeev, et al. "Linear algebraic structure of word senses, with applications to polysemy." *Transactions of the Association of Computational Linguistics* 6 (2018): 483-495.

[^91]: Kiros, Ryan, et al. "Skip-thought vectors." *Advances in neural information processing systems*. 2015.
[^92]: Conneau, Alexis, et al. "Supervised learning of universal sentence representations from natural language inference data." *arXiv preprint arXiv:1705.02364* (2017).
[^93]: Logeswaran, Lajanugen, and Honglak Lee. "An efficient framework for learning sentence representations." *arXiv preprint arXiv:1803.02893*(2018).
[^94]: Pagliardini, Matteo, Prakhar Gupta, and Martin Jaggi. "Unsupervised learning of sentence embeddings using compositional n-gram features." *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*. Vol. 1. 2018.

[^95]: Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." *arXiv preprint arXiv:1312.4400* (2013).
[^96]: Ma, Ji, Kuzman Ganchev, and David Weiss. "State-of-the-art Chinese word segmentation with bi-lstms." *arXiv preprint arXiv:1808.06511* (2018).
[^97]: He, Han, et al. "Effective Neural Solution for Multi-criteria Word Segmentation." *Smart Intelligent Computing and Applications*. Springer, Singapore, 2019. 133-142.
[^98]: Ye, Yuxiao, et al. "Improving Cross-Domain Chinese Word Segmentation with Word Embeddings." *arXiv preprint arXiv:1903.01698*(2019).

[^99]: Smith, Leslie N. "Cyclical learning rates for training neural networks." *2017 IEEE Winter Conference on Applications of Computer Vision (WACV)*. IEEE, 2017.
[^100]: Loshchilov, Ilya, and Frank Hutter. "Sgdr: Stochastic gradient descent with warm restarts." *arXiv preprint arXiv:1608.03983* (2016).
[^101]: Kim, Yoon. "Convolutional neural networks for sentence classification." *arXiv preprint arXiv:1408.5882* (2014).
[^102]: Zhao, Zhe, et al. "Ngram2vec: Learning improved word representations from ngram co-occurrence statistics." *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. 2017.

[^103]: Yang, Jie, Yue Zhang, and Shuailong Liang. "Subword Encoding in Lattice LSTM for Chinese Word Segmentation." *arXiv preprint arXiv:1810.12594* (2018).
[^104]: Heinzerling, Benjamin, and Michael Strube. "Bpemb: Tokenization-free pre-trained subword embeddings in 275 languages." *arXiv preprint arXiv:1710.02187* (2017).
[^105]: Clark, Christopher Andreas, and Santosh Divvala. "Looking beyond text: Extracting figures, tables and captions from computer science papers." *Workshops at the Twenty-Ninth AAAI Conference on Artificial Intelligence*. 2015.
[^106]: Clark, Christopher, and Santosh Divvala. "Pdffigures 2.0: Mining figures from research papers." *2016 IEEE/ACM Joint Conference on Digital Libraries (JCDL)*. IEEE, 2016.
[^107]: Li, Minghao, et al. "TableBank: Table Benchmark for Image-based Table Detection and Recognition." *arXiv preprint arXiv:1903.01949* (2019).
[^108]: Huang, Xiao, et al. "Knowledge graph embedding based question answering." *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining*. ACM, 2019.  Github: <https://github.com/xhuang31/KEQA_WSDM19/tree/master>
[^109]: Wang, Quan, et al. "Knowledge graph embedding: A survey of approaches and applications." *IEEE Transactions on Knowledge and Data Engineering*29.12 (2017): 2724-2743.
[^110]: Nickel, Maximilian, et al. "A review of relational machine learning for knowledge graphs." *Proceedings of the IEEE* 104.1 (2016): 11-33.

[^111]: Abujabal, Abdalghani, et al. "ComQA: A Community-sourced Dataset for Complex Factoid Question Answering with Paraphrase Clusters." *arXiv preprint arXiv:1809.09528* (2018).
[^112]: Liu, Xin, et al. "LCQMC: A Large-scale Chinese Question Matching Corpus." *Proceedings of the 27th International Conference on Computational Linguistics*. 2018.
[^113]: Hu, Dichao. "An Introductory Survey on Attention Mechanisms in NLP Problems." *arXiv preprint arXiv:1811.05544* (2018).
[^114]: Attention? Attention! <https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html>

[^115]: Ji, Jianshu, et al. "A nested attention neural hybrid model for grammatical error correction." *arXiv preprint arXiv:1707.02026* (2017).
[^116]: Lin, Zhouhan, et al. "A structured self-attentive sentence embedding." *arXiv preprint arXiv:1703.03130* (2017).
[^117]: Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning <http://timdettmers.com/2019/04/03/which-gpu-for-deep-learning/>
[^118]: Chaudhari, Sneha, et al. "An Attentive Survey of Attention Models." *arXiv preprint arXiv:1904.02874* (2019).
[^119]: Yang, Zichao, et al. "Hierarchical attention networks for document classification." *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. 2016.
[^120]: Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based LSTM for aspect-level sentiment classification." *Proceedings of the 2016 conference on empirical methods in natural language processing*. 2016.
[^121]: A Full Hardware Guide to Deep Learning <https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/>

[^122]: Sun, Chi, Luyao Huang, and Xipeng Qiu. "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence." *arXiv preprint arXiv:1903.09588* (2019).

[^123]: Lei Ba, Jimmy, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." *arXiv preprint arXiv:1607.06450* (2016).
[^124]: ERNIE: Enhanced Representation through Knowledge Integration Yu. <https://arxiv.org/abs/1904.09223>
[^125]: Hendrycks, Dan, and Kevin Gimpel. "Gaussian Error Linear Units (GELUs)." *arXiv preprint arXiv:1606.08415* (2016).

[^126]: Xu, Hu, et al. "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis." *arXiv preprint arXiv:1904.02232* (2019).  Github: https://github.com/howardhsu/BERT-for-RRC-ABSA
[^127]: Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional networks for text classification." *Advances in neural information processing systems*. 2015.
[^128]: https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
[^129]: https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/
[^130]: https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
[^131]: Micikevicius, Paulius, et al. "Mixed precision training." *arXiv preprint arXiv:1710.03740* (2017).
[^132]: Song, Youwei, et al. "Attentional Encoder Network for Targeted Sentiment Classification." *arXiv preprint arXiv:1902.09314* (2019).  Github: https://github.com/songyouwei/ABSA-PyTorch/tree/aen

[^133]: Wang, Chenguang, Mu Li, and Alexander J. Smola. "Language Models with Transformers." *arXiv preprint arXiv:1904.09408* (2019).

[^134]: Labach, Alex, Hojjat Salehinejad, and Shahrokh Valaee. "Survey of Dropout Methods for Deep Neural Networks." *arXiv preprint arXiv:1904.13310*(2019).

