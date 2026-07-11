# Embedding Model Comparison

Comparing **openai/text-embedding-3-small** (Model A) vs **baai/bge-large-en-v1.5** (Model B) for retrieval quality on the PDFs in `docs/`, using 3 test questions and top-3 retrieval from ChromaDB.

Relevance is scored as the fraction of meaningful question words that appear in the retrieved chunk (a model-agnostic proxy, since raw ChromaDB distances from different embedding models are not on the same scale and can't be compared directly).

## Per-question results

### What is the attention mechanism in transformers?

Chunks both models agreed on: 2/3

**Model A:**

1. `machinelearning1.pdf` chunk 9 (distance=0.6264, relevance=0.67): ttention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of depe...

2. `machinelearning1.pdf` chunk 14 (distance=0.8084, relevance=0.33): rst transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or c...

3. `machinelearning1.pdf` chunk 1 (distance=0.8212, relevance=0.67): University of Toronto aidan@cs.toronto.edu Łukasz Kaiser∗ Google Brain lukaszkaiser@google.com Illia Polosukhin∗ ‡ illia.polosukhin@gmail.com Abstract...

**Model B:**

1. `machinelearning1.pdf` chunk 27 (distance=0.4669, relevance=0.33): total computational cost is similar to that of single-head attention with full dimensionality. 3.2.3 Applications of Attention in our Model The Transf...

2. `machinelearning1.pdf` chunk 9 (distance=0.4677, relevance=0.67): ttention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of depe...

3. `machinelearning1.pdf` chunk 1 (distance=0.5410, relevance=0.67): University of Toronto aidan@cs.toronto.edu Łukasz Kaiser∗ Google Brain lukaszkaiser@google.com Illia Polosukhin∗ ‡ illia.polosukhin@gmail.com Abstract...


### How does multi-head attention work in the Transformer model?

Chunks both models agreed on: 1/3

**Model A:**

1. `machinelearning1.pdf` chunk 27 (distance=0.5089, relevance=0.83): total computational cost is similar to that of single-head attention with full dimensionality. 3.2.3 Applications of Attention in our Model The Transf...

2. `machinelearning1.pdf` chunk 9 (distance=0.6804, relevance=0.67): ttention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of depe...

3. `machinelearning1.pdf` chunk 65 (distance=0.7612, relevance=0.50): on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of t...

**Model B:**

1. `machinelearning1.pdf` chunk 27 (distance=0.3283, relevance=0.83): total computational cost is similar to that of single-head attention with full dimensionality. 3.2.3 Applications of Attention in our Model The Transf...

2. `machinelearning1.pdf` chunk 26 (distance=0.3324, relevance=0.50): tions. With a single attention head, averaging inhibits this. MultiHead(Q, K, V) = Concat(head1, ...,headh)W O where headi = Attention(QW Q i , KWK i ...

3. `machinelearning1.pdf` chunk 24 (distance=0.4510, relevance=0.50): .2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to ...


### What are dilated convolutions and how do they help with dense prediction tasks like semantic segmentation?

Chunks both models agreed on: 2/3

**Model A:**

1. `machinelearning2.pdf` chunk 1 (distance=0.6587, relevance=0.62): this work, we develop a new convolutional network module that is speciﬁcally designed for dense prediction. The presented module uses dilated convolut...

2. `machinelearning2.pdf` chunk 9 (distance=0.7190, relevance=0.75): e presented context module is designed speciﬁcally for dense prediction. It is a rectangular prism of convolutional layers, with no pooling or subsamp...

3. `machinelearning2.pdf` chunk 4 (distance=0.7655, relevance=0.38): lutional networks (LeCun et al., 1989) trained by backpropagation (Rumelhart et al., 1986). Speciﬁcally, Long et al. (2015) showed that convolutional ...

**Model B:**

1. `machinelearning2.pdf` chunk 1 (distance=0.2752, relevance=0.62): this work, we develop a new convolutional network module that is speciﬁcally designed for dense prediction. The presented module uses dilated convolut...

2. `machinelearning2.pdf` chunk 0 (distance=0.3325, relevance=0.75): Published as a conference paper at ICLR 2016 MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS Fisher Yu Princeton University Vladlen Koltun Int...

3. `machinelearning2.pdf` chunk 9 (distance=0.3352, relevance=0.75): e presented context module is designed speciﬁcally for dense prediction. It is a rectangular prism of convolutional layers, with no pooling or subsamp...


## Summary

- Average relevance, Model A (`openai/text-embedding-3-small`): **0.60**
- Average relevance, Model B (`baai/bge-large-en-v1.5`): **0.62**
- Better performer on this test set: **baai/bge-large-en-v1.5**

baai/bge-large-en-v1.5 retrieved chunks whose text overlapped more with the question's own vocabulary across the test questions, suggesting its embeddings placed the genuinely relevant passages closer to the question than the other model did. `openai/text-embedding-3-small` is a general-purpose commercial embedding model, while `baai/bge-large-en-v1.5` is an open-source model tuned specifically for retrieval tasks, so a close or reversed result here is plausible depending on the document domain.

**Caveats:** this is a tiny test set (2 papers, 3 questions) and the relevance score is lexical, not semantic - a chunk that answers the question in different words would score low even if a human would call it relevant. Treat this as a directional signal, not a rigorous benchmark.
