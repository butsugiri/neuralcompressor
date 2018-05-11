# chainer-nncompress: Chainer Implementations of Embedding Quantization 

This is the chainer port of `nncompress` (forked from Tensorflow impl. by the author of the paper).

The most of my implementation corresponds to the original implementation.

## Requirements

- Python 3.5 or higher
- Chainer 4.0.0
- CuPy 4.0.0
- logzero

## Usage

1. Download the project and prepare the data

```
> git clone https://github.com/zomux/neuralcompressor
> cd neuralcompressor
> bash scripts/download_glove_data.sh
```

2. Convert the Glove embeddings to numpy format

```
> python scripts/convert_glove2numpy.py data/glove.6B.300d.txt
```

3. Train the embedding quantization model

```
> python train_comp.py -b 64 -e 200 -g 0 -O Adam --M 32 --K 16 --tau 0.1 --input-matrix ./data/glove.6B.300d.npy
```

```
...
199         1.245e+06   11.519      11.8183               0.980877    0.982509
199         1.246e+06   11.7329     11.795                0.982026    0.982303
199         1.247e+06   11.6411     11.8103               0.983015    0.982784
199         1.248e+06   11.2741     11.8116               0.981655    0.982299
199         1.249e+06   11.9637     11.8323               0.982289    0.982389
200         1.25e+06    11.3458     11.8293               0.982983    0.982753
200         1.25e+06    11.3458     11.8293               0.982983    0.982753
[I 180511 00:46:57 train_comp:123] Training complete!!
[I 180511 00:46:57 resource:122] EXIT TIME: 20180511 - 00:46:57
[I 180511 00:46:57 resource:124] Duration: 2:03:28.673434
[I 180511 00:46:57 resource:125] Remember: log is saved in /home/kiyono/deploy/nncompress/result/20180510_224328_model_seed_0_optim_Adam_tau_0.1_batch_64_M_32_K_16
```

4. Export the word codes and the codebook matrix

```
> python python infer_comp.py --gpu 0 --model result/20180510_224328_model_seed_0_optim_Adam_tau_0.1_batch_64_M_32_K_16/iter_1211000.npz --vocab-path data/glove.6B.300d.word --matrix data/glove.6B.300d.npy
```

It will generate two files in result dir:
- iter_1211000.npz.codebook.npy
- iter_1211000.npz.codes

6. Check the codes

```
> head -100 iter_1211000.npz.codes
```

```
...
pao     9 11 10 12 7 9 5 2 0 14 2 12 6 3 2 12 0 7 2 4 9 5 3 10 15 1 5 15 2 10 15 4
comforts        12 8 6 8 15 5 1 4 10 4 15 3 1 15 14 5 6 12 2 12 1 12 12 6 4 14 7 0 9 11 13 3
nellie  2 15 1 13 14 0 5 3 7 13 0 8 1 0 14 4 3 6 2 3 1 4 8 7 15 1 7 15 4 11 9 4
trondheim       6 7 8 6 6 3 8 7 6 3 11 13 11 3 9 14 9 10 4 8 15 9 13 12 3 7 6 12 15 1 4 13
...
```

---
# Following is the README of Original Repo
# nncompress: Implementations of Embedding Quantization (Compress Word Embeddings)

Thank you for your interest on our paper.

I'm receieving mail basically everyday and happy to know many of you implemented the model correctly.

I'm glad to debug your code or have discussion with you.

Please do not hesitate to mail me for help.

`mail_address = "raph_ael@ua_ca.com".replace("_", "")`

### Requirements:

numpy and tensorflow (I also have the pytorch implementation, which will be uploaded)

### Tutorial of the code

1. Download the project and prepare the data

```
> git clone https://github.com/zomux/neuralcompressor
> cd neuralcompressor
> bash scripts/download_glove_data.sh
```

2. Convert the Glove embeddings to numpy format

```
> python scripts/convert_glove2numpy.py data/glove.6B.300d.txt
```

3. Train the embedding quantization model

```
> python bin/quantize_embed.py -M 32 -K 16 --train
```

```
...
[epoch198] train_loss=12.82 train_maxp=0.98 valid_loss=12.50 valid_maxp=0.98 bps=618 *
[epoch199] train_loss=12.80 train_maxp=0.98 valid_loss=12.53 valid_maxp=0.98 bps=605
Training Done
```

4. Evaluate the averaged euclidean distance

```
> python bin/quantize_embed.py -M 32 -K 16 --evaluate
```

```
Mean euclidean distance: 4.889592628145218
```

5. Export the word codes and the codebook matrix

```
> python bin/quantize_embed.py -M 32 -K 16 --export
```

It will generate two files:
- data/mymodel.codes
- data/mymodel.codebook.npy

6. Check the codes

```
> paste data/glove.6B.300d.word data/mymodel.codes | head -n 100
```

```
...
only    15 14 7 10 1 14 14 3 0 9 1 9 3 3 0 0 12 1 3 12 15 3 11 12 12 6 1 5 13 6 2 6
state   7 13 7 3 8 14 10 6 6 4 12 2 9 3 9 0 1 1 3 9 11 10 0 14 14 4 15 5 0 6 2 1
million 5 7 3 15 1 14 4 0 6 11 1 4 8 3 1 0 0 1 3 14 8 6 6 5 2 1 2 12 13 6 6 15
could   3 14 7 0 2 14 5 3 0 9 1 0 2 3 9 0 3 1 3 11 5 15 1 12 12 6 1 6 2 6 2 10
...
```

### Use it in python

```python
from nncompress import EmbeddingCompressor

# Load my embedding matrix
matrix = np.load("data/glove.6B.300d.npy")

# Initialize the compressor
compressor = EmbeddingCompressor(32, 16, "data/mymodel")

# Train the quantization model
compressor.train(matrix)

# Evaluate
distance = compressor.evaluate(matrix)
print("Mean euclidean distance:", distance)

# Export the codes and codebook
compressor.export(matrix, "data/mymodel")
```

### Citation

```
@inproceedings{shu2018compressing,
title={Compressing Word Embeddings via Deep Compositional Code Learning},
author={Raphael Shu and Hideki Nakayama},
booktitle={International Conference on Learning Representations (ICLR)},
year={2018},
url={https://openreview.net/forum?id=BJRZzFlRb},
}
```

Arxiv version: https://arxiv.org/abs/1711.01068
