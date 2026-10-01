# Results

FiQA2018 test set, nDCG@10, `google/embeddinggemma-300m`. Output of `python step3_ladder.py`:

```
none           0.4739   diff +0.0000   p nan
r1             0.4731   diff -0.0007   p 0.831
r2             0.4765   diff +0.0027   p 0.132
spectemp_768   0.4738   diff -0.0001   p 0.984
spectemp_256   0.4527   diff -0.0211   p 0.000183
spectemp_128   0.4242   diff -0.0497   p 4.93e-10
pca_128        0.4126   diff -0.0612   p 1.07e-12
whiten_128     0.4128   diff -0.0611   p 4.74e-13
```

Baseline `none` is 0.4739 vs 0.4774 on the MTEB leaderboard.

Setup: mteb 2.21.10, dataset `mteb/fiqa` @ `27a168819829fe9bcd655c2df245fb19452e8e06`, torch 2.11.0+cu128, sentence-transformers 6.1.0, transformers 5.18.0, Python 3.14.6, RTX 5060 Laptop (8 GB), document batch size 16.
