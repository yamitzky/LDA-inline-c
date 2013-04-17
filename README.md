LDA-inline-c
============

smoothed Latent Dirichlet Allocation implemented with Python and inline C++ using Scipy.weave.

This code is for only learning LDA. **Useless, and not for application.**

このコードは学習用です。実用には使えません。

## 必要なファイル - Requirements

You need to download a corpus on [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bag+of+Words).
I experimented with the NIPS corpus.

[UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bag+of+Words)から、コーパスのダウンロードが必要です。私は NIPS corpusを使いました。

## 実行方法 - Run

On the terminal, type the following command after cloning the repository.

リポジトリをcloneした後、ターミナル上で、以下のコマンドを実行して下さい。

    ./lda.py

## ファイル情報 - File imformation

`lda.py` is the main code. It calls three C++ source files, `*.embed.cpp`, to be embedded.

`lda.py`がメインとなるコードです。そこから、埋め込み用の３つの`*.embed.cpp`という名前のC++のソースが呼び出されてます。

## 参考 - References

- Taro Tezuka, 2010, [LDA (Latent Dirichlet Allocation)の更新式の導出](http://yattemiyou.net/docs/lda_gibbs.pdf)
- id:n_shuyo, 2011, [Latent Dirichlet Allocations の Python 実装](http://d.hatena.ne.jp/n_shuyo/20110214/lda)
