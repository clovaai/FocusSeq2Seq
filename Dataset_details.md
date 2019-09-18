# Source of data files

## SQuAD Question generation (`./squad`)

* `dev.txt.shuffle.dev.bio`,
`dev.txt.shuffle.dev.case`,
`dev.txt.shuffle.dev.ner`,
`dev.txt.shuffle.dev.pos`,
`dev.txt.shuffle.dev.source.txt`,
`dev.txt.shuffle.dev.target.txt`,
`dev.txt.shuffle.test.bio`,
`dev.txt.shuffle.test.case`,
`dev.txt.shuffle.test.ner`,
`dev.txt.shuffle.test.pos`,
`dev.txt.shuffle.test.source.txt`,
`dev.txt.shuffle.test.target.txt`,
`train.txt.bio`,
`train.txt.case`,
`train.txt.ner`,
`train.txt.pos`,
`train.txt.source.txt`,
`train.txt.target.txt`
are from [redistribute.zip](https://res.qyzhou.me/redistribute.zip) shared by [Zhou et al.](https://arxiv.org/abs/1704.01792).

* `vocab.txt` is created by running scripts in [NQG++ official repo](https://github.com/magic282/NQG).

## CNN-DM Abstract summarization (`./cnndm`)

* `train.txt.src`,
`train.txt.tgt.tagged`,
`val.txt.src`,
`val.txt.tgt.tagged`,
`test.txt.src`,
`test.txt.tgt.tagged`
are from [cnndm.tar.gz](https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz) shared by [Harvard NLP](https://github.com/harvardnlp/sent-summary).
(`*.tagged` files contains target for content selection used in [Gehrmann et al.](https://arxiv.org/abs/1808.10792)).

* `vocab` is created by running scripts in [cnn-dm preprocessing repo](https://github.com/abisee/cnn-dailymail) shared by [See et al.](https://arxiv.org/abs/1704.04368).

## GloVe word embedding ( `./glove`)

* `glove.6B.300d.txt` is from [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) shared by [Stanford NLP](https://nlp.stanford.edu/projects/glove/).
