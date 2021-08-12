# lab_transformer

## Branches: Everybody has one

__PLEASE ONLY WORK ON YOUR BRANCH!!!!!__ 


## Install instructions
__install with:__ 

```bash
conda env create --file environment.yml
conda activate pytorch_transformer
```
__and if data does not exist run the dataset (en-de, de-en) download:__

```bash
# cd is important, otherwise data goes into wrong folder
cd helpers 
python helpers/iwslt_setup.py 
```

## Data:
__Please Check before running training:__ The data should be in the same directory as the python script in which you are executing `torchtext.datasets.IWSLT.splits`, e.g.:

```
transformer-annotated
    |- transformer.py 
    |
    |- .data
        |- iwslt
            |- en-de.tgz
            |- de-en.tgz    
```

## The Annotated Transformer:
1. [blog post](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
2. [Code from](https://github.com/harvardnlp/annotated-transformer)

## Project specific
- [2021 DL Lab Project Group "Different Functionality, Different Optimization" - Google Slides](https://docs.google.com/presentation/d/1fF50j5xOnQ-COn8n_eC1UnehprtgSz1hkjHrbHNLQCQ/edit)


# Notes:

## copy files from remote to local:

```bash
scp -r fr_as1464@login.nemo.uni-freiburg.de:/work/ws/nemo/fr_as1464-transformer_work-0/transformer-main/experiments_save/<NAME_OF_EXPERIMENT> /home/mrrobot/PycharmProjects/transformer-main/experiments_save
```

## Start TENSORBOARD
```bash
# Within the project main folder:
tensorboard --logdir experiments_save/runs
```


## Bleu Score
- https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

## Transformer
- Paper link: https://arxiv.org/pdf/1706.03762.pdf
- Suggested theory: https://jalammar.github.io/illustrated-transformer/ 

### changes to Attention is all you need
Contains the implementation of the original transformer paper "Attention is all you need".

Paper link: https://arxiv.org/pdf/1706.03762.pdf

__Certain modifications:__
1. LayerNorm (before instead of after)
2. Dropout (Added additionally to attention weights and point-wise feed-forward net sublayer
