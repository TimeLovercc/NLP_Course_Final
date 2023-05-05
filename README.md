
## Structure
```
root-
    |-fakenewsnet_dataset
	|-gossipcop
        |-politifact
    |-FakeNewsNet-master
    |-main.py
    |-main_ssl.py
    |-main_ttt.py
    |-README.md
    |-requirements.txt
    |-search_ssl_imdb.sh
    |-search_ssl.sh
    |-search_ttt_imdb.sh
    |-search_ttt.sh	
```


## Requirements

```
torch==2.0.0
python=3.9.6
datasets==2.11.0
jieba3k==0.35.1
nltk==3.8.1
numpy==1.24.2
pandas==2.0.0
huggingface-hub==0.13.4
```

## Run the code
The file `main.py` is the code to reproduce the results of the Bert in our report. The file `main_ssl.py` is the code to reproduce the results of the Bert-SSL in our report. The file `main_ttt.py` is the code to reproduce the results of the Bert-TTT in our report.

Here is an example to run the experiment for Bert-TTT on the fake news dataset `Fake`.
```
python main_ttt.py \
        --weight_decay 0.001 --lr 0.01 --weight 0.7 --mask_ratio 0.05 --dataset fake
```


## Data Set
1. Fake is stored in `./fakenewsnet_dataset/`. 
You can run the code to get data in the `https://github.com/KaiDMML/FakeNewsNet`. 

```
@article{shu2020fakenewsnet,
  title={Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media},
  author={Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  journal={Big data},
  volume={8},
  number={3},
  pages={171--188},
  year={2020},
  publisher={Mary Ann Liebert, Inc., publishers 140 Huguenot Street, 3rd Floor New~…}
}
```
2. IMDB is loaded from the HuggingFace datasets library.

You can find the details of it in the website `https://huggingface.co/datasets/imdb`.
