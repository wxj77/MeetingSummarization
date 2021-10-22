## End-to-End Abstractive Summarization for Meetings (HMNet)
Implements the model described in the following paper [End-to-End Abstractive Summarization for Meetings](https://arxiv.org/pdf/2004.02016.pdf).

```
@article{zhu2020end,
  title={End-to-End Abstractive Summarization for Meetings},
  author={Zhu, Chenguang and Xu, Ruochen and Zeng, Michael and Huang, Xuedong},
  journal={arXiv preprint arXiv:2004.02016},
  year={2020}
}
```

### Differences from Paper Implementation
 
```
1. I used CNN in Transformer-PositionwiseFeedForward.
2. Role vectors, pos tags, and named entity tags are not used. 
(When using a role vector, performance was lower than not utilizing role vector. So please do not hesitate to advise me about this.)
```

### Preparation
 
```
1. Switch into the folder secondary_drive/HMNet-End-to-End-Abstractive-Summarization-for-Meetings
2. Run conda activate hmnet-simplified-new
(When using a role vector, performance was lower than not utilizing role vector. So please do not hesitate to advise me about this.)
```

### Train
```
Adjust the hyperparameters in config/hparams.py
python main.py --mode train --save_path path_to_save_the_model
```

### Evaluation
```
python main.py --mode eval --model_path trained_model_path --gen_max_length 500
[Andreas]: In order to just run the latest checkpoint, adjust the evaluation loop in main.py. E.g. if you trained with 200 epochs, change the loop to for i in range(int(200), 201): to just load the 200th checkpoint.
```

| Epoch | Rouge-1 | Rouge-2 | Rouge-L |
|:-----:|:-------:|:-------:|:-------:|
|   30  |  0.4762 |  0.1862 |  0.1767 |
|   40  |  0.4796 |  0.1935 |  0.1858 |


### Contact
- jude.lee@kakaocorp.com
