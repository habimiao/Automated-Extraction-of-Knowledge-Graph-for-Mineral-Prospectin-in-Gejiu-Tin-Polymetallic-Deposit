# pytorch-NER
RoBERTa + BiLSTM + CRF for Gejiu mineral NER Task

## Requirement
- python 3.8
- pytorch 1.8.1
- transformers 4.11
- tqdm
- wandb

## CLUENER
fine-grained named entity recognition dataset and benchmark for chinese [[see also]](NER/README.md)

## zh-RoBERTa
use the pretrained RoBERTa weight for chinese from @brightmart and @ymcui [[see also]](resource/README.md)

## NER Run

```shell
python main.py
```

### configure what you want

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # device
lr = 1e-3  # learning_rate
batch_size = 128  # batch size
accumulation_steps = 1  # accumulation steps for gradient accumulation
roberta_path = 'resource/RoBERTa_zh_Large_PyTorch'  # path to PLM files

lstm_hidden_dim = 768  # hidden dim for BiLSTM
lstm_dropout_rate = 0.2  # dropout rate for BiLSTM
```


## Predict run
```shell
python predict.py
python predict_one_sentence.py
```




