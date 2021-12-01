# COVID-19 Named Entity Recognition for Vietnamese

- [Dataset](https://github.com/VinAIResearch/PhoNER_COVID19)

## Methods

We use PhoBERT - an improved version of RoBERTa to fine-tune for state-of-the-art performance

## Setup

### Download:
- [python 3.8.10](https://www.python.org/downloads/release/python-3810/), choose **Installer** versions, tick add
PATH when download process finished
- [java](https://www.oracle.com/java/technologies/downloads/), choose **Installer** versions, tick add
PATH when download process finished
- File [VnCoreNLP-1.1.1.jar](https://github.com/vncorenlp/VnCoreNLP/raw/master/VnCoreNLP-1.1.1.jar) (27MB)

```shell
git clone https://github.com/huyenbui117/COVID19_PhoNER
```

### In the project directory

```
py -m pip install -r requirements.txt
py -m pip install jupyter
py -m pip install seqeval //conda install seqeval
```

## Notes
In case the input texts are raw, i.e. without word segmentation, a word segmenter must be applied to produce word-segmented texts before feeding to PhoBERT. As PhoBERT employed the RDRSegmenter from VnCoreNLP to pre-process the pre-training data (including Vietnamese tone normalization and word and sentence segmentation), it is recommended to also use the same word segmenter for PhoBERT-based downstream applications w.r.t. the input raw texts.

### Installation

```
# Install the vncorenlp python wrapper
pip3 install vncorenlp

# Download VnCoreNLP-1.1.1.jar & its word segmentation component (i.e. RDRSegmenter) 
mkdir -p vncorenlp/models/wordsegmenter
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/ 
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```

VnCoreNLP-1.1.1.jar (27MB) and folder models must be placed in the same working folder, here is vncorenlp!

### Example usage
```
# See more details at: https://github.com/vncorenlp/VnCoreNLP

# Load rdrsegmenter from VnCoreNLP
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("/Absolute-path-to/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# Input 
text = ""

# To perform word (and sentence) segmentation
sentences = rdrsegmenter.tokenize(text) 
for sentence in sentences:
	print(" ".join(sentence))
```

## Config:

- In phobert_config.json: 
  - "data_dir": The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task
  - "cahe_dir": Where do you want to store the pretrained models downloaded from huggingface.co
  - "model_name_or_path": Path to pretrained model or model identifier from huggingface.co/models
  - "output_dir": The output directory where the model predictions and checkpoints will be written,
  - "max_seq_length": The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded
  - "num_train_epochs": int,
  - "per_device_train_batch_size": int,
  - "save_steps": int,
  - "seed": int,
  - "do_train": Whether to run training or not. This argument is not directly used by Trainer, it’s intended to be used by your training/evaluation scripts instead,
  - "do_eval": Whether to run evaluation on the validation set or not. Will be set to True if evaluation_strategy is different from "no". This argument is not directly used by Trainer, it’s intended to be used by your training/evaluation scripts instead.,
  - "do_predict": Whether to run predictions on the test set or not. This argument is not directly used by Trainer, it’s intended to be used by your training/evaluation scripts instead,
  - "eval_steps ": int
- For more training argument, you can see them [here](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments)

## Execution

```
py main.py phobert_config.json 
```

## Inference

- Example output: 
- Output are then stored in: suffix/test_result.txt
```
test_loss = 0.08466359227895737
test_report =               precision    recall  f1-score   support

           B       0.97      0.97      0.97     11599
           I       0.95      0.93      0.94      6969

   micro avg       0.96      0.95      0.96     18568
   macro avg       0.96      0.95      0.95     18568
weighted avg       0.96      0.95      0.96     18568

test_runtime = 767.4659
test_samples_per_second = 3.909
test_steps_per_second = 0.489
```
