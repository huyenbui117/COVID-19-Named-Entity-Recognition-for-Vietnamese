# COVID-19 Named Entity Recognition for Vietnamese

- [Dataset](https://github.com/phuonglt26/Vietnamese-E-commerce-Dataset)
## Xử lý data:

- In phobert_config.json:
  - "data_dir": The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task
  - "cahe_dir": Where do you want to store the pretrained models downloaded from huggingface.co
  - "model_name_or_path": Path to pretrained model or model identifier from huggingface.co/models
  - "output_dir": suffix,
  - "max_seq_length": The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded
  - "num_train_epochs": 5,
  - "per_device_train_batch_size": 32,
  - "save_steps": 750,
  - "seed": 1,
  - "do_train": false,
  - "do_eval": true,
  - "do_predict": true,
  - "eval_steps ": 10
- class ModelArguments (Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.)


    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

- class DataTrainingArguments (Arguments pertaining to what data we are going to input our model for training and eval)

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

## Methods


### Feature selection


### Model


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

### Notes
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

## Training

```shell
py evaluate.py 
```

## Evaluation



## Inference

- Run

```shell
py main.py
```

- Example output:

- Output are then stored in 

