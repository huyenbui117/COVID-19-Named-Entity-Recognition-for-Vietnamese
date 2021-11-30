# COVID-19 Named Entity Recognition for Vietnamese

- [Dataset](https://github.com/phuonglt26/Vietnamese-E-commerce-Dataset)

## Methods


### Feature selection


### Model

## Setup

Download:
- [python 3.8.10](https://www.python.org/downloads/release/python-3810/), choose **Installer** versions, tick add
PATH when download process finished
- [java](https://www.oracle.com/java/technologies/downloads/), choose **Installer** versions, tick add
PATH when download process finished

```shell
git clone https://github.com/huyenbui117/COVID19_PhoNER
```

In the project directory

```shell
py -m pip install -r requirements.txt
py -m pip install vncorenlp
py -m pip install jupyter
py -m pip install seqeval //conda install seqeval
```

## Training

```shell
py evaluate.py 
```

## Evaluation



## Inference

- Input text in [data/text.csv](data/text.csv), check the appearance of aspects in the text by 1 or 0 in the aspect<sub>
  i</sub> collumn
- Note: aspect0, aspect1, aspect2, aspect3, aspect4, aspect5 are 'giá', 'dịch vụ', 'an toàn', 'chất lượng', 'ship', '
  chính hãng' respectively.
- Run

```shell
py main.py
```

- Example output:

- Output are then stored in 

