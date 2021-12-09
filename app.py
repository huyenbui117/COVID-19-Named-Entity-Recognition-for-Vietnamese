from transformers import AutoTokenizer, AutoModelForTokenClassification
from tasks import NER
import torch
from vncorenlp import VnCoreNLP
import pandas as pd

phobert_ner = AutoModelForTokenClassification.from_pretrained("huyenbui117/Covid19_phoNER")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

labels = NER().get_labels(None)

rdrsegmenter = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')


def run():
    df = pd.DataFrame(columns=["tokens", "predictions"])
    with open("sentences.txt", 'r', encoding="utf-8") as file:
        for line in file:
            sentences = rdrsegmenter.tokenize(line)
            for sentence in sentences:
                sequence = " ".join(sentence)  # tạo câu mới đã được tokenized

                input_ids = torch.tensor([tokenizer.encode(sequence)])  # lấy id của các tokens tương ứng
                # không dùng tokenize(decode(encode)), text sẽ bị lỗi khi tokenize do conflict với tokenizer mặc định
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence))  # lấy các token để đánh tags

                outputs = phobert_ner(input_ids).logits
                predictions = torch.argmax(outputs, dim=2)
                print(predictions)

                for i in [[token, labels[prediction]] for token, prediction in zip(tokens, predictions[0].numpy())]:
                    print(i)
                    a_series = pd.Series(i, index=df.columns)
                    df = df.append(a_series, ignore_index=True)
                print('---------------------------------------------')
    df.to_json("predict_sentences.json",orient="index")

