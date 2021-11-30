# COVID19_PhoNER

## PhoBERT
1. PhoBERT là gì
- PhoBERT bao gồm Pho (phở là món ăn nổi tiếng của Việt Nam) và BERT (Bidirectional Encoder Representations from Transformers) có nghĩa là mô hình biểu diễn từ theo 2 chiều ứng dụng kỹ thuật của Transformer. PhoBERT là một pre-train model được huấn luyện dành riêng cho tiếng Việt. 
- Việc huấn luyện PhoBERT dựa trên kiến trúc và cách tiếp cận giống RoBERTa của Facebook được Facebook giới thiệu giữa năm 2019. Đây là một cái tiến so với BERT trước đây. 
- Tương tự như BERT, PhoBERT cũng có 2 phiên bản là PhoBERT_base với 12 transformers block và PhoBERT_large với 24 transformers block.
- PhoBERT được huấn luyện trên khoảng 20GB dữ liệu bao gồm khoảng 1GB Vietnamese Wikipedia corpus và 19GB còn lại lấy từ Vietnamese news corpus. Đây là một lượng dữ liệu khả ổn để train một mô hình như BERT.
- PhoBERT sử dụng RDRSegmenter của VnCoreNLP để tách từ cho dữ liệu đầu vào trước khi qua BPE encoder. Việc huấn luyện của BERT tiếp cận theo tư tưởng của RoBERTa, chỉ sử dụng task Masked Language Model để train, bỏ đi task Next Sentence Prediction.

2. Thực thi PhoBERT: Nếu ta feed 1 câu văn bản qua PhoBERT thì sẽ lấy ra được embedding output của cả câu sau block encoder cuối cùng. Và chúng ta sẽ sử dụng output đó để làm đặc trưng classify, cụ thể gồm có những bước sau:
- Bước 1: Tiền xử lý (preprocess) câu văn bản
- Bước 2: Word segment câu văn bản trước khi đưa vào PhoBert (do PhoBert yêu cầu)
- Bước 3: Tokenize bằng bộ Tokenizer của PhoBert. Chú ý rằng khi tokenize ta sẽ thêm 2 token đặc biệt là [CLS] và [SEP] vào đầu và cuối câu.
- Bước 4: Đưa câu văn bản đã được tokenize vào model kèm theo attention mask.
  - Mô hình PhoBERT tạo ra các biểu diễn từ từ quá trình ẩn các vị trí token một cách ngẫu nhiên trong câu input và dự báo chính chính từ đó ở output dựa trên bối cảnh là các từ xung quanh
  - Như vậy khi đã biết các từ xung quanh, chúng ta hoàn toàn có thể dự báo được từ phù hợp nhất với vị trí đã được masking.
- Bước 5: Lấy output đầu ra và lấy vector output đầu tiên (chính là ở vị trí token đặc biệt [CLS]) làm đặc trưng cho câu để train hoặc để predict (tuỳ phase).

3. Một vài ứng dụng chính của PhoBERT:
- Tìm từ đồng nghĩa, trái nghĩa, cùng nhóm dựa trên khoảng cách của từ trong không gian biểu diễn đa chiều.
- Xây dựng các véc tơ embedding cho các tác vụ NLP như sentiment analysis, phân loại văn bản, NER, POS, huấn luyện chatbot.
- Gợi ý từ khóa tìm kiếm trong các hệ thống search.
- Xây dựng các ứng dụng seq2seq như robot viết báo, tóm tắt văn bản, sinh câu ngẫu nhiên với ý nghĩa tương đồng.
4. Đọc hiểu code
4.1 class TokenClassificationDataset(Dataset) trong Utils_ner.py:
- features: một mảng các InputFeature (thuộc tính của input)
- pad_token_label_id: Tập trung vào giá trị mà nó bị bỏ qua trong công đoạn tính gradient
- Hàm __init__: có 2 khả năng:
+ if os.path.exists(cached_features_file) and not overwrite_cache:
Đọc dữ liệu từ cache nếu đã có file cache và không có ý định đè lên file cache trước đó
+ else:
Đọc dữ liệu từ dataset và lưu nó vào trong cache nếu có file cache trước đó thì sẽ ghi đè, bằng cách gán feature bằng 1 hàm tokenizer
- Hàm __len__: Trả về số lượng feature
- Hàm __getitem__(self, i) -> InputFeatures: Trả về feature thứ I trong mảng các InputFeature
4.2 class TFTokenClassificationDataset trong Utils_ner.py:
- features: một mảng các InputFeature (thuộc tính của input)
- pad_token_label_id: tương tự như class trên, gán giá trị mặc định là 100
- Hàm __init__: 
+ Vẫn gán feature bằng 1 hàm tokenizer như hàm __init__ của class TokenClassificationDataset
+ Hàm gen(): với từng token trong self.features, đưa ra 1 object gồm inputs_ids, attention_mask, token_type_ids (nếu có) và 1 label_ids tương ứng
+ Hàm if, else: Tạo ra các mảng của inputs_ids, attention_mask, token_type_ids để chuẩn bị cho mask attention
- Hàm get_dataset(self): Trả về dataset sau khi đã xác minh được số lượng các feature
- Hàm __len__: Trả về số lượng feature
- Hàm __getitem__(self, i) -> InputFeatures: Trả về feature thứ I trong mảng các InputFeature


## RoBertaForTokenClassification

1.	Overview
-	RoBertaForTokenClassification dùng model của RoBERTa với một lớp tuyến tính ở cuối cùng sau output của lớp ẩn cuối cùng, được dùng trong một số nhiệm vụ liên quan đến xử lý ngôn ngữ tự nhiên như Named-Entity-Recognition và Part-Of-Speech Tagging
-	Model được kế thừa từ PreTrainedModel
2.	Implementation
```
class RobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (config.classifier_dropout 
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict
            if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```
3. Các hàm trong class RobertaForTokenClassification
   - Hàm __init__(self, config): Hàm khởi tạo của class dùng lại một số tham số của config (biến của class RobertaConfig) như num_label (số lượng nhãn), classifier_dropout, hidden_size (kích cỡ lớp ẩn)
   - Tham số của hàm forward: 
     - input_ids: Mảng vị trí của các token trong từ điển
     - attention_mask: 1 mảng các số 0,1 để thực hiện quá trình attention và tránh mask lên các token padding (các token chỉ để điền vào cho đủ kích cỡ mảng)
     - token_type_ids: Dùng để xác định token nằm ở chuỗi câu thứ nhất (0) hay chuỗi câu thứ 2 (1)
     - position_ids: Mảng vị trí của các token trong chuỗi đầu vào
     - inputs_embeds: Tham số này có thể thay thế cho input_ids để truyền trực tiếp một biểu diễn từ đã được embedded
     - output_attentions: Trả về attention tensor của tất cả các lớp attention (true) hoặc không (false)
     - output_hidden_states: Trả về trạng thái của các lớp ẩn (true) hoặc không (false)
     - return_dict: Trả về 1 ModelOutput (true) hoặc một plain tuple (false)
     - labels: 1 mảng các nhãn phục vụ cho việc tính hàm mất mát
   - Hàm forward trả về một TokenClassifierOutput hoặc một tuple của torch.FloatTensor (phụ thuộc vào return_dict) với các tham số:
     - loss: Giá trị của hàm mất mát
     - logits: Classification scores (trước khi đưa vào SoftMax).
     - hidden_states (nếu output_hidden_states là true): Trạng thái (ẩn) của đầu ra của từng mảng và đầu ra của công đoạn embedding
     - attentions (nếu output_attentions là true): Đầu ra của attention sau khi đưa vào softmax


