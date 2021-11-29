import logging
import os
from typing import List, TextIO, Union

from utils_ner import InputExample, Split, TokenClassificationTask

logger = logging.getLogger(__name__)


class NER(TokenClassificationTask):
    def __init__(self, label_idx=-1):
        self.label_idx = label_idx

    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        id_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=f"{mode}-{id_index}", words=words, labels=labels))
                        id_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[self.label_idx].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid=f"{mode}-{id_index}", words=words, labels=labels))
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                writer.write(line)
                if not preds_list[example_id]:
                    example_id += 1
            elif preds_list[example_id]:
                output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                writer.write(output_line)
            else:
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    def get_labels(self, path: str) -> List[str]:
        return ["O", "B-ORGANIZATION", "O-ORGANIZATION", "I-ORGANIZATION",
                "B-LOCATION", "I-LOCATION", "B-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE", "B-DATE", "I-DATE",
                "B-PATIENT_ID", "I-PATIENT_ID", "B-AGE", "I-AGE", "B-NAME", "I-NAME", "B-JOB", "I-JOB",
                "B-TRANSPORTATION", "I-TRANSPORTATION", "B-GENDER"]
