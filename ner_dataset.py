import datasets
from datasets import Dataset, DatasetInfo, Features, Value, ClassLabel, Sequence
import json

valid_labels = [
    "关注点_教材_少儿教育",
    "下一步行动",
    "关注点_上课形式_少儿教育",
    "客户问题",
    "异议_考虑一下",
    "关注点_师资_少儿教育",
    "同意加微信",
    "关注点_课程数量_少儿教育",
    "客户确认日期与时间",
    "异议_下次再说",
    "51talk_理念渗透",
    "关注点_上课内容_少儿教育",
    "销售提及加微信",
    "关注点_价格"
]

class NERDataset(Dataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            features=Features(
                {
                    "text": Value("string"),
                    "label": Sequence(
                        feature={
                            "entity_type": ClassLabel(num_classes = 14, names=valid_labels),
                            "start": Value("int32"),
                            "end": Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        data_file = r"D:\projects\BERT-NER-Pytorch\datasets\yiliang\train.json"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_file},
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                labels = []
                for entity_type, entity_list in data["label"].items():
                    for entity, spans in entity_list.items():
                        for start, end in spans:
                            labels.append({"entity_type": entity_type, "start": start, "end": end})
                yield data["id"], {"text": text, "label": labels}