""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_lens
torch.cuda.is_available()
def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a,list):
            example.text_a = " ".join(example.text_a)
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if len(label_ids) != max_seq_length:  ## 截断一下
            label_ids = label_ids[:max_seq_length]
        assert len(label_ids) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                      segment_ids=segment_ids, label_ids=label_ids))
    return features


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position','B-scene',"I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position','I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position',
                'S-scene','O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class SalerProcessor(DataProcessor):
    """Processor for the Saler ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-客户关注点-上课形式", "B-提出的询问内容", "B-客户异议-担心孩子跟不上", "B-销售接下来的行动", 'B-客户确认日期与时间', 'B-客户关注点-学习周期', 'B-销售提及添加微信',
                "I-客户关注点-上课形式", "I-提出的询问内容", "I-客户异议-担心孩子跟不上", "I-销售接下来的行动", 'I-客户确认日期与时间', 'I-客户关注点-学习周期', 'I-销售提及添加微信',
                "S-客户关注点-上课形式", "S-提出的询问内容", "S-客户异议-担心孩子跟不上", "S-销售接下来的行动", 'S-客户确认日期与时间', 'S-客户关注点-学习周期', 'S-销售提及添加微信',
                'O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class YiLiangProcessor_14labels(DataProcessor):
    """Processor for the Saler ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X",
                "B-客户问题", "I-客户问题", "S-客户问题",
                "B-销售提及加微信", "I-销售提及加微信", "S-销售提及加微信",
                "B-异议_考虑一下", "I-异议_考虑一下", "S-异议_考虑一下",
                "B-关注点_上课形式_少儿教育", "I-关注点_上课形式_少儿教育", "S-关注点_上课形式_少儿教育",
                "B-客户确认日期与时间", "I-客户确认日期与时间", "S-客户确认日期与时间",
                "B-同意加微信", "I-同意加微信", "S-同意加微信",
                "B-关注点_上课内容_少儿教育", "I-关注点_上课内容_少儿教育", "S-关注点_上课内容_少儿教育",
                "B-51talk_理念渗透", "I-51talk_理念渗透", "S-51talk_理念渗透",
                "B-异议_下次再说", "I-异议_下次再说", "S-异议_下次再说",
                "B-关注点_课程数量_少儿教育", "I-关注点_课程数量_少儿教育", "S-关注点_课程数量_少儿教育",
                "B-关注点_价格", "I-关注点_价格", "S-关注点_价格",
                "B-下一步行动", "I-下一步行动", "S-下一步行动",
                "B-关注点_师资_少儿教育", "I-关注点_师资_少儿教育", "S-关注点_师资_少儿教育",
                "B-关注点_教材_少儿教育", "I-关注点_教材_少儿教育", "S-关注点_教材_少儿教育", 
                "O", "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class YiLiangProcessor_28labels(DataProcessor):
    """Processor for the Saler ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [
            "X",
            "B-关注点_价格",
            "I-关注点_价格",
            "S-关注点_价格",
            "B-51talk_理念渗透",
            "I-51talk_理念渗透",
            "S-51talk_理念渗透",
            "B-关注点_上课内容_少儿教育",
            "I-关注点_上课内容_少儿教育",
            "S-关注点_上课内容_少儿教育",
            "B-客户问题",
            "I-客户问题",
            "S-客户问题",
            "B-51talk_自报家门",
            "I-51talk_自报家门",
            "S-51talk_自报家门",
            "B-异议_考虑一下",
            "I-异议_考虑一下",
            "S-异议_考虑一下",
            "B-销售提及加微信",
            "I-销售提及加微信",
            "S-销售提及加微信",
            "B-同意加微信",
            "I-同意加微信",
            "S-同意加微信",
            "B-下一步行动",
            "I-下一步行动",
            "S-下一步行动",
            "B-异议_下次再说",
            "I-异议_下次再说",
            "S-异议_下次再说",
            "B-关注点_询问地址_通用",
            "I-关注点_询问地址_通用",
            "S-关注点_询问地址_通用",
            "B-异议_价格异议",
            "I-异议_价格异议",
            "S-异议_价格异议",
            "B-51talk_确认跟进时间",
            "I-51talk_确认跟进时间",
            "S-51talk_确认跟进时间",
            "B-客户确认日期与时间",
            "I-客户确认日期与时间",
            "S-客户确认日期与时间",
            "B-关注点_上课形式_少儿教育",
            "I-关注点_上课形式_少儿教育",
            "S-关注点_上课形式_少儿教育",
            "B-异议_孩子跟不上_少儿教育",
            "I-异议_孩子跟不上_少儿教育",
            "S-异议_孩子跟不上_少儿教育",
            "B-需求_有报班意向_25752",
            "I-需求_有报班意向_25752",
            "S-需求_有报班意向_25752",
            "B-异议_孩子不愿意学_少儿教育",
            "I-异议_孩子不愿意学_少儿教育",
            "S-异议_孩子不愿意学_少儿教育",
            "B-关注点_课程数量_少儿教育",
            "I-关注点_课程数量_少儿教育",
            "S-关注点_课程数量_少儿教育",
            "B-51talk_家长试听效果_好",
            "I-51talk_家长试听效果_好",
            "S-51talk_家长试听效果_好",
            "B-51talk_提问授权",
            "I-51talk_提问授权",
            "S-51talk_提问授权",
            "B-51talk_了解客户需求",
            "I-51talk_了解客户需求",
            "S-51talk_了解客户需求",
            "B-关注点_教材_少儿教育",
            "I-关注点_教材_少儿教育",
            "S-关注点_教材_少儿教育",
            "O",
            "[START]",
            "[END]",
        ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


ner_processors = {
    "cner": CnerProcessor,
    'cluener':CluenerProcessor,
    'saler' :SalerProcessor,
    "yiliang" :YiLiangProcessor_28labels,
    "yiliang_multigual" :YiLiangProcessor_28labels
}
