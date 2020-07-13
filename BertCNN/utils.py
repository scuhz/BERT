import torch
from torch.utils.data import DataLoader, RandomSampler,TensorDataset


class InputExample(object):

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):

    def __init__(self,input_ids,input_mask,segment_ids,label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer):
    label_map = {label:i for i,label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" %(ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ['[CLS]'] + tokens_a +['[SEP]']
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] *(max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label_map[example.label]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeature(input_ids,input_mask,segment_ids,label_id))

    return features


def convert_features_to_tensors(features,batch_size):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long
    )

    data =TensorDataset(all_input_ids,all_input_mask,all_segment_ids,all_label_ids)

    sample = RandomSampler(data)
    dataloader = DataLoader(data,sampler=sample,batch_size=batch_size)
    return dataloader


def load_data(data_dir,tokenizer,processor,max_length,batch_size,data_type):
    label_list = processor.get_labels()

    if data_type == 'train':
        examples = processor.get_train_examples(data_dir)
    elif data_type == 'test':
        examples = processor.get_test_examples(data_dir)
    features = convert_examples_to_features(examples,label_list,max_length,tokenizer)

    dataloader =convert_features_to_tensors(features,batch_size)
    examples_length = len(examples)
    return dataloader, examples_length