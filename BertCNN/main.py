from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertAdam
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
from utils import InputExample,load_data
from BertCNN import BertCNN
import random
import os
import time
from sklearn import metrics
from tqdm import tqdm


class MRProcessor(object):

    def get_train_examples(self,lines):
        return self._create_examples(lines,'train')

    def get_test_examples(self,lines):
        return self._create_examples(lines,'test')

    def get_labels(self):
        return ['0','1']

    def _create_examples(self,lines,set_type):
        examples = []
        end = int(len(lines)* 0.9)
        if set_type == 'train':
            tmp_data = lines[:end]
        else:
            tmp_data = lines[end:]
        for i,item in enumerate(tmp_data):
            text_a = item[0]
            label = item[1]
            guid = "%s-%s" %(set_type,i)
            examples.append(
                InputExample(guid=guid,text_a=text_a,text_b=None,label=label))
        return examples

    def _read_data(self, files):
        lines = []
        for file in files:
            if 'neg' in file:
                label = '1'
            elif 'pos' in file:
                label = '0'
            with open(file,'r',encoding='utf-8-sig')as f:
                for line in f.readlines():
                    sentence = line.strip().lower()
                    tmp = (sentence,label)
                    lines.append(tmp)
        random.shuffle(lines)
        return lines


def classifiction_metric(preds, labels, label_list):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

    acc = metrics.accuracy_score(labels, preds)

    labels_list = [i for i in range(len(label_list))]

    report = metrics.classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5,
                                           output_dict=True)

    if len(label_list) > 2:
        auc = 0.5
    else:
        auc = metrics.roc_auc_score(labels, preds)
    return acc, report, auc



def evaluate(model, dataloader, criterion, device, label_list):

    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    epoch_loss = 0

    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Eval"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        epoch_loss += loss.mean().item()

    acc, report, auc = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss/len(dataloader), acc, report, auc


def train(epoch_num, model, train_loader, test_loader, optimizer, criterion,
          gradient_accumulation_steps, device, label_list,output_model_file,output_config_file, log_dir):
    early_stop_times = 0
    print_step = 200
    writer = SummaryWriter(log_dir=log_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())))
    best_test_loss = float('inf')

    global_step = 0
    for epoch in range(epoch_num):

        if early_stop_times >= 50:
            break

        print(f'---------------- Epoch: {epoch+1:02} ----------')
        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([],dtype=int)
        all_labels = np.array([],dtype=int)

        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask,segment_ids,label_ids = batch

            output = model(input_ids,segment_ids,input_mask, labels= None)
            loss = criterion(output.view(-1,len(label_list)),label_ids.view(-1))
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            train_steps += 1

            loss.backward()

            # 用于画图和分析的数据
            epoch_loss += loss.item()
            preds = output.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)
            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % print_step == 0 and global_step != 0:

                    """ 打印Train此时的信息 """
                    train_loss = epoch_loss / train_steps
                    train_acc, train_report, train_auc = classifiction_metric(all_preds, all_labels, label_list)

                    test_loss, test_acc, test_report, test_auc = evaluate(model, test_loader, criterion, device,
                                                                      label_list)

                    c = global_step // print_step
                    writer.add_scalar("loss/train", train_loss, c)
                    writer.add_scalar("loss/test", test_loss, c)

                    writer.add_scalar("acc/train", train_acc, c)
                    writer.add_scalar("acc/test", test_acc, c)

                    writer.add_scalar("auc/train", train_auc, c)
                    writer.add_scalar("auc/test", test_auc, c)

                    for label in label_list:
                        writer.add_scalar(label + ":" + "f1/train", train_report[label]['f1-score'], c)
                        writer.add_scalar(label + ":" + "f1/test",
                                          test_report[label]['f1-score'], c)

                    print_list = ['macro avg', 'weighted avg']
                    for label in print_list:
                        writer.add_scalar(label + ":" + "f1/train",
                                          train_report[label]['f1-score'], c)
                        writer.add_scalar(label + ":" + "f1/test",
                                          test_report[label]['f1-score'], c)
                    for label in label_list:
                        writer.add_scalar(label + ":" + "f1/train", train_report[label]['f1-score'], c)
                        writer.add_scalar(label + ":" + "f1/test", test_report[label]['f1-score'], c)

                    print_list = ['macro avg', 'weighted avg']
                    for label in print_list:
                        writer.add_scalar(label + ":" + "f1/train",
                                        train_report[label]['f1-score'], c)
                        writer.add_scalar(label + ":" + "f1/test",
                                        test_report[label]['f1-score'], c)

                    if test_loss < best_test_loss:
                        best_test_loss = test_loss

                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        torch.save(model_to_save.state_dict(), output_model_file)
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())

                        early_stop_times = 0
                    else:
                        early_stop_times += 1

    writer.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_vocab_file = './Source/bert-base-uncased-vocab.txt'
    bert_model_path = './Source/bert-base-uncased'
    output_dir = "./Result/Output/"
    cache_dir = "./Result/Cache/"
    log_dir = "./Result/Log/"
    max_seq_length = 70
    gradient_accumulation_steps = 8
    num_train_epoches = 5
    train_batch_size = 32 // gradient_accumulation_steps
    test_batch_size = 16


    if not os.path.exists(output_dir):
        os.makedirs(output_dir )

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    #Bert模型输出文件
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)


    myProcessor = MRProcessor()
    tokenizer =BertTokenizer.from_pretrained(
        bert_vocab_file, do_lower_case=True)
    label_list =myProcessor.get_labels()
    num_labels = len(label_list)
    files = [os.path.join("./rt-polaritydata", 'rt-polarity-pos.txt'), os.path.join("./rt-polaritydata", 'rt-polarity-neg.txt')]
    lines = myProcessor._read_data(files)
    train_dataloader, train_examples_len = load_data(
        lines, tokenizer, myProcessor, max_seq_length, train_batch_size, "train")
    test_dataloader, _ = load_data(
        lines, tokenizer, myProcessor, max_seq_length, test_batch_size, "test")

    num_train_optimization_steps = int(
        train_examples_len / train_batch_size / gradient_accumulation_steps) * num_train_epoches
    model = BertCNN.from_pretrained(
        bert_model_path, cache_dir= cache_dir, num_labels=num_labels, n_filters=100,filter_sizes=[3,4,5]
    )
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay =['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=3e-5,
                         warmup=0.1,
                         t_total=num_train_optimization_steps)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    train(1, model, train_dataloader, test_dataloader, optimizer,
          criterion, gradient_accumulation_steps, device, label_list, output_model_file, output_config_file,log_dir)

    """ Test """

    # 加载模型
    bert_config = BertConfig(output_config_file)

    model = BertCNN(bert_config, num_labels=num_labels,
                        n_filters=100, filter_sizes=[3,4,5])
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    # test
    test_loss, test_acc, test_report, dev_auc = evaluate(
        model, test_dataloader, criterion, device, label_list)
    print("-------------- Test -------------")
    print(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc * 100: .3f} % | AUC:{dev_auc}')

    for label in label_list:
        print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
            label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
    print_list = ['macro avg', 'weighted avg']

    for label in print_list:
        print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
            label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))


if __name__ == '__main__':
    main()