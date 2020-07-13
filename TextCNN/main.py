import gensim
import torch
from torch.utils.data import DataLoader, TensorDataset
import nltk
from collections import Counter
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from model import TextCNN


def make_vocab(data_file,output_file):
    if os.path.exists(output_file):
        return None
    vocab2num = Counter()
    lengths = []
    for i in range(2):
        with open(data_file[i],'r',encoding='utf-8-sig')as f:
            for line in f.readlines():
                sentence = line.strip().lower()
                vocabs = nltk.wordpunct_tokenize(sentence)
                lengths.append(len(vocabs))
                for vocab in vocabs:
                    vocab2num[vocab] +=1
    with open(output_file,'w',encoding='utf-8')as f:
        f.write('%s\t10000000\n'%'<PAD>')
        for vocab,num in vocab2num.most_common():
            f.write('%s\t%s\n' %(vocab,num))
    print("Vocab Size of all train data {}".format(len(vocab2num)))
    print("Train Data Size {}".format(len(lengths)))
    print("Average Sentence Length {}".format(sum(lengths) / len(lengths)))
    print("Max Sentence Length {}".format(max(lengths)))

def convert_vocab_to_idx(file):
    with open(file, "r", encoding="utf-8") as fr:
        vocabs = [line.split()[0] for line in fr.readlines() if int(line.split()[1]) >= 1]
    vocab2idx = {vocab: idx for idx, vocab in enumerate(vocabs)}
    return vocab2idx

def load_data(data_file,vocab2idx):
    x_list = []
    y_list = []
    for i in range(2):
        count = 0
        with open(data_file[i],'r',encoding='utf-8')as f:
            for line in f.readlines():
                count+=1
                sentence = line.strip().lower()
                x = [vocab2idx.get(vocab) for vocab in nltk.wordpunct_tokenize(sentence) if vocab in vocab2idx]
                x = x[:max_length]
                n_pad = max_length - len(x)
                x = x + n_pad * [PAD]
                x_list.append(x)
            y_list = y_list + [i] *  count
    X = np.array(x_list,dtype=np.int64)
    Y = np.array(y_list,dtype=np.int64)
    return X,Y


def load_word_embedding(vocab2idx,word2vec):
    vocab_size = len(vocab2idx)
    embedding_size = word2vec.vector_size
    word_embedding = np.zeros((vocab_size,embedding_size),dtype=np.float32)
    idx2vocab = {idx:vocab for vocab,idx in vocab2idx.items()}
    for idx in range(1,vocab_size):
        vocab = idx2vocab[idx]
        try:
            word_embedding[idx] = word2vec[vocab]
        except KeyError:
            word_embedding[idx] = np.random.randn(embedding_size)
    return word_embedding





def train(model,device,optimizer,loss_func):
    for epoch in range(23):
        model.train()
        for batch_x,batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_out = model(batch_x)
            loss = loss_func(batch_out,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        y_true = []
        y_pred = []
        for batch_x,batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_pred = model(batch_x).argmax(dim=-1)
            for y in batch_pred.cpu().numpy():
                y_pred.append(y)
            for y in batch_y.cpu().numpy():
                y_true.append(y)
        accuracy = metrics.accuracy_score(y_true,y_pred)
        f1_score = metrics.f1_score(y_true,y_pred)
        auc = metrics.roc_auc_score(y_true,y_pred)
        print("epoch %d\nLoss:%.9f  Test_accuracy: %.9f  Test_f1_score: %.9f Test_auc: %.9f" %(epoch+1,loss,accuracy,f1_score,auc))


if __name__ == '__main__':
    data_file = ['./rt-polaritydata/rt-polarity-pos.txt', './rt-polaritydata/rt-polarity-neg.txt']
    vocab_output_file = './vocab.txt'
    max_length = 68
    PAD = 0
    model_name = 'GoogleNews-vectors-negative300.bin'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)
    vocab_file = make_vocab(data_file,vocab_output_file)
    vocab2idx = convert_vocab_to_idx(vocab_output_file)
    word_embedding = load_word_embedding(vocab2idx,word2vec)

    #训练集划分
    X,Y = load_data(data_file,vocab2idx)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    #模型准备
    model = TextCNN(word_embedding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    #模型训练
    train(model,device,optimizer,loss_func)
