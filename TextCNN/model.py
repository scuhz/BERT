import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, word_embedding):
        super(TextCNN, self).__init__()
        self.word_embedding1 = nn.Embedding.from_pretrained(torch.Tensor(word_embedding), freeze=False)
        self.word_embedding2 = nn.Embedding.from_pretrained(torch.Tensor(word_embedding), freeze=True)
        self.conv1 = nn.Conv2d(2, 100, kernel_size=(3, 300))
        self.conv2 = nn.Conv2d(2, 100, kernel_size=(4, 300))
        self.conv3 = nn.Conv2d(2, 100, kernel_size=(5, 300))
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(300, 2)

    def conv_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(dim=3)
        #         print(x.shape)
        x = F.max_pool1d(x, x.size(2)).squeeze(dim=2)
        #         print(x.shape)
        return x

    def forward(self, x):
        embeds1 = self.word_embedding1(x)
        embeds2 = self.word_embedding2(x)
        #         print(embeds.shape)
        embeds = torch.stack((embeds1, embeds2), dim=1)
        #         print(embeds.shape)
        cnn_out1 = self.conv_pool(embeds, self.conv1)
        cnn_out2 = self.conv_pool(embeds, self.conv2)
        cnn_out3 = self.conv_pool(embeds, self.conv3)
        cnn_out = torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1)
        cnn_out = self.dropout(cnn_out)
        output = F.softmax(self.linear(cnn_out), dim=1)
        return output