# -*- coding: UTF-8 -*-
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(0)
# device = torch.device("cuda:1")
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
import datetime
import utils

# Hyper Parameters
num_epochs = 30
batch_size = 64
learning_rate = 0.002

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    start_time = datetime.datetime.now()
    cnn = CNN().to(device)
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # images = Variable(images)
            # labels = Variable(labels.float())
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
    print("save last model")
    end_time = datetime.datetime.now()
    utils.log_cost(start_time, end_time)

if __name__ == '__main__':
    main()


