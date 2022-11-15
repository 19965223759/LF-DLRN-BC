import csv

import sklearn
import torch
from sklearn.metrics import accuracy_score

from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import time

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#1.Data enhancement
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

#2 Load data set
dataset = '.\data/'
train_directory = os.path.join(dataset, 'train_T4')
valid_directory = os.path.join(dataset, 'valid_T4')
test_directory = os.path.join(dataset , 'test_T4')
batch_size = 32
num_classes = 2

data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
     #imagefolder（root, transform），前者是图片路径，后者是对图片的变换，生成的数据类型是dataset
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['valid'])
}
# Size of training set
train_data_size = len(data['train'])
# Size of validation set
valid_data_size = len(data['valid'])
# Size of test set
test_data_size = len(data['test'])

# Unbalanced treatment
# Count = [0.697 , 0.303]
# weight = [1/0.697 , 1 / 0.303]
weight = [1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
          3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,
          3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,
          3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,
          3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,
          3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,
          3.30,3.30,3.30,3.30,3.30,3.30]
# sampler_val = [1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
#           1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
#           1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
#           1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,1.43,
#           1.43,1.43,1.43,1.43,
#           3.30, 3.30, 3.30, 3.30, 3.30, 3.30, 3.30, 3.30, 3.30, 3.30,
#           3.30, 3.30, 3.30, 3.30, 3.30, 3.30
# ]

weight1 = [1.36,1.36,1.36,1.36,1.36,1.36,1.36,
          1.36,1.36,1.36,1.36,1.36,1.36,1.36,
          1.36,1.36,1.36,1.36,1.36,1.36,1.36,
          1.36,1.36,1.36,1.36,1.36,1.36,1.36,
          1.36,1.36,1.36,1.36,1.36,1.36,1.36,
          1.36,1.36,1.36,1.36,1.36,1.36,1.36,
          1.36,1.36,3.75,3.75,3.75,3.75,3.75,
          3.75,3.75,3.75,3.75,3.75,3.75,3.75,
          3.75,3.75,3.75,3.75]

sampler = WeightedRandomSampler(weight , num_samples = 185, replacement=True)
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=False,sampler = sampler)

valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=False)

print("T4 training set data amount is：{}，validation set data amount is：{},validation set data amount is: {}".format(train_data_size, valid_data_size,test_data_size))

#3 Load model, transfer learning
densenet161 = models.densenet161(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
densenet161.to(device)
for param in densenet161.parameters():
    param.requires_grad = False
fc_inputs = densenet161.classifier.in_features
densenet161.classifier = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1)
)

# Define the loss function and optimizer.
loss_func = nn.NLLLoss()
# loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet161.parameters())

#4.Training model and validation model
def train_and_valid(model, loss_function, optimizer, epochs=25):
    history = []
    best_acc = 0.0
    best_loss = 1.0
    best_epoch = 0

    for epoch in range(epochs):
        tensor_list = []
        predict_list = []
        label_list = []
        tensor_list1 = []
        predict1 = []
        label1 = []

        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):  # 训练数据
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss = (loss-0.1).abs() + 0.1
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

            outputs = outputs.cpu().detach().numpy()
            tensor_list.append(outputs)

            predictions = predictions.cpu().detach().numpy()
            predict_list.append(predictions)

            labels = labels.cpu().detach().numpy()
            label_list.append(labels)
        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs2 = model(inputs)
                loss = loss_function(outputs2, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs2.data, 1)

                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

                outputs2 = outputs2.cpu().detach().numpy()
                tensor_list1.append(outputs2)

                predictions = predictions.cpu().detach().numpy()
                predict1.append(predictions)

                labels = labels.cpu().detach().numpy()
                label1.append(labels)
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        # if best_acc < avg_valid_acc:
        #     best_acc = avg_valid_acc
        #     best_epoch = epoch + 1
        if best_loss > avg_valid_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), '.\dataset1' + '_model_' + str(epoch+1) + '.pth')
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        # print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        print("Best loss for validation : {:.4f} at epoch {:03d}".format(best_loss, best_epoch))
        # outputs = np.concatenate(tensor_list, axis=0)
        # print(outputs1.shape)
        # print(outputs)
        predict = np.concatenate(predict_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        matrix = sklearn.metrics.confusion_matrix(label, predict, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        auc = sklearn.metrics.roc_auc_score(label, predict)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        acc1 = accuracy_score(label, predict)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        print('sensitivity: ', sen)
        print('specificity: ', spe)
        print('auc: ', auc)
        print('acc: ', acc1)
        print('ppv:', ppv)
        print('npv:', npv)
        print('Prediction of training Set T4：\n', predict)
        # outputs1 = np.concatenate(outputs1, axis=0)
        # print(outputs1.shape)
        # print(outputs1)
        predict4 = np.concatenate(predict1, axis=0)
        label4 = np.concatenate(label1, axis=0)
        matrix = sklearn.metrics.confusion_matrix(label4, predict4, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        auc = sklearn.metrics.roc_auc_score(label4, predict4)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        acc1 = accuracy_score(label4, predict4)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        print('sensitivity: ', sen)
        print('specificity: ', spe)
        print('auc: ', auc)
        print('acc: ', acc1)
        print('ppv:', ppv)
        print('npv:', npv)
        print('Prediction of validation Set T4：\n', predict4)
    return model, history

istrain=1
if istrain:
    num_epochs = 1000
    trained_model, history = train_and_valid(densenet161, loss_func, optimizer, num_epochs)
    history = np.array(history)
    print(history)
    print(history[:,0:2])
    with open("./T4-train-loss3.csv", "a", newline='', encoding='utf-8') as file:
      writer = csv.writer(file, delimiter=',')
      writer.writerows(history[:,0:1])
    with open("./T4-val-loss3.csv", "a", newline='', encoding='utf-8') as file:
      writer = csv.writer(file, delimiter=',')
      writer.writerows(history[:,1:2])

#6 Test model
def test_T4(model,loss_function):
    densenet161.load_state_dict(torch.load('.\dataset1' + '_model_' + '4' + '.pt'))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss = 0.0
    test_acc = 0.0
    tensor_list1 = []
    predict = []
    label = []
    test_start = time.time()
    with torch.no_grad():
        model.eval()
    for j, (inputs, labels) in enumerate(test_data):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs1 = model(inputs)
        # print(outputs1.shape)
        # print(labels)
        loss = loss_function(outputs1, labels)

        test_loss += loss.item() * inputs.size(0)

        ret, predictions = torch.max(outputs1.data, 1)

        correct_counts = predictions.eq(labels.data.view_as(predictions))

        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        test_acc += acc.item() * inputs.size(0)


        outputs1 = outputs1.cpu().detach().numpy()
        tensor_list1.append(outputs1)

        predictions = predictions.cpu().detach().numpy()
        predict.append(predictions)

        labels = labels.cpu().detach().numpy()
        label.append(labels)

    avg_test_loss = test_loss / test_data_size
    avg_test_acc = test_acc / test_data_size
    test_end = time.time()

    print(
        "test: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            avg_test_loss, avg_test_acc * 100,
            test_end - test_start
        ))

    return tensor_list1,predict,label

istest=1
if istest:
    outputs1, predict4, label = test_T4(densenet161, loss_func)
    # print(outputs1)
    outputs1 = np.concatenate(outputs1, axis=0)
    # print(outputs1.shape)
    print(outputs1)
    predict4 = np.concatenate(predict4, axis=0)
    label = np.concatenate(label, axis=0)
    matrix = sklearn.metrics.confusion_matrix(label, predict4, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    auc = sklearn.metrics.roc_auc_score(label, predict4)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc1 = accuracy_score(label, predict4)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    print('sensitivity: ', sen)
    print('specificity: ', spe)
    print('auc: ', auc)
    print('acc: ', acc1)
    print('ppv:', ppv)
    print('npv:', npv)
    print('Prediction of Test Set T4：\n', predict4)



