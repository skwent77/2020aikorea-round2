import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse

from dataloader import *
from evaluation import *
from model import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix

'''
data
 \_ train
   \_ 0.tif, ..., train.csv
 \_ test
   \_ 0.tif, ..., test.csv
 \_ validate(생성 필요)
   \_ 0.tif, ..., validate.csv
 \_ prediction.csv (train시 validate 예측값, test시 test 예측값 저장)

model
 \_ 1.pth, ...

 evaluation.py를 통해 예측값의 f1_socre 확인 가능

# 데이터, 모델 경로 수정 바랍니다.
'''

# 디렉토리 체크 및 생성
def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


# 로드할 model 찾는 함수
def find_model(model_type):
    if model_type in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
        model = EffNet(type=model_type, =True, num_classes=2)
    elif model_type == 'xception':
        model = Xception(num_classes=num_classes)
    return model 


def _infer(model, cuda, data_loader):
    res_fc = None
    
    for index, data in enumerate(data_loader):
        if (index+1) % (len(data_loader)//10) == 0 and (index+1) == len(data_loader):
            print(index + 1,'/',len(data_loader))
        image = data['image']
        label = data['label']

        if cuda:
            image = image.cuda()
        fc = model(image)
        fc = fc.detach().cpu().numpy()

        if index == 0:
            res_fc = fc
        else:
            res_fc = np.concatenate((res_fc, fc), axis=0)

    res_cls = np.argmax(res_fc, axis=1)
    #print('res_cls{}\n{}'.format(res_cls.shape, res_cls))

    return res_cls


def _infer_ensemble(models, cuda, data_loader):
    res_fc = None
    total_res_fc = None

    for idx in range(len(models[0])):
        print(f"[Ensemble model {idx+1} load]")
        model_type = models[0][idx]
        model_name = models[1][idx]
        
        model = find_model(model_type)
        load_model(model_name, model)

        if cuda:
            model = model.cuda()

        model.eval()
        for index, data in enumerate(data_loader):
            if (index+1) % (len(data_loader)//10) == 0 and (index+1) == len(data_loader):
                print(index + 1,'/',len(data_loader))
            image = data['image']
            label = data['label']

            if cuda:
                image = image.cuda()
            fc = model(image)
            fc = fc.detach().cpu().numpy()

            if index == 0:
                res_fc = fc
            else:
                res_fc = np.concatenate((res_fc, fc), axis=0)

        if idx == 0:
            total_res_fc = res_fc
        else:
            total_res_fc += res_fc
    
    res_cls = np.argmax(total_res_fc, axis=1)

    return res_cls


def feed_infer(output_file, infer_func):
    prediction_class = infer_func()

    print('write output')
    predictions_str = []

    with open(output_file, 'w') as file_writer:
        for pred in prediction_class:
            file_writer.write("{}\n".format(pred))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def validate(prediction_file, model, validate_dataloader, validate_label_file, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=validate_dataloader))
    
    pred = pd.read_csv(prediction_file, header=None)
    pred = pred[0].to_list()
    
    metric_result = f1_score(pred, validate_label_file)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result


def test(prediction_file, model, test_dataloader, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=test_dataloader))


def ensemble(prediction_file, models, test_dataloader, cuda):
    feed_infer(prediction_file, lambda : _infer_ensemble(models, cuda, data_loader=test_dataloader))


def save_model(model_name, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join(model_name + '.pth'))
    print('model saved')


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=2)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--print_iter", type=int, default=50)
    args.add_argument("--model_name", type=str, default="./model/100.pth")
    args.add_argument("--prediction_file", type=str, default="/home/workspace/user-workspace/gh-153/validation.csv")
    args.add_argument("--batch", type=int, default=64)
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--net_type", type=str, default="xception")
    args.add_argument("--model_save_dir", type=str, default="./model/")

    config = args.parse_args()

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode
    net_type = config.net_type
    save_path = config.model_save_dir

    # Data Augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    if mode == 'train':
        # create model
        model = find_model(net_type)

        if cuda:
            model = model.cuda()

        # define loss function
        loss_fn = nn.CrossEntropyLoss()
        if cuda:
            loss_fn = loss_fn.cuda()

        # set optimizer
        optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

        # get data loader
        train_csv = pd.read_csv('./data/train/train.csv')
        train_csv = shuffle(train_csv)
        train_csv, val_csv = train_test_split(train_csv, test_size=0.1)

        print(train_csv['label'].value_counts())
        train_data = NewDataset(train_csv, root='./data', phase='train', transform=transform_train)
        val_data = NewDataset(val_csv, root='./data', phase='train', transform=transform_test)
        
        print(f'number of train data : {len(train_data)}')
        print(f'number of validate data : {len(val_data)}')

        train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_data, batch_size=batch, shuffle=False, drop_last=True)
        
        val_csv.to_csv(f'{prediction_file[:-4]}_gt.csv')
        validate_label = val_csv['label'].to_list()
        
        time_ = datetime.datetime.now()
        num_batches = len(train_dataloader)
        
        #check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter : ",total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :",trainable_params)
        print("------------------------------------------------------------")
        
        # train
        max_val_acc = 0
        for epoch in range(num_epochs):
            model.train()
            correct = 0
            total = 0
            print(f'[Epoch {epoch+1}] lr: {optimizer.param_groups[0]["lr"]:.7f}')
            for iter_, data in enumerate(train_dataloader):
                # fetch train data
                image = data['image']
                label = data['label']
                
                if cuda:
                    image = image.cuda()
                    label = label.cuda()
                
                # update weight
                pred = model(image)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()

                _, predicted = torch.max(pred, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                acc = correct / total * 100

                loss.backward()
                optimizer.step()
                
                # Validation
                if (iter_+1) % print_iter == 0 or (iter_+1) == len(train_dataloader):
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    val_loss_mean = 0
                    val_pred = None
                    val_gt = None
                    for _ , data in enumerate(val_dataloader):
                        # fetch validate data
                        image = data['image']
                        label = data['label']
                        
                        if cuda:
                            image = image.cuda()
                            label = label.cuda()

                        pred = model(image)
                        loss = loss_fn(pred, label)
                        val_loss_mean += loss.data.cpu()

                        _, predicted = torch.max(pred, 1)
                        val_total += label.size(0)
                        val_correct += (predicted == label).sum().item()
 
                        if val_pred is None and val_gt is None:
                            val_pred = predicted.data.cpu()
                            val_gt = label.data.cpu()
                        else:
                            val_pred = torch.cat((val_pred, predicted.data.cpu()))
                            val_gt = torch.cat((val_gt, label.data.cpu()))
                            
                    val_loss_mean /= len(val_dataloader)
                    val_acc = val_correct / val_total * 100
                    print(f'[{iter_+1:3d}/{len(train_dataloader)}] train_loss: {loss:.5f}, train_acc: {acc:.3f}, val_loss: {val_loss_mean:.5f}, val_acc: {val_acc:.3f}, val_score: {f1_score(val_gt, val_pred):.5f}')
                    if (iter_+1) == len(train_dataloader):
                        print(f'[Confusion Matrix]\n{confusion_matrix(val_gt, val_pred)}')
                    
                    model.train()
            
            # for save validation.csv
            validate(prediction_file, model, val_dataloader, validate_label, cuda)
            
            # scheduler update
            scheduler.step()
        
            # save model
            check_dir(save_path)
            save_model(f'{save_path}/{epoch+1}', model, optimizer, scheduler)
            if val_acc >= max_val_acc:
                save_model(f'{save_path}/best', model, optimizer, scheduler)
                max_val_acc = val_acc
            
            elapsed = datetime.datetime.now() - time_
            print(f'[epoch {epoch + 1}] elapsed: {elapsed}')

    elif mode == 'test' or mode == 'ensemble':

        # get data loader
        test_csv = pd.read_csv('./data/test/test.csv')
        test_data = NewDataset(test_csv, root='./data', phase='test', transform=transform_test)
        test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=False, drop_last=False)

        if mode == 'test':
            # create model
            model = find_model(net_type)
            load_model(model_name, model)

            if cuda:
                model = model.cuda()

            model.eval()
            test(prediction_file, model, test_dataloader, cuda)
        else:
            model_types = ['xception', 'xception', 'xception', 'b5', 'b5', 'b5']
            saved_model = ['./model/xception_0/best.pth', './model/xception_1/best.pth', './model/xception_2/best.pth', './model/15_b5/best.pth', './model/b5_0/best.pth', './model/b5_1/best.pth']
            models = [model_types, saved_model]
            
            ensemble(prediction_file, models, test_dataloader, cuda)
        # submit test result
