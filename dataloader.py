import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import os
from PIL import Image
import pandas as pd
import numpy as np

def get_transform(method=Image.BILINEAR):
    transform_list = []

    transform_list.append(transforms.Resize((68,68)))
    transform_list.append(transforms.Grayscale())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5), (0.5)))
    
    return transforms.Compose(transform_list)


class CustomDataset(Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        
        self.label_path = os.path.join(root, self.phase, self.phase+'.csv')
        with open(self.label_path, 'r', encoding='utf-8-sig') as f:
            file_list = []
            label = []
            
            for line in f.readlines()[1:]:
                v = line.strip().split(',')
                file_list.append(v[0])
                if self.phase != 'test':
                    label.append(v[2])

        self.imgs = list(file_list)
        self.labels = list(label)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.phase, self.imgs[index])
        
        if self.phase != 'test':
            label = self.labels[index]
            label = torch.tensor(int(label))

        transform = get_transform()
        image = Image.open(image_path)
        image = transform(image)

        if self.phase != 'test' :
            return (image, label)
        elif self.phase == 'test' :
            dummy = ""
            return (image, dummy)

    def __len__(self):
        return len(self.imgs)

    def get_label_file(self):
        return self.label_path

class NewDataset(Dataset):
    def __init__(self, csv, root='./data', phase='train', transform=None):
        self.csv = csv
        self.root = f'{root}/{phase}'
        self.transform = transform
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name = self.csv.iloc[idx][0]
        title_name = self.csv.iloc[idx][1]
        label = self.csv.iloc[idx][2]
        
        image = Image.open(f'{self.root}/{file_name}')
        # image = np.asarray(image)
        # image = ToTensor()(image)
        # image = Image.fromarray(image)
        # image = torch.Tensor(image)
        # print(image)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'title': title_name}


def data_loader(root, phase='train', batch_size=16):
    phase = phase.lowercase()
    assert phase in ['train', 'test'], 'phase should be train/test !!'
    if phase == 'train':
        shuffle = True
        
    else:
        shuffle = False
        dataset = CustomDataset(root, phase)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader, dataset.get_label_file()


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    TRAIN_LOC = '/home/workspace/user-workspace/gh-153/data/train/train.csv'
    train_csv = pd.read_csv(f'{TRAIN_LOC}')
    train_data = NewDataset(train_csv, root='./data', phase='train', transform=transform_train)
    trainloader = DataLoader(train_data, batch_size=32, shuffle=False)

    data = next(iter(trainloader))

    print(data['image'])
    print(data['label'])
    print(data['title'])