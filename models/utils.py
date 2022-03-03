import os
import re
import pickle
from skimage import io
from PIL import Image
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler

class RafFaceDataset(Dataset):
    """
    RAF-Face dataset for Face Expression Recognition
    """
    def __init__(self, train=True, data_path='../dataset/basic/', transform=None):
      manual_annotation_dir = os.path.join(data_path, 'Annotation/manual')
      emotion_label_txt_path = os.path.join(data_path, 'EmoLabel/list_patition_label.txt')

      emotion_dict = dict(np.loadtxt(emotion_label_txt_path, dtype=np.str))

      if train:
        face_files=[];genders=[];races=[];ages=[];emotions=[];ldmks=[]
        for file in os.listdir(manual_annotation_dir):
          if file.startswith('train_'):
            face_fname = file.replace('_manu_attri', '_aligned').replace('.txt', '.jpg')
            face_files.append(os.path.join(data_path, 'Image/aligned/'+face_fname))
            with open(os.path.join(manual_annotation_dir, file), mode='rt') as f:
              manu_info_list = f.readlines()
            genders.append(int(manu_info_list[5]))
            races.append(int(manu_info_list[6]))
            ages.append(int(manu_info_list[7]))
            emotions.append(int(emotion_dict[face_fname.replace('_aligned', '')].strip())-1)
            ldmks.append(np.array([[[float(_.replace('\n', ''))] for _ in re.split('\t| ',line)] for line in
                                     manu_info_list[0:5]]).flatten().tolist())
      else:
        face_files=[];genders=[];races=[];ages=[];emotions=[];ldmks=[]
        for file in os.listdir(manual_annotation_dir):
          if file.startswith('test_'):
            face_fname = file.replace('_manu_attri', '_aligned').replace('.txt', '.jpg')
            face_files.append(os.path.join(data_path, 'Image/aligned/'+face_fname))
            with open(os.path.join(manual_annotation_dir, file), mode='rt') as f:
                  manu_info_list = f.readlines()
            genders.append(int(manu_info_list[5]))
            races.append(int(manu_info_list[6]))
            ages.append(int(manu_info_list[7]))
            emotions.append(int(emotion_dict[face_fname.replace('_aligned', '')].strip())-1)
            ldmks.append(np.array([[[float(_.replace('\n', ''))] for _ in re.split('\t| ',line)] for line in
                                     manu_info_list[0:5]]).flatten().tolist())

      self.face_files = face_files
      self.genders = genders
      self.races = races
      self.ages = ages
      self.emotions = emotions
      self.ldmks = ldmks

      self.transform = transform

    def __len__(self):
      return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(self.face_files[idx])
        gender = self.genders[idx]
        race = self.races[idx]
        age = self.ages[idx]
        emotion = self.emotions[idx]
        ldmk = self.ldmks[idx]

        sample = {'image': image, 'gender': gender, 'race': race, 'age': age, 'emotion': emotion,
                  'landmark': np.array(ldmk), 'filename': self.face_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
        
def load_data(num_workers=2,batch_size=32,data_path='../dataset/basic/'):
    """
    load dataset
    :param dataset_name:
    :return:
    """

    print('loading %s dataset...' % 'RAF-DB')
    train_dataset = RafFaceDataset(train=True, data_path=data_path,
                    transform=transforms.Compose([
                        transforms.Resize(224),
                        transforms.ColorJitter(),
                        transforms.RandomRotation(30),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    weights = []
    for sample in train_dataset:
        label = sample['emotion']
        if label == 0:
            weights.append(3.68)
        elif label == 1:
            weights.append(16.78)
        elif label == 2:
            weights.append(6.8)
        elif label == 3:
            weights.append(1)
        elif label == 4:
            weights.append(2.42)
        elif label == 5:
            weights.append(6.87)
        elif label == 6:
            weights.append(1.86)
        else:
            print('label error')

    weighted_random_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             sampler=weighted_random_sampler)

    test_dataset = RafFaceDataset(train=False, data_path=data_path,
                    transform=transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader