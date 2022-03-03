import re
import os
import yaml
import pickle
from skimage import io
from PIL import Image
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler

from utils import RafFaceDataset, load_data

class Tester():
    '''
    train model
    '''
    def __init__(self, model, testloader, n_classes=7, num_epochs=50, batch_size=32, 
                 device='cuda', log_path='./save/log/', checkpoint_path='./save/checkpoint/'):
        self.device = device
        self.model = model.to(self.device)
        self.testloader = testloader
        
        self.n_classes = n_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path+'model_epoch'+str(self.num_epochs)+'.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Successfully load model of Epoch {}'.format(self.num_epochs))
    
    def logging(self,confusion_matrix):
        with open(self.log_path+'confusion_matrix.pkl','wb') as f:
            pickle.dump(confusion_matrix, f)
    
    def test_model(self):
        self.model.eval()
        self.load_checkpoint()
        
        with torch.no_grad():
            correct = 0
            total = 0
            
            for data in self.testloader:
                images_tmp,labels_tmp = data['image'],data['emotion']
                images_tmp = images_tmp.to(self.device)
                labels_tmp = labels_tmp.to(self.device)
                outputs = self.model(images_tmp)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels_tmp.size(0)
                correct += (predicted == labels_tmp).sum().item()
                
            print('Accuracy of model is : {} %' .format(100*correct/total))
        
        confusion_matrix = torch.zeros(self.n_classes, self.n_classes)
        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                images,labels = data['image'],data['emotion']
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                
                _, preds = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        print('confusion_matrix')
        print(confusion_matrix)
        print(confusion_matrix.diag()/confusion_matrix.sum(1))
        print('mean_precision: {}'.format((confusion_matrix.diag()/confusion_matrix.sum(1)).sum()/7))
        
        # log
        self.logging(confusion_matrix)
        
        print('Testing is complete!')
            
def main():
    configname = 'config.yaml'
    with open(configname, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    n_classes = config['data_params']['n_classes']
    data_path = config['data_params']['data_path']
    log_path  = config['data_params']['log_path']
    
    lr          = config['train_params']['lr']
    num_epochs  = config['train_params']['num_epochs']
    batch_size  = config['train_params']['batch_size']
    num_workers = config['train_params']['num_workers']
    gpu_id      = config['train_params']['gpu_id']
    checkpoint_path = config['train_params']['checkpoint_path']
    
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if torch.cuda.is_available() and gpu_id >= 0:
      device = torch.device('cuda:%d' % gpu_id)
    else:
      device = torch.device('cpu')
    
    # dataloader
    trainloader_path = log_path+'trainloader.pkl'
    testloader_path = log_path+'testloader.pkl'
    if not os.path.exists(trainloader_path):
        trainloader, testloader = load_data(num_workers = num_workers,
                                            batch_size  = batch_size,
                                            data_path   = data_path)
        with open(trainloader_path,'wb') as f:
            pickle.dump(trainloader,f)
        with open(testloader_path,'wb') as f:
            pickle.dump(testloader,f)
    else:
        with open (testloader_path, 'rb') as f:
            testloader = pickle.load(f)
    
    # construct model
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(in_features=1280, out_features=7, bias=True)
    
    # train model
    tester = Tester(model           = model,
                    testloader      = testloader,
                    n_classes       = n_classes,
                    num_epochs      = num_epochs,
                    batch_size      = batch_size,
                    device          = device,
                    log_path        = log_path,
                    checkpoint_path = checkpoint_path)
    tester.test_model()
    
if __name__=='__main__':
    main()