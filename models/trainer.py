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

class Trainer():
    '''
    train model
    '''
    def __init__(self, model, trainloader, lr=1e-4, num_epochs=50, batch_size=32, 
                 device='cuda', log_path='./save/log/', checkpoint_path='./save/checkpoint/'):
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
    
    def save_checkpoint(self, epoch):
        torch.save({'state_dict': self.model.state_dict()},
                    self.checkpoint_path+'model_epoch'+str(epoch)+'.pth')
    
    def logging(self, losses, train_accuracy):
        with open(self.log_path+'losses.pkl', 'wb') as f:
            pickle.dump(losses,f)
        with open(self.log_path+'train_accuracy.pkl', 'wb') as f:
            pickle.dump(train_accuracy,f)
    
    def train_model(self):
        self.model.train()
        optim = torch.optim.Adam(self.model.parameters() , lr = self.lr)
        crit = nn.CrossEntropyLoss()
        total_step = len(self.trainloader)
        
        losses = []
        train_accuracy = []
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.trainloader):
                images,labels = data['image'],data['emotion'] #(32,3,224,224)
                images = images.to(self.device)
                labels = labels.to(self.device)
                optim.zero_grad()
                
                outputs = self.model(images)
                loss = crit(outputs, labels)
                
                loss.backward()
                optim.step()
                
                # log
                _, argmax = torch.max(outputs, 1)
                accuracy = (labels == argmax.squeeze()).float().mean()
                losses.append(loss.item()/ total_step)
                train_accuracy.append(accuracy.item())
                
                if (i+1) % 70 == 1:
                    print ('Epoch [{}/{}], Step [{}/{}], Log_Loss: {:.4f} , Accuracy at the step {:.3f}' 
                           .format(epoch+1, self.num_epochs, i+1, total_step, loss.item() , accuracy.item() ))
            # save checkpoint
            if (epoch+1)%5==0:
                self.save_checkpoint(epoch+1)
        
        # log
        self.logging(losses, train_accuracy)
        print('Training is complete!')
            
def main():
    configname = 'config.yaml'
    with open(configname, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
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
        with open (trainloader_path, 'rb') as f:
            trainloader = pickle.load(f)
    
    # construct model
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(in_features=1280, out_features=7, bias=True)
    
    # train model
    trainer = Trainer(model           = model,
                      trainloader     = trainloader,
                      lr              = lr,
                      num_epochs      = num_epochs,
                      batch_size      = batch_size,
                      device          = device,
                      log_path        = log_path,
                      checkpoint_path = checkpoint_path)
    trainer.train_model()
    
if __name__=='__main__':
    main()