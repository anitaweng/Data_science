import torch.nn as nn
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import imageio
import cv2
import copy
import torch.nn.functional as F
import sys
import torchvision.models as models
import matplotlib.image as mpimg
import numpy as np

class PopBooClassifier(nn.Module):
    def __init__(self, bs):
        super().__init__()
        #self.resnet50 = resnet50
        self.batch_size = bs

        '''self.conv1 = nn.Conv2d(1000, 512, 1)
        self.conv2 = nn.Conv2d(512, 64, 1) #b*64*1*1
        self.conv3 = nn.Conv2d(64, 16, 1) #b*16*1*1
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)'''
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 5, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(128 * 26 * 26, 500)  # 5=>128*14*14 3=>128*16*16
        self.fc2 = nn.Linear(500, 2)


    def forward(self,x):
        #print(x.shape)
        '''x = self.resnet50(x.unsqueeze(0))
        x = F.relu(self.conv1(x.unsqueeze(2).unsqueeze(3)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(self.batch_size, -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x'''
        x = self.conv1(x.unsqueeze(0))
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(self.batch_size, -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def test(imgfilelistname):
    model_id = 'ptt_many'
    device = torch.device('cuda')
    #resnet50 = models.resnet50(pretrained=True)
    M = PopBooClassifier( 1).eval().to(device)
    image_size = 224
    m_checkpoint = torch.load(os.path.join('checkpoint', model_id, 'M.ckpt'))
    M.load_state_dict(m_checkpoint)
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


    with open(imgfilelistname, 'r') as f, open('classification.txt', 'w') as outf:
        lines = f.readlines()
        for line in lines:
            linef = line.replace("\n","")
            with open(linef, 'rb') as f:
                img = mpimg.imread(f)
                
                if len(img.shape) > 2 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                #print(linef)
                    #img = np.uint8(img)
                '''try:
                  if len(img.shape) > 2 and img.shape[2] == 4:
                      img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                except:
                  print('png'+linef)'''

                #PIL_image = Image.fromarray(img)
                #print(img.shape)
                if transform is not None:
                      timg = transform(np.uint8(img))
                
                #print('trans'+linef)

                timg = timg.to(device)
                y = M(timg)
                #print(y.shape)
                if y[0][0] >= y[0][1]:
                    outf.write('0')
                else:
                    outf.write('1')
                '''try:
                    
                except:
                    print(linef)'''



if __name__ == '__main__':
    test(sys.argv[1])