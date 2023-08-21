import os
import torch
import cv2
import torch.utils.data
from torch.utils.data import Dataset    


"""ProtegO dataset"""
def up_dataset(up_path):        
    up = cv2.imread(up_path)
    up = cv2.resize(up, (100, 32))
    up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB) 
    up  = torch.FloatTensor(up)
    up = up / 255 # normalization to [0,1]
    up = up.permute(2,0,1) # [C, H, W]

    return up

class train_dataset_builder(Dataset):
    def __init__(self, height, width, img_path):
        '''
        height: input height to model
        width: input width to model
        total_img_path: path with all images
        seq_len: sequence length
        '''
        self.height = height
        self.width = width
        self.img_path = img_path
        self.dataset = []
        
        img = [] 
        for i,j,k in os.walk(self.img_path):            
            for file in k:
                file_name = os.path.join(i ,file)
                img.append(file_name)
        self.total_img_name = img
        
        for img_name in self.total_img_name:
            _, label, _ = img_name.split('_')
            self.dataset.append([img_name, label])


    def __getitem__(self, index):
        img_name, label = self.dataset[index]
        IMG = cv2.imread(img_name)       
        IMG = cv2.resize(IMG, (self.width, self.height)) # resize
        
        # binarization processing
        gray = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        IMG = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        IMG = torch.FloatTensor(IMG) # [H, W, C]
        IMG = IMG / 255 # normalization to [0,1]
        IMG = IMG.permute(2,0,1) # [C, H, W]
    
        return IMG, label

    def __len__(self):
        return len(self.dataset)

class test_dataset_builder(Dataset):
    def __init__(self, height, width, img_path): 
        self.height = height
        self.width = width
        self.img_path = img_path
        self.dataset = []
        
        img = [] 
        for i,j,k in os.walk(self.img_path):            
            for file in k:
                file_name = os.path.join(i ,file)
                img.append(file_name)
        self.total_img_name = img
        
        for img_name in self.total_img_name:
            img_index, label, img_adv = img_name.split('_')  
            img_adv = img_adv.split('.') 
            index_or_advlogo = img_adv[0]
            self.dataset.append([img_name, label, img_index, index_or_advlogo])
        self.dataset = sorted(self.dataset)

    def __getitem__(self, index):
        img_name, label, img_index, index_or_advlogo = self.dataset[index]
        IMG = cv2.imread(img_name)
        ORG = cv2.resize(IMG, (self.width, self.height))

        IMG = cv2.cvtColor(ORG, cv2.COLOR_BGR2RGB)
        IMG = torch.FloatTensor(IMG) # convert to tensor [H, W, C]
        IMG = IMG / 255
        IMG = IMG.permute(2,0,1) # [C, H, W]
        
        gray = cv2.cvtColor(ORG, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        mask = torch.FloatTensor(mask) # [H, W, C]
        mask = mask / 255 # normalization to [0,1]
        mask = mask.permute(2,0,1) # [C, H, W]

        return IMG, label, img_index, index_or_advlogo, img_name, mask

    def __len__(self):
        return len(self.dataset)

class test_adv_dataset(Dataset):
    def __init__(self, height, width, img_path):      
        self.height = height
        self.width = width
        self.img_path = img_path
        self.dataset = []
        
        img = [] 
        for i,j,k in os.walk(self.img_path):            
            for file in k:
                file_name = os.path.join(i ,file)
                img.append(file_name)
        self.total_img_name = img
        
        for img_name in self.total_img_name:
            img_index, label, img_adv = img_name.split('_')  
            img_adv = img_adv.split('.') 
            index_or_advlogo = img_adv[0] 
            self.dataset.append([img_name, label, img_index, index_or_advlogo])
        self.dataset = sorted(self.dataset)

    def __getitem__(self, index):
        img_name, label, img_index, index_or_advlogo = self.dataset[index]        
        IMG = cv2.imread(img_name) 
        IMG = cv2.resize(IMG, (self.width, self.height))
        
        # binarization processing
        gray = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_b = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        img_b = torch.FloatTensor(img_b)
        img_b = img_b / 255 # normalization to [0,1]
        img_b = img_b.permute(2,0,1) # [C, H, W]

        img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
        img = torch.FloatTensor(img)
        img = img /255 # normalization to [0,1]
        img = img.permute(2,0,1) # [C, H, W]

        return img_b, img, label, img_index, index_or_advlogo, img_name

    def __len__(self):
        return len(self.dataset)


"""STR models dataset"""
class strdataset(Dataset):
    def __init__(self, height, width, total_img_path):
        '''
        height: input height to model
        width: input width to model
        total_img_path: path with all images
        seq_len: sequence length
        '''  
        self.total_img_path = total_img_path
        self.height = height
        self.width = width
        img = []
        self.dataset = [] 

        for i,_,k in os.walk(total_img_path):            
            for file in k:
                file_name = os.path.join(i ,file)
                img.append(file_name)
        self.total_img_name = img
                
        for img_name in self.total_img_name:
            _, label, _ = img_name.split('_')           
            self.dataset.append([img_name, label])

    def __getitem__(self, index):
        img_name, label = self.dataset[index]
        IMG = cv2.imread(img_name)
        IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)        
        IMG = cv2.resize(IMG, (self.width, self.height)) # resize
        IMG = (IMG - 127.5)/127.5 # normalization to [-1,1]
        IMG = torch.FloatTensor(IMG) # convert to tensor [H, W, C]
        IMG = IMG.permute(2,0,1) # [C, H, W]

        return IMG, label

    def __len__(self):
        return len(self.dataset)