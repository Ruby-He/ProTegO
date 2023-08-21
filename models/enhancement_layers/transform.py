import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as Ktrans

def image(path):
    IMG = cv2.imread(path)       
    IMG = cv2.resize(IMG, (100, 32)) # resize
    
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMG = torch.FloatTensor(IMG) # [H, W, C]
    IMG = IMG / 255 # normalization to [0,1]
    IMG = IMG.permute(2,0,1) # [C, H, W]

    return IMG.unsqueeze(0) # [B, C, H, W]

# real binarization
class R_Binarization(nn.Module):
    def __init__(self):
        super(R_Binarization, self).__init__()

    def RB(self, x):
        x_np = x.squeeze_(0).permute(1,2,0).cpu().numpy() #[h,w,c]
        maxValue = x_np.max()
        x_np = x_np * 255 / maxValue 
        x_uint = np.uint8(x_np)
        gray = cv2.cvtColor(x_uint, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        x_b = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        x_b = torch.FloatTensor(x_b)
        x_b = x_b / 255
        x_b = x_b.permute(2,0,1).unsqueeze_(0)
        return x_b.to(x.device)
    
    def forward(self, x):
        enhance_adv = self.RB(x)
        return enhance_adv

# differentiable binarization
class D_Binarization(nn.Module):
    def __init__(self):
        super(D_Binarization, self).__init__()
        self.k = 20

    def DB(self, x):
        y = 0.8 *torch.ones_like(x)
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        enhance_adv = self.DB(x)
        return enhance_adv

class Translate(nn.Module):
    """
    Translate the image.
    """
    def __init__(self):
        super(Translate, self).__init__()
        i = random.choice([0,1])
        if i == 0:
            self.translation = torch.tensor([[4., 4.]])
        else:
            self.translation = torch.tensor([[-4., -4.]])

    def forward(self, x): 
        device = x.device
        enhance_adv = Ktrans.translate(x, self.translation.to(device), mode='bilinear', padding_mode='border', align_corners=True)

        return enhance_adv
    
class Resize(nn.Module):
    """
    Resize the image. The target size is 
    """

    def __init__(self):
        super(Resize, self).__init__()
        self.resize_rate = 1.60
        self.img_h = 32
        self.img_w = 100

    def input_diversity(self, x):
        img_resize_h = int(self.img_h * self.resize_rate)
        img_resize_w = int(self.img_w * self.resize_rate)

        rnd_h = torch.randint(low=self.img_h, high=img_resize_h, size=(1,), dtype=torch.int32)
        rnd_w = torch.randint(low=self.img_w , high=img_resize_w, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd_h, rnd_w], mode='bilinear', align_corners=False)
        h_rem = img_resize_h - rnd_h
        w_rem = img_resize_w - rnd_w
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        padded = F.interpolate(padded, size=[self.img_h, self.img_w])
        return padded

    def forward(self, x):
        enhance_adv = self.input_diversity(x)
        return enhance_adv 