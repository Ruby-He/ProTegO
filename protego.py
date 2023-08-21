import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils

import models.GAN_models as G_models
from models.enhancer import Enhancer
from dataset import up_dataset

""" custom weights initialization called on netG and netD """
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

class framework():
    def __init__(self,
                 device,
                 model,
                 dt_model,
                 converter,
                 criterion,
                 batch_max_length,
                 up_path,
                 dark,
                 batch_size,
                 image_nc,
                 height,
                 width,
                 eps,
                 lambda1,
                 lambda2,
                 lambda3,
                 lambda4,
                 use_eh,
                 use_guide):
        self.device = device
        self.model = model
        self.dt_model = dt_model
        self.converter = converter
        self.criterion = criterion
        self.batch_max_length = batch_max_length
        self.up_path = up_path
        self.dark = dark
        self.batch_size = batch_size
        self.image_nc = image_nc
        self.height = height
        self.width = width  
        self.box_min = 0
        self.box_max = 1
        self.c = 0.1  # user-specified bound 
        self.eps = eps     
        self.lambda1 = lambda1
        self.lambda2 = lambda2 
        self.lambda3 = lambda3 
        self.lambda4 = lambda4
        self.use_eh = use_eh
        self.use_guide = use_guide

        self.model.apply(fix_bn).to(self.device)
        self.netG = G_models.Generator(self.image_nc).to(self.device)
        self.netD = G_models.Discriminator(self.image_nc).to(self.device)
        # initialize all weights
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.001)


    def train_batch(self, images, labels, up, gui_net, epoch):
    # -----------optimize D-----------
        if self.dark:
            mask = images.detach()
            images = (1-images) + torch.mul(mask, up)
        else:
            mask = images.detach()
            images = torch.mul(images, up) 
            
        perturbation = self.netG(up)
        perturbation = torch.clamp(perturbation, -self.eps, self.eps)

        if self.use_guide:
            map = G_models.guided_net(perturbation, gui_net)
            loss_guide = - torch.mean(torch.tanh(map*1000))

        else: 
            map = torch.zeros_like(perturbation).to(self.device)
            loss_guide = torch.zeros(1).to(self.device)
        
        perturbation = torch.mul(mask, perturbation)

        if self.use_eh:
            mixed_layers = ["Combined([Identity(), Translate(), Resize(), D_Binarization()])"]
            print('epoch{}---enhancement_layers{}'.format(epoch, mixed_layers[0]))
            enhance = Enhancer(mixed_layers).to(self.device)
            adv_images1 = images - perturbation
            adv_images1 = enhance(adv_images1)
            adv_images1 = torch.clamp(adv_images1, self.box_min, self.box_max)
            adv_images2 = images + perturbation
            adv_images2 = enhance(adv_images2)
            adv_images2 = torch.clamp(adv_images2, self.box_min, self.box_max)
        else:
            adv_images1 = images - perturbation
            adv_images1 = torch.clamp(adv_images1, self.box_min, self.box_max)
            adv_images2 = images + perturbation
            adv_images2 = torch.clamp(adv_images2, self.box_min, self.box_max)
        
        self.optimizer_D.zero_grad()
        pred_real = self.netD(images)
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))

        pred_fake1 = self.netD(adv_images1.detach())  
        loss_D_fake1 = F.mse_loss(pred_fake1, torch.zeros_like(pred_fake1, device=self.device))
        pred_fake2 = self.netD(adv_images2.detach())  
        loss_D_fake2 = F.mse_loss(pred_fake2, torch.zeros_like(pred_fake2, device=self.device))

        loss_D_gan = loss_D_fake1 + loss_D_fake2 + loss_D_real
        
        loss_D_gan.backward()
        self.optimizer_D.step()
    # -----------optimize G-----------
        self.optimizer_G.zero_grad()

        # the hinge Loss part of L (calculate perturbation norm)
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        loss_hinge = torch.max(torch.zeros(1, device=self.device), loss_perturb - self.c)

        # the adv Loss part of L 
        torch.backends.cudnn.enabled=False
        """actually, text_for_pred is the param for attention model"""
        text_for_pred = torch.LongTensor(self.batch_size, self.batch_max_length + 1).fill_(0).to(self.device)
        targets, target_len = self.converter.encode(labels, self.batch_max_length)

        preds1 = self.model(adv_images1, text_for_pred)   
        preds_size1 = torch.IntTensor([preds1.size(1)] * self.batch_size) 
        preds1 = preds1.log_softmax(2).permute(1, 0, 2)
        loss_adv1 = - self.criterion(preds1, targets, preds_size1, target_len)
        preds2 = self.model(adv_images2, text_for_pred) 
        preds_size2 = torch.IntTensor([preds2.size(1)] * self.batch_size) 
        preds2 = preds2.log_softmax(2).permute(1, 0, 2)      
        loss_adv2 = - self.criterion(preds2, targets, preds_size2, target_len) 
        loss_adv = loss_adv1 + loss_adv2 
        
        # cal G's loss in GAN
        pred_fake1 = self.netD(adv_images1)
        loss_G_gan1 = F.mse_loss(pred_fake1, torch.ones_like(pred_fake1, device=self.device))
        pred_fake2 = self.netD(adv_images2)
        loss_G_gan2 = F.mse_loss(pred_fake2, torch.ones_like(pred_fake2, device=self.device))
        loss_G_gan = loss_G_gan1 + loss_G_gan2
        loss_G_gan.backward(retain_graph=True)

        loss_G = self.lambda1*loss_hinge + self.lambda2*loss_guide + self.lambda3*loss_G_gan + self.lambda4*loss_adv
        self.model.zero_grad()   
        loss_G.backward()
        self.optimizer_G.step() 

        return loss_G.item(), loss_D_gan.item(), loss_G_gan.item(), loss_hinge.item(), loss_adv.item(), loss_guide.item(), \
               map, perturbation, adv_images1, adv_images2

    def train(self, train_dataloader, epochs, train_adv_path, train_per_path, Generator_path, loss_path):

        loss_G, loss_D_gan, loss_G_gan, loss_hinge, loss_adv, loss_guide= [], [], [], [], [], []

        if self.use_eh and self.use_guide:
            print("==> Use enhancement and guidance module !")
        elif self.use_eh:
            print("==> ONLY Use enhancement layer ...")
        elif self.use_guide:
            print("==> ONLY Use guided network ...")
        else:
            print("Do not use any trick !")

        up = up_dataset(self.up_path)
        up = up.repeat(self.batch_size,1,1,1).to(self.device)
        # up = torch.clamp(up, self.eps, 1-self.eps)
        # vutils.save_image(up, "{}/up.png".format(loss_path))  # TODO remove when release
        
        print('Loading text detection model from \"%s\" as our guided net!' % self.dt_model)
        gui_net = torch.load(self.dt_model).to(self.device)
        gui_net.eval()

        for epoch in range(1, epochs+1):
            if epoch == 20:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=0.0001)
            if epoch == 40:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=0.00001)
            if epoch == 60:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.000001)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=0.000001)

            loss_G_sum, loss_D_gan_sum, loss_G_gan_sum, loss_hinge_sum, loss_adv_sum, loss_guide_sum = 0, 0, 0, 0, 0, 0
            
            for i_batch, data in enumerate(train_dataloader, start=0): 
                images, labels = data         
                images = images.to(self.device)

                loss_G_batch, loss_D_gan_batch, loss_G_gan_batch, \
                loss_hinge_batch, loss_adv_batch, loss_guide_batch,\
                map, perturbation, adv_images1, adv_images2 = self.train_batch(images, labels, up, gui_net, epoch)

                loss_G_sum += loss_G_batch
                loss_D_gan_sum += loss_D_gan_batch
                loss_G_gan_sum += loss_G_gan_batch
                loss_hinge_sum += loss_hinge_batch
                loss_adv_sum += loss_adv_batch
                loss_guide_sum += loss_guide_batch
                
            vutils.save_image(adv_images1, "{}/{}_{}_adv-.png".format(train_adv_path, epoch, i_batch))
            vutils.save_image(adv_images2, "{}/{}_{}_adv+.png".format(train_adv_path, epoch, i_batch))
            vutils.save_image(map*1000, "{}/{}_{}map.png".format(train_per_path, epoch, i_batch))
            vutils.save_image(perturbation, "{}/{}_{}per.png".format(train_per_path, epoch, i_batch))

            # print statistics
            batch_size = len(train_dataloader)
            print('epoch {}: \nloss G: {}, \n\tloss_G_gan: {}, \n\tloss_hinge: {}, \n\tloss_adv: {}, \n\tloss_guide: {}, \nloss D_gan: {}\n'.format(
                 epoch, 
                 loss_G_sum/batch_size, 
                 loss_G_gan_sum/batch_size, 
                 loss_hinge_sum/batch_size,
                 loss_adv_sum/batch_size,
                 loss_guide_sum/batch_size,
                 loss_D_gan_sum/batch_size, 
            ))

            loss_G.append(loss_G_sum / batch_size)
            loss_D_gan.append( loss_D_gan_sum / batch_size)
            loss_G_gan.append(loss_G_gan_sum / batch_size)
            loss_hinge.append(loss_hinge_sum / batch_size)
            loss_adv.append(loss_adv_sum / batch_size)
            loss_guide.append(loss_guide_sum / batch_size)
            
            # save generator
            if epoch % 2== 0:
                netG_file_name = Generator_path + '/netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
        
        plt.figure()
        plt.plot(loss_G)
        plt.title("loss_G")
        plt.savefig(loss_path + '/loss_G.png')

        plt.figure()
        plt.plot(loss_D_gan)
        plt.title("loss_D_gan")
        plt.savefig(loss_path + '/loss_D_gan.png')

        plt.figure()
        plt.plot(loss_G_gan)
        plt.title("loss_G_gan")
        plt.savefig(loss_path + '/loss_G_gan.png')

        plt.figure()
        plt.plot(loss_hinge)
        plt.title("loss_hinge")
        plt.savefig(loss_path + '/loss_hinge.png')
        
        plt.figure()
        plt.plot(loss_adv)
        plt.title("loss_adv")
        plt.savefig(loss_path + '/loss_adv.png')

        plt.figure()
        plt.plot(loss_guide)
        plt.title("loss_guide")
        plt.savefig(loss_path + '/loss_guide.png')

        plt.close('all')