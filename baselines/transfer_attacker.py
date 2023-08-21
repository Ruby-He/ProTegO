import os
import time
import torch

from utils import CTCLabelConverter, AttnLabelConverter
from STR_modules.model import Model

r"""
    Base class for transfer-based attack [SI-NI-FGSM, VMI-FGSM], 
    modified from "https://github.com/Harry24k/adversarial-attacks-pytorch".

    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020

    VMI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021.

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 40/255)
        iter_num (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        alpha (float): step size. (Default: 2/255)
        beta (float): the upper bound of neighborhood. (Default: 3/2)
        m (int): number of scale copies. (Default: 5)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)

"""

class transfer_Attacker(object):

    def __init__(self, c_para):
        self.device = c_para.device
        self.batch_size = c_para.batch_size
        self.eps = c_para.eps
        self.iter_num = c_para.iter_num
        self.decay = c_para.decay
        self.alpha = c_para.alpha
        self.beta = c_para.beta
        self.m = c_para.m
        self.N = c_para.N        
        
        self.converter = self._load_converter(c_para)
        self.model = self._load_model(c_para)
        self.Transformation = c_para.Transformation
        self.FeatureExtraction = c_para.FeatureExtraction
        self.SequenceModeling = c_para.SequenceModeling
        self.Prediction = c_para.Prediction
        self.batch_max_length = c_para.batch_max_length

        self.criterion = self._load_base_loss(c_para)
        self.l2_loss = self._load_l2_loss()

    @staticmethod
    def _load_l2_loss():
        return torch.nn.MSELoss()
    @staticmethod
    def _load_base_loss(c_para):
        if c_para.Prediction == 'CTC':
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(c_para.device)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(c_para.device)
        return criterion
    @staticmethod
    def _load_converter(c_para):
        if 'CTC' in c_para.Prediction:
            converter = CTCLabelConverter(c_para.character)
        else:
            converter = AttnLabelConverter(c_para.character)
        c_para.num_class = len(converter.character)
        return converter
    @staticmethod
    def _load_model(c_para):
        if not os.path.exists(c_para.str_model):
            raise FileNotFoundError("cannot find pth file in {}".format(c_para.str_model))
        # load model
        with torch.no_grad():
            model = Model(c_para).to(c_para.device)
            model.load_state_dict(torch.load(c_para.str_model, map_location=c_para.device))
        for name, para in model.named_parameters():
            para.requires_grad = False
        return model.eval()
  
    def model_pred(self, img, mode='CTC'):
        text_for_pred = torch.LongTensor(self.batch_size, self.batch_max_length + 1).fill_(0).to(self.device)
        # self.model.eval()
        with torch.no_grad():
            if mode == 'CTC':
                pred = self.model(img, text_for_pred).log_softmax(2)
                size = torch.IntTensor([pred.size(1)] * self.batch_size).to(self.device)
                _, index = pred.permute(1, 0, 2).max(2)
                index = index.transpose(1, 0).contiguous().view(-1)
                pred_str = self.converter.decode(index.data, size.data)
                pred_str = pred_str[0]
            else:  # ATTENTION
                pred = self.model(img, text_for_pred)
                size = torch.IntTensor([pred.size(1)] * self.batch_size)
                _, index = pred.max(2)
                pred_str = self.converter.decode(index, size)
                pred_s = pred_str[0]
                pred_str = pred_s[:pred_s.find('[s]')]
        # self.model.train()
        return pred_str
        
    def SINIFGSM(self, x, raw_text):
        torch.backends.cudnn.enabled=False
        x = x.clone().detach().to(self.device)
        text_for_pred = torch.LongTensor(self.batch_size, self.batch_max_length + 1).fill_(0).to(self.device)
        text, length = self.converter.encode(raw_text, batch_max_length=self.batch_max_length)
        
        momentum = torch.zeros_like(x).detach().to(self.device)

        adv_x = x.clone().detach()
        time_each = 0
        t_st = time.time()
        pred_org = self.model_pred(x, self.Prediction)
        for iters in range(self.iter_num):
            adv_x.requires_grad = True
            nes_x = adv_x + self.decay*self.alpha*momentum
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(x).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_x = nes_x / torch.pow(2, i)
                preds = self.model(nes_x, text_for_pred).log_softmax(2)
                # Calculate loss
                if 'CTC' in self.Prediction:
                    preds_size = torch.IntTensor([preds.size(1)] * self.batch_size).to(self.device)
                    preds = preds.permute(1, 0, 2)
                    cost = self.criterion(preds, text, preds_size, length)
                else: # ATTENTION
                    target_text = text[:, 1:]
                    cost = self.criterion(preds.view(-1, preds.shape[-1]), target_text.contiguous().view(-1))
                adv_grad += torch.autograd.grad(cost, adv_x,
                                                retain_graph=True, create_graph=False)[0]
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay*momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - x, min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

            pred_adv = self.model_pred(adv_x, self.Prediction)
            if pred_adv != pred_org:
                break
        t_en = time.time()
        time_each = t_en - t_st
        return adv_x, delta, pred_adv, time_each, iters

    def VMIFGSM(self, x, raw_text):
        torch.backends.cudnn.enabled=False
        x = x.clone().detach().to(self.device)
        text_for_pred = torch.LongTensor(self.batch_size, self.batch_max_length + 1).fill_(0).to(self.device)
        text, length = self.converter.encode(raw_text, batch_max_length=self.batch_max_length)
        
        momentum = torch.zeros_like(x).detach().to(self.device)
        v = torch.zeros_like(x).detach().to(self.device)

        adv_x = x.clone().detach()
        time_each = 0
        t_st = time.time()
        pred_org = self.model_pred(x, self.Prediction)
        for iters in range(self.iter_num):
            adv_x.requires_grad = True
            preds = self.model(adv_x, text_for_pred).log_softmax(2)

            # Calculate loss
            if 'CTC' in self.Prediction:
                preds_size = torch.IntTensor([preds.size(1)] * self.batch_size).to(self.device)
                preds = preds.permute(1, 0, 2)
                cost = self.criterion(preds, text, preds_size, length)
            else: # ATTENTION
                target_text = text[:, 1:]
                cost = self.criterion(preds.view(-1, preds.shape[-1]), target_text.contiguous().view(-1))

            # Update adversarial images
            adv_grad = torch.autograd.grad(cost, adv_x,
                                       retain_graph=False, create_graph=False)[0]
            
            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(x).detach().to(self.device)
            for _ in range(self.N):
                neighbor_x = adv_x.detach() + \
                                  torch.randn_like(x).uniform_(-self.eps*self.beta, self.eps*self.beta)
                neighbor_x.requires_grad = True
                preds = self.model(neighbor_x, text_for_pred).log_softmax(2)

                # Calculate loss
                if 'CTC' in self.Prediction:
                    preds_size = torch.IntTensor([preds.size(1)] * self.batch_size).to(self.device)
                    preds = preds.permute(1, 0, 2)
                    cost = self.criterion(preds, text, preds_size, length)
                else: # ATTENTION
                    target_text = text[:, 1:]
                    cost = self.criterion(preds.view(-1, preds.shape[-1]), target_text.contiguous().view(-1))
                    
                GV_grad += torch.autograd.grad(cost, neighbor_x,
                                               retain_graph=False, create_graph=False)[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - x, min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()
            pred_adv = self.model_pred(adv_x, self.Prediction)
            if pred_adv != pred_org:
                break
        t_en = time.time()
        time_each = t_en - t_st
        

        return adv_x, delta, pred_adv, time_each, iters

