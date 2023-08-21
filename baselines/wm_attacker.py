import os
import time
import numpy as np
import cv2
import torch

from skimage import morphology
from trdg.generators import GeneratorFromStrings
from nltk.metrics import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, RGB2Hex
from STR_modules.model import Model

r"""
    Base class for fawa.

    Distance Measure : Linf

"""
class WM_Attacker(object):
    def __init__(self, c_para):
        r"""
        Arguments:
            c_para : all arguments from Parser which are prepared for <fawa>
        """
        self.device = c_para.device
        self.batch_size = c_para.batch_size
        self.eps = c_para.eps
        self.iter_num = c_para.iter_num
        self.decay = c_para.decay
        self.alpha = c_para.alpha
        
        self.converter = self._load_converter(c_para)
        self.model = self._load_model(c_para)
        self.Transformation = c_para.Transformation
        self.FeatureExtraction = c_para.FeatureExtraction
        self.SequenceModeling = c_para.SequenceModeling
        self.Prediction = c_para.Prediction
        self.batch_max_length = c_para.batch_max_length

        self.criterion = self._load_base_loss(c_para)
        self.l2_loss = self._load_l2_loss()

        self.img_size = (c_para.batch_size, c_para.input_channel, c_para.imgH, c_para.imgW)
        self.best_img = 100 * torch.ones(self.img_size)
        self.best_iter = -1
        self.best_delta = 100 * torch.ones(self.img_size)
        self.preds = ''
        self.suc = False

    def init(self):
        self.best_img = 100* torch.ones(self.img_size)
        self.best_iter = -1
        self.best_delta = 100 * torch.ones(self.img_size)
        self.preds = ''
        self.suc = False

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

    def basic_attack(self, x, raw_text):
        x = x.clone().detach().to(self.device)
        text_for_pred = torch.LongTensor(self.batch_size, self.batch_max_length + 1).fill_(0).to(self.device)
        text, length = self.converter.encode(raw_text, batch_max_length=self.batch_max_length)
        pred_org = self.model_pred(x, self.Prediction)  #TODO change to raw_text as label
        
        momentum = torch.zeros_like(x).detach().to(self.device)
        
        adv_x = x.clone().detach()

        num = 0
        time_each = 0
        t_st = time.time()
        for iter in range(self.iter_num):
            adv_x.requires_grad = True

            # erlier stop
            pred_adv = self.model_pred(adv_x, self.Prediction)
            if pred_adv != pred_org: 
                num += 1
                # print('Best results!')
                self.best_iter = iter
                self.best_img = adv_x.detach().clone()
                self.best_delta = delta.detach().clone()
                self.preds = pred_adv
                self.suc = True
                if num == 1:
                    break 
            
            # Calculate loss
            torch.backends.cudnn.enabled=False
            preds = self.model(adv_x, text_for_pred).log_softmax(2)
            if 'CTC' in self.Prediction:
                preds_size = torch.IntTensor([preds.size(1)] * self.batch_size).to(self.device)
                preds = preds.permute(1, 0, 2)
                cost = self.criterion(preds, text, preds_size, length)
            else: # ATTENTION
                target_text = text[:, 1:]
                cost = self.criterion(preds.view(-1, preds.shape[-1]), target_text.contiguous().view(-1))

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_x,
                                       retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - x, min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

            if iter == self.iter_num and self.best_iter == -1:  
                print('[!] Attack failed: No optimal results were found in effective iter_num!')
                self.best_iter = -1
                self.best_img = adv_x.detach().clone()
                self.best_delta = delta.data.detach().clone()
                self.preds = pred_adv
                self.suc = False

        t_end = time.time()
        time_each = t_end - t_st

        return self.best_img, self.best_delta, self.best_iter, self.preds, self.suc, time_each

    def find_wm_pos(self, adv_img, input_img, ret_frame_img=False):
        pert = np.abs(adv_img - input_img)
        pert = (pert > 1e-2) * 255.0
        wm_pos_list = []
        frame_img_list = []
        for src in pert:
            kernel = np.ones((3, 3), np.uint8)  # kernel_size 3*3
            dilate = cv2.dilate(src, kernel, iterations=2)  
            erode = cv2.erode(dilate, kernel, iterations=2)  
            remove = morphology.remove_small_objects(erode.astype('bool'), min_size=0)
            contours, _ = cv2.findContours((remove * 255).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            wm_pos, frame_img = [], []
            for cont in contours:
                left_point = cont.min(axis=1).min(axis=0)
                right_point = cont.max(axis=1).max(axis=0)
                wm_pos.append(np.hstack((left_point, right_point)))
                if ret_frame_img:
                    img = cv2.rectangle(
                        (remove * 255).astype('uint8'), (left_point[0], left_point[1]),
                        (right_point[0], right_point[1]), (255, 255, 255), 2)
                    frame_img.append(img)
            wm_pos_list.append(wm_pos)
            frame_img_list.append(frame_img)

        if ret_frame_img:
            return (wm_pos_list, frame_img_list)
        else:
            return wm_pos_list

    def gen_wm(self, RGB):
        generator = GeneratorFromStrings(
            strings=['ecml'],
            count=1, 
            fonts=['baselines/Impact.ttf'],  # TODO change ['Impact.tff']]
            language='en',
            size=78, # default: 32
            skewing_angle=15,
            random_skew=False,
            blur=0,
            random_blur=False,
            background_type=1, # gaussian noise (0), plain white (1), quasicrystal (2) or picture (3)
            distorsion_type=0,  # None(0), Sine wave(1),Cosine wave(2),Random(3)
            distorsion_orientation=0,
            is_handwritten=False,
            width=-1,
            alignment=1,
            text_color=RGB2Hex(RGB),
            orientation=0,
            space_width=1.0,
            character_spacing=0,
            margins=(0, 0, 0, 0),
            fit=True,
        )
        img_list = [img for img, _ in generator]
        return img_list[0]

    def wm_attack(self, wm_x, raw_text, wm_mask, adv_text_mask):
        wm_x = wm_x.clone().detach().to(self.device)

        text_for_pred = torch.LongTensor(self.batch_size, self.batch_max_length + 1).fill_(0).to(self.device)
        text, length = self.converter.encode(raw_text, batch_max_length=self.batch_max_length)
        pred_org = self.model_pred(wm_x, self.Prediction)

        wm_mask = wm_mask.clone().detach().to(self.device)
        adv_text_mask = adv_text_mask.clone().detach().to(self.device)

        momentum = torch.zeros_like(wm_x).detach().to(self.device)

        adv_x = wm_x.clone().detach()

        num = 0
        ED_num= 0
        time_each = 0
        t_st = time.time()
        for iter in range(self.iter_num):
            adv_x.requires_grad = True
            
            # erlier stop
            pred_adv = self.model_pred(adv_x, self.Prediction)
            if pred_adv != pred_org:
                    ED_num = edit_distance(pred_org, pred_adv)
                    num += 1
                    # print('Best results!')
                    self.best_iter = iter
                    self.best_img = adv_x.detach().clone()
                    self.best_delta = delta.detach().clone()
                    self.preds = pred_adv
                    self.suc = True
                    if num == 1:
                        break 
            
            # Calculate loss
            torch.backends.cudnn.enabled=False
            preds = self.model(adv_x, text_for_pred).log_softmax(2)
            if 'CTC' in self.Prediction:
                preds_size = torch.IntTensor([preds.size(1)] * self.batch_size).to(self.device)
                preds = preds.permute(1, 0, 2)
                cost = self.criterion(preds, text, preds_size, length)
            else: # ATTENTION
                target_text = text[:, 1:]
                cost = self.criterion(preds.view(-1, preds.shape[-1]), target_text.contiguous().view(-1))

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_x,
                                       retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - wm_x, min=-self.eps, max=self.eps)
            delta = torch.mul(delta, adv_text_mask)
            delta = torch.mul(delta, wm_mask)
            adv_x = torch.clamp(wm_x + delta, min=0, max=1).detach()

            if iter == self.iter_num and self.best_iter == -1:  
                print('[!] Attack failed: No optimal results were found in effective iter_num!')
                ED_num = 0
                self.best_iter = -1
                self.best_img = adv_x.detach().clone()
                self.best_delta = delta.data.detach().clone()
                self.preds = pred_adv
                self.suc = False

        t_end = time.time()
        time_each = t_end - t_st

        return self.best_img, self.best_delta, self.best_iter, self.preds, self.suc, ED_num, time_each


    