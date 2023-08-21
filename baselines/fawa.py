import os
import sys
# sys.path.append('xx/ProTegO/')
import string
import shutil
import argparse
from PIL import Image
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import utils as vutils

from dataset import test_dataset_builder
from utils import Logger, np2tensor, tensor2np, get_text_mask, cvt2Image, color_map
from wm_attacker import WM_Attacker


def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print('cannot create dirs: {}'.format(path))
            exit(0)

def fawa(opt):
    """prepare log with model_name """
    model_name = os.path.basename(opt.str_model.split('-')[0])
    print('-----Attack Settings< model_name:{} --iter_num:{} --eps:{} --decay:{} --alpha:{} >'
            .format(model_name, opt.iter_num, opt.eps, opt.decay, opt.alpha))
    
    save_root = os.path.join(opt.save_attacks, model_name)
    makedirs(save_root)
    del_list = os.listdir(save_root)
    for f in del_list:
        file_path = os.path.join(save_root, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    wm_save_adv_path = os.path.join(save_root, 'wmadv')
    wm_save_per_path = os.path.join(save_root, 'wmper')
    makedirs(wm_save_adv_path)
    makedirs(wm_save_per_path)
    
    """ save all the print content as log """
    log_file= os.path.join(save_root, 'train.log')
    sys.stdout = Logger(log_file)

    dataset = test_dataset_builder(opt.imgH, opt.imgW, opt.root)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=opt.batch_size,
                    shuffle=False, num_workers=4,
                    drop_last=True,pin_memory=True)

    attacker = WM_Attacker(opt)
    time_all, suc, ED_sum = 0, 0, 0
    for i, data in enumerate(dataloader, start=0):
        label = data[1]
        img = data[5] # binary image
        img_index = data[2][0]
        img_path = data[4][0]
        img_tensor = img.to(opt.device)
        
        """basic attack"""
        attacker.init()
        adv_img, delta, epoch, preds, flag, time = attacker.basic_attack(img_tensor, label)
        print('img-{}_path:{} --iters:{} --Success:{} --prediction:{} --groundtruth:{} --time:{}'
                .format(i, img_path, epoch, flag, preds, label[0], time))

        """wm attack""" 
        # find position to add watermark
        adv_np = tensor2np(adv_img.squeeze(0))
        img_np = tensor2np(img_tensor.squeeze(0))
        pos, frames = attacker.find_wm_pos(adv_np, img_np, True)
        # 按面积大小把pos从大到小排个序
        new_pos = []
        for _pos in pos:
            if len(_pos) > 1:
                new_pos.append(sorted(_pos, key=lambda x: (x[3]-x[1])*(x[2]-x[0]), reverse=True))
            else:
                new_pos.append(_pos)
        pos = new_pos

        # get watermark mask
        grayscale = 0
        color = (grayscale, grayscale, grayscale)
        wm_img = attacker.gen_wm(color)
        wm_arr = np.array(wm_img.convert('L'))
        bg_mask = ~(wm_arr != 255)

        # # get grayscale watermark
        # """
        # 灰度值在 76-226 有对应的彩色水印值，为了增加扰动后还在范围内，128-174,
        # *note by hyr:paper setting = 255*0.682 =174
        # """
        # grayscale = 174 
        # color = (grayscale, grayscale, grayscale)
        # wm_img = np.array(Image.new(mode="RGB", size=wm_img.size, color=color))
        # wm_img[bg_mask] = 255
        # wm_img = Image.fromarray(wm_img)

        # # get color watermark
        grayscale = 174 
        green_v = color_map(grayscale)
        color = (255, green_v, 0)
        wm_img = np.array(Image.new(mode="RGB", size=wm_img.size, color=color))
        wm_img[bg_mask] = 255
        wm_img = Image.fromarray(wm_img)


        text_img = cvt2Image(img_np)
        text_mask = get_text_mask(np.array(text_img))  # bool, 1 channel
        rgb_img = Image.new(mode="RGB", size=(text_img.size), color=(255, 255, 255)) # white bg
        p = -int(wm_img.size[0] * np.tan(10 * np.pi / 180))
        right_shift = 20
        xp = pos[0][0][0]+right_shift if len(pos) != 0 else right_shift
        rgb_img.paste(wm_img, box=(xp, p))  # first to add wm
        wm_mask = (np.array(rgb_img) != 255)  # bool, 3 channel
        rgb_img.paste(text_img, mask=cvt2Image(text_mask))  # then add text

        wm0_img = np.array(rgb_img)
        wm_img = np2tensor(wm0_img)
        wm_mask = np2tensor(wm_mask) 
        adv_text_mask = ~text_mask
        adv_text_mask = np2tensor(adv_text_mask)
        
        attacker.init()
        adv_img_wm, delta, epoch, preds, flag, ED, time = attacker.wm_attack(wm_img, label, wm_mask, adv_text_mask)  

        print('wmimg-{}_path:{} --iters:{} --Success:{} --prediction:{} --groundtruth:{} --edit_distance:{} --time:{}'
                .format(i, img_path, epoch, flag, preds, label[0], ED, time))
        
        time_all +=time
        if flag:
            suc += 1
            ED_sum += ED
        vutils.save_image(adv_img_wm,"{}/{}_{}_adv.png".format(wm_save_adv_path, os.path.basename(img_index), label[0]))
        vutils.save_image(delta*100,"{}/{}_{}_delta.png".format(wm_save_per_path, os.path.basename(img_index), label[0]))
    
    print('FAWA_Total_attack_time:{} '.format(time_all))
    print('ASR:{:.2f}% '.format((suc/len(dataloader))*100))
    print('Average Edit_distance: {}'.format(ED_sum/suc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/hyr/dataset/PTAU/new/test100-times/',help='path of original text images')
    parser.add_argument('--save_attacks', type=str, default='res-baselines/fawa', help='path of save adversarial images')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--iter_num', type=int, default=2000, help='number of iterations')
    parser.add_argument('--eps', type=float, default=40/255, help='maximnum perturbation setting in paper')
    parser.add_argument('--decay', type=float, default=1.0, help='momentum factor')
    parser.add_argument('--alpha', type=float, default=0.05, help='the step size of the iteration')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_false', help='for sensitive character mode')
    """ Model Architecture """
    parser.add_argument('--str_model', type=str, help="well-trained models for evaluation",
                        default='STR_modules/downloads_models/STARNet-TPS-ResNet-BiLSTM-CTC-sensitive.pth')
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    opt = parser.parse_args()
    
    """ vocab / character number configuration """
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.device  = torch.device(device_type)

    opt.save_attacks = opt.save_attacks + "-eps" + str(int(opt.eps*255))

    if opt.sensitive:
        opt.character = string.printable[:62]
    
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    torch.cuda.synchronize()
    fawa(opt)
