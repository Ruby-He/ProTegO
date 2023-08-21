import os
import sys
# sys.path.append('xx/ProTegO/')
import string
import shutil
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import utils as vutils
from nltk.metrics import edit_distance

from transfer_attacker import transfer_Attacker
from dataset import test_dataset_builder
from utils import Logger


def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print('cannot create dirs: {}'.format(path))
            exit(0)

def transfer_attack(opt):
    """prepare log with model_name """
    model_name = os.path.basename(opt.str_model.split('-')[0])
    print('----------------------------------------------------------------------------------------')
    print('Start attacking: <model_name>:{} <attack_name>:{} \t\n<iter_num>:{} <eps>:{:2f} <alpha>:{:2f} <beta>:{:2f} <m>:{} <N>:{}'
            .format(model_name, opt.name, opt.iter_num, opt.eps, opt.alpha, opt.beta, opt.m, opt.N))
    print('----------------------------------------------------------------------------------------')

    save_root = os.path.join(opt.save_attacks, model_name)
    makedirs(save_root)
    del_list = os.listdir(save_root)
    for f in del_list:
        file_path = os.path.join(save_root, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    save_adv_path = os.path.join(save_root, 'adv')
    save_per_path = os.path.join(save_root, 'per')
    makedirs(save_adv_path)
    makedirs(save_per_path)
    
    """ save all the print content as log """
    log_file= os.path.join(save_root, 'train.log')
    sys.stdout = Logger(log_file)

    dataset = test_dataset_builder(opt.imgH, opt.imgW, opt.root)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=opt.batch_size,
                    shuffle=False, num_workers=4,
                    drop_last=True,pin_memory=True)

    # up = up_dataset(opt.up_path)
    # up = up.repeat(opt.batch_size,1,1,1)

    attacker = transfer_Attacker(opt)
    time_all, suc, ED_sum = 0, 0, 0
    for i, data in enumerate(dataloader, start=0): 
        # image = data[0]
        label = data[1]
        img_index = data[2][0]
        img_path = data[4][0]
        mask = data[5]
        mask_tensor = mask.to(opt.device)
        # img_tensor = torch.mul(mask, up).to(opt.device)
        if opt.name == 'SINIFGSM':
            adv_img, delta, preds, time, iters  = attacker.SINIFGSM(mask_tensor, label)
        elif opt.name == 'VMIFGSM':
            adv_img, delta, preds, time, iters  = attacker.VMIFGSM(mask_tensor, label)
        
        flag = (preds != label[0])
        ED = edit_distance(label[0], preds)

        print('img-{}_path:{} --Success:{} --prediction:{} --groundtruth:{} --edit_distance:{} --time:{:2f} --iter_num:{}'
                .format(i, img_path, flag, preds, label[0], ED, time, iters))
        
        time_all +=time

        vutils.save_image(adv_img,"{}/{}_{}_adv.png".format(save_adv_path, os.path.basename(img_index), label[0]))
        vutils.save_image(torch.abs(delta*100),"{}/{}_{}_delta.png".format(save_per_path, os.path.basename(img_index), label[0]))

        if flag:
            suc += 1
            ED_sum +=ED
            
        
    print('{} Total_attack_time:{} '.format(opt.name, time_all))
    print('ASR:{:.2f}% '.format((suc/len(dataloader))*100))
    print('Average Edit_distance: {}'.format(ED_sum/suc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='attack name [SINIFGSM, VMIFGSM]')
    parser.add_argument('--root', default='/data/hyr/dataset/PTAU/new/test100-times/',help='path of images')
    parser.add_argument('--save_attacks', type=str, default='res-baselines/', help='path of save adversarial text images')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--eps', type=float, default=40/255, help='maximum perturbation')
    parser.add_argument('--iter_num', type=int, default=30, help='number of max iterations')
    parser.add_argument('--decay', type=float, default=1, help='momentum factor')
    parser.add_argument('--alpha', type=float, default=2/255, help='step size of each iteration')
    parser.add_argument('--beta', type=float, default=2/3, help='the upper bound of neighborhood.')
    parser.add_argument('--m', type=int, default=5, help='number of scale copies.')
    parser.add_argument('--N', type=int, default=5, help='the number of sampled examples in the neighborhood.')
    
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    # parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', 
                        help='character label')
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
    print(opt)
    
    """ vocab / character number configuration """
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.device  = torch.device(device_type)
    
    if opt.sensitive:
        opt.character = string.printable[:62]

    opt.save_attacks = opt.save_attacks + opt.name
    opt.save_attacks = opt.save_attacks + "-eps" + str(int(opt.eps*255))
    
    cudnn.benchmark = True
    cudnn.deterministic = True

    torch.cuda.synchronize()
    transfer_attack(opt)