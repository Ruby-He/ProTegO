import os
import argparse
import cv2
import numpy as np
import torch
from torchvision import utils as vutils

import models.GAN_models as G_models
from dataset import test_dataset_builder, up_dataset

def gen_underpaintings(opt, device, Generator_path, adv_output_path1, adv_output_path2, per_output_path, map_output_path): 
    BOX_MIN = 0
    BOX_MAX = 255

    # load the well-trained generator 
    pretrained_generator_path = os.path.join(Generator_path + '/netG_epoch_' + str(opt.epochs) + '.pth')
    pretrained_G = G_models.Generator(opt.img_channel).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    test_dataset = test_dataset_builder(opt.imgH, opt.imgW, opt.test_path)
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=1,
                    shuffle=False, num_workers=4)

    up = up_dataset(opt.up_path).to(device)
    up = up.repeat(1,1,1,1)
    gui_net = torch.load(opt.dt_model).to(device)
    gui_net.eval()

    for i, data in enumerate(test_dataloader, 0):
        ori_labels = data[1][0]
        img_index = data[3][0]
        test_img = data[5]
        test_img = test_img.to(device)
        mask = test_img.detach().to(device)
        if opt.dark:
            test_img = (1-test_img) + torch.mul(mask, up)
        else:
            test_img = torch.mul(test_img, up)
        
        # gen adv_img
        test_map = G_models.guided_net(test_img, gui_net)
        vutils.save_image(test_map,"{}/{}_{}_map.png".format(map_output_path, img_index, ori_labels))

        perturbation = pretrained_G(up)
        perturbation = torch.clamp(perturbation, -opt.eps, opt.eps)
        vutils.save_image(perturbation, "{}/{}_{}_per.png".format(per_output_path, img_index, ori_labels))

        permap = G_models.guided_net(perturbation, gui_net)
        vutils.save_image(permap, "{}/{}_{}_permap.png".format(map_output_path, img_index, ori_labels))
        vutils.save_image(permap*100, "{}/{}_{}_permap100.png".format(map_output_path, img_index, ori_labels))
        perturbation = torch.mul(mask, perturbation)
  
        """convert float32 to uint8:
        Avoid the effect of float32 on 
        generating fully complementary frames
        """
        perturbation_int = (perturbation*255).type(torch.int8)
        adv_img1_uint = (test_img*255).type(torch.uint8) - perturbation_int
        adv_img1_uint = torch.clamp(adv_img1_uint, BOX_MIN, BOX_MAX)
        adv_img2_uint = (test_img*255).type(torch.uint8) + perturbation_int
        adv_img2_uint = torch.clamp(adv_img2_uint, BOX_MIN, BOX_MAX)
        print((adv_img1_uint + perturbation_int).equal(adv_img2_uint - perturbation_int))
        
        adv1_map = G_models.guided_net(adv_img1_uint/255, gui_net)
        adv2_map = G_models.guided_net(adv_img2_uint/255, gui_net)
        vutils.save_image(adv1_map,"{}/{}_{}_map-.png".format(map_output_path, img_index, ori_labels))
        vutils.save_image(adv2_map,"{}/{}_{}_map+.png".format(map_output_path, img_index, ori_labels))
        
        adv_img1_uint = adv_img1_uint.squeeze(0).permute(1,2,0)
        adv_img1_uint = np.uint8(adv_img1_uint.cpu())
        adv_img1 = cv2.cvtColor(adv_img1_uint, cv2.COLOR_RGB2BGR)
        adv_img2_uint = adv_img2_uint.squeeze(0).permute(1,2,0)
        adv_img2_uint = np.uint8(adv_img2_uint.cpu())
        adv_img2 = cv2.cvtColor(adv_img2_uint, cv2.COLOR_RGB2BGR)

        cv2.imwrite("{}/{}_{}_adv-.png".format(adv_output_path1, img_index, ori_labels), adv_img1)
        cv2.imwrite("{}/{}_{}_adv+.png".format(adv_output_path2, img_index, ori_labels), adv_img2)
    
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type= str, required=True, help='path of font test dataset')
    parser.add_argument('--up_path', type= str, default='data/protego/up/5.png', help='underpaintings path')
    parser.add_argument('--dark', action='store_true', help='use dark background and white text.')
    parser.add_argument('--dt_model', type=str, default='/models/dbnet++.pth', 
                        help='path of our guided network DBnet++')
    parser.add_argument('--batchsize', type= int, default=4, help='batchsize of training ProTegO')
    parser.add_argument('--epochs', type= int, default=60, help='epochs of training ProTegO')
    parser.add_argument('--eps', type=float, default=40/255, help='maximum perturbation')
    parser.add_argument('--use_eh', action='store_true', help='Use enhancement layers')
    parser.add_argument('--use_guide', action='store_true', help='use guided network')
    parser.add_argument('--img_channel', type=int, default=3,
                            help='the number of input channel of text images')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--character', type=str,default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_false', help='default for sensitive character mode')

    parser.add_argument('--output', default='res-font', help='the path to save all ouput results')
    parser.add_argument('--test_output', default='test-out', help='the path to save output of test results')
    parser.add_argument('--adv_output', default='adv', help='the path to save adversarial text images')
    parser.add_argument('--per_output', default='perturbation', help='the path to save output of adversarial perturbation')
    parser.add_argument('--map_output', default='map', help='the path to save mapping results')
    parser.add_argument('--train_output', default='train-out', help='the path to save output of intermediate training results')
    
    parser.add_argument('--saveG', required=True, help='the path to save generator which is used for generated AEs')
    
    opt = parser.parse_args()
    print(opt)

    output_path = opt.output
    Generator_path = opt.saveG

    font_name = opt.test_path.split('/')[-1]
    test_output_path = os.path.join(output_path, font_name, opt.test_output)
    adv_output_path1 = os.path.join(test_output_path, opt.adv_output, 'adv-')
    adv_output_path2 = os.path.join(test_output_path, opt.adv_output, 'adv+')
    per_output_path = os.path.join(test_output_path, opt.per_output)
    map_output_path = os.path.join(test_output_path, opt.map_output)
    
    
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)
    if not os.path.exists(adv_output_path1):
        os.makedirs(adv_output_path1)
    if not os.path.exists(adv_output_path2):
        os.makedirs(adv_output_path2)
    if not os.path.exists(per_output_path):
        os.makedirs(per_output_path)
    if not os.path.exists(map_output_path):
        os.makedirs(map_output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_underpaintings(opt, device, Generator_path, adv_output_path1, adv_output_path2, per_output_path, map_output_path)

