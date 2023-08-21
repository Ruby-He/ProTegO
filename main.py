import os
import sys
import time
import random
import string
import shutil
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from utils import Logger
from train_protego import run_train
from gen_udp import gen_underpaintings
from test_udp import test_udp


""" Basic parameters settings """
parser = argparse.ArgumentParser()
parser.add_argument('--manualSeed', type=int, default=3407, 
                        help='Refer to the settings in paper <Torch. manual_seed (3407) is all you need>')
parser.add_argument('--train_path', type= str, required=True, help='path of training dataset') 
parser.add_argument('--test_path', type= str, required=True, help='path of test dataset')
parser.add_argument('--up_path', type= str, default= "data/protego/up/5.png", 
                        help='path of the pre-processed underpaintings')
parser.add_argument('--dt_model', type=str, default='models/dbnet++.pth', help='path of our guided network DBnet++')
parser.add_argument('--batchsize', type= int, default=4, help='batchsize of training ProTegO')
parser.add_argument('--epochs', type= int, default=60, help='epochs of training ProTegO')
parser.add_argument('--eps', type=float, default=40/255, help='maximum perturbation')
parser.add_argument('--lambda1', type= float, default=1e-3, help='the weight of hinge_loss')
parser.add_argument('--lambda2', type= float, default=2, help='the weight of guide_loss')
parser.add_argument('--lambda3', type= float, default=1, help='the weight of gan_loss')
parser.add_argument('--lambda4', type= float, default=10, help='the weight of adv_loss')
parser.add_argument('--dark', action='store_true', help='use dark background and white text')
parser.add_argument('--b', action='store_true', help='robust test for both frames of adversarial text images')
parser.add_argument('--use_eh', action='store_true', help='Use enhancement layers')
parser.add_argument('--use_guide', action='store_true', help='use guided network')

""" Model Architecture """
parser.add_argument('--str_model', type=str, help="path of pretrainted STR models for evaluation",
                        default='STR_modules/downloads_models/STARNet-TPS-ResNet-BiLSTM-CTC-sensitive.pth')
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=3,
                    help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

""" Data processing """
parser.add_argument('--img_channel', type=int, default=3,
                        help='the number of input channel of text images')
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--character', type=str,default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_false', help='default for sensitive character mode')

""" Output settings """
parser.add_argument('--output', default=f'{time.strftime("res-%m%d-%H%M")}', 
                    help='the path to save all ouput results')
parser.add_argument('--train_output', default='train-out', help='the path to save intermediate training results')
parser.add_argument('--saveG', default='Generators', help='the path to save generators')
parser.add_argument('--loss', default='losses', help='the path to save all training losses')
parser.add_argument('--test_output', default='test-out', help='the path to save output of test results')
parser.add_argument('--adv_output', default='adv', help='the path to save adversarial text images')
parser.add_argument('--per_output', default='perturbation', help='the path to save adversarial perturbation')
parser.add_argument('--map_output', default='map', help='the path to save mapping results')

opt = parser.parse_args()
# print(opt)

""" Seed and GPU setting """
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
""" vocab / character number configuration """
if opt.sensitive:
    opt.character = string.printable[:62] # use 62 char (0~9, a~z, A~Z)

""" output configuration """""
output_path = opt.output
if not os.path.exists(output_path):
    os.makedirs(output_path)

# del all the output directories and files
del_list = os.listdir(output_path)
for f in del_list:
    file_path = os.path.join(output_path, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

""" save all the print content as log """
log_file= os.path.join(output_path, 'protego.log')
sys.stdout = Logger(log_file)

""" make all save directories """
train_adv_path = os.path.join(output_path, opt.train_output, 'adv')
train_per_path = os.path.join(output_path, opt.train_output, 'per')
Generator_path = os.path.join(output_path,opt.saveG)
loss_path = os.path.join(output_path, opt.loss)
test_output_path = os.path.join(output_path, opt.test_output)
adv_output_path1 = os.path.join(test_output_path, opt.adv_output, 'adv-')
adv_output_path2 = os.path.join(test_output_path, opt.adv_output, 'adv+')
per_output_path = os.path.join(test_output_path, opt.per_output)
map_output_path = os.path.join(test_output_path, opt.map_output)

if not os.path.exists(train_adv_path):
    os.makedirs(train_adv_path)
if not os.path.exists(train_per_path):
    os.makedirs(train_per_path)
if not os.path.exists(Generator_path):
    os.makedirs(Generator_path)
if not os.path.exists(loss_path):
    os.makedirs(loss_path)
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

torch.cuda.synchronize()
time_start = time.time()

print(opt)

# train ProTegO
run_train(opt, device, train_adv_path, train_per_path, Generator_path, loss_path)

# Generate adversarial underpaintings fot text images
gen_start = time.time()
gen_underpaintings(opt, device, Generator_path, adv_output_path1, adv_output_path2, per_output_path, map_output_path) 
gen_end = time.time() - gen_start
print('Generation time:' + str(gen_end))

# Test ProTegO performance
test_udp(opt, device, adv_output_path1, adv_output_path2, test_output_path)
time_end = time.time()
time_sum = time_end - time_start
print('Total time:' + str(time_sum))