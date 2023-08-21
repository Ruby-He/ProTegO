import os
import sys
import argparse
import shutil
import string
import torch
from utils import CTCLabelConverter, AttnLabelConverter, Logger
from dataset import test_adv_dataset
from STR_modules.model import Model
from models.enhancer import Enhancer
from nltk.metrics import edit_distance



def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
def process_line(line):
    adv_img_path, recog_result = line.split(':')
    label, adv_preds = recog_result.split('--->')
    adv_preds = adv_preds.strip('\n')
    return adv_preds, label, adv_img_path

def test_udp(opt, device, adv_output_path1, adv_output_path2, test_output_path):
    batch_size = 1
    save_success_adv = os.path.join(test_output_path , 'attack-success-adv')
    save_binary = os.path.join(test_output_path , 'binary_adv')
    attack_success_result1 = os.path.join(test_output_path , 'attack_success_result1.txt')
    attack_success_result2 = os.path.join(test_output_path , 'attack_success_result2.txt') 
    if not os.path.exists(save_success_adv):
        os.makedirs(save_success_adv)
    if not os.path.exists(save_binary):
        os.makedirs(save_binary)

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt).to(device)
    print('Loading a STR model from \"%s\" as the target model!' % opt.str_model)
    model.load_state_dict(torch.load(opt.str_model, map_location=device),strict=False)
    model.eval()
    model.apply(fix_bn)

    mixed_layers = ["Combined([Identity(), Translate(), D_Binarization()])"]
    robust_test = Enhancer(mixed_layers).to(device)
    
    test_dataset1= test_adv_dataset(opt.imgH, opt.imgW, adv_output_path1)  
    test_dataloader1= torch.utils.data.DataLoader(
                    test_dataset1,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=1)

    result1 = dict() # adv-
    for i, data in enumerate(test_dataloader1):
        adv_img1 = data[1]
        label1 = data[2]      
        adv_index1 = data[3][0]  
        adv_path1 = data[5][0]     
        if opt.b:
            adv_img1 = robust_test(adv_img1)
        else:
            adv_img1

        adv_img1= adv_img1.to(device)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        if 'CTC' in opt.Prediction:
            preds1 = model(adv_img1, text_for_pred).log_softmax(2)
            preds_size1 = torch.IntTensor([preds1.size(1)] * batch_size)
            _, preds_index1 = preds1.permute(1, 0, 2).max(2)
            preds_index1 = preds_index1.transpose(1, 0).contiguous().view(-1)
            preds_output1 = converter.decode(preds_index1.data, preds_size1)
            preds_output1 = preds_output1[0]
            result1[adv_index1] = '{}:{}--->{}\n'.format(adv_path1, label1[0], preds_output1)
        else: # Attention
            preds1 = model(adv_img1, text_for_pred, is_train=False)
            _, preds_index1 = preds1.max(2)
            preds_output1 = converter.decode(preds_index1, length_for_pred)
            preds_output1 = preds_output1[0]
            preds_output1 = preds_output1[:preds_output1.find('[s]')]
            result1[adv_index1] = '{}:{}--->{}\n'.format(adv_path1, label1[0], preds_output1)
    result1 = sorted(result1.items(), key=lambda x:x[0])
    with open(attack_success_result1, 'w+') as f:
        for item in result1:
            f.write(item[1])

    
    test_dataset2= test_adv_dataset(opt.imgH, opt.imgW, adv_output_path2) 
    test_dataloader2= torch.utils.data.DataLoader(
                    test_dataset2,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4)
    result2 = dict() # adv+   
    for i, data in enumerate(test_dataloader2):
        adv_img2= data[1]
        label2 = data[2]
        adv_index2 = data[3][0]
        adv_path2 = data[5][0]

        if opt.b:
            adv_img2 = robust_test(adv_img2)
        else:
            adv_img2

        adv_img2= adv_img2.to(device)
        if 'CTC' in opt.Prediction:
            preds2 = model(adv_img2, text_for_pred).log_softmax(2)
            preds_size2 = torch.IntTensor([preds2.size(1)] * batch_size)
            _, preds_index2 = preds2.permute(1, 0, 2).max(2)
            preds_index2 = preds_index2.transpose(1, 0).contiguous().view(-1)
            preds_output2 = converter.decode(preds_index2.data, preds_size2)
            preds_output2 = preds_output2[0]
            result2[adv_index2] = '{}:{}--->{}\n'.format(adv_path2, label2[0], preds_output2)
        else:
            preds2 = model(adv_img2, text_for_pred, is_train=False)
            _, preds_index2 = preds2.max(2)
            preds_output2 = converter.decode(preds_index2, length_for_pred)
            preds_output2 = preds_output2[0]
            preds_output2 = preds_output2[:preds_output2.find('[s]')]
            result2[adv_index2] = '{}:{}--->{}\n'.format(adv_path2, label2[0], preds_output2)
    result2 = sorted(result2.items(), key=lambda x:x[0])
    with open(attack_success_result2, 'w+') as f:
        for item in result2:
            f.write(item[1])
    
    # calculate ASR
    with open(attack_success_result1, 'r') as f:
        alladv1 = f.readlines()
    with open(attack_success_result2, 'r') as f:
        alladv2 = f.readlines()
    
    attack_success_num,asc1,asc2, = 0, 0, 0

    ED_num1, ED_num2 = 0, 0
    for line1, line2 in zip(alladv1, alladv2):
        adv_preds1,label1,adv_img_path1 = process_line(line1)
        adv_preds2,label2,adv_img_path2 = process_line(line2)
        if adv_preds1 != label1:
            asc1 +=1
        if adv_preds2 != label2:
            asc2 +=1
        if adv_preds1 != label1 and adv_preds2 != label2:
            ED_num1 += edit_distance(label1, adv_preds1)
            ED_num2 += edit_distance(label2, adv_preds2)
            attack_success_num += 1
            shutil.copy(adv_img_path1, save_success_adv)
            shutil.copy(adv_img_path2, save_success_adv)
    print("***********Test Finished !***********")
    psr1 = asc1 / len(test_dataset1)
    psr2 = asc2 / len(test_dataset2)
    attack_success_rate = attack_success_num / len(test_dataset1)
    print('PSR1:{:.2%}'.format(psr1))
    print('PSR2:{:.2%}'.format(psr2))
    print('PSR:{:.2%}'.format(attack_success_rate))
    if attack_success_num != 0:
        ED_num1_avr = ED_num1 / attack_success_num
        ED_num2_avr = ED_num2 / attack_success_num
        ED_avr = (ED_num1_avr + ED_num2_avr) / 2
        print('Average Edit_distance-: {:.2f}'.format(ED_num1_avr))
        print('Average Edit_distance+: {:.2f}'.format(ED_num2_avr))
        print('Average Edit_distance-2: {:.2f}'.format(ED_avr))

  
    
if __name__ == '__main__':
    """transfer attack for black-box models"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='the path of the white-box model (STARNet) results')
    parser.add_argument('--STR_name', required=True, help='the path to save ouput results of different models')
    parser.add_argument('--saveG', default='Generators', help='the path to save generator which is used for generated AEs')
    parser.add_argument('--adv_output', default='adv', help='the path to save adversarial examples results')
    parser.add_argument('--per_output', default='perturbation', help='the path to save output of adversarial perturbation')
    parser.add_argument('--up_output', default='up', help='the path to save underpainting and mapping results')
    parser.add_argument('--b', action='store_true', help='Use binarization processing to test AEs.')
    """ Data processing """
    parser.add_argument('--img_channel', type=int, default=3, help='the number of input channel of image')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--character', type=str,default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_false', help='default for sensitive character mode')
    """ Model Architecture """
    parser.add_argument('--str_model', type=str, required=True, help='the model path of the target model')
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    opt = parser.parse_args()
    print(opt)

    """ create new test model output path """
    if opt.b:
        test_output_path = os.path.join(opt.output, opt.STR_name, 'RB')
    else:
        test_output_path = os.path.join(opt.output, opt.STR_name)
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)
    
    log_file= os.path.join(test_output_path, 'test.log')
    sys.stdout = Logger(log_file)
    
    """ already exist """
    Generator_path = os.path.join(opt.output, opt.saveG)
    adv_output_path1 = os.path.join(opt.output, 'test-out', opt.adv_output, 'adv-')
    adv_output_path2 = os.path.join(opt.output, 'test-out', opt.adv_output, 'adv+')
    per_output_path = os.path.join(opt.output, 'test-out', opt.per_output)
    up_output_path = os.path.join(opt.output, 'test-out', opt.up_output)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:62] # use 62 char (0~9, a~z, A~Z)

    test_udp(opt, device, adv_output_path1, adv_output_path2, test_output_path)