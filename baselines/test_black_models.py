"""
    Test baseline methods on 4 Black-box models: CRNN, Rosetta, RARA, TRBA"""

import os, sys, time, string, argparse, shutil
sys.path.append('/data/hyr/ocr/ProTegO/release/')
import torch
from nltk.metrics import edit_distance
from dataset import test_adv_dataset
from STR_modules.model import Model
from utils import Logger, CTCLabelConverter, AttnLabelConverter

def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print('cannot create dirs: {}'.format(path))
            exit(0)

def process_line(line):
    adv_img_path, recog_result = line.split(':')
    label, adv_preds = recog_result.split('--->')
    adv_preds = adv_preds.strip('\n')
    return adv_preds, label, adv_img_path

def test(opt):
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt).to(opt.device)
    print('Loading a STR model from \"%s\" as the target model!' % opt.str_model)
    model.load_state_dict(torch.load(opt.str_model, map_location=opt.device),strict=False)
    model.eval()

    """ create new test model output path """
    makedirs(opt.output)

    # str_name = opt.str_model.split('/')[-2].split('-')[0]
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, opt.attack_name, str_name)
    attack_success_result = os.path.join(test_output_path , 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path , 'attack-success-adv')

    makedirs(test_output_path)
    makedirs(save_success_adv)

    log_file= os.path.join(test_output_path, 'test.log')
    sys.stdout = Logger(log_file)
    
    test_dataset = test_adv_dataset(opt.imgH, opt.imgW, opt.adv_img)  
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=1)
    
    result = dict()
    for i, data in enumerate(test_dataloader):     
        if opt.b:
            adv_img = data[0] 
        else:
            adv_img = data[1]
        adv_img= adv_img.to(opt.device)
        label = data[2]    
        adv_index = data[3][0]  
        adv_path = data[5][0]
        
        length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(opt.device)
        text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(opt.device)
        if 'CTC' in opt.Prediction:
            preds = model(adv_img, text_for_pred).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_output = converter.decode(preds_index.data, preds_size)
            preds_output = preds_output[0]
            result[adv_index] = '{}:{}--->{}\n'.format(adv_path, label[0], preds_output)
        else: # Attention
            preds = model(adv_img, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_output = converter.decode(preds_index, length_for_pred)
            preds_output = preds_output[0]
            preds_output = preds_output[:preds_output.find('[s]')]
            result[adv_index] = '{}:{}--->{}\n'.format(adv_path, label[0], preds_output)
    result = sorted(result.items(), key=lambda x:x[0])
    with open(attack_success_result, 'w+') as f:
        for item in result:
            f.write(item[1])

    # calculate ASR
    with open(attack_success_result, 'r') as f:
        alladv = f.readlines()
    attack_success_num, ED_sum = 0, 0
    for line in alladv:
        adv_preds, label, adv_img_path = process_line(line)
        
        if adv_preds != label:
            ED_num = edit_distance(label, adv_preds)
            attack_success_num += 1
            shutil.copy(adv_img_path, save_success_adv)
            ED_sum += ED_num
    attack_success_rate = attack_success_num / len(test_dataset)
    print('ASR:{:.2f} %'.format(attack_success_rate * 100))
    if attack_success_num != 0:
        ED_num_avr = ED_sum / attack_success_num
        print('Average Edit_distance: {:.2f}'.format(ED_num_avr))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='res-BlackModelTest/up5a', help='path to save attack results')
    parser.add_argument('--attack_name', required=True, help='baseline attack method name')
    """ Data processing """
    parser.add_argument('--adv_img', required=True, help='the path of adv_x which generated from STARNet model')
    parser.add_argument('--b', action='store_true', help='Use binarization processing to adv_img.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_channel', type=int, default=3, help='the number of input channel of image')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--character', type=str,default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_false', help='default for sensitive character mode')
    """ Model Architecture """
    parser.add_argument('--str_model', type=str, help='the model path of the target model',
                        default='/STR_modules/downloads_models/CRNN-None-VGG-BiLSTM-CTC-sensitive.pth')
    parser.add_argument('--Transformation', type=str, default='None', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='VGG', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    opt = parser.parse_args()
    print(opt)
    
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:62] # use 62 char (0~9, a~z, A~Z)

    torch.cuda.synchronize()
    time_st = time.time()
    test(opt)
    time_end = time.time()
    time_all = time_end - time_st
    print('Testing time:{:.2f}'.format(time_all))