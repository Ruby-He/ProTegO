import os
import sys
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from STR_modules.prediction import CTCLabelConverter, AttnLabelConverter
from STR_modules.model import Model
from dataset import strdataset, train_dataset_builder
from utils import Averager, Logger
from torchvision import utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print('cannot create dirs: {}'.format(path))
            exit(0)

def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0

    infer_time = 0
    valid_loss_avg = Averager()
    
    for i, data in enumerate(evaluation_loader):
        image_tensors, labels = data
        image = image_tensors.to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(device)
        text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        
        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            # if opt.sensitive:
            #     pred = pred.lower()
            #     gt = gt.lower()
            #     alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            #     out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            #     pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
            #     gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
            if pred == gt:
                n_correct += 1
                vutils.save_image(image, "{}/{}_{}_{}.png".format(opt.test_out, i, gt, i))  # 删选正确样本作为测试集  
                
                # if not opt.train_mode:
                #     print('GoundTruth: %-10s => Prediction: %-10s' % (gt, pred))
            if not opt.train_mode:
                print('Success:{},\t GoundTruth:{:20} => Prediction:{:20}'.format(pred == gt, gt, pred)) 
                  
            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(len(evaluation_loader)) * 100

    return valid_loss_avg.val(), accuracy, preds_str, confidence_score_list, labels, infer_time, len(evaluation_loader)


def test(opt):
    """ save all the print content as log """
    opt.test_out = os.path.join(opt.output, opt.name)
    makedirs(opt.test_out)
    # log_file= os.path.join(opt.test_out, 'test.log')
    # sys.stdout = Logger(log_file)

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt).to(device)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    # model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device),strict=False)
    # opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        # eval_dataset = strdataset(opt.imgH, opt.imgW, opt.eval_data)
        eval_dataset = train_dataset_builder(opt.imgH, opt.imgW, opt.eval_data)
        evaluation_loader = torch.utils.data.DataLoader(
                            eval_dataset, batch_size=opt.batch_size,
                            shuffle=False, num_workers=int(opt.workers),
                            # drop_last=True, pin_memory=True
                            )

        _, accuracy_by_best_model, _, _, _, _, _ = validation(
            model, criterion, evaluation_loader, converter, opt)
        print('SR:', f'{accuracy_by_best_model:0.2f}', '%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='Test output path')
    parser.add_argument('--name', required=True, help='Test model name')
    parser.add_argument('--train_mode', action='store_true', help='defalut is Test mode')
    parser.add_argument('--eval_data', type=str, required=True, help='path to evaluation dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_false', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_false', help='for sensitive character mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:62]  # use 62 char (0~9, a~z, A~Z)

    cudnn.benchmark = True
    cudnn.deterministic = True

    test(opt)