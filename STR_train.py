import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from STR_modules.prediction import CTCLabelConverter, AttnLabelConverter
from STR_modules.model import Model
from STR_test import validation
from dataset import strdataset
from utils import Averager



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
   
    """ dataset preparation """
    train_dataset = strdataset(opt.imgH, opt.imgW, opt.train_data)
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=opt.batch_size,
                        shuffle=True, num_workers=int(opt.workers),
                        drop_last=True, pin_memory=True)


    valid_dataset = strdataset(opt.imgH, opt.imgW, opt.valid_data)
    valid_loader = torch.utils.data.DataLoader(
                        valid_dataset, batch_size=opt.batch_size,
                        shuffle=False, num_workers=int(opt.workers),
                        drop_last=True, pin_memory=True)

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt).to(device)

     # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # # data parallel for multi-GPU
    # model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./STR_modules/saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)
    
    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass
    
    start_time = time.time()
    best_accuracy = -1
    iteration = start_iter
    while(True):
        # train part 
        for i, data in enumerate(train_loader):
            image_tensors, labels = data
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)

            # validation part
            if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
                elapsed_time = time.time() - start_time
                # for log
                with open(f'./STR_modules/saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                    model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, preds, confidence_score, labels, infer_time, length_of_data = validation(
                            model, criterion, valid_loader, converter, opt)
                    model.train()

                    # training loss and validation loss
                    loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                    loss_avg.reset()

                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}'

                    # keep best accuracy model (on valid dataset)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'./STR_modules/saved_models/{opt.exp_name}/best_accuracy.pth')
                    
                    best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'

                    loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                    print(loss_model_log)
                    log.write(loss_model_log + '\n')

                    # show some predicted results
                    dashed_line = '-' * 80
                    head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                    predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                    for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                        if 'Attn' in opt.Prediction:
                            gt = gt[:gt.find('[s]')]
                            pred = pred[:pred.find('[s]')]

                        predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                    predicted_result_log += f'{dashed_line}'
                    print(predicted_result_log)
                    log.write(predicted_result_log + '\n')

            # save model per 1e+4 iter.
            if (iteration + 1) % 1e+4 == 0:
                torch.save(
                    model.state_dict(), f'./STR_modules/saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

            if (iteration + 1) == opt.num_iter:
                print('end the training')
                sys.exit()
            iteration += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, help='Where to store logs and models')
    parser.add_argument('--train_mode', action='store_true', help='defalut is test mode')
    parser.add_argument('--train_data', type=str, required=True, help='path to training dataset')
    parser.add_argument('--valid_data', type=str, required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=2023, help='for random seed setting')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=5e+4, help='number of iterations to train for') #train 3e+4, finetune 1e+4
    parser.add_argument('--valInterval', type=int, default=1000, help='Interval between each validation')
    parser.add_argument('--saved_model', type=str, default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    # parser.add_argument('--rgb', action='store_false', help='default to use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_false', help='default for sensitive character mode')
    """ Model Architecture """
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


    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        # opt.exp_name += f'-Seed{opt.manualSeed}'
        if opt.sensitive:
            opt.exp_name += f'-sensitive'
        print(opt.exp_name)
    else:
        opt.exp_name = f'{opt.exp_name}-{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        # opt.exp_name += f'-Seed{opt.manualSeed}'
        if opt.sensitive:
            opt.exp_name += f'-sensitive'
        print(opt.exp_name)
    
    if opt.FT:
        pass
    else:
        os.makedirs(f'./STR_modules/saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:62] # use 62 char (0~9, a~z, A~Z)

    """ Seed and GPU setting """
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    train(opt)
