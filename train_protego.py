import torch

from protego import framework
from STR_modules.model import Model

from dataset import train_dataset_builder
from utils import CTCLabelConverter, AttnLabelConverter


def run_train(opt, device, train_adv_path, train_per_path, Generator_path, loss_path):
    """ data preparing """
    train_dataset = train_dataset_builder(opt.imgH, opt.imgW, opt.train_path)
    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=opt.batchsize,
                    shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt).to(device)
    print('Loading STR pretrained model from %s' % opt.str_model)
    model.load_state_dict(torch.load(opt.str_model, map_location=device),strict=False)
    model.eval()
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
   
    """ attack setting """
    ProTegO = framework(device, model, opt.dt_model, converter, criterion, opt.batch_max_length, 
                opt.up_path, opt.dark, opt.batchsize, opt.img_channel, opt.imgH, opt.imgW, 
                opt.eps, opt.lambda1, opt.lambda2, opt.lambda3, opt.lambda4,
                opt.use_eh, opt.use_guide)

    # train
    ProTegO.train(train_dataloader, opt.epochs, train_adv_path, train_per_path, Generator_path, loss_path)


