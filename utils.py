#!/usr/bin/python
# encoding: utf-8
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Logger(object):
    def __init__(self, filename = "train.log"):
        self.terminal =sys.stdout
        self.log = open(filename,"w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

""" utils for fawa """
def tensor2np(img): 
    trans = transforms.Grayscale(1)
    img = trans(img).squeeze(0)
    return img.detach().cpu().numpy()
def np2tensor(img: np.array):
    if len(img.shape) == 2:
        img_tensor = torch.from_numpy(img).float()  # bool to float
        img_tensor = img_tensor.unsqueeze_(0).repeat(3, 1, 1)
    if len(img.shape) == 3:
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2,0,1)
        if torch.max(img_tensor) <= 1:
            img_tensor = img_tensor * 255
        img_tensor = img_tensor / 255.
    return img_tensor.unsqueeze_(0).to(device)  # add batch dim

def get_text_mask(img: np.array):
    if img.max() <= 1:
        return img  < 1 / 1.25
    else:
        return img < 255 / 1.25

def cvt2Image(array):
    if array.max() <= 0.5:
        return Image.fromarray(((array + 0.5) * 255).astype('uint8'))
    elif array.max() <= 1:
        return Image.fromarray((array * 255).astype('uint8'))
    elif array.max() <= 255:
        return Image.fromarray(array.astype('uint8'))

def RGB2Hex(RGB): # RGB is a 3-tuple
        color = '#'
        for num in RGB:
            color += str(hex(num))[-2:].replace('x', '0').upper()
        return color

def color_map(grayscale):
    gray_map = (grayscale - 255*0.299 - 0*0.114) / 0.587
    return int(gray_map)


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. Note:in our dataset,label is list, and len=batch_size
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]  #index of char in text, shape=[len(text)]单词长度
            batch_text[i][:len(text)] = torch.LongTensor(text)

        return (batch_text.to(device), torch.IntTensor(length).to(device))  # [b, 25], list:b(16)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


