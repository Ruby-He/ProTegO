from .identity import Identity
from .transform import Translate, Resize, D_Binarization, R_Binarization
import torch.nn as nn
import random

class Combined(nn.Module):
	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list

	def forward(self, adv_image):
		id =  random.randint(0, len(self.list) - 1)
		print(f"[+] Batch Combined {self.list[id]}")
		return self.list[id](adv_image)

		