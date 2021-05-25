from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
import random
import math
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch import nn, optim

from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torchvision
from .model import StyledGenerator, Discriminator
# from .model_org import StyledGenerator, Discriminator

class StyleGAN():

	def __init__(self, lr):
		code_size = 512
		# generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
		# discriminator = nn.DataParallel( Discriminator(from_rgb_activate=True)).cuda()
		# g_running = StyledGenerator(code_size).cuda()
		# g_running.train(False)

		self.netG = StyledGenerator(code_size).cuda()
		self.netD = Discriminator(from_rgb_activate=True).cuda()
		self.g_running = StyledGenerator(code_size).cuda()
		self.g_running.train(False)

		self.learningRate = lr
		self.optimizerD = self.getOptimizerD()
		self.optimizerG = self.getOptimizerG()
		# self.one = torch.FloatTensor([1]).cuda()
		self.one = torch.tensor(1, dtype=torch.float).cuda()
		self.mone = (self.one * -1).cuda()


	def optimizeParameters(self, input_batch, inputLabels=None):
		pass

	def getOptimizerG(self):
		g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.netG.generator.parameters()),
						  betas=[0.0, 0.999], lr=self.learningRate)
		g_optimizer.add_param_group(
			{
				'params': self.netG.style.parameters(),
				'lr': self.learningRate,
				'mult': 0.01,
			}
		)
		return g_optimizer

	def getOptimizerD(self):
		return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
						  betas=[00, 0.999], lr=self.learningRate)

	def accumulate(self, model1, model2, decay=0.999):
		par1 = dict(model1.named_parameters())
		par2 = dict(model2.named_parameters())

		for k in par1.keys():
			par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag

	def getSize(self):
		size = 2**(self.config.depth + 3)
		return (size, size)
