import torch, torchvision
import numpy as np
from . import constants, stylegan
from utils import image
import torch.nn as nn
from torch.autograd import Variable, grad
# from .gradient_penalty import gradient_penalty
from torch import nn, optim
import functools
import torch.nn.functional as F
from . import constants


class walk_embed(nn.Module):
	def __init__(self, dim_z, Nsliders, attrList):
		super(walk_embed, self).__init__()
		self.dim_z = dim_z
		self.Nsliders = Nsliders
		self.w = nn.ParameterDict()
		for i in attrList:
			self.w.update(
				{i: nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [6, 1, self.dim_z, Nsliders])).cuda())})

	def forward(self, z, name, alpha, index_):
		z_new = z  # .cpu()
		for i in range(self.Nsliders):
			z_new = z_new + self.w[name][index_, :, :, i]
		# al = torch.unsqueeze(alpha[:, i], axis=1)
		# z_new = z_new + al.cpu() * self.w_embed[name][index_, :, :, i]
		return z_new

#
# class walk_linear_w(nn.Module):
# 	def __init__(self, dim_z, step, Nsliders, attrList):
# 		super(walk_linear_w, self).__init__()
# 		self.dim_z = dim_z
# 		self.step = step
# 		self.Nsliders = Nsliders
#
# 		self.w = nn.ParameterDict()
# 		for i in attrList:
# 			self.w.update(
# 				{i: nn.Parameter(
# 					torch.Tensor(np.random.normal(0.0, 0.02, [(self.step + 1) * 2, self.dim_z, Nsliders])).cuda())})
#
# 	def forward(self, input, layers, name, alpha, index_):
#
# 		w_transformed = []
# 		if layers == None:
# 			print('Layers == None in walk')
# 			al = torch.unsqueeze(alpha[:, 0], axis=1)
# 			for i in range(len(input)):
# 				# w_new = input[i] + al * self.w[name][i, :, 0]
# 				w_new = input[i] + al * self.w[name][0, :, 0]
# 				w_transformed.append(w_new)
# 			print('before: ', input[0][0, :5])
# 			print('after: ', w_transformed[0][0, :5])
# 			return w_transformed
#
# 		for i in range(len(input)):
# 			if i in layers:
# 				al = torch.unsqueeze(alpha[:, 0], axis=1)
# 				w_new = input[i] + al * self.w[name][i, :, 0]
# 			else:
# 				w_new = input[i]
#
# 			w_transformed.append(w_new)
# 		return w_transformed

class Normalization(nn.Module):
	def __init__(self):
		super(Normalization, self).__init__()
		mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
		std = torch.tensor([0.229, 0.224, 0.225]).cuda()

		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, img):
		return (img - self.mean) / self.std


class ContentLoss(nn.Module):
	def __init__(self):
		super(ContentLoss, self).__init__()

	def forward(self, org, shifted):
		self.loss = F.mse_loss(org.detach(), shifted)
		return self.loss


class walk_linear(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(walk_linear, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		self.w = nn.ParameterDict()
		for i in attrList:
			self.w.update(
				{i: nn.Parameter(torch.Tensor(np.random.normal(0.0, 0.02, [1, self.dim_z, Nsliders])).cuda())})

	def forward(self, z, name, alpha, index_):
		z_new = z.cpu()
		for i in range(self.Nsliders):
			al = torch.unsqueeze(alpha[:, i], axis=1)
			z_new = z_new + al.cpu() * self.w[name][:, :, i].cpu()
		return z_new.cuda()


class walk_mlp_multi_z(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(walk_mlp_multi_z, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders
		direction = np.zeros((1, 10))
		direction[0, 0] = 1
		self.direction = torch.Tensor(direction).cuda()
		self.embed = nn.Linear(10, self.dim_z)
		self.linear = nn.Sequential(*[nn.Linear(self.dim_z + self.dim_z, self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z, self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(self.dim_z, self.dim_z)])

	def forward(self, input, name, alpha, index_, layers=None):
		al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()  # Batch, 1

		out = self.embed(self.direction.repeat(al.size(0), 1))
		print('MLP z')
		out2 = self.linear(torch.cat([out, input], 1))
		out2 = al * out2 / torch.norm(out2, dim=1, keepdim=True)
		z_new = input + out2

		return z_new


class walk_linear_single_w(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(walk_linear_single_w, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		self.w = nn.ParameterDict()
		for i in attrList:
			self.w.update(
				{i: nn.Parameter(
					torch.Tensor(np.random.normal(0.0, 0.1, [1, self.dim_z, Nsliders])).cuda())})

	def forward(self, input, name, alpha, index_, layers=None):
		output = []
		for i in range(len(input)):
			al = torch.unsqueeze(alpha[:, i], axis=1)
			# print('before: ', input[i].size(), al)
			out = input[i] + al * self.w[name][:, :, 0]
			output.append(out)
		return output


class walk_linear_multi_w(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(walk_linear_multi_w, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		self.w = nn.ParameterDict()
		for i in attrList:
			self.w.update(
				{i: nn.Parameter(
					torch.Tensor(np.random.normal(0.0, 0.02, [(self.step + 1) * 2, self.dim_z, Nsliders])).cuda())})

	def forward(self, input, name, alpha, index_, layers=None):
		w_transformed = []
		al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()
		if layers == None:
			for i in range(len(input)):
				# out = input[i] + al * self.w[name][i, :, 0]
				# Unit norm
				# print('norm: ', self.w[name][i, :, 0].size(), torch.norm(self.w[name][i, :, 0], dim=0, keepdim=True).size())
				# out = input[i] + al * self.w[name][i, :, 0] / torch.norm(self.w[name][i, :, 0], dim=0, keepdim=True)
				out = input[i] + al * self.w[name][i, :, 0]
				w_transformed.append(out)
			return w_transformed

		# select layers
		for i in range(len(input)):
			if i in layers:
				print('in the layer', i, )
				out = input[i] + al * self.w[name][i, :, 0]
			else:
				out = input[i]
			w_transformed.append(out)
		return w_transformed


class walk_mlp_multi_w(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(walk_mlp_multi_w, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		self.linear = nn.Sequential(*[nn.Linear(self.dim_z, 2 * self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(2 * self.dim_z, self.dim_z)])

	def forward(self, input, name, alpha, index_, layers=None):

		w_transformed = []
		al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()  # Batch, 1

		if layers == None:
			for i in range(len(input)):
				out2 = self.linear(input[i])
				# out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
				w_new = input[i] + al * out2
				# w_new = torch.clamp(w_new, min=-1, max=2)
				w_transformed.append(w_new)
			return w_transformed

		for i in range(len(input)):
			if i in layers:
				out2 = self.linear(input[i], 1)
				# out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
				w_new = input[i] + al * out2
			else:
				w_new = input[i]
			w_transformed.append(w_new)
		return w_transformed

class walk_nonlinear_w(nn.Module):
	def __init__(self, dim_z, step, Nsliders, attrList):
		super(walk_nonlinear_w, self).__init__()
		self.dim_z = dim_z
		self.step = step
		self.Nsliders = Nsliders

		# self.w = nn.ParameterDict()
		# for i in attrList:
		# 	self.w.update(
		# 		{i: nn.Parameter(
		# 			torch.Tensor(np.random.normal(0.0, 0.1, [(self.step + 1 ) * 2, self.dim_z, Nsliders])).cuda())})

		self.embed = nn.Linear(10, self.dim_z // 2)
		self.linear = nn.Sequential(*[nn.Linear(self.dim_z // 2 + self.dim_z, 2 * self.dim_z),
									  nn.LeakyReLU(0.2, True),
									  nn.Linear(2 * self.dim_z, self.dim_z)])

	def forward(self, input, name, alpha, index_, layers=None):
		w_transformed = []
		al = torch.unsqueeze(alpha[:, 0], axis=1).cuda()  # Batch, 1
		out = self.embed(al.repeat(1, 10))
		print('Nonlinear MLP w')
		if layers == None:
			for i in range(len(input)):
				# print('Max min before: ', input[i].max(), input[i].min())

				out2 = self.linear(torch.cat([out, input[i]], 1))
				out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
				w_new = input[i] + out2
				# w_new = torch.clamp(w_new, min=-1, max=2)
				w_transformed.append(w_new)

			# print('Max min after: ', w_new.max(), w_new.min())

			return w_transformed

		for i in range(len(input)):
			if i in layers:
				out2 = self.linear(torch.cat([out, input[i]], 1))
				# out2 = out2 / torch.norm(out2, dim=1, keepdim=True)
				w_new = input[i] + out2
			else:
				w_new = input[i]
			w_transformed.append(w_new)
		return w_transformed


class TransformGraph():
	def __init__(self, lr, walk_type, nsliders, loss_type, eps, N_f,
				 trainEmbed, attrList, layers, stylegan_opts):
		assert (loss_type in ['l2', 'lpips']), 'unimplemented loss'

		# module inputs
		self.lr = lr
		self.useGPU = constants.useGPU
		self.module = self.get_stylegan_module()
		self.one = torch.tensor(1, dtype=torch.float).cuda()
		self.mone = (self.one * -1).cuda()
		self.regressor, self.reg_optmizer = self.get_reg_module()
		self.vgg19 = self.get_vgg_module()

		self.attrTabel = {
			'daylight': 1, 'night': 2, 'sunrisesunset': 3, 'sunny': 5,
			'clouds': 6, 'fog': 7, 'snow': 9, 'warm': 10, 'cold': 11,
			'beautiful': 13, 'flowers': 14, 'spring': 15, 'summer': 16,
			'autumn': 17, 'winter': 18, 'colorful': 20, 'dark': 24,
			'bright': 25, 'rain': 29, 'boring': 37, 'lush': 39}

		# self.module.netG.train(False)
		# self.module.netD.train(False)
		# self.regressor.train(False)

		self.module.netG.eval()
		self.module.netD.eval()
		self.regressor.eval()

		self.reg_criterion = nn.MSELoss().cuda()
		# self.reg_criterion = BCEloss()

		self.dim_z = constants.DIM_Z
		self.Nsliders = Nsliders = nsliders
		self.img_size = constants.resolution
		self.num_channels = constants.NUM_CHANNELS
		# self.CRITIC_ITERS = CRITIC_ITERS = constants.CRITIC_ITERS
		# self.OUTPUT_DIM = constants.OUTPUT_DIM
		self.BATCH_SIZE = constants.BATCH_SIZE
		self.LAMBDA = 0.05

		self.BCE_loss = nn.BCELoss()
		self.BCE_loss_logits = nn.BCEWithLogitsLoss()
		self.MSE_loss = nn.MSELoss()
		self.ContentLoss = ContentLoss()

		self.trainEmbed = trainEmbed
		self.attrList = attrList

		# StyleGAN 256
		self.step = 6
		self.alpha = 1
		self.stylegan_opts = stylegan_opts
		self.layers = layers

		self.is_mlp = False
		# self.is_single = False

		# walk pattern
		print('walk_type and tylegan_opts.latent: ', walk_type, stylegan_opts.latent)

		if walk_type == 'linear':
			if self.trainEmbed == True:
				print('Walk in non-linear embed')
				self.walk = walk_embed(self.dim_z, Nsliders, self.attrList)
			else:
				if stylegan_opts.latent == 'z':
					# self.walk = walk_linear(self.dim_z, self.step, Nsliders, self.attrList).cuda()
					if self.is_mlp:
						self.walk = walk_mlp_multi_z(self.dim_z, self.step, Nsliders, self.attrList).cuda()
						self.optimizers = torch.optim.Adam(self.walk.parameters(),
														   lr=self.lr,
														   betas=(0.5, 0.99))

				elif stylegan_opts.latent == 'w':
					# self.walk = walk_linear_single_w(self.dim_z, self.step, Nsliders, self.attrList)
					self.walk = walk_linear_multi_w(self.dim_z, self.step, Nsliders, self.attrList)
					if self.is_mlp:
						self.walk = walk_mlp_multi_w(self.dim_z, self.step, Nsliders, self.attrList).cuda()
				else:
					raise NotImplementedError('Not implemented latent walk type:' '{}'.format(stylegan_opts.latent))
		elif 'NN' in walk_type:
			self.walk = walk_nonlinear_w(self.dim_z, self.step, Nsliders, self.attrList).cuda()

		# self.optimizers = torch.optim.Adam(self.walk.parameters(),
		# 								   lr=self.lr,
		# 								   betas=(0.5, 0.99))

		self.optimizers = {}
		for i in self.attrList:
			self.optimizers[i] = torch.optim.Adam([self.walk.w[i]],
												  lr=self.lr,
												  betas=(0.5, 0.99))

		self.walk.train()
		# # set class vars
		self.Nsliders = Nsliders

		self.y = None
		self.z = None
		self.truncation = None

		self.walk_type = walk_type
		self.N_f = N_f  # NN num_steps
		self.eps = eps  # NN step_size

	def get_logits(self, inputs_dict, reshape=True):

		if self.stylegan_opts.latent == 'z':
			outputs_orig = self.module.netG(inputs_dict['z'],
											step=self.step,
											alpha=self.alpha)

		elif self.stylegan_opts.latent == 'w':
			outputs_orig = self.module.netG(inputs_dict['w'],
											step=self.step,
											alpha=self.alpha)

		return outputs_orig

	def get_z_new(self, z, alpha):
		if self.walk_type == 'linear' or self.walk_type == 'NNz':
			z_new = z
			for i in range(self.Nsliders):
				# TODO: PROBLEM HERE
				al = torch.unsqueeze(torch.Tensor(alpha[:, i]), axis=1)
				z_new = (z_new + al * self.w[:, :, i]).cuda()
		return z_new

	def get_z_new_tensor(self, z, alpha, name=None, trainEmbed=False, index_=None):
		z = z.squeeze()
		z_new = self.walk(z, name, alpha, index_)
		# return z_new.cuda()
		return z_new

	def get_w(self, z, is_single=False):
		# SINGLE W
		w = self.module.netG.style(z)
		if is_single:
			return [w]

		multi_ws = [w] * (self.step + 1) * 2
		return multi_ws

	def get_w_new_tensor(self, multi_ws, alpha, layers=None,
						 name=None, trainEmbed=False, index_=None):
		"""
		For single w and single transformation
		"""
		multi_ws_new = self.walk(multi_ws, name, alpha, index_, layers=layers)

		# multi_ws_new = self.walk(multi_ws, layers, name, alpha, index_)
		# print('tenser2: ', multi_ws_new.size())
		return multi_ws_new

	def get_edit_loss(self, feed_dict):
		# L2 loss
		target = feed_dict['target']
		mask = feed_dict['mask_out']
		logit = feed_dict['logit']
		diff = (logit - target) * mask
		return torch.sum(diff.pow(2)) / torch.sum(mask)

	def get_reg_preds(self, logit):
		preds = self.regressor(logit)[:, self.attrTabel[self.attrList[0]]]
		preds = preds.unsqueeze(1)
		return preds

	def get_alphas(self, alpha_org, alpha_delta):
		alpha_target = torch.clamp(alpha_org + alpha_delta, min=0, max=1)
		alpha_delta_new = alpha_target - alpha_org
		return alpha_target, alpha_delta_new

	def BCEloss(self, pred, y, eps=1e-12):
		loss = -(y * pred.clamp(min=eps).log() + (1 - y) * (1 - pred).clamp(min=eps).log()).mean()
		return loss

	def get_reg_loss(self, feed_dict):
		logit = feed_dict['logit']
		alpha_gt = feed_dict['alpha'].to(torch.double)
		# print(self.attrList, self.attrTabel[self.attrList[0]])
		preds = self.regressor(logit)[:, self.attrTabel[self.attrList[0]]]
		preds = preds.unsqueeze(1).to(torch.double)
		# loss = self.BCEloss(preds, alpha_gt)
		loss = self.reg_criterion(preds, alpha_gt)
		return loss.mean()

	def get_content_loss(self, org_img, shifted_img):
		# content_layers = ['conv_4']
		content_layers = [ 'conv_1', 'conv_2', 'conv_3']  # 'conv_0',# , 'conv_4'

		# norm = Normalization().cuda()
		# model = nn.Sequential(norm)

		model = nn.Sequential()
		i = 0
		content_losses = []
		for layer in self.vgg19.children():
			if isinstance(layer, nn.Conv2d):
				i += 1
				name = 'conv_{}'.format(i)
			elif isinstance(layer, nn.ReLU):
				name = 'relu_{}'.format(i)
				layer = nn.ReLU(inplace=False)
			elif isinstance(layer, nn.MaxPool2d):
				name = 'pool_{}'.format(i)
			elif isinstance(layer, nn.BatchNorm2d):
				name = 'bn_{}'.format(i)
			else:
				raise RuntimeError('Unrecognized layer: {}'
								   .format(layer.__class__.__name__))

			model.add_module(name, layer)

			if name in content_layers:
				org_content = model(org_img).detach()
				shifted_content = model(shifted_img)
				content_loss = self.ContentLoss(org_content, shifted_content)

				# model.add_module("content_loss_{}".format(i), content_loss)
				content_losses.append(content_loss)

		return content_losses

	def optimizeParametersAll(self, feed_dict, trainEmbed, updateGAN):
		# FOR loaded PGAN
		# if updateGAN:
		# 	print('Update GAN')
		# 	# target = feed_dict['target']
		# 	mask = feed_dict['mask_out']
		# 	logit = feed_dict['logit']
		# 	x_real = feed_dict['real_target']
		#
		# 	y_real = Variable(torch.ones(logit.size()[0]).cuda())
		# 	y_fake = Variable(torch.zeros(logit.size()[0]).cuda())
		#
		# 	# Update D
		# 	self.module.optimizerD.zero_grad()
		#
		# 	# D real
		# 	D_real_result = self.module.netD(x_real).squeeze()
		# 	D_real_result = D_real_result.mean() - 0.001 * (D_real_result ** 2).mean()
		# 	D_real_result.backward(self.module.mone, retain_graph=True)
		#
		# 	# D fake
		# 	D_fake_result = self.module.netD(logit).squeeze()
		# 	D_fake_result = D_fake_result.mean()
		# 	D_fake_result.backward(self.module.one, retain_graph=True)
		#
		# 	# TRAIN WITH GRADIENT PENALTY
		# 	# gp = gradient_penalty(functools.partial(self.module.netD), x_real, logit,
		# 	# 					  gp_mode='1-gp',
		# 	# 					  sample_mode='line')
		# 	# gradient_penalty = calc_gradient_penalty(self.module.netD, x_real.data, logit.data, self.BATCH_SIZE)
		#
		# 	eps = torch.rand(constants.BATCH_SIZE, 1, 1, 1).cuda()
		# 	x_hat = eps * x_real.data + (1 - eps) * logit.data
		# 	x_hat = Variable(x_hat, requires_grad=True)
		# 	hat_predict = self.module.netD(x_hat)
		# 	grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
		# 	grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
		# 	grad_penalty = 10 * grad_penalty
		# 	grad_penalty.backward(retain_graph=True)
		#
		# 	# grad_loss_val = grad_penalty.data
		# 	# disc_loss_val = (real_predict - fake_predict).data
		# 	self.module.optimizerD.step()
		#
		# 	# Update G
		# 	self.module.optimizerG.zero_grad()
		# 	new_logit = self.get_logits(feed_dict)
		# 	feed_dict['logit'] = new_logit
		#
		# 	D_fake_result = self.module.netD(new_logit).squeeze()
		# 	G_train_loss = self.BCE_loss_logits(D_fake_result, y_real)
		#
		# 	Edit_loss = self.get_edit_loss(feed_dict)
		# 	G_train_loss += self.LAMBDA * Edit_loss
		# 	G_train_loss.backward(retain_graph=True)
		# 	self.module.optimizerG.step()
		# 	self.module.accumulate(self.module.g_running, self.module.netG)
		if updateGAN:
			raise ('No implementation for updateGAN')

		# self.optimizers.zero_grad()
		# logit = feed_dict['logit']
		# y_real = Variable(torch.ones(logit.size()[0]).cuda())
		# D_fake_result = self.module.netD(logit,
		# 								 step=self.step,
		# 								 alpha=self.alpha).squeeze()
		# GAN_Loss = self.BCE_loss_logits(D_fake_result, y_real)
		# Content_loss = self.get_content_loss(feed_dict['org'], feed_dict['logit'])
		# Edit_loss = self.get_reg_loss(feed_dict)
		# content_losses = 0
		# weights = 1 - np.linspace(0.2, 1, len(Content_loss))
		# for i in range(len(Content_loss)):
		# 	content_losses += weights[i] * Content_loss[i]
		# loss = 10 * Edit_loss + 0.05 * content_losses / len(Content_loss) + 0.05 * GAN_Loss  ## + 0.1 * MSE_loss #
		# loss.backward()
		# self.optimizers.step()
		#
		# return loss

		for name in self.optimizers.keys():
			self.optimizers[name].zero_grad()
			logit = feed_dict['logit']
			y_real = Variable(torch.ones(logit.size()[0]).cuda())
			D_fake_result = self.module.netD(logit,
											 step=self.step,
											 alpha=self.alpha).squeeze()
			GAN_Loss = self.BCE_loss_logits(D_fake_result, y_real)
			Content_loss = self.get_content_loss(feed_dict['org'], feed_dict['logit'])
			Edit_loss = self.get_reg_loss(feed_dict)
			content_losses = 0
			weights = 1 # 1 - np.linspace(0.2, 1, len(Content_loss))
			for i in range(len(Content_loss)):
				# content_losses += weights[i] * Content_loss[i]
				content_losses += Content_loss[i]
			loss = 10 * Edit_loss + 0.05 * content_losses / len(Content_loss) + 0.05 * GAN_Loss  ## + 0.1 * MSE_loss #
			loss.backward()
			self.optimizers[name].step()

		return loss


	def save_multi_models(self, save_path_w, save_path_gan, trainEmbed=False, updateGAN=True,
						  single_transform_name=None):
		print('Save W and GAN in %s and %s' % (save_path_w, save_path_gan))
		if updateGAN == True:
			print('Save GAN')
			torch.save(self.module, save_path_gan)
		torch.save(self.walk, save_path_w + '_walk_module.ckpt')

	# if trainEmbed:
	# 	if single_transform_name:
	# 		print('Save %s only ' % single_transform_name)
	# 		cur_path_w = save_path_w + '_' + single_transform_name
	# 		np.save(cur_path_w, self.walk.w_embe[single_transform_name].detach().cpu().numpy())
	# 		return
	# 	print('Save embed W')
	# 	for i, cur_w in self.walk.w_embed.items():
	# 		cur_path_w = save_path_w + '_' + i
	# 		np.save(cur_path_w, cur_w.detach().cpu().numpy())
	# 	return
	# else:
	# 	print('Save ws')
	# 	for i, cur_w in self.walk.w_embed.items():
	# 		cur_path_w = save_path_w + '_' + i
	# 		np.save(cur_path_w, cur_w.detach().cpu().numpy())

	def load_multi_models(self, save_path_w, save_path_gan, trainEmbed=False, updateGAN=False,
						  single_transform_name=None):

		if updateGAN:
			# Load GAN
			print('Load GAN in %s' % save_path_gan)
			self.module = torch.load(save_path_gan)

		print('Load w in %s' % save_path_w)
		self.walk = torch.load(save_path_w)

	# try:
	# 	self.walk = torch.load(save_path_w)
	# except:
	# 	for name in self.walk.w.keys():
	# 		new_w_path = save_path_w + '_' + name + '.npy'
	# 		print('Load W of %s' % name)
	# 		print('Before w: ', self.walk.w[name].size())
	# 		self.walk.w_embed[name] = torch.nn.Parameter(torch.Tensor(np.load(new_w_path)).cuda())
	# 		print('After w: ', self.walk.w[name].size())

	# if trainEmbed:
	# 	# Load w
	# 	if single_transform_name:
	# 		print('Load %s only ' % single_transform_name)
	# 		new_w_path = save_path_w + '_' + single_transform_name + '.npy'
	# 		print('Before w: ', self.w_embed[single_transform_name][0, :5, 0])
	# 		self.walk.w_embed[single_transform_name] = torch.Tensor(np.load(new_w_path))
	# 		print('After w: ', self.w_embed[single_transform_name][0, :5, 0])
	# 		return
	#
	# 	print('Load W for embedding')
	# 	for name in self.w_embed.keys():
	# 		new_w_path = save_path_w + '_' + name + '.npy'
	# 		print('Load W of %s' % name)
	# 		print('Before w: ', self.w_embed[name][0, :5, 0])
	# 		self.walk.w_embed[name] = torch.Tensor(np.load(new_w_path))
	# 		print('After w: ', self.w_embed[name][0, :5, 0])
	# else:
	# 	print('Load continuous Ws ')
	# 	for name in self.ws.keys():
	# 		new_w_path = save_path_w + '_' + name + '.npy'
	# 		print('Load W of %s' % name)
	# 		print('Before w: ', self.ws[name][0, :5, 0])
	# 		self.ws[name] = torch.Tensor(np.load(new_w_path))
	# 		print('After w: ', self.ws[name][0, :5, 0])

	def load_model(self, save_path_w, save_path_gan):
		# Load w
		print('Load W in %s' % save_path_w)
		print('Before w: ', self.w[0, :5, 0])
		self.w = torch.Tensor(np.load(save_path_w))
		print('After w: ', self.w[0, :5, 0])
		# Load GAN
		print('Load GAN in %s' % save_path_gan)
		# self.module.load(save_path_gan)
		self.module = torch.load(save_path_gan)

	def get_reg_module(self):
		#####
		model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=False)
		model.fc = torch.nn.Linear(2048, 40)
		model = model.cuda()
		optimizer = optim.Adam(model.parameters(), lr=1e-4)
		#####
		base_dir = '/home/peiye/ImageEditing/scene_regressor/checkpoint_256/'
		ckpt = torch.load(base_dir + '500_dict.model')

		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optm'])

		return model, optimizer

	def get_vgg_module(self):
		#####
		import torchvision.models as models
		vgg19 = models.vgg19(pretrained=True).features.cuda().eval()
		return vgg19

	def get_stylegan_module(self):
		base_dir = '/home/peiye/ImageEditing/style-based-gan-pytorch/checkpoint_256_combine_2/'
		module = stylegan.StyleGAN(lr=self.lr)
		ckpt = torch.load(base_dir + '500000.model')
		module.netG.load_state_dict(ckpt['g_running'])
		module.netD.load_state_dict(ckpt['discriminator'])
		module.optimizerG.load_state_dict(ckpt['g_optimizer'])
		module.optimizerD.load_state_dict(ckpt['d_optimizer'])

		print('Finish loading the pretrained model')
		return module

	def clip_ims(self, ims):
		return np.uint8(np.clip(((ims + 1) / 2.0) * 255, 0, 255))

	def apply_alpha(self, graph_inputs, alpha_to_graph,
					layers=None, name=None, trainEmbed=False, index_=None,
					given_w=None):
		zs_batch = graph_inputs['z']  # tensor.cuda() # [Batch, DIM_Z]
		if self.stylegan_opts.latent == 'w':
			if not given_w :
				latent_w = self.get_w(zs_batch)
			else:
				latent_w = given_w
			feed_dict = {'w': latent_w}
			out_zs = self.get_logits(feed_dict)
			alpha_org = self.get_reg_preds(out_zs)
			alpha_delta = torch.Tensor(alpha_to_graph).cuda() - alpha_org

		if self.stylegan_opts.latent == 'z':

			z_new = self.get_z_new_tensor(zs_batch, alpha_to_graph, name, trainEmbed=trainEmbed, index_=index_)
			best_inputs = {'z': z_new}
			print('zs z new: ', zs_batch[0, :4], z_new[0, :4])
			best_im_out = self.get_logits(best_inputs)

		elif self.stylegan_opts.latent == 'w':
			# Get W
			# latent_w = self.get_w(zs_batch)  # list
			# Get transferred W
			latent_w_new = self.get_w_new_tensor(latent_w, alpha_delta,
												layers=layers,
												name=name,
												trainEmbed=trainEmbed,
												index_=index_)

			# latent_w_new = self.get_w_new_tensor(latent_w, alpha_to_graph,
			# 									 layers=layers,
			# 									 name=name,
			# 									 trainEmbed=trainEmbed,
			# 									 index_=index_)

			# layers=[i for i in range((self.step + 1) * 2)]

			best_inputs = {'w': latent_w_new}
			best_im_out = self.get_logits(best_inputs)

		else:
			raise ('Non implemented')
		return best_im_out

	def L2_loss(self, img1, img2, mask):
		return np.sum(np.square((img1 - img2) * mask), (1, 2, 3))

	def vis_image_batch_alphas(self, graph_inputs, filename,
							   alphas_to_graph, alphas_to_target,
							   batch_start, name=None, wgt=False, wmask=False,
							   trainEmbed=False, computeL2=True):

		zs_batch = graph_inputs['z']  # numpy
		filename_base = filename
		ims_target = []
		ims_transformed = []
		ims_mask = []
		L2_loss = {}

		for index_, (ag, at) in enumerator(zip(alphas_to_graph, alphas_to_target)):
			# print('Index: ', index)
			input_test = {'z': torch.Tensor(zs_batch).cuda()}
			out_input_test = self.get_logits(input_test)
			out_input_test = out_input_test.detach().cpu().numpy()  # on Cuda
			target_fn, mask_out = self.get_target_np(out_input_test, at)

			best_im_out = self.apply_alpha(input_test, ag, layers=layers, name=name,
										   index=index_).detach().cpu().numpy()

			L2_loss[at] = self.L2_loss(target_fn, best_im_out, mask_out)
			ims_target.append(target_fn)
			ims_transformed.append(best_im_out)
			ims_mask.append(mask_out)
		if computeL2:
			## Compute L2 loss for drawing the plots
			return L2_loss

		######### ######### ######### ######### #########
		print('wgt: ', wgt)
		for ii in range(zs_batch.shape[0]):
			arr_gt = np.stack([x[ii, :, :, :] for x in ims_target], axis=0)

			if wmask:
				arr_transform = np.stack([x[j, :, :, :] * y[j, :, :, :] for x, y
										  in zip(ims_transformed, ims_mask)], axis=0)
			else:
				arr_transform = np.stack([x[ii, :, :, :] for x in
										  ims_transformed], axis=0)
			arr_gt = self.clip_ims(arr_gt)
			arr_transform = self.clip_ims(arr_transform)
			if wgt:
				ims = np.concatenate((arr_gt, arr_transform), axis=0)
			else:
				ims = arr_transform
			filename = filename_base + '_sample{}'.format(ii + batch_start)
			if wgt:
				filename += '_wgt'
			if wmask:
				filename += '_wmask'
			# (7, 3, 64, 64)
			if ims.shape[1] == self.num_channels:
				# N C W H -> N W H C
				ims = np.transpose(ims, [0, 2, 3, 1])
			# ims = np.squeeze(ims)
			# print('ims.shape: ', ims.shape)
			image.save_im(image.imgrid(ims, cols=len(alphas_to_graph)), filename)

	def vis_multi_image_batch_alphas(self, graph_inputs, filename,
									 alphas_to_graph, alphas_to_target,
									 batch_start,
									 layers=None,
									 name=None, wgt=False, wmask=False,
									 trainEmbed=False, computeL2=False,
									 given_w=None):

		zs_batch = graph_inputs['z']  # numpy
		filename_base = filename
		ims_target = []
		ims_transformed = []
		ims_mask = []
		index_ = 0


		for ag, at in zip(alphas_to_graph, alphas_to_target):
			input_test = {'z': torch.Tensor(zs_batch).cuda()}

			best_im_out = self.apply_alpha(input_test, ag, name=name, layers=layers, trainEmbed=trainEmbed,
										   index_=index_, given_w=given_w)
			#best_im_out = F.interpolate(best_im_out, size=256)
			best_im_out = best_im_out.detach().cpu().numpy()
			best_im_out = np.uint8(np.clip(((best_im_out + 1) / 2.0) * 255, 0, 255))
			ims_transformed.append(best_im_out)
			index_ += 1

		for ii in range(zs_batch.shape[0]):
			if wmask:
				arr_transform = np.stack([x[j, :, :, :] * y[j, :, :, :] for x, y
										  in zip(ims_transformed, ims_mask)], axis=0)
			else:
				arr_transform = np.stack([x[ii, :, :, :] for x in
										  ims_transformed], axis=0)
			# arr_gt = self.clip_ims(arr_gt)
			# arr_transform = self.clip_ims(arr_transform)
			ims = arr_transform
			filename = filename_base + '_sample{}'.format(ii + batch_start)
			if wgt:
				filename += '_wgt'
			if wmask:
				filename += '_wmask'
			# (7, 3, 64, 64)
			if ims.shape[1] == 1 or ims.shape[1] == 3:
				# N C W H -> N W H C
				ims = np.transpose(ims, [0, 2, 3, 1])
			# ims = np.squeeze(ims)
			image.save_im(image.imgrid(ims, cols=len(alphas_to_graph)), filename)

	def vis_image_batch(self, graph_inputs, filename,
						batch_start, wgt=False, wmask=False, num_panels=7):
		raise NotImplementedError('Subclass should implement vis_image_batch')


class BboxTransform(TransformGraph):
	def __init__(self, *args, **kwargs):
		TransformGraph.__init__(self, *args, **kwargs)

	def get_distribution_statistic(self, img, channel=None):
		raise NotImplementedError('Subclass should implement get_distribution_statistic')

	def distribution_data_per_category(self, num_categories, num_samples,
									   output_path, channel=None):
		raise NotImplementedError('Coming soon')

	def distribution_model_per_category(self, num_categories, num_samples,
										a, output_path, channel=None):
		raise NotImplementedError('Coming soon')

	def get_distributions_per_category(self, num_categories, num_samples,
									   output_path, palpha, nalpha,
									   channel=None):
		raise NotImplementedError('Coming soon')

	def get_distributions_all_categories(self, num_samples, output_path,
										 channel=None):
		raise NotImplementedError('Coming soon')


class PixelTransform(TransformGraph):
	def __init__(self, *args, **kwargs):
		TransformGraph.__init__(self, *args, **kwargs)

	def get_distribution_statistic(self, img, channel=None):
		raise NotImplementedError('Subclass should implement get_distribution_statistic')

	def get_distribution(self, num_samples, channel=None):
		random_seed = 0
		rnd = np.random.RandomState(random_seed)
		inputs = graph_input(self, num_samples, seed=random_seed)
		batch_size = constants.BATCH_SIZE
		model_samples = []
		for a in self.test_alphas():
			distribution = []
			start = time.time()
			print("Computing attribute statistic for alpha={:0.2f}".format(a))
			for batch_num, batch_start in enumerate(range(0, num_samples, batch_size)):
				s = slice(batch_start, min(num_samples, batch_start + batch_size))
				inputs_batch = util.batch_input(inputs, s)
				zs_batch = inputs_batch[self.z]
				a_graph = self.scale_test_alpha_for_graph(a, zs_batch, channel)
				ims = self.clip_ims(self.apply_alpha(inputs_batch, a_graph))
				for img in ims:
					img_stat = self.get_distribution_statistic(img, channel)
					distribution.extend(img_stat)
			end = time.time()
			print("Sampled {} images in {:0.2f} min".format(num_samples, (end - start) / 60))
			model_samples.append(distribution)

		model_samples = np.array(model_samples)
		return model_samples
