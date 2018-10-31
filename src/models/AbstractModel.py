from abc import ABCMeta, abstractmethod, abstractproperty
from src.utils.utility_functions import variable,FloatTensor,ByteTensor,LongTensor,select_optimizer
import torch
import os,copy,sys, logging

logger = logging.getLogger(__name__)

class AbstractModel():
	__metaclass__ = ABCMeta

	def __init__(self, args, inputs):
		self.args = args
		self.updates = 0

	def cuda(self):
		self.network = self.network.cuda()

	def parallelize(self):
		"""Use data parallel to copy the model across several gpus.
		This will take all gpus visible with CUDA_VISIBLE_DEVICES.
		"""
		self.parallel = True
		self.network = torch.nn.DataParallel(self.network)

	def init_optimizer(self):
		parameters = [p for p in self.network.parameters() if p.requires_grad]
		self.optimizer = select_optimizer(self.args, parameters)

	def vectorize(self, batch, mode = "train"):
		raise NotImplementedError

	def save(self):
		# model parameters; metrics;
		if self.args.parallel:
			network = self.network.module
		else:
			network = self.network
		state_dict = copy.copy(network.state_dict())
		# Pop layers if required
		params = {
			'args': self.args,
			'state_dict': state_dict
		}
		try:
			torch.save(params, os.path.join(self.args.model_dir, self.args.model_path))
		except BaseException:
			logger.warning('WARN: Saving failed... continuing anyway.')

	def checkpoint(self, file_path, epoch_no):
		raise NotImplementedError

	@staticmethod
	def add_args(parser):
		raise NotImplementedError

	def load(self, pretrained_model_path=None, pretrained_embed_path=None):
		raise NotImplementedError