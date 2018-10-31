import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import pdb
import copy,os,logging

from src.models.abstract_model import AbstractModel
from src.models import factory as model_factory
from src.learn import factory as learn_factory
from src.models.factory import RegisterModel
from src.models.components.output_models.dialogue_embedder import DialogueEmbedder
from src.utils.utility_functions import variable,FloatTensor,ByteTensor,LongTensor,select_optimizer
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

#########################################
############### NETWORK #################
#########################################
class RefExpVanillaNetwork(nn.Module):
	def __init__(self, args):
		raise NotImplementedError

	def forward(self, *input):
		raise NotImplementedError

	def eval(self, *input):
		raise NotImplementedError


#################################################
############### NETWORK WRAPPER #################
#################################################
class RefExpVanilla(AbstractModel):
	def __init__(self, args):

		## Initialize environment
		self.args = args
		self.updates = 0
		self.network = RefExpVanillaNetwork(self.args)

	def predict(self, inputs):
		raise NotImplementedError

	def vectorize(self, batch, mode = "train"):
		raise NotImplementedError

	def init_optimizer(self):
		parameters = [p for p in self.network.parameters() if p.requires_grad]
		self.optimizer = select_optimizer(self.args, parameters)