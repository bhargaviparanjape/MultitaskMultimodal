import torch
import torch.optim as optim
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

def variable(v, arg_use_cuda=True, volatile=False):
	if torch.cuda.is_available() and arg_use_cuda:
		return Variable(v, volatile=volatile).cuda()
	return Variable(v, volatile=volatile)


def select_optimizer(args, parameters):
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(parameters, args.l_rate,
								   momentum=args.momentum,
								   weight_decay=args.weight_decay)
	elif args.optimizer == 'adamax':
		optimizer = optim.Adamax(parameters,
									  weight_decay=args.weight_decay)
	elif args.optimizer == "adam":
		optimizer = optim.Adam(parameters, lr=args.l_rate)
	else:
		raise RuntimeError('Unsupported optimizer: %s' %
						   args.optimizer)

	return optimizer