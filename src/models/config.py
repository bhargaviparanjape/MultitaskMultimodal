def add_args(parser):
	## General Model parameters
	model = parser.add_argument_group("Model Parameters")
	model.add_argument("--dropout", type=float, default=0.2)