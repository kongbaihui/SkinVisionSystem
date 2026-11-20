from datetime import datetime
import os

from backend.skinvision.training.utils import BceDiceLoss


class setting_config:
	"""Training configuration."""

	network = "LMDAUNet"
	model_config = {
		"num_classes": 1,
		"kernel_sizes": [1, 3, 5],
		"expansion_factor": 2,
		"dw_parallel": True,
		"add": True,
		"dcag_ks": 3,
		"activation": "relu",
		"encoder": "Encoder",
		"pretrain": True,
	}
	test_weights = ""
	criterion = BceDiceLoss()
	num_classes = 1
	input_size_h = 256
	input_size_w = 256
	input_channels = 3
	distributed = False
	local_rank = -1
	num_workers = 0 if os.name == "nt" else 4
	device = "auto"
	seed = 42
	world_size = None
	rank = None
	amp = False
	batch_size = 8
	gpu_id = "0"
	epochs = 200

	datasets = "ISIC2016"
	if datasets == "ISIC2016":
		data_path = "./data/dataset_isic16/"
	elif datasets == "ISIC2017":
		data_path = "./data/dataset_isic17/"
	elif datasets == "ISIC2018":
		data_path = "./data/dataset_isic18/"
	elif datasets == "PH2":
		data_path = "./data/dataset_PH2/"
	else:
		raise Exception("datasets in not right!")

	work_dir = "outputs/training/" + network + "_" + datasets + "_" + datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss") + "/"
	print_interval = 20
	val_interval = 30
	save_interval = 100
	threshold = 0.5
	opt = "AdamW"

	if opt == "AdamW":
		lr = 0.001
		betas = (0.9, 0.999)
		eps = 1e-8
		weight_decay = 1e-2
		amsgrad = False

	sch = "CosineAnnealingLR"
	if sch == "CosineAnnealingLR":
		T_max = 50
		eta_min = 0.00001
		last_epoch = -1