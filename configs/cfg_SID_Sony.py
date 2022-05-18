from easydict import EasyDict

Cfg = EasyDict()
Cfg.gpu_id = '0'

# whether to resume from a ckecpoint
Cfg.resume = True
Cfg.resume_tar = None

Cfg.dataset_dir = '/home/jovyan/dataset/Sony/'

Cfg.train_input_dir = Cfg.dataset_dir + 'train/input_image/'
Cfg.train_gt_dir = Cfg.dataset_dir + 'train/gt_image/'
Cfg.train_edge_dir = Cfg.dataset_dir + 'train/edge/'
Cfg.train_list_file = Cfg.dataset_dir + 'icpr_train_list.txt'

Cfg.test_input_dir = Cfg.dataset_dir + 'test/input_image/'
Cfg.test_gt_dir = Cfg.dataset_dir + 'test/gt_image/'
Cfg.test_edge_dir = Cfg.dataset_dir + 'test/edge/'
Cfg.test_list_file = Cfg.dataset_dir + 'icpr_test_list.txt'

# all the results will be under result_dir
Cfg.result_dir = './results/result_SID_Sony/'

# training batch and size
Cfg.bs = 1
Cfg.ps = 512  # patch size for training
Cfg.test_freq = 100  # frequency to perform CRAFT analysis and to visulize the processed testing images
Cfg.model_save_freq = 100  # frequency to save the model

Cfg.training_epoch = 4001
Cfg.learning_rate = 1e-4
Cfg.scheduler_gamma = 0.1
Cfg.target_size = 4096

# loss weightings for each term
Cfg.mae_loss_w = 0.85
Cfg.ms_ssim_loss_w = 0.15
Cfg.text_loss_w = 0.85 * 0.5

# testing parameters
Cfg.save_test_image = True
Cfg.test_tar = './results/result_SID_Sony/epoch_4.tar'

# craft text detection related hyper parameters
Cfg.text_threshold = 0.7
Cfg.link_threshold = 0.4
Cfg.low_text = 0.4
Cfg.craft_pretrained_model = './CRAFTpytorch/craft_ic15_20k.pth'
