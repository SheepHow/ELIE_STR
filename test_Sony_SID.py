import os
import time
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from configs.cfg_SID_Sony import Cfg
from CRAFTpytorch.craft import CRAFT
from dataset.SID_Sony import SIDSonyTestDataset_fast_cc as sid_test_dataset
from unet import GrayEdgeAttentionUNet


def main():
    # logging to text file
    dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
    logger.add(f'{Cfg.result_dir}/{dt_string}_test_console.log', format='{time:YYYY-MM-DD at HH:mm:ss} | {level} |  \
                {message}', mode='w', backtrace=True, diagnose=True)

    logger.info(Cfg)

    if Cfg.save_test_image:
        out_img_path = os.path.join(Cfg.result_dir, 'output_image')
        if not os.path.isdir(out_img_path):
            os.makedirs(out_img_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # initialize network
    unet = GrayEdgeAttentionUNet()

    # load pretrained for evaluation
    unet, lastepoch = utils.load_checkpoint_state_infer(Cfg.test_tar, device, unet)
    logger.info(f'------Load pretrained model at epoch {lastepoch}!')

    unet.to(device)

    # load text detection model
    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith('module'):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = '.'.join(k.split('.')[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    # CRAFT
    craft_net = CRAFT()
    craft_net.load_state_dict(copyStateDict(torch.load(Cfg.craft_pretrained_model)))
    craft_net.to(device)
    craft_net.eval()

    # preparing dataloader for test
    logger.info(f'------Evaluated using image size: {Cfg.target_size}!')
    test_dataset = sid_test_dataset(Cfg.target_size, list_file=Cfg.test_list_file,
                                    root_dir=Cfg.dataset_dir, edge_dir=Cfg.test_edge_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                 pin_memory=True, persistent_workers=True)

    with torch.no_grad():
        eval_time = time.perf_counter()

        unet.eval()
        psnr_list, ssim_list = [], []

        for sample in tqdm(test_dataloader, desc='Testing Sample:'):
            in_fn = sample['in_fn'][0]  # input filename
            in_img = sample['in_img'].to(device)
            gt_img = sample['gt_img'].to(device)
            in_gray_img = sample['in_gray_img'].to(device)
            in_edge_img = sample['in_edge_img'].to(device)

            out_img = unet(in_img, in_gray_img, in_edge_img)

            psnr_list.append(utils.PSNR(out_img, gt_img).item())
            ssim_list.append(utils.SSIM(out_img, gt_img).item())

            out_img = utils.Tensor2OpenCV(out_img)
            if Cfg.save_test_image:
                cv2.imwrite(os.path.join(out_img_path, in_fn), out_img)

            del in_img
            del gt_img
            del in_gray_img
            del in_edge_img
            del out_img

        total_eval_time = time.perf_counter()-eval_time
        logger.info('------Total_eval_time=%.3f' % (total_eval_time))
        logger.info('------PSNR={}, SSIM={}'.format(np.mean(psnr_list), np.mean(ssim_list)))


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.gpu_id
    main()
