import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from configs.cfg_SID_Sony import Cfg
from CRAFTpytorch.craft import CRAFT
from dataset.SID_Sony import SIDSonyTestDataset_fast_cc as sid_test_dataset
from dataset.SID_Sony import SIDSonyTrainDataset_fast_cc as sid_train_dataset
from unet import GrayEdgeAttentionUNet


def main():
    # logging to text file
    dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
    logger.add(f'{Cfg.result_dir}/{dt_string}_console.log', format='{time:YYYY-MM-DD at HH:mm:ss} | {level} | \
                {message}', mode='w', backtrace=True, diagnose=True)
    logger.info(Cfg)

    # tensorboard log
    writer = SummaryWriter(log_dir=Cfg.result_dir + 'logs')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # initialize network
    unet = GrayEdgeAttentionUNet()
    unet.to(device)

    # set up solver and scheduler
    optimizer = optim.Adam(unet.parameters(), lr=Cfg.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=Cfg.scheduler_gamma)

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

    # load pretrained for resume
    if Cfg.resume and Cfg.resume_tar is not None:
        unet, lastepoch, optimizer, scheduler = utils.load_checkpoint_state(Cfg.resume_tar, device, unet,
                                                                            optimizer, scheduler)
        logger.info(f'------Load pretrained model at epoch {lastepoch}!')
    else:
        lastepoch = 1
        logger.info('------No pretrained model!')

    # preparing dataloader for train
    train_dataset = sid_train_dataset(
        list_file=Cfg.train_list_file, root_dir=Cfg.dataset_dir, edge_dir=Cfg.train_edge_dir, ps=Cfg.ps)
    train_dataloader = DataLoader(train_dataset, batch_size=Cfg.bs, shuffle=True, num_workers=8,
                                  pin_memory=True, persistent_workers=True)

    test_dataset = sid_test_dataset(
        Cfg.target_size, list_file=Cfg.test_list_file, root_dir=Cfg.dataset_dir, edge_dir=Cfg.test_edge_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                 pin_memory=True, persistent_workers=True)

    for epoch in tqdm(range(lastepoch, Cfg.training_epoch), desc='Training Epoch:'):

        epoch_time = time.perf_counter()
        logger.info(f'------Starting epoch {epoch}')

        if os.path.isdir(Cfg.result_dir + '%04d' % epoch):
            continue

        cnt = 0

        mae_loss_list = []
        ms_ssim_loss_list = []
        text_loss_list = []
        all_loss_list = []

        unet.train()

        for sample in train_dataloader:
            sample_time = time.perf_counter()
            cnt += Cfg.bs

            gpu_time = time.perf_counter()
            in_imgs = sample['in_img'].to(device)
            gt_imgs = sample['gt_img'].to(device)
            in_gray_imgs = sample['in_gray_img'].to(device)
            in_edge_imgs = sample['in_edge_img'].to(device)
            gpu_end_time = time.perf_counter()-gpu_time

            optimizer.zero_grad()

            unet_time = time.perf_counter()
            out_imgs = unet(in_imgs, in_gray_imgs, in_edge_imgs)
            unet_end_time = time.perf_counter()-unet_time

            loss_time = time.perf_counter()
            mae_loss = utils.MAELoss(out_imgs, gt_imgs)
            ms_ssim_loss = utils.MS_SSIMLoss(out_imgs, gt_imgs)
            text_loss = utils.TextDetectionLoss(out_imgs, gt_imgs, craft_net)
            loss = Cfg.mae_loss_w * mae_loss + Cfg.ms_ssim_loss_w * ms_ssim_loss + Cfg.text_loss_w * text_loss
            loss_end_time = time.perf_counter()-loss_time

            bp_time = time.perf_counter()
            loss.backward()
            optimizer.step()
            bp_end_time = time.perf_counter()-bp_time

            # loss for the entire epoch
            mae_loss_list.append(mae_loss.item())
            ms_ssim_loss_list.append(ms_ssim_loss.item())
            text_loss_list.append(text_loss.item())
            all_loss_list.append(loss.item())

            sample_end_time = time.perf_counter()-sample_time
            logger.info('%d %d All_Loss=%.3f TOGPU_Time=%.3f UNET_Time=%.3f'
                        'LOSS_Time=%.3f BP_Time=%.3f Total_Time=%.3f' %
                        (epoch, cnt, np.mean(all_loss_list), gpu_end_time,
                         unet_end_time, loss_end_time, bp_end_time, sample_end_time))

        per_epoch_time = time.perf_counter()-epoch_time
        logger.info('------per_epoch_time=%.3f' % (per_epoch_time))

        writer.add_scalar('Train/MAE_Loss', np.mean(mae_loss_list), epoch)
        writer.add_scalar('Train/MS_SSIM_Loss', np.mean(ms_ssim_loss_list), epoch)
        writer.add_scalar('Train/Text_Loss', np.mean(text_loss_list), epoch)
        writer.add_scalar('Train/All_Loss', np.mean(all_loss_list), epoch)

        if epoch % Cfg.model_save_freq == 0:
            utils.save_checkpoint_state(os.path.join(Cfg.result_dir, 'epoch_{}.tar'.format(epoch)),
                                        epoch, unet, optimizer, scheduler)

        if epoch % Cfg.test_freq == 0:
            with torch.no_grad():
                eval_time = time.perf_counter()

                unet.eval()
                psnr_list, ssim_list = [], []

                for sample in iter(test_dataloader):
                    in_img = sample['in_img'].to(device)
                    gt_img = sample['gt_img'].to(device)
                    in_gray_img = sample['in_gray_img'].to(device)
                    in_edge_img = sample['in_edge_img'].to(device)

                    out_img = unet(in_img, in_gray_img, in_edge_img)

                    psnr_list.append(utils.PSNR(out_img, gt_img).item())
                    ssim_list.append(utils.SSIM(out_img, gt_img).item())

                    del in_img
                    del gt_img
                    del in_gray_img
                    del in_edge_img
                    del out_img

                total_eval_time = time.perf_counter()-eval_time
                logger.info('------Total_eval_time=%.3f' % (total_eval_time))
                logger.info('------PSNR={}, SSIM={}'.format(np.mean(psnr_list), np.mean(ssim_list)))

                writer.add_scalar('Test/psnr', np.mean(psnr_list), epoch)
                writer.add_scalar('Test/ssim', np.mean(ssim_list), epoch)

        scheduler.step()
    writer.close()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.gpu_id
    main()
