import math

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from pytorch_msssim import ms_ssim, ssim


def SSIM(out_image, gt_image):
    out_image = torch.clamp(out_image, min=0.0, max=1.0)
    return ssim(out_image, gt_image, data_range=1, size_average=True)


def PSNR(out_image, gt_image):
    out_image = torch.clamp(out_image, min=0.0, max=1.0)
    mse = torch.mean((out_image - gt_image)**2)
    return 10 * torch.log10(1.0 / mse)


def MAELoss(out_image, gt_image):
    return torch.mean(torch.abs(out_image - gt_image))


def MS_SSIMLoss(out_image, gt_image):
    return 1 - ms_ssim(out_image, gt_image, data_range=1, size_average=True)


def Tensor2OpenCV(img):
    # BxCxHxW (B=1) to HxWxC
    if img.size(dim=0) == 1:
        img = img.squeeze(0)
    img = img.permute(1, 2, 0).cpu().data.numpy()
    img = np.minimum(np.maximum(img, 0), 1)
    img = img * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def normalizeMeanVarianceTensor(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = TF.normalize(in_img, mean=(0.485*255.0, 0.456*255.0, 0.406*255.0),
                       std=(0.229*255.0, 0.224*255.0, 0.225*255.0))
    return img


# CRAFT Section
def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex = x - niter, x + w + niter + 1
        sy, ey = y - niter, y + h + niter + 1
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    return boxes


def TextDetectionLoss(out_image, gt_image, net):
    # out_image:  BxCxHxW
    out_image = torch.clamp(out_image, min=0, max=1)
    out_image = out_image * 255
    out_image = normalizeMeanVarianceTensor(out_image)

    gt_image = torch.clamp(gt_image, min=0, max=1)
    gt_image = gt_image * 255
    gt_image = normalizeMeanVarianceTensor(gt_image)

    out_pred, _ = net(out_image)
    gt_pred, _ = net(gt_image)

    out_text = out_pred[:, :, :, 0]
    gt_text = gt_pred[:, :, :, 0]

    return torch.mean(torch.abs(out_text - gt_text))


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)

        if isinstance(ratio_w, torch.Tensor):
            ratio_w = ratio_w.item()

        if isinstance(ratio_h, torch.Tensor):
            ratio_h = ratio_h.item()

        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)

    return polys


# for checkpoint saving and loading
def save_checkpoint_state(path, epoch, model, optimizer, scheduler):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }
    torch.save(checkpoint, path)


def load_checkpoint_state(path, device, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return model, epoch, optimizer, scheduler


def load_checkpoint_state_infer(path, device, model):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    return model, epoch
