import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SIDSonyTrainDataset_fast_cc(Dataset):
    def __init__(self, list_file, root_dir, edge_dir, ps):
        self.ps = ps
        self.list_file = open(list_file, 'r')
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir
        self.edge_dir = edge_dir
        self.ratio_keys = [100, 250, 300]

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]

        in_exposure = float(input_img_name.split('/')[-1].split('_')[2].split('.png')[0][:-1])
        gt_exposure = float(gt_img_name.split('/')[-1].split('_')[2].split('.png')[0][:-1])

        # either 100, 250 or 303
        ratio = int(min(gt_exposure / in_exposure, 300))
        assert ratio in self.ratio_keys, print(in_exposure, gt_exposure, ratio)

        # this is to get the index of image
        in_fn = input_img_name.split('/')[-1]

        gt_img_path = os.path.join(self.root_dir, gt_img_name)
        assert os.path.exists(gt_img_path)
        gt_img = cv2.cvtColor(cv2.imread(gt_img_path), cv2.COLOR_BGR2RGB)
        gt_img_h, gt_img_w, _ = gt_img.shape
        gt_img = np.float32(gt_img / 255.0)

        input_img_path = os.path.join(self.root_dir, input_img_name)
        assert os.path.exists(input_img_path)
        input_img = cv2.cvtColor(cv2.imread(input_img_path), cv2.COLOR_BGR2RGB)
        input_img_h, input_img_w, _ = input_img.shape
        input_img = np.float32(input_img / 255.0)

        edge_path = os.path.join(self.edge_dir, in_fn)
        assert os.path.exists(edge_path)
        input_edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        input_edge_h, input_edge_w = input_edge_img.shape
        input_edge_img = np.expand_dims(np.float32(input_edge_img / 255.0), axis=2)

        assert input_img_h == gt_img_h == input_edge_h
        assert input_img_w == gt_img_w == input_edge_w

        # Random crop
        xx = np.random.randint(0, input_img_w - self.ps)
        yy = np.random.randint(0, input_img_h - self.ps)
        input_patch = input_img[yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = gt_img[yy:yy + self.ps, xx:xx + self.ps, :]
        input_edge_patch = input_edge_img[yy:yy + self.ps, xx:xx + self.ps, :]

        # random flip
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
            input_edge_patch = np.flip(input_edge_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
            input_edge_patch = np.flip(input_edge_patch, axis=2)
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (1, 0, 2))
            gt_patch = np.transpose(gt_patch, (1, 0, 2))
            input_edge_patch = np.transpose(input_edge_patch, (1, 0, 2))

        input_patch = np.minimum(np.maximum(input_patch, 0), 1)
        gt_patch = np.minimum(np.maximum(gt_patch, 0), 1)
        input_edge_patch = np.minimum(np.maximum(input_edge_patch, 0), 1)

        input_patch = torch.from_numpy(input_patch).permute(2, 0, 1)
        gt_patch = torch.from_numpy(gt_patch).permute(2, 0, 1)
        input_edge_patch = torch.from_numpy(input_edge_patch).permute(2, 0, 1)

        r, g, b = input_patch[0, :, :], input_patch[1, :, :], input_patch[2, :, :]
        input_gray_patch = (1.0 - (0.299 * r + 0.587 * g + 0.114 * b)).unsqueeze(0)

        sample = {
            'in_img': input_patch,
            'gt_img': gt_patch,
            'in_gray_img': input_gray_patch,
            'in_edge_img': input_edge_patch,
        }

        return sample


class SIDSonyTestDataset_fast_cc(Dataset):
    def __init__(self, target_size, list_file, root_dir, edge_dir):
        self.list_file = open(list_file, 'r')
        self.list_file_lines = self.list_file.readlines()
        self.edge_dir = edge_dir
        self.root_dir = root_dir
        self.target_size = target_size
        self.ratio_keys = [100, 250, 300]

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]

        in_exposure = float(input_img_name.split('/')[-1].split('_')[2].split('.png')[0][:-1])
        gt_exposure = float(gt_img_name.split('/')[-1].split('_')[2].split('.png')[0][:-1])

        # either 100, 250 or 303
        ratio = int(min(gt_exposure / in_exposure, 300))
        assert ratio in self.ratio_keys, print(in_exposure, gt_exposure, ratio)

        # this is to get the index of image
        in_fn = input_img_name.split('/')[-1]

        input_img_path = os.path.join(self.root_dir, input_img_name)
        assert os.path.exists(input_img_path)
        input_img = cv2.cvtColor(cv2.imread(input_img_path), cv2.COLOR_BGR2RGB)
        input_img_h, input_img_w, _ = input_img.shape

        if max(input_img_h, input_img_w) > self.target_size:
            resize_ratio = self.target_size / max(input_img_h, input_img_w)
        else:
            resize_ratio = 1.0

        target_H, target_W = int(input_img_h * resize_ratio), int(input_img_w * resize_ratio)

        dim = (target_W, target_H)
        input_img = cv2.resize(input_img, dim, interpolation=cv2.INTER_AREA)
        H1 = input_img.shape[0]
        W1 = input_img.shape[1]

        if (W1 % 32) != 0:
            W1 = W1 + 32 - (W1 % 32)
        if (H1 % 32) != 0:
            H1 = H1 + 32 - (H1 % 32)

        dim = (W1, H1)
        final_ratio_w = W1/input_img_w
        final_ratio_h = H1/input_img_h
        input_img = cv2.resize(input_img, dim, interpolation=cv2.INTER_AREA)
        input_img_h, input_img_w, _ = input_img.shape
        input_img = np.float32(input_img / 255.0)

        gt_img_path = os.path.join(self.root_dir, gt_img_name)
        assert os.path.exists(gt_img_path)
        gt_img = cv2.cvtColor(cv2.imread(gt_img_path), cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, dim, interpolation=cv2.INTER_AREA)
        gt_img_h, gt_img_w, _ = gt_img.shape
        gt_img = np.float32(gt_img / 255.0)

        edge_path = os.path.join(self.edge_dir, in_fn)
        assert os.path.exists(edge_path)
        input_edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        input_edge_img = cv2.resize(input_edge_img, dim, interpolation=cv2.INTER_AREA)
        input_edge_h, input_edge_w = input_edge_img.shape
        input_edge_img = np.expand_dims(np.float32(input_edge_img / 255.0), axis=2)

        assert input_img_h == gt_img_h == input_edge_h
        assert input_img_w == gt_img_w == input_edge_w

        # in_fn example: 10213_00_0.1s.png
        img_file_name_only_no_ext = in_fn.split('.')[0] + in_fn.split('.')[1][:-1]
        first_part_name = img_file_name_only_no_ext.split('_')[0]
        second_part_name = img_file_name_only_no_ext.split('_')[1]
        third_part_name = img_file_name_only_no_ext.split('_')[2]
        filename_no_ext = first_part_name + second_part_name + third_part_name

        input_img = np.minimum(np.maximum(input_img, 0), 1)
        gt_img = np.minimum(np.maximum(gt_img, 0), 1)
        input_edge_img = np.minimum(np.maximum(input_edge_img, 0), 1)

        input_img = torch.from_numpy(input_img).permute(2, 0, 1)
        gt_img = torch.from_numpy(gt_img).permute(2, 0, 1)
        input_edge_img = torch.from_numpy(input_edge_img).permute(2, 0, 1)

        r, g, b = input_img[0, :, :], input_img[1, :, :], input_img[2, :, :]
        in_gray_img = (1.0 - (0.299 * r + 0.587 * g + 0.114 * b)).unsqueeze(0)

        sample = {
            'in_img': input_img,
            'gt_img': gt_img,
            'in_edge_img': input_edge_img,
            'in_gray_img': in_gray_img,
            'in_fn': in_fn,
            'final_ratio_w': final_ratio_w,
            'final_ratio_h': final_ratio_h,
            'file_name': filename_no_ext
            }

        return sample
