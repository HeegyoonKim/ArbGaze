import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os, h5py, cv2, random


class GazeData(Dataset):
    def __init__(self, dataset, data_path, subject_ids, scale=1.0):
        
        self.dataset = dataset
        self.data_path = data_path
        self.scale = scale

        with h5py.File(data_path, 'r') as h5f:
            if dataset == 'utmv':
                self.subject_ids = ['s%02d' % id for id in subject_ids]
            else:   # mpiifacegaze
                self.subject_ids = ['p%02d' % id for id in subject_ids]
            
            self.index_to_query = sum([
                [(subject_id, i) for i in range(len(next(iter(h5f[subject_id].values()))))]
                for subject_id in self.subject_ids
            ], [])

    def __len__(self):
        return len(self.index_to_query)
    
    def set_scale(self, scale):
        self.scale = scale
        
    def prepare_images(self, image):
        HR_img = image
        HR_H, HR_W = HR_img.shape[0], HR_img.shape[1]

        LR_img = cv2.resize(HR_img, dsize=None, fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_CUBIC)
        LR_H, LR_W = LR_img.shape[0], LR_img.shape[1]
        GT_H, GT_W = int(self.scale * LR_H), int(self.scale * LR_W)
        
        GT_img = cv2.resize(HR_img, dsize=(GT_W,GT_H), interpolation=cv2.INTER_CUBIC)
        BC_img = cv2.resize(LR_img, dsize=(HR_W,HR_H), interpolation=cv2.INTER_CUBIC)

        HR_img = np.expand_dims(HR_img, axis=0)
        LR_img = np.expand_dims(LR_img, axis=0)
        GT_img = np.expand_dims(GT_img, axis=0)
        BC_img = np.expand_dims(BC_img, axis=0)

        HR_img = torch.FloatTensor(HR_img)
        LR_img = torch.FloatTensor(LR_img)
        GT_img = torch.FloatTensor(GT_img)
        BC_img = torch.FloatTensor(BC_img)

        HR_img = 2 * HR_img / 255.0 - 1
        LR_img = LR_img / 255.0
        GT_img = GT_img / 255.0
        BC_img = 2 * BC_img / 255.0 - 1

        return HR_img, LR_img, BC_img, GT_img
    
    def __getitem__(self, idx):
        hdf = h5py.File(self.data_path, 'r')

        key, index = self.index_to_query[idx]
        group = hdf[key]

        imgs = group['pixels'][index]
        gazes = group['labels'][index]

        if self.dataset == 'utmv':
            p = random.uniform(0.0, 1.0)
            if p < 0.5: # left
                img = imgs[:, 60:120]
                HR, LR, BC, SR_GT = self.prepare_images(img)
                gaze = gazes[1].reshape(3)
                eye = 0
            else:       # right
                img = imgs[:, 0:60]
                HR, LR, BC, SR_GT = self.prepare_images(img)
                gaze = gazes[0].reshape(3)
                eye = 1
        else:   # mpii
            HR, LR, BC, SR_GT = self.prepare_images(imgs)
            gaze = gazes
            eye = 2 # pre-defined
                
        data_dict = {
            'HR_img': HR,
            'LR_img': LR,
            'BC_img': BC,
            'SR_GT_img': SR_GT,
            'gaze': torch.tensor(gaze),
            'which_eye': torch.tensor(eye),
            'scale': torch.tensor(self.scale)
        }

        return data_dict


if __name__ == '__main__':
    data_path = '../datasets/mpii.h5'
    subjects = [i for i in range(15)]
    test_subject_ids = [0]
    for i in range(len(test_subject_ids)):
        subjects.remove(test_subject_ids[i])
    train_subject_ids = subjects

    train_dataset = GazeData('mpii', data_path, train_subject_ids)
    test_dataset = GazeData('mpii', data_path, test_subject_ids)

    print(len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=7, shuffle=True, num_workers=0, pin_memory=True)

    for i, (data_dict) in enumerate(train_loader, 1):
        print(data_dict['HR_img'].size(), data_dict['gaze'].size())
        if i > 1:
            break