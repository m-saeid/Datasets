import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

CLASS_MAP = {"car":0, "bus":1, "motorcycle":2}

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    s = 0 if partition == 'train' else 96
    e = 96 if partition == 'train' else 120
    for clss in ["car", "bus", "motorcycle"]:
      for i in range(s,e):
          pcd_path = os.path.join(DATA_DIR, "Vehicles360/PCDs_1024", clss, f"{i}.pcd")
          pcd = o3d.io.read_point_cloud(pcd_path)
          points = np.asarray(pcd.points).astype('float32')
          all_data.append([points])

          all_label.append([CLASS_MAP[clss]])

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.array(all_label).astype('int64')
    return all_data, all_label



def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class vehicles360(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]



if __name__ == '__main__':
    train = vehicles360(1024)
    test = vehicles360(1024, 'test')
    for data, label in train:
        print('*', data.shape)
        print('*', label.shape)