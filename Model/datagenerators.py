import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data

'''
用于构建PyTorch的数据加载器，
以便在深度学习模型训练过程中加载和处理医学图像数据
'''


class Dataset(Data.Dataset):
    # files表示要处理的医学图像文件的路径
    def __init__(self, files):
        # 初始化
        self.files = files

    # 数据集中包含的医学图像文件数量
    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    # __getitem__采用sitk读取nii图像转为array
    def __getitem__(self, index):
        # 经过裁切后图像的尺寸大小为：(160, 192, 160)
        # 使用sitk.GetArrayFromImage函数将医学图像转换为NumPy数组
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr
