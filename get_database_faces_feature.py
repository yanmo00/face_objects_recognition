import torch
import os
import pickle
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from models.mtcnn import MTCNN
from models.face_alignment import Alignment
from models.inception_resnet_v1 import InceptionResnetV1

mtcnn = MTCNN(keep_all=True, post_process=False, device='cpu')    # 实例化得到mtcnn人脸检测器
resnet = InceptionResnetV1(pretrained=True).eval()                # 实例化得到一个facenet人脸特征提取器


def collate_fn(x):
    """ 自定义取出一个batch数据的格式 """
    # 因为 batch_size=1，故索引只能取0
    return x[0]


def get_database_faces_feature(database_path='./data/database_images',
                               save=False,
                               pkl_name='faces_feature_database.pkl'):

    dataset = datasets.ImageFolder(database_path)                              # 获取指定路径中的所有文件夹及其其中的照片信息

    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}     # 关联类别和人名
    loader = DataLoader(dataset, collate_fn=collate_fn)                        # 加载器加载数据

    faces_after_aligned = []     # 用于保存检测到的人脸图像数据
    names = []                   # 用于保存人名
    for x, y in loader:
        faces, prob, bboxes, points = mtcnn(x, return_prob=True)        # 执行人脸检测
        if faces is not None:
            for face, point in zip(faces, points):                      # 遍历检测出的所有人脸框
                face = face.permute(1, 2, 0).numpy().astype(np.uint8)   # 调整数据类型和维度顺序
                face_aligned, _ = Alignment(face, point)  # 人脸对齐
                face_aligned = torch.tensor(face_aligned, dtype=torch.float32).permute(2, 0, 1)   # 调整数据类型和维度顺序
                face_aligned = (face_aligned - 127.5) / 128.0           # 数据标准化
                faces_after_aligned.append(face_aligned)                # 将处理后的人脸图像数据存储起来
                names.append(dataset.idx_to_class[y])                   # 保存相应的人名信息

    faces_stack = torch.stack(faces_after_aligned)             # 将所有人脸图像信息堆叠至一个tensor中
    faces_feature = resnet(faces_stack).detach()               # 提取人脸特征信息
    faces_feature_database = tuple(zip(names, faces_feature))  # 将数据整合成元组形式，以便后续保存

    # 将检测到的人脸特征数据及人名以pkl格式保存
    if save:
        pkl_path = os.path.dirname(database_path)
        with open(os.path.join(pkl_path, pkl_name), mode='wb') as f:
            pickle.dump(faces_feature_database, f)

    return faces_feature_database


if __name__ == '__main__':
    faces_database = get_database_faces_feature(save=True)
