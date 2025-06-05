import os.path
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.mtcnn import MTCNN  # 导入人脸检测模块
from models.face_alignment import Alignment  # 导入人脸对齐模块
from models.inception_resnet_v1 import InceptionResnetV1  # 导入FaceNet
from get_database_faces_feature import get_database_faces_feature  # 导入人脸识别模块
from PIL import Image, ImageDraw, ImageFont


# 主函数
# save ----是否保存（默认不保存）
# save_path --- 默认保存地址
def main(image_path="./data/examples_images/example02.jpg", save=False,
         save_path='./data/examples_images_results'):  # 待检测照片
    image = plt.imread(image_path)  # 读取照片

    # -------实例化得到一个人脸检测器---------

    # keep_all --- 是否将照片中的全部人脸都保存下来
    # device ------ 表示计算的平台
    mtcnn = MTCNN(keep_all=True, post_process=False, device='cpu')
    # 实例化得到一个特征提取器
    # 调用eval()是因为我们不去训练，而只是用于推理
    resnet = InceptionResnetV1(pretrained=True).eval()

    # -------执行人脸检测操作---------

    # 用于存储人脸图像数据
    faces_after_aligned = []

    # faces ---- 人脸检测的图像
    # prob ----- 图像为人脸的概率有多少(将相对比较大的返回给我们)
    # bboxs ---- 将图片中若干的人脸的人脸框的坐标返回
    # points --- 五个关键点
    faces, prob, bboxs, points = mtcnn(image, return_prob=True)

    # -------执行人脸对齐操作---------
    # 一个人脸图像
    # face = faces[0]
    # 第一个人脸图像的关键点坐标
    # point = points[0]

    # 在faces图像为空的时候，不进行下面的计算
    if faces is not None:
        # face在faces中做遍历访问，pint在pints中做遍历访问
        for face, point in zip(faces, points):
            # 调整照片的通道顺序并修改数据类型
            # 书写顺序分先写分辨率的*行*，随后是*列*，最后是*通道数*
            # permute的作用是调整通道的顺序
            # numpy()-----转换为数组
            # astype变为---转为无符号的八位整形
            face = face.permute(1, 2, 0).numpy().astype(np.uint8)
            # 执行人脸对齐操作
            # 此处下划线是为了接收另外一个值
            face_aligned, _ = Alignment(face, point)
            # 使用torch中的tensor函数将上面的face_aligned图像转换为tensor
            # dtype申明数据类型为32为的浮点型
            # 使用permute将通道（维度）的顺序调整回来
            face_aligned = torch.tensor(face_aligned, dtype=torch.float32).permute(2, 0, 1)
            # 数据标准化(数据当前的取值范围在0~255，为了方便后续的计算我们同意将数据整理到-1~1之间)
            face_aligned = (face_aligned - 127.5) / 128
            # 保存人脸图像数据
            faces_after_aligned.append(face_aligned)

    # -------执行人脸特征提取---------
    # 将所有的人脸图像进行堆叠，得到的就是一个tensor
    faces_stack = torch.stack(faces_after_aligned)
    # 提取人脸特征信息（特征向量）
    faces_feature = resnet(faces_stack)

    # -------人脸识别(人脸特征匹配)---------
    # 获取后台数据库中的人脸特征向量
    # database_path ---- 后台数据库的路径
    faces_feature_database = get_database_faces_feature(database_path='./data/database_images')

    # 获取数据库中的人脸图像对应的人名
    faces_name_database = [i[0] for i in faces_feature_database]
    # 获取数据库中的人脸图像的特征向量
    features_database = [i[1] for i in faces_feature_database]

    # 求解距离矩阵
    # 运用欧氏距离对特征向量进行判断
    # e1表示待检测的人脸识别向量，e2表示数据库中的人脸识别向量
    dist = [[(e1 - e2).norm().item() for e1 in features_database] for e2 in faces_feature]
    # 转为数组形式
    dist = np.array(dist)

    # 距离最小值（按行计算）
    dist_min = dist.min(axis=1)
    # 距离最小值的位置（按行计算）
    idx_min = dist.argmin(axis=1)

    # 识别结果
    is_who = [faces_name_database[idx] if num < 0.7 else 'Unknown' for idx, num in zip(idx_min, dist_min)]

    # 原图用于裁剪人脸，转换为PIL的图用于绘制边界框等
    img_draw = Image.fromarray(image)
    draw = ImageDraw.Draw(img_draw)
    # box ----边框
    # point ---- 人脸的坐标
    # text ---- 人名
    for i, (box, point, text) in enumerate(zip(bboxs, points, is_who)):
        # 设置边框线宽度
        # 边框宽度小于300时，rate赋值为1，大于300则将整除后的结果乘以三
        rate = 1 if (box[2] - box[0]) // 300 == 0 else (box[2] - box[0]) // 300 * 3
        # 绘制人脸边框
        draw.rectangle(box.tolist(), width=2 * int(rate))
        # 设置字体并放大，要用自己电脑有的字体
        ft = ImageFont.truetype('bgtx.ttf', size=20 * int(rate))
        # 绘制文本（识别出的人名）
        draw.text(box[:2], text=text, font=ft, fill='blue')
    # 直接show()的，是调用电脑上的图片查看器show的
    img_draw.show()

    if save:
        save_path = os.path.join(os.getcwd(), save_path)  # 保存路径
        os.makedirs(save_path, exist_ok=True)  # 创建路径
        img_draw.save(os.path.join(save_path, os.path.basename(image_path)))  # 保存识别结果


if __name__ == '__main__':
    main(save=True)
