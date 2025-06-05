import torch
from torch import nn
import numpy as np
import os

from models.detect_face import detect_face, extract_face


class PNet(nn.Module):
    """MTCNN PNet.
    Pnet网络粗略获取人脸框, 输出bbox位置和是否有人脸

    Arguments:
        pretrained {bool}: 是否使用预训练权重 (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:  # __file__：当前py脚本所在目录
            state_dict_path = os.path.join(os.path.dirname(__file__), '../weights/pnet.pt')
            state_dict = torch.load(state_dict_path, weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """MTCNN RNet.
    精修框
    
    Arguments:
        pretrained {bool} -- 是否使用预训练权重 (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../weights/rnet.pt')
            state_dict = torch.load(state_dict_path, weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet.
    精修框并获得五个点

    Arguments:
        pretrained {bool} -- 是否使用预训练权重 (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)  # a
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)  # b
        self.dense6_3 = nn.Linear(256, 10) # c

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../weights/onet.pt')
            state_dict = torch.load(state_dict_path, weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class MTCNN(nn.Module):
    """MTCNN 人脸检测模块.

    该类加载预训练的P-Net、R-Net和 O-Net网络，并返回裁剪为仅包括面部的图像,
    输入的图像必须是以下类型之一:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) 一张图像 (3D) 或 一个批次图像 (4D).
    也可以选择将裁剪的图片保存到本地.
    
    Arguments:
        image_size {int} -- 输出图像的大小（以像素为单位）. 图像是方形的. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            注意，这里margin的使用和源tf版的作者(davidsandberg/facenet)在使用策略上有所不同.
            后者在调整大小之前将margin应用于原始图像，使margin取决于原始图像的大小 (这是tf版里的的bug).
            (default: {0})
        min_face_size {int} -- 要搜索的最小面部大小. (default: {20})
        thresholds {list} -- MTCNN 人脸检测阈值 (default: {[0.6, 0.7, 0.7]})
        factor {float} -- 用于创建人脸尺寸缩放金字塔的因子. (default: {0.709})
        post_process {bool} -- 返回前是否对图像张量进行后期处理. (default: {True})
        select_largest {bool} -- If True, 如果检测到多个人脸, 则返回面积最大的那个.
            If False, 则返回检测概率最高的人脸.
            (default: {True})
        selection_method {string} -- 选择人脸的方法. (Default None).
            若指定，则覆盖select_largest的结果:
                    "probability": 选择检测到人脸概率最大的
                    "largest": 选择检测到人脸面积最大的
                    "largest_over_threshold": 概率相同的情况下选择面积大的
                    "center_weighted_size": 计算方框大小减去图像中心的加权平方偏移, 然后选择结果最大的
                (default: {None})
        keep_all {bool} -- If True, 则无视select_largest参数, 返回所有检测到的人脸.
            如果指定了save_path，则将第一个面保存到该路径, 其余面保存到＜save_path＞1、＜save_path＞2等.
            (default: {False})
        device {torch.device} -- 指定模型在device上运行. 在正向传播之前将Image tensor和models复制到device.
            (default: {None})
    """

    def __init__(
        self, image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        select_largest=True, selection_method=None, keep_all=False, device=None
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

        if not self.selection_method:
            self.selection_method = 'largest' if self.select_largest else 'probability'

    def forward(self, img, save_path=None, return_prob=False):
        """
        在PIL.Image或np.ndarray上运行MTCNN人脸检测。
        该方法执行人脸检测和提取，返回表示检测到的人脸而不是边界框的张量。
        要访问边界框，请参阅下面的 MTCNN.detect() 方法.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.
        
        Keyword Arguments:
            save_path {str} -- 裁剪图像的可选保存路径。请注意，当self.post_process=True时，虽然返回的张量是处理后的，
                但保存的人脸图像不是，因此它是输入图像中人脸的真实表示。
                如果“img”是图像列表，则“save_path”应该是一个长度相等的列表.
                (default: {None})
            return_prob {bool} -- 是否返回每张人脸的检测概率.
                (default: {False})
        
        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] --
                如果检测到，则返回 3×160×160的人脸图像.
                并且何以选择是否输出检测到人脸概率, 人脸边界框, 人脸关键点(5个—两个眼睛一个鼻子两边嘴角).
                若 self.keep_all=True, 则返回检测到的多个人脸, shape=[n, 3, 160, 160].
                probabilities.shape = [n,]
                如果输入一个批次图像, 则返回的项具有一个额外的维度（批）作为第一个维度.

        Example:
        >>> from models.mtcnn import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """

        # Detect faces 检测人脸
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces 选择人脸，是否全部人脸都输入，若False，则只输出第一个
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces 裁剪人脸
        faces = self.extract(img, batch_boxes, save_path)

        if return_prob:  # return: 裁剪的人脸, 检测到人脸概率, 人脸边界框, 人脸关键点(5个—两个眼睛一个鼻子两边嘴角)
            return faces, batch_probs, batch_boxes, batch_points
        else:
            return faces, batch_boxes

    def detect(self, img, landmarks=False):
        """检测 PIL图像中的所有人脸，并返回 bounding box和可选的人脸关键点.
        该方法可以使用‘MTCNN().detect()’调用.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
            landmarks {bool} -- 是否返回人脸关键点. (default: {False})
        
        Returns:
            tuple(numpy.ndarray, list) --
                输入的一张图片中：
                1、若检测到 N个人脸，则返回的数组 shape=[N, 4], 其中, 4表示4个bboexs回归值，同时也返回 N个检测到人脸的概率；
                2、如果self.select_largest=False，则返回的框将按检测概率降序排序，否则优先返回面积最大的

                如果“img”是一个图像列表，则返回的项目有一个额外的维度（批）作为第一个维度。
                可选地，如果`landmarks=True，则返回第三项，即人脸关键点`.

        Example:
        >>> from PIL import Image, ImageDraw
        >>> from models.mtcnn import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes)
        probs = np.array(probs)
        points = np.array(points)

        if (
            not isinstance(img, (list, tuple)) and 
            not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
            not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]

        if landmarks:
            return boxes, probs, points

        return boxes, probs

    def select_boxes(
        self, all_boxes, all_probs, all_points, imgs, method='probability', threshold=0.9,
        center_weight=2.0
    ):
        """使用多‘method’参数的方法从给定图像的多个框中选择一个框.

        Arguments:
                all_boxes {np.ndarray} -- self.detect()的输出.
                all_probs {np.ndarray} -- self.detect()的输出.
                all_points {np.ndarray} -- self.detect()的输出.
                imgs {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
                method {str} -- 选择人脸的方法.
                    "probability": 选择检测到人脸概率最大的
                    "largest": 选择检测到人脸面积最大的
                    "largest_over_threshold": 概率相同的情况下选择面积大的
                    "center_weighted_size": 计算方框大小减去图像中心的加权平方偏移, 然后选择结果最大的
                     (default: {'probability'})
                threshold {float} -- 阈值, 当且仅当‘method="largest_over_threshold"’时使用. (default: {0.9})
                center_weight {float} -- 中心加权尺寸法中平方偏移的权重, 当且仅当‘method="center_weighted_size"’时使用.
                    (default: {2.0})

        Returns:
                tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray) --
                    bboxes, shape=[N, 4], 其中, N表示图片数量,
                    probabilities, shape=[I, ], 其中, I表示 bboxes的数量,
                    关键点.
        """

        # copying batch detection from extract, but would be easier to ensure detect creates consistent output.
        # 从 'self.extract()'复制 batch detection, 确保创建一致的输出
        batch_mode = True
        if (
                not isinstance(imgs, (list, tuple)) and
                not (isinstance(imgs, np.ndarray) and len(imgs.shape) == 4) and
                not (isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4)
        ):
            imgs = [imgs]
            all_boxes = [all_boxes]
            all_probs = [all_probs]
            all_points = [all_points]
            batch_mode = False

        selected_boxes, selected_probs, selected_points = [], [], []
        for boxes, points, probs, img in zip(all_boxes, all_points, all_probs, imgs):
            
            if boxes is None:
                selected_boxes.append(None)
                selected_probs.append([None])
                selected_points.append(None)
                continue
            
            # If at least 1 box found
            boxes = np.array(boxes)
            probs = np.array(probs)
            points = np.array(points)
                
            if method == 'largest':
                # 计算每个检测到的人脸面积，返回面积最大的
                # >>> np.argsort([1, 2, 0, 8, 9])  # 先给输入的数组标上index, 然后对数组进行排序, 排序的同时index也会相应调换顺序
                # array([2, 0, 1, 4, 3], dtype=int64)
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
            elif method == 'probability':
                box_order = np.argsort(probs)[::-1]
            elif method == 'center_weighted_size':
                box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                img_center = (img.width / 2, img.height/2)
                box_centers = np.array(list(zip((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)))
                offsets = box_centers - img_center
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 1)
                box_order = np.argsort(box_sizes - offset_dist_squared * center_weight)[::-1]
            elif method == 'largest_over_threshold':
                box_mask = probs > threshold
                boxes = boxes[box_mask]
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
                if sum(box_mask) == 0:
                    selected_boxes.append(None)
                    selected_probs.append([None])
                    selected_points.append(None)
                    continue

            box = boxes[box_order][[0]]
            prob = probs[box_order][[0]]
            point = points[box_order][[0]]
            selected_boxes.append(box)
            selected_probs.append(prob)
            selected_points.append(point)

        if batch_mode:
            selected_boxes = np.array(selected_boxes)
            selected_probs = np.array(selected_probs)
            selected_points = np.array(selected_points)
        else:
            selected_boxes = selected_boxes[0]
            selected_probs = selected_probs[0][0]
            selected_points = selected_points[0]

        return selected_boxes, selected_probs, selected_points

    def extract(self, img, batch_boxes, save_path):
        """ 裁剪人脸 """
        # Determine if a batch or single image was passed
        batch_mode = True
        if (
                not isinstance(img, (list, tuple)) and
                not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
                not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_mode = False

        # 处理保存路径
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]

        # 处理所有 bounding boxes
        faces = []
        for im, box_im, path_im in zip(img, batch_boxes, save_path):
            if box_im is None:
                faces.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                face = extract_face(im, box, self.image_size, self.margin, face_path)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]

            faces.append(faces_im)

        if not batch_mode:
            faces = faces[0]

        return faces


def fixed_image_standardization(image_tensor):
    """ 图像标准化方法 """
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor
