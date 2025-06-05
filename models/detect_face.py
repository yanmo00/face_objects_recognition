import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms
from PIL import Image
import numpy as np
import os
import math

# OpenCV是可选的，但如果使用numpy数组而不是PIL，则需要OpenCV
try:
    import cv2
except:
    pass


def fixed_batch_process(im_data, model):
    """固定批次处理，防止电脑内存不够中断代码运行, 同时也传入 rnet或onet进行bboxes回归，和判断bboxes是否包含人脸"""
    batch_size = 512
    out = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i:(i + batch_size)]
        out.append(model(batch))

    return tuple(torch.cat(v, dim=0) for v in zip(*out))


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    """"人脸检测"""
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        if isinstance(imgs, np.ndarray):
            imgs = torch.as_tensor(imgs.copy(), device=device)

        if isinstance(imgs, torch.Tensor):
            imgs = torch.as_tensor(imgs, device=device)

        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
        imgs = np.stack([np.uint8(img) for img in imgs])
        imgs = torch.as_tensor(imgs.copy(), device=device)

    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # 创建图像缩放比例金字塔
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # stage1
    # 使用P-Net是一个全卷积网络，用来生成候选窗和边框回归向量(bounding box regression vectors)。
    # 使用Bounding box regression的方法来校正这些候选窗，使用非极大值抑制（NMS）合并重叠的候选框
    boxes = []
    image_inds = []

    scale_picks = []

    all_i = 0
    offset = 0
    # for循环生成不同尺寸的候选框
    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)

        # 生成候选框
        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)

        # 使用NMS初步抑制 IoU>0.5的候选框
        pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0)
    scale_picks = torch.cat(scale_picks, dim=0)

    # NMS within each scale + image
    boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

    # 为每张图像执行NMS
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]  # 候选框的宽
    regh = boxes[:, 3] - boxes[:, 1]  # 候选框的高
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)  # 粗精修框
    y, ey, x, ex = pad(boxes, w, h)  # 保证人脸的边界框不会越界

    # stage2
    # 使用R-Net改善候选窗。将通过P-Net的候选窗输入R-Net中，
    # 拒绝掉大部分false的窗口，继续使用Bounding box regression和NMS
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))  # 重采样至 24×24 大小
        im_data = torch.cat(im_data, dim=0)  # shape=[102, 3, 24, 24]
        im_data = (im_data - 127.5) * 0.0078125  # 标准化

        # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, rnet)  # out[0].shape=[102, 4],  out[1].shape=[102, 2]

        out0 = out[0].permute(1, 0)  # shape=[4, 102], 通过rnet后的bboxes回归值
        out1 = out[1].permute(1, 0)  # shape=[2, 102], 通过rnet后的bboxes中包含人脸的概率
        score = out1[1, :]
        ipass = score > threshold[1]  # 取出包含人脸概率>0.7的bboxes
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # 为每张图像执行NMS
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)  # output: 执行nms后保留下来的index
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick] # 取出上一步保留下来的bboxes、图片index、bboxes_index
        boxes = bbreg(boxes, mv)  # 细精修框
        boxes = rerec(boxes)      # 粗精修框

    # stage3
    # 最后使用O-Net输出最终的人脸框和特征点位置。和第二步类似，但是不同的是生成5个特征点位置
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125

        # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, onet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[1, :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)  # 细精修框

        # 每张图片均执行 NMS并采用 "Min" 策略
        # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)  # 官方定义的batched_nms方法
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    image_inds = image_inds.cpu()

    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds].copy())
        batch_points.append(points[b_i_inds].copy())

    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)

    return batch_boxes, batch_points


def bbreg(boundingbox, reg):
    """
    精修框
    Args:
        boundingbox: 针对于图片的bboxes
        reg: mv值，表示经过rnet或onet模型卷积之后的bboxes回归值
    """
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
    """生成 bounding box"""
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
    """非极大值抑制"""
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:  # numel(): 计算tensor中有多少个值
        return torch.empty((0,), dtype=torch.int64, device=device)
    # 策略：为每个类独立执行NMS。我们对所有框添加偏移量。
    # 偏移量仅取决于类idx，并且足够大，使得来自不同类的框不会重叠。
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def pad(boxes, w, h):
    """保证人脸的边界框不会越界"""
    # w, h = 输入图片的宽高
    boxes = boxes.trunc().int().cpu().numpy()  # trunc(): 向下取整
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    # 以下操作保证人脸的边界框不会越界
    x[x < 1] = 1    # bboxes中，小于1的表示越界，设置为1
    y[y < 1] = 1
    ex[ex > w] = w  # bboxes中，大于w的表示越界，设置为w
    ey[ey > h] = h

    return y, ey, x, ex


def rerec(bboxA):
    """精修框"""
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]

    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bboxA


def imresample(img, sz):
    """
    插值采样，将图片采用插值方法放大或缩小
    这里，不管原图多大或多小，统一缩放致给定的size大小，原作者设置size=(160, 160)
    """
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    """裁剪图像并重置大小"""
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def save_img(img, path):
    """保存图像"""
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def get_size(img):
    """获取图像尺寸"""
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """
    利用给定的bboxes从图像中提取人脸
    参数:
        img {PIL.Image}: PIL图像.
        box {numpy.ndarray}: 4个值的bounding box.
        image_size {int}: 输出图像大小（以像素为单位）, 图像为方形.
        margin {int}: Margin to add to bounding box, in terms of pixels in the final image.
            注意，这里margin的使用和源tf版的作者(davidsandberg/facenet)在使用策略上有所不同.
            后者在调整大小之前将margin应用于原始图像，使margin取决于原始图像的大小 (这是tf版里的的bug).
        save_path {str}: 提取到的人脸图像的保存路径. (default: {None})
    
    Returns:
        torch.tensor -- 脸部特征张量.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)  # -> shape=(160, 160, 3)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    face = F.to_tensor(np.float32(face))  # -> shape=torch.Size([3, 160, 160])

    return face
