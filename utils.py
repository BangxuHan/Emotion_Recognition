"""
使用的工具
"""
import numpy as np

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}


def get_face_target(gray_img: np.ndarray, face):
    """
    从gray_img中提取面部图像
    :param gray_img:
    :param face:
    :return:
    """
    img_size = gray_img.shape
    # print(img_size)
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

    def reset_point(point, max_size):
        """
        重置点的位置
        :param point: 当前点的坐标
        :param max_size: 点在坐标轴的上的最大位置
        :return:
        """
        if point < 0:
            point = 0
        if point > max_size:
            point = max_size - 1
        return point

    x1 = reset_point(x1, img_size[0])
    x2 = reset_point(x2, img_size[0])
    y1 = reset_point(y1, img_size[1])
    y2 = reset_point(y2, img_size[1])
    return gray_img[y1:y2, x1:x2]


def soft_max(x):
    """classify"""
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x)
    y = exp_x / sum_x
    return y


def cvt_R2N(result):
    """将神经网络预测的结果转化为0-100的数值"""
    if result is None:
        return 80
    cvt_M = [70, 20, 60, 100, 10, 90, 80]
    cvt_M = np.array(cvt_M)
    level = np.sum(cvt_M * result)
    return level
