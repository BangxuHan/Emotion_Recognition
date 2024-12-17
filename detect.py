"""
表情识别
"""
import warnings

import cv2
import dlib
import numpy as np
import torch

from model import FaceCNN
from utils import get_face_target, labels, soft_max, cvt_R2N

warnings.filterwarnings("ignore")


class EmotionIdentify:
    """
    用于表情识别的类
    """

    def __init__(self, gui_open=False):
        # 使用特征提取器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # 加载CNN模型
        self.model = FaceCNN()
        self.model.load_state_dict(torch.load('models/cnn_model.pkl').state_dict())
        # 建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture('/home/kls/data/archive_new/happy/contempt000007.jpg')
        # 设置视频参数，propId设置的视频参数，value设置的参数值
        self.cap.set(3, 480)
        # 截图screenshot的计数器
        self.img = None
        self.gui_open = gui_open

    def get_img_and_predict(self):
        """
        表情识别
        :return:(img, emotion)
        """
        gray_img, left_top, right_bottom, result = None, None, None, None
        if self.cap.isOpened():
            # 从摄像头读取图像
            flag, img = self.cap.read()
            self.img = img
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
            faces = self.detector(img_gray, 0)

            if len(faces) > 0:
                # 只取第一张脸
                face = faces[0]
                left_top = (face.left(), face.top())
                right_bottom = (face.right(), face.bottom())
                # 转化为灰度图片并提取目标
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = get_face_target(gray_img, face)
                # 将目标大小重置为神经网络的输入大小
                gray_img = cv2.resize(gray_img, (48, 48))
                target_img = gray_img.reshape((1, 1, 48, 48)) / 255.0
                img_torch = torch.from_numpy(target_img).type('torch.FloatTensor')
                result = self.model(img_torch).detach().numpy()
                result = soft_max(result)

        if self.gui_open:
            """显示图像"""
            show_img = self.img
            if gray_img is not None:
                # cv2.imshow('gray_img', gray_img)
                color = (255, 34, 78)
                cv2.rectangle(show_img, left_top, right_bottom, color, 2)
                emotion = labels[np.argmax(result)]
                cv2.putText(show_img, emotion, left_top, 0, 0.75, color, 2)
            cv2.imshow('cv', show_img)
            cv2.waitKey(0)
            self.cap.release()
            cv2.destroyAllWindows()

            return self.img, result

        raise Exception('No camera')


if __name__ == '__main__':
    ed = EmotionIdentify(gui_open=True)
    flag = True
    while flag is not None:
        flag = ed.get_img_and_predict()

        result = flag[1]
        print(cvt_R2N(result))
