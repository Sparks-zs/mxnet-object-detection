import cv2
import time
import threading
from collections import deque
import mxnet as mx
from tools.tools import try_gpu, import_module
lock = threading.Lock()


def img_transform(img, img_size=500):
    img = mx.image.imresize(img, img_size, img_size, 3).astype('float32')
    orig_img = img.asnumpy().astype('uint8')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = mx.nd.array(mean).reshape((1, 1, -1))
    std = mx.nd.array(std).reshape((1, 1, -1))
    out_img = (img / 255.0 - mean) / std
    out_img = out_img.transpose((2, 0, 1)).expand_dims(axis=0)  # 通道 h w c->c h w

    return out_img, orig_img


# 预测目标
def predict(test_image, net):
    anchors, bbox_preds, cls_preds = net(test_image)
    cls_probs = mx.nd.SoftmaxActivation(cls_preds.transpose((0, 2, 1)), mode='channel')
    output = mx.nd.contrib.MultiBoxDetection(cls_probs, bbox_preds, anchors,
                                             force_suppress=True, clip=True,
                                             threshold=0.5, nms_threshold=.45)

    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    if idx:
        return output[0, idx]


# 摄像头的显示
class WebcamThread(threading.Thread):
    def __init__(self, input, output, img_height, img_width, threshold=0.5, labels=None):
        super(WebcamThread).__init__()
        self._jobq = input
        self._output = output
        self._num = 0
        self.cap = cv2.VideoCapture(0)
        self.img_height = img_height
        self.img_width = img_width
        self.labels = labels
        self.threshold = threshold
        threading.Thread.__init__(self)

    def run(self):
        start = time.time()
        cv2.namedWindow('camera', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        if not self.cap.isOpened():
            print('摄像头打开失败')
        while self.cap.isOpened():
            # 计算fps
            if self._num < 60:
                self._num += 1
            else:
                end = time.time()
                fps = self._num / (end - start)

                start = time.time()
                self._num = 0
                print('fps:', fps)

            ret, frame = self.cap.read()
            lock.acquire()
            if len(self._jobq) == 10:
                self._jobq.popleft()
            else:
                self._jobq.append(frame)
            lock.release()
            frame = cv2.resize(frame, (self.img_width, self.img_height))
            if self._output[0] is not None:
                output = self._output[0]
                for row in output:
                    score = row[1].asscalar()
                    if score < self.threshold:
                        cv2.imshow('camera', frame)
                        continue
                    bounding_boxes = [
                        row[2:6] * mx.nd.array((self.img_width, self.img_height, self.img_width, self.img_height),
                                               ctx=row.context)]
                    for bbox in bounding_boxes:
                        bbox = bbox.asnumpy()
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (205, 0, 0), 2)
                        if self.labels:
                            label = self.labels[int(row[0].asscalar())]
                            font = cv2.FONT_HERSHEY_TRIPLEX
                            cv2.putText(frame, label, (bbox[0], bbox[1]), font, 0.8, (205, 0, 0), 1, cv2.LINE_8)
                        cv2.imshow('camera', frame)
            else:
                cv2.imshow('camera', frame)
            if cv2.waitKey(1) == ord('q'):
                # 退出程序
                break
        print("实时读取线程退出！！！！")
        cv2.destroyWindow('camera')
        self._jobq.clear()  # 读取进程结束时清空队列
        self.cap.release()


# 处理摄像头传来的数据
class ModelDealhread(threading.Thread):
    def __init__(self, input, output, img_size, ctx=mx.cpu()):
        super(ModelDealhread).__init__()
        self._jobq = input
        self._output = output
        self.img_size = img_size
        self.ctx = ctx
        threading.Thread.__init__(self)

    def run(self):
        flag = False
        while True:
            if len(self._jobq) != 0:
                lock.acquire()
                im_new = self._jobq.pop()
                lock.release()

                frame = mx.nd.array(cv2.cvtColor(im_new, cv2.COLOR_BGR2RGB)).astype('uint8')
                img, frame = img_transform(frame, img_size=self.img_size)
                output = predict(img.as_in_context(self.ctx), net)

                lock.acquire()
                self._output[0] = output
                lock.release()
                # cv2.waitKey(500)
                flag = True
            elif flag is True and len(self._jobq) == 0:
                break

        print("间隔1s获取图像线程退出！！！！")


if __name__ == "__main__":
    ctx = try_gpu()
    # net = vgg_ssd.get_model(3, pretrained_model='model/mask_SSD_model.params', pretrained=True,ctx=ctx)
    net = import_module('model.resnet_ssd').get_model(3, pretrained_model='model/mask_resnet18_SSD_model.params',
                                                      pretrained=True, ctx=ctx)
    net.hybridize()

    q = deque([], 10)   # 双端队列，存储当前帧
    output_q = [None]   # 模型的输出
    labels = ['without_mask', 'with_mask', 'mask_weared_incorrect']
    th1 = WebcamThread(q, output_q, 500, 500, labels=labels)
    th2 = ModelDealhread(q, output_q, 500, ctx=ctx)

    # 开启两个线程
    th1.start()
    th2.start()

    th1.join()
    th2.join()
