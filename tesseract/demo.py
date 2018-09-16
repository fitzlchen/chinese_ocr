# -*- coding:utf-8 -*-
from PIL import Image
from keras.applications.vgg16 import preprocess_input, VGG16
import pytesseract as pyt
import glob
import re
import numpy as np
from keras.optimizers import SGD


def load_model():
    """加载模型"""
    sgd = SGD(lr=0.00001, momentum=0.9)
    model = VGG16(weights=None, classes=4)
    model.load_weights('../angle/modelAngle.h5', by_name=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def angle_detect(img):
    ROTATE = [0, 270, 180, 90]
    w, h = img.size
    thesh = 0.05
    xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
    im = img.crop((xmin, ymin, xmax, ymax))  ##剪切图片边缘，清楚边缘噪声
    im = im.resize((224, 224))
    img = np.array(im)
    img = preprocess_input(img.astype(np.float32))
    model = load_model()
    pred = model.predict(np.array([img]))
    index = np.argmax(pred, axis=1)[0]
    return ROTATE[index]


def rotate_image(angle, img):
    if angle == 90:
        img = img.transpose(Image.ROTATE_90)
    elif angle == 180:
        img = img.transpose(Image.ROTATE_180)
    elif angle == 270:
        img = img.transpose(Image.ROTATE_270)
    img = np.array(img)
    return img


def ocr(img):
    result = pyt.image_to_string(img)
    return result


def extract_id_number(sentence):
    pattern = re.compile(r'([a-zA-Z0-9]{18})')
    matcher = re.search(pattern, sentence)
    if matcher is None:
        return None
    else:
        return matcher.group(1)


if __name__ == '__main__':
    images = glob.glob('../test_images/*.*')
    for image in images:
        image_name = image.title()
        image = Image.open(image).convert('RGB')
        # 图片倾斜矫正
        angle = angle_detect(image)
        image = rotate_image(angle, image)
        # 内容识别
        sentence = ocr(image)
        id = extract_id_number(sentence)
        print('image:{}, recognize id:{}'.format(image_name, id))
