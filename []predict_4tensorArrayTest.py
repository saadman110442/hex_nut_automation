"in this code we have tested the alternative of saving each image in a file"
"the soluting could be converting the numpy array to a tensor array having 4 dimensions"
"and inserting this 4d array directly to the CNN"
"without prediction for loop speed 29fps"
"commenting the score printing statement offered no significant improvement"
"commenting the class finding statement had offered no significant improvement"
"the problem seems to be speed of prediction"
"proceeded to install gpu relevant files" \
"installed CUDA10.1 cudnn 7.6.5 following https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d" \
"faced problem with cudnn 7.6.5" \
"fixed with http://devmartin.com/blog/2020/11/no-libcudnn-ubuntu-focal-tensorflow231/" \
"new problem arrised of CUDA runtime implicit initialization on GPU:0 failed. Status: device kernel image is invalid"
"from we it was found that GTX960m has compute capability of 5 which is not supported by CUDA10.1" \
"gtx960m driver installed 460"
'try this https://peshmerge.io/how-to-install-cuda-9-0-on-with-cudnn-7-1-4-on-ubuntu-18-04/'
'upgrading tensorflow from 2.3.0 to 2.4.1 did not solve the issue'
"i had to downgrade to 2.2.0 in order to get the gpu to work"
'now saving and loading models in json is creating issues'
'given up on GPU processing due to unavailable old gpu support of tensorflow' \
'changing model.predict to model solved the issue speed increased from .8fps to 7.5fps https://github.com/tensorflow/tensorflow/issues/40261'

"25rpm 200rpm"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow import keras
import cv2
from pypylon import pylon
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

cropped = [0] * 41
resized = [0] * 41
tablet = [0] * 41
compareList = ['good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good']

t1x1 = 229
t1y1 = 345
t1x2 = 313
t1y2 = 392


# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Loading the model
json_file = open("[]model-bw.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("[]model-bw.h5")
print("Loaded model from disk")

batch_size = 32
img_height = 64
img_width = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/test',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
class_names = train_ds.class_names

class_names = ['blank', 'broken', 'good']

# fps variables
fpsReport = 0
timeStamp = time.time()

while camera.IsGrabbing():

    # # img = keras.preprocessing.image.load_img(
    # #     "marked.jpg", target_size=(img_height, img_width)
    # # )`
    # imgRaw = imgDisplay = cv2.imread("sample4.jpg")
    # #    imgDisplay = cv2.imread("sample3.jpg")
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        imgRaw = image.GetArray()
        imgDisplay = image.GetArray()

    # imgRaw = cv2.imread("blank.bmp")
    # imgDisplay = cv2.imread("blank.bmp")

    cropped[1] = imgRaw[t1y1:t1y2, t1x1:t1x2]




    i = 1
    resized[i] = cv2.resize(cropped[i], (64, 64))
    img_array = np.expand_dims(resized[i], axis=0)
    predictions = model(img_array)
    # tablet[i] = class_names[np.argmax(predictions)]
    # tablet[i] = model.predict_classes(img_array)
    # print(tablet[i])

    score = tf.nn.softmax(predictions[0])
    tablet[i] = class_names[np.argmax(score)]

    print(
        "tablet {} {} with a {:.2f} percent confidence."
        .format(i, class_names[np.argmax(score)], 100 * np.max(score))
    )

    if tablet[1] == "good":
        cv2.rectangle(imgDisplay, (t1x1, t1y1), (t1x2, t1y2), (0, 255, 0), 1)
    if tablet[1] == "blank":
        cv2.rectangle(imgDisplay, (t1x1, t1y1), (t1x2, t1y2), (0, 0, 255), 1)
    if tablet[1] == "broken":
        cv2.rectangle(imgDisplay, (t1x1, t1y1), (t1x2, t1y2), (0, 255, 255), 1)


## decision sent to pyserial
    # fpsCounter
    dt = time.time() - timeStamp
    fps = 1 / dt
    fpsReport = .90 * fpsReport + .1 * fps
    # print('fps is: ', round(fpsReport, 1))
    timeStamp = time.time()
    cv2.rectangle(imgDisplay, (0, 0), (100, 40), (0, 0, 255), -1)
    cv2.putText(imgDisplay, str(round(fpsReport, 1)) + 'fps', (0, 25), cv2.FONT_HERSHEY_COMPLEX, .75,
                (0, 255, 255, 2))

    cv2.imshow('frame', imgDisplay)
    k = cv2.waitKey(1)
    if k == 27:
        break
