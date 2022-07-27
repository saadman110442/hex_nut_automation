import cv2
import os
import datetime
from pypylon import pylon

mx = 0
my = 0

good = False
blank = False
broken = False
marked = False
train = False
test = False
goodAll = False
blankAll = False
brokenAll = False
markedAll = False

tabletNo = 00

t1x1 = 548
t1y1 = 224
t1x2 = t1x1 + 300
t1y2 = t1y1 + 300



textFontSize = .85
textThickness = 2
# """length 130 height 40"""
# good sample button rectangle
b1x1 = 3
b1y1 = 3  # 349
b1x2 = 130
b1y2 = 43  # 398

# blank sample button rectangle
b2x1 = 3
b2y1 = b1y1 + 50
b2x2 = b1x2
b2y2 = b1y1 + 90

# broken sample button rectangle
b3x1 = 3
b3y1 = b2y1 + 50
b3x2 = b1x2
b3y2 = b2y1 + 90

# marked sample button rectangle
b4x1 = 3
b4y1 = b3y1 + 50
b4x2 = b1x2
b4y2 = b3y1 + 90

# train mode button
b5x1 = 130
b5y1 = b4y2
b5x2 = 215
b5y2 = b4y2 + 40

# test mode button
b6x1 = 220
b6y1 = b4y2
b6x2 = 300
b6y2 = b4y2 + 40



n1x1 = 135
n1y1 = b1y1

n2x1 = n1x1
n2y1 = b2y1

n3x1 = n1x1
n3y1 = b3y1

n4x1 = n1x1
n4y1 = b4y1

n5x1 = 220
n5y1 = b1y1

n6x1 = n5x1
n6y1 = b2y1

n7x1 = n5x1
n7y1 = b3y1

n8x1 = n5x1
n8y1 = b4y1

"""
t3x1=
t3y1=
t3x2=
t3y2=
"""

# Create the directory structure
if not os.path.exists("[]data"):
    os.makedirs("[]data")
    os.makedirs("[]data/train")
    os.makedirs("[]data/test")
    os.makedirs("[]data/train/good")
    os.makedirs("[]data/train/broken")
    os.makedirs("[]data/train/blank")
    os.makedirs("[]data/test/good")
    os.makedirs("[]data/test/broken")
    os.makedirs("[]data/test/blank")


# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# defining mouse events
def mouse_drawing(event, x, y, flags, params):
    global mx, my, good, blank, broken, marked, train, test, goodAll, blankAll, brokenAll, markedAll
    mx = x
    my = y
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imwrite("[]data/a_" + str(dateTimeNow) + ".jpg", img)
        if (b1x1 < mx < b1x2) and (b1y1 < my < b1y2):
            good = True
            blank = False
            broken = False
            marked = False
        if (b2x1 < mx < b2x2) and (b2y1 < my < b2y2):
            good = False
            blank = True
            broken = False
            marked = False
        if (b3x1 < mx < b3x2) and (b3y1 < my < b3y2):
            good = False
            blank = False
            broken = True
            marked = False
        if (b4x1 < mx < b4x2) and (b4y1 < my < b4y2):
            good = False
            blank = False
            broken = False
            marked = True
        if (b5x1 < mx < b5x2) and (b5y1 < my < b5y2):
            train = True
            test = False
        if (b6x1 < mx < b6x2) and (b6y1 < my < b6y2):
            train = False
            test = True

        if good is True:
            if train is True:
                if (t1x1 < mx < t1x2) and (t1y1 < my < t1y2):
                    cropped = img[t1y1:t1y2, t1x1:t1x2]
                    resized = cv2.resize(cropped, (64, 64))
                    cv2.imwrite("[]data/train/good/1_" + str(dateTimeNow) + ".jpg", resized)
            else:
                if (t1x1 < mx < t1x2) and (t1y1 < my < t1y2):
                    cropped = img[t1y1:t1y2, t1x1:t1x2]
                    resized = cv2.resize(cropped, (64, 64))
                    cv2.imwrite("[]data/test/good/1_" + str(dateTimeNow) + ".jpg", resized)

        if blank is True:
            if train is True:
                if (t1x1 < mx < t1x2) and (t1y1 < my < t1y2):
                    cropped = img[t1y1:t1y2, t1x1:t1x2]
                    resized = cv2.resize(cropped, (64, 64))
                    cv2.imwrite("[]data/train/blank/1_" + str(dateTimeNow) + ".jpg", resized)
 
            else:
                if (t1x1 < mx < t1x2) and (t1y1 < my < t1y2):
                    cropped = img[t1y1:t1y2, t1x1:t1x2]
                    resized = cv2.resize(cropped, (64, 64))
                    cv2.imwrite("[]data/test/blank/1_" + str(dateTimeNow) + ".jpg", resized)
 
        if broken is True:
            if train is True:
                if (t1x1 < mx < t1x2) and (t1y1 < my < t1y2):
                    cropped = img[t1y1:t1y2, t1x1:t1x2]
                    resized = cv2.resize(cropped, (64, 64))
                    cv2.imwrite("[]data/train/broken/1_" + str(dateTimeNow) + ".jpg", resized)
 
            else:
                if (t1x1 < mx < t1x2) and (t1y1 < my < t1y2):
                    cropped = img[t1y1:t1y2, t1x1:t1x2]
                    resized = cv2.resize(cropped, (64, 64))
                    cv2.imwrite("[]data/test/broken/1_" + str(dateTimeNow) + ".jpg", resized)
 

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_drawing)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    dateTimeNow = datetime.datetime.now()
    if grabResult.GrabSucceeded():
        # Access the image []data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        imgDisplay = image.GetArray()


        cv2.rectangle(imgDisplay, (t1x1, t1y1), (t1x2, t1y2), (255, 0, 0), 1)


        cv2.rectangle(imgDisplay, (0, 0), (380, 250), (0, 0, 0), -1)  # shape filled black
        cv2.rectangle(imgDisplay, (b1x1, b1y1), (b1x2, b1y2), (0, 255, 0), 1)
        cv2.rectangle(imgDisplay, (b2x1, b2y1), (b2x2, b2y2), (0, 255, 0), 1)
        cv2.rectangle(imgDisplay, (b3x1, b3y1), (b3x2, b3y2), (0, 255, 0), 1)
        cv2.rectangle(imgDisplay, (b4x1, b4y1), (b4x2, b4y2), (0, 255, 0), 1)
        cv2.rectangle(imgDisplay, (b5x1, b5y1), (b5x2, b5y2), (0, 255, 0), 1)  # train mode button rectangle
        cv2.rectangle(imgDisplay, (b6x1, b6y1), (b6x2, b6y2), (0, 255, 0), 1)  # test mode button rectangle

        # count the number of files
        count = {'train_good': len(os.listdir("[]data/train/good")),
                 'train_blank': len(os.listdir("[]data/train/blank")),
                 'train_broken': len(os.listdir("[]data/train/broken")),

                 'test_good': len(os.listdir("[]data/test/good")),
                 'test_blank': len(os.listdir("[]data/test/blank")),
                 'test_broken': len(os.listdir("[]data/test/broken"))}
        # display the number over the screen
        cv2.putText(imgDisplay, str(count['train_good']), (n1x1, n1y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize,
                    (0, 255, 0),
                    textThickness)
        cv2.putText(imgDisplay, str(count['train_blank']), (n2x1, n2y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize,
                    (0, 255, 0),
                    textThickness)
        cv2.putText(imgDisplay, str(count['train_broken']), (n3x1, n3y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize,
                    (0, 255, 0),
                    textThickness)

        cv2.putText(imgDisplay, str(count['test_good']), (n5x1, n5y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize,
                    (0, 255, 0),
                    textThickness)
        cv2.putText(imgDisplay, str(count['test_blank']), (n6x1, n6y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize,
                    (0, 255, 0),
                    textThickness)
        cv2.putText(imgDisplay, str(count['test_broken']), (n7x1, n7y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize,
                    (0, 255, 0),
                    textThickness)


        if good:
            cv2.putText(imgDisplay, "Good", (b1x1 + 10, b1y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 255, 0),
                        textThickness)
        if not good:
            cv2.putText(imgDisplay, "Good", (b1x1 + 10, b1y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 0, 255),
                        textThickness)
        if blank:
            cv2.putText(imgDisplay, "Blank", (b2x1 + 10, b2y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 255, 0),
                        textThickness)
        if not blank:
            cv2.putText(imgDisplay, "Blank", (b2x1 + 10, b2y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 0, 255),
                        textThickness)
        if broken:
            cv2.putText(imgDisplay, "Broken", (b3x1 + 10, b3y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 255, 0),
                        textThickness)
        if not broken:
            cv2.putText(imgDisplay, "Broken", (b3x1 + 10, b3y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 0, 255),
                        textThickness)
        if marked:
            cv2.putText(imgDisplay, "Marked", (b4x1 + 10, b4y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 255, 0),
                        textThickness)
        if not marked:
            cv2.putText(imgDisplay, "Marked", (b4x1 + 10, b4y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 0, 255),
                        textThickness)
        if train:
            cv2.putText(imgDisplay, "Train", (b5x1 + 5, b5y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 255, 0),
                        textThickness)
        if not train:
            cv2.putText(imgDisplay, "Train", (b5x1 + 5, b5y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 0, 255),
                        textThickness)
        if test:
            cv2.putText(imgDisplay, "Test", (b6x1 + 10, b6y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 255, 0),
                        textThickness)
        if not test:
            cv2.putText(imgDisplay, "Test", (b6x1 + 10, b6y1 + 28), cv2.FONT_HERSHEY_COMPLEX, textFontSize, (0, 0, 255),
                        textThickness)

        cv2.imshow('frame', imgDisplay)
        cv2.imwrite('acc.jpg', imgDisplay)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()