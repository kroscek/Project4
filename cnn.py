from keras.models import load_model
import numpy as np
import os
from keras.models import model_from_json

os.chdir('/home/lemma/MNIST-GUI')
model = load_model("MnistModel.h5")
from PIL import ImageFilter
from PIL import Image


def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if nheight == 0:  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if nwidth == 0:  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
    tv = list(newImage.getdata())  # get pixel values
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


class Model:
    def __init__(self):
        with open("model.json") as json_model:
            self.model = model_from_json(json_model.read())
            json_model.close()

        self.model.load_weights("model.weights")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def Predict(self, image):
        array = [imageprepare(image)]
        imageArray = [[0 for d in range(28)] for y in range(28)]
        k = 0
        for i in range(28):
            for j in range(28):
                imageArray[i][j] = array[0][k]
                k = k + 1

        imageArray = np.array(imageArray)
        imageArray = imageArray.reshape(1, 28, 28, 1)
        scores = self.model.predict(np.array(imageArray))

        number = 0
        bestScore = -1
        prediction = -1
        for score in scores[0]:
            if score > bestScore:
                bestScore = score
                prediction = number

            number += 1

        return prediction, scores[0]
