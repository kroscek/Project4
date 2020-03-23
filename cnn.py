from keras.models import load_model
from scipy import misc
import numpy as np
import os

os.chdir('/home/lemma/MNIST-GUI')
# need scipy==1.2.1 to run this script
model = load_model("MnistModel.h5")


def predict(inpout):
    image = misc.imread(inpout, mode="L")
    image = np.invert(image)
    image = misc.imresize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)

    return model.predict(image)[0].tolist().index(1)
