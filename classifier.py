from pydoc import importfile
import numpy as np;
import pandas as pd;
from sklearn.datasets import fetch_openml;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;
from PIL import Image;
import PIL.ImageOps;


X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)

xtrain_scaled = xtrain / 255
xtest_scaled = xtest / 255

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(xtrain_scaled, ytrain)

def get_prediction(image) :
    # opening image
    im_pil = Image.open(image)
    # Converting into gray and scalar(same extension ex- png, jpeg) quantity so that extension and colors don't change our prediction
    image_bw = im_pil.convert("L")
    # Resizing the image
    image_bw_resize = image_bw.resize((22, 30), Image.ANTIALIAS)
    # Using percentile to set minimum pixels to use clip function
    pixel_filter = 20
    minimum_pixel = np.percentile(image_bw_resize, pixel_filter)
    # Inverting the image using clip
    image_inverted = np.clip(image_bw_resize - minimum_pixel, 0, 255)
    # Changing/Maximizing the pixels again 
    maximum_pixel = np.max(image_bw_resize)
    # Shifting all the images to one array
    image_array = np.asarray(image_inverted) / maximum_pixel
    test_sample = np.array(image_array).reshape(1, 660)
    test_prediction = clf.predict(test_sample)
    return test_prediction[0]