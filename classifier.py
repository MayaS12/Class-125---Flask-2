import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.datasets import fetch_openml
import PIL.ImageOps 
from PIL import Image


X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state = 42, train_size = 7500, test_size = 2500)

xTrain_Scale = xTrain/255.0
xTest_Scale = xTest/255.0

lr = LogisticRegression(solver = 'saga', multi_class = 'multinomial')

lr.fit(xTrain_Scale, yTrain)

def get_prediction(image):
        im_pil  = Image.fromarray(image)
        img_bw = im_pil.convert('L')
        img_bw_resized = img_bw.resize((28,28), Image.ANTIALIAS)
        img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
        pixel_filter = 20
        minPixel = np.percentile(img_bw_resized_inverted, pixel_filter)
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted - minPixel, 0, 255)
        maxPixel = np.max(img_bw_resized_inverted)
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/maxPixel
        testSample = np.array(img_bw_resized_inverted_scaled).reshape(1,74)
        testPredict = lr.predict(testSample)
        return testPredict[0]