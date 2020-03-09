from builtins import int

import cv2
import numpy as np
import tf
import evaluation as ev
import os

cwd = os.getcwd()
files = os.listdir(cwd + "/static/uploads")
inputFolder = cwd + "/static/uploads"
outputFolder = cwd + "/static/result"
compFolder = cwd + "/static/compare"
for f in files:
    print("f = " + f)
    file_name_without_ext = f.split(".")[0]
    outputFileName = file_name_without_ext + "_enhanced"
    f = tf.getInputPhoto(f)
    tf.processImg(f, outputFileName)
    img1 = cv2.imread(inputFolder + "/" + f)
    img2 = cv2.imread(outputFolder + "/" + outputFileName + ".png")
    vis = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(os.path.join(compFolder, file_name_without_ext + "_vs_" + file_name_without_ext + "_enhanced.jpg"), vis)

# inp = cv2.imread('static/uploads/input.png', 1)
# outp = cv2.imread('static/result/output.png', 1)

# cv2.imshow('Input', inp)
# cv2.imshow('Output', outp)
# cv2.waitKey()

'''
inp = cv2.imread('static/result-db/1a.png', 1)
outp = cv2.imread('static/result-db/1b.png', 1)

psnr = ev.psnr(inp, outp, 1)
print(psnr)
'''
