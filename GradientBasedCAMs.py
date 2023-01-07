import itertools
import random
from random import shuffle
import cv2
import os

from keras import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf

from DatabaseGeneration.DatabaseGenerator import getTrainingsDataHeightsDatabase, loadNumpyDataSet
from PIL import Image
import random
from itertools import product

from LearnFromParameters.ModelEvaluation import evaluatePredictions
from SaliencyMetrics import *

"""
Contains routines which could be used for generating and visualisation of gradient based class activation maps. 

"""


# Code adapted from:
# https://colab.research.google.com/drive/1rxmXus_nrGEhxlQK_By38AjwDxwmLn9S?usp=sharing#scrollTo=2orhPMN2Wxbq

def generateGradCamImage(model, img_array, layer_name, eps=1e-8, method="gradCAM", useInterpolation=False,
                         indexOutputClass=0):
    """
    Generates a generateGradCamImage- or HiResGradCam-image of given CNN model and a given layer

    Args:
        useInterpolation: Indicates if an interpolation method should be used when theCAM is upsampled
        model: The Keras model which should be analyzed
        img_array: The numpy image which should be analyzed
        layer_name: The name of the layer for which the GradCam should be computed. Usually the last Conv-layer is used
        for this process
        eps: This parameter avoids the division with 0
        method: Indicates which methode should be used
        useInterpolation: The values after applying the Grad-CAM algorithm have the resolution of the Conv-layer which
        was used for this algorithm. To get the same resolution as the input the result has to be scaled up. This para-
        meter specifies if for this process interpolation should be used
        indexOutputClass: Classifies for which output neuron (output class) the CAM should be generated. For regression
        models this value should be set to 0.

    Returns:
        The resulting GradCam image
    """

    gradModel = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output,
                 model.output])

    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(img_array, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, indexOutputClass]

    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)

    v3 = convOutputs.numpy()

    # compute the guided gradients
    # castConvOutputs = tf.cast(convOutputs > 0, "float32")
    # castGrads = tf.cast(grads > 0, "float32")

    # Calculates the CAM using the algorithm HiRes-CAM
    if method == "hiResCAM":
        outputsNumpy = convOutputs.numpy()
        gradNumpy = grads.numpy()
        guidedGrads = outputsNumpy * gradNumpy
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        multipliedGradientsAndFeatures = tf.multiply(guidedGrads, convOutputs)
        cam = tf.reduce_mean(multipliedGradientsAndFeatures, axis=(-1))
        # cam = tf.nn.relu(cam1)
        # if np.array_equal(cam1.numpy(), cam.numpy()):
        #    print("cams are equal")

    # Calculates the CAM using an adapted version of HiRes-CAM
    # applying the ReLU-function on the gradients before the multiplitcation
    # with the activation map
    elif method == "layerCAM":
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = convOutputs * (grads * castGrads)
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        multipliedGradientsAndFeatures = tf.multiply(guidedGrads, convOutputs)
        cam = tf.reduce_mean(multipliedGradientsAndFeatures, axis=(-1))
        # cam = tf.nn.relu(cam1)
        # if np.array_equal(cam1.numpy(), cam.numpy()):
        #    print("cams are equal")

    # Calculates the CAM using the Grad-CAM algorithm
    elif method == "gradCAM":
        guidedGrads = convOutputs * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        # cam = tf.nn.relu(cam1)
        # if np.array_equal(cam1.numpy(), cam.numpy()):
        #    print("cams are equal")

    # Like the Grad CAM algorithm but using only the positive gradients
    elif method == "gradCAM_posGrads":
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = convOutputs * (grads * castGrads)
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        # cam = tf.nn.relu(cam1)
        # if np.array_equal(cam1.numpy(), cam.numpy()):
        #    print("cams are equal")
    else:
        print("Method \"" + method + "\" is not suppoerted")

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])

    if useInterpolation:
        heatmap = cv2.resize(cam.numpy(), (w, h))
    # use no interpolation when upsampling
    else:
        heatmap = Image.fromarray(cam.numpy())
        heatmap = heatmap.resize(size=(w, h), resample=Image.NEAREST)
        heatmap = np.array(heatmap)

    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    # heatmap = (heatmap * 255).astype("uint8")
    # return the resulting heatmap to the calling function
    return heatmap


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))


def superimpose(img_bgr, cam, thresh, emphasize=False):
    '''
    Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.

    Args:
      image: (img_width x img_height x 3) numpy array
      grad-cam heatmap: (img_width x img_width) numpy array
      threshold: float
      emphasize: boolean

    Returns
      uint8 numpy array with shape (img_height, img_width, 3)

    '''
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    hif = 0.8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img_rgb


def convertTrainingsDataToRGBimage(xData, reshape=True):
    if reshape:
        xData = xData.reshape(xData.shape[:-1])

    rgbImage = np.stack((xData,) * 3, axis=-1)
    # plt.imshow(stacked_img)
    # plt.show()
    # img = Image.fromarray(np.uint8(xData), 'L')
    numer = rgbImage - np.min(rgbImage)
    denom = (rgbImage.max() - rgbImage.min()) + 0.0000000001
    rgbImage = numer / denom
    rgbImage = np.uint8(255 * rgbImage)

    return rgbImage


def convertTraingDataToRGBimage2(imgArray):
    convertedImage = np.minimum(imgArray, 255.0).astype(np.uint8)  # scale 0 to 255
    convertedImage_rgb = cv2.cvtColor(convertedImage, cv2.COLOR_BGR2RGB)
    return convertedImage_rgb


def generateCAMsWithDifferentMethods(nameOfLastHiddenLayer, imageDataInput, outputName="cam"):
    originalImage = convertTrainingsDataToRGBimage(imageDataInput)

    grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer,
                                    method="hiResCAM", useInterpolation=True)
    grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    plotCAMimage(grad_cam_superimposed, originalImage, outputName + "_hiResCAM_interpolation")

    grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer,
                                    method="layerCAM", useInterpolation=True)
    grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    plotCAMimage(grad_cam_superimposed, originalImage, outputName + "_layerCAM_interpolation")

    grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer,
                                    method="gradCAM", useInterpolation=True)
    grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    plotCAMimage(grad_cam_superimposed, originalImage, outputName + "_gradCAM_interpolation")

    grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer,
                                    method="gradCAM_posGrads", useInterpolation=True)
    grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    plotCAMimage(grad_cam_superimposed, originalImage, outputName + "_gradCAM_posGrads_interpolation")

    # grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer, useHiresGradCam=True, useInterpolation=False)
    # grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    # plotCAMimage(grad_cam_superimposed, originalImage, outputName+ "_HiRes_CAM_no_interpolation")


def saveCAMImagesOnly(nameOfLastHiddenLayer, imageDataInput, outputName="cam"):
    originalImage = convertTrainingsDataToRGBimage(imageDataInput)
    saveArrayAsImage(originalImage, outputName + "_original")

    grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer,
                                    method="hiResCAM", useInterpolation=False)
    grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    saveArrayAsImage(grad_cam_superimposed, outputName + "_hiResCAM_interpolation")

    grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer,
                                    method="layerCAM", useInterpolation=False)
    grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    saveArrayAsImage(grad_cam_superimposed, outputName + "_layerCAM_interpolation")

    grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer,
                                    method="gradCAM", useInterpolation=False)
    grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    saveArrayAsImage(grad_cam_superimposed, outputName + "_gradCAM_interpolation")

    grad_cam = generateGradCamImage(model, np.expand_dims(imageDataInput, axis=0), nameOfLastHiddenLayer,
                                    method="gradCAM_posGrads", useInterpolation=False)
    grad_cam_superimposed = superimpose(originalImage, grad_cam, 0.3, emphasize=False)
    saveArrayAsImage(grad_cam_superimposed, outputName + "_gradCAM_posGrads_interpolation")


def plotCAMimage(grad_cam_superimposed, originalImage, outputName):
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(originalImage)
    plt.axis('off')
    plt.title(outputName)
    ax = plt.subplot(1, 2, 2)
    plt.imshow(grad_cam_superimposed)
    plt.axis('off')
    plt.title('Conv_1 Grad-CAM heat-map')
    plt.tight_layout()
    plt.savefig(outputName + ".png")
    # plt.show()


def saveArrayAsImage(arr, imageName):
    img = Image.fromarray(arr)
    imgRescaled = img.resize(size=(512, 512), resample=Image.NEAREST)
    imgRescaled.save(imageName + '.png')


def generateSaliencyMaps(model, nameOfLastHiddenLayer, nameOfMethod, testData):
    saliencyMaps = []
    counter = 0
    for testImage in testData:
        saliencyMap = generateGradCamImage(model, np.expand_dims(testImage, axis=0),
                                           nameOfLastHiddenLayer,
                                           method=nameOfMethod,
                                           useInterpolation=True)
        saliencyMaps.append(saliencyMap)
        counter += 1

    return saliencyMaps


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

domainResolution = 32

modelName = "TransformedData"
#pathToHeightsFile = "/home/thomas/Dokumente/HiWi/vorticity_prediction/Databases/HeightsProfiles_15k/Heights_32_32.csv"
#pathToMaxVorticityFile = "/home/thomas/Dokumente/Studium/Semester_6/Bachelorarbeit/Daten_Bachelorarbeit/RoughnessDatabases/MaxVorticityDatabases/1_6_0_All.csv"
# xTrain, xTest, yTrain, yTest = getTrainingsDataHeightsDatabase(pathToMaxVorticityFile, pathToHeightsFile,
#                                                               domainResolution)
# np.save("xTest", xTest)
# np.save("yTest", yTest)

xTrain, xTest, yTrain, yTest, targetScaler = loadNumpyDataSet("max_omegaX_vort_std")

model = tf.keras.models.load_model("../Models/HeightModels/32_32_model_std")
saliencyMaps = generateSaliencyMaps(model, "thirdConv", "hiResCAM", xTest)
numberReplacedPixels = 100
valuesUsedForTheReplacement = xTest[0].flatten()

mostSalientAreasReplacedByOneValue = replaceHighstValuesForWholeDataSetByOnlyOneValue(xTest, saliencyMaps,
                                                                            numberReplacedPixels,
                                                                            xTest[0][0][0][0])

randomReplacedWithTheSameValue = applyRandomDeletionMetricForWholeDataSetUsingOneValue(xTest, numberReplacedPixels, xTest[0][0][0][0])

mostSalientAreasReplacedWithCertainArea = replaceHighstValuesForWholeDataSetByCertainValues(xTest,
                                                                                            saliencyMaps,
                                                                                            numberReplacedPixels,
                                                                                            valuesUsedForTheReplacement)

randomReplacedWithCertainAreas = applyRandomDeletionMetricForWholeDataSetUsingDifferentValues(xTest,
                                                                                              numberReplacedPixels,
                                                                                              valuesUsedForTheReplacement)

saliencyMapFlippedVertical = applyBitMaskTransformationUsingCAMOnWholeDataSet(xTest, saliencyMaps, flipVertical, numberReplacedPixels,
                                                  valuesUsedForTheReplacement)

saliencyMapFlippedHorizontal = applyBitMaskTransformationUsingCAMOnWholeDataSet(xTest, saliencyMaps, flipHorizontal, numberReplacedPixels,
                                                  valuesUsedForTheReplacement)

def func(x):
    return moveDomain(numberStepsX=0, numberStepsY=3, domain=x)

saliencyMapMovedX = applyBitMaskTransformationUsingCAMOnWholeDataSet(xTest, saliencyMaps,
                                                                     func,
                                                                     numberReplacedPixels, valuesUsedForTheReplacement)
def func2(x):
    return moveDomain(numberStepsX=3, numberStepsY=0, domain=x)

saliencyMapMovedY = applyBitMaskTransformationUsingCAMOnWholeDataSet(xTest, saliencyMaps,
                                                                     func2,
                                                                     numberReplacedPixels, valuesUsedForTheReplacement)

print("")
print("---------------------------------------")
print("Without deletion: ")
predictions = model.predict(xTest)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("Replacement using Saliency map with same values: ")
predictions = model.predict(mostSalientAreasReplacedByOneValue)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("Random replacement using same value: ")
predictions = model.predict(randomReplacedWithTheSameValue)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("Replacement using Saliency map with different values: ")
predictions = model.predict(mostSalientAreasReplacedWithCertainArea)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("Random replacement with different values: ")
predictions = model.predict(randomReplacedWithCertainAreas)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("Saliency mask flipped vertical: ")
predictions = model.predict(saliencyMapFlippedVertical)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("Saliency mask flipped horizontal: ")
predictions = model.predict(saliencyMapFlippedHorizontal)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("Saliency mask moved X: ")
predictions = model.predict(saliencyMapMovedX)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("Saliency mask moved Y: ")
predictions = model.predict(saliencyMapMovedY)
evaluatePredictions(yTest, predictions, targetScaler)
print("---------------------------------------")

path = "../Output/"
for i in range(3):
    saveArrayAsImage(convertTrainingsDataToRGBimage(mostSalientAreasReplacedByOneValue[i]), path + str(i) + "_replacement_saliency_map_same_value")
    saveArrayAsImage(convertTrainingsDataToRGBimage(xTest[i]), path + str(i) + "_original")
    saveArrayAsImage(convertTrainingsDataToRGBimage(saliencyMaps[i], reshape=False), path + str(i) + "_saliency_map")
    saveArrayAsImage(convertTrainingsDataToRGBimage(randomReplacedWithCertainAreas[i]), path + str(i) + "_random_replacement_different_values")
    saveArrayAsImage(convertTrainingsDataToRGBimage(mostSalientAreasReplacedWithCertainArea[i]),
                     path + str(i) + "_replacement_saliency_map_different_value")
    saveArrayAsImage(convertTrainingsDataToRGBimage(randomReplacedWithCertainAreas[i]),
                     path + str(i) + "_random_replacement_different_values")
    saveArrayAsImage(convertTrainingsDataToRGBimage(saliencyMapFlippedVertical[i]),
                     path + str(i) + "_saliency_mask_flipped_vertical")
    saveArrayAsImage(convertTrainingsDataToRGBimage(saliencyMapFlippedHorizontal[i]),
                     path + str(i) + "_saliency_mask_flipped_horizontal")
    saveArrayAsImage(convertTrainingsDataToRGBimage(saliencyMapMovedX[i]),
                     path + str(i) + "_saliency_mask_moved_x")
    saveArrayAsImage(convertTrainingsDataToRGBimage(saliencyMapMovedY[i]),
                     path + str(i) + "_saliency_mask_moved_y")

# for i in range(5):
#    saveCAMImagesOnly("secondConv", xTest[i], "cam2_no_interpol_" + str(i))
# for i, layer in enumerate(model.layers):
#    layer._name = 'layer_' + str(i)

# for i in range(5):
#    saveCAMImagesOnly("secondConv", xTrain[i], "cam2_no_interpol_" + str(i))
