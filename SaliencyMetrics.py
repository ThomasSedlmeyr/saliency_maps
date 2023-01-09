import itertools
from itertools import product
import random
import numpy as np



"""
This file contains routines for the evaluation of certain CAMs (Class Activation maps)
"""


def applyRandomDeletionMetricQuadraticAreaOnWholeDataSet(xData, sizeX, sizeY, valueUsedForTheDeletion):
    resultData = np.empty(xData.shape)
    width = xData.shape[1]

    for i in range(xData.shape[0]):
        randomPosX = random.randint(0, width - sizeX - 1)
        randomPoxY = random.randint(0, width - sizeY - 1)
        for x in range(sizeX):
            for y in range(sizeY):
                resultData[i][randomPosX + x][randomPoxY + y] = valueUsedForTheDeletion

    return resultData


def applyRandomDeletionMetricForWholeDataSetUsingOneValue(xData, numberElementsWhichShouldBeReplaced,
                                                          valueUsedForTheReplacement):
    resultData = np.empty(xData.shape)

    for i in range(xData.shape[0]):
        resultData[i] = applyRandomDeletionMetricForOneImageUsingOneValue(xData[i], numberElementsWhichShouldBeReplaced,
                                                                     valueUsedForTheReplacement)
    return resultData


def applyRandomDeletionMetricForWholeDataSetUsingDifferentValues(xData, numberValues, valuesUsedForTheReplacement):
    resultData = np.empty(xData.shape)

    for i in range(xData.shape[0]):
        resultData[i] = applyRandomDeletionMetricForOneImageUsingDifferentValues(xData[i], numberValues, valuesUsedForTheReplacement)

    return resultData


def applyRandomDeletionMetricForOneImageUsingOneValue(xData, numberReplacedValues, valueUsedForTheReplacement):
    valuesUsedForTheReplacement = [valueUsedForTheReplacement] * numberReplacedValues
    return applyRandomDeletionMetricForOneImageUsingDifferentValues(xData, numberReplacedValues, valuesUsedForTheReplacement)


def applyRandomDeletionMetricForOneImageUsingDifferentValues(xData, numberValues, valuesUsedForTheReplacement):
    """
    Replaces the numberReplacedValues pixels at random positions by valuesUsedForTheReplacement

    Args:
        inputData: The input image
        numberReplacedValues: The number values which should be replaced
        valuesUsedForTheReplacement: The pixels which should be used for the replacement

    Returns:
        resultData: The transformed data
    """

    #For each area the salient area with the same value

    resultData = np.copy(xData)
    width = xData.shape[1]

    randomTuples = random.sample(list(product(range(width), repeat=2)), k=numberValues)
    counter = 0
    for (x, y) in randomTuples:
        resultData[x][y][0] = valuesUsedForTheReplacement[counter]
        counter += 1

    return resultData

def replaceHighstValuesForWholeDataSetByCertainValues(xData, saliencyMaps, numberReplacedValues, valuesUsedForTheReplacement):
    resultData = np.empty(xData.shape)

    for i in range(xData.shape[0]):
        resultData[i] = replaceHighstValuesForOneImageByCertainValues(xData[i], saliencyMaps[i], numberReplacedValues,
                                                                      valuesUsedForTheReplacement)
    return resultData


def replaceHighstValuesForWholeDataSetByOnlyOneValue(xData, saliencyMaps, numberElementsWhichShouldBeReplaced,
                                                          valueUsedForTheReplacement):
    resultData = np.empty(xData.shape)

    for i in range(xData.shape[0]):
        resultData[i] = replaceHighstValuesForOneImageByOnlyOneValue(xData[i], saliencyMaps[i],
                                                                     numberElementsWhichShouldBeReplaced,
                                                                     valueUsedForTheReplacement)
    return resultData


def replaceHighstValuesForOneImageByOnlyOneValue(xData, saliencyMap, numberReplacedValues, valueUsedForTheReplacement):
    valuesUsedForTheReplacement = [valueUsedForTheReplacement] * numberReplacedValues
    return replaceHighstValuesForOneImageByCertainValues(xData, saliencyMap, numberReplacedValues, valuesUsedForTheReplacement)


def replaceHighstValuesForOneImageByCertainValues(xData, saliencyMap, numberReplacedValues, valuesUsedForTheReplacement):
    """
    Replaces the numberReplacedValues highest values from the input image with the values form
    valuesUsedForTheReplacement

    Args:
        inputData: The input image
        numberReplacedValues: The number values which should be replaced
        valuesUsedForTheReplacement: The pixels which should be used for the replacement

    Returns:
        resultData: The transformed data
    """

    #The data after applying the transformation
    resultData = np.copy(xData)

    flattened = saliencyMap.flatten()
    sortedArray = np.sort(flattened)[::-1]
    smallestValueWhichShouldBeReplaced = sortedArray[numberReplacedValues-1]
    counter = 0

    for i in range(xData.shape[0]) :
        for j in range(xData.shape[1]):
            if saliencyMap[i][j] >= smallestValueWhichShouldBeReplaced and counter < numberReplacedValues:
                resultData[i][j][0] = valuesUsedForTheReplacement[counter]
                counter += 1
    #print("counter: " + str(counter))
    return resultData


def createBitMaskForSaliencyMap(xData, saliencyMap, numberReplacedValues):
    """
    TODO
    """

    #The data after applying the transformation
    resultData = np.full((xData.shape), 0)

    flattened = saliencyMap.flatten()
    sortedArray = np.sort(flattened)[::-1]
    smallestValueWhichShouldBeReplaced = sortedArray[numberReplacedValues-1]
    counter = 0

    for i in range(xData.shape[0]) :
        for j in range(xData.shape[1]):
            if saliencyMap[i][j] >= smallestValueWhichShouldBeReplaced and counter < numberReplacedValues:
                resultData[i][j][0] = 1
                counter += 1
    #print("counter: " + str(counter))
    return resultData


def rotateBitMask(bitMask, degree):
    return np.rot90(bitMask, degree)


def flipHorizontal(bitMask):
    return np.fliplr(bitMask)


def flipVertical(bitMask):
    return np.flipud(bitMask)


def applyBitMaskToOneImage(bitMask, xData, valuesUsedForTheReplacement):
    resultData = np.copy(xData)
    counter = 0

    for i in range(bitMask.shape[0]):
        for j in range(bitMask.shape[1]):
            # We only change the value where the bitmask has ones
            if bitMask[i][j] == 1:
                resultData[i][j] = valuesUsedForTheReplacement[counter]

    return resultData


def applyBitMaskTransformationUsingCAMforOneImage(xData, saliencyMap, bitMapTransformationFunc, numberReplacedValues,
                                                  valuesUsedForTheReplacement):
    bitMask = createBitMaskForSaliencyMap(xData, saliencyMap, numberReplacedValues)
    transformedBitMask = bitMapTransformationFunc(bitMask)
    resultData = applyBitMaskToOneImage(transformedBitMask, xData, valuesUsedForTheReplacement)
    return resultData


def applyBitMaskTransformationUsingCAMOnWholeDataSet(xData, saliencyMaps, bitMapTransformationFunc, numberReplacedValues,
                                                  valuesUsedForTheReplacement):
    resultData = np.empty(xData.shape)

    for i in range(xData.shape[0]):
        resultData[i] = applyBitMaskTransformationUsingCAMforOneImage(xData[i], saliencyMaps[i], bitMapTransformationFunc,
                                                                      numberReplacedValues, valuesUsedForTheReplacement)
    return resultData
