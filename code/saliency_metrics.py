import itertools
from itertools import product
import random
import numpy as np



"""
this file contains routines for the evaluation of certain cams (class activation maps)
"""


def apply_random_deletion_metric_quadratic_area_on_whole_data_set(x_data, size_x, size_y, value_used_for_the_deletion):
    result_data = np.empty(x_data.shape)
    width = x_data.shape[1]

    for i in range(x_data.shape[0]):
        random_pos_x = random.randint(0, width - size_x - 1)
        random_pox_y = random.randint(0, width - size_y - 1)
        for x in range(size_x):
            for y in range(size_y):
                result_data[i][random_pos_x + x][random_pox_y + y] = value_used_for_the_deletion

    return result_data


def apply_random_deletion_metric_for_whole_data_set_using_one_value(x_data, number_elements_which_should_be_replaced,
                                                          value_used_for_the_replacement):
    result_data = np.empty(x_data.shape)

    for i in range(x_data.shape[0]):
        result_data[i] = apply_random_deletion_metric_for_one_image_using_one_value(x_data[i], number_elements_which_should_be_replaced,
                                                                     value_used_for_the_replacement)
    return result_data


def apply_random_deletion_metric_for_whole_data_set_using_different_values(x_data, number_values, values_used_for_the_replacement):
    result_data = np.empty(x_data.shape)

    for i in range(x_data.shape[0]):
        result_data[i] = apply_random_deletion_metric_for_one_image_using_different_values(x_data[i], number_values, values_used_for_the_replacement)

    return result_data


def apply_random_deletion_metric_for_one_image_using_one_value(x_data, number_replaced_values, value_used_for_the_replacement):
    values_used_for_the_replacement = [value_used_for_the_replacement] * number_replaced_values
    return apply_random_deletion_metric_for_one_image_using_different_values(x_data, number_replaced_values, values_used_for_the_replacement)


def apply_random_deletion_metric_for_one_image_using_different_values(x_data, number_values, values_used_for_the_replacement):
    """
    replaces the number_replaced_values pixels at random positions by values_used_for_the_replacement

    args:
        input_data: the input image
        number_replaced_values: the number values which should be replaced
        values_used_for_the_replacement: the pixels which should be used for the replacement

    returns:
        result_data: the transformed data
    """

    #for each area the salient area with the same value

    result_data = np.copy(x_data)
    width = x_data.shape[1]

    random_tuples = random.sample(list(product(range(width), repeat=2)), k=number_values)
    counter = 0
    for (x, y) in random_tuples:
        result_data[x][y][0] = values_used_for_the_replacement[counter]
        counter += 1

    return result_data

def replace_highst_values_for_whole_data_set_by_certain_values(x_data, saliency_maps, number_replaced_values, values_used_for_the_replacement):
    result_data = np.empty(x_data.shape)

    for i in range(x_data.shape[0]):
        result_data[i] = replace_highst_values_for_one_image_by_certain_values(x_data[i], saliency_maps[i], number_replaced_values,
                                                                      values_used_for_the_replacement)
    return result_data


def replace_highst_values_for_whole_data_set_by_only_one_value(x_data, saliency_maps, number_elements_which_should_be_replaced,
                                                          value_used_for_the_replacement):
    result_data = np.empty(x_data.shape)

    for i in range(x_data.shape[0]):
        result_data[i] = replace_highst_values_for_one_image_by_only_one_value(x_data[i], saliency_maps[i],
                                                                     number_elements_which_should_be_replaced,
                                                                     value_used_for_the_replacement)
    return result_data


def replace_highst_values_for_one_image_by_only_one_value(x_data, saliency_map, number_replaced_values, value_used_for_the_replacement):
    values_used_for_the_replacement = [value_used_for_the_replacement] * number_replaced_values
    return replace_highst_values_for_one_image_by_certain_values(x_data, saliency_map, number_replaced_values, values_used_for_the_replacement)


def replace_highst_values_for_one_image_by_certain_values(x_data, saliency_map, number_replaced_values, values_used_for_the_replacement):
    """
    replaces the number_replaced_values highest values from the input image with the values form
    values_used_for_the_replacement

    args:
        input_data: the input image
        number_replaced_values: the number values which should be replaced
        values_used_for_the_replacement: the pixels which should be used for the replacement

    returns:
        result_data: the transformed data
    """

    #the data after applying the transformation
    result_data = np.copy(x_data)

    flattened = saliency_map.flatten()
    sorted_array = np.sort(flattened)[::-1]
    smallest_value_which_should_be_replaced = sorted_array[number_replaced_values-1]
    counter = 0

    for i in range(x_data.shape[0]) :
        for j in range(x_data.shape[1]):
            if saliency_map[i][j] >= smallest_value_which_should_be_replaced and counter < number_replaced_values:
                result_data[i][j][0] = values_used_for_the_replacement[counter]
                counter += 1
    #print("counter: " + str(counter))
    return result_data


def create_bit_mask_for_saliency_map(x_data, saliency_map, number_replaced_values):
    """
    todo
    """

    #the data after applying the transformation
    result_data = np.full((x_data.shape), 0)

    flattened = saliency_map.flatten()
    sorted_array = np.sort(flattened)[::-1]
    smallest_value_which_should_be_replaced = sorted_array[number_replaced_values-1]
    counter = 0

    for i in range(x_data.shape[0]) :
        for j in range(x_data.shape[1]):
            if saliency_map[i][j] >= smallest_value_which_should_be_replaced and counter < number_replaced_values:
                result_data[i][j][0] = 1
                counter += 1
    #print("counter: " + str(counter))
    return result_data


def rotate_bit_mask(bit_mask, degree):
    return np.rot90(bit_mask, degree)


def flip_horizontal(bit_mask):
    return np.fliplr(bit_mask)


def flip_vertical(bit_mask):
    return np.flipud(bit_mask)


def apply_bit_mask_to_one_image(bit_mask, x_data, values_used_for_the_replacement):
    result_data = np.copy(x_data)
    counter = 0

    for i in range(bit_mask.shape[0]):
        for j in range(bit_mask.shape[1]):
            # we only change the value where the bitmask has ones
            if bit_mask[i][j] == 1:
                result_data[i][j] = values_used_for_the_replacement[counter]

    return result_data


def apply_bit_mask_transformation_using_camfor_one_image(x_data, saliency_map, bit_map_transformation_func, number_replaced_values,
                                                  values_used_for_the_replacement):
    bit_mask = create_bit_mask_for_saliency_map(x_data, saliency_map, number_replaced_values)
    transformed_bit_mask = bit_map_transformation_func(bit_mask)
    result_data = apply_bit_mask_to_one_image(transformed_bit_mask, x_data, values_used_for_the_replacement)
    return result_data


def apply_bit_mask_transformation_using_camon_whole_data_set(x_data, saliency_maps, bit_map_transformation_func, number_replaced_values,
                                                  values_used_for_the_replacement):
    result_data = np.empty(x_data.shape)

    for i in range(x_data.shape[0]):
        result_data[i] = apply_bit_mask_transformation_using_camfor_one_image(x_data[i], saliency_maps[i], bit_map_transformation_func,
                                                                      number_replaced_values, values_used_for_the_replacement)
    return result_data
