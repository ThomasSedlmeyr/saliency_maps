import itertools
import random
from random import shuffle
import cv2
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from PIL import Image
import random
from itertools import product

from saliency_metrics import *

"""
-1 contains routines which could be used for generating and visualisation of gradient based class activation maps. 

"""


# code adapted from:
# https://colab.research.google.com/drive/1rxm_xus_nr_gehxl_qk_by38_ajw_dxwm_ln9_s?usp=sharing#scroll_to=2orh_pmn2_wxbq

def generate_grad_cam_image(model, img_array, layer_name, eps=1e-8, method="grad_cam", use_interpolation=False,
                            index_output_class=0):
    """
    generates a generate_grad_cam_image- or hi_res_grad_cam-image of given cnn model and a given layer

    args:
        use_interpolation: indicates if an interpolation method should be used when the_cam is upsampled
        model: the keras model which should be analyzed
        img_array: the numpy image which should be analyzed
        layer_name: the name of the layer for which the grad_cam should be computed. usually the last conv-layer is used
        for this process
        eps: this parameter avoids the division with 0
        method: indicates which methode should be used
        use_interpolation: the values after applying the grad-cam algorithm have the resolution of the conv-layer which
        was used for this algorithm. to get the same resolution as the input the result has to be scaled up. this para-
        meter specifies if for this process interpolation should be used
        index_output_class: classifies for which output neuron (output class) the cam should be generated. for regression
        models this value should be set to 0.

    returns:
        the resulting grad_cam image
    """

    # we need to add an additional dimension to get a batch
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output,
                 model.output])

    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(img_array, tf.float32)
        (conv_outputs, predictions) = grad_model(inputs)
        loss = predictions[:, index_output_class]

    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, conv_outputs)
    grads = grads[0]
    conv_outputs = conv_outputs[0]

    if method == "hi_res_cam":
        multiplied_gradients_and_features = tf.multiply(grads, conv_outputs)
        cam = tf.reduce_sum(multiplied_gradients_and_features, axis=(-1))

    # calculates the cam using an adapted version of hi_res-cam
    # applying the re_lu-function on the gradients before the multiplication
    # with the activation map
    elif method == "layer_cam":
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = grads * cast_grads
        multiplied_gradients_and_features = tf.multiply(guided_grads, conv_outputs)
        cam = tf.reduce_sum(multiplied_gradients_and_features, axis=(-1))

    # calculates the cam using the grad-cam algorithm
    elif method == "grad_cam":
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        #Check that we get the same results as in the keras code
        #pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        #last_conv_layer_output = conv_outputs
        #heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        #heatmap = tf.squeeze(heatmap)
        #arr1 = cam.numpy().round(3)
        #arr2 = heatmap.numpy().round(3)
        #if np.array_equal(arr1, arr2):
        #    print("cams are equal")

    # like the grad cam algorithm but using only the positive gradients
    elif method == "grad_cam_pos_grads":
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = grads * cast_grads
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

    else:
        print("method \"" + method + "\" is not suppoerted")

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])

    if use_interpolation:
        heatmap = cv2.resize(cam.numpy(), (w, h))
    # use no interpolation when upsampling
    else:
        heatmap = Image.fromarray(cam.numpy())
        heatmap = heatmap.resize(size=(w, h), resample=Image.nearest)
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
    superimposes a grad-cam heatmap onto an image for model interpretation and visualization.

    args:
      image: (img_width x img_height x 3) numpy array
      grad-cam heatmap: (img_width x img_width) numpy array
      threshold: float
      emphasize: boolean

    returns
      uint8 numpy array with shape (img_height, img_width, 3)

    '''
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLOR_BGR2RGB)

    hif = 0.8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img_rgb


def convert_trainings_data_to_rgbimage(x_data, reshape=True):
    if reshape:
        x_data = x_data.reshape(x_data.shape[:-1])

    rgb_image = np.stack((x_data,) * 3, axis=-1)
    # plt.imshow(stacked_img)
    # plt.show()
    # img = image.fromarray(np.uint8(x_data), 'l')
    numer = rgb_image - np.min(rgb_image)
    denom = (rgb_image.max() - rgb_image.min()) + 0.0000000001
    rgb_image = numer / denom
    rgb_image = np.uint8(255 * rgb_image)

    return rgb_image


def convert_traing_data_to_rgbimage2(img_array):
    converted_image = np.minimum(img_array, 255.0).astype(np.uint8)  # scale 0 to 255
    converted_image_rgb = cv2.cvt_color(converted_image, cv2.COLOR_BGR2RGB)
    return converted_image_rgb


def generate_cams_with_different_methods(model, name_of_last_hidden_layer, image_data_input, output_name="cam",
                                         index_output_class=0):
    original_image = convert_trainings_data_to_rgbimage(image_data_input)

    grad_cam = generate_grad_cam_image(model, np.expand_dims(image_data_input, axis=0), name_of_last_hidden_layer,
                                       method="hi_res_cam", use_interpolation=True, index_output_class=0)
    grad_cam_superimposed = superimpose(original_image, grad_cam, 0.3, emphasize=False)
    plot_camimage(grad_cam_superimposed, original_image, output_name + "_hi_res_cam_interpolation")

    grad_cam = generate_grad_cam_image(model, np.expand_dims(image_data_input, axis=0), name_of_last_hidden_layer,
                                       method="layer_cam", use_interpolation=True, index_output_class=0)
    grad_cam_superimposed = superimpose(original_image, grad_cam, 0.3, emphasize=False)
    plot_camimage(grad_cam_superimposed, original_image, output_name + "_layer_cam_interpolation")

    grad_cam = generate_grad_cam_image(model, np.expand_dims(image_data_input, axis=0), name_of_last_hidden_layer,
                                       method="grad_cam", use_interpolation=True, index_output_class=0)
    grad_cam_superimposed = superimpose(original_image, grad_cam, 0.3, emphasize=False)
    plot_camimage(grad_cam_superimposed, original_image, output_name + "_grad_cam_interpolation")

    grad_cam = generate_grad_cam_image(model, np.expand_dims(image_data_input, axis=0), name_of_last_hidden_layer,
                                       method="grad_cam_pos_grads", use_interpolation=True, index_output_class=0)
    grad_cam_superimposed = superimpose(original_image, grad_cam, 0.3, emphasize=False)
    plot_camimage(grad_cam_superimposed, original_image, output_name + "_grad_cam_pos_grads_interpolation")

    # grad_cam = generate_grad_cam_image(model, np.expand_dims(image_data_input, axis=0), name_of_last_hidden_layer, use_hires_grad_cam=True, use_interpolation=False)
    # grad_cam_superimposed = superimpose(original_image, grad_cam, 0.3, emphasize=False)
    # plot_camimage(grad_cam_superimposed, original_image, output_name+ "_hi_res_cam_no_interpolation")


def plot_camimage(grad_cam_superimposed, original_image, output_name):
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title(output_name)
    ax = plt.subplot(1, 2, 2)
    plt.imshow(grad_cam_superimposed)
    plt.axis('off')
    plt.title('conv_1 grad-cam heat-map')
    plt.tight_layout()
    plt.savefig(output_name + ".png")
    # plt.show()


def save_array_as_image(arr, image_name):
    img = Image.fromarray(arr)
    img_rescaled = img.resize(size=(512, 512), resample=Image.nearest)
    img_rescaled.save(image_name + '.png')


def generate_saliency_maps_for_one_class(model, name_of_last_hidden_layer, name_of_method, test_data,
                                         index_output_class=0):
    saliency_maps = []
    counter = 0
    for test_image in test_data:
        saliency_map = generate_grad_cam_image(model, test_image,
                                               name_of_last_hidden_layer,
                                               method=name_of_method,
                                               use_interpolation=True,
                                               index_output_class=0)
        saliency_maps.append(saliency_map)
        counter += 1

    return saliency_maps


def generate_saliency_maps_for_every_class(model, name_of_last_hidden_layer, name_of_method, test_data,
                                           number_classes=1):
    saliency_maps = []
    counter = 0
    for test_image in test_data:
        saliency_maps_per_class = []
        for i in range(number_classes):
            saliency_map = generate_grad_cam_image(model, test_image,
                                                   name_of_last_hidden_layer,
                                                   method=name_of_method,
                                                   use_interpolation=True,
                                                   index_output_class=i)
            saliency_maps_per_class.append(saliency_map)
            counter += 1
        saliency_maps.append(saliency_maps_per_class)
    return saliency_maps


def save_multiple_saliency_maps_for_every_class(cams, labels, input_data):
    i = 0
    for cams_for_one_class in cams:
        j = 0
        for current_cam in cams_for_one_class:
            input_image = input_data[i].numpy()
            input_image = input_image.astype(np.uint8)
            superimposed_img = superimpose(input_image, current_cam, 0.3, emphasize=False)
            plot_camimage(superimposed_img, input_image, "cam_hires" + str(i) + labels[j])
            j += 1
        i += 1


"""
os.environ['tf_force_gpu_allow_growth'] = 'True'

domain_resolution = 32

model_name = "transformed_data"
#path_to_heights_file = "/home/thomas/dokumente/hi_wi/vorticity_prediction/databases/heights_profiles_15k/heights_32_32.csv"
#path_to_max_vorticity_file = "/home/thomas/dokumente/studium/semester_6/bachelorarbeit/daten_bachelorarbeit/roughness_databases/max_vorticity_databases/1_6_0_all.csv"
# x_train, x_test, y_train, y_test = get_trainings_data_heights_database(path_to_max_vorticity_file, path_to_heights_file,
#                                                               domain_resolution)
# np.save("x_test", x_test)
# np.save("y_test", y_test)

x_train, x_test, y_train, y_test, target_scaler = load_numpy_data_set("max_omega_x_vort_std")

model = tf.keras.models.load_model("../models/height_models/32_32_model_std")
saliency_maps = generate_saliency_maps(model, "third_conv", "hi_res_cam", x_test)
number_replaced_pixels = 100
values_used_for_the_replacement = x_test[0].flatten()

most_salient_areas_replaced_by_one_value = replace_highst_values_for_whole_data_set_by_only_one_value(x_test, saliency_maps,
                                                                            number_replaced_pixels,
                                                                            x_test[0][0][0][0])

random_replaced_with_the_same_value = apply_random_deletion_metric_for_whole_data_set_using_one_value(x_test, number_replaced_pixels, x_test[0][0][0][0])

most_salient_areas_replaced_with_certain_area = replace_highst_values_for_whole_data_set_by_certain_values(x_test,
                                                                                            saliency_maps,
                                                                                            number_replaced_pixels,
                                                                                            values_used_for_the_replacement)

random_replaced_with_certain_areas = apply_random_deletion_metric_for_whole_data_set_using_different_values(x_test,
                                                                                              number_replaced_pixels,
                                                                                              values_used_for_the_replacement)

saliency_map_flipped_vertical = apply_bit_mask_transformation_using_camon_whole_data_set(x_test, saliency_maps, flip_vertical, number_replaced_pixels,
                                                  values_used_for_the_replacement)

saliency_map_flipped_horizontal = apply_bit_mask_transformation_using_camon_whole_data_set(x_test, saliency_maps, flip_horizontal, number_replaced_pixels,
                                                  values_used_for_the_replacement)

def func(x):
    return move_domain(number_steps_x=0, number_steps_y=3, domain=x)

saliency_map_moved_x = apply_bit_mask_transformation_using_camon_whole_data_set(x_test, saliency_maps,
                                                                     func,
                                                                     number_replaced_pixels, values_used_for_the_replacement)
def func2(x):
    return move_domain(number_steps_x=3, number_steps_y=0, domain=x)

saliency_map_moved_y = apply_bit_mask_transformation_using_camon_whole_data_set(x_test, saliency_maps,
                                                                     func2,
                                                                     number_replaced_pixels, values_used_for_the_replacement)

print("")
print("---------------------------------------")
print("without deletion: ")
predictions = model.predict(x_test)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("replacement using saliency map with same values: ")
predictions = model.predict(most_salient_areas_replaced_by_one_value)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("random replacement using same value: ")
predictions = model.predict(random_replaced_with_the_same_value)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("replacement using saliency map with different values: ")
predictions = model.predict(most_salient_areas_replaced_with_certain_area)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("random replacement with different values: ")
predictions = model.predict(random_replaced_with_certain_areas)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("saliency mask flipped vertical: ")
predictions = model.predict(saliency_map_flipped_vertical)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("saliency mask flipped horizontal: ")
predictions = model.predict(saliency_map_flipped_horizontal)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("saliency mask moved x: ")
predictions = model.predict(saliency_map_moved_x)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

print("")
print("---------------------------------------")
print("saliency mask moved y: ")
predictions = model.predict(saliency_map_moved_y)
evaluate_predictions(y_test, predictions, target_scaler)
print("---------------------------------------")

path = "../output/"
for i in range(3):
    save_array_as_image(convert_trainings_data_to_rgbimage(most_salient_areas_replaced_by_one_value[i]), path + str(i) + "_replacement_saliency_map_same_value")
    save_array_as_image(convert_trainings_data_to_rgbimage(x_test[i]), path + str(i) + "_original")
    save_array_as_image(convert_trainings_data_to_rgbimage(saliency_maps[i], reshape=False), path + str(i) + "_saliency_map")
    save_array_as_image(convert_trainings_data_to_rgbimage(random_replaced_with_certain_areas[i]), path + str(i) + "_random_replacement_different_values")
    save_array_as_image(convert_trainings_data_to_rgbimage(most_salient_areas_replaced_with_certain_area[i]),
                     path + str(i) + "_replacement_saliency_map_different_value")
    save_array_as_image(convert_trainings_data_to_rgbimage(random_replaced_with_certain_areas[i]),
                     path + str(i) + "_random_replacement_different_values")
    save_array_as_image(convert_trainings_data_to_rgbimage(saliency_map_flipped_vertical[i]),
                     path + str(i) + "_saliency_mask_flipped_vertical")
    save_array_as_image(convert_trainings_data_to_rgbimage(saliency_map_flipped_horizontal[i]),
                     path + str(i) + "_saliency_mask_flipped_horizontal")
    save_array_as_image(convert_trainings_data_to_rgbimage(saliency_map_moved_x[i]),
                     path + str(i) + "_saliency_mask_moved_x")
    save_array_as_image(convert_trainings_data_to_rgbimage(saliency_map_moved_y[i]),
                     path + str(i) + "_saliency_mask_moved_y")

# for i in range(5):
#    save_camimages_only("second_conv", x_test[i], "cam2_no_interpol_" + str(i))
# for i, layer in enumerate(model.layers):
#    layer._name = 'layer_' + str(i)

# for i in range(5):
#    save_camimages_only("second_conv", x_train[i], "cam2_no_interpol_" + str(i))
"""
