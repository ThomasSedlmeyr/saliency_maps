import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import os
from tensorflow.keras import models
from PIL import Image


from gradient_based_cams import generate_grad_cam_image, superimpose, generate_saliency_maps_for_every_class, \
    save_multiple_saliency_maps_for_every_class

img_size = 224

os.environ['tf_force_gpu_allow_growth'] = 'True'
#gpus = tf.config.experimental.list_physical_devices('gpu')
#for gpu in gpus:
    #print("der")
#    tf.config.experimental.set_memory_growth(gpu, True)

#tf.config.experimental.virtual_device_configuration(memory_limit=3000)
#model = EfficientNetB0(weights='imagenet', drop_connect_rate=0.4, include_top=False,)

img_augmentation = Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def build_model(num_classes=2):
    inputs = tf.keras.layers.input(shape=(img_size, img_size, 3))
    #x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # freeze the pretrained weights
    model.trainable = False

    # rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_Dropout_rate = 0.2
    x = layers.Dropout(top_Dropout_rate, name="top_Dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # compile
    model = tf.keras.model(inputs, outputs, name="efficient_net")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"],
    )
    
    model.summary()
    return model



def create_own_model(lr_rate=0.00025, num_classes=2):
    # norm_layer = normalization()
    # norm_layer.adapt(x_train)
    model = models.Sequential()
    # model.add(norm_layer)
    model.add(layers.Conv2D(64, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(img_size, img_size, 3), padding="same",name="first_conv"))
    model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.3))
    model.add(layers.Flatten((2, 2)))

    model.add(layers.Conv2D(64, (4, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(img_size, img_size, 3), padding="same",name="first_conv2"))
    model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.3))
    model.add(layers.Flatten((2, 2)))

    # model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(128, (4, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", name="second_conv3"))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", name="third_conv"))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.flatten())

    model.add(layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="softmax", name="pred"))

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate), loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def unfreeze_model(model):
    # we unfreeze the top 20 layers while leaving batch_norm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def create_numpy_ds():
    from keras.utils import load_img
    from keras.utils import img_to_array


    # define location of dataset
    folder = '/home/thomas/downloads/dogs-vs-cats/train/'
    photos, labels = list(), list()
    # enumerate files in the directory
    counter = 0
    for file in listdir(folder):
        # determine class
        output = [1.0, 0.0]
        if file.startswith('dog'):
            output = [0.0, 1.0]
        # load image
        photo = load_img(folder + file, target_size=(img_size, img_size))
        # convert to numpy array
        photo = img_to_array(photo)
        # store
        photos.append(photo)
        labels.append(output)
        if counter == 10000:
            break
        counter += 1

    # convert to a numpy arrays
    photos = np.asarray(photos)
    labels = np.asarray(labels)
    print(photos.shape, labels.shape)
    # save the reshaped photos
    np.save('../data_set/dogs_vs_cats_photos.npy', photos)
    np.save('../data_set/dogs_vs_cats_labels.npy', labels)


def create_train_test_numpy():
    x = np.load("../data_set/dogs_vs_cats_photos.npy")
    y = np.load("../data_set/dogs_vs_cats_labels.npy")

    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

    # use 20% for testing
    split_index = int(x.shape[0] * 0.8)
    np.save('../data_set/x_train_small.npy', x[:split_index])
    np.save('../data_set/x_test_small.npy', x[split_index:])
    np.save('../data_set/y_train_small.npy', y[:split_index])
    np.save('../data_set/y_test_small.npy', y[split_index:])


def read_numpy_data_set(name_prefix, name_postfix):
    x_train = np.load(name_prefix + 'x_train' + name_postfix)
    x_test = np.load(name_prefix + 'x_test' + name_postfix)
    y_train = np.load(name_prefix + 'y_train' + name_postfix)
    y_test = np.load(name_prefix + 'y_test' + name_postfix)

    # todo tensoren sind kacke für die bilder

    return tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test), \
        tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)
    #return x_train, x_test, y_train, y_test


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


labels = ["cat", "dog"]

#create_numpy_ds()
#create_train_test_numpy()
#print(tf.version.version)

x_train, x_test, y_train, y_test = read_numpy_data_set("../data_set/", "_small.npy")

#unfreeze_model(model)

#epochs = 10  # @param {type: "slider", min:8, max:50}
#model = build_model(2)
#model = create_own_model()
#hist = model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.1, verbose=1, batch_size=2, use_multiprocessing=True, workers=8)
#model.evaluate(x_test, y_test)
#model.save("saved_models/test")

#name: top_conv

model = models.load_model("../saved_models/test")
model.summary()
#model.evaluate(x_test, y_test)

input_image = x_test[0]
#cam_image = generate_grad_cam_image(model, input_image, "top_conv", eps=1e-8, method="grad_cam", use_interpolation=False,
#                         index_output_class=1)


#input_image = cv2.cvt_color(np.array(input_image), cv2.color_rgb2_bgr)
#plot_camimage(cam_image, input_image, "test")
#superimposed_img = superimpose(input_image, cam_image, 0.3, emphasize=False)
#plot_camimage(superimposed_img, input_image, "test_2")


cams = generate_saliency_maps_for_every_class(model, "top_conv", "hi_res_cam", x_test[:10], number_classes=2)

save_multiple_saliency_maps_for_every_class(cams, labels, x_test[:10])

#for i in range(10):
#    input_image = x_test[i].numpy()
#    input_image = input_image.astype(np.uint8)
#    superimposed_img = superimpose(input_image, cams[i], 0.3, emphasize=True)
#    plot_camimage(superimposed_img, input_image, "cam_" + str(i))

print("test")
#plot_hist(hist)
