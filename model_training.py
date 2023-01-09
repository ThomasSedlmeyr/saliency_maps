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


from GradientBasedCAMs import generateGradCamImage, superimpose, generateSaliencyMapsForEveryClass, \
    saveMultipleSaliencyMapsForEveryClass

IMG_SIZE = 224

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
    #print("Der")
#    tf.config.experimental.set_memory_growth(gpu, True)

#tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)
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
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    #x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"],
    )
    
    model.summary()
    return model



def create_own_model(lr_rate=0.00025, num_classes=2):
    # normLayer = Normalization()
    # normLayer.adapt(xTrain)
    model = models.Sequential()
    # model.add(normLayer)
    model.add(layers.Conv2D(64, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(IMG_SIZE, IMG_SIZE, 3), padding="same",name="firstConv"))
    model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.3))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (4, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(IMG_SIZE, IMG_SIZE, 3), padding="same",name="firstConv2"))
    model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.3))
    model.add(layers.MaxPooling2D((2, 2)))

    # model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(128, (4, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", name="secondConv3"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding="same", name="thirdConv"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

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
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
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

    print("Servus!")
    # define location of dataset
    folder = '/home/thomas/Downloads/dogs-vs-cats/train/'
    photos, labels = list(), list()
    # enumerate files in the directory
    counter = 0
    for file in listdir(folder):
        # determine class
        output = [1.0, 0.0]
        if file.startswith('dog'):
            output = [0.0, 1.0]
        # load image
        photo = load_img(folder + file, target_size=(IMG_SIZE, IMG_SIZE))
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
    np.save('data_set/dogs_vs_cats_photos.npy', photos)
    np.save('data_set/dogs_vs_cats_labels.npy', labels)


def create_train_test_numpy():
    x = np.load("data_set/dogs_vs_cats_photos.npy")
    y = np.load("data_set/dogs_vs_cats_labels.npy")

    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

    # use 20% for testing
    split_index = int(x.shape[0] * 0.8)
    np.save('data_set/x_train_small.npy', x[:split_index])
    np.save('data_set/x_test_small.npy', x[split_index:])
    np.save('data_set/y_train_small.npy', y[:split_index])
    np.save('data_set/y_test_small.npy', y[split_index:])


def read_numpy_dataSet(name_prefix, name_postfix):
    x_train = np.load(name_prefix + 'x_train' + name_postfix)
    x_test = np.load(name_prefix + 'x_test' + name_postfix)
    y_train = np.load(name_prefix + 'y_train' + name_postfix)
    y_test = np.load(name_prefix + 'y_test' + name_postfix)

    # TODO tensoren sind kacke für die bilder

    return tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test), \
        tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)
    #return x_train, x_test, y_train, y_test


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


labels = ["cat", "dog"]

#create_numpy_ds()
#create_train_test_numpy()
#print(tf.version.VERSION)

x_train, x_test, y_train, y_test = read_numpy_dataSet("data_set/", "_small.npy")

#unfreeze_model(model)

#epochs = 10  # @param {type: "slider", min:8, max:50}
#model = build_model(2)
#model = create_own_model()
#hist = model.fit(x=x_train, y=y_train, epochs=1, validation_split=0.1, verbose=1, batch_size=2, use_multiprocessing=True, workers=8)
#model.evaluate(x_test, y_test)
#model.save("saved_models/test")

#name: top_conv

model = models.load_model("saved_models/test")
#model.evaluate(x_test, y_test)

input_image = x_test[0]
#cam_image = generateGradCamImage(model, input_image, "top_conv", eps=1e-8, method="gradCAM", useInterpolation=False,
#                         indexOutputClass=1)


#input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
#plotCAMimage(cam_image, input_image, "test")
#superimposed_img = superimpose(input_image, cam_image, 0.3, emphasize=False)
#plotCAMimage(superimposed_img, input_image, "test_2")


cams = generateSaliencyMapsForEveryClass(model, "top_conv", "gradCAM", x_test[:10], numberClasses=2)

saveMultipleSaliencyMapsForEveryClass(cams, labels, x_test[:10])

#for i in range(10):
#    input_image = x_test[i].numpy()
#    input_image = input_image.astype(np.uint8)
#    superimposed_img = superimpose(input_image, cams[i], 0.3, emphasize=True)
#    plotCAMimage(superimposed_img, input_image, "cam_" + str(i))

print("Test")
#plot_hist(hist)
