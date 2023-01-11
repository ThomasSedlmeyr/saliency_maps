import glob
import os
import re
import shutil
from pathlib import Path

import inflection


def to_snake_case(name):
    #name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    name = name.lower()
    #name = inflection.underscore(name)
    name = name.replace('true', 'True')
    name = name.replace('false', 'False')
    name = name.replace('sequential', 'Sequential')
    name = name.replace('random_rotation', 'RandomRotation')
    name = name.replace('random_translation', 'RandomTranslation')
    name = name.replace('random_flip', 'RandomFlip')
    name = name.replace('random_contrast', 'RandomContrast')

    name = name.replace('global_average_pooling2_d', 'GlobalAveragePooling2D')
    name = name.replace('batch_normalization', 'BatchNormalization')
    name = name.replace('dropout', 'Dropout')
    name = name.replace('leaky_re_lu', 'LeakyReLU')
    name = name.replace('adam', 'Adam')
    name = name.replace('efficient_net_b0', 'EfficientNetB0')
    name = name.replace('dense', 'Dense')
    name = name.replace('conv2_d', 'Conv2D')
    name = name.replace('max_pooling2_d', 'Flatten')

    return name


def copy_camel_case_file_to_snake_case(path_to_cc, destination, content_should_be_transformed=True):
    file_name = Path(path_to_cc).name
    converted_name = to_snake_case(file_name)
    destination += converted_name

    if content_should_be_transformed:
        with open(path_to_cc) as f:
            content_of_cc_file = f.readlines()

        content_of_cc_file = "".join(content_of_cc_file)


        converted_code_sc = to_snake_case(content_of_cc_file)


        with open(destination, "a") as f2:
            f2.write(converted_code_sc)
    else:
        shutil.copyfile(path_to_cc, destination)


def transform_complete_module(path_to_module, destination):
    module_name = Path(path_to_module).name
    converted_name = to_snake_case(module_name)
    destination += converted_name + "/"
    if not os.path.exists(destination):
        os.mkdir(destination)

    files_to_transform = glob.glob(path_to_module + "*.py")
    for path in files_to_transform:
        copy_camel_case_file_to_snake_case(path, destination)

    files_to_transform = glob.glob(path_to_module + "*.png")
    for path in files_to_transform:
        copy_camel_case_file_to_snake_case(path, destination, content_should_be_transformed=False)

destination_folder = "/home/thomas/Downloads/Test2/"
transform_complete_module("/code/", destination_folder)




