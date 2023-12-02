import tensorflow as tf
import numpy as np
import os
import json
import glob
import cv2
import h5py

REDUCTION_FACTOR = 4

# Load the data
script_dir = os.path.dirname(__file__)
learning_data_path = os.path.join(script_dir, 'learning_data')
# run_folders = os.listdir(os.path.join(script_dir, 'learning_data'))
# https://stackoverflow.com/a/142535
run_folders = next(os.walk(os.path.join(learning_data_path, '.')))[1]
run_folders.sort()
# print(run_folders)
run_dicts = []
for run_folder in run_folders:
    # get file named "inputs_*.json"
    json_file = glob.glob(os.path.join(
        learning_data_path, run_folder, 'inputs_*.json'))[0]
    # load the json file
    with open(json_file) as f:
        run_dicts.append(json.load(f))

# count total number of samples
sample_count = 0
for run_dict in run_dicts:
    sample_count += len(run_dict)

total_data = np.zeros(
    (sample_count, 1080 // REDUCTION_FACTOR, 1920//REDUCTION_FACTOR, 3), dtype=np.uint8)
result_data = np.zeros((sample_count, 2))
i = 0
run_num = 0
for run_dict in run_dicts:
    print("Now processing run " + str(run_num) + "...")
    for entry in run_dict:
        # print(entry)
        image = cv2.imread(os.path.join(
            learning_data_path, run_folders[run_num], entry[0]))
        test = cv2.resize(
            image, (1920//REDUCTION_FACTOR, 1080//REDUCTION_FACTOR))
        total_data[i] = test
        result_data[i, 0] = entry[1]["stick_x"]
        result_data[i, 1] = entry[1]["RT"] - entry[1]["LT"]
        i += 1
    run_num += 1
    # print("shape" + str(total_data[i-1].shape))

# print("total_data shape: " + str(total_data.shape))
# zip data for shuffling
zipped = list(zip(total_data, result_data))
# shuffle the data
np.random.shuffle(zipped)

# unpack the data
total_data, result_data = zip(*zipped)

# split the data into training and validation sets
training_data = total_data[:int(0.8*sample_count)]
validation_data = total_data[int(0.8*sample_count):]

training_result = result_data[:int(0.8*sample_count)]
validation_result = result_data[int(0.8*sample_count):]


# # put the data into tf.data.Dataset
# training_dataset = tf.data.Dataset.from_tensor_slices(
#     (training_data, training_result))

# validation_dataset = tf.data.Dataset.from_tensor_slices(
#     (validation_data, validation_result))

# save the datasets into h5 files
with h5py.File(os.path.join(script_dir, 'learning_data', 'training_data.h5'), 'w') as f:
    f.create_dataset('training_data', data=training_data)
    f.create_dataset('training_result', data=training_result)

with h5py.File(os.path.join(script_dir, 'learning_data', 'validation_data.h5'), 'w') as f:
    f.create_dataset('validation_data', data=validation_data)
    f.create_dataset('validation_result', data=validation_result)
