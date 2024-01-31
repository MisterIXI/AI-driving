import vgamepad as vg
import tensorflow as tf
import numpy as np
import os
import pyautogui as gui
import cv2
import keyboard as kb
from helper_classes import auto_gui_helper as ag
from helper_classes import gamepad_helper as gh
import time
from time import sleep
import enum as en
import typing as tp
import pickle as pkl
import h5py as h5
from collections import deque


# Debug levels
DEBUG_NONE = 0
DEBUG_BASIC = 1
DEBUG_DETAILED = 2


class DataPoint:
    def __init__(self, image_buffer: np.ndarray) -> None:
        pass


class Model:
    def __init__(self, debug_level: int = 2) -> None:
        """
        Initialize the model.
        output_shape: shape of the output tensor (e.g. (None, 2, 21) for steering and acceleration with 21 bins each)
        debug_level: 0 = no prints
        """
        # set file path one dir up from the current file in the rl_data folder
        self.FILE_PATH = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), "rl_data")
        self.debug_level = debug_level
        self.MEMORY_SIZE = 4
        self.framebuffer = deque(maxlen=self.MEMORY_SIZE)
        self.data_states = deque()
        self.data_rewards = deque()
        self.LEARNING_RATE = 0.002
        self.DISCOUNT_FACTOR = 0.8
        self.EPSILON_MAX = 1.0
        self.EPSILON_MIN = 0.02
        self.EPSILON_DECAY = 0.9
        self.epsilon = self.EPSILON_MAX
        self.BATCH_SIZE = 32
        self.UPDATE_TARGET_EVERY = 10
        self.IMG_WIDTH = 1920//4
        self.IMG_HEIGHT = 1080//4
        self.IMG_CHANNELS = 3
        self.update_counter = 0

    def reset_memory(self) -> None:
        """
        Resets the image buffer
        """
        self.framebuffer.clear()
        self.data.clear()

    def save_running_model(self, name: str = "rl_model") -> None:
        """
        Saves the running model to the given name
        """
        self.running_model.save(os.path.join(self.FILE_PATH, name))
        # save the output shape and layer names to a pickle file
        with open(os.path.join(self.FILE_PATH, f"model_params.pkl"), "wb") as f:
            pkl.dump((self.output_shape, self.layer_names, self.epsilon), f)
        if self.debug_level >= DEBUG_BASIC:
            print(f"Saved model to {os.path.join(self.FILE_PATH, name)}")
    def load_old_data(self) -> None:
        """
        Loads the old data from the rl_data folder
        """
        # count the number of existing datasets
        h5_count = len([name for name in os.listdir(
            self.FILE_PATH) if name.endswith(".h5")])
        # load the datasets
        for i in range(h5_count):
            with h5.File(os.path.join(self.FILE_PATH, f"{i}.h5"), "r") as f:
                self.data_states.append(f["dataset"])
                self.data_rewards.append(f["rewards"])
        if self.debug_level >= DEBUG_BASIC:
            print(f"Loaded {h5_count} datasets from {self.FILE_PATH}")
    def load_model(self, name: str = "rl_model") -> None:
        """
        Loads the model from the given name
        """
        # check if the model exists
        if not os.path.exists(os.path.join(self.FILE_PATH, name)):
            if self.debug_level >= DEBUG_BASIC:
                print(
                    f"Model {os.path.join(self.FILE_PATH, name)} does not exist")
            return False
        self.running_model = tf.keras.models.load_model(
            os.path.join(self.FILE_PATH, name))
        self.target_model = tf.keras.models.load_model(
            os.path.join(self.FILE_PATH, name))
        # load the output shape and layer names
        with open(os.path.join(self.FILE_PATH, f"model_params.pkl"), "rb") as f:
            self.output_shape, self.layer_names, self.epsilon = pkl.load(f)
        if self.debug_level >= DEBUG_BASIC:
            print(f"Loaded model from {os.path.join(self.FILE_PATH, name)}")
        return True

    def create_model(self, output_shape: tp.Tuple, layer_names: tp.Tuple) -> None:
        if self.debug_level >= DEBUG_BASIC:
            print("Initializing model from scratch...")
        self.output_shape = output_shape
        self.layer_names = layer_names
        # define model
        # input layer:
        # stacked images of the last MEMORY_SIZE frames, each resized to IMG_WIDTH x IMG_HEIGHT (retaining RGB channels)
        inputs = tf.keras.Input(shape=(
            self.MEMORY_SIZE, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS), name="inputs")
        # output layer:
        # 2 one-hot encoded values for the gamepad inputs, steering and acceleration
        # both are in the range [-1, 1]
        # the one-hot vectors have BIN_COUNT entries, each representing a bin of the range (e.g. with 20 bins, the first bin represents [-1, -0.9], the second [-0.9, -0.8], etc., the middle bin representing empty input)

        # first layer: conv2d with 32 filters, 5x5 kernel, batch normalization, relu activation
        x = tf.keras.layers.Conv2D(
            32, 5, activation='relu', kernel_initializer='HeNormal')(inputs)
        # 1x1 convolution for better memory understanding
        x = tf.keras.layers.Conv2D(
            32, 1, activation='relu', kernel_initializer='HeNormal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(3))(x)

        # second layer: conv2d with 64 filters, 3x3 kernel, batch normalization, relu activation
        x = tf.keras.layers.Conv2D(
            64, 3, activation='relu', kernel_initializer='HeNormal')(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(3))(x)

        # x = tf.keras.layers.BatchNormalization()(x)
        # pooling layer: max pooling with 3x3 kernel
        # x = tf.keras.layers.MaxPooling2D(3)(x)
        # third layer: conv2d with 128 filters, 3x3 kernel, batch normalization, relu activation
        x = tf.keras.layers.Conv2D(
            128, 3, activation='relu', kernel_initializer='HeNormal')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # pooling layer: max pooling with 3x3 kernel
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(3))(x)
        # x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            32, activation='relu', kernel_initializer='HeNormal')(x)
        x = tf.keras.layers.Dense(
            64, activation='relu', kernel_initializer='HeNormal')(x)
        # Output layers: 2 dense layers with BIN_COUNT neurons each, linear activation
        # these layers represent the Q-values for the steering and acceleration input bins

        if len(output_shape) == 2:
            # no bins, only normal outputs
            output = tf.keras.layers.Dense(
                output_shape[1], activation='linear', name="q_value_output")(x)
        else:
            # bins
            outputs = []
            for i in range(output_shape[1]):
                layername = f"output_{i}"
                if len(layer_names) > i:
                    layername = layer_names[i]
                outputs.append(tf.keras.layers.Dense(
                    output_shape[2], activation='linear', name=layername)(x))
            output = tf.keras.layers.concatenate(
                outputs, name="output")
        # define the model
        self.running_model = tf.keras.Model(
            inputs=inputs, outputs=output)
        # define the target model
        self.target_model = tf.keras.Model(
            inputs=inputs, outputs=output)
        # compile the model
        self.running_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.LEARNING_RATE),
            loss="mse"
        )
        # initialize the target model with the running model
        self.target_model.set_weights(self.running_model.get_weights())
        self.target_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.LEARNING_RATE),
            loss="mse"
        )
        if self.debug_level >= DEBUG_BASIC:
            print("Model initialized")
        if self.debug_level >= DEBUG_DETAILED:
            print(self.running_model.summary())

    def predict(self, image_buffer: np.ndarray) -> np.ndarray:
        """
        Predicts the Q-values for the given image buffer
        """
        return self.running_model.predict(image_buffer)

    def predict_bin_indices(self, image_buffer: np.ndarray, suppress_exploration: bool = False) -> tp.Tuple[list, list]:
        """
        Predicts the bin indices for the given image buffer
        returns a tuple (steering, acceleration)
        """
        # if self.debug_level >= DEBUG_DETAILED:
        #     print(f"image_buffer shape: {image_buffer.shape}")
        # if no bins present, raise an error
        if len(self.output_shape) == 2:
            raise Exception("Model has no bins")
        if not suppress_exploration and np.random.rand() <= self.epsilon:
            # perform exploration
            if self.debug_level >= DEBUG_DETAILED:
                print("Performing exploration")
            q_values = np.random.rand(1,
                                      self.output_shape[1] * self.output_shape[2])
            bins = []
            for i in range(self.output_shape[1]):
                bins.append(np.random.randint(self.output_shape[2]))

            return bins,  q_values

        input = np.asarray(image_buffer)
        # rehsape input from (MEM, 1, W, H, C) to (1, MEM, W, H, C)
        input = np.reshape(input, (1, self.MEMORY_SIZE,
                                   self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS))
        q_values = self.predict(np.asarray(input))
        bins = []
        # split the q_values into bins
        for i in range(self.output_shape[1]):
            start = i * self.output_shape[2]
            end = (i+1) * self.output_shape[2]
            bins.append(q_values[0][start:end])
        # get the indices of the max values
        indices = []
        for bin in bins:
            indices.append(np.argmax(bin))
        if self.debug_level >= DEBUG_DETAILED:
            print("Prediction: ", indices)
            print("Q-Values: ", q_values)
        return indices, q_values

    def _preprocess_image(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Preprocesses the given screenshot
        """
        screenshot = np.array(screenshot)
        if screenshot.shape[2] != self.IMG_CHANNELS:
            raise Exception("Wrong number of channels")
        # resize the image
        image = cv2.resize(screenshot, (self.IMG_WIDTH, self.IMG_HEIGHT))
        # reshape the image to the correct shape
        image = np.reshape(
            image, (1, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS))
        return image

    def step(self, screenhsot: np.ndarray) -> np.ndarray:
        """
        Perform one step of the model
        screenshot: the current screenshot (full rgb resolution image WxHxChannel)
        """
        image = self._preprocess_image(screenhsot)
        self.framebuffer.append(image)
        # check if the framebuffer is filled
        if len(self.framebuffer) < self.MEMORY_SIZE:
            # not filled, return empty outputs
            if len(self.output_shape) == 2:
                return 0
            else:
                return np.zeros(self.output_shape[1])

        # predict the bin indices
        indices, q_values = self.predict_bin_indices(
            np.array(self.framebuffer), suppress_exploration=False)
        # safe current framebuffer
        self.data_states.append(np.asarray(self.framebuffer))
        self.data_rewards.append(q_values)
        # if self.debug_level >= DEBUG_DETAILED:
        #     print(f"Step: indices: {indices}, q_values: {q_values}")
        return indices

    def _save_dataset(self, dataset: np.ndarray, rewards: np.ndarray) -> None:
        """
        Save the given dataset with the given rewards
        """
        # count the number of existing datasets
        h5_count = len([name for name in os.listdir(
            self.FILE_PATH) if name.endswith(".h5")])
        # save the dataset
        with h5.File(os.path.join(self.FILE_PATH, f"{h5_count}.h5"), "w") as f:
            f.create_dataset("dataset", data=dataset)
            f.create_dataset("rewards", data=rewards)
        if self.debug_level >= DEBUG_BASIC:
            print(
                f"Saved dataset with {len(dataset)} data points to {os.path.join(self.FILE_PATH, f'{h5_count}.h5')}")

    def _pull_batch_from_dataset(self, dataset: np.ndarray, rewards: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Pull a batch from the given dataset and rewards
        """
        # get the indices of the batch
        indices = np.random.choice(
            np.arange(len(dataset)), size=self.BATCH_SIZE, replace=True)
        # get the batch
        batch = dataset[indices]
        batch_rewards = rewards[indices]
        return batch, batch_rewards

    def train(self, epochs: int, save_model: bool) -> None:
        """
        Stop the current game and train the model. This will clear the current data and asign rewards to the data points according to the given has_won parameter.
        """
        # create np array from the deque
        states = np.asarray(self.data_states)
        rewards = np.asarray(self.data_rewards)
        # remove the extra dimension (n, 1, MEM, W, H, C) -> (n, MEM, W, H, C)

        states = np.squeeze(states, axis=2)

        for i in range(epochs):
            data_batch, reward_batch = self._pull_batch_from_dataset(
                states, rewards)
            # train the model
            if self.debug_level >= DEBUG_BASIC:
                print("Training model...")
            self.target_model.fit(data_batch, reward_batch, epochs=1, verbose=0)
            self.update_counter += 1
            # update the target model
            if self.update_counter % self.UPDATE_TARGET_EVERY == 0:
                self.running_model.set_weights(self.target_model.get_weights())
                if self.debug_level >= DEBUG_BASIC:
                    print("Updated target model")
            # decay epsilon
            old_epsilon = self.epsilon
        self.epsilon = max(
            self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
        if save_model:
            # save the dataset
            self._save_dataset(states, rewards)
            self.save_running_model()
        if self.debug_level >= DEBUG_DETAILED:
            print(f"Trained model for {epochs} epochs")
            print(f"Epsilon: {old_epsilon} -> {self.epsilon}")
