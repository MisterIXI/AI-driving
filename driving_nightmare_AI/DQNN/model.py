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
import enum as en


# Debug levels
DEBUG_NONE = 0
DEBUG_BASIC = 1
DEBUG_DETAILED = 2


class Reward(en.Enum):
    NEUTRAL = 0
    WIN = 1
    LOSE = 2
    Intermediate = 3


class Model:
    def __init__(self, debug_level: int = 2, IMG_WIDTH=1920//4, IMG_HEIGHT=1080//4, path_name="rl_data") -> None:
        """
        Initialize the model.
        output_shape: shape of the output tensor (e.g. (None, 2, 21) for steering and acceleration with 21 bins each)
        debug_level: 0 = no prints
        """
        # set file path one dir up from the current file in the rl_data folder
        self.FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), path_name)
        self.debug_level = debug_level
        self.MEMORY_SIZE = 4
        self.framebuffer = deque(maxlen=self.MEMORY_SIZE + 1)
        self.data = deque()
        self.run_data = deque()
        self.last_action = None
        self.LEARNING_RATE = 0.0001
        self.DISCOUNT_FACTOR = 0.99
        self.EPSILON_MAX = 1.0
        self.EPSILON_MIN = 0.02
        self.EPSILON_DECAY = 0.9
        self.epsilon = self.EPSILON_MAX
        self.BATCH_SIZE = 32
        self.UPDATE_TARGET_EVERY = 20
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_CHANNELS = 1
        self.update_counter = 0
        self.NEUTRAL_REWARD = 1
        self.LOSE_REWARD = -10
        self.WIN_REWARD = 10

    def reset_memory(self) -> None:
        """
        Resets the image buffer
        """
        self.framebuffer.clear()

    def save_models(self) -> None:
        self.running_model.save(os.path.join(self.FILE_PATH, "running_model"))
        self.target_model.save(os.path.join(self.FILE_PATH, "target_model"))
        with open(os.path.join(self.FILE_PATH, "model_params.pkl"), "wb") as f:
            pkl.dump((self.actions_shape, self.action_length, self.layer_names, self.epsilon, self.update_counter), f)
        if self.debug_level >= DEBUG_BASIC:
            print(f"Saved model to {self.FILE_PATH}")

    def load_old_data(self) -> None:
        """
        Loads the old data from the rl_data folder
        """
        # count the number of existing datasets
        h5_count = len([name for name in os.listdir(self.FILE_PATH) if name.endswith(".h5")])
        # load the datasets
        for i in range(h5_count):
            with h5.File(os.path.join(self.FILE_PATH, f"{i}.h5"), "r") as f:
                self.data_states.append(f["dataset"])
                self.data_rewards.append(f["rewards"])
        if self.debug_level >= DEBUG_BASIC:
            print(f"Loaded {h5_count} datasets from {self.FILE_PATH}")

    def load_model(self, epsilon_override=None) -> None:
        """
        Loads the model from the given name
        """
        # check if the model exists
        if not os.path.exists(os.path.join(self.FILE_PATH, "running_model")):
            if self.debug_level >= DEBUG_BASIC:
                print(f"Model {os.path.join(self.FILE_PATH, 'running_model')} does not exist")
            return False
        self.running_model = tf.keras.models.load_model(os.path.join(self.FILE_PATH, "running_model"))
        self.target_model = tf.keras.models.load_model(os.path.join(self.FILE_PATH, "target_model"))
        # load the output shape and layer names
        with open(os.path.join(self.FILE_PATH, f"model_params.pkl"), "rb") as f:
            self.actions_shape, self.action_length, self.layer_names, self.epsilon, self.update_counter = pkl.load(f)
        self.create_action_permutations(self.actions_shape)
        if epsilon_override is not None:
            self.epsilon = epsilon_override
            print(f"Overriding epsilon with {epsilon_override}")
        if self.debug_level >= DEBUG_BASIC:
            print(f"Loaded model from {self.FILE_PATH}")
        return True

    def create_model(self, actions_shape: list[list], layer_names: tp.Tuple) -> None:
        if self.debug_level >= DEBUG_BASIC:
            print("Initializing model from scratch...")
        self.actions_shape = actions_shape
        self.action_length = 0
        for action in actions_shape:
            self.action_length += action
        self.layer_names = layer_names
        self.create_action_permutations(self.actions_shape)
        # define model
        # input layers:
        inputA = tf.keras.Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS*self.MEMORY_SIZE), name="input_state")
        inputB = tf.keras.Input(shape=(self.action_length), name="input_actions")
        # convolutional layers
        x = tf.keras.layers.Conv2D(32, 5, activation='relu', kernel_initializer='HeNormal')(inputA)
        x = tf.keras.layers.MaxPooling2D(3)(x)
        x = tf.keras.layers.Conv2D(64, 5, activation='relu', kernel_initializer='HeNormal')(x)
        x = tf.keras.layers.MaxPooling2D(3)(x)
        x = tf.keras.layers.Conv2D(128, 5, activation='relu', kernel_initializer='HeNormal')(x)
        x = tf.keras.layers.MaxPooling2D(5)(x)
        x = tf.keras.layers.Flatten()(x)
        # add player inputs to dense layer
        x = tf.keras.layers.concatenate((x, inputB))
        x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='HeNormal')(x)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='HeNormal')(x)
        # output: the q-value for the given action
        output = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='HeNormal')(x)
        # define the running model
        self.running_model = tf.keras.Model(inputs=[inputA, inputB], outputs=output)
        # define the target model
        self.target_model = tf.keras.Model(inputs=[inputA, inputB], outputs=output)
        # compile the model
        self.running_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), loss="mse")
        # initialize the target model with the running model
        self.target_model.set_weights(self.running_model.get_weights())
        self.target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), loss="mse")
        if self.debug_level >= DEBUG_BASIC:
            print("Model initialized")
        if self.debug_level >= DEBUG_DETAILED:
            print(self.running_model.summary())

    def predict_all_actions(self, image_buffer: np.ndarray, use_target_model: bool = False) -> np.ndarray:
        """
        Predicts the Q-values for the given image buffer in combination with all possible actions
        """
        # duplicate the image buffer to the number of actions
        buffer_arr = np.asarray(image_buffer)
        buffer_arr = np.repeat(buffer_arr, self.action_permutations.shape[0], axis=0)
        if use_target_model:
            result = self.target_model.predict((buffer_arr, self.action_permutations))
        else:
            print(buffer_arr.shape)
            print(self.action_permutations.shape)
            result = self.running_model.predict((buffer_arr, self.action_permutations))
        return result

    def _rec_perms(self, actions, actions_shape, depth, offset):
        total_length = actions.shape[0]
        set_length = 0
        if depth == 0:
            set_length = 1
        else:
            for i in range(depth):
                if i == 0:
                    set_length = actions_shape[i]
                else:
                    set_length *= actions_shape[i]
        set_count = total_length // (set_length*actions_shape[depth])
        n = 0
        # iterate over all sets of actions
        for i in range(set_count):
            # iterate over all actions in the set
            for j in range(actions_shape[depth]):
                for k in range(set_length):
                    actions[n][offset+j] = 1
                    n += 1
        if depth < len(actions_shape) - 1:
            self._rec_perms(actions, actions_shape, depth + 1, offset + actions_shape[depth])

    def create_action_permutations(self, action_shape) -> None:
        """Creates all permutations of the given actions in self.actions_shape"""
        # each action is a onehot encoded vector and every entry in self.actions_shape is the length of the action vector
        # example: self.actions_shape = [3, 2] -> actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0], [0, 1]] -> permutations = [[1,0,0, 1,0], [1,0,0, 0,1], [0,1,0, 1,0], [0,1,0, 0,1], [0,0,1, 1,0], [0,0,1, 0,1]]
        action_count = 0
        action_length = 0
        for i in range(len(action_shape)):
            action_length += action_shape[i]
            if i == 0:
                action_count = action_shape[i]
            else:
                action_count *= action_shape[i]
        self.action_permutations = np.zeros((action_count, action_length))
        self._rec_perms(self.action_permutations, action_shape, 0, 0)

    def choose_next_action(self, image_buffer: np.ndarray, suppress_exploration: bool = False) -> np.ndarray:
        """Predicts the Q-values for the given image buffer and returns either the best action, or a random for exploration"""
        if not suppress_exploration and np.random.rand() <= self.epsilon:
            # perform exploration
            if self.debug_level >= DEBUG_DETAILED:
                print("Performing exploration")
            # choose random array from action_permutations
            return self.action_permutations[np.random.randint(self.action_permutations.shape[0])]
        q_values = self.predict_all_actions(image_buffer)
        # get the index of the max value
        index = np.argmax(q_values)
        # index = np.argmin(q_values)
        if self.debug_level >= DEBUG_DETAILED:
            print("Predicted Action: ", self.action_permutations[index])
        return self.action_permutations[index]

    def _preprocess_image(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Preprocesses the given screenshot
        """
        screenshot = np.array(screenshot)
        # resize the image
        image = cv2.resize(screenshot, (self.IMG_WIDTH, self.IMG_HEIGHT))

        # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # reshape the image to the correct shape
        image = np.reshape(image, (1,  self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        return image

    def step(self, screenshot: np.ndarray, bonus_reward: int) -> np.ndarray:
        """
        Perform one step of the model
        screenshot: the current screenshot (full rgb resolution image WxHxChannel)
        returns: null when the framebuffer is not filled, the bins otherwise
        """
        image = self._preprocess_image(screenshot)
        if len(self.framebuffer) == 0:
            # fill the buffer with MEMORY_SIZE -1 empty black images
            for _ in range(self.MEMORY_SIZE - 1):
                self.framebuffer.append(np.zeros((1, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)))
        self.framebuffer.append(image)
        # check if the framebuffer is filled
        if len(self.framebuffer) < (self.MEMORY_SIZE):
            # not filled, return None
            return None
        elif len(self.framebuffer) == self.MEMORY_SIZE:
            # if buffer not completely filled, only predict this round
            buffer = np.array(self.framebuffer)
            print("buffer" + str(buffer.shape))
            state = np.squeeze(buffer, axis=1)
            print("state" + str(state.shape))
            state = np.transpose(state, (3,1,2,0))
            # state = np.reshape(state, (1, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS*self.MEMORY_SIZE))
            print("state" + str(state.shape))
            self.last_action = self.choose_next_action(state, suppress_exploration=False)
            return self.last_action
        # predict the bin indices
        squished_buffer = np.squeeze(np.array(self.framebuffer), axis=1)
        # old_state is the buffer without the last element
        old_state = squished_buffer[:-1]
        old_state = np.transpose(old_state, (3,1,2,0))
        # new_state is the buffer without the first element
        new_state = squished_buffer[1:]
        new_state = np.transpose(new_state, (3,1,2,0))
        selected_action = self.choose_next_action(new_state, suppress_exploration=False)
        # safe current data
        self.run_data.append((old_state, self.last_action, bonus_reward, Reward.NEUTRAL.value, new_state))
        self.last_action = selected_action
        # if self.debug_level >= DEBUG_DETAILED:
        #     print(f"Step: indices: {indices}, q_values: {q_values}")
        return selected_action

    def _add_intermediate_rewards(self, intermediate: tp.Tuple[int, int] = None) -> None:
        """Add intermediate rewards to the run data"""
        if intermediate is not None:
            for i in range(len(self.run_data)):
                self.run_data[i] = (self.run_data[i][0], self.run_data[i][1], self.run_data[i][2], Reward.Intermediate.value, self.run_data[i][4])
        self.run_data[-1] = (self.run_data[-1][0], self.run_data[-1][1], self.run_data[-1][2], Reward.NEUTRAL.value, self.run_data[-1][4])

    def finish_run(self, has_won: bool, intermediate: tp.Tuple[int, int] = None) -> None:
        """Finish run, and assign states to the run.
        intermediate: (every_x_steps, min_distance_to_end)"""
        if has_won == None:
            self.run_data[-1] = (self.run_data[-1][0], self.run_data[-1][1], self.run_data[-1][2], Reward.NEUTRAL.value, self.run_data[-1][4])
        elif has_won:
            self.run_data[-1] = (self.run_data[-1][0], self.run_data[-1][1], self.run_data[-1][2], Reward.WIN.value, self.run_data[-1][4])
        else:
            self.run_data[-1] = (self.run_data[-1][0], self.run_data[-1][1], self.run_data[-1][2], Reward.LOSE.value, self.run_data[-1][4])
        if intermediate is not None:
            self._add_intermediate_rewards(intermediate)
        self._save_dataset(self.run_data)
        self.run_data.clear()
        self.framebuffer.clear()

    def _save_dataset(self, dataset: deque) -> None:
        """
        Save the given dataset with the given rewards
        """
        # count the number of existing datasets
        h5_count = len([name for name in os.listdir(self.FILE_PATH) if name.endswith(".h5")])
        # save the dataset
        with h5.File(os.path.join(self.FILE_PATH, f"{h5_count}.h5"), "w") as f:
            for i, (img_old, action, reward, done, img_new) in enumerate(dataset):
                grp = f.create_group(f"step_{i}")
                grp.create_dataset("img_old", data=img_old)
                grp.create_dataset("action", data=action)
                grp.create_dataset("reward", data=reward)
                grp.create_dataset("done", data=done)
                grp.create_dataset("img_new", data=img_new)

    def _pull_batch_from_dataset(self, batch_size: int, force_latest_set: bool = False) -> tp.Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
        """
        Pull a batch from the given dataset and rewards
        """
        # get all h5 files
        h5_files = [name for name in os.listdir(self.FILE_PATH) if name.endswith(".h5")]
        # load enough data to fill the batch
        batch = []
        # Create a probability distribution that is weighted towards the end
        weights = np.arange(1, len(h5_files) + 1)
        weights = weights / weights.sum()
        # print("weights: ",weights)
        # select a random h5 file, weighted towards the end to get more relevant data
        h5_file = np.random.choice(h5_files, p=weights)
        print("h5_file chosen: ", h5_file)
        
        with h5.File(os.path.join(self.FILE_PATH, h5_file), "r") as f:
            # print("total steps in file: ", len(f))
            for i in range(batch_size):
                # select a random step^
                rand_int = np.random.randint(len(f))
                # print("rand_int: ", rand_int)
                step = f[f"step_{rand_int}"]
                img_old = step["img_old"][:]
                action = step["action"][()]
                reward = step["reward"][()]
                done = step["done"][()]
                img_new = step["img_new"][:]
                batch.append((img_old, action, reward, done, img_new))
                # TODO: Check if images are correct here
        return batch

    def train(self, batch_count: int, epochs: int, force_latest_set_once: bool = False) -> None:
        """
        Stop the current game and train the model. This will clear the current data and asign rewards to the data points according to the given has_won parameter.
        """
        if self.debug_level >= DEBUG_BASIC:
            print("Training model...")
        avg_loss = 0
        for i in range(batch_count):
            if self.debug_level >= DEBUG_DETAILED:
                print(f"Loading data for batch {i+1}/{batch_count}...")
            data_batch = self._pull_batch_from_dataset(self.BATCH_SIZE, force_latest_set_once)
            batch_size = len(data_batch)
            #######################
            # complete_batch = np.zeros((batch_size * self.action_permutations.shape[0], self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS*self.MEMORY_SIZE))
            # action_count = self.action_permutations.shape[0]
            # for i in range(batch_size):
            #     repeated_arr = np.repeat(data_batch[i][0], action_count, axis=0)
            #     complete_batch[i * action_count:(i+1) * action_count] = repeated_arr
            # repeated_actions = np.repeat(self.action_permutations, batch_size, axis=0)
            # new_preds = np.zeros((batch_size, self.action_permutations.shape[0],1))
            # results = self.running_model.predict((complete_batch, repeated_actions))
            # # change results from (batch_size * action_permutations, 1) to (batch_size, action_permutations, 1)
            # for i in range(batch_size):
            #     new_preds[i] = results[i * self.action_permutations.shape[0]:(i+1) * self.action_permutations.shape[0]]

            #########################
            # repeat states as often as there are actions (so in total, batch_size * action_permutations)
            action_count = self.action_permutations.shape[0]
            stacked_states = np.zeros((batch_size * action_count, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS*self.MEMORY_SIZE))
            for i in range(batch_size):
                start_i = i * action_count
                end_i = (i+1) * action_count
                stacked_states[start_i:end_i] = np.repeat(data_batch[i][4], action_count, axis=0)
            # repeat actions batch_size times
            stacked_actions = np.repeat(self.action_permutations, batch_size, axis=0)
            # predict the q-values for the new states
            predictions = self.target_model.predict((stacked_states, stacked_actions))
            # take the one dimensional predictions output and reshape it to (batch_size, action_count)
            new_preds = np.zeros((batch_size, action_count, 1))
            for i in range(batch_size):
                new_preds[i] = predictions[i * action_count:(i+1) * action_count]


            # new_preds = np.zeros((batch_size, self.action_permutations.shape[0], 1))
            # for i in range(batch_size):
            #     new_preds[i] = self.predict_all_actions(data_batch[i][4], use_target_model=True)
            #######################
            # reward prediction
            rewards = np.zeros((batch_size, 1))
            for i in range(batch_size):
                # get max reward of the given new_preds array
                rewards[i] = np.max(new_preds[i]) * self.DISCOUNT_FACTOR
                rewards[i] += data_batch[i][2]
                if data_batch[i][3] == Reward.WIN.value:
                    rewards[i] += self.WIN_REWARD
                elif data_batch[i][3] == Reward.LOSE.value:
                    rewards[i] += self.LOSE_REWARD
                elif data_batch[i][3] == Reward.NEUTRAL.value:
                    rewards[i] += self.NEUTRAL_REWARD
            # train the model
            if self.debug_level >= DEBUG_BASIC:
                print(f"Training {epochs} epochs...")
            # unpack data_batch
            states = np.array([x[0] for x in data_batch])
            states = np.squeeze(states, axis=1)
            actions = np.array([x[1] for x in data_batch])
            print("states " + str(states.shape))
            print("actions " + str(actions.shape))
            print("rewards " + str(rewards.shape))
            # return states, actions, rewards
            history = self.running_model.fit((states, actions), rewards, epochs=epochs, verbose=self.debug_level)
            # calculate the average loss of the batch
            current_loss_avg = np.mean(history.history["loss"])
            avg_loss += current_loss_avg

            # TODO: Check if rewards are correctly calculated, seems like the best possible outcomes are predicted and used 
            self.update_counter += 1
            # update the target model
            if self.update_counter > self.UPDATE_TARGET_EVERY:
                self.update_counter = 0
                self.target_model.set_weights(self.running_model.get_weights())
                if self.debug_level >= DEBUG_BASIC:
                    print("========== Updated target model ==========================================")
            # decay epsilon
            old_epsilon = self.epsilon
            if self.debug_level >= DEBUG_DETAILED:
                print(f"Trained model for {epochs} epochs")
        avg_loss /= batch_count
        self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
        if self.debug_level >= DEBUG_DETAILED:
            print(f"Epsilon: {old_epsilon} -> {self.epsilon}")
        if self.debug_level >= DEBUG_BASIC:
            print("Saving running model...")
        self.save_models()
        if self.debug_level >= DEBUG_BASIC:
            print("Model training finished")
        return avg_loss
