import torch as th
import numpy as np
import os
import cv2
import enum as en
import typing as tp
import pickle as pkl
import h5py as h5
from collections import deque
import enum as en
import torchsummary as ts

DEBUG_NONE = 0
DEBUG_BASIC = 1
DEBUG_DETAILED = 2


class Reward(en.Enum):
    NEUTRAL = 0
    WIN = 1
    LOSE = 2
    INTERMEDIATE = 3


class Model:
    """
    Model class for the DQCNN model
    """

    def __init__(self, debug_level: int = 2, IMG_WIDTH=1920//4, IMG_HEIGHT=1080//4, path_name="rl_data") -> None:
        # general
        self.FILE_PATH = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), path_name)
        self.debug_level = debug_level
        self.cuda = th.cuda.is_available()
        # model parameters
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_CHANNELS = 1
        self.MEMORY_SIZE = 4
        # step parameters
        self.LEARNING_RATE = 0.0001
        self.EPSILON_MAX = 1.0
        self.EPSILON_MIN = 0.02
        self.EPSILON_DECAY = 0.9
        self.epsilon = self.EPSILON_MAX
        self.NEUTRAL_REWARD = 0
        self.LOSE_REWARD = -5
        self.WIN_REWARD = 5
        self.BONUS_REWARD = 1
        # training parameters
        self.DISCOUNT_FACTOR = 0.99
        self.BATCH_SIZE = 32
        self.UPDATE_TARGET_EVERY = 20
        self.update_counter = 0
        # data
        self.framebuffer = deque(maxlen=self.MEMORY_SIZE + 1)
        # deque which stores run data: (old_state, action, state_specific_reward, reward_class, new_state, just_explored)
        self.run_data = deque()
        self.last_action = None

    def reset_memory(self) -> None:
        self.framebuffer.clear()
        self.run_data.clear()
        self.last_action = None
# region action permutation

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
            self._rec_perms(actions, actions_shape, depth +
                            1, offset + actions_shape[depth])

    def _create_action_permutations(self, action_shape) -> None:
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
# endregion

    def create_model(self, actions_shape: list[list], layer_names: tp.Tuple) -> None:
        if self.debug_level >= DEBUG_BASIC:
            print("Initializing model from scratch...")
        self.actions_shape = actions_shape
        self.action_length = 0
        for action in actions_shape:
            self.action_length += action
        self.layer_names = layer_names
        self._create_action_permutations(self.actions_shape)
        input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH,
                       self.IMG_CHANNELS*self.MEMORY_SIZE)
        self.model_shape = [th.nn.Conv2d(input_shape[2], 32, 4),  # 32x480x270
                            th.nn.MaxPool2d(2),  # 32x240x135
                            th.nn.Conv2d(32, 64, 4, 2),  # 64x240x135
                            th.nn.MaxPool2d(2),  # 64x120x67
                            th.nn.Conv2d(64, 64, 3, 1),  # 64x120x67
                            th.nn.AdaptiveMaxPool2d((15, 30)),  # 64x60x33
                            th.nn.Flatten(),
                            th.nn.Linear(28800, 512),
                            th.nn.ReLU(),
                            th.nn.Linear(512, self.action_permutations.shape[0])]

        self.running_model = th.nn.Sequential(*self.model_shape)
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print(ts.summary(self.running_model.to(device), input_size=(
            input_shape[2], input_shape[0], input_shape[1])))
        self.optimizer = th.optim.Adam(
            self.running_model.parameters(), lr=self.LEARNING_RATE)
        self.loss_fn = th.nn.MSELoss()
        self.target_model = th.nn.Sequential(*self.model_shape)
        self.target_model.load_state_dict(self.running_model.state_dict())
        # print running model to console
        if self.debug_level >= DEBUG_DETAILED:
            print(self.running_model)
# region action prediction

    def predict_all_actions(self, image_buffer: np.ndarray, use_target_model: bool = False) -> np.ndarray:
        """Predicts the Q-Values for all possible actions"""
        if use_target_model:
            return self.target_model(image_buffer)
        else:
            return self.running_model(image_buffer)

    def choose_next_action(self, image_buffer: np.ndarray, suppress_exploration: bool = False) -> np.ndarray:
        """Predicts the Q-values for the given image buffer and returns either the best action, or a random for exploration"""
        if not suppress_exploration and np.random.rand() <= self.epsilon:
            # perform exploration
            if self.debug_level >= DEBUG_DETAILED:
                print("Performing exploration")
            # choose random array from action_permutations
            return np.random.randint(self.action_permutations.shape[0]), True
        q_values = self.predict_all_actions(image_buffer)
        # get the index of the max value
        index = th.argmax(q_values)
        # convert tensor to normal number
        index = index.item()
        # index = np.argmin(q_values)
        if self.debug_level >= DEBUG_DETAILED:
            print("Predicted Action: ", self.action_permutations[index])
        return index, False
# endregion
# region helper functions

    def _preprocess_image(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Preprocesses the given screenshot
        """
        screenshot = np.array(screenshot)
        # resize the image
        image = cv2.resize(screenshot, (self.IMG_WIDTH, self.IMG_HEIGHT))

        # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.reshape(image, (self.IMG_HEIGHT, self.IMG_WIDTH, 1))
        return image

    def _save_dataset(self, dataset: deque) -> None:
        """
        Save the given dataset with the given rewards
        """
        h5_files = [name for name in os.listdir(
            self.FILE_PATH) if name.endswith(".h5")]
        h5_highest_num = -1
        if h5_files:
            h5_highest_num = max([int(name.split(".")[0])
                                 for name in h5_files])
        # save the dataset
        with h5.File(os.path.join(self.FILE_PATH, f"{h5_highest_num + 1}.h5"), "w") as f:
            for i, (img_old, action, reward, done, img_new, just_explored) in enumerate(dataset):
                grp = f.create_group(f"step_{i}")
                grp.create_dataset("img_old", data=img_old)
                grp.create_dataset("action", data=action)
                grp.create_dataset("reward", data=reward)
                grp.create_dataset("done", data=done)
                grp.create_dataset("img_new", data=img_new)
                grp.create_dataset("just_explored", data=just_explored)

# endregion

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
                self.framebuffer.append(
                    np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)))
        self.framebuffer.append(image)
        if len(self.framebuffer) == self.MEMORY_SIZE:
            # if buffer not completely filled, only predict this round
            buffer = np.array(self.framebuffer)
            # print("state" + str(state.shape))
            # state = np.reshape(state, (1, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS*self.MEMORY_SIZE))
            # print("state" + str(state.shape))
            state = th.from_numpy(buffer).permute(3, 0, 1, 2).float().cuda()
            print("state: ", state.shape)
            self.last_action, self.just_explored = self.choose_next_action(
                state, suppress_exploration=False)
            return self.action_permutations[self.last_action]
        # predict the bin indices
        buffer_arr = np.array(self.framebuffer)
        # old_state is the buffer without the last element
        old_state_np = buffer_arr[:-1]
        old_state_th = th.from_numpy(old_state_np).permute(
            3, 0,  1, 2).float().cuda()
        # new_state is the buffer without the first element
        new_state_np = buffer_arr[1:]
        new_state_th = th.from_numpy(new_state_np).permute(3, 0, 1, 2).float().cuda()
        print("old_state: ", old_state_th.shape)
        print("new_state: ", new_state_th.shape)

        selected_action, self.just_explored = self.choose_next_action(
            new_state_th, suppress_exploration=False)
        # safe current data
        buffer_arr = np.array(self.framebuffer)
        new_np = buffer_arr[1:]
        new_np = np.transpose(new_np, (3,1,2,0))
        old_np = buffer_arr[:-1]
        old_np = np.transpose(old_np, (3,1,2,0))
        self.run_data.append((old_np, self.last_action, bonus_reward,
                             Reward.NEUTRAL.value, new_np, self.just_explored))
        self.last_action = selected_action
        # if self.debug_level >= DEBUG_DETAILED:
        #     print(f"Step: indices: {indices}, q_values: {q_values}")
        return self.action_permutations[selected_action]

    def finish_run(self, reward_state: Reward, intermediate: tp.Tuple[int, int] = None) -> None:
        """Finish run, and assign states to the run.
        intermediate: (every_x_steps, min_distance_to_end)"""
        x = self.run_data[-1]
        self.run_data[-1] = (x[0], x[1], x[2], reward_state, x[4], x[5])

        self._save_dataset(self.run_data)
        self.run_data.clear()
        self.framebuffer.clear()

    def save_models(self) -> None:
        # save the running model
        th.save(self.running_model.state_dict(), os.path.join(
            self.FILE_PATH, "running_model.pth"))
        # save the target model
        th.save(self.target_model.state_dict(), os.path.join(
            self.FILE_PATH, "target_model.pth"))
        # save model parameters
        with open(os.path.join(self.FILE_PATH, "model_params.pickle"), "wb") as f:
            pkl.dump({
                "IMG_WIDTH": self.IMG_WIDTH,
                "IMG_HEIGHT": self.IMG_HEIGHT,
                "IMG_CHANNELS": self.IMG_CHANNELS,
                "MEMORY_SIZE": self.MEMORY_SIZE,
                "LEARNING_RATE": self.LEARNING_RATE,
                "EPSILON_MAX": self.EPSILON_MAX,
                "EPSILON_MIN": self.EPSILON_MIN,
                "EPSILON_DECAY": self.EPSILON_DECAY,
                "epsilon": self.epsilon,
                "NEUTRAL_REWARD": self.NEUTRAL_REWARD,
                "LOSE_REWARD": self.LOSE_REWARD,
                "WIN_REWARD": self.WIN_REWARD,
                "BONUS_REWARD": self.BONUS_REWARD,
                "DISCOUNT_FACTOR": self.DISCOUNT_FACTOR,
                "BATCH_SIZE": self.BATCH_SIZE,
                "UPDATE_TARGET_EVERY": self.UPDATE_TARGET_EVERY,
                "update_counter": self.update_counter,
                "model_shape": self.model_shape
            }, f)

    def load_models(self) -> bool:
        # check if all files exist
        if not os.path.exists(os.path.join(self.FILE_PATH, "model_params.pickle")):
            return False
        if not os.path.exists(os.path.join(self.FILE_PATH, "running_model.pth")):
            return False
        if not os.path.exists(os.path.join(self.FILE_PATH, "target_model.pth")):
            return False
        # load model parameters
        with open(os.path.join(self.FILE_PATH, "model_params.pickle"), "rb") as f:
            params = pkl.load(f)
            self.IMG_WIDTH = params["IMG_WIDTH"]
            self.IMG_HEIGHT = params["IMG_HEIGHT"]
            self.IMG_CHANNELS = params["IMG_CHANNELS"]
            self.MEMORY_SIZE = params["MEMORY_SIZE"]
            self.LEARNING_RATE = params["LEARNING_RATE"]
            self.EPSILON_MAX = params["EPSILON_MAX"]
            self.EPSILON_MIN = params["EPSILON_MIN"]
            self.EPSILON_DECAY = params["EPSILON_DECAY"]
            self.epsilon = params["epsilon"]
            self.NEUTRAL_REWARD = params["NEUTRAL_REWARD"]
            self.LOSE_REWARD = params["LOSE_REWARD"]
            self.WIN_REWARD = params["WIN_REWARD"]
            self.BONUS_REWARD = params["BONUS_REWARD"]
            self.DISCOUNT_FACTOR = params["DISCOUNT_FACTOR"]
            self.BATCH_SIZE = params["BATCH_SIZE"]
            self.UPDATE_TARGET_EVERY = params["UPDATE_TARGET_EVERY"]
            self.update_counter = params["update_counter"]
            self.model_shape = params["model_shape"]
        # load the running model
        self.running_model = th.nn.Sequential(*self.model_shape)
        self.running_model.load_state_dict(
            th.load(os.path.join(self.FILE_PATH, "running_model.pth")))
        # load the target model
        self.target_model = th.nn.Sequential(*self.model_shape)
        self.target_model.load_state_dict(
            th.load(os.path.join(self.FILE_PATH, "target_model.pth")))

    def _pull_batch_from_dataset(self, batch_size: int, force_latest_set: bool = False) -> tp.Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
        """
        Pull a batch from the given dataset and rewards
        """
        # TODO: Improve pulling by using X many h5 files to fill up the batch_size
        # get all h5 files
        h5_files = [name for name in os.listdir(
            self.FILE_PATH) if name.endswith(".h5")]
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
        return batch

    def _fit(self, states, target_values, epochs: int) -> int:
        avg_loss = 0
        for i in range(epochs):
            states_copy = states.clone()
            # Forward pass
            outputs = self.running_model(states_copy)
            loss = self.loss_fn(outputs, target_values)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            avg_loss += loss.item()
        return avg_loss/epochs

    def train(self, batch_count: int, epochs: int, force_latest_set_once: bool = False) -> None:
        """
        Stop the current game and train the model. This will clear the current data and asign rewards to the data points according to the given has_won parameter.
        """
        th.autograd.set_detect_anomaly(True)
        if self.debug_level >= DEBUG_BASIC:
            print("Training model...")
        avg_loss = 0
        for i in range(batch_count):
            if self.debug_level >= DEBUG_DETAILED:
                print(f"Loading data for batch {i+1}/{batch_count}...")
            data_batch = self._pull_batch_from_dataset(
                self.BATCH_SIZE, force_latest_set_once)
            batch_size = len(data_batch)
            state_stack = th.zeros(
                (batch_size*2, self.IMG_CHANNELS*self.MEMORY_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH))
            for i in range(batch_size):
                state_stack[i] = th.from_numpy(
                    data_batch[i][0]).permute(0, 3, 1, 2).float()
                state_stack[i+batch_size] = th.from_numpy(
                    data_batch[i][4]).permute(0, 3, 1, 2).float()
            # state_stack = np.zeros((batch_size*2, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS*self.MEMORY_SIZE))
            # state_stack[:batch_size] = [data[0] for data in data_batch]
            # state_stack[batch_size:] = [data[4] for data in data_batch]
            if self.cuda:
                state_stack = state_stack.cuda()
            total_predictions = self.target_model(state_stack)
            print("total_predictions: ", total_predictions.shape)
            # convert to np array
            target_values = total_predictions[:batch_size]
            future_rewards = total_predictions[batch_size:]
            for i in range(batch_size):
                # replace the taken action with actual reward
                actual_reward = data_batch[i][2]
                # add end state rewards
                if data_batch[i][3] == Reward.WIN.value:
                    actual_reward += self.WIN_REWARD
                elif data_batch[i][3] == Reward.LOSE.value:
                    actual_reward += self.LOSE_REWARD
                elif data_batch[i][3] == Reward.NEUTRAL.value:
                    actual_reward += self.NEUTRAL_REWARD
                    # add the best reward of the next state with the discount factor if not dead next turn
                    fr_clone = future_rewards.clone()
                    actual_reward += th.max(fr_clone[i]) * self.DISCOUNT_FACTOR
                    # print("actual_reward: ", actual_reward)
                target_values[i][data_batch[i][1]] = actual_reward

            # train the model
            if self.debug_level >= DEBUG_BASIC:
                print(f"Training {epochs} epochs...")

            print("states " + str(state_stack[:batch_size].shape))
            print("rewards " + str(target_values.shape))
            avg_loss += self._fit(state_stack[:batch_size].clone(),
                                  target_values, epochs)

            # TODO: Check if rewards are correctly calculated, seems like the best possible outcomes are predicted and used
            self.update_counter += 1
            # update the target model
            if self.update_counter > self.UPDATE_TARGET_EVERY:
                self.update_counter = 0
                self.target_model.load_state_dict(
                    self.running_model.state_dict())
                if self.debug_level >= DEBUG_BASIC:
                    print(
                        "========== Updated target model ==========================================")
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
