import vgamepad as vg
import tensorflow as tf
import pyautogui as gui
import numpy as np
import os
import cv2
import keyboard as kb
from helper_classes import auto_gui_helper as ag
from helper_classes import gamepad_helper as gh
import time
from time import sleep
import enum as en
import typing as tp
import DQNN.model as md
import pickle as pkl
import flappy_bird as fb

PATH = os.path.join(os.path.dirname(__file__), "fb_model")
# os.environ["SDL_VIDEODRIVER"] = "dummy"

class fb_player:
    def __init__(self) -> None:
        self.model = md.Model(path_name="fb_model", IMG_HEIGHT=800//4, IMG_WIDTH=600//4)
        self.model.LOSE_REWARD = -4
        self.model.LEARNING_RATE = 0.001
        self.model.DISCOUNT_FACTOR = 0.9
        self.action_shape = [2]
        has_loaded = self.model.load_model()
        if not has_loaded:
            self.model.create_model(self.action_shape, ["jump"])
        self.obstacle_reward = -2
        # self.model.epsilon = 0.5
        kb.hook(self.react_on_key)
        self.stats = []
        if os.path.exists(os.path.join(PATH, "stats.pickle")):
            with open(os.path.join(PATH, "stats.pickle"), "rb") as f:
                self.stats = pkl.load(f)
        self.fb = None
        self.model_running = True
        self.run()

    def react_on_key(self, key) -> None:
        if key.name == "esc":
            self.model_running = False
            print("ESC pressed, stopping game after this step...")

    def run(self) -> None:
        while self.model_running:
            # start new game
            self.fb = fb.FlappyBird()
            self.fb.start_game()
            while self.fb.game_running:
                ss = self.fb.get_screen()
                reward = self.fb.check_for_reward()
                reward *= 10
                reward += 5
                # reward += self.fb.steps_taken
                action = self.model.step(ss, reward)

                if action is not None and action[1] > 0.5:
                    self.fb.jump()
                self.fb.multiple_steps(7)
                if not self.model_running:
                    break
            ss = self.fb.get_screen()
            game_over_reward = self.obstacle_reward
            if self.fb.was_OOB:
                game_over_reward = self.model.LOSE_REWARD
            self.model.step(ss, game_over_reward)
            self.model.finish_run(False)
            stat = {"score": self.fb.steps_taken, "epsilon": self.model.epsilon}
            self.fb.close()
            self.fb = None
            avg_loss = self.model.train(10, 5)
            stat["avg_loss"] = avg_loss
            self.stats.append(stat)
            with open(os.path.join(PATH, "stats.pickle"), "wb") as f:
                pkl.dump(self.stats, f)



if __name__ == "__main__":
    player = fb_player()
