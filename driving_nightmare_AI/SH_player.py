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
import DQNN.model as md
import pickle as pkl

PATH = os.path.join(os.path.dirname(__file__),"super_hexagon")
IMG_PATH = os.path.join(PATH, "images")
START_IMAGE = os.path.join(IMG_PATH, "start_game.png")
RETRY_IMAGE = os.path.join(IMG_PATH, "retry.png")

class SH_player:
    def __init__(self) -> None:
        self.model: md.Model = md.Model(path_name="sh_model")
        self.model.LOSE_REWARD = -1000
        print("Setting lose reward to -1000")
        self.stop_game = False
        self.action_shape = [3]
        self.last_input = 1
        has_loaded = self.model.load_model()
        if not has_loaded:
            self.model.create_model(self.action_shape, ["steering"])
        self.model.epsilon = 1
        self.model.EPSILON_DECAY = 0.9999
        kb.hook(self.react_on_key)
        self.stats = []
            # if stats pickle exists, load it
        self.SCRIPT_DIR = os.path.dirname(__file__)
        if os.path.exists(os.path.join(self.SCRIPT_DIR, "sh_model", "stats.pickle")):
            with open(os.path.join(self.SCRIPT_DIR, "sh_model", "stats.pickle"), "rb") as f:
                self.stats = pkl.load(f)
        self._wait_for_game()
        self.run()

    def _wait_for_game(self, delay=0.1) -> None:
        while True:
            try:
                location = gui.locateCenterOnScreen(START_IMAGE, confidence=0.9)
                gui.click(location.x, location.y)
                return
            except:
                print("Waiting for the game to be visible...")
                time.sleep(delay)

    def check_for_lose_scren(self) -> bool:
        # check for defeat
        try:
            gui.locateCenterOnScreen(RETRY_IMAGE, confidence=0.9)
            return True
        except:
            return False
        
    def react_on_key(self, key) -> None:
        if key.name == "esc":
            self.stop_game = True
            print("ESC pressed, stopping game after this step...")
    def convert_action_to_input(self, action: np.ndarray) -> int:
        return np.argmax(action)
    
    def press_input(self, action: int) -> None:
        if self.last_input == 0:
            kb.release("left")
        if self.last_input == 2:
            kb.release("right")
        if action == 0:
            kb.press("left")
        if action == 1:
            pass
        if action == 2:
            kb.press("right")
        self.last_input = action

    def release_input(self) -> None:
        if self.last_input == 0:
            kb.release("left")
        if self.last_input == 2:
            kb.release("right")
        self.last_input = 1
    def run(self) -> None:
        while not self.stop_game:
            # start game by pushing spacebar
            print("Starting game from menu")
            kb.press("space")
            sleep(0.1)
            kb.release("space")
            sleep(0.5)
            # start level 1 by pushing spacebar
            print("Starting level 1")
            kb.press("space")
            sleep(0.1)
            kb.release("space")
            sleep(0.3)
            reward = 1
            start_time = time.time()
            while not self.check_for_lose_scren():
                screenshot = gui.screenshot()
                action = self.model.step(screenshot, reward)
                reward+=reward
                if action is None:
                    action = 1
                self.press_input(self.convert_action_to_input(action))
            end_time = time.time()
            self.release_input()
            print(f"Game lasted: {end_time - start_time} seconds")
            self.stats.append({"time": end_time - start_time, "epsilon": self.model.epsilon, "reward": reward})
            with open(os.path.join(self.SCRIPT_DIR, "sh_model", "stats.pickle"), "wb") as f:
                pkl.dump(self.stats, f)
            self.model.finish_run(False)
            self.model.train(4,5,True)


if __name__ == "__main__":
    player = SH_player()

