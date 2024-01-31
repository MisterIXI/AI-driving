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


class DN_Player:
    def __init__(self) -> None:
        self.model: md.Model = md.Model()
        has_loaded = self.model.load_model()
        if not has_loaded:
            self.model.create_model((1, 2, 21), ("steering", "acceleration"))
        kb.hook(self.react_on_key)
        self.game_running = True
        self.gamepad = gh.gamepad()
        self.SCRIPT_DIR = os.path.dirname(__file__)

    def _open_game(self) -> None:
        exepath = os.path.join(self.SCRIPT_DIR, "DN_Game",
                               r"Driving Nightmare.exe")
        os.startfile(exepath)
        ag.wait_for_game()

    def react_on_key(self, key) -> None:
        if key.name == "esc":
            self.game_running = False
            print("ESC pressed, stopping game after this step...")
    def convert_bin_indices(self, indices: tp.Tuple[float, float]) -> tp.Tuple[float, float]:
        steer_id = indices[0]
        accel_id = indices[1]
        steer = (steer_id-10)/10
        accel = (accel_id-10)/10
        return (steer, accel)

    def convert_bins(self, bins: tp.List[int]) -> tp.List[float]:
        steering_bins = bins[0]
        acceleration_bins = bins[1]
        steering = np.argmax(steering_bins)
        acceleration = np.argmax(acceleration_bins)
        steering, acceleration = self.convert_bin_indices((steering, acceleration))
        # if all values of steering_bins are 0, steering is 0
        if np.all(steering_bins == 0):
            steering = 0
            if self.model.debug_level > 1:
                print("All steering bins are 0, steering is 0")
        # if all values of acceleration_bins are 0, acceleration is 0
        if np.all(acceleration_bins == 0):
            acceleration = 0
            if self.model.debug_level > 1:
                print("All acceleration bins are 0, acceleration is 0")
        return [steering, acceleration]

    def run(self) -> None:
        self._open_game()
        while self.game_running:
            # start game^
            ag.wait_for_game()
            ag.click_start_button()
            state = None
            while state == None:
                if self.game_running == False:
                    break
                screenshot = gui.screenshot()
                input = self.model.step(screenshot)
                # input = self.convert_bins(arr)
                input = self.convert_bin_indices(input)
                self.gamepad.apply_input(input[0], input[1])
                self.gamepad.print_input()
                if ag.check_for_lose_screen():
                    state = False
                elif ag.check_for_win_screen():
                    state = True
            if state == None:
                self.model.save_running_model()
            else:
                self.model.train(5, True)
            if self.game_running:
                ag.click_again_button()
            else:
                # ag.try_click_exit_button()
                pass
        


if __name__ == "__main__":
    player = DN_Player()
    player.run()
