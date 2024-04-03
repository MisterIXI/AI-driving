import vgamepad as vg
import numpy as np
import os
import pyautogui as gui
import keyboard as kb
from helper_classes import auto_gui_helper as ag
from helper_classes import gamepad_helper as gh
import time
from time import sleep
import enum as en
import typing as tp
import DQCNN.model as md
import pickle as pkl


class DN_Player:
    def __init__(self) -> None:
        self.model: md.Model = md.Model()
        has_loaded = self.model.load_models()
        self.action_shape = [3, 3]
        if not has_loaded:
            self.model.create_model(self.action_shape, ["steering", "acceleration"])
        self.model.LOSE_REWARD = -10
        self.model.EPSILON_DECAY = 0.9
        self.model.LEARNING_RATE = 0.001
        # self.model.epsilon = 0.2
        kb.hook(self.react_on_key)
        self.game_running = True
        self.gamepad = gh.gamepad()
        self.SCRIPT_DIR = os.path.dirname(__file__)
        self.stats = []
        # if stats pickle exists, load it
        if os.path.exists(os.path.join(self.SCRIPT_DIR, "rl_data", "stats.pickle")):
            with open(os.path.join(self.SCRIPT_DIR, "rl_data", "stats.pickle"), "rb") as f:
                self.stats = pkl.load(f)

    def _open_game(self) -> None:
        exepath = os.path.join(self.SCRIPT_DIR, "DN_Game",
                               r"Driving Nightmare.exe")
        os.startfile(exepath)
        ag.wait_for_game()

    def react_on_key(self, key) -> None:
        if key.name == "esc":
            self.game_running = False
            print("ESC pressed, stopping game after this step...")

    def convert_action_to_input(self, action: np.ndarray) -> tp.Tuple[float, float]:
        arr1 = action[:3]
        arr2 = action[3:]
        input1 = np.argmax(arr1)
        input2 = np.argmax(arr2)
        # steer = (input1 - 10) / 10
        steer = input1 - 1.0
        accell = input2 - 1.0
        return steer, accell

    def run(self) -> None:
        self._open_game()
        while self.game_running:
            # start game^
            run_num = len(self.stats)
            ag.wait_for_game()
            ag.click_start_button()
            state = None
            start_time = time.time()
            while state == None:
                
                if self.game_running == False:
                    break
                screenshot = gui.screenshot()
                action = self.model.step(screenshot, 5)
                # ##############
                # action = self.model.step(screenshot, 5)
                # state = False
                # break
                # ##############
                # input = self.convert_bins(arr)
                if action is None:
                    input = (0, 0)
                else:
                    input = self.convert_action_to_input(action)
                self.gamepad.apply_input(input[0], input[1])
                self.gamepad.print_input()
                if ag.check_for_lose_screen():
                    state = False
                elif ag.check_for_win_screen():
                    state = True
            self.gamepad.reset()
            final_time = time.time() - start_time
            if state != None:
                self.model.finish_run(state)
                avg_loss = self.model.train(5, 1, True)
                self.stats.append({"run": run_num, "time": final_time, "epsilons": self.model.epsilon, "avg_loss": avg_loss})
                print("Run: " + str(run_num) + " Time: " + str(final_time))
                with open(os.path.join(self.SCRIPT_DIR, "rl_data", "stats.pickle"), "wb") as f:
                    pkl.dump(self.stats, f)
            if self.game_running:
                ag.click_again_button()
            else:
                # ag.try_click_exit_button()
                pass


if __name__ == "__main__":
    player = DN_Player()
    player.run()
    # model = md.Model()
    # model.create_model((1, 2, 21), ("steering", "acceleration"))
