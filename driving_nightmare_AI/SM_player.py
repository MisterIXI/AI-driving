import tensorflow as tf
import DQNN.model as md
import typing as tp
import enum as en
from time import sleep
import time
import keyboard as kb
import cv2
import pyautogui as gui
import os
import numpy as np
import sys
import pytesseract as pt
import subprocess

pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

START_IMAGE = os.path.join(os.path.dirname(__file__), "SnakManiac", "images", "start_image.png")
START_BUTTON = os.path.join(os.path.dirname(__file__), "SnakManiac", "images", "start.png")
RESTART_BUTTON = os.path.join(os.path.dirname(__file__), "SnakManiac", "images", "restart.png")
SCORE = os.path.join(os.path.dirname(__file__), "SnakManiac", "images", "score.png")
GAME_REGION = (560, 20, 800, 1010)
IMG_WIDTH = 800//2
IMG_HEIGHT = 1010//2


class sm_player():
    def __init__(self) -> None:
        self.model: md.Model = md.Model(IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, path_name="sm_model")
        self.path = os.path.dirname(__file__)
        has_loaded = self.model.load_model()
        self.action_shape = [4]
        if not has_loaded:
            self.model.create_model(self.action_shape, ["steering"])
        kb.hook(self.react_on_key)
        self.game_running = True
        self.SCRIPT_DIR = os.path.dirname(__file__)
        self.end_early = False

    def _open_game(self) -> None:
        exepath = os.path.join(self.SCRIPT_DIR, "SnakManiac", "Snakii_Game", r"Snakii.exe")
        subprocess.Popen(exepath, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.wait_for_game()

    def react_on_key(self, key) -> None:
        if key.name == "esc":
            self.game_running = False
            print("ESC pressed, stopping game after this step...")
        if key.name == "f":
            self.end_early = True
            print("F pressed, stopping game after this step...")

    def wait_for_game(self, delay=0.1) -> None:
        print("Waiting for game to start")
        # wait until header visible
        while True:
            try:
                gui.locateCenterOnScreen(START_IMAGE, confidence=0.9)
                return
            except:
                print("Waiting for the game to be visible...")
                time.sleep(delay)

    def check_for_lose_screen(self) -> bool:
        # check for defeat
        try:
            gui.locateCenterOnScreen(RESTART_BUTTON, confidence=0.9)
            return True
        except:
            return False

    def click_restart_button(self) -> None:
        print("Clicking restart button")
        x, y = gui.locateCenterOnScreen(RESTART_BUTTON, confidence=0.9)
        gui.click(x, y)
        self.move_mouse_away()

    def click_start_button(self) -> None:
        print("Clicking start button")
        x, y = gui.locateCenterOnScreen(START_BUTTON, confidence=0.9)
        gui.click(x, y)
        self.move_mouse_away()

    def move_mouse_away(self) -> None:
        screenWidth, screenHeight = gui.size()
        gui.moveTo(30, screenHeight)

    def wait_for_game_step(self, old_screenshot: np.ndarray) -> tp.Tuple[np.ndarray, int]:
        diff = 0
        while diff < 1000:
            screenshot = gui.screenshot(region=GAME_REGION)
            arr = np.array(screenshot)
            img = self._preprocess_image(arr)
            # check for differences
            diff = np.sum(cv2.absdiff(old_screenshot, img))
            print("Difference: ", diff)
            sleep(0.1)
        score = self.read_score(np.array(screenshot))
        return img, score

    def _preprocess_image(self, screenshot: np.ndarray) -> np.ndarray:
        cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        return cv2.resize(screenshot, (IMG_HEIGHT, IMG_WIDTH))

    def push_action(self, action: np.ndarray) -> None:
        if action is None:
            return
        arr1 = action
        input1 = np.argmax(arr1)
        # if input1 == 0: press arrow left on kb
        # if input1 == 1: press arrow right on kb
        # if input1 == 2: press arrow up on kb
        # if input1 == 3: press arrow down on kb
        if input1 == 0:
            kb.press_and_release("a")
            print("Left" + " \u2190")
        elif input1 == 1:
            kb.press_and_release("d")
            print("Right" + " \u2192")
        elif input1 == 2:
            kb.press_and_release("w")
            print("Up" + " \u2191")
        elif input1 == 3:
            kb.press_and_release("s")
            print("Down" + " \u2193")

    def read_score(self, unprocessed_img: np.ndarray):
        # W620, H160 -> w25,h170
        region = unprocessed_img[155:175, 620:780]
        # read text from region
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        text = pt.image_to_string(gray, config='--psm 6')
        text = text.replace("Oo", "0")
        print("Read text: ", text)
        try:
            int_text = int(text)
            print("Current score: ", int_text)
        except:
            int_text = None
            print("Could not read score")
        return int_text

    def calc_score_reward(self, old_score: int, new_score: int) -> int:
        if old_score == None or new_score == None:
            return 0
        diff = new_score - old_score
        return diff * 10

    def run(self) -> None:
        self._open_game()
        self.wait_for_game()
        self.click_start_button()
        self.first_round = True
        self.last_score = 0
        while self.game_running:
            img = gui.screenshot(region=GAME_REGION)
            self.last_img = self._preprocess_image(np.array(img))
            if self.first_round == False:
                self.click_restart_button()
                self.wait_for_game()
                self.click_start_button()
            else:
                self.first_round = False
            state = None
            while state == None:
                if self.game_running == False:
                    break
                img, score = self.wait_for_game_step(self.last_img)
                if score == None:
                    score = self.last_score
                reward = self.calc_score_reward(self.last_score, score)
                action = self.model.step(img, reward)
                self.push_action(action)
                if self.check_for_lose_screen():
                    state = True
                if self.end_early:
                    state = False
                    self.game_running = False
                self.last_img = img
                self.last_score = score

            if state:
                self.model.finish_run(False)
            elif state == False:
                self.model.finish_run(None)
            
            self.model.train(10, 5, True)


if __name__ == "__main__":
    player = sm_player()
    player.run()
    print("Game stopped")
    sys.exit(0)
    # unprocessed_img = np.array(gui.screenshot(region=GAME_REGION))
    # region = unprocessed_img[155:175, 620:780]
    # # read text from region
    # gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    # # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)
    # text = pt.image_to_string(gray, config='--psm 6')
    # print(text)
