import vgamepad as vg
import time
import tensorflow as tf
import numpy as np
import os
import pyautogui as gui
import cv2
import math
import inputs
import keyboard as kb

import auto_gui_helper as ag
gamepad = vg.VX360Gamepad()
script_dir = os.path.dirname(__file__)

def react_on_key(event) -> None:
    if event.name == "esc":
        global stats
        print("Stats:")
        print(stats)
        os._exit(0)

kb.hook(react_on_key)
# import model
model = tf.keras.models.load_model(os.path.join(script_dir, "the_model"))
stats = {
    "max": 0,
    "min": 0
}
ag.wait_for_game()
ag.click_start_button()
while True:
# for i in range(1000):
    screenshot = gui.screenshot()
    screenshot = np.asarray(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    screenshot = cv2.resize(screenshot, (1920//4, 1080//4))
    screenshot = np.reshape(screenshot, (1, 1080//4, 1920//4, 3))
    prediction = model.predict(screenshot)
    x_input = max(min(prediction[0][0], 1.0), -1.0)
    trigger_input = max(min(prediction[0][1], 1.0), -1.0)
    if trigger_input > 0:
        LT_input = 0
        RT_input = trigger_input
    else:
        LT_input = -trigger_input
        RT_input = 0
    gamepad.left_joystick_float(x_value_float=x_input, y_value_float=0.0)
    gamepad.left_trigger_float(value_float=LT_input)
    gamepad.right_trigger_float(value_float=RT_input)
    gamepad.update()
    print(prediction[0])
    if(prediction[0][0] > stats["max"]):
        stats["max"] = prediction[0][0]
    if(prediction[0][0] < stats["min"]):
        stats["min"] = prediction[0][0]
    if ag.check_for_win_screen():
        ag.click_again_button()
        ag.click_start_button()
    elif ag.check_for_lose_screen():
        ag.click_again_button()
        ag.click_start_button()
    # print(model.predict(screenshot))
