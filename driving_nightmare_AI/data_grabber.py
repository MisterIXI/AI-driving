import pyautogui as gui
import os
import time
import json
import keyboard as kb
from helper_classes import gamepad_tracker as gt
from helper_classes import auto_gui_helper as ag

tracker = gt.tracker()
tracker.start_tracking_agent()

COLLECTION_DELAY = 0.01  # time waited between each data point collection
DELETION_TIME = 1  # seconds of data to delete on fail
# convert to number of data points to delete
DELETION_COUNT = DELETION_TIME / COLLECTION_DELAY

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = script_path + "\\learning_data"

# load images into variables
again_button = script_path + "\\reference_images\\again_button.png"
start_button = script_path + "\\reference_images\\start_button.png"
header = script_path + "\\reference_images\\HeaderTransparent.png"
victory_image = script_path + "\\reference_images\\Victory.png"
defeat_image = script_path + "\\reference_images\\YouLose.png"

screenWidth, screenHeight = gui.size()  # Get the size of the primary monitor.

# FUNCTIONS
def determine_run_number(current : bool = False) -> int:
    # get the run number
    run_number = 1
    for file in os.listdir(data_path):
        if file.startswith("run_"):
            run_number += 1
    if current and run_number > 1:
        run_number -= 1
    return run_number

def determine_run_path() -> str:
    # get the run number
    run_number = determine_run_number()
    return data_path + "\\run_" + str(run_number)

def save_data() -> None:
    global input_data
    # save input data as json dump
    name = str("inputs_" + str(determine_run_number(True))) + "_" + str(len(input_data)) + ".json"
    with open(run_path + "\\" + name, "w") as file:
        json.dump(input_data, file)
    print("Saved data to " + name)

def on_victory() -> None:
    print("Victory")
    ag.click_again_button()
    save_data()
    start_game()

def on_defeat() -> None:
    print("Defeat")
    ag.click_again_button()
    save_data()
    start_game()

def start_game() -> None:
    print("Starting game")
    # wait for eventual scene loading times
    time.sleep(0.1)
    ag.click_start_button()
    global run_path 
    run_path = determine_run_path()
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    global input_data
    input_data = []


def react_on_key(event) -> None:
    if event.name == "esc":
        global game_running
        print("Escape pressed, stopping run and exiting")
        if game_running:
            save_data()
        tracker.stop_tracking_agent()
        os._exit(0)

kb.hook(react_on_key)
# start game
run_path = determine_run_path()
input_data = []
game_running = False
ag.wait_for_game()
game_running = True
start_game()
time.sleep(COLLECTION_DELAY)
while game_running:
    if ag.check_for_win_screen():
        on_victory()
    elif ag.check_for_lose_screen():
        on_defeat()
    # get inputs
    inputs = tracker.get_inputs()
    # collect data
    screenshot_name = str(len(input_data)) + ".png"
    screenshot = gui.screenshot(run_path + "\\" + screenshot_name)
    # save data
    input_data.append((screenshot_name, inputs))
    print("Collected data point " + str(len(input_data)))
    # wait for some time
    # time.sleep(COLLECTION_DELAY)

