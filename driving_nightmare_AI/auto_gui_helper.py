import pyautogui as gui
import time
import os
SCRIPT_DIR = os.path.dirname(__file__)
HEADER_PNG = os.path.join(SCRIPT_DIR, "reference_images", "Header.png")
YOU_LOSE_PNG = os.path.join(SCRIPT_DIR, "reference_images", "YouLose.png")
YOU_WIN_PNG = os.path.join(SCRIPT_DIR, "reference_images", "Victory.png")
AGAIN_BUTTON_PNG = os.path.join(SCRIPT_DIR, "reference_images", "again_button.png")
START_BUTTON_PNG = os.path.join(SCRIPT_DIR, "reference_images", "start_button.png")

def move_mouse_away() -> None:
    screenWidth, screenHeight = gui.size()  # Get the size of the primary monitor.
    # move mouse out of the way
    gui.moveTo(30, screenHeight)

def wait_for_game(delay = 0.1) -> None:
    print("Waiting for game to start")
    # wait until header visible
    while True:
        try:
            gui.locateCenterOnScreen(HEADER_PNG, confidence=0.9)
            return
        except:
            print("Waiting for the game to be visible...")
            time.sleep(delay)

def check_for_lose_screen() -> bool:
    # check for defeat
    try:
        gui.locateCenterOnScreen(YOU_LOSE_PNG, confidence=0.9)
        return True
    except:
        return False
    
def check_for_win_screen() -> bool:
    # check for victory
    try:
        gui.locateCenterOnScreen(YOU_WIN_PNG, confidence=0.9)
        return True
    except:
        return False
    
def click_again_button() -> None:
    print("Clicking again button")
    x, y = gui.locateCenterOnScreen(AGAIN_BUTTON_PNG, confidence=0.9)
    gui.click(x, y)
    move_mouse_away()

def click_start_button() -> None:
    print("Clicking start button")
    x, y = gui.locateCenterOnScreen(START_BUTTON_PNG, confidence=0.9)
    gui.click(x, y)
    move_mouse_away()