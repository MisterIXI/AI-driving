# Additional Helper Classes
## Sanity Checker
You can use the [`SanityChecker`](../dqcnn_tensorflow/sanity_checker.py) to view the past experiences present in the model folders. It will use opencv to display the images next to each other, and put predictions in the titlebar, alongside additional information.  
## Stat Plotter
The [`StatPlotter`](../dqcnn_tensorflow/stat_plotter.py) can be used to plot the statistics of the model. It will plot the loss, reward and a moving average of the reward over time.
## Gamepad helper classes
The Gamepad helper classes ([`gamepad_helper.py`](../dqcnn_tensorflow/helper_classes/gamepad_helper.py) and [`gamepad_tracker.py`](../dqcnn_tensorflow/helper_classes/gamepad_tracker.py)) can be used to interact with the gamepad. The `GamepadHelper` class can be used to simulate gamepad input to control games with controller input. This is used for the `Driving Nightmare` game. The `GamepadTracker` class can be used to track the gamepad input and use the information elserwhere. This was used for the initial "offline playing" sessions when a human played the game and the input was recorded. The latter is not used anymore.  
## Autogui helper class
The autogui helper class [`autogui_helper.py`](../dqcnn_tensorflow/helper_classes/auto_gui_helper.py) can be used to wait for specific game screens, like start screen or game over screens. This uses the `pyautogui` library to take screenshots and compare them to a reference image. Currently this is not written very modular, but can be adapted for other games quickly.  
