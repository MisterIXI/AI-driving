import cv2
import tensorflow as tf
import numpy as np
import os
import h5py
import keyboard as kb

script_dir = os.path.dirname(__file__)


def react_on_key(event) -> None:
    if event.name == "esc":
        print("shutting down...")
        global continue_running
        continue_running = False
        # kb.press_and_release("q")


def on_window_mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        global window
        cv2.setWindowTitle("image", f"image: {x}, {y}")


kb.hook(react_on_key)
model = tf.keras.models.load_model(os.path.join(script_dir, "the_model"))
# train_data = h5py.File(os.path.join(
#     script_dir, 'learning_data', 'training_data.h5'), 'r')
test_data = h5py.File(os.path.join(
    script_dir, 'learning_data', 'validation_data.h5'), 'r')
continue_running = True
window = cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.setMouseCallback("image", on_window_mouse_move)

while continue_running:
    # take a random sample from the test data
    image = test_data.get('validation_data')[np.random.randint(
        0, len(test_data.get('validation_data')))]
    # reshape the image to fit the model
    preview_image = image.copy()
    image = np.reshape(image, (1, 1080//4, 1920//4, 3))
    prediction = model.predict(image)
    # draw two boxes on the image to visualize the prediction
    cv2.rectangle(preview_image, (5, 190), (70, 220), (50, 50, 50), -1)
    # indicator gas
    cv2.rectangle(preview_image, (15, 195), (60, 203), (255, 255, 255), -1)
    gas_x = 15 + int(22.5 * prediction[0][1] + 22.5)
    cv2.rectangle(preview_image, (gas_x-1, 195), (gas_x+1, 203), (0, 0, 255), -1)
    # indicator steering
    cv2.rectangle(preview_image, (15, 207), (60, 215), (255, 255, 255), -1)
    steering_x = 15 + int(22.5 * prediction[0][0] + 22.5)
    cv2.rectangle(preview_image, (steering_x-1, 207),
                  (steering_x+1, 215), (0, 0, 255), -1)
    cv2.setWindowTitle("image", f"image: steering: {prediction[0][0]:.2f}, gas: {prediction[0][1]:.2f}")
    print(prediction[0])

    # preview_image = cv2.resize(preview_image, (1920//2, 1080//2))
    cv2.imshow("image", preview_image)
    cv2.waitKey(0)
