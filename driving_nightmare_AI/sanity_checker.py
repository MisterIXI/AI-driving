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
import h5py as h5

MODEL_STRING = "sh_model"
PATH = os.path.join(os.path.dirname(__file__), MODEL_STRING)
MODEL_PATH = os.path.join(PATH, MODEL_STRING, "running_model")

running = True
dataset = 0
image = 0
model = md.Model(path_name=MODEL_STRING)
model.load_model()
ds_path = os.path.join(PATH, str(dataset) + ".h5")
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
while running:
    with h5.File(ds_path, "r") as ds:
        same_set = True
        while same_set:
            # load first set in dataset
            print(ds)
            print(ds["step_0"])
            current = ds["step_" + str(image)]["img_old"]
            print("current_shape: " + str(current.shape))
            # q_values = model.predict_all_actions(current)
            q_values = model.choose_next_action(current)
            print("prediction: " + str(q_values))
            cv2.setWindowTitle("image", "prediction: " + str(q_values))
            cv2.imshow("image", current[0][:,:,3:])
            key = cv2.waitKey(0)
            # if key: "a"
            if key == 97:
                image = (image - 1) % len(ds)
            # if key: "d"
            if key == 100:
                image = (image + 1) % len(ds)
            # if key: "w"
            if key == 119:
                dataset += 1
                image = 0
                ds_path = os.path.join(PATH, str(dataset) + ".h5")
                same_set = False
            # if key: "s"
            if key == 115 and dataset > 0:
                dataset -= 1
                image = 0
                ds_path = os.path.join(PATH, str(dataset) + ".h5")
                same_set = False
            # if key: "q"
            if key == 113:
                running = False
                break