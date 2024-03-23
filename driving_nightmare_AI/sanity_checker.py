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

MODEL_STRING = "fb_model"
PATH = os.path.join(os.path.dirname(__file__), MODEL_STRING)
MODEL_PATH = os.path.join(PATH, MODEL_STRING, "running_model")

running = True
dataset = 0
image = 0
model = md.Model(path_name=MODEL_STRING)
model.load_model()
ds_path = os.path.join(PATH, str(dataset) + ".h5")
h5_count = len([name for name in os.listdir(PATH) if name.endswith(".h5")])
cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
while running:
    with h5.File(ds_path, "r") as ds:
        same_set = True
        while same_set:
            # load first set in dataset
            print(ds)
            print(ds["step_0"])
            curr_raw = ds["step_" + str(image)]["img_old"] 
            curr_step = ds["step_" + str(image)]
            taken_action = curr_step["action"][()]
            current = curr_raw[0]
            print("current_shape: " + str(current.shape))
            # q_values = model.predict_all_actions(current)
            q_values = model.predict_all_actions(curr_raw)
            actions = model.choose_next_action(curr_raw)
            print("prediction: " + str(actions))
            title = "ds: {} im: {} ds_len: {} taken: {} pred: {} q_values: {} reward: {}".format(dataset, image,len(ds), taken_action, actions, q_values, ds["step_" + str(image)]["reward"][()])
            cv2.setWindowTitle("image", title)
            img = np.zeros((current.shape[0], current.shape[1]*5, 1), np.uint8)
            # place the for images next to each other
            for i in range(4):
                img[:,i*current.shape[1]:i*current.shape[1]+current.shape[1], :] = current[:,:,i:i+1]
            # add latest image from img_new to the right
            img[:,4*current.shape[1]:4*current.shape[1]+current.shape[1], :] = ds["step_" + str(image)]["img_new"][0][:, :, 3:4]
            cv2.imshow("image", img)
            # cv2.imshow("image", current[0][:, :, 3:])
            key = cv2.waitKey(0)
            # if key: "a"
            if key == 97:
                print("a pressed, going back one image: ", image, " len(ds): ", len(ds))
                image = (image - 1) % len(ds)
            # if key: "d"
            if key == 100:
                print("d pressed, going forward one image: ", image, " len(ds): ", len(ds))
                image = (image + 1) % len(ds)
            # if key: "w"
            if key == 119:
                dataset = (dataset + 1) % h5_count
                image = 0
                ds_path = os.path.join(PATH, str(dataset) + ".h5")
                print("w pressed, changing dataset to: ", dataset, " ds_path: ", ds_path)
                same_set = False
            # if key: "s"
            if key == 115:
                dataset = (dataset - 1) % h5_count
                image = 0
                ds_path = os.path.join(PATH, str(dataset) + ".h5")
                print("s pressed, changing dataset to: ", dataset, " ds_path: ", ds_path)
                same_set = False
            # if key: "q"
            if key == 113:
                running = False
                break
