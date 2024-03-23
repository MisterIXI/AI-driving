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


model = md.Model(path_name="fb_model")
model.load_model()
model.IMG_HEIGHT = 800//4
model.IMG_WIDTH = 600//4

# batch = model._pull_batch_from_dataset(32)

# print("Checking batch: ")
# for i in range(32):
#     print(i,": ","action: ",batch[i][1]," reward: ",batch[i][2], " finished: ",batch[i][3])

states, actions, rewards = model.train(1,5)
print(actions)
print(rewards)