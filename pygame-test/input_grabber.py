import pygame as pg
import os
import keyboard as kb
def check(event):
    print(event)
    if event.name == "esc":
        print("Exiting")
        kb.write("Exiting")
        os._exit(1)
pg.init()
screen = pg.display.set_mode((640, 480))
# https://pypi.org/project/vgamepad/
# https://pypi.org/project/keyboard/
# grab keyboard input even without focus
# pg.event.set_grab(Truepip)
kb.hook(check)

while True:
    # for event in pg.event.get():
    #     if event.type == pg.QUIT:
    #         pg.quit()
    #         raise SystemExit
    #     elif event.type == pg.KEYDOWN:
    #         print(event.key)
    pass