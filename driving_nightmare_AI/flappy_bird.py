import pygame as pg
import random
import time 
import os
import cv2
import numpy as np

class FlappyBird:
    def __init__(self):
        self.width = 800
        self.height = 600
        pg.init()
        self.screen = pg.display.set_mode((self.width, self.height))
        self.clock = pg.time.Clock()
        self.clock.tick(60)
        self.game_running = True
        self.jump_input = False
        self.obstacles = []
        self.bird_rect = pg.Rect(100,50,20,20)
        self.obstacle_cd = 0
        self.passed_since_last_check = 0
        self.steps_taken = 0
        self.obstacles_passed = 0

    def start_game(self):
        self.bird_rect.top = 200
        self.bird_vel = 0
        self.obstacles = []
        self.steps_taken = 0
        self.game_running = True
        self.was_OOB = False
        self.obstacles_passed = 0
        self.obstacle_cd = 0
        self.step()
        # set window title
        pg.display.set_caption("Flappy Bird; Score: 0")

    def spawn_new_obstacle(self):
        gap_oneside = 100
        rnd_boundary = 100
        rnd_height = random.randint(rnd_boundary, self.height - rnd_boundary)
        obst_top = pg.Rect(self.width, 0, 50, rnd_height - gap_oneside)
        obst_bottom = pg.Rect(self.width, rnd_height + gap_oneside, 50, self.height - rnd_height - gap_oneside)
        self.obstacles.append((obst_top, obst_bottom))
    
    def check_for_reward(self) -> int:
        passed = self.passed_since_last_check
        self.passed_since_last_check = 0
        return passed
    
    def step(self) -> bool:
        pg.event.pump()
        # check for jump input
        if self.jump_input:
            self.bird_vel = -10
            self.jump_input = False
        self.bird_rect.top += self.bird_vel
        self.bird_vel += 1
        # move obstacles
        for obstacle in self.obstacles:
            ob1, ob2 = obstacle
            # check if bird passed obstacle
            if ob1.right >= self.bird_rect.left and ob1.right - 5 < self.bird_rect.left:
                self.passed_since_last_check += 1
                self.obstacles_passed += 1
                pg.display.set_caption("Flappy Bird; Score: " + str(self.obstacles_passed))
            ob1.left -= 5
            ob2.left -= 5
        cleanup = len(self.obstacles) > 0
        while cleanup:
            cleanup = False
            # check if leftmost obstacle is out of screen
            if self.obstacles[0][0].right < 0:
                self.obstacles.pop(0)
                cleanup = True

        # check for collision with ground or ceiling
        if self.bird_rect.top + 10 > self.height or self.bird_rect.top - 10 < 0:
            self.game_running = False
            self.was_OOB = True
        # check for collision with obstacles
        for obstacle in self.obstacles:
            if self.bird_rect.colliderect(obstacle[0]) or self.bird_rect.colliderect(obstacle[1]):
                self.game_running = False
                self.was_OOB = False
        if self.game_running == False:
            print("Game Over")
        # clear screen
        self.screen.fill((0,0,0))
        # draw obstacles
        for obstacle in self.obstacles:
            pg.draw.rect(self.screen, (50,180,100), obstacle[0])
            pg.draw.rect(self.screen, (50,180,100), obstacle[1])
        # draw bird
        pg.draw.rect(self.screen, (255,255,255), self.bird_rect)
        # spawn new obstacles
        self.obstacle_cd -= 1
        if self.obstacle_cd <= 0:
            self.spawn_new_obstacle()
            self.obstacle_cd = 70
        # advance one frame in the game
        self.clock.tick(200)
        pg.display.flip()
        self.steps_taken += 1


    def multiple_steps(self, n_steps: int):
        for _ in range(n_steps):
            self.step()

    def jump(self):
        self.jump_input = True
    
    def close(self):
        pg.quit()

    def spin(self):
        pg.display.flip()

    def check_for_human_input(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.game_running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    print("Jump pressed!")
                    self.jump()
                if event.key == pg.K_f:
                    print("F pressed!")
                    # self.step()  
                    self.multiple_steps(7)       
                if event.key == pg.K_q:
                    img = self.get_screen()
                    cv2.imshow("screen", img)
        pg.display.update()

    def get_screen(self):
        sur = pg.surfarray.array3d(self.screen)
        print(sur.shape)
        return np.transpose(sur, (1,0,2))

if __name__ == "__main__":
    game = FlappyBird()
    game.start_game()
    while game.game_running:
        game.check_for_human_input()
        # time.sleep(0.05)
    game.close()







