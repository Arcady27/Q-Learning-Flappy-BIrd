import numpy
from math import exp

import flappybird_original as flappybird

class frame:
    def __init__(self,bird_y_,bird_velocity,x1_,y11_,y21_):
        self.bird_y = (flappybird.WIN_HEIGHT - float(bird_y_))/flappybird.WIN_HEIGHT
        self.bird_speed = -bird_velocity
        self.pipe1_x = float(x1_)/flappybird.WIN_WIDTH
        self.pipe1_y1 = float(y11_ + 40)/flappybird.WIN_HEIGHT
        self.pipe2_y1 = float(y21_ + 40)/flappybird.WIN_HEIGHT

        self.array = [self.bird_speed,self.bird_y,self.pipe1_x,self.pipe1_y1,self.pipe2_y1]

class game_client:
    def __init__(self):
        self.game = flappybird.Main()

    def perform_action(self, action_num):
        if action_num == 1:
            self.game.jump()

    def change_screen_values(self,Qs):
        self.game.change_Qs(Qs[0][0]*100,Qs[0][1]*100)

    def run(self):
        previous_delta = 0
        current_action = 0
        current_pipe = 0

        losts = 0
        iter = 0
        current_gamescore = 0

        for collision,gamescore,score_updated,bird_y,bird_x,bird_velocity,pipes in self.game.run():
            bird_y = flappybird.WIN_HEIGHT - bird_y
            bird_y = max(bird_y,0)

            iter += 1
            if iter % 2 != 0:
                continue

            pipes = list(pipes)
            if len(pipes) == 1:
                current_pipe = pipes[0]
                continue

            elif len(pipes) == 2:
                #pipe=(x,y1,y2)
                pipe1=(pipes[0].x, pipes[0].bottom_height_px, pipes[0].bottom_height_px+pipes[0].WIDTH)
                pipe2=(pipes[1].x, pipes[1].bottom_height_px, pipes[1].bottom_height_px+pipes[1].WIDTH)
                frame_ = frame(bird_y,bird_velocity,pipe1[0],pipe1[1],pipe2[1])

            middle = pipes[0].bottom_height_px + 40
            delta = abs(bird_y - middle)

            reward = 0
            if bird_y < 30 or bird_y > 500:
                reward = -1

            if delta >= 100:
                reward = -0.5

            if delta >= 60:
                reward = -0.35

            if delta >= 40:
                reward = -0.2

            if delta < 60:
                reward = 0.1

            if delta < 40:
                reward = 0.15

            if delta < 20:
                reward = 0.3

            if score_updated:
                reward = 1
                print reward

            if collision:
                reward = -1


            current_pipe = pipe1
            previous_delta = delta

            yield reward, frame_
