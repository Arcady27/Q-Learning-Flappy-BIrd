#! /usr/bin/env python3

"""Flappy Bird, implemented using Pygame."""

import math
import os
from collections import deque
from random import randint

import pygame
from pygame.locals import *


FPS = 60
WIN_WIDTH = 284 * 2     # BG image size: 284x512 px; tiled twice
WIN_HEIGHT = 512

G = 0.002
V_JUMP = 0.5
V_X = 2 * 0.18
DT = 1.5


class Bird(pygame.sprite.Sprite):
    """Represents the bird controlled by the player.

    The bird is the 'hero' of this game.  The player can make it climb
    (ascend quickly), otherwise it sinks (descends more slowly).  It must
    pass through the space in between pipes (for every pipe passed, one
    point is scored); if it crashes into a pipe, the game ends.

    Constants:
    WIDTH: The width, in pixels, of the bird's image.
    HEIGHT: The height, in pixels, of the bird's image.
    """

    WIDTH = HEIGHT = 32

    def __init__(self, x, y, vx, images):
        """Initialise a new Bird instance.
        images: A tuple containing the images used by this bird.  It
            must contain the following images, in the following order:
                0. image of the bird with its wing pointing upward
                1. image of the bird with its wing pointing downward
        """
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.vx = vx
        self.vy = 0
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, dt):
        """Update the bird's position.

        This function uses the cosine function to achieve a smooth climb:
        In the first and last few frames, the bird climbs very little, in the
        middle of the climb, it climbs a lot.
        One complete climb lasts CLIMB_DURATION milliseconds, during which
        the bird ascends with an average speed of CLIMB_SPEED px/ms.
        This Bird's msec_to_climb attribute will automatically be
        decreased accordingly if it was > 0 when this method was called.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        self.y += self.vy * dt
        self.vy = min((self.vy + G * dt), 1)
        

    @property
    def image(self):
        """Get a Surface containing this bird's image.

        This will decide whether to return an image where the bird's
        visible wing is pointing upward or where it is pointing downward
        based on pygame.time.get_ticks().  This will animate the flapping
        bird, even though pygame doesn't support animated GIFs.
        """
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        else:
            return self._img_wingdown

    @property
    def mask(self):
        """Get a bitmask for use in collision detection.

        The bitmask excludes all pixels in self.image with a
        transparency greater than 127."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        """Get the bird's position, width, and height, as a pygame.Rect."""
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    """Represents an obstacle.

    A PipePair has a top and a bottom pipe, and only between them can
    the bird pass -- if it collides with either part, the game is over.

    Attributes:
    x: The PipePair's X position.  This is a float, to make movement
        smoother.  Note that there is no y attribute, as it will only
        ever be 0.
    image: A pygame.Surface which can be blitted to the display surface
        to display the PipePair.
    mask: A bitmask which excludes all pixels in self.image with a
        transparency greater than 127.  This can be used for collision
        detection.
    top_pieces: The number of pieces, including the end piece, in the
        top pipe.
    bottom_pieces: The number of pieces, including the end piece, in
        the bottom pipe.

    Constants:
    WIDTH: The width, in pixels, of a pipe piece.  Because a pipe is
        only one piece wide, this is also the width of a PipePair's
        image.
    PIECE_HEIGHT: The height, in pixels, of a pipe piece.
    ADD_INTERVAL: The interval, in milliseconds, in between adding new
        pipes.
    """

    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 500

    def __init__(self, pipe_end_img, pipe_body_img):
        """Initialises a new random PipePair.

        The new PipePair will automatically be assigned an x attribute of
        float(WIN_WIDTH - 1).

        Arguments:
        pipe_end_img: The image to use to represent a pipe's end piece.
        pipe_body_img: The image to use to represent one horizontal slice
            of a pipe's body.
        """
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()   # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -                  # fill window from top to bottom
             3 * Bird.HEIGHT -             # make room for bird to fit through
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT          # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))
        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        self.Y = (self.top_height_px + WIN_HEIGHT - self.bottom_height_px) / 2

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self):
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, dt):
        self.x -=  V_X * dt

    def collides_with(self, bird):
        """Get whether the bird collides with a pipe in this PipePair.

        Arguments:
        bird: The Bird which should be tested for collision with this
            PipePair.
        """
        return pygame.sprite.collide_mask(self, bird)


def load_images():
    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folder
        (./images/).  All images are converted before being returned to
        speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        file_name = os.path.join('.', 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png')}


def frames_to_msec(frames, fps=FPS):
    return 1000.0 * frames / fps

def msec_to_frames(milliseconds, fps=FPS):
    return fps * milliseconds / 1000.0




class Main:
    bird = 0

    def __init__(self):
        self.Q = [0,0]
        self.lost = 0

    def jump(self):
        self.bird.vy = -V_JUMP

    def change_Qs(self, Q0, Q1):
        self.Q = [Q0, Q1]

    def run(self):


        pygame.init()

        display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Flappy Bird - Python')


        clock = pygame.time.Clock()
        score_font = pygame.font.SysFont(None, 32, bold=False)  # default font
        Q_font = pygame.font.SysFont(None, 30, bold=False)  # default font
        images = load_images()
        max_score = 0
        exit = False
        while not exit:
            self.bird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 0.01,
                        (images['bird-wingup'], images['bird-wingdown']))

            pipes = deque()

            frame_clock = 0  # this counter is only incremented if the game isn't paused
            score = 0
            done = paused = False
            self.lost = 0
            while not done:
                clock.tick(FPS)

                # Handle this 'manually'.  If we used pygame.time.set_timer(),
                # pipe addition would be messed up when paused.
                if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
                    pp = PipePair(images['pipe-end'], images['pipe-body'])
                    pipes.append(pp)

                for e in pygame.event.get():
                    if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        done = exit = True
                        break
                    elif e.type == KEYUP and e.key in (K_PAUSE, K_p):
                        paused = not paused
                    elif e.type == MOUSEBUTTONUP or (e.type == KEYUP and
                            e.key in (K_UP, K_RETURN, K_SPACE)):
                        self.bird.vy = -V_JUMP

                if paused:
                    continue  # don't draw anything

                # check for collisions
                pipe_collision = any(p.collides_with(self.bird) for p in pipes)
                '''if pipe_collision or 0 >= self.bird.y or self.bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                    done = True
                    max_score = max(max_score, score)'''

                if 0 >= self.bird.y:
                    #self.bird.vy = 0
                    self.bird.y = 0
                if self.bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                    #self.bird.vy = 0
                    self.bird.y = WIN_HEIGHT - Bird.HEIGHT



                for x in (0, WIN_WIDTH / 2):
                    display_surface.blit(images['background'], (x, 0))

                while pipes and not pipes[0].visible:
                    pipes.popleft()

                dt = frames_to_msec(1) * DT
                for p in pipes:
                    p.update(dt)
                    display_surface.blit(p.image, p.rect)

                # update and display score
                for p in pipes:
                    if p.x + PipePair.WIDTH < self.bird.x and not p.score_counted:
                        score += 1
                        p.score_counted = True

                score_surface = Q_font.render("Score = %d" % score, True, (255, 255, 255))
                score_x = WIN_WIDTH * 0.25
                display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

                Q3 = Q_font.render("Max   = %d" % max_score, True, (255, 255, 255))
                display_surface.blit(Q3, (score_x, PipePair.PIECE_HEIGHT * 2))

                if len(pipes) > 1:
                    for i, p in enumerate(pipes):
                        if p.score_counted == False:
                            break

                    pipe0_x, pipe0_y = pipes[i].x, WIN_HEIGHT - pipes[i].Y
                    pipe1_y = WIN_HEIGHT - pipes[i + 1].Y

                    Q_x = WIN_WIDTH * 0.65
                    Q1 = Q_font.render("Q[0] = %.2f" % self.Q[0], True, (255, 0, 0))
                    display_surface.blit(Q1, (Q_x, PipePair.PIECE_HEIGHT))
                    Q2 = Q_font.render("Q[1] = %.2f " % self.Q[1], True, (255, 0, 0))
                    display_surface.blit(Q2, (Q_x, PipePair.PIECE_HEIGHT * 2))
                else:
                    self.bird.vy = 0

                self.bird.update(dt)
                display_surface.blit(self.bird.image, self.bird.rect)

                pygame.display.flip()

                frame_clock += 1

                yield pipe_collision,score,self.bird.y,self.bird.x,self.bird.vy,pipes
            #print('Game over! Score: %i' % score)
            self.lost = 1
            #yield score,self.bird.y,self.bird.x,self.bird.vy,pipes
        pygame.quit()

def main():
    obj = Main()
    obj.run()

if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    main()
