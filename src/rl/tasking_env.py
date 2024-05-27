import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dynamics.linear_dynamics import xdot
from quat.quat_helpers import *
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront

class TaskingEnv(gym.Env):
    def __init__(self, field_of_regard=30, dt=0.1, umax = 0.1, x0 = torch.tensor([0, 0, 0, 1, 0, 0, 0])):
        
        self.x = x0

        # Next to set up environment with task density...
        # Start with just one task...
        tasks = [torch.tensor([5, 0])]

        # Define action and observation spaces

    def step(self, action): 
        # Split torques and cameras...
        t, c = torch.split(action, 3, dim=0)
        # observation, reward, terminated, truncated, info
        return self.__get_obs(), self.__get_reward(), False, {}

    def reset(self):
        pass    

    def _get_obs(self):
        pass

    def _get_reward(self):
        pass

    def render(self, mode='human'):
        if mode == 'human':
            self._render_pygame()
    

    def _render_pygame(self):
        pygame.init()
        display = (640, 480)
        screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        gluLookAt(10.0, 10.0, 10.0,  # Camera position
              0.0, 0.0, 0.0,     # Look at this point
              0.0, 1.0, 0.0)     # Up vector
        self.model = pywavefront.Wavefront('../assets/cubesat_model.obj', collect_faces=True)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._setup_lighting()

            self._draw_model()
            # self._draw_points()
            pygame.display.flip()
            pygame.time.wait(10)

        pygame.quit()

    def _draw_model(self):
        q = self.x[3:7]
        rotation_matrix = q2mat(q, homogenous=True)

        glPushMatrix()
        glMultMatrixf(rotation_matrix)

        glColor3f(1, 0.5, 0.31)  # Set a standard material color

        for mesh in self.model.mesh_list:
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for vertex_i in face:
                    glVertex3fv(self.model.vertices[vertex_i])
            glEnd()

        glPopMatrix()

    def _setup_lighting(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        ambient_light = [0.2, 0.2, 0.2, 1.0]
        diffuse_light = [0.4, 0.4, 0.4, 0.5]
        specular_light = [0.5, 0.5, 0.5, 0.5]
        light_position = [7.0, 5.0, 5.0, 1.0]

        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [1.0, 0.5, 0.31, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 10.0)

    def close(self):    
        pass
