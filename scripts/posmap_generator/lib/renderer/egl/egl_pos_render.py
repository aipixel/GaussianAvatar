import numpy as np
import random

from .egl_framework import *
from .egl_cam_render import CamRender

from OpenGL.GL import *

class PosRender(CamRender):
    def __init__(self, width=256, height=256, name='Position Renderer'):
        CamRender.__init__(self, width, height, name, program_files=['pos_uv.vs', 'pos_uv.fs'])

    def draw(self):
        self.draw_init()

        glEnable(GL_MULTISAMPLE)

        glUseProgram(self.program)
        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix.transpose())
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix.transpose())

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, self.vertex_dim, GL_DOUBLE, GL_FALSE, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)

        glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

        glDisable(GL_MULTISAMPLE)

        self.draw_end()