import numpy as np

from .framework import *
from .cam_render import CamRender


class PosRender(CamRender):
    def __init__(self, width=256, height=256, name='Position Renderer'):
        CamRender.__init__(self, width, height, name, program_files=['pos_uv.vs', 'pos_uv.fs'])

        self.uv_buffer = glGenBuffers(1)
        self.uv_data = None

    def set_mesh(self, vertices, faces, uvs, faces_uv):
        self.vertex_data = vertices[faces.reshape([-1])]
        self.vertex_dim = self.vertex_data.shape[1]
        self.n_vertices = self.vertex_data.shape[0]

        self.uv_data = uvs[faces_uv.reshape([-1])]

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.uv_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.uv_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self):
        self.draw_init()

        glUseProgram(self.program)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, self.vertex_dim, GL_DOUBLE, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.uv_buffer)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_DOUBLE, GL_FALSE, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)

        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

        self.draw_end()