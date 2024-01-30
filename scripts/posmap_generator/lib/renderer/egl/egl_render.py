import numpy as np
from .glcontext import create_opengl_context
from .egl_framework import *

_context_inited = None
import OpenGL.GL as gl

class EGLRender:
    def __init__(self, width=1600, height=1200, name='GL Renderer',
                 program_files=['simple.fs', 'simple.vs']):
        self.width = width
        self.height = height
        self.name = name
        self.use_inverse_depth = False

        self.start = 0

        global _context_inited
        if _context_inited is None:
            create_opengl_context((width, height))
            _context_inited = True

        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glClampColor(gl.GL_CLAMP_READ_COLOR, gl.GL_FALSE)
        gl.glClampColor(gl.GL_CLAMP_FRAGMENT_COLOR, gl.GL_FALSE)
        gl.glClampColor(gl.GL_CLAMP_VERTEX_COLOR, gl.GL_FALSE)

        # init program
        shader_list = []

        for program_file in program_files:
            _, ext = os.path.splitext(program_file)
            if ext == '.vs':
                shader_list.append(loadShader(gl.GL_VERTEX_SHADER, program_file))
            elif ext == '.fs':
                shader_list.append(loadShader(gl.GL_FRAGMENT_SHADER, program_file))
            elif ext == '.gs':
                shader_list.append(loadShader(gl.GL_GEOMETRY_SHADER, program_file))

        self.program = createProgram(shader_list)

        for shader in shader_list:
            gl.glDeleteShader(shader)

        # Init uniform variables
        self.model_mat_unif = gl.glGetUniformLocation(self.program, 'ModelMat')
        self.persp_mat_unif = gl.glGetUniformLocation(self.program, 'PerspMat')

        self.vertex_buffer = gl.glGenBuffers(1)

        # Configure frame buffer
        self.frame_buffer = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer)

        # Configure texture buffer to render to
        self.color_buffer = gl.glGenTextures(1)
        multi_sample_rate = 32
        gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, self.color_buffer)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2DMultisample(gl.GL_TEXTURE_2D_MULTISAMPLE, multi_sample_rate, gl.GL_RGBA32F, self.width, self.height, gl.GL_TRUE)
        gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D_MULTISAMPLE, self.color_buffer, 0)

        self.render_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.render_buffer)
        gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, multi_sample_rate, gl.GL_DEPTH24_STENCIL8, self.width, self.height)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self.render_buffer)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        self.intermediate_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.intermediate_fbo)

        self.screen_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.screen_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, self.width, self.height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.screen_texture, 0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Configure texture buffer if needed
        self.render_texture = None

        # NOTE: original render_texture only support one input
        # this is tentative member of this issue
        self.render_texture_v2 = {}

        # Inner storage for buffer data
        self.vertex_data = None
        self.vertex_dim = None
        self.n_vertices = None

        self.model_view_matrix = None
        self.projection_matrix = None

    def set_mesh(self, vertices, faces):
        self.vertex_data = vertices[faces.reshape([-1])]
        self.vertex_dim = self.vertex_data.shape[1]
        self.n_vertices = self.vertex_data.shape[0]

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertex_data, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def select_array(self, start, size):
        self.start = start
        self.size = size

    def set_viewpoint(self, projection, model_view):
        self.projection_matrix = projection
        self.model_view_matrix = model_view

    def draw_init(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.frame_buffer)
        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        if self.use_inverse_depth:
            gl.glDepthFunc(gl.GL_GREATER)
            gl.glClearDepth(0.0)
        else:
            gl.glDepthFunc(gl.GL_LESS)
            gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def draw_end(self):
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.frame_buffer)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.intermediate_fbo)
        gl.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glClearDepth(1.0)

    def draw(self):
        self.draw_init()

        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.model_mat_unif, 1, gl.GL_FALSE, self.model_view_matrix.transpose())
        gl.glUniformMatrix4fv(self.persp_mat_unif, 1, gl.GL_FALSE, self.projection_matrix.transpose())

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, self.vertex_dim, gl.GL_DOUBLE, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.n_vertices)

        gl.glDisableVertexAttribArray(0)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glUseProgram(0)

        self.draw_end()

    def get_color(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.intermediate_fbo)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        data = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, outputType=None)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        rgb = data.reshape(self.height, self.width, -1)
        rgb = np.flip(rgb, 0)
        return rgb

    def get_z_value(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.intermediate_fbo)
        data = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, outputType=None)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        z = data.reshape(self.height, self.width)
        z = np.flip(z, 0)
        return z

