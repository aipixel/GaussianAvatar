from .egl_render import EGLRender


class CamRender(EGLRender):
    def __init__(self, width=1600, height=1200, name='Cam Renderer',
                 program_files=['simple.fs', 'simple.vs']):
        EGLRender.__init__(self, width, height, name, program_files)
        self.camera = None

    def set_camera(self, camera):
        self.camera = camera
        self.projection_matrix, self.model_view_matrix = camera.get_gl_matrix()
