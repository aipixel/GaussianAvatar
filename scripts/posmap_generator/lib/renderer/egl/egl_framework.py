# Mario Rosasco, 2016
# adapted from framework.cpp, Copyright (C) 2010-2012 by Jason L. McKesson
# This file is licensed under the MIT License.
#
# NB: Unlike in the framework.cpp organization, the main loop is contained
# in the tutorial files, not in this framework file. Additionally, a copy of
# this module file must exist in the same directory as the tutorial files
# to be imported properly.

import os



# Function that creates and compiles shaders according to the given type (a GL enum value) and
# shader program (a file containing a GLSL program).
def loadShader(shaderType, shaderFile):
    import OpenGL.GL as gl

    # check if file exists, get full path name
    strFilename = findFileOrThrow(shaderFile)
    shaderData = None
    with open(strFilename, 'r') as f:
        shaderData = f.read()

    shader = gl.glCreateShader(shaderType)

    gl.glShaderSource(shader, shaderData)  # note that this is a simpler function call than in C

    # This shader compilation is more explicit than the one used in
    # framework.cpp, which relies on a glutil wrapper function.
    # This is made explicit here mainly to decrease dependence on pyOpenGL
    # utilities and wrappers, which docs caution may change in future versions.
    gl.glCompileShader(shader)

    status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if status == gl.GL_FALSE:
        # Note that getting the error log is much simpler in Python than in C/C++
        # and does not require explicit handling of the string buffer
        strInfoLog = gl.glGetShaderInfoLog(shader)
        strShaderType = ""
        if shaderType is gl.GL_VERTEX_SHADER:
            strShaderType = "vertex"
        elif shaderType is gl.GL_GEOMETRY_SHADER:
            strShaderType = "geometry"
        elif shaderType is gl.GL_FRAGMENT_SHADER:
            strShaderType = "fragment"

        print("Compilation failure for " + strShaderType + " shader:\n" + str(strInfoLog))

    return shader


# Function that accepts a list of shaders, compiles them, and returns a handle to the compiled program
def createProgram(shaderList):
    import OpenGL.GL as gl

    program = gl.glCreateProgram()

    for shader in shaderList:
        gl.glAttachShader(program, shader)

        gl.glLinkProgram(program)

    status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if status == gl.GL_FALSE:
        # Note that getting the error log is much simpler in Python than in C/C++
        # and does not require explicit handling of the string buffer
        strInfoLog = gl.glGetProgramInfoLog(program)
        print("Linker failure: \n" + str(strInfoLog))

    for shader in shaderList:
        gl.glDetachShader(program, shader)

    return program


# Helper function to locate and open the target file (passed in as a string).
# Returns the full path to the file as a string.
def findFileOrThrow(strBasename):
    # Keep constant names in C-style convention, for readability
    # when comparing to C(/C++) code.
    if os.path.isfile(strBasename):
        return strBasename

    LOCAL_FILE_DIR = "data" + os.sep
    GLOBAL_FILE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep + "../gl/data" + os.sep

    strFilename = LOCAL_FILE_DIR + strBasename
    if os.path.isfile(strFilename):
        return strFilename

    strFilename = GLOBAL_FILE_DIR + strBasename
    if os.path.isfile(strFilename):
        return strFilename

    raise IOError('Could not find target file ' + strBasename)
