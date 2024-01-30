#version 330

layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec2 a_TextureCoord;

out VertexData {
    vec3 Position;
} VertexOut;

void main()
{ 
    VertexOut.Position = a_Position;
    VertexOut.Texcoord = a_TextureCoord;

    gl_Position = vec4(a_TextureCoord, 0.0, 1.0) - vec4(0.5, 0.5, 0, 0);
    gl_Position[0] *= 2.0;
    gl_Position[1] *= 2.0;
}
