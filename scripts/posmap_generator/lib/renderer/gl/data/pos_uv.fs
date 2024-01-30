#version 410

in VertexData {
    vec3 Position;
} VertexIn;

//in int gl_PrimitiveID;

layout (location = 0) out vec4 FragPosition;

void main()
{
    FragPosition = vec4(VertexIn.Position,1.0+float(gl_PrimitiveID));
}