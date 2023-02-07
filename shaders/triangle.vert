#version 460
#extension GL_EXT_scalar_block_layout : enable

struct Vertex
{
    vec3 pos;
    vec3 color;
};

vec3 vertices[] = {
    vec3(-0.5, 0.5, 0.0),
    vec3(0.5, 0.5, 0.0),
    vec3(0.0, -0.5, 0.0),
};

layout(set = 0, binding = 0, scalar) readonly buffer vertex_buffer
{
    Vertex verts[];
};

layout(location = 0) out vec3 vertex_color;

layout( push_constant ) uniform constants
{
    float time;
} control;

void main()
{
    mat3 rot = mat3(
        cos(control.time), sin(control.time), 0,
        -sin(control.time), cos(control.time), 0,
        0, 0, 1
    );
    vertex_color = verts[gl_VertexIndex].color;
    gl_Position = vec4(rot * verts[gl_VertexIndex].pos, 1.0);
}