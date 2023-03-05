#version 460
#extension GL_EXT_scalar_block_layout : enable

struct Vertex
{
    vec3 pos;
    vec3 normal;
    vec4 tangent;
    vec3 color;
    vec2 uv0;
    vec2 uv1;
};


layout(set = 0, binding = 0, scalar) readonly buffer vertex_buffer
{
    Vertex verts[];
};

layout(location = 0) out vec3 vertex_color;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv0;

layout( push_constant ) uniform constants
{
    mat4 viewproj;
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
    normal = verts[gl_VertexIndex].normal;
    uv0 = verts[gl_VertexIndex].uv0;
    gl_Position = control.viewproj * vec4(verts[gl_VertexIndex].pos, 1.0);
}