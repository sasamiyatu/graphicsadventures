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

struct Camera_Params
{
    mat4 view;
    mat4 projection;
    mat4 viewproj;
    vec3 pos;
};

layout(set = 0, binding = 0, scalar) readonly buffer vertex_buffer
{
    Vertex verts[];
};

layout(set = 0, binding = 7) readonly buffer camera_buffer
{
    Camera_Params camera_params;
};

layout(location = 0) out vec3 vertex_color;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv0;
layout(location = 3) out vec3 pos;

layout( push_constant ) uniform constants
{
    mat4 light_vp;
} control;


void main()
{
    vertex_color = verts[gl_VertexIndex].color;
    pos = verts[gl_VertexIndex].pos;
    normal = verts[gl_VertexIndex].normal;
    uv0 = verts[gl_VertexIndex].uv0;
    gl_Position = camera_params.viewproj * vec4(verts[gl_VertexIndex].pos, 1.0);
}