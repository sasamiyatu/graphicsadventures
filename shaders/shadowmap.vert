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

layout( push_constant ) uniform constants
{
    mat4 viewproj;
} control;

void main()
{
    gl_Position = control.viewproj * vec4(verts[gl_VertexIndex].pos, 1.0);
}