#version 460

layout(location=0) in vec3 vertex_color;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv0;
layout(location=0) out vec4 out_color;

layout(set = 0, binding = 1) uniform sampler2D base_color_sampler;

void main()
{
    vec4 sampled = texture(base_color_sampler, uv0);
    if (sampled.a < 0.5) discard;
    vec3 N = normalize(normal);
    out_color = vec4(sampled.rgb, 1.0);
}