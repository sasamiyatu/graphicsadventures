#version 460

layout(location=0) in vec3 vertex_color;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv0;
layout(location=3) in vec3 pos;
layout(location=0) out vec4 out_color;

#define M_PI 3.14159

struct Directional_Light
{
    vec4 dir;
    vec4 color;
};

struct Material_Data
{
    vec4 base_color;
    float roughness;
    float metallic;
    int base_tex;
    int metallic_roughness_tex;
    int normal_tex;
};

struct Camera_Params
{
    mat4 view;
    mat4 projection;
    mat4 viewproj;
    vec3 pos;
};

layout(set = 0, binding = 1) uniform sampler2D base_color_sampler;
layout(set = 0, binding = 2) readonly buffer lights
{
    Directional_Light directional_light;
};
layout(set = 0, binding = 3) uniform sampler2D shadowmap_sampler;
layout(set = 0, binding = 4) uniform sampler2D metallic_roughness_sampler;
layout(set = 0, binding = 5) uniform sampler2D normal_map_sampler;
layout(set = 0, binding = 6) readonly buffer material_buffer
{
    Material_Data[] materials;
};
layout(set = 0, binding = 7) readonly buffer camera_buffer
{
    Camera_Params camera_params;
};

layout( push_constant ) uniform constants
{
    mat4 light_vp;
} control;

vec3 f_schlick(vec3 f0, vec3 f90, float u)
{
    return f0 + (f90 - f0) * pow(1.f - u, 5.f);
}

float v_smith_ggx_correlated(float NoL, float NoV, float alpha)
{
    float a2 = alpha * alpha;
    float lambda_ggxv = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
    float lambda_ggxl = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);

    return 0.5f / (lambda_ggxv + lambda_ggxl);
}

float d_ggx(float NoH, float alpha)
{
    float a2 = alpha * alpha;
    float f = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (M_PI * f * f);
}

void main()
{
    vec4 light_clip_pos = control.light_vp * vec4(pos, 1.0);
    light_clip_pos.xyz /= light_clip_pos.w;

    float shadow_multiplier = 1.0f;
    vec2 light_uv = light_clip_pos.xy * 0.5 + 0.5;
    light_uv.y = 1.0f - light_uv.y;

    vec4 base_color = texture(base_color_sampler, uv0);
    base_color.rgb = pow(base_color.rgb, vec3(2.2));

    vec4 shadow_sample = texture(shadowmap_sampler, light_uv);
    if (shadow_sample.x < light_clip_pos.z)
        shadow_multiplier = 0.0f;

    vec4 sampled = texture(base_color_sampler, uv0);
    vec3 mapped_normal = texture(normal_map_sampler, uv0).rgb;
    vec3 metallic_roughness = texture(metallic_roughness_sampler, uv0).rgb;
    float roughness = metallic_roughness.g * metallic_roughness.g;
    float metallic = metallic_roughness.b;

    vec3 L = normalize(directional_light.dir.xyz);
    vec3 N = normalize(normal);
    vec3 V = normalize(camera_params.pos - pos);
    float NoV = abs(dot(N, V)) + 1e-5f;
    vec3 H = normalize(V + L);
    float NoH = max(0.0, dot(N, H));
    float NoL = max(0.0, dot(N, L));
    float LoH = max(0.0, dot(L, H));

    vec3 ambient = base_color.rgb * 0.01;

    vec3 f0 = mix(vec3(0.04), base_color.rgb, metallic);
    vec3 f90 = vec3(1.0);
    vec3 F = f_schlick(f0, f90, LoH);
    float vis = v_smith_ggx_correlated(NoV, NoL, roughness);
    float D = d_ggx(NoH, roughness);
    vec3 Fr = D * F * vis;
    vec3 kD = vec3(1.0 - F) * (1.0 - metallic);
    vec3 Fd = kD * base_color.rgb / M_PI;

    vec3 final_shading = (Fr + Fd) * NoL * shadow_multiplier + ambient;
    vec3 gamma_corrected = pow(final_shading, vec3(0.4545));
    out_color = vec4(gamma_corrected, sampled.a);
    //out_color = vec4(N * 0.5 + 0.5, 1.0);
    //out_color = vec4(1.0, 0.0, 1.0, 1.0);
}