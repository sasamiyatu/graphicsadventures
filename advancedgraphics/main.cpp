#include "SDL2/SDL.h"
#include <stdio.h>
#include <stdlib.h>
#define VOLK_IMPLEMENTATION
#include "volk/volk.h"
#include "SDL2/SDL_vulkan.h"
#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"
#include <assert.h>
#include <vector>
#include "windows.h"
#include "glm/glm.hpp"
#define CGLTF_IMPLEMENTATION
#include "cgltf/cgltf.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_PERLIN_IMPLEMENTATION
#include "stb/stb_perlin.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define PREFERRED_DEVICE_INDEX 0 // Hack to pick the correct GPU on my machine
#define ENABLE_VALIDATION

#define VK_CHECK(expression)        \
    do                              \
    {                               \
        VkResult res = expression;  \
        assert(res == VK_SUCCESS);  \
    } while (0);

static const int SCREEN_FULLSCREEN = 0;
static const int SCREEN_WIDTH = 1280;
static const int SCREEN_HEIGHT = 720;
SDL_Window* window = nullptr;

static void sdl_die(const char* message) {
	fprintf(stderr, "%s: %s\n", message, SDL_GetError());
	exit(2);
}

void init_screen(const char* caption) {
	// To be filled in the next section
    // Initialize SDL 

    // Create the window
    if (SCREEN_FULLSCREEN) {
        window = SDL_CreateWindow(
            caption,
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            0, 0, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_VULKAN
        );
    }
    else {
        window = SDL_CreateWindow(
            caption,
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_VULKAN
        );
    }
    if (window == NULL) sdl_die("Couldn't set video mode");


    int w, h;
    SDL_GetWindowSize(window, &w, &h);
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    const char* type = (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" :
        (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ? "WARNING" :
        "INFO";
    printf("%s: %s\n", type, pCallbackData->pMessage);
#ifndef DEBUG_VERBOSE
    assert(false);
#endif
    return VK_FALSE;
}

VkDebugUtilsMessengerEXT register_debug_callback(VkInstance instance)
{
    VkDebugUtilsMessengerEXT debug_messenger = 0;
    VkDebugUtilsMessengerCreateInfoEXT ci{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
#ifdef DEBUG_VERBOSE
    ci.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
#endif
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;// | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debug_callback;
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &ci, nullptr, &debug_messenger));

    return debug_messenger;
}

VkInstance create_instance()
{
    VkInstanceCreateInfo info{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    VkApplicationInfo app_info{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app_info.apiVersion = VK_API_VERSION_1_3;
    app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    app_info.engineVersion = VK_MAKE_VERSION(0, 0, 1);
    app_info.pApplicationName = "Vulkan app";
    app_info.pEngineName = "GigaTek";
    info.pApplicationInfo = &app_info;

    init_screen("Vulkan");

    std::vector<const char*> extensions = {};
    u32 count;
    SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr);
    extensions.resize(count);
    SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data());
    extensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);

#ifdef ENABLE_VALIDATION 
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    std::vector<const char*> validation_layers = {};

#ifdef _DEBUG
    validation_layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    info.enabledExtensionCount = (u32)extensions.size();
    info.ppEnabledExtensionNames = extensions.data();
    info.enabledLayerCount = (u32)validation_layers.size();
    info.ppEnabledLayerNames = validation_layers.data();

    VkInstance instance = 0;
    VK_CHECK(vkCreateInstance(&info, nullptr, &instance));

    volkLoadInstanceOnly(instance);

    return instance;
}

VkPhysicalDevice pick_physical_device(VkInstance instance)
{
    u32 count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());


    assert(count != 0);
    for (u32 i = 0; i < count; ++i)
    {
        VkPhysicalDeviceProperties2 props{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        vkGetPhysicalDeviceProperties2(devices[i], &props);
        if (props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
#ifdef PREFERRED_DEVICE_INDEX
            if (i != PREFERRED_DEVICE_INDEX) continue;
#endif
            printf("Selecting GPU %s\n", props.properties.deviceName);
            return devices[i];
        }
    }

    if (count > 0)
    {
        VkPhysicalDeviceProperties props{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        vkGetPhysicalDeviceProperties(devices[0], &props);
        {
            printf("Selecting GPU %s\n", props.deviceName);
            return devices[0];
        }
    }

    return 0;
}

VkSurfaceKHR create_surface(VkInstance instance)
{
    VkSurfaceKHR surface = 0;
    SDL_Vulkan_CreateSurface(window, instance, &surface);
    return surface;
}

VkDevice create_device(VkPhysicalDevice physical_device)
{
    VkDevice device = 0;
    VkDeviceCreateInfo info{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };

    float prio = 1.0f;
    VkDeviceQueueCreateInfo queue_info{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queue_info.pQueuePriorities = &prio;
    queue_info.queueCount = 1;
    queue_info.queueFamilyIndex = 0; //FIXME

    info.pQueueCreateInfos = &queue_info;
    info.queueCreateInfoCount = 1;


    std::vector<const char*> extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
    info.ppEnabledExtensionNames = extensions.data();
    info.enabledExtensionCount = (u32)extensions.size();
    
    VkPhysicalDeviceVulkan12Features vulkan_12_feats = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    vulkan_12_feats.scalarBlockLayout = true;
    VkPhysicalDeviceVulkan13Features vulkan_13_feats = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    vulkan_13_feats.synchronization2 = true;
    vulkan_13_feats.dynamicRendering = true;
    vulkan_13_feats.pNext = &vulkan_12_feats;
    info.pNext = &vulkan_13_feats;

    vkCreateDevice(physical_device, &info, nullptr, &device);

    volkLoadDevice(device);

    return device;
}

VkSurfaceFormat2KHR get_swapchain_format(VkPhysicalDevice physical_device, VkSurfaceKHR surface)
{
    VkPhysicalDeviceSurfaceInfo2KHR surface_info{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR };
    surface_info.surface = surface;
    u32 format_count = 0;
    vkGetPhysicalDeviceSurfaceFormats2KHR(physical_device, &surface_info, &format_count, nullptr);
    std::vector<VkSurfaceFormat2KHR> formats(format_count);
    for (auto& f : formats) f.sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR;
    vkGetPhysicalDeviceSurfaceFormats2KHR(physical_device, &surface_info, &format_count, formats.data());

    for (const auto& f : formats)
    {
        if (f.surfaceFormat.format == VK_FORMAT_R8G8B8A8_UNORM || f.surfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
            return f;
    }

    return formats[0];
}

VkSwapchainKHR create_swapchain(VkDevice device, VkPhysicalDevice physical_device, VkSurfaceKHR surface, VkFormat* swapchain_format)
{
    VkSwapchainCreateInfoKHR create_info{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    VkSurfaceFormat2KHR format = get_swapchain_format(physical_device, surface);

    create_info.minImageCount = 3;
    create_info.imageFormat = format.surfaceFormat.format;
    create_info.imageColorSpace = format.surfaceFormat.colorSpace;
    int w, h;
    SDL_Vulkan_GetDrawableSize(window, &w, &h);
    create_info.imageExtent = { (u32)w, (u32)h };
    create_info.surface = surface;
    create_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 1;
    u32 queue_indices = 0;
    create_info.pQueueFamilyIndices = &queue_indices;

    VkSwapchainKHR swapchain = 0;
    VK_CHECK(vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain));

    *swapchain_format = create_info.imageFormat;

    return swapchain;
}

VkSemaphore create_semaphore(VkDevice device)
{
    VkSemaphore semaphore = 0;
    VkSemaphoreCreateInfo info{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
  
    vkCreateSemaphore(device, &info, nullptr, &semaphore);

    return semaphore;
}

VkCommandPool create_command_pool(VkDevice device)
{
    VkCommandPoolCreateInfo info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    info.queueFamilyIndex = 0;
    VkCommandPool command_pool = 0;
    vkCreateCommandPool(device, &info, nullptr, &command_pool);
    return command_pool;
}

void allocate_command_buffers(VkDevice device, VkCommandPool pool, u32 count, VkCommandBuffer* command_buffers)
{
    VkCommandBufferAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    alloc_info.commandBufferCount = count;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    vkAllocateCommandBuffers(device, &alloc_info, command_buffers);
}

int read_entire_file(const char* path, u8** data)
{
    // FIXME: Make this work on other platforms
    HANDLE f = CreateFileA(path, GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0);
    DWORD error = GetLastError();
    assert(error == 0);
    DWORD filesize = GetFileSize(f, 0);
    u8* buffer = (u8*)malloc(filesize);
    DWORD bytes_read = 0;
    ReadFile(f, buffer, filesize, &bytes_read, 0);
    assert(bytes_read == filesize);

    *data = buffer;
    return bytes_read;
}

VkShaderModule load_shader_module(VkDevice device, const char* path)
{
    u8* data = 0;
    int filesize = read_entire_file(path, &data);

    VkShaderModuleCreateInfo info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = filesize;
    info.pCode = (u32*)data;
    VkShaderModule module = 0;
    vkCreateShaderModule(device, &info, nullptr, &module);

    free(data);

    return module;
}

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec3 color;
    glm::vec2 uv0;
    glm::vec2 uv1;
};

VkPipelineLayout create_pipeline_layout(VkDevice device, const std::vector<VkDescriptorSetLayout>& layouts)
{
    VkPipelineLayoutCreateInfo info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    info.pSetLayouts = layouts.data();
    info.setLayoutCount = (u32)layouts.size();
    VkPushConstantRange pc_range = {};
    pc_range.offset = 0;
    pc_range.size = 128;
    pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    info.pPushConstantRanges = &pc_range;
    info.pushConstantRangeCount = 1;
    VkPipelineLayout layout = 0;
    vkCreatePipelineLayout(device, &info, 0, &layout);

    return layout;
}

VkPipeline create_graphics_pipeline(VkDevice device, VkShaderModule vert_shader, VkShaderModule frag_shader, VkPipelineLayout layout, VkFormat swapchain_format)
{

    VkGraphicsPipelineCreateInfo info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].module = vert_shader;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].module = frag_shader;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo input_assembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport_state = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo raster_state = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster_state.cullMode = VK_CULL_MODE_NONE;
    raster_state.polygonMode = VK_POLYGON_MODE_FILL;
    raster_state.lineWidth = 1.0f;


    VkPipelineMultisampleStateCreateInfo multisample_state = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth_stencil_state.depthTestEnable = VK_TRUE;
    depth_stencil_state.depthWriteEnable = VK_TRUE;
    depth_stencil_state.minDepthBounds = 0.f;
    depth_stencil_state.maxDepthBounds = 1.0f;
    depth_stencil_state.depthCompareOp = VK_COMPARE_OP_LESS;
    
    VkPipelineColorBlendStateCreateInfo blend_state = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };

    VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamic_state = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_state.dynamicStateCount = (u32)std::size(dynamic_states);
    dynamic_state.pDynamicStates = dynamic_states;

    VkPipelineRenderingCreateInfo rendering_create_info{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    //rendering_create_info.viewMask = 0;
    //rendering_create_info.colorAttachmentCount = 1;
    //rendering_create_info.pColorAttachmentFormats = &swapchain_format;
    rendering_create_info.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    info.pNext = &rendering_create_info;
    info.stageCount = (u32)std::size(stages);
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pDepthStencilState = &depth_stencil_state;
    info.pMultisampleState = &multisample_state;
    info.pViewportState = &viewport_state;
    info.pDynamicState = &dynamic_state;
    info.pInputAssemblyState = &input_assembly;
    info.layout = layout;
    info.pRasterizationState = &raster_state;
    VkPipeline pipeline = 0;
    vkCreateGraphicsPipelines(device, 0, 1, &info, nullptr, &pipeline);

    return pipeline;
}

VmaAllocator create_allocator(VkDevice device, VkPhysicalDevice physical_device, VkInstance instance)
{
    VmaAllocator allocator = 0;
    VmaAllocatorCreateInfo info = {}; 
    info.device = device;
    info.instance = instance;
    info.physicalDevice = physical_device;
    VmaVulkanFunctions funcs = {};
    funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    info.pVulkanFunctions = &funcs;
    vmaCreateAllocator(&info, &allocator);

    return allocator;
}

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device)
{
    VkDescriptorSetLayout set_layout;
    VkDescriptorSetLayoutCreateInfo info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    info.bindingCount = (u32)std::size(bindings);
    info.pBindings = bindings;
    vkCreateDescriptorSetLayout(device, &info, nullptr, &set_layout);

    return set_layout;
}

struct Buffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
};

struct Image
{
    VkImage image;
    VkImageView image_view;
    VkImageLayout layout;
    VmaAllocation allocation;
};

struct Texture
{
    Image image;
    std::string name;
};

struct Material
{
    glm::vec4 base_color = glm::vec4(0.5f, 0.5f, 0.5f, 1.f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    int base_tex = -1;
    int metallic_roughness_tex = -1;
    int normal_tex = -1;
};

struct Mesh_Primitive
{
    std::vector<Vertex> vertices;
    std::vector<u32> indices;

    Buffer vertex_buffer;
    Buffer index_buffer;

    int material = -1;
};

struct Mesh
{
    std::vector<Mesh_Primitive> primitives;
};



Image allocate_image(VmaAllocator allocator, VkDevice device, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    VkImageCreateInfo cinfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    cinfo.arrayLayers = 1;
    cinfo.extent = extent;
    cinfo.format = format;
    cinfo.imageType = extent.depth == 1 ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_3D;
    cinfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    cinfo.mipLevels = 1;
    cinfo.samples = VK_SAMPLE_COUNT_1_BIT;
    cinfo.usage = usage;
    cinfo.tiling = VK_IMAGE_TILING_OPTIMAL;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    Image img;

    vmaCreateImage(allocator, &cinfo, &alloc_info, &img.image, &img.allocation, nullptr);

    VkImageViewCreateInfo view_info{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    view_info.format = cinfo.format;
    view_info.image = img.image;
    view_info.viewType = extent.depth == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.subresourceRange.aspectMask = aspect;

    vkCreateImageView(device, &view_info, nullptr, &img.image_view);

    return img;
}

VkImageMemoryBarrier2 image_memory_barrier2(
    VkImage image, 
    VkImageAspectFlags aspect,
    VkImageLayout old_layout, 
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage_mask,
    VkPipelineStageFlags2 dst_stage_mask)
{
    VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcStageMask = src_stage_mask;
    barrier.dstStageMask = dst_stage_mask;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

    return barrier;
}

VkDependencyInfo dependency_info(
    u32 memory_barrier_count, VkMemoryBarrier2* memory_barriers, 
    u32 buffer_memory_barrier_count, VkBufferMemoryBarrier2* buffer_memory_barriers,
    u32 image_memory_barrier_count, VkImageMemoryBarrier2* image_memory_barriers)
{
    VkDependencyInfo dep_info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep_info.memoryBarrierCount = memory_barrier_count;
    dep_info.pMemoryBarriers = memory_barriers;
    dep_info.bufferMemoryBarrierCount = buffer_memory_barrier_count;
    dep_info.pBufferMemoryBarriers = buffer_memory_barriers;
    dep_info.imageMemoryBarrierCount = image_memory_barrier_count;
    dep_info.pImageMemoryBarriers = image_memory_barriers;

    return dep_info;
}

Buffer allocate_buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage, VmaAllocationCreateFlags vma_flags)
{
    VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    buffer_info.size = size;
    buffer_info.usage = usage;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = memory_usage;
    alloc_info.flags = vma_flags;

    Buffer buffer;
    vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer.buffer, &buffer.allocation, nullptr);


    return buffer;
}

void allocate_command_buffers(VkDevice device, u32 command_buffer_count, VkCommandPool command_pool, VkCommandBufferLevel level, VkCommandBuffer* out_cmd)
{
    VkCommandBufferAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc_info.commandBufferCount = command_buffer_count;
    alloc_info.commandPool = command_pool;
    alloc_info.level = level;
    vkAllocateCommandBuffers(device, &alloc_info, out_cmd);
}

void begin_command_buffer(VkCommandBuffer cmd)
{
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &begin_info);
}

Texture create_default_texture(VmaAllocator allocator, VkDevice device, VkCommandPool command_pool, VkQueue queue)
{
    constexpr u32 default_tex_size = 256;
    constexpr VkFormat default_format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr u32 n_comps = 4;
    Image img = allocate_image(allocator, device, { default_tex_size, default_tex_size, 1 }, default_format, 
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkDeviceSize buffer_size = default_tex_size * default_tex_size * n_comps;

    Buffer staging_buffer = allocate_buffer(
        allocator,
        buffer_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    void* mapped;
    vmaMapMemory(allocator, staging_buffer.allocation, &mapped);
    u8* texture_data = (u8*)mapped;

    for (int y = 0; y < default_tex_size; ++y)
    {
        for (int x = 0; x < default_tex_size; ++x)
        {
            int v = (x ^ y) & 0xFF;
            texture_data[(y * default_tex_size + x) * 4 + 0] = (u8)v;
            texture_data[(y * default_tex_size + x) * 4 + 1] = (u8)v;
            texture_data[(y * default_tex_size + x) * 4 + 2] = (u8)v;
            texture_data[(y * default_tex_size + x) * 4 + 3] = 255;
        }
    }
    vmaUnmapMemory(allocator, staging_buffer.allocation);

    VkCommandBuffer cmd = 0;
    allocate_command_buffers(device, command_pool, 1, &cmd);
    
    begin_command_buffer(cmd);

    {
        VkImageMemoryBarrier2 barrier = image_memory_barrier2(
            img.image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
        );

        VkDependencyInfo dep_info = dependency_info(0, 0, 0, 0, 1, &barrier);
        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    VkImageSubresourceLayers subres{};
    subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.baseArrayLayer = 0;
    subres.layerCount = 1;
    subres.mipLevel = 0;
    VkBufferImageCopy regions{};
    regions.bufferOffset = 0;
    regions.bufferImageHeight = 0;
    regions.bufferRowLength = 0;
    regions.imageSubresource = subres;
    regions.imageOffset = { 0, 0, 0 };
    regions.imageExtent = { (u32)default_tex_size, (u32)default_tex_size, 1 };

    vkCmdCopyBufferToImage(cmd, staging_buffer.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &regions);

    {
        VkImageMemoryBarrier2 barrier = image_memory_barrier2(
            img.image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
        );

        VkDependencyInfo dep_info = dependency_info(0, 0, 0, 0, 1, &barrier);
        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submit, 0);

    vkQueueWaitIdle(queue);

    vmaFreeMemory(allocator, staging_buffer.allocation);
    vkDestroyBuffer(device, staging_buffer.buffer, nullptr);

    Texture t;
    t.image = img;
    t.name = "default";

    return t;
}

Texture load_texture(const char* path)
{
    return Texture();
}

VkSampler create_sampler(VkDevice device)
{
    VkSampler sampler = 0;
    VkSamplerCreateInfo create_info = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    vkCreateSampler(device, &create_info, nullptr, &sampler);

    return sampler;
}

u32 get_graphics_queue_family(VkPhysicalDevice physical_device)
{
    u32 prop_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &prop_count, nullptr);
    std::vector<VkQueueFamilyProperties> props(prop_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &prop_count, props.data());
    
    u32 i = 0;
    for (const auto& prop : props)
    {
        if (prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            return i;
        ++i;
    }

    printf("No graphics queue family??");

    return 0;
}

void free_mesh(VkDevice device, VmaAllocator allocator, Mesh* mesh)
{
    for (auto& prim : mesh->primitives)
    {
        vmaFreeMemory(allocator, prim.vertex_buffer.allocation);
        vmaFreeMemory(allocator, prim.index_buffer.allocation);
        vkDestroyBuffer(device, prim.vertex_buffer.buffer, nullptr);
        vkDestroyBuffer(device, prim.index_buffer.buffer, nullptr);
    }
    mesh->primitives.clear();
}

void destroy_image(VkDevice device, VmaAllocator allocator, Image image)
{
    vmaFreeMemory(allocator, image.allocation);
    vkDestroyImageView(device, image.image_view, nullptr);
    vkDestroyImage(device, image.image, nullptr);
}

void load_gltf(VmaAllocator allocator, VkDevice device, const char* path, Mesh* mesh, std::vector<Texture>& textures, std::vector<Material>& materials)
{
    cgltf_options options = {};
    cgltf_data* data = NULL;
    cgltf_result result = cgltf_parse_file(&options, path, &data);
    assert(result == cgltf_result_success);
    result = cgltf_load_buffers(&options, data, path);
    assert(result == cgltf_result_success);
    std::string base_path = path;
    for (int i = (int)base_path.length(); i >= 0; --i)
    {
        if (base_path[i] == '/')
        {
            base_path = base_path.substr(0, i + 1);
            break;
        }
        else if (i == 0)
        {
            base_path = "";
        }
    }

    // Load all images
    VkCommandPool command_pool = create_command_pool(device);
    VkCommandBuffer cmd = 0;
    allocate_command_buffers(device, command_pool, 1, &cmd);
    begin_command_buffer(cmd);
    textures.reserve(data->images_count);
    std::vector<Buffer> buffers_to_free(data->images_count);
    for (int i = 0; i < data->images_count; ++i)
    {
        int w, h, c;
        u8* image_data = stbi_load((base_path + std::string(data->images[i].uri)).c_str(), &w, &h, &c, 4);
        assert(image_data);

        u32 required_size = w * h * 4;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        Image img = allocate_image(allocator, device, { (u32)w, (u32)h, 1 }, format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

        Buffer staging_buffer = allocate_buffer(allocator, required_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        buffers_to_free[i] = staging_buffer;

        void* mapped;
        vmaMapMemory(allocator, staging_buffer.allocation, &mapped);
        memcpy(mapped, image_data, required_size);
        vmaUnmapMemory(allocator, staging_buffer.allocation);

        stbi_image_free(image_data);

        {
            VkImageMemoryBarrier2 barrier = image_memory_barrier2(img.image, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT
            );
            VkDependencyInfo dep_info = dependency_info(0, nullptr, 0, nullptr, 1, &barrier);
            vkCmdPipelineBarrier2(cmd, &dep_info);
        }

        VkBufferImageCopy2 img_copy = { VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2 };
        img_copy.bufferRowLength = 0;
        img_copy.bufferImageHeight = 0;
        img_copy.bufferOffset = 0;
        img_copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        img_copy.imageOffset = { 0, 0, 0 };
        img_copy.imageExtent = { (u32)w, (u32)h, 1 };

        VkCopyBufferToImageInfo2 copy = { VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2 };
        copy.dstImage = img.image;
        copy.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copy.srcBuffer = staging_buffer.buffer;
        copy.regionCount = 1;
        copy.pRegions = &img_copy;

        vkCmdCopyBufferToImage2(cmd, &copy);

        {
            VkImageMemoryBarrier2 barrier = image_memory_barrier2(img.image, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            );
            VkDependencyInfo dep_info = dependency_info(0, nullptr, 0, nullptr, 1, &barrier);
            vkCmdPipelineBarrier2(cmd, &dep_info);
        }

        Texture new_tex;
        new_tex.image = img;
        new_tex.name = data->images[i].uri;
        textures.push_back(new_tex);
    }

    vkEndCommandBuffer(cmd);

    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);
    VkSubmitInfo2 submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    VkCommandBufferSubmitInfo submit_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    submit_info.commandBuffer = cmd;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &submit_info;
    vkQueueSubmit2(queue, 1, &submit, 0);

    vkDeviceWaitIdle(device);
    vkDestroyCommandPool(device, command_pool, nullptr);

    materials.reserve(data->materials_count);
    for (int i = 0; i < data->materials_count; ++i)
    {
        Material new_mat = {};
        cgltf_material* mat = &data->materials[i];
        assert(mat->has_pbr_metallic_roughness);
        new_mat.base_color = glm::make_vec4(mat->pbr_metallic_roughness.base_color_factor);
        new_mat.metallic = mat->pbr_metallic_roughness.metallic_factor;
        new_mat.roughness = mat->pbr_metallic_roughness.roughness_factor;
        new_mat.normal_tex = mat->normal_texture.texture ? int(mat->normal_texture.texture->image - data->images) : -1;
        new_mat.base_tex = mat->pbr_metallic_roughness.base_color_texture.texture ? (int)(mat->pbr_metallic_roughness.base_color_texture.texture->image - data->images) : -1;
        new_mat.metallic_roughness_tex = mat->pbr_metallic_roughness.metallic_roughness_texture.texture ? (int)(mat->pbr_metallic_roughness.metallic_roughness_texture.texture->image - data->images) : -1;
        materials.push_back(new_mat);
    }

    for (auto& buf : buffers_to_free)
    {
        vmaFreeMemory(allocator, buf.allocation);
        vkDestroyBuffer(device, buf.buffer, nullptr);
    }

    assert(data->meshes_count == 1);
    for (u32 m = 0; m < data->meshes_count; ++m)
    {
        for (u32 p = 0; p < data->meshes[m].primitives_count; ++p)
        {
            const cgltf_primitive* prim = &data->meshes[m].primitives[p];

            Mesh_Primitive new_prim;
            new_prim.indices.resize(prim->indices->count);
            new_prim.vertices.resize(prim->attributes[0].data->count);
            new_prim.material = (int)(prim->material - data->materials);

            for (int i = 0; i < prim->indices->count; ++i)
            {
                u8* src = (u8*)prim->indices->buffer_view->buffer->data +
                    (u32)(prim->indices->offset + prim->indices->buffer_view->offset);
                if (prim->indices->component_type == cgltf_component_type_r_16u)
                {
                    u16 index = *(u16*)(src + prim->indices->stride * i);
                    u32 index32 = u32(index);
                    new_prim.indices[i] = index32;
                }
                else
                {
                    u16 index = *(u32*)(src + prim->indices->stride * i);
                    new_prim.indices[i] = index;
                }
            }
            for (u32 a = 0; a < prim->attributes_count; ++a)
            {
                cgltf_attribute* attrib = &prim->attributes[a];
                switch (attrib->type)
                {
                case cgltf_attribute_type_position:
                    assert(prim->attributes[a].data->stride == 12);

                    for (int v = 0; v < attrib->data->count; ++v)
                    {
                        u8* src = (u8*)attrib->data->buffer_view->buffer->data + attrib->data->offset + attrib->data->buffer_view->offset + prim->attributes[a].data->stride * v;
                        new_prim.vertices[v].pos = *(glm::vec3*)src;
                    }
                    break;
                case cgltf_attribute_type_normal:
                    assert(prim->attributes[a].data->stride == 12);
                    for (int v = 0; v < attrib->data->count; ++v)
                    {
                        u8* src = (u8*)attrib->data->buffer_view->buffer->data + attrib->data->offset + attrib->data->buffer_view->offset + prim->attributes[a].data->stride * v;
                        new_prim.vertices[v].normal = *(glm::vec3*)src;
                    }
                    break;
                case cgltf_attribute_type_tangent:
                    assert(prim->attributes[a].data->stride == 16);
                    for (int v = 0; v < attrib->data->count; ++v)
                    {
                        u8* src = (u8*)attrib->data->buffer_view->buffer->data + attrib->data->offset + attrib->data->buffer_view->offset + prim->attributes[a].data->stride * v;
                        new_prim.vertices[v].tangent = *(glm::vec4*)src;
                    }
                    break;
                case cgltf_attribute_type_texcoord:
                    assert(prim->attributes[a].data->stride == 8);
                    for (int v = 0; v < attrib->data->count; ++v)
                    {
                        u8* src = (u8*)attrib->data->buffer_view->buffer->data + attrib->data->offset + attrib->data->buffer_view->offset + prim->attributes[a].data->stride * v;
                        new_prim.vertices[v].uv0 = *(glm::vec2*)src;
                    }
                    break;
                default:
                    break;
                }
            }

            mesh->primitives.emplace_back(std::move(new_prim));
        }
    }

    cgltf_free(data);
}

int main(int argc, char** argv)
{
    u64 pfreq = SDL_GetPerformanceFrequency();
    double inv_pfreq = 1.0 / (double)pfreq;

    VK_CHECK(volkInitialize());

    VkInstance instance = create_instance();
    VkDebugUtilsMessengerEXT debug_messenger = register_debug_callback(instance);

    VkPhysicalDevice physical_device = pick_physical_device(instance);
    VkDevice device = create_device(physical_device);
    VkSurfaceKHR surface = create_surface(instance);
    VkFormat swapchain_format = {};
    VkSwapchainKHR swapchain = create_swapchain(device, physical_device, surface, &swapchain_format);
    VkSemaphore acquire_semaphore = create_semaphore(device);
    VkSemaphore release_semaphore = create_semaphore(device);

    VkDeviceQueueInfo2 queue_info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2 };
    queue_info.queueFamilyIndex = get_graphics_queue_family(physical_device);;
    queue_info.queueIndex = 0;
    VkQueue queue = 0;
    vkGetDeviceQueue2(device, &queue_info, &queue);

    u32 image_count = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
    std::vector<VkImage> swapchain_images(image_count);
    vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());

    std::vector<VkImageView> swapchain_image_views(image_count);
    for (u32 i = 0; i < image_count; ++i)
    {
        VkImageViewCreateInfo info = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        info.format = swapchain_format;
        info.image = swapchain_images[i];
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
        VkImageSubresourceRange range = {};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseArrayLayer = 0;
        range.baseMipLevel = 0;
        range.layerCount = 1;
        range.levelCount = 1;
        info.subresourceRange = range;
        vkCreateImageView(device, &info, nullptr, &swapchain_image_views[i]);
    }

    VkCommandPool command_pool = create_command_pool(device);

    VkCommandBuffer command_buffer = 0;
    allocate_command_buffers(device, command_pool, 1, &command_buffer);

    VkShaderModule vertex = load_shader_module(device, "../shaders/spirv/triangle.vert.spv");
    VkShaderModule fragment = load_shader_module(device, "../shaders/spirv/triangle.frag.spv");

    std::vector<VkDescriptorSetLayout> layouts = { create_descriptor_set_layout(device)};
    VkPipelineLayout layout = create_pipeline_layout(device, layouts);

    VkPipeline pipeline = create_graphics_pipeline(device, vertex, fragment, layout, swapchain_format);

    vkDestroyShaderModule(device, vertex, nullptr);
    vkDestroyShaderModule(device, fragment, nullptr);

    VmaAllocator allocator = create_allocator(device, physical_device, instance);

    VkSampler sampler = create_sampler(device);

    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 512;
    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1024;
    VkDescriptorPool descriptor_pool = 0;
    VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool));


    glm::mat4 proj = glm::perspective(glm::radians(60.f), (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 1.0f, 2000.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(3.0f, 100.f, 0.f), glm::vec3(0.0f, 100.f, 0.f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 viewproj = proj * view;

    SDL_Event event;
    bool quit = false;

    Mesh mesh;
    std::vector<Texture> textures;
    std::vector<Material> materials;
    load_gltf(allocator, device, "../data/Sponza/glTF/Sponza.gltf", &mesh, textures, materials);

    for (int i = 0; i < mesh.primitives.size(); ++i)
    {
        {
            VkDeviceSize size = sizeof(Vertex) * mesh.primitives[i].vertices.size();
            mesh.primitives[i].vertex_buffer = allocate_buffer(allocator, size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

            void* mapped = 0;
            vmaMapMemory(allocator, mesh.primitives[i].vertex_buffer.allocation, &mapped);
            memcpy(mapped, mesh.primitives[i].vertices.data(), size);
            vmaUnmapMemory(allocator, mesh.primitives[i].vertex_buffer.allocation);
        }
        {
            VkDeviceSize size = sizeof(u32) * mesh.primitives[i].indices.size();
            mesh.primitives[i].index_buffer = allocate_buffer(allocator, size,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

            void* mapped = 0;
            vmaMapMemory(allocator, mesh.primitives[i].index_buffer.allocation, &mapped);
            memcpy(mapped, mesh.primitives[i].indices.data(), size);
            vmaUnmapMemory(allocator, mesh.primitives[i].index_buffer.allocation);
        }
    }

    std::vector<VkDescriptorSet> mesh_desc_sets(mesh.primitives.size());
    {
        VkDescriptorSetAllocateInfo desc_alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        desc_alloc_info.descriptorPool = descriptor_pool;
        desc_alloc_info.descriptorSetCount = (u32)mesh.primitives.size();
        std::vector<VkDescriptorSetLayout> layouts_(mesh.primitives.size());
        for (int i = 0; i < mesh.primitives.size(); ++i)
            layouts_[i] = layouts[0];
        desc_alloc_info.pSetLayouts = layouts_.data();;
        vkAllocateDescriptorSets(device, &desc_alloc_info, mesh_desc_sets.data());
    }

    Texture default_texture = create_default_texture(allocator, device, command_pool, queue);

    for (int i = 0; i < mesh.primitives.size(); ++i)
    {
        VkWriteDescriptorSet writes[2] = {};

        VkDescriptorBufferInfo desc_buffer_info = {};
        desc_buffer_info.buffer = mesh.primitives[i].vertex_buffer.buffer;
        desc_buffer_info.offset = 0;
        desc_buffer_info.range = VK_WHOLE_SIZE;

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].dstSet = mesh_desc_sets[i];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].pBufferInfo = &desc_buffer_info;

        VkDescriptorImageInfo image_info = {};
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        //image_info.imageView = default_texture.image.image_view;
        assert(mesh.primitives[i].material >= 0);
        image_info.imageView = textures[materials[mesh.primitives[i].material].base_tex].image.image_view;
        image_info.sampler = sampler;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].dstSet = mesh_desc_sets[i];
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].pImageInfo = &image_info;
        vkUpdateDescriptorSets(device, (u32)std::size(writes), writes, 0, 0);
    }

    Image depth_buffer = allocate_image(
        allocator, 
        device, 
        { (u32)SCREEN_WIDTH, (u32)SCREEN_HEIGHT, 1 }, 
        VK_FORMAT_D32_SFLOAT, 
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 
        VK_IMAGE_ASPECT_DEPTH_BIT
    );
    {
        vkResetCommandPool(device, command_pool, 0);
        VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        {
            VkImageMemoryBarrier2 barrier = image_memory_barrier2(
                depth_buffer.image,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
            );

            VkDependencyInfo dep_info = dependency_info(0, 0, 0, 0, 1, &barrier);
            vkCmdPipelineBarrier2(command_buffer, &dep_info);
        }
        vkEndCommandBuffer(command_buffer);
        VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        vkQueueSubmit(queue, 1, &submit, 0);
        vkQueueWaitIdle(queue);
    }


    u64 prev_time = SDL_GetPerformanceCounter();
    double elapsed = 0.0;

    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            }
            if (event.type == SDL_KEYDOWN)
            {
                if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
                {
                    goto shutdown;
                }
            }
        }

        u64 curr_time = SDL_GetPerformanceCounter();
        double delta = (double)(curr_time - prev_time) * inv_pfreq;
        elapsed += delta;
        prev_time = curr_time;

        VkAcquireNextImageInfoKHR info{ VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR };
        info.deviceMask = 1;
        info.semaphore = acquire_semaphore;
        info.swapchain = swapchain;
        info.timeout = UINT64_MAX;
        u32 image_index = 0;
        vkAcquireNextImage2KHR(device, &info, &image_index);

        vkResetCommandPool(device, command_pool, 0);

        VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        {
            VkImageMemoryBarrier2 barrier = image_memory_barrier2(
                swapchain_images[image_index],
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
            );

            VkDependencyInfo dep_info = dependency_info(0, 0, 0, 0, 1, &barrier);
            vkCmdPipelineBarrier2(command_buffer, &dep_info);
        }

        VkRenderingInfo rendering_info = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering_info.renderArea = { {0, 0,}, {SCREEN_WIDTH, SCREEN_HEIGHT} };
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        VkRenderingAttachmentInfo color_attachment = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        VkClearValue clr = {};
        clr.color = { 0.1f, 0.2f, 0.1f, 1.0f };
        color_attachment.clearValue = clr;
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        color_attachment.imageView = swapchain_image_views[image_index];
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkClearValue depth_clear{};
        depth_clear.depthStencil.depth = 1.f;

        VkRenderingAttachmentInfo depth_attachment_info{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depth_attachment_info.imageView = depth_buffer.image_view;
        depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment_info.clearValue = depth_clear;

        rendering_info.pColorAttachments = &color_attachment;
        rendering_info.pDepthAttachment = &depth_attachment_info;

        VkViewport viewport = {};
        viewport.x = 0;
        viewport.y = (float)SCREEN_HEIGHT;
        viewport.width = float(SCREEN_WIDTH);
        viewport.height = -float(SCREEN_HEIGHT);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdBeginRendering(command_buffer, &rendering_info);

        VkRect2D scissor = {};
        scissor.extent = { SCREEN_WIDTH, SCREEN_HEIGHT };
        scissor.offset = { 0, 0 };
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        float time = (float)elapsed;
        struct {
            glm::mat4 viewproj;
            float time;
        } pc;
        pc.viewproj = viewproj;
        pc.time = time;

        vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        for (int i = 0; i < mesh.primitives.size(); ++i)
        {
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &mesh_desc_sets[i], 0, 0);
            vkCmdBindIndexBuffer(command_buffer, mesh.primitives[i].index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(command_buffer, (u32)mesh.primitives[i].indices.size(), 1, 0, 0, 0);
        }

        vkCmdEndRendering(command_buffer);

        {
            VkImageMemoryBarrier2 barrier = image_memory_barrier2(
                swapchain_images[image_index],
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
            );

            VkDependencyInfo dep_info = dependency_info(0, 0, 0, 0, 1, &barrier);
            vkCmdPipelineBarrier2(command_buffer, &dep_info);
        }

        vkEndCommandBuffer(command_buffer);

        VkSubmitInfo2 submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        VkCommandBufferSubmitInfo submit_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        submit_info.commandBuffer = command_buffer;
        submit_info.deviceMask = 1;
        submit.pCommandBufferInfos = &submit_info;
        submit.commandBufferInfoCount = 1;
        submit.waitSemaphoreInfoCount = 1;
        VkSemaphoreSubmitInfo sem_info = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        sem_info.deviceIndex = 0;
        sem_info.semaphore = acquire_semaphore;
        sem_info.stageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        submit.pWaitSemaphoreInfos = &sem_info;
        VkSemaphoreSubmitInfo signal_info = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        signal_info.deviceIndex = 0;
        signal_info.semaphore = release_semaphore;
        signal_info.stageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        submit.pSignalSemaphoreInfos = &signal_info;
        submit.signalSemaphoreInfoCount = 1;
        vkQueueSubmit2(queue, 1, &submit, nullptr);

        VkPresentInfoKHR present_info{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain;
        present_info.pImageIndices = &image_index;
        present_info.pWaitSemaphores = &release_semaphore;
        present_info.waitSemaphoreCount = 1;

        vkQueuePresentKHR(queue, &present_info);

        vkDeviceWaitIdle(device);
    }

shutdown:
    for (auto& tex : textures)
    {
        vmaFreeMemory(allocator, tex.image.allocation);
        vkDestroyImageView(device, tex.image.image_view, nullptr);
        vkDestroyImage(device, tex.image.image, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroySampler(device, sampler, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    for (auto& l : layouts)
        vkDestroyDescriptorSetLayout(device, l, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroySemaphore(device, acquire_semaphore, nullptr);
    vkDestroySemaphore(device, release_semaphore, nullptr);
    for (int i = 0; i < swapchain_image_views.size(); ++i)
        vkDestroyImageView(device, swapchain_image_views[i], nullptr);
    destroy_image(device, allocator, default_texture.image);
    destroy_image(device, allocator, depth_buffer);
    free_mesh(device, allocator, &mesh);
    vmaDestroyAllocator(allocator);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}