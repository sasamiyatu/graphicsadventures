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

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define VK_CHECK(expression)        \
    do                              \
    {                               \
        VkResult res = expression;  \
        assert(res == VK_SUCCESS);  \
    } while (0);

static const int SCREEN_FULLSCREEN = 0;
static const int SCREEN_WIDTH = 600;
static const int SCREEN_HEIGHT = 600;
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

    volkLoadInstance(instance);

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
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            printf("Selecting GPU %s\n", props.deviceName);
            return devices[i];
        }
    }

    if (count > 0)
    {
        VkPhysicalDeviceProperties props{};
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

VkSwapchainKHR create_swapchain(VkDevice device, VkPhysicalDevice physical_device, VkSurfaceKHR surface, VkFormat* swapchain_format)
{
    VkSwapchainCreateInfoKHR create_info{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    VkPhysicalDeviceSurfaceInfo2KHR surface_info{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR };
    surface_info.surface = surface;
    u32 format_count = 0;
    vkGetPhysicalDeviceSurfaceFormats2KHR(physical_device, &surface_info, &format_count, nullptr);
    std::vector<VkSurfaceFormat2KHR> formats(format_count);
    for (auto& f : formats)
        f.sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR;
    vkGetPhysicalDeviceSurfaceFormats2KHR(physical_device, &surface_info, &format_count, formats.data());

    create_info.minImageCount = 3;
    //create_info.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    create_info.imageFormat = formats[0].surfaceFormat.format;
    create_info.imageColorSpace = formats[0].surfaceFormat.colorSpace;
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
    glm::vec3 color;
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

VkPipeline create_graphics_pipeline(VkDevice device, VkShaderModule vert_shader, VkShaderModule frag_shader, VkPipelineLayout layout)
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
    
    VkPipelineColorBlendStateCreateInfo blend_state = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };

    VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamic_state = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_state.dynamicStateCount = std::size(dynamic_states);
    dynamic_state.pDynamicStates = dynamic_states;

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
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    info.bindingCount = 1;
    info.pBindings = &binding;
    vkCreateDescriptorSetLayout(device, &info, nullptr, &set_layout);

    return set_layout;
}

struct Buffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
};

int main(int argc, char** argv)
{
    u64 pfreq = SDL_GetPerformanceFrequency();
    double inv_pfreq = 1.0 / (double)pfreq;

    VK_CHECK(volkInitialize());

    VkInstance instance = create_instance();
    volkLoadInstanceOnly(instance);

    VkPhysicalDevice physical_device = pick_physical_device(instance);
    VkDevice device = create_device(physical_device);
    VkSurfaceKHR surface = create_surface(instance);
    VkFormat swapchain_format = {};
    VkSwapchainKHR swapchain = create_swapchain(device, physical_device, surface, &swapchain_format);
    VkSemaphore acquire_semaphore = create_semaphore(device);
    VkSemaphore release_semaphore = create_semaphore(device);

    VkDeviceQueueInfo2 queue_info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2 };
    queue_info.queueFamilyIndex = 0;
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

    VkPipeline pipeline = create_graphics_pipeline(device, vertex, fragment, layout);

    VmaAllocator allocator = create_allocator(device, physical_device, instance);

    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = sizeof(Vertex) * 3;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VmaAllocation allocation = 0;
    VkBuffer buf = 0;
    vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buf, &allocation, nullptr);
    void* mapped = 0;
    vmaMapMemory(allocator, allocation, &mapped);
    std::vector<Vertex> verts(3);
    verts[0].pos = glm::vec3(-0.5, 0.5, 0.0);
    verts[1].pos = glm::vec3(0.5, 0.5, 0.0);
    verts[2].pos = glm::vec3(0.0, -0.5, 0.0);
    verts[0].color = glm::vec3(1.0f, 0.0f, 0.0f);
    verts[1].color = glm::vec3(0.0f, 1.0f, 0.0f);
    verts[2].color = glm::vec3(0.0f, 0.0f, 1.0f);
    memcpy(mapped, verts.data(), sizeof(Vertex) * verts.size());
    vmaUnmapMemory(allocator, allocation);

    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 1;
    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    pool_info.maxSets = 4;
    VkDescriptorPool descriptor_pool = 0;
    VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool));

    VkDescriptorSet desc_set = 0;
    VkDescriptorSetAllocateInfo desc_alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    desc_alloc_info.descriptorPool = descriptor_pool;
    desc_alloc_info.descriptorSetCount = 1;
    desc_alloc_info.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &desc_alloc_info, &desc_set);

    VkDescriptorBufferInfo desc_buffer_info = {};
    desc_buffer_info.buffer = buf;
    desc_buffer_info.offset = 0;
    desc_buffer_info.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet writes = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes.descriptorCount = 1;
    writes.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes.dstSet = desc_set;
    writes.dstBinding = 0;
    writes.dstArrayElement = 0;
    writes.pBufferInfo = &desc_buffer_info;
    vkUpdateDescriptorSets(device, 1, &writes, 0, 0);

    SDL_Event event;
    bool quit = false;

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
            VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            barrier.image = swapchain_images[image_index];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.levelCount = 1;

            VkDependencyInfo dep_info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
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
        rendering_info.pColorAttachments = &color_attachment;

        VkViewport viewport = {};
        viewport.x = 0;
        viewport.y = 0;
        viewport.width = float(SCREEN_WIDTH);
        viewport.height = float(SCREEN_HEIGHT);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdBeginRendering(command_buffer, &rendering_info);

        VkRect2D scissor = {};
        scissor.extent = { SCREEN_WIDTH, SCREEN_HEIGHT };
        scissor.offset = { 0, 0 };
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &desc_set, 0, 0);
        float time = (float)elapsed;
        vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(time), &time);
        vkCmdDraw(command_buffer, 3, 1, 0, 0);

        vkCmdEndRendering(command_buffer);

        {
            VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            barrier.image = swapchain_images[image_index];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.levelCount = 1;

            VkDependencyInfo dep_info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep_info.imageMemoryBarrierCount = 1;
            dep_info.pImageMemoryBarriers = &barrier;
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

    return 0;
}