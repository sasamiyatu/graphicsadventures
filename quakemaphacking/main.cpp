#include "SDL2/SDL.h"
#include <stdio.h>
#include <stdlib.h>
#define VOLK_IMPLEMENTATION
#include "volk/volk.h"
#include "SDL2/SDL_vulkan.h"
#include "vma/vk_mem_alloc.h"
#include <assert.h>
#include <vector>

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
static const int SCREEN_WIDTH = 960;
static const int SCREEN_HEIGHT = 540;
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
    
    VkPhysicalDeviceVulkan13Features vulkan_13_feats = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    vulkan_13_feats.synchronization2 = true;
    info.pNext = &vulkan_13_feats;

    vkCreateDevice(physical_device, &info, nullptr, &device);

    volkLoadDevice(device);

    return device;
}

VkSwapchainKHR create_swapchain(VkDevice device, VkPhysicalDevice physical_device, VkSurfaceKHR surface)
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

int main(int argc, char** argv)
{
    VK_CHECK(volkInitialize());

    VkInstance instance = create_instance();
    volkLoadInstanceOnly(instance);

    VkPhysicalDevice physical_device = pick_physical_device(instance);
    VkDevice device = create_device(physical_device);
    VkSurfaceKHR surface = create_surface(instance);
    VkSwapchainKHR swapchain = create_swapchain(device, physical_device, surface);
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

    VkCommandPool command_pool = create_command_pool(device);

    VkCommandBuffer command_buffer = 0;
    allocate_command_buffers(device, command_pool, 1, &command_buffer);

    SDL_Event event;
    bool quit = false;
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

        VkClearColorValue clear = { 1, 0, 1, 1 };
        VkImageSubresourceRange range = {};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseArrayLayer = 0;
        range.baseMipLevel = 0;
        range.layerCount = 1;
        range.levelCount = 1;
        vkCmdClearColorImage(command_buffer, swapchain_images[image_index], VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);

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