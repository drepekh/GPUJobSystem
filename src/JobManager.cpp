#include "JobManager.h"

#include "DeviceMemoryAllocator.h"

#include <spirv_reflect.h>
#include <cassert>

#ifdef USE_VMA
using DefaultMemoryAllocator = VMADeviceMemoryAllocator;
#else
using DefaultMemoryAllocator = SimpleDeviceMemoryAllocator;
#endif

JobManager::JobManager(const std::vector<std::string> extensions, DeviceMemoryAllocator* memoryAllocator) :
    manageInstance(true),
    deviceExtensions(extensions),
    allocator(memoryAllocator)
{
    initVulkan();

    if (!allocator)
    {
        allocator = new DefaultMemoryAllocator;
    }
    if (allocator->initialize(this, physicalDevice, device, instance) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to initialize memory allocator");
    }
}

JobManager::JobManager(VkPhysicalDevice physicalDevice, VkDevice device) :
    physicalDevice(physicalDevice),
    device(device),
    manageInstance(false)
{
    initVulkan();
}

JobManager::~JobManager()
{
    cleanupVulkan();
}

Task JobManager::createTask(const std::string &shaderPath)
{
    return _createTask(shaderPath);
}

Buffer JobManager::createBuffer(size_t size, Buffer::Type type)
{
    VkBuffer buffer;
    AllocatedMemory bufferMemory;
    switch(type)
    {
    case Buffer::Type::DeviceLocal:
        createBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            buffer,
            bufferMemory);
        break;
    case Buffer::Type::Uniform:
        createBuffer(
            size,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            buffer,
            bufferMemory);
        break;
    case Buffer::Type::Staging:
        createBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            buffer,
            bufferMemory,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        break;
    }

    buffers.push_back(buffer);
    allocatedMemory.push_back(bufferMemory);

    Buffer *staging = nullptr;
    if (type == Buffer::Type::DeviceLocal)
    {
        VkBuffer stagingBuffer;
        AllocatedMemory stagingBufferMemory;
        createBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        
        staging = new Buffer(stagingBuffer, stagingBufferMemory, size, Buffer::Type::Staging);

        buffers.push_back(stagingBuffer);
        allocatedMemory.push_back(stagingBufferMemory);
    }

    return Buffer{ buffer, bufferMemory, size, type, staging };
}

Image JobManager::createImage(size_t width, size_t height)
{
    VkImage image;
    AllocatedMemory imageMemory;
    createImage(static_cast<uint32_t>(width), static_cast<uint32_t>(height),
        VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory);

    VkImageView imageView = createImageView(image, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

    images.push_back(image);
    imageViews.push_back(imageView);
    allocatedMemory.push_back(imageMemory);

    VkBuffer stagingBuffer;
    AllocatedMemory stagingBufferMemory;
    size_t imageSize = width * height * 4;
    createBuffer(
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory,
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    
    Buffer *staging = new Buffer(stagingBuffer, stagingBufferMemory, imageSize, Buffer::Type::Staging);

    buffers.push_back(stagingBuffer);
    allocatedMemory.push_back(stagingBufferMemory);

    return { image, imageMemory, imageView, width, height, 4, staging };
}

ResourceSet JobManager::createResourceSet(const std::vector<Resource *> &resources)
{
    VkDescriptorSetLayout layout = createDescriptorSetLayout(resourceToDescriptorType(resources));
    VkDescriptorSet descriptorSet = createDescriptorSet(resourceToDescriptorType(resources), resources, layout);

    descriptorSetLayouts.push_back(layout);

    return { descriptorSet, resources };
}

Job JobManager::createJob(VkCommandBuffer commandBuffer)
{
    if (commandBuffer != VK_NULL_HANDLE)
        return { this, commandBuffer, VK_NULL_HANDLE, VK_NULL_HANDLE };
    
    auto fence = createFence();
    commandBuffer = createCommandBuffer();

    fences.push_back(fence);

    return { this, commandBuffer, computeQueue, fence };
}

VkDevice JobManager::getDevice()
{
    return device;
}

DeviceMemoryAllocator* JobManager::getDeviceMemoryAllocator()
{
    return allocator;
}

DeviceComputeLimits JobManager::getComputeLimits()
{
    return computeLimits;
}

void JobManager::initVulkan()
{
    if (manageInstance)
    {
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();
    }
    cacheComputeLimits();
    createCommandPool();
    createDescriptorPool();
}

void JobManager::cleanupVulkan()
{
    cleanupResources();

    allocator->deinitialize();

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);

    for (const auto& [key, shaderModule] : shaderModules)
        vkDestroyShaderModule(device, shaderModule.vkModule, nullptr);
    
    shaderModules.clear();

    if (manageInstance)
    {
        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers)
        {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        vkDestroyInstance(instance, nullptr);
    }
}

void JobManager::cleanupResources()
{
    for (auto fence: fences)
        vkDestroyFence(device, fence, nullptr);
    fences.clear();

    for (auto semaphore: semaphores)
        vkDestroySemaphore(device, semaphore, nullptr);
    semaphores.clear();

    for (auto buffer: buffers)
        vkDestroyBuffer(device, buffer, nullptr);
    buffers.clear();
    
    for (auto imageView: imageViews)
        vkDestroyImageView(device, imageView, nullptr);
    imageViews.clear();
    
    for (auto image: images)
        vkDestroyImage(device, image, nullptr);
    images.clear();
    
    for (auto memory: allocatedMemory)
        allocator->freeMemory(memory);
    allocatedMemory.clear();

    for (auto pipeline: pipelines)
        vkDestroyPipeline(device, pipeline, nullptr);
    pipelines.clear();
    
    for (auto pipelineLayout: pipelineLayouts)
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    pipelineLayouts.clear();

    for (auto descriptorSetLayout: descriptorSetLayouts)
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    descriptorSetLayouts.clear();
}

void JobManager::createInstance()
{
    if (enableValidationLayers && !checkValidationLayerSupport())
    {
        throw std::runtime_error("Validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPU Job System";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if (enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    }
    else
    {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create instance!");
    }
}

bool JobManager::checkValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers)
    {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

std::vector<const char*> JobManager::getRequiredExtensions()
{
    std::vector<const char*> extensions;

    if (enableValidationLayers)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

void JobManager::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

void JobManager::setupDebugMessenger()
{
    if (!enableValidationLayers)
        return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void JobManager::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device: devices)
    {
        if (isDeviceSuitable(device))
        {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE)
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void JobManager::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.computeFamily.value(), indices.transferFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily: uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    // deviceFeatures.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    std::vector<const char*> extensions;
    for (const auto &ext: deviceExtensions)
        extensions.push_back(ext.data());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
}

bool JobManager::isDeviceSuitable(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    if (VK_VERSION_MAJOR(deviceProperties.apiVersion) == 1 && VK_VERSION_MINOR(deviceProperties.apiVersion) < 1)
        return false;

    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    return indices.isComplete() && extensionsSupported;
}

bool JobManager::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

void JobManager::cacheComputeLimits()
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    computeLimits = DeviceComputeLimits{};
    computeLimits.maxComputeSharedMemorySize = deviceProperties.limits.maxComputeSharedMemorySize;
    computeLimits.maxComputeWorkGroupInvocations = deviceProperties.limits.maxComputeWorkGroupInvocations;
    std::copy(deviceProperties.limits.maxComputeWorkGroupCount, deviceProperties.limits.maxComputeWorkGroupCount + 3,
        computeLimits.maxComputeWorkGroupCount);
    std::copy(deviceProperties.limits.maxComputeWorkGroupSize, deviceProperties.limits.maxComputeWorkGroupSize + 3,
        computeLimits.maxComputeWorkGroupSize);
}

VkDescriptorSetLayout JobManager::createDescriptorSetLayout(std::vector<VkDescriptorType> types)
{
    // TODO
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    for (size_t i = 0; i < types.size(); ++i)
    {
        VkDescriptorSetLayoutBinding layoutBinding;
        layoutBinding.binding = static_cast<uint32_t>(i);
        layoutBinding.descriptorType = types[i];
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings.push_back(layoutBinding);
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout descriptorSetLayout;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    return descriptorSetLayout;
}

VkDescriptorSetLayout JobManager::createDescriptorSetLayout(std::vector<ResourceType> types)
{
    std::vector<VkDescriptorType> descriptorTypes;
    for (auto type: types)
        descriptorTypes.push_back(resourceToDescriptorType(type));
    
    return createDescriptorSetLayout(descriptorTypes);
}

VkPipelineLayout JobManager::createPipelineLayout(const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts,
    uint32_t pushConstantSize)
{
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    VkPushConstantRange range{};
    if (pushConstantSize > 0)
    {
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.offset = 0;
        range.size = pushConstantSize;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &range;
    }

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    return pipelineLayout;
}

VkPipeline JobManager::createComputePipeline(VkShaderModule vkModule, VkPipelineLayout pipelineLayout,
    VkSpecializationInfo *specializationInfo)
{
    VkPipelineShaderStageCreateInfo shaderStage = {};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = vkModule;
    shaderStage.pName = "main";
    shaderStage.pSpecializationInfo = specializationInfo;

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    // computePipelineCreateInfo.flags;
    computePipelineCreateInfo.stage = shaderStage;
    computePipelineCreateInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, nullptr, 1, &computePipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create compute pipeline!");
    }

    return pipeline;
}

void JobManager::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
    AllocatedMemory& bufferMemory, VkMemoryPropertyFlags optionalProperties)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    bufferMemory = allocator->createBuffer(buffer, bufferInfo, properties, optionalProperties);
}

VkImageView JobManager::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

void JobManager::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties, VkImage& image, AllocatedMemory& imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    imageMemory = allocator->createImage(image, imageInfo, properties, 0);
}

void JobManager::transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    switch (oldLayout)
    {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        barrier.srcAccessMask = 0;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
    default:
        throw std::invalid_argument("Unsupported layout transition!");
    }

    switch (newLayout)
    {
    case VK_IMAGE_LAYOUT_GENERAL:
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        barrier.dstAccessMask = 0;
        destinationStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        break;
    default:
        throw std::invalid_argument("Unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void JobManager::copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, size_t width, size_t height)
{
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1
    };

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void JobManager::copyImageToBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, size_t width, size_t height)
{
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1
    };

    vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
}

void JobManager::copyImageToImage(VkCommandBuffer commandBuffer, VkImage src, VkImageLayout srcLayout, VkImage dst, VkImageLayout dstLayout,
    size_t width, size_t height)
{
    VkImageSubresourceLayers layer{};
    layer.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    layer.mipLevel = 0;
    layer.baseArrayLayer = 0;
    layer.layerCount = 1;

    VkImageCopy region{};
    region.srcSubresource = layer;
    region.dstSubresource = layer;
    region.extent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1
    };

    vkCmdCopyImage(commandBuffer, src, srcLayout, dst, dstLayout, 1, &region);
}

void JobManager::copyBufferToBuffer(VkCommandBuffer commandBuffer, VkBuffer src, VkBuffer dst, size_t size,
    size_t srcOffset, size_t dstOffset)
{
    VkBufferCopy region = {srcOffset, dstOffset, size};

    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &region);
}

void JobManager::copyDataToHostVisibleMemory(const void *data, size_t size, const AllocatedMemory& memory)
{
    void* stagingData;
    allocator->mapMemory(memory, size, &stagingData);
        memcpy(stagingData, data, size);
    allocator->unmapMemory(memory);
}

void JobManager::copyDataFromHostVisibleMemory(void *data, size_t size, const AllocatedMemory& memory)
{
    void* stagingData;
    allocator->mapMemory(memory, size, &stagingData);
        memcpy(data, stagingData, size);
    allocator->unmapMemory(memory);
}

uint32_t JobManager::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    optionalProperties |= properties;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & optionalProperties) == optionalProperties)
        {
            return i;
        }
    }

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

JobManager::QueueFamilyIndices JobManager::findQueueFamilies(VkPhysicalDevice device)
{
    // TODO dedicated transfer queue?
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily: queueFamilies)
    {
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT &&
            queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)
        {
            indices.computeFamily = i;
            indices.transferFamily = i;
        }

        if (indices.isComplete())
            break;

        i++;
    }

    return indices;
}

void JobManager::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // poolInfo.flags; // TODO
    poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics command pool!");
    }
}

void JobManager::createDescriptorPool()
{
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 256;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 256;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 256;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

VkDescriptorSet JobManager::createDescriptorSet(std::vector<VkDescriptorType> types, const std::vector<Resource *> &resources,
    VkDescriptorSetLayout descriptorSetLayout)
{
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    std::vector<VkWriteDescriptorSet> descriptorWrites{};
    std::vector<std::function<void()>> deleters;

    for (size_t i = 0; i < types.size(); ++i)
    {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet;
        write.dstBinding = static_cast<uint32_t>(i);
        write.dstArrayElement = 0;
        write.descriptorCount = 1;

        if (types[i] == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        {
            VkDescriptorBufferInfo *bufferInfo = new VkDescriptorBufferInfo{};
            bufferInfo->buffer = static_cast<Buffer*>(resources[i])->getBuffer();
            bufferInfo->offset = 0;
            bufferInfo->range = VK_WHOLE_SIZE;
            deleters.push_back([bufferInfo](){
                delete bufferInfo;
            });

            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = bufferInfo;
            
            descriptorWrites.push_back(write);
        }
        else if (types[i] == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
        {
            VkDescriptorImageInfo *imageInfo = new VkDescriptorImageInfo{};
            imageInfo->imageView = static_cast<Image*>(resources[i])->getView();;
            imageInfo->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            deleters.push_back([imageInfo](){
                delete imageInfo;
            });

            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = imageInfo;

            descriptorWrites.push_back(write);
        }
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

    for (auto &del: deleters)
        del();

    return descriptorSet;
}

VkCommandBuffer JobManager::createCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate command buffer!");
    }

    return commandBuffer;
}

VkFence JobManager::createFence()
{
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence fence;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create fence!");
    }

    return fence;
}

VkSemaphore JobManager::createSemaphore()
{
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore semaphore;
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create semaphore");
    }

    semaphores.push_back(semaphore);

    return semaphore;
}

VkShaderModule JobManager::createVkShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

JobManager::ShaderModule& JobManager::getShaderModule(const std::string& shaderPath)
{
    auto it = shaderModules.find(shaderPath);
    if (it != shaderModules.end())
    {
        return it->second;
    }
    else
    {
        auto shaderCode = readFile(shaderPath, true);
        ShaderModule shaderModule;
        shaderModule.vkModule = createVkShaderModule(shaderCode);
        shaderModule.reflectModule = std::make_shared<spv_reflect::ShaderModule>(shaderCode.size(), shaderCode.data());
        reflectDescriptorSets(shaderModule.reflectModule.get(), shaderModule.layouts, shaderModule.resourceAccessFlags);
        shaderModule.pushConstantSize = reflectPushConstantSize(shaderModule.reflectModule.get());

        auto result = shaderModules.insert({ shaderPath, shaderModule });
        return result.first->second;
    }
}

VkResult JobManager::CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void JobManager::DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL JobManager::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, 
    VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

std::vector<char> JobManager::readFile(const std::string &filename, bool binaryMode)
{
    std::ios::openmode openmode = std::ios::in;
    if (binaryMode)
    {
        openmode |= std::ios::binary;
    }

    std::ifstream ifs(filename, openmode);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Failed to open file " + filename);
    }

    ifs.ignore(std::numeric_limits<std::streamsize>::max());
    std::vector<char> data(ifs.gcount());
    ifs.clear();
    ifs.seekg(0, std::ios_base::beg);
    ifs.read(data.data(), data.size());
    ifs.close();

    return data;
}

Task JobManager::_createTask(const std::string &shaderPath, VkSpecializationInfo *specializationInfo)
{
    ShaderModule& shaderModule = getShaderModule(shaderPath);

    std::vector<VkDescriptorSetLayout> layouts;
    for (const auto &descriptorSetLayoutTypes: shaderModule.layouts)
    {
        layouts.push_back(createDescriptorSetLayout(descriptorSetLayoutTypes));
        descriptorSetLayouts.push_back(layouts.back());
    }
    auto pipelineLayout = createPipelineLayout(layouts, static_cast<uint32_t>(shaderModule.pushConstantSize));
    pipelineLayouts.push_back(pipelineLayout);

    auto pipeline = createComputePipeline(shaderModule.vkModule, pipelineLayout, specializationInfo);
    pipelines.push_back(pipeline);

    return { pipeline, pipelineLayout, layouts, shaderModule.resourceAccessFlags };
}

ResourceType reflectDescriptorTypeToResourceType(SpvReflectDescriptorType type)
{
    switch(type)
    {
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            return ResourceType::StorageBuffer;
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            return ResourceType::StorageImage;
        default:
            throw std::runtime_error("Unsupported descriptor type: " + std::to_string(type));
    }
}

void JobManager::reflectDescriptorSets(spv_reflect::ShaderModule* reflectModule,
    std::vector<std::vector<ResourceType>>& outLayout,
    std::vector<std::vector<AccessTypeFlags>>& outResourceAccessFlags)
{
    outLayout.clear();
    outResourceAccessFlags.clear();

    uint32_t count;
    SpvReflectResult result = reflectModule->EnumerateDescriptorSets(&count, nullptr);
    assert(result == SPV_REFLECT_RESULT_SUCCESS);

    std::vector<SpvReflectDescriptorSet*> sets(count);
    result = reflectModule->EnumerateDescriptorSets(&count, sets.data());
    assert(result == SPV_REFLECT_RESULT_SUCCESS);

    outLayout.resize(sets.size());
    outResourceAccessFlags.resize(sets.size());
    for (size_t i = 0; i < sets.size(); ++i)
    {
        for (size_t j = 0; j < sets[i]->binding_count; ++j)
        {
            outLayout[i].push_back(reflectDescriptorTypeToResourceType(sets[i]->bindings[j]->descriptor_type));
            outResourceAccessFlags[i].push_back((sets[i]->bindings[j]->block.flags & SPV_REFLECT_VARIABLE_FLAGS_UNUSED)
                ? AccessType::None
                : ((sets[i]->bindings[j]->block.decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE)
                    ? AccessType::Read
                    : AccessType:: Read | AccessType::Write));
        }
    }
}

uint32_t JobManager::reflectPushConstantSize(spv_reflect::ShaderModule* reflectModule)
{
    uint32_t count;
    SpvReflectResult result = reflectModule->EnumeratePushConstantBlocks(&count, nullptr);
    assert(result == SPV_REFLECT_RESULT_SUCCESS);

    if (count == 0)
    {
        return 0;
    }

    std::vector<SpvReflectBlockVariable*> blocks(count);
    result = reflectModule->EnumeratePushConstantBlocks(&count, blocks.data());
    assert(result == SPV_REFLECT_RESULT_SUCCESS);

    return blocks[0]->size;
}