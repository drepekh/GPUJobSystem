#include "DeviceMemoryAllocator.h"

#define VMA_IMPLEMENTATION
#define VMA_VULKAN_VERSION 1001000
#include <vk_mem_alloc.h>


void DefaultDeviceMemoryAllocator::initialize(JobManager* manager, VkPhysicalDevice physicalDevice, VkDevice device, VkInstance)
{
    this->manager = manager;
    this->device = device;
    this->physicalDevice = physicalDevice;
}

void DefaultDeviceMemoryAllocator::deinitialize()
{
}

AllocatedMemory DefaultDeviceMemoryAllocator::createBuffer(VkBuffer& buffer, const VkBufferCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties)
{
    if (vkCreateBuffer(device, &createInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, optionalProperties);

    AllocatedMemory allocatedMemory{};
    if (vkAllocateMemory(device, &allocInfo, nullptr, &allocatedMemory.memory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, allocatedMemory.memory, 0);

    return allocatedMemory;
}

AllocatedMemory DefaultDeviceMemoryAllocator::createImage(VkImage& image, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags properties)
{
    if (vkCreateImage(device, &createInfo, nullptr, &image) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    AllocatedMemory allocatedMemory{};
    if (vkAllocateMemory(device, &allocInfo, nullptr, &allocatedMemory.memory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, allocatedMemory.memory, 0);

    return allocatedMemory;
}

void DefaultDeviceMemoryAllocator::FreeMemory(const AllocatedMemory& allocatedMemory)
{
    vkFreeMemory(device, allocatedMemory.memory, nullptr);
}

uint32_t DefaultDeviceMemoryAllocator::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties)
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

// ----- VMA -----

void VMADeviceMemoryAllocator::initialize(JobManager* manager, VkPhysicalDevice physicalDevice, VkDevice device, VkInstance instance)
{
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_1;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device;
    allocatorCreateInfo.instance = instance;

    vmaCreateAllocator(&allocatorCreateInfo, &allocator);
}

void VMADeviceMemoryAllocator::deinitialize()
{
    vmaDestroyAllocator(allocator);
}

AllocatedMemory VMADeviceMemoryAllocator::createBuffer(VkBuffer& buffer, const VkBufferCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties)
{
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.requiredFlags = properties;
    allocInfo.preferredFlags = optionalProperties;
    
    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
    vmaCreateBuffer(allocator, &createInfo, &allocInfo, &buffer, &allocation, &allocationInfo);
    
    AllocatedMemory allocatedMemory{};
    allocatedMemory.memory = allocationInfo.deviceMemory;
    allocatedMemory.offset = allocationInfo.offset;
    allocatedMemory.customData = new VMACustomData{allocation};

    return allocatedMemory;
}

AllocatedMemory VMADeviceMemoryAllocator::createImage(VkImage& image, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags properties)
{
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.requiredFlags = properties;

    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
    vmaCreateImage(allocator, &createInfo, &allocInfo, &image, &allocation, &allocationInfo);

    AllocatedMemory allocatedMemory{};
    allocatedMemory.memory = allocationInfo.deviceMemory;
    allocatedMemory.offset = allocationInfo.offset;
    allocatedMemory.customData = new VMACustomData{allocation};

    return allocatedMemory;
}

void VMADeviceMemoryAllocator::FreeMemory(const AllocatedMemory& allocatedMemory)
{
    VMACustomData* customData = static_cast<VMACustomData*>(allocatedMemory.customData);
    vmaFreeMemory(allocator, customData->allocation);
    delete customData;
}
