#include "DeviceMemoryAllocator.h"

#define VMA_IMPLEMENTATION
#define VMA_VULKAN_VERSION 1001000
#include <vk_mem_alloc.h>


VkResult SimpleDeviceMemoryAllocator::initialize(JobManager* manager, VkPhysicalDevice physicalDevice, VkDevice device, VkInstance)
{
    this->manager = manager;
    this->device = device;
    this->physicalDevice = physicalDevice;

    return VK_SUCCESS;
}

void SimpleDeviceMemoryAllocator::deinitialize()
{
}

AllocatedMemory SimpleDeviceMemoryAllocator::createBuffer(VkBuffer& buffer, const VkBufferCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties)
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

AllocatedMemory SimpleDeviceMemoryAllocator::createImage(VkImage& image, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties)
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
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, optionalProperties);

    AllocatedMemory allocatedMemory{};
    if (vkAllocateMemory(device, &allocInfo, nullptr, &allocatedMemory.memory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, allocatedMemory.memory, 0);

    return allocatedMemory;
}

void SimpleDeviceMemoryAllocator::freeMemory(const AllocatedMemory& allocatedMemory)
{
    vkFreeMemory(device, allocatedMemory.memory, nullptr);
}

VkResult SimpleDeviceMemoryAllocator::mapMemory(const AllocatedMemory& allocatedMemory, VkDeviceSize size, void **ppData)
{
    return vkMapMemory(device, allocatedMemory.memory, allocatedMemory.offset, size, 0, ppData);
}

void SimpleDeviceMemoryAllocator::unmapMemory(const AllocatedMemory& allocatedMemory)
{
    vkUnmapMemory(device, allocatedMemory.memory);
}

uint32_t SimpleDeviceMemoryAllocator::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties)
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

VkResult VMADeviceMemoryAllocator::initialize(JobManager* manager, VkPhysicalDevice physicalDevice, VkDevice device, VkInstance instance)
{
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_1;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device;
    allocatorCreateInfo.instance = instance;

    return vmaCreateAllocator(&allocatorCreateInfo, &allocator);
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

AllocatedMemory VMADeviceMemoryAllocator::createImage(VkImage& image, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties)
{
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.requiredFlags = properties;
    allocInfo.preferredFlags = optionalProperties;

    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
    vmaCreateImage(allocator, &createInfo, &allocInfo, &image, &allocation, &allocationInfo);

    AllocatedMemory allocatedMemory{};
    allocatedMemory.memory = allocationInfo.deviceMemory;
    allocatedMemory.offset = allocationInfo.offset;
    allocatedMemory.customData = new VMACustomData{allocation};

    return allocatedMemory;
}

void VMADeviceMemoryAllocator::freeMemory(const AllocatedMemory& allocatedMemory)
{
    VMACustomData* customData = getCustomData(allocatedMemory);
    vmaFreeMemory(allocator, customData->allocation);
    delete customData;
}

VkResult VMADeviceMemoryAllocator::mapMemory(const AllocatedMemory& allocatedMemory, VkDeviceSize size, void **ppData)
{
    VMACustomData* customData = getCustomData(allocatedMemory);
    return vmaMapMemory(allocator, customData->allocation, ppData);
}

void VMADeviceMemoryAllocator::unmapMemory(const AllocatedMemory& allocatedMemory)
{
    VMACustomData* customData = getCustomData(allocatedMemory);
    vmaUnmapMemory(allocator, customData->allocation);
}

VMACustomData* VMADeviceMemoryAllocator::getCustomData(const AllocatedMemory& allocatedMemory)
{
    return static_cast<VMACustomData*>(allocatedMemory.customData);;
}
