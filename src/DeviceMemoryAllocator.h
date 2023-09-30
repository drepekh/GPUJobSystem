#ifndef DEVICE_MEMORY_ALLOCATOR_H
#define DEVICE_MEMORY_ALLOCATOR_H

#include "Resources.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

class JobManager;

/**
 * @brief Interface for device memory allocator used by JobManager.
 * 
 */
class DeviceMemoryAllocator
{
public:
    virtual ~DeviceMemoryAllocator() {}

    /**
     * @brief Initialize allocator.
     * 
     * Called on object after Vulkan has been properly initialized.
     * 
     * @param manager Instance of JobManager class that is initializing this allocator.
     * @param physicalDevice Physical device used by calling JobManager instance.
     * @param device Logical device used by calling JobManager instance.
     * @param instance Vulkan instance created by calling JobManager instance.
     */
    virtual VkResult initialize(JobManager* manager, VkPhysicalDevice physicalDevice, VkDevice device, VkInstance instance) = 0;

    /**
     * @brief Deinitilize allocator.
     * 
     * Called after all resources (images and buffers) have been freed but before Vulkan instance/device has been destroyed.
     */
    virtual void deinitialize() = 0;

    /**
     * @brief Create a Buffer object.
     * 
     * Create buffer, allocate appropriate chunk of memory according to required and optional properties,
     * and bind them together.
     * 
     * @param buffer Created buffer
     * @param createInfo Info used to create buffer
     * @param properties Required properties of memory allocated for this buffer
     * @param optionalProperties Optional propertie of memory allocated for this buffer
     * @return AllocatedMemory Information about the memory that was allocated for this buffer
     */
    virtual AllocatedMemory createBuffer(VkBuffer& buffer, const VkBufferCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties) = 0;

    /**
     * @brief Create an Image object
     * 
     * Create image, allocate appropriate chunk of memory according to required and optional properties,
     * and bind them together.
     * 
     * @param image Created image
     * @param createInfo Info used to create image
     * @param properties Required memory properties
     * @param optionalProperties Optional memory properties
     * @return AllocatedMemory Information about the memory that was allocated for this image
     */
    virtual AllocatedMemory createImage(VkImage& image, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties) = 0;

    /**
     * @brief Free memory that was previously allocated by this allocator.
     * 
     * @param allocatedMemory Allocated memory object
     */
    virtual void freeMemory(const AllocatedMemory& allocatedMemory) = 0;

    /**
     * @brief Map host-visible memory and return pointer to it.
     * 
     * Needed mostly because Vulkan prohibits mapping of a single memory allocation more than once.
     * Thus, if allocator implementation can use same allocated chunk for multiple buffers, it should
     * also make sure that if some of those buffers are mapped at the same time, underlying memory is
     * mapped only once.
     * 
     * @param allocatedMemory Memory allocated with this allocator. Must be Host-Visible.
     * @param size Amount of the memory that we want to map.
     * @param ppData Returned pointer to the mapped memory.
     * @return VkResult Operation result.
     */
    virtual VkResult mapMemory(const AllocatedMemory& allocatedMemory, VkDeviceSize size, void** ppData) = 0;

    /**
     * @brief Unmap host-visible memory, mapped previously by mapMemory().
     * 
     * Same as with mapMemory, implementation might need to count amount of times single memory allocation
     * was mapped/unmapped.
     * 
     * @param allocatedMemory Host-visible memory allocated with this allocator that was previously mapped.
     */
    virtual void unmapMemory(const AllocatedMemory& allocatedMemory) = 0;
};

/**
 * @brief Simplest implementation of DeviceMemoryAllocator. Does not have any special allocation strategy.
 * Allocates and frees memory exactly when requested and in exactly specified amounts.
 * 
 */
class SimpleDeviceMemoryAllocator : public DeviceMemoryAllocator
{
    JobManager* manager;
    VkDevice device;
    VkPhysicalDevice physicalDevice;

public:
    virtual VkResult initialize(JobManager*, VkPhysicalDevice, VkDevice, VkInstance) override;
    virtual void deinitialize() override;

    virtual AllocatedMemory createBuffer(VkBuffer& buffer, const VkBufferCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties) override;
    virtual AllocatedMemory createImage(VkImage& image, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties) override;

    virtual void freeMemory(const AllocatedMemory&) override;

    virtual VkResult mapMemory(const AllocatedMemory& allocatedMemory, VkDeviceSize size, void** ppData) override;
    virtual void unmapMemory(const AllocatedMemory& allocatedMemory) override;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties = 0);
};

/**
 * @brief Struct that holds custom data created by VMADeviceMemoryAllocator and stored in AllocatedMemory object.
 * 
 */
struct VMACustomData
{
    VmaAllocation allocation;
};

/**
 * @brief Implementation of DeviceMemoryAllocator that uses VMA library underneath with all
 * benefits provided by it. These include allocations in big chunks for better performance.
 * 
 */
class VMADeviceMemoryAllocator : public DeviceMemoryAllocator
{
public:
    VmaAllocator allocator;

    virtual VkResult initialize(JobManager*, VkPhysicalDevice, VkDevice, VkInstance) override;
    virtual void deinitialize() override;

    virtual AllocatedMemory createBuffer(VkBuffer& buffer, const VkBufferCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties) override;
    virtual AllocatedMemory createImage(VkImage& image, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties) override;

    virtual void freeMemory(const AllocatedMemory&) override;

    virtual VkResult mapMemory(const AllocatedMemory& allocatedMemory, VkDeviceSize size, void** ppData) override;
    virtual void unmapMemory(const AllocatedMemory& allocatedMemory) override;

private:
    static VMACustomData* getCustomData(const AllocatedMemory& allocatedMemory);
};


#endif // DEVICE_MEMORY_ALLOCATOR_H