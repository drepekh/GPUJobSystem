#ifndef RESOURCES_H
#define RESOURCES_H

#include <vulkan/vulkan.h>

#include <vector>
#include <memory>

class JobManager;

enum class ResourceType {
    StorageBuffer,
    StorageImage
};

enum AccessType : uint8_t {
    None = 0,
    Read = 1,
    Write = 2
};
using AccessTypeFlags = uint8_t;

struct AllocatedMemory
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;

    void* customData = nullptr;
};

class Resource
{
    ResourceType resourceType;
    size_t size;
    AllocatedMemory allocatedMemory;
    size_t ID;

protected:
    Resource() = delete;
    Resource(ResourceType resourceType, size_t size, const AllocatedMemory& allocatedMemory) :
        resourceType(resourceType),
        size(size),
        allocatedMemory(allocatedMemory)
    {
        static size_t nextID = 1;
        ID = nextID++;
    }

public:
    ResourceType getResourceType() const
    {
        return resourceType;
    }

    size_t getSize() const
    {
        return size;
    }

    const AllocatedMemory& GetAllocatedMemory() const
    {
        return allocatedMemory;
    }

    VkDeviceMemory getMemory() const
    {
        return allocatedMemory.memory;
    }

    VkDeviceSize getMemoryOffset() const
    {
        return allocatedMemory.offset;
    }

    size_t getID() const
    {
        return ID;
    }
};


class Buffer : public Resource
{
public:
    enum class Type {
        DeviceLocal,
        Staging,
        Uniform
    };

private:
    VkBuffer buffer;
    Type bufferType;

    std::shared_ptr<Buffer> stagingBuffer;

public:
    Buffer() :
        Resource(ResourceType::StorageBuffer, 0, {}),
        buffer(VK_NULL_HANDLE),
        bufferType(Type::DeviceLocal),
        stagingBuffer(nullptr)
    {}

    Buffer(VkBuffer buffer, const AllocatedMemory& allocatedMemory, size_t size, Type type = Type::DeviceLocal, Buffer *staging = nullptr) :
        Resource(ResourceType::StorageBuffer, size, allocatedMemory),
        buffer(buffer),
        bufferType(type),
        stagingBuffer(staging)
    {}

    VkBuffer getBuffer() const
    {
        return buffer;
    }

    Buffer* getStagingBuffer() const
    {
        return stagingBuffer.get();
    }

    Buffer::Type getBufferType() const
    {
        return bufferType;
    }
};


class Image : public Resource
{
    VkImage image;
    VkImageView imageView;
    size_t width;
    size_t height;
    size_t channels;
    VkImageLayout layout;

    std::shared_ptr<Buffer> stagingBuffer;
    
public:
    Image() :
        Resource(ResourceType::StorageImage, 0, {}),
        image(VK_NULL_HANDLE),
        imageView(VK_NULL_HANDLE),
        width(0),
        height(0),
        channels(0),
        layout(VK_IMAGE_LAYOUT_UNDEFINED),
        stagingBuffer(nullptr)
    {}

    Image(VkImage image, const AllocatedMemory& allocatedMemory, VkImageView imageView, size_t width, size_t height,
            size_t channels, Buffer *staging = nullptr, VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED) :
        Resource(ResourceType::StorageImage, width * height * channels, allocatedMemory),
        image(image),
        imageView(imageView),
        width(width),
        height(height),
        channels(channels),
        layout(layout),
        stagingBuffer(staging)
    {}

    VkImage getImage() const
    {
        return image;
    }

    VkImageView getView() const
    {
        return imageView;
    }

    size_t getWidth() const
    {
        return width;
    }

    size_t getHeight() const
    {
        return height;
    }

    size_t getChannels() const
    {
        return channels;
    }

    VkImageLayout getLayout() const
    {
        return layout;
    }

    void setLayout(VkImageLayout layout)
    {
        this->layout = layout;
    }

    Buffer* getStagingBuffer() const
    {
        return stagingBuffer.get();
    }
};


class ResourceSet
{
    VkDescriptorSet descriptorSet;
    std::vector<Resource *> resources;

public:
    ResourceSet() :
        descriptorSet(VK_NULL_HANDLE)
    {}

    ResourceSet(VkDescriptorSet descriptorSet, const std::vector<Resource *> &resources = {}) :
        descriptorSet(descriptorSet),
        resources(resources)
    {}

    VkDescriptorSet getDescriptorSet() const
    {
        return descriptorSet;
    }

    const std::vector<Resource *>& getResources() const
    {
        return resources;
    }
};


class Task
{
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<std::vector<AccessTypeFlags>> resourceAccessFlags; 

public:

    Task(VkPipeline pipeline, VkPipelineLayout pipelineLayout, const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts, const std::vector<std::vector<AccessTypeFlags>>& resourceAccessFlags) :
        pipeline(pipeline),
        pipelineLayout(pipelineLayout),
        descriptorSetLayouts(descriptorSetLayouts),
        resourceAccessFlags(resourceAccessFlags)
    {}

    VkPipeline getPipeline() const
    {
        return pipeline;
    }

    VkPipelineLayout getPipelineLayout() const
    {
        return pipelineLayout;
    }

    VkDescriptorSetLayout getDescriptorSetLayout(size_t ind) const
    {
        return descriptorSetLayouts[ind];
    }

    size_t getDescriptorSetLayoutsCount() const
    {
        return descriptorSetLayouts.size();
    }

    const std::vector<std::vector<AccessTypeFlags>>& getResourceAccessFlags() const
    {
        return resourceAccessFlags;
    }
};


class Semaphore
{
    VkSemaphore semaphore;

public:
    Semaphore(VkSemaphore semaphore) :
        semaphore(semaphore)
    {}

    VkSemaphore getSemaphore()
    {
        return semaphore;
    }

    bool isValid()
    {
        return semaphore != VK_NULL_HANDLE;
    }
};


static VkDescriptorType resourceToDescriptorType(ResourceType type)
{
    switch (type)
    {
    case ResourceType::StorageBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case ResourceType::StorageImage:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }

    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

static std::vector<VkDescriptorType> resourceToDescriptorType(const std::vector<ResourceType> &types)
{
    std::vector<VkDescriptorType> res;
    res.reserve(types.size());
    for (ResourceType type: types)
        res.push_back(resourceToDescriptorType(type));
    
    return res;
}

static std::vector<VkDescriptorType> resourceToDescriptorType(const std::vector<Resource> &resources)
{
    std::vector<VkDescriptorType> res;
    res.reserve(resources.size());
    for (const auto &resource: resources)
        res.push_back(resourceToDescriptorType(resource.getResourceType()));
    
    return res;
}

static std::vector<VkDescriptorType> resourceToDescriptorType(const std::vector<Resource*> &resources)
{
    std::vector<VkDescriptorType> res;
    res.reserve(resources.size());
    for (const auto &resource: resources)
        res.push_back(resourceToDescriptorType(resource->getResourceType()));
    
    return res;
}


#endif // RESOURCES_H
