#ifndef RESOURCES_H
#define RESOURCES_H

#include <vulkan/vulkan.h>

#include <vector>

class JobManager;

enum class ResourceType {
    StorageBuffer,
    StorageImage
};

class Resource
{
    ResourceType resourceType;
    size_t size;
    size_t ID;

protected:
    Resource(ResourceType resourceType, size_t size) :
        resourceType(resourceType),
        size(size)
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
    VkDeviceMemory bufferMemory;
    Type bufferType;

    Buffer *stagingBuffer;

public:

    Buffer(VkBuffer buffer, VkDeviceMemory bufferMemory, size_t size, Type type = Type::DeviceLocal, Buffer *staging = nullptr) :
        Resource(ResourceType::StorageBuffer, size),
        buffer(buffer),
        bufferMemory(bufferMemory),
        bufferType(type),
        stagingBuffer(staging)
    {}

    ~Buffer()
    {
        if (stagingBuffer != nullptr)
            delete stagingBuffer;
    }

    VkBuffer getBuffer() const
    {
        return buffer;
    }

    VkDeviceMemory getMemory() const
    {
        return bufferMemory;
    }

    Buffer* getStagingBuffer() const
    {
        return stagingBuffer;
    }

    Buffer::Type getBufferType() const
    {
        return bufferType;
    }
};


class Image : public Resource
{
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;
    size_t width;
    size_t height;
    size_t channels;
    VkImageLayout layout;
    
public:
    Image(VkImage image, VkDeviceMemory imageMemory, VkImageView imageView, size_t width, size_t height,
            size_t channels, VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED) :
        Resource(ResourceType::StorageImage, width * height * channels),
        image(image),
        imageMemory(imageMemory),
        imageView(imageView),
        width(width),
        height(height),
        channels(channels),
        layout(layout)
    {}

    VkImage getImage() const
    {
        return image;
    }

    VkImageView getView() const
    {
        return imageView;
    }

    VkDeviceMemory getMemory() const
    {
        return imageMemory;
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
};


class ResourceSet
{
    VkDescriptorSet descriptorSet;
    std::vector<size_t> IDs;

public:
    ResourceSet(VkDescriptorSet descriptorSet) :
        descriptorSet(descriptorSet)
    {}

    VkDescriptorSet getDescriptorSet() const
    {
        return descriptorSet;
    }
};


class Task
{
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;

public:

    Task(VkPipeline pipeline, VkPipelineLayout pipelineLayout, const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts) :
        pipeline(pipeline),
        pipelineLayout(pipelineLayout),
        descriptorSetLayouts(descriptorSetLayouts)
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
