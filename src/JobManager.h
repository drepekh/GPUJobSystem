#ifndef JOB_MANAGER_H
#define JOB_MANAGER_H

#include <vulkan/vulkan.h>

#include "Job.h"
#include "Resources.h"

#include <stdexcept>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <optional>
#include <set>
#include <functional>


template<typename T>
constexpr size_t argsSize()
{
    return sizeof(T);
}

template<typename T1, typename T2, typename... Args>
constexpr size_t argsSize()
{
    return sizeof(T1) + argsSize<T2, Args...>();
}

template<typename T>
void copyArgs(char *buffer, VkSpecializationMapEntry *entry, size_t offset, uint32_t id, const T& t)
{
    entry->constantID = id;
    entry->offset = static_cast<uint32_t>(offset);
    entry->size = sizeof(T);
    memcpy(buffer, &t, sizeof(T));
}

template<typename T1, typename T2, typename... Args>
void copyArgs(char *buffer, VkSpecializationMapEntry *entry, size_t offset, uint32_t id, const T1& t1, T2&& t2, Args&&... args)
{
    entry->constantID = id;
    entry->offset = offset;
    entry->size = sizeof(T1);
    memcpy(buffer, &t1, sizeof(T1));
    copyArgs<T2, Args...>(buffer + sizeof(T1), entry + 1, offset + sizeof(T1), id + 1, std::forward<T2>(t2), std::forward<Args>(args)...);
}


struct DeviceComputeLimits
{
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupSize[3];
};


class JobManager
{
private:

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> computeFamily;
        std::optional<uint32_t> transferFamily;

        bool isComplete()
        {
            return computeFamily.has_value() && transferFamily.has_value();
        }
    };

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    bool manageInstance;

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkPipelineLayout> pipelineLayouts;
    std::vector<VkPipeline> pipelines;

    // allocated resources
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceMemory> allocatedMemory;
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

    std::vector<VkFence> fences;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<std::string> deviceExtensions = {
        // VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    #ifdef NDEBUG
    const bool enableValidationLayers = false;
    #else
    const bool enableValidationLayers = true;
    #endif

    friend class Job;

public:

    JobManager(const std::vector<std::string> extensions = {});
    JobManager(VkPhysicalDevice physicalDevice, VkDevice device);
    ~JobManager();

    Task createTask(const std::string &shaderPath, const std::vector<std::vector<ResourceType>> &layout,
        uint32_t pushConstantSize = 0);

    // special case for specialization constants
    template<typename... Args>
    Task createTask(const std::string &shaderPath, const std::vector<std::vector<ResourceType>> &layout,
        uint32_t pushConstantSize, Args&&... specializationConstants)
    {
        VkSpecializationMapEntry specializationMapEntries[sizeof...(Args)];
        constexpr size_t dataSize = argsSize<Args...>();
        char buffer[dataSize];
        copyArgs<Args...>(buffer, specializationMapEntries, 0, 0, std::forward<Args>(specializationConstants)...);

        VkSpecializationInfo specializationInfo{};
        specializationInfo.mapEntryCount = sizeof...(Args);
        specializationInfo.pMapEntries = specializationMapEntries;
        specializationInfo.dataSize = static_cast<uint32_t>(dataSize);
        specializationInfo.pData = buffer;

        return _createTask(shaderPath, layout, pushConstantSize, &specializationInfo);
    }

    Buffer createBuffer(size_t size, Buffer::Type type = Buffer::Type::DeviceLocal);
    Image createImage(size_t width, size_t height);
    ResourceSet createResourceSet(const std::vector<Resource *> &resources);
    Job createJob(VkCommandBuffer commandBuffer = VK_NULL_HANDLE);

    VkDevice getDevice();
    DeviceComputeLimits getComputeLimits();

private:

    void initVulkan();
    void cleanupVulkan();

    void createInstance();

    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    void setupDebugMessenger();

    void pickPhysicalDevice();
    void createLogicalDevice();

    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);

    VkDescriptorSetLayout createDescriptorSetLayout(std::vector<VkDescriptorType> types);
    VkDescriptorSetLayout createDescriptorSetLayout(std::vector<ResourceType> types);

    VkPipelineLayout createPipelineLayout(const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts,
        uint32_t pushConstantSize);
    VkPipeline createComputePipeline(const std::string &shaderPath, VkPipelineLayout pipelineLayout,
        VkSpecializationInfo *specializationInfo);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
        VkDeviceMemory& bufferMemory);

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

    void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);

    void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, size_t width, size_t height);
    void copyImageToBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, size_t width, size_t height);
    void copyImageToImage(VkCommandBuffer commandBuffer, VkImage src, VkImageLayout srcLayout, VkImage dst, VkImageLayout dstLayout,
        size_t width, size_t height);
    void copyBufferToBuffer(VkCommandBuffer commandBuffer, VkBuffer src, VkBuffer dst, size_t size,
        size_t srcOffset = 0, size_t dstOffset = 0);

    void copyDataToHostVisibleMemory(const void *data, size_t size, VkDeviceMemory memory);
    void copyDataFromHostVisibleMemory(void *data, size_t size, VkDeviceMemory memory);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    void createCommandPool();
    void createDescriptorPool();

    VkDescriptorSet createDescriptorSet(std::vector<VkDescriptorType> types, const std::vector<Resource *> &resources,
        VkDescriptorSetLayout descriptorSetLayout);
    VkCommandBuffer createCommandBuffer();

    VkFence createFence();
    VkSemaphore createSemaphore();

    VkShaderModule createShaderModule(const std::vector<char>& code);

    static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
    static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks* pAllocator);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, 
        VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

    static std::vector<char> readFile(const std::string &filename, bool binaryMode = false);

    Task _createTask(const std::string &shaderPath, const std::vector<std::vector<ResourceType>> &layout,
        uint32_t pushConstantSize, VkSpecializationInfo *specializationInfo = nullptr);
};

#endif // JOB_MANAGER_H
