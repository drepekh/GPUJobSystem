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

#define USE_VMA

class DeviceMemoryAllocator;

namespace spv_reflect
{
    class ShaderModule;
}

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


/**
 * @brief Various device limits related to compute shaders.
 * 
 */
struct DeviceComputeLimits
{
    /**
     * @brief Maximum total storage size, in bytes, available for variables declared
     * with the Workgroup storage class in shader modules
     */
    uint32_t maxComputeSharedMemorySize;
    /**
     * @brief Maximum number of local workgroups that can be dispatched by a single
     * dispatching command
     */
    uint32_t maxComputeWorkGroupCount[3];
    /**
     * @brief Maximum total number of compute shader invocations in a single local
     * workgroup
     */
    uint32_t maxComputeWorkGroupInvocations;
    /**
     * @brief Maximum size of a local compute workgroup, per dimension
     */
    uint32_t maxComputeWorkGroupSize[3];
};


/**
 * @brief Class responsible for creation and management of all GPU-side resources.
 * 
 * Can be initialized either to create its own Vulkan resources or make use of already
 * created instance/logical device, which may be used for integration with already
 * existing pipeline.
 */
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

    struct ShaderModule
    {
        VkShaderModule vkModule;
        std::shared_ptr<spv_reflect::ShaderModule> reflectModule;

        std::vector<std::vector<ResourceType>> layouts;
        std::vector<std::vector<AccessTypeFlags>> resourceAccessFlags;
        size_t pushConstantSize = 0;
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

    std::map<std::string, ShaderModule> shaderModules;

    DeviceMemoryAllocator* allocator = nullptr;
    // allocated resources
    std::vector<VkBuffer> buffers;
    std::vector<class AllocatedMemory> allocatedMemory;
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

    std::vector<VkFence> fences;

    DeviceComputeLimits computeLimits;

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
    /**
     * @brief Construct a new Job Manager object.
     * 
     * Picks physical device that will be used for all following operations and
     * creates all needed GPU-side resources.
     * 
     * @param extensions List of the device extensions that should be ebanbled
     */
    JobManager(const std::vector<std::string> extensions = {}, DeviceMemoryAllocator* memoryAllocator = nullptr);

    /**
     * @brief Construct a new Job Manager object
     * 
     * Makes use of already initialized physical and logical devices instead of
     * creating its own. Useful for integration into exsisting pipeline.
     * 
     * @param physicalDevice Physical device that is going to be used for all
     * following operations
     * @param device Logical device that was created on picked \p physicalDevice
     */
    JobManager(VkPhysicalDevice physicalDevice, VkDevice device);

    /**
     * @brief Destroy the Job Manager object
     * 
     */
    ~JobManager();

    /**
     * @brief Create a Task object
     * 
     * Create Task using provided shader.
     * 
     * @param shaderPath Path to the already compiled SPIR-V shader
     * @return Created Task
     */
    Task createTask(const std::string &shaderPath);

    /**
     * @brief Create a Task object
     * 
     * Create Task using provided shader. Additionally makes use of provided
     * specialization constants.
     * 
     * @tparam Args Typenames of the specialized constants
     * @param shaderPath Path to the already compiled SPIR-V shader
     * @param specializationConstants Specialization constants to be used in the shader
     * @return Created Task
     */
    template<typename... Args>
    Task createTask(const std::string &shaderPath, Args&&... specializationConstants)
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

        return _createTask(shaderPath, &specializationInfo);
    }

    /**
     * @brief Create a Buffer object
     * 
     * Memory type used by the buffer depends on its type. For every buffer
     * with DeviceLocal \p type additional staging buffer of the same size will be
     * allocated and used during the transfer operations to/from host.
     * 
     * @param size Size of the buffer in bytes
     * @param type Type of the buffer
     * @return Created Buffer
     */
    Buffer createBuffer(size_t size, Buffer::Type type = Buffer::Type::DeviceLocal);

    /**
     * @brief Create an Image object
     * 
     * Created storage image has 4 channels and is allocated in the device-local memory.
     * Initial layout is undefined, so call to Job::syncResourceToDevice() may be needed
     * to change image layout before using it in the shader.
     * 
     * @param width Width of the image (in pixels)
     * @param height Height of the image (in pixels)
     * @return Created Image
     */
    Image createImage(size_t width, size_t height);

    /**
     * @brief Create a Resource Set object
     * 
     * Combines list of resources into single ResourceSet. Useful when these resources
     * are going to be used by the task that is going to be submitted multiple times
     * because ResourceSet is going to be created on every call to Job::addTask(),
     * which might be costly.  
     * 
     * @param resources List of resources
     * @return Created ResourceSet
     */
    ResourceSet createResourceSet(const std::vector<Resource *> &resources);

    /**
     * @brief Create a Job object
     * 
     * Either creates new command buffer or makes use of one passed as a parameter
     * to create a Job object.
     * 
     * @param commandBuffer Already existing command buffer or nullptr to create
     * a new one
     * @return Created Job
     */
    Job createJob(VkCommandBuffer commandBuffer = VK_NULL_HANDLE);

    /**
     * @brief Get the Device object
     * 
     * @return VkDevice used by this manager
     */
    VkDevice getDevice();

    /**
     * @brief Get the Device Memory Allocator object
     * 
     * @return DeviceMemoryAllocator* 
     */
    DeviceMemoryAllocator* getDeviceMemoryAllocator();

    /**
     * @brief Get the Compute Limits object
     * 
     * Returns object with info about various compute limits of the currently used
     * device.
     * 
     * @return DeviceComputeLimits
     */
    DeviceComputeLimits getComputeLimits();

    /**
     * @brief Cleanup allocated resources.
     * 
     * Everything that was created with functions create* will become invalid
     * after this call. Using any of those resources will cause UB.
     * 
     */
    void cleanupResources();

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

    void cacheComputeLimits();

    VkDescriptorSetLayout createDescriptorSetLayout(std::vector<VkDescriptorType> types);
    VkDescriptorSetLayout createDescriptorSetLayout(std::vector<ResourceType> types);

    VkPipelineLayout createPipelineLayout(const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts,
        uint32_t pushConstantSize);
    VkPipeline createComputePipeline(VkShaderModule vkModule, VkPipelineLayout pipelineLayout,
        VkSpecializationInfo *specializationInfo);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
        AllocatedMemory& bufferMemory, VkMemoryPropertyFlags optionalProperties = 0);

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties, VkImage& image, AllocatedMemory& imageMemory);

    void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);

    void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, size_t width, size_t height);
    void copyImageToBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, size_t width, size_t height);
    void copyImageToImage(VkCommandBuffer commandBuffer, VkImage src, VkImageLayout srcLayout, VkImage dst, VkImageLayout dstLayout,
        size_t width, size_t height);
    void copyBufferToBuffer(VkCommandBuffer commandBuffer, VkBuffer src, VkBuffer dst, size_t size,
        size_t srcOffset = 0, size_t dstOffset = 0);

    void copyDataToHostVisibleMemory(const void *data, size_t size, VkDeviceMemory memory, VkDeviceSize memoryOffset);
    void copyDataFromHostVisibleMemory(void *data, size_t size, VkDeviceMemory memory, VkDeviceSize memoryOffset);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkMemoryPropertyFlags optionalProperties = 0);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    void createCommandPool();
    void createDescriptorPool();

    VkDescriptorSet createDescriptorSet(std::vector<VkDescriptorType> types, const std::vector<Resource *> &resources,
        VkDescriptorSetLayout descriptorSetLayout);
    VkCommandBuffer createCommandBuffer();

    VkFence createFence();
    VkSemaphore createSemaphore();

    VkShaderModule createVkShaderModule(const std::vector<char>& code);

    ShaderModule& getShaderModule(const std::string& shaderPath);

    static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
    static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks* pAllocator);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, 
        VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

    static std::vector<char> readFile(const std::string &filename, bool binaryMode = false);


    Task _createTask(const std::string &shaderPath, VkSpecializationInfo *specializationInfo = nullptr);

    void reflectDescriptorSets(spv_reflect::ShaderModule* reflectModule,
        std::vector<std::vector<ResourceType>>& outLayout,
        std::vector<std::vector<AccessTypeFlags>>& outResourceAccessFlags);
    uint32_t reflectPushConstantSize(spv_reflect::ShaderModule* reflectModule);
};

#endif // JOB_MANAGER_H
