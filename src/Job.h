#ifndef JOB_H
#define JOB_H

#include <vulkan/vulkan.h>
#include <queue>
#include <variant>

class JobManager;
class Task;
class Buffer;
class Resource;
class ResourceSet;

class Job
{
    class JobManager *manager;

    VkQueue computeQueue;
    VkCommandBuffer commandBuffer;
    VkFence fence;
    bool recorded = false;
    bool transfersComplete = true;
    std::vector<std::pair<size_t, std::variant<ResourceSet, std::vector<Resource *>>>> pendingBindings;

    struct TransferInfo
    {
        const Buffer *buffer;
        size_t size;
        void *dst;
    };

    std::queue<TransferInfo> transfers;

public:

    Job(JobManager *manager, VkQueue computeQueue, VkCommandBuffer commandBuffer, VkFence fence);
    
    void addTask(const Task &, uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1);
    void addTask(const Task &, const std::vector<std::vector<Resource *>> &resources,
        uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1);
    void addTask(const Task &task, const std::vector<ResourceSet> &resources,
        uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1);
    
    void useResources(size_t, const ResourceSet &);
    void useResources(size_t, const std::vector<Resource *> &);
    
    void syncResourceToDevice(const Buffer &buffer, void *data, size_t size, bool waitTillTransferDone = true);
    void syncResourceToHost(const Buffer &buffer, void *data, size_t size, bool waitTillShaderDone = true);

    void waitForTasksFinish();

    void submit();
    bool await(uint64_t timeout = UINT64_MAX);
    bool isComplete();

private:

    void completeTransfers();
    void bindPendingResources(const Task &);
};

#endif // JOB_H
