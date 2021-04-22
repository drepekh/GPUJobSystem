#ifndef JOB_H
#define JOB_H

#include <vulkan/vulkan.h>
#include <queue>

class JobManager;
class Task;
class Buffer;
class Resource;

class Job
{
    class JobManager *manager;

    VkQueue computeQueue;
    VkCommandBuffer commandBuffer;
    VkFence fence;
    bool recorded = false;
    bool transfersComplete = true;

    struct TransferInfo
    {
        const Buffer *buffer;
        size_t size;
        void *dst;
    };

    std::queue<TransferInfo> transfers;

public:

    Job(JobManager *manager, VkQueue computeQueue, VkCommandBuffer commandBuffer, VkFence fence);
    
    void addTask(const Task &, const std::vector<std::pair<size_t, std::vector<Resource *>>> &resources,
        uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1);
    void syncResourceToDevice(const Buffer &buffer, void *data, size_t size, bool waitTillTransferDone = true);
    void syncResourceToHost(const Buffer &buffer, void *data, size_t size, bool waitTillShaderDone = true);
    void submit();
    bool await(uint64_t timeout = UINT64_MAX);
    bool isComplete();

private:

    void completeTransfers();
};

#endif // JOB_H
