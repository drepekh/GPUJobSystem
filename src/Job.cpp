#include "Job.h"

#include "JobManager.h"

Job::Job(JobManager *manager, VkQueue computeQueue, VkCommandBuffer commandBuffer, VkFence fence) :
    manager(manager),
    computeQueue(computeQueue),
    commandBuffer(commandBuffer),
    fence(fence)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // beginInfo.flags;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to begin recording command buffer!");
    }
}

void Job::addTask(const Task &task, const Buffer &buffer, uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.getPipeline());
    auto descriptorSet = buffer.getDescriptorSet();
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.getPipelineLayout(), 0, 1,
        &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, groupX, groupY, groupZ);
}

void Job::syncResourceToDevice(const Buffer &buffer, void *data, size_t size, bool waitTillDone)
{    
    void* stagingData;
    vkMapMemory(manager->device, buffer.getStagingBuffer()->getMemory(), 0, size, 0, &stagingData);
        memcpy(stagingData, data, size);
    vkUnmapMemory(manager->device, buffer.getStagingBuffer()->getMemory());

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, buffer.getStagingBuffer()->getBuffer(), buffer.getBuffer(), 1, &copyRegion);

    if (waitTillDone)
    {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        // barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        // barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        // barrier.buffer = buffer.getBuffer();
        // barrier.offset = 0;
        // barrier.size = size;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &barrier,
            0, nullptr,
            0, nullptr);
    }
}

void Job::syncResourceToHost(const Buffer &buffer, void *data, size_t size, bool waitTillShaderDone)
{
    if (waitTillShaderDone)
    {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            1, &barrier,
            0, nullptr,
            0, nullptr);
    }

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, buffer.getBuffer(), buffer.getStagingBuffer()->getBuffer(), 1, &copyRegion);

    transfers.push({ &buffer, size, data });
}

void Job::submit()
{
    if (!recorded)
    {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
        recorded = true;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkResetFences(manager->device, 1, &fence);

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    transfersComplete = false;
}

bool Job::await(uint64_t timeout)
{
    VkResult res = vkWaitForFences(manager->device, 1, &fence, VK_TRUE, timeout);
    if (res != VK_SUCCESS && res != VK_TIMEOUT)
    {
        throw std::runtime_error("Failed to wait for fence!");
    }

    if (res == VK_SUCCESS && !transfersComplete)
        completeTransfers();
    
    return res == VK_SUCCESS;
}

bool Job::isComplete()
{
    return await(0);
}

void Job::completeTransfers()
{
    while (!transfers.empty())
    {
        auto transferInfo = transfers.front();
        transfers.pop();

        void* stagingData;
        vkMapMemory(manager->device, transferInfo.buffer->getStagingBuffer()->getMemory(), 0, transferInfo.size, 0, &stagingData);
            memcpy(transferInfo.dst, stagingData, transferInfo.size);
        vkUnmapMemory(manager->device, transferInfo.buffer->getStagingBuffer()->getMemory());
    }

    transfersComplete = true;
}
