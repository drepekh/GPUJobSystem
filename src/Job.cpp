#include "Job.h"

#include "JobManager.h"
#include "Resources.h"

Job::Job(JobManager *manager, VkQueue computeQueue, VkCommandBuffer commandBuffer, VkFence fence) :
    manager(manager),
    computeQueue(computeQueue),
    commandBuffer(commandBuffer),
    fence(fence),
    signalSemaphore(VK_NULL_HANDLE)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // beginInfo.flags;

    if (fence != VK_NULL_HANDLE)
    {
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
    }
}

void Job::addTask(const Task &task, uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.getPipeline());
    bindPendingResources(task);

    vkCmdDispatch(commandBuffer, groupX, groupY, groupZ);
}

void Job::addTask(const Task &task, const std::vector<std::vector<Resource *>> &resources,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.getPipeline());
    bindPendingResources(task);

    if (resources.size() > 0)
    {
        std::vector<VkDescriptorSet> descriptorSets;
        descriptorSets.reserve(resources.size());
        for (size_t i = 0; i < resources.size(); ++i)
        {
            descriptorSets.push_back(manager->createDescriptorSet(
                resourceToDescriptorType(resources.at(i)),
                resources.at(i),
                task.getDescriptorSetLayout(i)
            ));
        }

        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            task.getPipelineLayout(),
            0,
            static_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            0, nullptr);
    }

    vkCmdDispatch(commandBuffer, groupX, groupY, groupZ);
}

void Job::addTask(const Task &task, const std::vector<ResourceSet> &resources,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.getPipeline());
    bindPendingResources(task);

    if (resources.size() > 0)
    {
        std::vector<VkDescriptorSet> descriptorSets;
        descriptorSets.reserve(resources.size());
        for (size_t i = 0; i < resources.size(); ++i)
        {
            descriptorSets.push_back(resources[i].getDescriptorSet());
        }

        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            task.getPipelineLayout(),
            0,
            static_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
            0, nullptr);
    }

    vkCmdDispatch(commandBuffer, groupX, groupY, groupZ);
}

void Job::useResources(size_t index, const ResourceSet &resources)
{
    pendingBindings.push_back({index, resources});
}

void Job::useResources(size_t index, const std::vector<Resource *> &resources)
{
    pendingBindings.push_back({index, resources});
}

void Job::syncResourceToDevice(Resource &resource, const void *data, size_t size, bool waitTillTransferDone)
{
    if (resource.getResourceType() == ResourceType::StorageBuffer)
    {
        const Buffer &buffer = static_cast<const Buffer&>(resource);
        switch(buffer.getBufferType())
        {
        case Buffer::Type::DeviceLocal:
        {
            manager->copyDataToHostVisibleMemory(data, size, buffer.getStagingBuffer()->getMemory());

            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, buffer.getStagingBuffer()->getBuffer(), buffer.getBuffer(), 1, &copyRegion);
            break;
        }
        case Buffer::Type::Staging:
        case Buffer::Type::Uniform:
            manager->copyDataToHostVisibleMemory(data, size, buffer.getMemory());
            break;
        }
    }
    else if (resource.getResourceType() == ResourceType::StorageImage)
    {
        Image &image = static_cast<Image&>(resource);
        VkDeviceSize imageSize = image.getWidth() * image.getHeight() * 4;

        if (data == nullptr)
        {
            transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);
            return;
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        manager->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        manager->copyDataToHostVisibleMemory(data, size, stagingBufferMemory);

        transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        manager->copyBufferToImage(commandBuffer, stagingBuffer, image.getImage(), image.getWidth(), image.getHeight());
        transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);
        image.setLayout(VK_IMAGE_LAYOUT_GENERAL);

        transfers.push({ new Buffer(stagingBuffer, stagingBufferMemory, 0, Buffer::Type::Staging), 0, nullptr, true });
    }

    if (waitTillTransferDone)
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

void Job::syncResourceToHost(Resource &resource, void *data, size_t size, bool waitTillShaderDone)
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

    if (resource.getResourceType() == ResourceType::StorageBuffer)
    {
        const Buffer &buffer = static_cast<const Buffer&>(resource);
        if (buffer.getBufferType() == Buffer::Type::DeviceLocal)
        {
            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, buffer.getBuffer(), buffer.getStagingBuffer()->getBuffer(), 1, &copyRegion);

            transfers.push({ buffer.getStagingBuffer(), size, data });
        }
        else
        {
            transfers.push({ &buffer, size, data });
        }
    }
    else if (resource.getResourceType() == ResourceType::StorageImage)
    {
        Image &image = static_cast<Image&>(resource);
        VkDeviceSize imageSize = image.getWidth() * image.getHeight() * 4;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        manager->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        manager->copyImageToBuffer(commandBuffer, stagingBuffer, image.getImage(), image.getWidth(), image.getHeight());
        transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);
        image.setLayout(VK_IMAGE_LAYOUT_GENERAL);

        Buffer *buffer = new Buffer(stagingBuffer, stagingBufferMemory, imageSize, Buffer::Type::Staging);
        
        transfers.push({ buffer, imageSize, data, true });
    }
    
}

void Job::syncResources(Resource &src, Resource &dst)
{
    if (src.getResourceType() == ResourceType::StorageImage && dst.getResourceType() == ResourceType::StorageImage)
    {
        Image &srcImg = static_cast<Image&>(src);
        Image &dstImg = static_cast<Image&>(dst);

        transitionImageLayout(srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        transitionImageLayout(dstImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        manager->copyImageToImage(commandBuffer, srcImg.getImage(), srcImg.getLayout(), dstImg.getImage(), dstImg.getLayout(), 
            std::min(srcImg.getWidth(), dstImg.getWidth()), std::min(srcImg.getHeight(), dstImg.getHeight()));
        transitionImageLayout(srcImg, VK_IMAGE_LAYOUT_GENERAL);
        transitionImageLayout(dstImg, VK_IMAGE_LAYOUT_GENERAL);
    }
    else if (src.getResourceType() == ResourceType::StorageBuffer && dst.getResourceType() == ResourceType::StorageBuffer)
    {
        Buffer &srcBuffer = static_cast<Buffer&>(src);
        Buffer &dstBuffer = static_cast<Buffer&>(dst);

        manager->copyBufferToBuffer(commandBuffer, srcBuffer.getBuffer(), dstBuffer.getBuffer(),
            std::min(srcBuffer.getSize(), dstBuffer.getSize()));
    }
    else
    {
        throw std::runtime_error("Unsupported sync between resources");
    }
    
}

void Job::waitForTasksFinish()
{
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr);
}

Semaphore Job::submit(bool signal)
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

    if (signal)
    {
        if (signalSemaphore == VK_NULL_HANDLE)
        {
            signalSemaphore = manager->createSemaphore();
        }

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &signalSemaphore;
    }

    vkResetFences(manager->device, 1, &fence);

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    transfersComplete = false;

    return { signal ? signalSemaphore : VK_NULL_HANDLE };
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

VkCommandBuffer Job::getCommandBuffer() const
{
    return commandBuffer;
}

void Job::completeTransfers()
{
    while (!transfers.empty())
    {
        auto transferInfo = transfers.front();
        transfers.pop();

        if (transferInfo.dst != nullptr)
            manager->copyDataFromHostVisibleMemory(transferInfo.dst, transferInfo.size, transferInfo.buffer->getMemory());

        if (transferInfo.destroyAfterTransfer)
        {
            vkDestroyBuffer(manager->device, transferInfo.buffer->getBuffer(), nullptr);
            vkFreeMemory(manager->device, transferInfo.buffer->getMemory(), nullptr);
            delete transferInfo.buffer;
        }
    }

    transfersComplete = true;
}

void Job::bindPendingResources(const Task &task)
{
    for (const auto &resource: pendingBindings)
    {
        VkDescriptorSet descriptorSet;
        if (std::holds_alternative<ResourceSet>(resource.second))
        {
            descriptorSet = std::get<ResourceSet>(resource.second).getDescriptorSet();
        }
        else
        {
            const auto &val = std::get<std::vector<Resource *>>(resource.second);
            descriptorSet = manager->createDescriptorSet(
                resourceToDescriptorType(val),
                val,
                task.getDescriptorSetLayout(resource.first)
            );
        }
        
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            task.getPipelineLayout(),
            static_cast<uint32_t>(resource.first),
            1,
            &descriptorSet,
            0, nullptr);
    }

    pendingBindings.clear();
}

void Job::transitionImageLayout(Image &image, VkImageLayout newLayout)
{
    if (image.getLayout() == newLayout)
        return;
    manager->transitionImageLayout(commandBuffer, image.getImage(), image.getLayout(), newLayout);
    image.setLayout(newLayout);
}
