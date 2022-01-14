#include "Job.h"

#include "JobManager.h"
#include "Resources.h"

Job::Job(JobManager *manager, VkCommandBuffer commandBuffer, VkQueue computeQueue, VkFence fence) :
    manager(manager),
    computeQueue(computeQueue),
    commandBuffer(commandBuffer),
    fence(fence),
    signalSemaphore(VK_NULL_HANDLE)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // beginInfo.flags;

    if (computeQueue != VK_NULL_HANDLE && fence != VK_NULL_HANDLE)
    {
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }
    }
}

void Job::setAutoDataDependencyManagement(bool value)
{
    autoDataDependencyManagement = value;
}

void Job::addTask(const Task &task, uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    if (autoDataDependencyManagement && lastRecordedOperation == Operation::Transfer)
    {
        waitAfterTransfers();
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.getPipeline());
    bindPendingResources(task);

    vkCmdDispatch(commandBuffer, groupX, groupY, groupZ);
    lastRecordedOperation = Operation::Task;
}

void Job::addTask(const Task &task, const std::vector<std::vector<Resource *>> &resources,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    if (autoDataDependencyManagement && lastRecordedOperation == Operation::Transfer)
    {
        waitAfterTransfers();
    }

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
    lastRecordedOperation = Operation::Task;
}

void Job::addTask(const Task &task, const std::vector<ResourceSet> &resources,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    if (autoDataDependencyManagement && lastRecordedOperation == Operation::Transfer)
    {
        waitAfterTransfers();
    }

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
    lastRecordedOperation = Operation::Task;
}

void Job::useResources(size_t index, const ResourceSet &resources)
{
    pendingBindings.push_back({index, resources});
}

void Job::useResources(size_t index, const std::vector<Resource *> &resources)
{
    pendingBindings.push_back({index, resources});
}

void Job::syncResourceToDevice(Resource &resource, const void *data, size_t size)
{
    if (resource.getResourceType() == ResourceType::StorageBuffer)
    {
        const Buffer &buffer = static_cast<const Buffer&>(resource);
        size = std::min(size, buffer.getSize());
        switch(buffer.getBufferType())
        {
        case Buffer::Type::DeviceLocal:
        {
            manager->copyDataToHostVisibleMemory(data, size, buffer.getStagingBuffer()->getMemory());

            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, buffer.getStagingBuffer()->getBuffer(), buffer.getBuffer(), 1, &copyRegion);
            lastRecordedOperation = Operation::Transfer;
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
        VkDeviceSize imageSize = image.getSize();

        if (size < imageSize)
            throw std::runtime_error("The size of the passed data is smaller than the size of the image");

        if (data == nullptr)
        {
            transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);
            return;
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        manager->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        manager->copyDataToHostVisibleMemory(data, imageSize, stagingBufferMemory);

        transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        manager->copyBufferToImage(commandBuffer, stagingBuffer, image.getImage(), image.getWidth(), image.getHeight());
        transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);
        // mark staging buffer/memory for deletion after the task is finished
        transfers.push({ new Buffer(stagingBuffer, stagingBufferMemory, 0, Buffer::Type::Staging), 0, nullptr, true });
    }
}

void Job::syncResourceToHost(Resource &resource, void *data, size_t size)
{
    if (autoDataDependencyManagement && lastRecordedOperation == Operation::Task)
    {
        waitBeforeTransfers();
        lastRecordedOperation = Operation::None;
    }

    if (resource.getResourceType() == ResourceType::StorageBuffer)
    {
        const Buffer &buffer = static_cast<const Buffer&>(resource);
        if (buffer.getBufferType() == Buffer::Type::DeviceLocal)
        {
            VkBufferCopy copyRegion{};
            size = std::min(size, buffer.getSize());
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, buffer.getBuffer(), buffer.getStagingBuffer()->getBuffer(), 1, &copyRegion);
            // TODO mark lastRecordedOperation?
            
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
        VkDeviceSize imageSize = image.getSize();

        if (size < imageSize)
            throw std::runtime_error("The size of the passed data is smaller than the size of the image");

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        manager->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        manager->copyImageToBuffer(commandBuffer, stagingBuffer, image.getImage(), image.getWidth(), image.getHeight());
        transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);

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
    // TODO buffer to image, image to buffer
    else
    {
        throw std::runtime_error("Unsupported sync between resources");
    }
    
}

void Job::pushConstants(void *data, size_t size)
{
    std::shared_ptr<void> dst{ new char[size] };
    std::memcpy(dst.get(), data, size);
    pendingConstants = { dst, static_cast<uint32_t>(size) };
}

void Job::waitForTasksFinish()
{
    addMemoryBarrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
}

void Job::waitAfterTransfers()
{
    addMemoryBarrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT /*| VK_ACCESS_TRANSFER_READ_BIT*/,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void Job::waitBeforeTransfers()
{
    addMemoryBarrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT);
}

void Job::addMemoryBarrier(VkPipelineStageFlags srcStageMask, VkAccessFlags srcAccessMask,
    VkPipelineStageFlags dstStageMask, VkAccessFlags dstAccessMask)
{
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;

    vkCmdPipelineBarrier(
        commandBuffer,
        srcStageMask,
        dstStageMask,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr);
}

Semaphore Job::submit(bool signal)
{
    if (!isRecorded)
    {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
        isRecorded = true;
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
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    return { signal ? signalSemaphore : VK_NULL_HANDLE };
}

bool Job::await(uint64_t timeout)
{
    VkResult res = vkWaitForFences(manager->device, 1, &fence, VK_TRUE, timeout);
    if (res != VK_SUCCESS && res != VK_TIMEOUT)
    {
        throw std::runtime_error("Failed to wait for fence!");
    }

    if (res == VK_SUCCESS)
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

    if (pendingConstants.has_value())
    {
        vkCmdPushConstants(
            commandBuffer,
            task.getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            pendingConstants.value().second,
            pendingConstants.value().first.get());
    }

    pendingBindings.clear();
    pendingConstants.reset();
}

void Job::transitionImageLayout(Image &image, VkImageLayout newLayout)
{
    if (image.getLayout() == newLayout)
        return;
    manager->transitionImageLayout(commandBuffer, image.getImage(), image.getLayout(), newLayout);
    image.setLayout(newLayout);
}
