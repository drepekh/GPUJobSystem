#include "Job.h"

#include "JobManager.h"

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

Job& Job::addTask(const Task &task, uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    checkDataDependencyInPendingBindings(task);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, task.getPipeline());
    bindPendingResources(task);

    vkCmdDispatch(commandBuffer, groupX, groupY, groupZ);

    return *this;
}

Job& Job::addTask(const Task &task, const std::vector<std::vector<Resource *>> &resources,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    for (size_t i = 0; i < resources.size(); ++i)
    {
        useResources(i, resources.at(i));
    }

    return addTask(task, groupX, groupY, groupZ);
}

Job& Job::addTask(const Task &task, const std::vector<ResourceSet> &resources,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ)
{
    for (size_t i = 0; i < resources.size(); ++i)
    {
        useResources(i, resources.at(i));
    }

    return addTask(task, groupX, groupY, groupZ);
}

Job& Job::useResources(size_t index, const ResourceSet &resources)
{
    pendingBindings.insert_or_assign(index, resources);

    return *this;
}

Job& Job::useResources(size_t index, const std::vector<Resource *> &resources)
{
    pendingBindings.insert_or_assign(index, resources);

    return *this;
}

Job& Job::syncResourceToDevice(Resource &resource, const void *data, size_t size)
{
    if (resource.getResourceType() == ResourceType::StorageBuffer)
    {
        const Buffer &buffer = static_cast<const Buffer&>(resource);
        size = std::min(size, buffer.getSize());
        switch(buffer.getBufferType())
        {
        case Buffer::Type::DeviceLocal:
        {
            preExecutionTransfers.push_back({ buffer.getStagingBuffer(), size, data, false });

            checkDataDependency({ &resource }, Operation::Transfer, AccessType::Write);
            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, buffer.getStagingBuffer()->getBuffer(), buffer.getBuffer(), 1, &copyRegion);
            break;
        }
        case Buffer::Type::Staging:
        case Buffer::Type::Uniform:
            preExecutionTransfers.push_back({ &buffer, size, data, false });
            break;
        }
    }
    else if (resource.getResourceType() == ResourceType::StorageImage)
    {
        Image &image = static_cast<Image&>(resource);
        VkDeviceSize imageSize = image.getSize();

        if (data == nullptr)
        {
            transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);
            return *this;
        }

        if (size != imageSize)
            throw std::runtime_error("The size of the passed data does not match the size of the image");

        preExecutionTransfers.push_back({ image.getStagingBuffer(), size, data, false });

        transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        manager->copyBufferToImage(commandBuffer, image.getStagingBuffer()->getBuffer(), image.getImage(), image.getWidth(), image.getHeight());
        transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);
    }

    return *this;
}

Job& Job::syncResourceToHost(Resource &resource, void *data, size_t size)
{
    if (resource.getResourceType() == ResourceType::StorageBuffer)
    {
        const Buffer &buffer = static_cast<const Buffer&>(resource);
        if (buffer.getBufferType() == Buffer::Type::DeviceLocal)
        {
            checkDataDependency({ &resource }, Operation::Transfer, AccessType::Read);
            
            VkBufferCopy copyRegion{};
            size = std::min(size, buffer.getSize());
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, buffer.getBuffer(), buffer.getStagingBuffer()->getBuffer(), 1, &copyRegion);
            
            postExecutionTransfers.push_back({ buffer.getStagingBuffer(), size, data });
        }
        else
        {
            postExecutionTransfers.push_back({ &buffer, size, data });
        }
    }
    else if (resource.getResourceType() == ResourceType::StorageImage)
    {
        Image &image = static_cast<Image&>(resource);
        VkDeviceSize imageSize = image.getSize();

        if (size < imageSize)
            throw std::runtime_error("The size of the passed data is smaller than the size of the image");

        transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        manager->copyImageToBuffer(commandBuffer, image.getStagingBuffer()->getBuffer(), image.getImage(), image.getWidth(), image.getHeight());
        transitionImageLayout(image, VK_IMAGE_LAYOUT_GENERAL);

        postExecutionTransfers.push_back({ image.getStagingBuffer(), imageSize, data });
    }

    return *this;
}

Job& Job::syncResources(Resource &src, Resource &dst)
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

        checkDataDependency({ &src, &dst }, Operation::Transfer, { AccessType::Read, AccessType::Write });

        manager->copyBufferToBuffer(commandBuffer, srcBuffer.getBuffer(), dstBuffer.getBuffer(),
            std::min(srcBuffer.getSize(), dstBuffer.getSize()));
    }
    // TODO buffer to image, image to buffer
    else
    {
        throw std::runtime_error("Unsupported sync between resources");
    }

    return *this;
}

Job& Job::pushConstants(const void *data, size_t size)
{
    std::shared_ptr<void> dst{ new char[size] };
    std::memcpy(dst.get(), data, size);
    pendingConstants = { dst, static_cast<uint32_t>(size) };

    return *this;
}

Job& Job::waitForTasksFinish()
{
    unguardedResourceAccess.clear();
    addMemoryBarrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);

    return *this;
}

Job& Job::waitAfterTransfers()
{
    addMemoryBarrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT /*| VK_ACCESS_TRANSFER_READ_BIT*/,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    return *this;
}

Job& Job::waitBeforeTransfers()
{
    addMemoryBarrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT);

    return *this;
}

Job& Job::addMemoryBarrier(VkPipelineStageFlags srcStageMask, VkAccessFlags srcAccessMask,
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
    
    return *this;
}

Job& Job::addExecutionBarrier(VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask)
{
    vkCmdPipelineBarrier(
        commandBuffer,
        srcStageMask,
        dstStageMask,
        0,
        0, nullptr,
        0, nullptr,
        0, nullptr);

    return *this;
}

Semaphore Job::submit(bool signal, const std::vector<VkSemaphore>& waitSemaphores)
{
    if (!isRecorded)
    {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
        isRecorded = true;
    }

    if (isSubmitted)
    {
        throw std::runtime_error("Tried to submit job again without awaiting for its completion");
    }

    completePreExecutionTransfers();

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

    if (waitSemaphores.size() != 0)
    {
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = waitSemaphores.data();
    }

    vkResetFences(manager->device, 1, &fence);

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    isSubmitted = true;

    return { signal ? signalSemaphore : VK_NULL_HANDLE };
}

Job& Job::submit()
{
    submit(false);

    return *this;
}

bool Job::await(uint64_t timeout)
{
    VkResult res = vkWaitForFences(manager->device, 1, &fence, VK_TRUE, timeout);
    if (res != VK_SUCCESS && res != VK_TIMEOUT)
    {
        throw std::runtime_error("Failed to wait for fence!");
    }

    if (res == VK_SUCCESS)
    {
        completePostExecutionTransfers();
        isSubmitted = false;
    }
    
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

void Job::completePostExecutionTransfers()
{
    for (size_t i = 0; i < postExecutionTransfers.size(); ++i)
    {
        const auto& transferInfo = postExecutionTransfers[i];

        if (transferInfo.hostBuffer != nullptr)
            manager->copyDataFromHostVisibleMemory(transferInfo.hostBuffer, transferInfo.size, transferInfo.deviceBuffer->GetAllocatedMemory());

        if (transferInfo.destroyAfterTransfer)
        {
            vkDestroyBuffer(manager->device, transferInfo.deviceBuffer->getBuffer(), nullptr);
            vkFreeMemory(manager->device, transferInfo.deviceBuffer->getMemory(), nullptr); // TODO use allocator
            delete transferInfo.deviceBuffer;
            
            postExecutionTransfers.erase(postExecutionTransfers.begin() + i);
            --i;
        }
    }
}

void Job::completePreExecutionTransfers()
{
    for (size_t i = 0; i < preExecutionTransfers.size(); ++i)
    {
        const auto& transferInfo = preExecutionTransfers[i];

        if (transferInfo.hostBuffer != nullptr)
            manager->copyDataToHostVisibleMemory(transferInfo.hostBuffer, transferInfo.size, transferInfo.deviceBuffer->GetAllocatedMemory());

        if (transferInfo.destroyAfterTransfer)
        {
            vkDestroyBuffer(manager->device, transferInfo.deviceBuffer->getBuffer(), nullptr);
            vkFreeMemory(manager->device, transferInfo.deviceBuffer->getMemory(), nullptr); // TODO use allocator
            delete transferInfo.deviceBuffer;
            
            preExecutionTransfers.erase(preExecutionTransfers.begin() + i);
            --i;
        }
    }
}

void Job::bindPendingResources(const Task &task)
{
    std::vector<VkDescriptorSet> descriptorSets;
    size_t currentFirstPos = pendingBindings.size() > 0 ? pendingBindings.begin()->first : 0;
    for (const auto &[pos, resources]: pendingBindings)
    {
        if (pos != currentFirstPos + descriptorSets.size())
        {
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                task.getPipelineLayout(),
                static_cast<uint32_t>(currentFirstPos),
                static_cast<uint32_t>(descriptorSets.size()),
                descriptorSets.data(),
                0, nullptr);
            
            currentFirstPos = pos;
            descriptorSets.clear();
        }

        if (std::holds_alternative<ResourceSet>(resources))
        {
            descriptorSets.push_back(std::get<ResourceSet>(resources).getDescriptorSet());
        }
        else
        {
            const auto &val = std::get<std::vector<Resource *>>(resources);
            descriptorSets.push_back(
                manager->createDescriptorSet(
                    resourceToDescriptorType(val),
                    val,
                    task.getDescriptorSetLayout(pos)));
        }
    }
    if (descriptorSets.size() > 0)
    {
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            task.getPipelineLayout(),
            static_cast<uint32_t>(currentFirstPos),
            static_cast<uint32_t>(descriptorSets.size()),
            descriptorSets.data(),
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

void Job::checkDataDependencyInPendingBindings(const Task& task)
{
    if (!autoDataDependencyManagement)
        return;

    const auto& accessFlags = task.getResourceAccessFlags();
    std::vector<Resource *> allResources;
    std::vector<AccessTypeFlags> allAccessFlags;
    for (const auto &[pos, resources] : pendingBindings)
    {
        const auto &rs = std::holds_alternative<ResourceSet>(resources) ?
            std::get<ResourceSet>(resources).getResources() :
            std::get<std::vector<Resource *>>(resources);
        
        if (pos >= accessFlags.size() || rs.size() > accessFlags[pos].size())
            throw std::runtime_error("Binded resources does not match shader layout");
        
        allResources.insert(allResources.end(), rs.begin(), rs.end());
        allAccessFlags.insert(allAccessFlags.end(), accessFlags[pos].begin(), accessFlags[pos].begin() + rs.size());
    }

    checkDataDependency(allResources, Operation::Task, allAccessFlags);
}

void Job::checkDataDependency(const std::vector<Resource *> &requiredResources,
    Operation accessStage, AccessTypeFlags accessType)
{
    std::vector<AccessTypeFlags> accessTypes(requiredResources.size(), accessType);
    checkDataDependency(requiredResources, accessStage, accessTypes);
}

void Job::checkDataDependency(const std::vector<Resource *> &requiredResources,
    Operation accessStage, const std::vector<AccessTypeFlags> &accessTypes)
{
    if (!autoDataDependencyManagement)
        return;
    
    if (requiredResources.size() != accessTypes.size())
        throw std::runtime_error("Number of resources does not match number of elements in accessTypes array");
    
    // Different calls for different source stages
    // TODO switch to synchronization2 ?
    std::vector<VkBufferMemoryBarrier> shaderStageBarriers;
    std::vector<VkBufferMemoryBarrier> transferStageBarriers;

    // Accumulate access types for every unique resource
    std::map<Resource *, AccessTypeFlags> uniqueResources;
    for (size_t i = 0; i < requiredResources.size(); ++i)
    {
        auto it = uniqueResources.find(requiredResources.at(i));
        if (it != uniqueResources.end())
        {
            it->second |= accessTypes.at(i);
        }
        else
        {
            uniqueResources.insert({ requiredResources.at(i), accessTypes.at(i) });
        }
    }

    for (const auto &[resource, accessTypeFlags] : uniqueResources)
    {
        auto it = unguardedResourceAccess.find(resource);
        if (it != unguardedResourceAccess.end())
        {
            const auto &info = it->second;

            // two read operations do not require synchronization
            if (info.accessType == AccessType::Read && accessTypeFlags == AccessType::Read)
                continue;

            if (info.accessType == AccessType::None || accessTypeFlags == AccessType::None)
                continue;

            if (it->first->getResourceType() == ResourceType::StorageBuffer)
            {
                const auto* buffer = static_cast<const Buffer *>(it->first);

                auto [srcStage, srcAccessMask] = mapStageAndAccessMask(info.accessStage, info.accessType);
                auto [dstStage, dstAccessMask] = mapStageAndAccessMask(accessStage, accessTypeFlags);
                
                if (info.accessStage == Operation::Task)
                {
                    shaderStageBarriers.push_back(makeBufferMemoryBarrier(
                        *buffer, srcAccessMask, dstAccessMask));
                }
                else if (info.accessStage == Operation::Transfer)
                {
                    transferStageBarriers.push_back(makeBufferMemoryBarrier(
                        *buffer, srcAccessMask, dstAccessMask));
                }
            }
            else if (it->first->getResourceType() == ResourceType::StorageImage)
            {
                // TODO images
                throw std::runtime_error("Unsupported resource type");
            }
            else
            {
                throw std::runtime_error("Unsupported resource type");
            }
        }

        unguardedResourceAccess.insert_or_assign(resource, ResourceAccesInfo{ accessTypeFlags, accessStage });
    }

    auto dstStage = mapStage(accessStage);

    if (shaderStageBarriers.size() > 0)
    {
        addResourceMemoryBarriers(shaderStageBarriers, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dstStage);
    }

    if (transferStageBarriers.size() > 0)
    {
        addResourceMemoryBarriers(transferStageBarriers, {}, VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage);
    }
}

VkBufferMemoryBarrier Job::makeBufferMemoryBarrier(const Buffer &buffer,
    VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask)
{
    VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer.getBuffer();
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
    
    return barrier;
}

void Job::addResourceMemoryBarriers(
    const std::vector<VkBufferMemoryBarrier> &bufferBarriers,
    const std::vector<VkImageMemoryBarrier> &imageBarriers,
    VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask)
{
    vkCmdPipelineBarrier(
        commandBuffer,
        srcStageMask,
        dstStageMask,
        0,
        0, nullptr,
        static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.size() ? bufferBarriers.data() : nullptr,
        static_cast<uint32_t>(imageBarriers.size()), imageBarriers.size() ? imageBarriers.data() : nullptr);
}

std::pair<VkPipelineStageFlags, VkAccessFlags> Job::mapStageAndAccessMask(Operation accessStage, AccessTypeFlags accessType)
{
    VkAccessFlags accessFlags = 0;
    switch (accessStage)
    {
        case Operation::Task:
        {
            if (accessType & AccessType::Read)
                accessFlags |= VK_ACCESS_SHADER_READ_BIT;
            if (accessType & AccessType::Write)
                accessFlags |= VK_ACCESS_SHADER_WRITE_BIT;
            return { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, accessFlags };
        }
        case Operation::Transfer:
        {
            if (accessType & AccessType::Read)
                accessFlags |= VK_ACCESS_TRANSFER_READ_BIT;
            if (accessType & AccessType::Write)
                accessFlags |= VK_ACCESS_TRANSFER_WRITE_BIT;
            return { VK_PIPELINE_STAGE_TRANSFER_BIT, accessFlags };
        }
        default:
            throw std::runtime_error("Unsupported operation");
    }
}

VkPipelineStageFlags Job::mapStage(Operation accessStage)
{
    switch (accessStage)
    {
        case Operation::Task: return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        case Operation::Transfer: return VK_PIPELINE_STAGE_TRANSFER_BIT;
        default:
            throw std::runtime_error("Unsupported operation");
    }
}
