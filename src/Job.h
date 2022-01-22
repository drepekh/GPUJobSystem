#ifndef JOB_H
#define JOB_H

#include <vulkan/vulkan.h>
#include <variant>
#include <memory>
#include <optional>
#include <map>
#include <vector>

class JobManager;
class Task;
class Buffer;
class Image;
class Resource;
class ResourceSet;
class Semaphore;


/**
 * @brief Unit of work that can record various operations and submit them to the
 * GPU in batches.
 * 
 * It is possible to resubmit single job multiple times,
 * however it is not possible to add operations after job was submitted
 * for the first. Job has to be in completed state before it can be
 * submited again.
 */
class Job
{
    class JobManager *manager;

    VkQueue computeQueue;
    VkCommandBuffer commandBuffer;
    VkFence fence;
    VkSemaphore signalSemaphore;

    bool isRecorded = false;
    bool autoDataDependencyManagement = true;

    std::map<size_t, std::variant<ResourceSet, std::vector<Resource *>>> pendingBindings;
    std::optional<std::pair<std::shared_ptr<void>, uint32_t>> pendingConstants;

    template <typename T>
    struct TransferInfo
    {
        const Buffer *deviceBuffer;
        size_t size;
        T hostBuffer;
        bool destroyAfterTransfer = false;
    };
    using TransferInfoToHost = TransferInfo<void *>;
    using TransferInfoFromHost = TransferInfo<const void *>;

    // Pending host to host-visible transfers
    std::vector<TransferInfoFromHost> preExecutionTransfers;
    // Pending host-visible to host transfers
    std::vector<TransferInfoToHost> postExecutionTransfers;

    enum class Operation {
        None,
        Transfer,
        Task
    };

    enum AccessType : uint8_t {
        Read = 1,
        Write = 2
    };
    using AccessTypeFlags = uint8_t;

    struct ResourceAccesInfo
    {
        AccessTypeFlags accessType = 0;
        Operation accessStage = Operation::None;
    };

    // information about last access to resource that was not synchronised with barrier
    std::map<const Resource*, ResourceAccesInfo> unguardedResourceAccess;

public:
    /**
     * @brief Initialize object.
     * 
     * Begins command buffer if both computeQueue and fence is not null.
     * Otherwise job can be used only as wrapper for command buffer and functions
     * like submit() and await() should never be called. If called not from JobManager,
     * make sure that instance of job manager possesses same physical and logical
     * devices that were used to create command buffer.
     * 
     * @param manager Instance of JobManager class
     * @param computeQueue Queue to which recorded commands will be submitted
     * @param commandBuffer Command buffer where all commands will be recorded
     * @param fence Fence used to determine whether submitted work is done
     */
    Job(JobManager *manager, VkCommandBuffer commandBuffer, VkQueue computeQueue, VkFence fence);

    /**
     * @brief Set the Auto Data Dependency Management setting.
     * 
     * When set to true, job manager will automatically place barriers between 
     * transfers and tasks. Set to false if you want to manually control memory
     * barriers placement. You might find this useful when integrating manager
     * with external pipeline. Should be called only prior to any other
     * recording command. Right now this supports only transfer-task and 
     * taks-transfer barriers. Task-task and transfer-transfer barriers should
     * be placed manually regardless of the setting (see waitForTasksFinish()
     * and waitAfterTransfers() for more details).
     * 
     * @param value New setting value.
     */
    void setAutoDataDependencyManagement(bool newValue);
    
    /**
     * @brief Add task execution to the job.
     * 
     * Records shader dispatch command into underlying command buffer.
     * 
     * @param task Task that sould be executed on the GPU
     * @param groupX Number of local workgroups to dispatch in the X dimension
     * @param groupY Number of local workgroups to dispatch in the Y dimension
     * @param groupZ Number of local workgroups to dispatch in the Z dimension
     */
    void addTask(const Task &task, uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1);

    /**
     * @brief Add task execution to the job.
     * 
     * Binds GPU resources (if any provided) to be used in the task and records
     * shader dispatch command into underlying command buffer.
     * 
     * @param task Task that sould be executed
     * @param resources Resources that should be bound to the task and used during
     * its execution on the GPU. Each element of the vector is interpreted as a separate
     * ResourceSet (DescriptorSet)
     * @param groupX Number of local workgroups to dispatch in the X dimension
     * @param groupY Number of local workgroups to dispatch in the Y dimension
     * @param groupZ Number of local workgroups to dispatch in the Z dimension
     */
    void addTask(const Task &task, const std::vector<std::vector<Resource *>> &resources,
        uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1);
    
    /**
     * @brief Add task execution to the job.
     * 
     * Binds GPU resources (if any provided) to be used in the task and records
     * shader dispatch command into underlying command buffer.
     * 
     * @param task Task that sould be executed
     * @param resources Resources that should be bound to the task and used during
     * its execution on the GPU
     * @param groupX Number of local workgroups to dispatch in the X dimension
     * @param groupY Number of local workgroups to dispatch in the Y dimension
     * @param groupZ Number of local workgroups to dispatch in the Z dimension
     */
    void addTask(const Task &task, const std::vector<ResourceSet> &resources,
        uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1);
    
    /**
     * @brief Bind resources that should be used during the execution of the next
     * added task. 
     * 
     * @param set Set number of the resources to be bound
     * @param resources Resource set that should be bound
     */
    void useResources(size_t set, const ResourceSet &resources);

    /**
     * @brief Bind resources that should be used during the execution of the next
     * added task.
     * 
     * Content of the vector will be turned into single ResourceSet.
     * 
     * @param set Set number of the resources to be bound
     * @param resources Array of the resources that should be bound as a single
     * ResourceSet
     */
    void useResources(size_t set, const std::vector<Resource *> &resources);

    /**
     * @brief Copy data from the host to the device.
     * 
     * Copying to the host-visible memory takes place either when the submit()
     * is called or manually with the completePreExecutionTransfers() call,
     * meaning that \p data should be available at the moment of either of these calls.
     * If needed, copy command is added to the command buffer to copy data from
     * host-visible to device-local data. Actual amount of bytes that will be copied
     * is minimum between \p size and actual size of the resource. Should be called
     * at least once on every image resource before using it in the task, even if
     * there is no data to copy - in such cases \p data should be set to nullptr.
     * 
     * @param resource Resource that will be a destination for this copy operation
     * @param data Source for the copy command, allocated on the host. Could be set to
     * nullptr to prepare image layout
     * @param size Amount of bytes to copy
     */
    void syncResourceToDevice(Resource &resource, const void *data, size_t size = UINT64_MAX);

    /**
     * @brief Copy data from the device to the host.
     * 
     * If needed, copy command is added to the command buffer to copy data from
     * device-local to host-visible momory. Copying from the host-visible to the
     * host memory takes place only on call to await(). Actual amount of bytes
     * that will be copied is minimum between \p size and actual size of the
     * resource.
     * 
     * @param resource Resource that will be a source for this copy operation
     * @param data Destination for the copy command, allocated on the host
     * @param size Amount of bytes to copy
     */
    void syncResourceToHost(Resource &resource, void *data, size_t size = UINT64_MAX);

    /**
     * @brief Copy data from one resource to another.
     * 
     * Copying takes place entirely on the GPU without transfers to/from host.
     * 
     * @param src Source for copy operation
     * @param dst Destination for copy operation
     */
    void syncResources(Resource &src, Resource &dst);

    /**
     * @brief Push constants that should be used by the next added task.
     * 
     * Only stores copy of the passed data, actual command is written to the command
     * buffer on the next call of addTask().
     * 
     * @param data Location of the data that should be pushed to the device
     * @param size Size of the data
     */
    void pushConstants(void *data, size_t size);

    /**
     * @brief Wait until tasks are completed.
     * 
     * Make GPU wait for the completion of the previously added tasks before
     * execution of the tasks that are added after this function's call.
     * Usuful in cases when result (data) of one task should be used by
     * another task.
     */
    void waitForTasksFinish();

    /**
     * @brief Wait until resource transfers are complete.
     * 
     * Make GPU wait for the completion of the previously recorded resource
     * transfers, thereby making them safe to use in the tasks that are added
     * after call to this function.
     * 
     */
    void waitAfterTransfers();

    /**
     * @brief Wait until tasks finish writing resources before they can be
     * safely transfered or copied.
     * 
     * Make GPU wait for the completion of the previously added tasks before
     * execution of the followed transfer operations. This ensures that tasks
     * finish write operations on the resources and valid data is transfered.
     * 
     */
    void waitBeforeTransfers();

    /**
     * @brief Record memory berrier.
     * 
     * Records global memory barrier with specific source and destination
     * stages and masks.
     * 
     * @param srcStageMask Source stage mask
     * @param srcAccessMask Source access mask
     * @param dstStageMask Destination stage mask
     * @param dstAccessMask Destination access mask
     */
    void addMemoryBarrier(VkPipelineStageFlags srcStageMask, VkAccessFlags srcAccessMask,
        VkPipelineStageFlags dstStageMask, VkAccessFlags dstAccessMask);

    /**
     * @brief Submit job with all recorded operations to be executed on the GPU.
     * 
     * Resulting semaphore can be used to chain operations on the GPUs pipeline.
     * 
     * @param signal Indicates whether semaphore that signals job completion
     * should be created.
     * @return Empty Semaphore object if \p signal is false, otherwise - Sempahore
     * object that signals when the job is completed.
     */
    Semaphore submit(bool signal = false);

    /**
     * @brief Wait for the GPU to finish operations sumbitted by this job.
     * 
     * After GPU-side operations are done, it also completes CPU-side pending 
     * operations such as device-host data transfers.
     * 
     * @param timeout Timeout period in units of nanoseconds
     * @return True if the job was successfully completed, false if timeout has
     * expired  
     */
    bool await(uint64_t timeout = UINT64_MAX);

    /**
     * @brief Check whether job is complete. Non-blocking call.
     */
    bool isComplete();

    /**
     * @brief Get underlying command buffer.
     * 
     * @return VkCommandBuffer
     */
    VkCommandBuffer getCommandBuffer() const;

    /**
     * @brief Manually transition image layout.
     * 
     * Should be used only when integrating job system into external pipeline.
     * Inside the job system layouts managed automatically.
     * 
     * @param image Image which layout should be changed
     * @param layout New layout for image
     */
    void transitionImageLayout(Image &image, VkImageLayout layout);

    /**
     * @brief Complete transfers from device to host.
     * 
     * Should be called in cases when commands were submitted from outside Job
     * class (meaning submit() was not called) and only after all submited work
     * is done.
     */
    void completePostExecutionTransfers();

    /**
     * @brief Complete transfers from host memory to host-visible memory of
     * the device.
     * 
     * Usually Job automatically takes care of this by calling this function
     * from inside the submit(), but you might want to call it manually when
     * integrating JobManager with the external pipeline where command buffer
     * is submited outside of the Job class (i.e. submit() is never called).
     * 
     */
    void completePreExecutionTransfers();

private:
    void bindPendingResources(const Task &);

    void checkDataDependencyInPendingBindings();

    void checkDataDependency(const std::vector<Resource *> &requiredResources,
        Operation accessStage, AccessTypeFlags accessType);
    
    void checkDataDependency(const std::vector<Resource *> &requiredResources,
        Operation accessStage, const std::vector<AccessTypeFlags> &accessTypes);

    VkBufferMemoryBarrier makeBufferMemoryBarrier(const Buffer &buffer,
        VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask);
    
    void addResourceMemoryBarriers(const std::vector<VkBufferMemoryBarrier> &barriers,
        VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask);
    
    std::pair<VkPipelineStageFlags, VkAccessFlags> mapStageAndAccessMask(Operation accessStage, AccessTypeFlags accessType);
    VkPipelineStageFlags mapStage(Operation accessStage);
};

#endif // JOB_H
