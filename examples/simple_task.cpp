#include "JobManager.h"

constexpr size_t arraySize = 20;

int main()
{
    uint32_t data[arraySize];
    size_t dataSize = sizeof(uint32_t) * arraySize;

    std::cout << "Before: ";
    for (size_t i = 0; i < arraySize; ++i)
    {
        data[i] = i;
        std::cout << data[i] << ' ';
    }
    std::cout << std::endl;

    JobManager manager;
    Task task = manager.createTask("shaders/fibonacci.spv", {{ ResourceType::StorageBuffer }});
    Buffer buffer = manager.createBuffer(dataSize);
    Job job = manager.createJob();

    // record commands
    job.syncResourceToDevice(buffer, data, dataSize);
    job.addTask(task, {{0, { &buffer }}}, arraySize);
    job.syncResourceToHost(buffer, data, dataSize);
    // submit and wait until done
    job.submit();
    job.await();

    std::cout << "After:  ";
    for (size_t i = 0; i < arraySize; ++i)
        std::cout << data[i] << ' ';
}