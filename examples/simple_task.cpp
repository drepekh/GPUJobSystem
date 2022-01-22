#include "JobManager.h"

constexpr uint32_t arraySize = 20;

int main()
{
    uint32_t data[arraySize];
    size_t dataSize = sizeof(uint32_t) * arraySize;

    std::cout << "Before: ";
    for (uint32_t i = 0; i < arraySize; ++i)
    {
        data[i] = i;
        std::cout << data[i] << ' ';
    }
    std::cout << std::endl;

    JobManager manager;
    Task task = manager.createTask("../examples/shaders/fibonacci.spv", {{ ResourceType::StorageBuffer }}, 0, arraySize);
    Buffer buffer = manager.createBuffer(dataSize);
    Job job = manager.createJob();

    // record commands
    job.syncResourceToDevice(buffer, data, dataSize);
    job.addTask(task, {{ &buffer }}, arraySize);
    job.syncResourceToHost(buffer, data, dataSize);
    // submit and wait until done
    job.submit();
    job.await();

    std::cout << "After:  ";
    for (size_t i = 0; i < arraySize; ++i)
        std::cout << data[i] << ' ';
}
