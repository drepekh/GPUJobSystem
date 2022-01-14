#include "JobManager.h"

constexpr uint32_t arraySize = 20;

void printArray(std::string name, uint32_t *data)
{
    std::cout << "\n" << name << ": ";
    for (size_t i = 0; i < arraySize; ++i)
        std::cout << data[i] << ' ';
}

int main()
{
    uint32_t data1[arraySize];
    uint32_t data2[arraySize];
    size_t dataSize = sizeof(uint32_t) * arraySize;

    std::cout << "Before:";
    for (uint32_t i = 0; i < arraySize; ++i)
    {
        data1[i] = i;
        data2[i] = i * 10;
    }
    printArray("Array 1", data1);
    printArray("Array 2", data2);

    JobManager manager;
    Task task = manager.createTask("shaders/sum.spv",
        {{ ResourceType::StorageBuffer, ResourceType::StorageBuffer }});
    Buffer buffer1 = manager.createBuffer(dataSize);
    Buffer buffer2 = manager.createBuffer(dataSize);
    ResourceSet resourceSet = manager.createResourceSet({ &buffer1, &buffer2 });
    ResourceSet resourceSet2 = manager.createResourceSet({ &buffer2, &buffer1 });
    Job job = manager.createJob();

    // copy data to device
    job.syncResourceToDevice(buffer1, data1, dataSize);
    job.syncResourceToDevice(buffer2, data2, dataSize);
    // execute task
    job.addTask(task, { resourceSet }, arraySize);
    // wait until task finishes all write operations
    job.waitForTasksFinish();
    // execute task again with differently binded buffers
    job.useResources(0, resourceSet2);
    job.addTask(task, arraySize);
    // copy data back to host
    job.syncResourceToHost(buffer1, data1, dataSize);
    job.syncResourceToHost(buffer2, data2, dataSize);
    // submit and wait until done
    job.submit();
    job.await();

    std::cout << "\nAfter:";
    printArray("Array 1", data1);
    printArray("Array 2", data2);
}
