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
    std::vector<uint32_t> data1(arraySize);
    std::vector<uint32_t> data2(arraySize);
    size_t dataSize = sizeof(uint32_t) * arraySize;

    std::cout << "Before:";
    for (uint32_t i = 0; i < arraySize; ++i)
    {
        data1[i] = i;
        data2[i] = i * 10;
    }
    printArray("Array 1", data1.data());
    printArray("Array 2", data2.data());

    JobManager manager;
    Task task = manager.createTask("../examples/shaders/sum.spv");
    Buffer buffer1 = manager.createBuffer(dataSize);
    Buffer buffer2 = manager.createBuffer(dataSize);
    ResourceSet resourceSet = manager.createResourceSet({ &buffer1, &buffer2 });
    ResourceSet resourceSet2 = manager.createResourceSet({ &buffer2, &buffer1 });
    // Job job = manager.createJob();
    manager.createJob()
        // copy data to device
        .syncResourceToDevice(buffer1, data1.data())
        .syncResourceToDevice(buffer2, data2.data())
        // execute task
        .addTask(task, { resourceSet }, arraySize)
        // wait until task finishes all write operations
        .waitForTasksFinish()
        // execute task again with differently binded buffers
        .useResources(0, resourceSet2)
        .addTask(task, arraySize)
        // copy data back to host
        .syncResourceToHost(buffer1, data1.data())
        .syncResourceToHost(buffer2, data2.data())
        // submit and wait until done
        .submit()
        .await();

    std::cout << "\nAfter:";
    printArray("Array 1", data1.data());
    printArray("Array 2", data2.data());
}
