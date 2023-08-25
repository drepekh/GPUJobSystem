#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "JobManager.h"

#include "TestUtils.h"


TEST_CASE("JobManager resource creation", "[JobManager]")
{
    JobManager manager;
    
    REQUIRE(manager.getDevice() != VK_NULL_HANDLE);

    SECTION("Buffer created")
    {
        auto type = GENERATE(Buffer::Type::DeviceLocal, Buffer::Type::Staging, Buffer::Type::Uniform);
        SECTION(bufferTypeName(type))
        {
            size_t size = 10;
            Buffer buffer = manager.createBuffer(size, type);

            REQUIRE(buffer.getResourceType() == ResourceType::StorageBuffer);
            REQUIRE(buffer.getBufferType() == type);
            REQUIRE(buffer.getBuffer() != VK_NULL_HANDLE);
            REQUIRE(buffer.getMemory() != VK_NULL_HANDLE);
            REQUIRE(buffer.getSize() == size);

            if (type == Buffer::Type::DeviceLocal)
            {
                REQUIRE(buffer.getStagingBuffer() != nullptr);
            }
            else
            {
                REQUIRE(buffer.getStagingBuffer() == nullptr);
            }
        }
    }

    SECTION("Image created")
    {
        size_t width = 10, height = 10;
        Image image = manager.createImage(width, height);

        REQUIRE(image.getResourceType() == ResourceType::StorageImage);
        REQUIRE(image.getImage() != VK_NULL_HANDLE);
        REQUIRE(image.getMemory() != VK_NULL_HANDLE);
        REQUIRE(image.getView() != VK_NULL_HANDLE);
        REQUIRE(image.getWidth() == width);
        REQUIRE(image.getHeight() == height);
        REQUIRE(image.getStagingBuffer() != nullptr);
    }

    SECTION("Job created")
    {
        Job job = manager.createJob();

        REQUIRE(job.isComplete() == true);
        REQUIRE(job.getCommandBuffer() != VK_NULL_HANDLE);
    }

    SECTION("Task created")
    {
        SECTION("without specialized constants")
        {
            Task task = manager.createTask("../examples/shaders/fibonacci.spv");
        }

        SECTION("With specialized constants")
        {
            Task task = manager.createTask("../examples/shaders/fibonacci.spv", 20);
        }
    }

    SECTION("ResourceSet created")
    {
        size_t size = 10;
        Buffer buffer = manager.createBuffer(size);
        size_t width = 10, height = 10;
        Image image = manager.createImage(width, height);

        ResourceSet set = manager.createResourceSet({ &buffer, &image });
        REQUIRE(set.getDescriptorSet() != VK_NULL_HANDLE);
    }
}

