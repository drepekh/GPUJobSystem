#include "catch.hpp"

#include "JobManager.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

TEST_CASE("Job transfer tests", "[Job]")
{
    JobManager manager;
    Job job = manager.createJob();

    SECTION("Buffer transfer")
    {
        constexpr size_t count = 5;
        constexpr size_t dataSize = count * sizeof(uint32_t);
        uint32_t data[count] = {1, 2, 3, 4, 5};
        uint32_t result[count];

        SECTION("To/from device")
        {
            auto bufferType = GENERATE(Buffer::Type::DeviceLocal, Buffer::Type::Staging, Buffer::Type::Uniform);
            Buffer buffer = manager.createBuffer(dataSize, bufferType);
            job.syncResourceToDevice(buffer, data, dataSize);
            job.syncResourceToHost(buffer, result, dataSize);
        }

        SECTION("Between buffers")
        {
            auto bufferType1 = GENERATE(Buffer::Type::DeviceLocal, Buffer::Type::Staging);
            auto bufferType2 = GENERATE(Buffer::Type::DeviceLocal, Buffer::Type::Staging);
            Buffer buffer1 = manager.createBuffer(dataSize, bufferType1);
            Buffer buffer2 = manager.createBuffer(dataSize, bufferType2);
            job.syncResourceToDevice(buffer1, data, dataSize);
            job.syncResources(buffer1, buffer2);
            job.syncResourceToHost(buffer2, result, dataSize);
        }

        job.submit();
        REQUIRE(job.await());
        
        REQUIRE(std::equal(data, data + count, result));
    }

    SECTION("Image transfer")
    {
        std::string path = "../examples/resources/test.png";
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        REQUIRE(pixels != nullptr);
        REQUIRE(texChannels == 4);

        stbi_uc* result = new stbi_uc[texWidth * texHeight * 4];
        Image image = manager.createImage(texWidth, texHeight);

        job.syncResourceToDevice(image, pixels, image.getSize());
        
        SECTION("To/from device")
        {
            job.syncResourceToHost(image, result, image.getSize());
        }

        SECTION("Between images")
        {
            Image image2 = manager.createImage(texWidth, texHeight);
            job.syncResources(image, image2);
            job.syncResourceToHost(image2, result, image2.getSize());
        }

        job.submit();
        REQUIRE(job.await());

        REQUIRE(std::equal(pixels, pixels + image.getSize(), result));
    }
}

TEST_CASE("Job execute tests", "[Job]")
{
    JobManager manager;
    Job job = manager.createJob();

    SECTION("Empty job")
    {
        job.submit();
        REQUIRE(job.await());
        REQUIRE(job.isComplete());
    }

    SECTION("Simple job")
    {
        constexpr size_t count = 5;
        constexpr size_t dataSize = count * sizeof(uint32_t);
        Buffer buffer = manager.createBuffer(dataSize);
        Task task = manager.createTask("../examples/shaders/fibonacci.spv",
            {{ ResourceType::StorageBuffer }}, count);
        
        uint32_t data[count] = {1, 2, 3, 4, 5};
        uint32_t expected[count] = {1, 1, 2, 3, 5};
        
        job.syncResourceToDevice(buffer, data, dataSize);
        job.addTask(task, {{ &buffer }}, count);
        job.syncResourceToHost(buffer, data, dataSize);

        job.submit();
        REQUIRE(job.await());
        
        REQUIRE(std::equal(data, data + count, expected));
    }
}
