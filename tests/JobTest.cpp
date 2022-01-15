#include "catch.hpp"

#include "JobManager.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "helpers.h"


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
            SECTION(bufferTypeName(bufferType))
            {
                Buffer buffer = manager.createBuffer(dataSize, bufferType);
                job.syncResourceToDevice(buffer, data, dataSize);
                job.waitAfterTransfers();
                job.syncResourceToHost(buffer, result, dataSize);
            }
        }

        SECTION("Between buffers")
        {
            auto bufferType1 = GENERATE(Buffer::Type::DeviceLocal, Buffer::Type::Staging);
            auto bufferType2 = GENERATE(Buffer::Type::DeviceLocal, Buffer::Type::Staging);
            SECTION(bufferTypeName(bufferType1) + " - " + bufferTypeName(bufferType2))
            {
                Buffer buffer1 = manager.createBuffer(dataSize, bufferType1);
                Buffer buffer2 = manager.createBuffer(dataSize, bufferType2);
                job.syncResourceToDevice(buffer1, data, dataSize);
                job.waitAfterTransfers();
                job.syncResources(buffer1, buffer2);
                job.waitAfterTransfers();
                job.syncResourceToHost(buffer2, result, dataSize);
            }
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

        stbi_image_free(pixels);
        delete[] result;
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
            {{ ResourceType::StorageBuffer }}, 0, (uint32_t)count);
        
        uint32_t data[count] = {1, 2, 3, 4, 5};
        uint32_t expected[count] = {1, 1, 2, 3, 5};
        
        job.syncResourceToDevice(buffer, data, dataSize);
        
        SECTION("with ResourceSet")
        {
            ResourceSet set = manager.createResourceSet({ &buffer });
            SECTION("using useResources")
            {
                job.useResources(0, set);
                job.addTask(task, count);
            }

            SECTION("using direct resource binding")
            {
                job.addTask(task, { set }, count);
            }
        }

        SECTION("without ResourceSet")
        {
            SECTION("using useResources")
            {
                job.useResources(0, { &buffer });
                job.addTask(task, count);
            }

            SECTION("using direct resource binding")
            {
                job.addTask(task, {{ &buffer }}, count);
            }
        }

        job.syncResourceToHost(buffer, data, dataSize);
        job.submit();
        REQUIRE(job.await());
        
        REQUIRE(std::equal(data, data + count, expected));
    }

    SECTION("Multiple submit")
    {
        constexpr size_t count = 5;
        constexpr size_t dataSize = count * sizeof(uint32_t);
        Buffer buffer = manager.createBuffer(dataSize);
        Task task = manager.createTask("../examples/shaders/fibonacci.spv",
            {{ ResourceType::StorageBuffer }}, 0, (uint32_t)count);
        
        uint32_t inData[count] = {1, 2, 3, 4, 5};
        uint32_t outData[count];
        uint32_t expectedData[count] = {1, 1, 2, 3, 5};

        job.syncResourceToDevice(buffer, inData, dataSize);
        job.addTask(task, {{ &buffer }}, count);
        job.syncResourceToHost(buffer, outData, dataSize);

        for (size_t i = 0; i < 5; ++i)
        {
            std::fill(outData, outData + count, 0);
            job.submit();
            REQUIRE(job.await());
            REQUIRE(job.isComplete());
            REQUIRE(std::equal(outData, outData + count, expectedData));
        }
    }

    SECTION("Multiple task invokations")
    {
        constexpr size_t count = 5;
        constexpr size_t dataSize = count * sizeof(uint32_t);
        Task task = manager.createTask("../examples/shaders/sum.spv",
            {{ ResourceType::StorageBuffer, ResourceType::StorageBuffer }});

        Buffer buffer1 = manager.createBuffer(dataSize);
        Buffer buffer2 = manager.createBuffer(dataSize);
        ResourceSet resourceSet = manager.createResourceSet({ &buffer1, &buffer2 });
        ResourceSet resourceSet2 = manager.createResourceSet({ &buffer2, &buffer1 });

        uint32_t data1[count] = {1, 2, 3, 4, 5};
        uint32_t data2[count] = {10, 20, 30, 40, 50};
        uint32_t expected1[count] = {12, 24, 36, 48, 60};
        uint32_t expected2[count] = {11, 22, 33, 44, 55};

        job.syncResourceToDevice(buffer1, data1, dataSize);
        job.syncResourceToDevice(buffer2, data2, dataSize);
        job.addTask(task, { resourceSet }, count);
        job.waitForTasksFinish();
        job.addTask(task, { resourceSet2 }, count);
        job.syncResourceToHost(buffer1, data1, dataSize);
        job.syncResourceToHost(buffer2, data2, dataSize);
        job.submit();
        REQUIRE(job.await());

        REQUIRE(std::equal(data1, data1 + count, expected1));
        REQUIRE(std::equal(data2, data2 + count, expected2));
    }
}
