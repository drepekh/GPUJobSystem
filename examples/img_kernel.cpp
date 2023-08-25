#include "JobManager.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main()
{
    // load texture
    std::string path = "../examples/resources/vulkan_11_rgba.png";
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (pixels == nullptr)
    {
        std::cout << "Could not load image.";
        return 0;
    }

    int imageSize = texWidth * texHeight * texChannels;
    std::shared_ptr<unsigned char[]> imgOut(new unsigned char[imageSize]);
    int localGroupSize = 16;

    // create manager and resources
    JobManager manager;
    Task task = manager.createTask("../examples/shaders/edgedetect.spv", localGroupSize, localGroupSize);
    Image imageIn = manager.createImage(texWidth, texHeight);
    Image imageOut = manager.createImage(texWidth, texHeight);
    Job job = manager.createJob();

    // record commands
    job.syncResourceToDevice(imageIn, pixels, imageSize)
        .syncResourceToDevice(imageOut, 0, 0)
        .addTask(task, { {&imageIn, &imageOut} }, texWidth / localGroupSize, texHeight / localGroupSize)
        .syncResourceToHost(imageOut, imgOut.get(), imageSize)
        .submit()
        .await();

    stbi_image_free(pixels);
    stbi_write_png("../examples/resources/out.png", texWidth, texHeight, texChannels, imgOut.get(), texWidth * texChannels);
}