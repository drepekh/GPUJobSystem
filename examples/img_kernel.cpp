#include "JobManager.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main()
{
    // load texture
    std::string path = "resources/vulkan_11_rgba.png";
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    unsigned char *imgOut = new unsigned char[texWidth * texHeight * 4];

    // create manager and resources
    JobManager manager;
    Task task = manager.createTask("shaders/edgedetect.spv",
        {{ ResourceType::StorageImage, ResourceType::StorageImage }});
    Image imageIn = manager.createImage(texWidth, texHeight);
    Image imageOut = manager.createImage(texWidth, texHeight);
    Job job = manager.createJob();

    job.syncResourceToDevice(imageIn, pixels, texWidth * texHeight * 4);
    job.syncResourceToDevice(imageOut, 0, 0);
    job.addTask(task, { {&imageIn, &imageOut} }, texWidth / 16, texHeight / 16);
    job.syncResourceToHost(imageOut, imgOut, texWidth * texHeight * 4);

    job.submit();
    job.await();

    stbi_image_free(pixels);
    stbi_write_png("resources/out.png", texWidth, texHeight, 4, imgOut, texWidth * 4);
}