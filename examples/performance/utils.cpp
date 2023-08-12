#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <random>


using namespace utils;

void addMeasure(size_t key, float time)
{
    if (auto it = utils::measures.find(key); it != utils::measures.end())
    {
        it->second.push_back(time);
    }
    else
    {
        utils::measures.insert({ key, { time } });
    }
}

void printMeasures(size_t trim, bool onlyResults)
{
    for (auto& [key, times] : utils::measures)
    {
        std::cout << "Case \"" << key << "\": ";
        if (!onlyResults)
        {
            printVector(times);
        }

        auto total = std::accumulate(times.begin(), times.end(), (decltype (times)::value_type)(0));
        std::cout << " | Average: " << float(total) / times.size();

        if (times.size() > trim * 2)
        {
            std::sort(times.begin(), times.end());
            total = std::accumulate(times.begin() + trim, times.end() - trim, (decltype (times)::value_type)(0));
            std::cout << " | Trimmed: " << float(total) / (times.size() - trim * 2);
        }

        std::cout << '\n';
    }
}

void clearMeasures()
{
    measures.clear();
}

Image::Image(const std::string& path)
{
    load(path);
}

Image::Image(int width, int height, int channels) :
    width(width),
    height(height),
    channels(channels),
    size(width * height * channels)
{
    data = new unsigned char[size];
}

Image::~Image()
{
    free();
}

void Image::save(const std::string& path)
{
    stbi_write_png(path.c_str(), width, height, channels, data, width * channels);
}

void Image::fill()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < size; ++i)
    {
        data[i] = dis(gen);
    }
}

void Image::load(const std::string& path)
{
    free();
    data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    size = width * height * channels;
}

void Image::free()
{
    if (data != nullptr)
    {
        stbi_image_free(data);
    }
}