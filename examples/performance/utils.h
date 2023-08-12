#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>

namespace utils
{
    inline std::map<size_t, std::vector<float>> measures;
    
    class Image;
}

template <typename T>
void printVector(const std::vector<T> &vec)
{
    std::cout << "{ ";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec.at(i) << (i == (vec.size() - 1) ? "" : ", ");
    }
    std::cout << " }";
}

void addMeasure(size_t key, float time);

void printMeasures(size_t trim = 2, bool onlyResults = false);

void clearMeasures();

namespace utils
{
    class Image
    {
    public:
        Image(const std::string& path);
        Image(int width, int height, int channels);
        ~Image();

        void save(const std::string& path);
        void fill();

        bool isValid() { return data != nullptr; }

        unsigned char* data = nullptr;
        int width = 0;
        int height = 0;
        int channels = 0;
        int size = 0;

    private:
        void load(const std::string& path);
        void free();
    };
}

#endif // UTILS_H