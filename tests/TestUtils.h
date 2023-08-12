#ifndef HELPERS_H
#define HELPERS_H

#include "JobManager.h"

#include <map>
#include <string>

#define D(name) default: return #name
#define C(name) case ResourceType::name: return #name
static std::string resourceTypeName(ResourceType type)
{
    switch(type)
    {
        C(StorageBuffer);
        C(StorageImage);
        D(none);
    }
}
#undef C


#define C(name) case Buffer::Type::name: return #name
static std::string bufferTypeName(Buffer::Type type)
{
    switch(type)
    {
        C(DeviceLocal);
        C(Staging);
        C(Uniform);
        D(none);
    }
}
#undef C
#undef D

#endif // HELPERS_H