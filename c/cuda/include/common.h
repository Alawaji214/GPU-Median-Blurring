#ifndef COMMMON_H
#define COMMMON_H

#include "cuda.h"

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line);

#endif /* COMMMON_H */
