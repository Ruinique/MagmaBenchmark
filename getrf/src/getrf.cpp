#include <curand.h>
#include <magma_lapack.h>
#include <magma_v2.h>

#include <cstddef>
#include <cstdio>

#define MAGIC_SEED 261825L

template <typename T>
void generateNormalMatrix(T* dA, size_t m, size_t n, size_t seed_offset = 0) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    unsigned long long seed = MAGIC_SEED;
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    if constexpr (std::is_same_v<T, float>)
        curandGenerateNormal(gen, dA, m * n, 0.0f, 1.0f);
    else if constexpr (std::is_same_v<T, double>)
        curandGenerateNormalDouble(gen, dA, m * n, 0.0, 1.0);
    curandDestroyGenerator(gen);
}

template <typename T>
double test_magma_getrf(magma_int_t m, magma_int_t n, magma_queue_t queue) {
    T* test;
    magma_malloc(reinterpret_cast<void**>(&test), m * n * sizeof(T));
    T* test_cpy_h;
    magma_malloc_cpu(reinterpret_cast<void**>(&test_cpy_h), m * n * sizeof(T));
    magma_getmatrix(m, n, sizeof(T), test, m, test_cpy_h, m, queue);
    if (test == nullptr) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1.0;
    }
    generateNormalMatrix(test, m, n);

    magma_int_t* ipiv;
    magma_imalloc_cpu(&ipiv, n);
    magma_int_t info;

    if constexpr (std::is_same_v<T, float>)
        magma_sgetrf_native(m, n, test, m, ipiv, &info);
    else if constexpr (std::is_same_v<T, double>)
        magma_dgetrf_native(m, n, test, m, ipiv, &info);

    double sum = 0;
    const int roll = 3;

    for (int i = 0; i < roll; i++) {
        magma_setmatrix(m, n, sizeof(T), test_cpy_h, m, test, m, queue);

        magma_queue_sync(queue);
        auto start = magma_sync_wtime(queue);

        if constexpr (std::is_same_v<T, float>)
            magma_sgetrf_native(m, n, test, m, ipiv, &info);
        else if constexpr (std::is_same_v<T, double>)
            magma_dgetrf_native(m, n, test, m, ipiv, &info);

        magma_queue_sync(queue);
        auto end = magma_sync_wtime(queue);
        double time = end - start;
        // printf("Time taken for magma_getrf_native: %f seconds\n", time);
        sum += time;
    }

    return sum / 3;
}

int main() {
    magma_init();
    magma_print_environment();

    magma_queue_t queue;
    magma_device_t device;
    magma_getdevice(&device);
    printf("Using device %d\n", device);
    magma_queue_create(device, &queue);

    double float_time = test_magma_getrf<float>(32768, 32768, queue);
    // double double_time = test_magma_getrf<double>(32768, 32768, queue);

    printf("Average time for magma_getrf_native with float: %f seconds\n", float_time);
    // printf("Average time for magma_getrf_native with double: %f seconds\n", double_time);

    return 0;
}