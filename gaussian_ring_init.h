#pragma once
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <math.h>
#include <stdint.h>
#include <string>
#include <sstream>


// Initialize particles in a Ring
__global__
void InitParticlesRing(float4 *positions, float4 *forces, float4 *elasticforces, unsigned int *chain_indices, int N, float L, float r0){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float4 &pos = positions[idx];
    float4 &force = forces[idx];
    float4 &elasticforce = elasticforces[idx];
    float x0 = box_size/2;
    float y0 = box_size/2;
    float z0 = box_size/2;
    float dTheta = 2*ppi/N;
    float R = r0 / (16.0f * sinf(dTheta / 2.0f));
    pos.x = x0+R*cos(2*ppi/N*(idx));
    pos.y = y0+R*sin(2*ppi/N*(idx));
    pos.z = z0;
    pos.w = 0.0f;
    force.x = 0.0f;
    force.y = 0.0f;
    force.z = 0.0f;
    force.w = 0.0f;
    elasticforce.x = 0.0f;
    elasticforce.y = 0.0f;
    elasticforce.z = 0.0f;
    elasticforce.w = 0.0f;
    chain_indices[idx] = idx;
}

// Initialize particle as random walk
// Simple hash-based RNG (deterministic per seed)
__device__ inline float rand_uniform(unsigned int seed, unsigned int ext_seed) {
    seed ^= 2747636419u;
    seed *= 2654435769u;
    seed ^= seed >> 16;
    seed *= 2654435769u;
    seed ^= seed >> 16;
    seed *= ext_seed;
    return (seed & 0x00FFFFFF) / float(0x01000000);
}

// Initialize particles in a random-walk ring
__global__
void InitParticlesRW(float4 *positions,
                       float4 *forces,
                       float4 *elasticforces,
                       unsigned int *chain_indices, const unsigned int ext_seed,
                       int N,
                       float r0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 &pos = positions[idx];
    float4 &force = forces[idx];
    float4 &elasticforce = elasticforces[idx];

    const float PI = 3.14159265358979323846f;

    // ---- Step 1: compute total end-to-end vector R of open RW ----
    float Rx = 0.0f;
    float Ry = 0.0f;
    float Rz = 0.0f;

    for (int i = 0; i < N; i++) {
        unsigned int seed = i * 9781u + 6271u;

        float u = rand_uniform(seed, ext_seed);
        float v = rand_uniform(seed + 17u, ext_seed);

        float theta = 2.0f * PI * u;
        float z = 2.0f * v - 1.0f;
        float s = sqrtf(1.0f - z*z);

        float dx = r0 * s * cosf(theta);
        float dy = r0 * s * sinf(theta);
        float dz = r0 * z;

        Rx += dx;
        Ry += dy;
        Rz += dz;
    }

    // ---- Step 2: compute position of bead idx in open walk ----
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    for (int i = 0; i < idx; i++) {
        unsigned int seed = i * 9781u + 6271u;

        float u = rand_uniform(seed, ext_seed);
        float v = rand_uniform(seed + 17u, ext_seed);

        float theta = 2.0f * PI * u;
        float zz = 2.0f * v - 1.0f;
        float s = sqrtf(1.0f - zz*zz);

        x += r0 * s * cosf(theta);
        y += r0 * s * sinf(theta);
        z += r0 * zz;
    }

    // ---- Step 3: apply ring closure correction ----
    float factor = float(idx) / float(N);
    x -= factor * Rx;
    y -= factor * Ry;
    z -= factor * Rz;

    // ---- Step 4: shift to box center ----
    x += box_size * 0.5f;
    y += box_size * 0.5f;
    z += box_size * 0.5f;

    // ---- Write positions ----
    pos.x = x;
    pos.y = y;
    pos.z = z;
    pos.w = 0.0f;   // initial epigenetic state

    // ---- Zero forces ----
    force.x = 0.0f;
    force.y = 0.0f;
    force.z = 0.0f;
    force.w = 0.0f;

    elasticforce.x = 0.0f;
    elasticforce.y = 0.0f;
    elasticforce.z = 0.0f;
    elasticforce.w = 0.0f;

    chain_indices[idx] = idx;
}

// Initialize epigenetic field to fixed configuration
__global__
void InitEpiAvg(float4 *positions, const int N, const float avg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float4& pos = positions[idx];
    pos.w = avg;
}



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Function for general initial condition setting
__host__
void initialize_system(const std::string init_conf,
                       float4 *positions, float4 *forces, float4 *elasticforces, unsigned int *chain_indices,
                       const int N, const float L, const float r0, const unsigned int ext_seed,
                       const int num_blocks, const int threads_per_block)
{
    
    if(init_conf == "Ring"){
        printf("Initializing chain as ring polymer. \n");
        InitParticlesRing<<<num_blocks, threads_per_block>>>(
            positions,
            forces,
            elasticforces,
            chain_indices,
            N, L, r0
        );
    }
    else if(init_conf=="RW"){
        printf("Initializing chain as random walk. \n");
        InitParticlesRW<<<num_blocks, threads_per_block>>>(
            positions,
            forces,
            elasticforces,
            chain_indices, ext_seed,
            N, r0
        );
    }
}