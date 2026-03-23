#pragma once

// verlet_skin.h
// Verlet skin algorithm for neighbor list rebuild triggering.
// Include this after particleslist.h in main.cu.
// Requires: box_size (defined in particleslist.h)

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

// ---------------------------------------------------------------------------
// Snapshot kernel: copies current positions into a reference snapshot
// ---------------------------------------------------------------------------
__global__
void SnapshotPositions(const float4 *positions, float4 *snapshot, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    snapshot[idx] = positions[idx];
}

// ---------------------------------------------------------------------------
// Functor: computes per-particle displacement squared since last snapshot,
//          using minimum-image convention consistent with particleslist.h
// ---------------------------------------------------------------------------
struct DisplacementFunctor
{
    const float4 *current;
    const float4 *snapshot;

    DisplacementFunctor(const float4 *c, const float4 *s)
        : current(c), snapshot(s) {}

    __host__ __device__
    float operator()(int i) const
    {
        float dx = current[i].x - snapshot[i].x;
        float dy = current[i].y - snapshot[i].y;
        float dz = current[i].z - snapshot[i].z;

        dx -= box_size * roundf(dx / box_size);
        dy -= box_size * roundf(dy / box_size);
        dz -= box_size * roundf(dz / box_size);

        return dx*dx + dy*dy + dz*dz;
    }
};

// ---------------------------------------------------------------------------
// Host function: returns the maximum displacement squared across all particles
// ---------------------------------------------------------------------------
inline float ComputeMaxDisplacementSqThrust(const float4 *current,
                                            const float4 *snapshot,
                                            int N)
{
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(N);

    return thrust::transform_reduce(
        thrust::device,
        begin, end,
        DisplacementFunctor(current, snapshot),
        0.0f,
        thrust::maximum<float>()
    );
}