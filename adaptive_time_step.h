
#pragma once

//Compute maximum force
struct ForceMagnitudeFunctor
{
    const float4 *forces;
    const float4 *elasticforces;

    ForceMagnitudeFunctor(const float4 *f,
                          const float4 *ef)
        : forces(f), elasticforces(ef) {}

    __host__ __device__
    float operator()(const int &i) const
    {
        float fx = forces[i].x + elasticforces[i].x;
        float fy = forces[i].y + elasticforces[i].y;
        float fz = forces[i].z + elasticforces[i].z;
        float fw = forces[i].w + elasticforces[i].w;

        return (fx*fx + fy*fy + fz*fz+ fw*fw);
    }
};

float ComputeMaxForceSqThrust(const float4 *d_forces,
                            const float4 *d_elasticforces,
                            int N)
{
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(N);

    float max_force_sq = thrust::transform_reduce(
        thrust::device,
        begin,
        end,
        ForceMagnitudeFunctor(d_forces,
                              d_elasticforces),
        0.0f,
        thrust::maximum<float>());

    return max_force_sq;
}

float calculate_dt_adaptive(const float4 *d_forces,
                            const float4 *d_elasticforces,
                            const int N, const float dx_thresh, const float dt)
{
    // Calculate maximum force
    float max_force = sqrtf(ComputeMaxForceSqThrust(
                d_forces,
                d_elasticforces,
                N));
    
    
    // Set time step
    float dt_adaptive = dx_thresh / max_force;

    if(dt_adaptive < 1e-6){
        // printf("Warning: very small time step encountered. \n");
    }
    
    if(dt_adaptive >= dt){
        dt_adaptive = dt;
    }

    return dt_adaptive;
}