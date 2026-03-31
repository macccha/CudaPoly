#pragma once


//Function to calculate repulsive force module: normal power-law scaling
__device__
inline float force_module(const float A, const float distance, const float lambda, const float exponent, const float shift){
    
    float inverse = 1/(distance);
    inverse = inverse*inverse*inverse;//inverse-b;
    return 1.0*inverse;
        
}

//Function to calculate repulsive force module: WCA potential
__device__
inline float wca_force_mod(float dist_sq, float sigma, float epsilon) {
    
        // Only active below 2^(1/6) * sigma ≈ 1.1224 * sigma
        float sigma_sq = sigma * sigma;
        if (dist_sq >= 1.25992105f * sigma_sq) return 0.0f;  // 2^(1/3) * sigma^2
        float sr2  = sigma_sq / dist_sq;
        float sr6  = sr2 * sr2 * sr2;
        float sr12 = sr6 * sr6 * sr2;
        // Magnitude of force: 48*eps*(sr12 - 0.5*sr6) / dist_sq
        // (already divided by r, ready to multiply by displacement components)
        return 48.0f * epsilon * (sr12 - 0.5f * sr6);

}

//Function to calculate soft repulsive force module
__device__
inline float soft_rep_mod(float dist, float sigma, float epsilon) {
    
        // Only active below 2^(1/6) * sigma ≈ 1.1224 * sigma
        float d_c = 1.12246204831 * sigma;
        if (dist >= d_c) return 0.0f;  // 2^(1/3) * sigma^2
        return epsilon*cosf(ppi2*dist/d_c);

}

//Search kernel function
__global__
void neighbor_search_kernel(const int mode,
                            const float4 *positions, float4 *forces, unsigned int *chain_indices, float *forces_x, int *cell_start, int *cell_end, int *particle_hashes,int *num_neighbors,
                            const int N, const int grid_size, const float cutoff_squared,
                            const float ds, float rep, const float A, const float lambda, const float shiftrep, const int expn, const float gamma)
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;

    // // Get particle index from list of indices
    int p_idx = chain_indices[idx];
    // Get particle position, force and chain index
    const float4& pos = positions[idx];
    float4& force = forces[idx];

    //Reset forces
    force.x = 0.0f;
    force.y = 0.0f;
    force.z = 0.0f;
    force.w = 0.0f;
    
    // Get integer coordinates of particle
    int c_x = static_cast<int>(pos.x / cell_size);
    int c_y = static_cast<int>(pos.y / cell_size);
    int c_z = static_cast<int>(pos.z / cell_size);

    // Check nearby cells in a 3x3x3 neighborhood
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                
                int n_x = pmod(c_x+dx,grid_size);
                int n_y = pmod(c_y+dy,grid_size);
                int n_z = pmod(c_z+dz,grid_size);

                int neighbor_hash = n_x + grid_size * (n_y + grid_size * n_z);

                // Check the range of particles in this cell
                int start = cell_start[neighbor_hash];
                int end = cell_end[neighbor_hash];

                // Exclude cases of empty or single-cell cells
                if (start == -1 || end == -1) continue;

                for (int j = start; j < end; j++) {
                    
                    if(j != idx){
                        
                        const float4& neighbor = positions[j];
                        float distance_sq = (distance_squared_float4(pos, neighbor));
                        int idx_neigh = chain_indices[j];

                        // Check whether neighbors are nearest neighbors on chain
                        bool are_bonded = (idx_neigh == (p_idx + 1) % N) ||
                                            (idx_neigh == (p_idx - 1 + N) % N);


                        // Force calculation on beads happens only for non-nearest neighbor beads
                        if (!are_bonded && distance_sq <= cutoff_squared && distance_sq > 0.00005f) {

                            float distance = sqrtf(distance_sq);
                            // float rep_mod = ds*A*force_module(A,distance,-6.0,-expn-1,shiftrep);
                            float rep_mod = mode == 1 ? ds*soft_rep_mod(distance, A, 30) : ds*wca_force_mod(distance_sq, A, expn);
                            if(isnan(rep_mod)){
                                printf("Force is NaN, because distance is %f. \n", distance);
                            }
                            float att_mod = rep * ds * gamma * lambda * expf(lambda * distance) * neighbor.w * pos.w;

                            force.x += (rep_mod+att_mod)*((pos.x-neighbor.x)/distance);
                            force.y += (rep_mod+att_mod)*((pos.y-neighbor.y)/distance);
                            force.z += (rep_mod+att_mod)*((pos.z-neighbor.z)/distance);
                            force.w += ds * gamma * expf(lambda * distance) * neighbor.w;
                        
                        }
                        // Charge coupling applies to ALL neighbors including bonded ones
                        // force.w += ds * gamma * expf(lambda * distance) * neighbor.w;
                    }
                    
                };
            }
        }
    }

    // Save force
    // forces_x[idx] = force.w;
};

// Wrapper of neighbor_search_kernel
inline void ComputeNonBondedForces(
    thrust::device_vector<float4>         &positions,
    thrust::device_vector<float4>         &forces,
    thrust::device_vector<unsigned int>   &chain_indices,
    thrust::device_vector<int>            &particle_hashes,
    thrust::device_vector<int>            &cell_start,
    thrust::device_vector<int>            &cell_end,
    thrust::device_vector<float>          &forces_x,
    thrust::device_vector<int>            &num_neighbors,
    const int N, const int num_blocks, const int threads_per_block,
    const float ds, const float rep,
    const float A, const float lambda,
    const float shiftrep, const int expn, const float gammam,
    const int mode)
{
    neighbor_search_kernel<<<num_blocks, threads_per_block>>>(mode,
        thrust::raw_pointer_cast(positions.data()),
        thrust::raw_pointer_cast(forces.data()),
        thrust::raw_pointer_cast(chain_indices.data()),
        thrust::raw_pointer_cast(forces_x.data()),
        thrust::raw_pointer_cast(cell_start.data()),
        thrust::raw_pointer_cast(cell_end.data()),
        thrust::raw_pointer_cast(particle_hashes.data()),
        thrust::raw_pointer_cast(num_neighbors.data()),
        N, grid_size, cutoff_squared, ds, rep, A, lambda, shiftrep, expn, gammam
    );
}

// Function to update particles position and scalar fields
__global__
void update_particles(float4 *positions, const float4 *forces, const float4 *elasticforces,  int N, thrust::random::default_random_engine* engines, thrust::random::normal_distribution<float>* ndists,
                      const float dt, const float D, const float lambda, const float rd, const float rm, const float dtnoise, const float epiev)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;


    thrust::random::default_random_engine& eng = engines[idx];
    thrust::random::normal_distribution<float>& distr = ndists[idx];

    float4& pos = positions[idx];
    const float4& force = forces[idx];
    const float4 &elasticforce = elasticforces[idx];

    //Evolve according to Langevin equation
    pos.x += dt*(force.x+elasticforce.x) + dtnoise*distr(eng);
    pos.y += dt*(force.y+elasticforce.y) + dtnoise*distr(eng);
    pos.z += dt*(force.z+elasticforce.z) + dtnoise*distr(eng);

    pos.x = wrap_float(pos.x, box_size);
    pos.y = wrap_float(pos.y, box_size);
    pos.z = wrap_float(pos.z, box_size);

    if(epiev != 0.0){

        pos.w = pos.w + epiev*dt*(force.w + elasticforce.w + rm - lambda*pos.w*pos.w*pos.w-pos.w*pos.w*pos.w*pos.w*pos.w) + sqrtf(epiev)*dtnoise*distr(eng);
        
        // Implicit modification through rd*pos.w term
        pos.w = pos.w/(1.0f+epiev*dt*rd);

    }

};

// Corrected elastic force kernel: minimum-image + Hookean springs
__global__
void elastic_force(const float4 *positions,
                   const unsigned int *chain_indices,
                   const unsigned int *d_id_to_index,
                   float4 *elasticforces,
                   const float kel,
                   const float r0,
                   const float Dmfactor,
                   const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float4 pos = positions[idx];
    float4 &ef = elasticforces[idx];
    ef.x = ef.y = ef.z = ef.w = 0.0f;

    unsigned int idx_chain = chain_indices[idx];
    unsigned int idx_prev = (idx_chain == 0u) ? (unsigned int)(N - 1) : idx_chain - 1u;
    unsigned int idx_forw = (idx_chain == (unsigned int)(N - 1)) ? 0u : idx_chain + 1u;

    unsigned int vprev = d_id_to_index[idx_prev];
    unsigned int vforw = d_id_to_index[idx_forw];

    const float4 neigh_prev = positions[vprev];
    const float4 neigh_forw = positions[vforw];

    // helper inline for min-image (unchanged semantics)
    auto min_image = [&](float dx)->float{
        return dx - box_size * roundf(dx / box_size);
    };

    // prev
    float dx = min_image(pos.x - neigh_prev.x);
    float dy = min_image(pos.y - neigh_prev.y);
    float dz = min_image(pos.z - neigh_prev.z);
    float r2 = dx*dx + dy*dy + dz*dz;
    const float eps = 1e-8f;
    if (r2 > eps) {
        float r = sqrtf(r2);
        float coeff = (r - r0) / r;               // (r - r0)/r
        float fmag = -kel * coeff;                // -k * (r - r0) / r
        ef.x += fmag * dx;
        ef.y += fmag * dy;
        ef.z += fmag * dz;
    }

    // forward
    dx = min_image(pos.x - neigh_forw.x);
    dy = min_image(pos.y - neigh_forw.y);
    dz = min_image(pos.z - neigh_forw.z);
    r2 = dx*dx + dy*dy + dz*dz;
    if (r2 > eps) {
        float r = sqrtf(r2);
        float coeff = (r - r0) / r;
        float fmag = -kel * coeff;
        ef.x += fmag * dx;
        ef.y += fmag * dy;
        ef.z += fmag * dz;
    }

    // optionally set ef.w (Laplacian for scalar field) if desired below:
    // ef.w = Dmfactor * (-2.0f * pos.w + neigh_prev.w + neigh_forw.w);
}

// Calculate elastic force with FENE force
// __global__
// void elastic_force_FENE(particle *particles, const float kel, const float r0, const float r0sq, const float r0threshold, const float req,
//                         const int N){
    
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= (N)) return;
    
//     //Define geometric variables
//     particle& p = particles[idx];
//     int idx_prev = 0;
//     int idx_forw = 0;
//     float rdprev = 0.0f;
//     float rdforw = 0.0f;
    
//     //Get indices of previous and next particle on the chain
//     if (idx == 0){
//         idx_prev = N-1;
//     }
//     else{
//         idx_prev = idx-1;
//     }
//     if(idx == N-1){
//         idx_forw = 0;
//     }
//     else{
//         idx_forw = idx+1;
//     }

//     // Get nearest neighbor particles
//     particle& neigh_prev = particles[idx_prev];
//     particle& neigh_forw = particles[idx_forw];
//     //Calculate distances of nearest neighbours
//     // rdprev = distance_squared(p, neigh_prev);
//     // rdforw = distance_squared(p, neigh_forw);
//     float3 vecprev = make_float3(p.pos.x-neigh_prev.pos.x-req,p.pos.y-neigh_prev.pos.y-req,p.pos.z-neigh_prev.pos.z-req);
//     float3 vecforw = make_float3(p.pos.x-neigh_forw.pos.x-req,p.pos.y-neigh_forw.pos.y-req,p.pos.z-neigh_forw.pos.z-req);
//     rdprev = distance_squared_v(vecprev);
//     rdforw = distance_squared_v(vecforw);
    
//     //Calculate elastic force
//     if(rdprev > r0threshold){
//         p.forceel.x = -100*vecprev.x;
//         p.forceel.y = -100*vecprev.y;
//         p.forceel.z = -100*vecprev.z;
//     }
//     else{
//         p.forceel.x = -kel*(vecprev.x)/(1-rdprev/r0sq);
//         p.forceel.y = -kel*(vecprev.y)/(1-rdprev/r0sq);
//         p.forceel.z = -kel*(vecprev.z)/(1-rdprev/r0sq);

//     }
//     if(rdforw > r0threshold){
//         p.forceel.x -= 100*vecforw.x;
//         p.forceel.y -= 100*vecforw.y;
//         p.forceel.z -= 100*vecforw.z;
//     }
//     else{
//         p.forceel.x -= kel*(vecforw.x)/(1-rdforw/r0sq);
//         p.forceel.y -= kel*(vecforw.y)/(1-rdforw/r0sq);
//         p.forceel.z -= kel*(vecforw.z)/(1-rdforw/r0sq);

//     }

// };