#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>

//PI constant
const float ppi = 3.14159265358979323846;
//System geometry variables
const int grid_size = 200;
const float b = 0.3333333333333f;
const float cutoff_rep = 3.0f;//0.408248290464f;
const float cutoff = 3.0f;
const float cell_size = 3.0f;
const float cutoff_squared = cutoff * cutoff;
const float box_size = grid_size* cell_size;
const int cell_numbers = grid_size*grid_size*grid_size;

//Define global timer
class Timer{
    public:
        void start()
        {
            m_StartTime = std::chrono::system_clock::now();
            m_bRunning = true;
        }
        
        void stop()
        {
            m_EndTime = std::chrono::system_clock::now();
            m_bRunning = false;
        }
        
        double elapsedMilliseconds()
        {
            std::chrono::time_point<std::chrono::system_clock> endTime;
            
            if(m_bRunning)
            {
                endTime = std::chrono::system_clock::now();
            }
            else
            {
                endTime = m_EndTime;
            }
            
            return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
        }
        
        double elapsedSeconds()
        {
            return elapsedMilliseconds() / 1000.0;
        }

    private:
        std::chrono::time_point<std::chrono::system_clock> m_StartTime;
        std::chrono::time_point<std::chrono::system_clock> m_EndTime;
        bool                                               m_bRunning = false;
};
Timer clock_sim;

//Random engine initialization for particles positions
struct initialize_engines {
    __device__ void operator()(thrust::random::default_random_engine &engine) {
        // You can seed the engine with any value; using thread ID is common
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        engine = thrust::random::default_random_engine(thread_id);
    }
};

//Random engine initialization for epigenetic field
struct initialize_engines_m {
    __device__ void operator()(thrust::random::default_random_engine &engine) {
        // You can seed the engine with any value; using thread ID is common
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        engine = thrust::random::default_random_engine(thread_id*7);
    }
};

// Function to initialize normal distributions
struct initialize_distributions {
    __device__ void operator()(thrust::random::normal_distribution<float> &dist) {
        // For normal distributions, mean = 0.0f and stddev = 1.0f by default
        dist = thrust::random::normal_distribution<float>(0.0f, 1.0f);
    }
};

// Function to initialize normal distributions for field
struct initialize_distributions_m {
    __device__ void operator()(thrust::random::normal_distribution<float> &dist) {
        // For normal distributions, mean = 0.0f and stddev = 1.0f by default
        dist = thrust::random::normal_distribution<float>(0.0f, 1.0f);
    }
};

__device__ int pmod(int i, int n) {
    return (i % n + n) % n;
};

__device__  __host__ float wrap_float(float x, float s) {
    return x - s * floor(x / s);
};

// Particle structure
struct particle {
    float3 pos;
    float3 force;
    float3 forceel;
    float m;
    float forcem;
    int idx_chain;
};

// Particle hash
struct particleHash {
    int grid_size;
    float cell_size;

    __host__ __device__
    int operator()(const particle &p) const {
        int cell_x = static_cast<int>(p.pos.x / cell_size);
        int cell_y = static_cast<int>(p.pos.y / cell_size);
        int cell_z = static_cast<int>(p.pos.z / cell_size);
        
        return cell_x + grid_size * (cell_y + grid_size * cell_z);
    }
};

// Particle chain index
struct particleChainIndex {
    __host__ __device__
    int operator()(const particle &p) const {
        return p.idx_chain;
    }
};

// Kernel that calculates the hashes for particles given grid and cell size
__global__ 
void calculateParticleHashes(float4 *positions, int*hashes, const int grid_size, const float cell_size, const int N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float4 &pos = positions[idx];
    int cell_x = static_cast<int>(pos.x / cell_size);
    int cell_y = static_cast<int>(pos.y / cell_size);
    int cell_z = static_cast<int>(pos.z / cell_size);
    hashes[idx] = cell_x + grid_size * (cell_y + grid_size * cell_z);
}


// Kernel that maps particle chain indices to array position indices
__global__ void MapChainIDToIndex(unsigned int* chain_indices, unsigned int* d_map, int N) {
    
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) {  
        unsigned int id = chain_indices[idx];
        d_map[id] = idx;
    }
}

// Kernel that calculates in one call the lower and upper bound position of cells
__global__ 
void findCellStartEnd(const int* hashes, int* cell_start, int* cell_end, const int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int current = hashes[i];
    int prev = (i == 0) ? -1 : hashes[i - 1];
    int next = (i == N - 1) ? -1 : hashes[i + 1];

    if (i == 0 || current != prev) cell_start[current] = i;
    if (i == N - 1 || current != next) cell_end[current] = i+1;
}

// Initialize particles in a Ring
__global__
void InitParticlesRing(float4 *positions, float4 *forces, float4 *elasticforces, unsigned int *chain_indices, int N, float L){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float4 &pos = positions[idx];
    float4 &force = forces[idx];
    float4 &elasticforce = elasticforces[idx];
    float x0 = box_size/2;
    float y0 = box_size/2;
    float z0 = box_size/2;
    float dTheta = 2*ppi/N;
    float R = 0.005/dTheta;
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

// Initialize epigenetic field to fixed configuration
__global__
void InitEpiAvg(float4 *positions, const int N, const float avg){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float4& pos = positions[idx];
    pos.w = avg;
}

// Initialize epigenetic field to fixed configuration
__global__
void CopyArrayToSave(float4 *save_vector, const float4 *positions, const unsigned int *chain_indices, const int start, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int index_copy = chain_indices[idx];
    save_vector[index_copy+start] = positions[idx];
}

// Initialize particle with random position and zero force 
struct particleInit {
    __host__ __device__
    particle operator()(const unsigned int seed) {
        thrust::random::default_random_engine rng(seed);
        rng.discard(seed);
        thrust::random::uniform_real_distribution<float> dist(0.45f, 0.55f);
        particle p;
        p.pos.x = dist(rng)*box_size;
        p.pos.y = dist(rng)*box_size; 
        p.pos.z = dist(rng)*box_size;
        p.force.x = 0.0f;
        p.force.y = 0.0f;
        p.force.z = 0.0f;
        p.forcem = 0.0f;
        return p;
    }
};

struct rngInit {
    __host__ __device__
    thrust::random::default_random_engine& operator()(thrust::random::default_random_engine & rng) {
        rng.seed(1101252662);
        
        return rng;
    }
};

__device__
inline float distance_squared(const particle &a, const particle &b) {
    float dx = a.pos.x - b.pos.x;
    dx  -= box_size*round(dx/box_size);
    float dy = a.pos.y - b.pos.y;
    dy  -= box_size*round(dy/box_size);
    float dz = a.pos.z - b.pos.z;
    dz  -= box_size*round(dz/box_size);
    return dx * dx + dy * dy + dz * dz;
};

//Function for float4 vectors to calculate distance
__device__
inline float distance_squared_float4(const float4 &a, const float4 &b) {
    float dx = a.x - b.x;
    dx  -= box_size*round(dx/box_size);
    float dy = a.y - b.y;
    dy  -= box_size*round(dy/box_size);
    float dz = a.z - b.z;
    dz  -= box_size*round(dz/box_size);
    return dx * dx + dy * dy + dz * dz;
};

__device__
inline float distance_squared_v(const float3 &a) {
    float dx = a.x;
    dx  -= box_size*round(dx/box_size);
    float dy = a.y;
    dy  -= box_size*round(dy/box_size);
    float dz = a.z;
    dz  -= box_size*round(dz/box_size);
    return dx * dx + dy * dy + dz * dz;
};

//Function to calculate force module
__device__
inline float force_module(const float A, const float distance, const float lambda, const float exponent, const float shift){
    
    float inverse = 1/(distance+shift);
    inverse = inverse-b;
    return 10.0*inverse;
        
}


//Search kernel function
__global__
void neighbor_search_kernel(const float4 *positions, float4 *forces, unsigned int *chain_indices, float *forces_x, int *cell_start, int *cell_end, int *particle_hashes,int *num_neighbors,  const int N, const int grid_size, const float cutoff_squared, const float ds, float rep, const float A, const float lambda, const float shiftrep, const int expn, const float gamma) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;

    // // Get particle index from list of indices
    int p_idx = chain_indices[idx];
    // Get particle position, force
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

    float force_modconst = lambda*rep*ds*gamma;
    float epi_firce_modconst = ds*gamma;

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
                    
                    
                    const float4& neighbor = positions[j];
                    float distance_squared = distance_squared_float4(pos, neighbor);
                    int idx_neigh = chain_indices[j];

                    if(distance_squared != 0 && distance_squared <= cutoff_squared){

                        // Forces moduli
                        float distance = sqrtf(distance_squared);
                        if (distance < 1e-4f) {
                            printf("WARNING: distance too small between particles %d and neighbor at distance = %g\n",
                                idx, distance);
                        }
                        
                        float force_mod = 0.0f;
                        float force_mod_epi = 0.0f;
                        // float repulsive_force_mod = 0.0f;

                        // Calculate forces
                        float repulsive_force_mod = ds*A*force_module(A,distance,-6.0,-expn-1,shiftrep); // Repulsive force
                        float attractive_force_mod = force_modconst*exp(lambda*distance)*neighbor.w*pos.w; // Attractive force;

                        // Add attractive and repulsive forces only among non-nearest neighbors neighbors
                        if(p_idx != N-1 && p_idx != 0 && p_idx != (idx_neigh - 1) && (p_idx != idx_neigh + 1)){
                            force_mod = repulsive_force_mod+attractive_force_mod;
                            force_mod_epi = epi_firce_modconst;
                        }
                        else if((p_idx == N-1 && idx_neigh != 0 && idx_neigh != N-2)){
                            force_mod = repulsive_force_mod+attractive_force_mod;
                            force_mod_epi = epi_firce_modconst;

                        }
                        else if((p_idx == 0 && idx_neigh != N-1 && idx_neigh != 1)){
                            force_mod = repulsive_force_mod+attractive_force_mod;
                            force_mod_epi = epi_firce_modconst;

                        }

                        // Total force
                        // float force_mod = repulsive_force_mod+attractive_force_mod;

                        //Update force
                        force.x += force_mod*(pos.x-neighbor.x)/distance;
                        force.y += force_mod*(pos.y-neighbor.y)/distance;
                        force.z += force_mod*(pos.z-neighbor.z)/distance;
                        force.w += epi_firce_modconst*exp(lambda*distance)*neighbor.w;

                    }
                    
                };
            }
        }
    }

    // Save force
    // forces_x[idx] = force.w;
};


// Function to solve implicitly up to third order the EQMs for the scalar field
__device__ float solve_cubic_implicit(const float R, const float dt, const float rd, const float lambda)
{
    const float a = dt * lambda;
    const float b = 1.0f + dt * rd;

    // Solve: a*u^3 + b*u - R = 0

    // Initial guess: R/b (linearized solution)
    float u = R / b;

    // Newton iterations with damping
    for(int it = 0; it < 12; ++it)
    {
        float f  = a*u*u*u + b*u - R;
        float df = 3.0f*a*u*u + b;

        // Safety: if derivative tiny, break (shouldn't normally happen)
        if(fabsf(df) < 1e-12f) break;

        float du = f / df;
        u -= du;

        // Convergence reached
        if(fabsf(du) < 1e-6f) break;

        // Print if convergence failed
        if(it == 11 && fabsf(du)>1e-6f) printf("Convergence failed. \n");
    }

    // Optional safety clamp to avoid large excursions
    const float WMAX = 1e3f;
    if(u >  WMAX) u =  WMAX;
    if(u < -WMAX) u = -WMAX;

    return u;
}


// Function to update particles position and scalar fields
__global__
void update_particles(float4 *positions, const float4 *forces, const float4 *elasticforces,  int N, thrust::random::default_random_engine* engines, thrust::random::default_random_engine* engines_m, thrust::random::normal_distribution<float>* ndists, thrust::random::normal_distribution<float>* ndists_m, const float dt, const float D, const float lambda, const float rd, const float rm, const float dtnoise, const float epiev)
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

        // Quantities for third-order implicit solver
        // float w = pos.w;
        // float R = w
        //         + dt * (force.w + rm - w*w*w*w*w)  // explicit parts
        //         + dtnoise*distr(eng);
       
        pos.w = pos.w + dt*(force.w + rm - lambda*pos.w*pos.w*pos.w - pos.w*pos.w*pos.w*pos.w*pos.w) + dtnoise*distr(eng);
        
        // Implicit modification through rd*pos.w term
        pos.w = pos.w/(1.0f+dt*rd);
        // pos.w = solve_cubic_implicit(R, dt, rd, lambda);
    }

};

// Calculate elastic force
__global__
void elastic_force(const float4 *positions, unsigned int* chain_indices, unsigned int* d_id_to_index, float4 *elasticforces, const float kel, const float r0, const float Dmfactor, const int N){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;
    
    //Define geometric variables
    const float4& pos = positions[idx];
    float4& elasticforce = elasticforces[idx];
    unsigned int idx_curr = chain_indices[idx];
    unsigned int idx_prev = 0;
    unsigned int idx_forw = 0;
    
    //Get indices of previous and next particle on the chain
    if (idx_curr == 0){
        idx_prev = N-1;
    }
    else{
        idx_prev = idx_curr-1;
    }
    if(idx_curr == N-1){
        idx_forw = 0;
    }
    else{
        idx_forw = idx_curr+1;
    }

    
    unsigned int vec_idx_prev = d_id_to_index[idx_prev];
    unsigned int vec_idx_forw = d_id_to_index[idx_forw];
    // Get nearest neighbor particles
    const float4& neigh_prev = positions[vec_idx_prev];
    const float4& neigh_forw = positions[vec_idx_forw];
    //Calculate distances of nearest neighbours
    float3 vecprev = make_float3(pos.x-neigh_prev.x,pos.y-neigh_prev.y,pos.z-neigh_prev.z);
    float3 vecforw = make_float3(pos.x-neigh_forw.x,pos.y-neigh_forw.y,pos.z-neigh_forw.z);
    float distprev = sqrtf(distance_squared_v(vecprev));
    float distforw = sqrtf(distance_squared_v(vecforw));
    float coeffprev = (distprev-r0)/distprev;
    float coeffforw = (distforw-r0)/distforw;

    //Calculate elastic force
    elasticforce.x = -kel*(pos.x-neigh_prev.x)*coeffprev;
    elasticforce.y = -kel*(pos.y-neigh_prev.y)*coeffprev;
    elasticforce.z = -kel*(pos.z-neigh_prev.z)*coeffprev;
    elasticforce.x -= kel*(pos.x-neigh_forw.x)*coeffforw;
    elasticforce.y -= kel*(pos.y-neigh_forw.y)*coeffforw;
    elasticforce.z -= kel*(pos.z-neigh_forw.z)*coeffforw;
    //Laplacian calculation for scalar field
    // elasticforce.w = Dmfactor*(-2*pos.w+neigh_prev.w+neigh_forw.w);


};

// Calculate elastic force with FENE force
__global__
void elastic_force_FENE(particle *particles, const float kel, const float r0, const float r0sq, const float r0threshold, const float req,
                        const int N){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;
    
    //Define geometric variables
    particle& p = particles[idx];
    int idx_prev = 0;
    int idx_forw = 0;
    float rdprev = 0.0f;
    float rdforw = 0.0f;
    
    //Get indices of previous and next particle on the chain
    if (idx == 0){
        idx_prev = N-1;
    }
    else{
        idx_prev = idx-1;
    }
    if(idx == N-1){
        idx_forw = 0;
    }
    else{
        idx_forw = idx+1;
    }

    // Get nearest neighbor particles
    particle& neigh_prev = particles[idx_prev];
    particle& neigh_forw = particles[idx_forw];
    //Calculate distances of nearest neighbours
    // rdprev = distance_squared(p, neigh_prev);
    // rdforw = distance_squared(p, neigh_forw);
    float3 vecprev = make_float3(p.pos.x-neigh_prev.pos.x-req,p.pos.y-neigh_prev.pos.y-req,p.pos.z-neigh_prev.pos.z-req);
    float3 vecforw = make_float3(p.pos.x-neigh_forw.pos.x-req,p.pos.y-neigh_forw.pos.y-req,p.pos.z-neigh_forw.pos.z-req);
    rdprev = distance_squared_v(vecprev);
    rdforw = distance_squared_v(vecforw);
    
    //Calculate elastic force
    if(rdprev > r0threshold){
        p.forceel.x = -100*vecprev.x;
        p.forceel.y = -100*vecprev.y;
        p.forceel.z = -100*vecprev.z;
    }
    else{
        p.forceel.x = -kel*(vecprev.x)/(1-rdprev/r0sq);
        p.forceel.y = -kel*(vecprev.y)/(1-rdprev/r0sq);
        p.forceel.z = -kel*(vecprev.z)/(1-rdprev/r0sq);

    }
    if(rdforw > r0threshold){
        p.forceel.x -= 100*vecforw.x;
        p.forceel.y -= 100*vecforw.y;
        p.forceel.z -= 100*vecforw.z;
    }
    else{
        p.forceel.x -= kel*(vecforw.x)/(1-rdforw/r0sq);
        p.forceel.y -= kel*(vecforw.y)/(1-rdforw/r0sq);
        p.forceel.z -= kel*(vecforw.z)/(1-rdforw/r0sq);

    }

};