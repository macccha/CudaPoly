#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>

//PI constant
const float ppi = 3.14159265358979323846;
//System geometry variables
const int grid_size = 100;
const float b = 6.0f;
const float cutoffb = 1/sqrtf(6);
const float cutoff_rep = 0.408248290464f;
const float cutoff = 1.0f;
const float cell_size = 1.0f;
const float cutoff_squared = cutoff * cutoff;
const float box_size = grid_size* cell_size;

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

// Kernel that set the cell start and end vectors to -1
__global__ 
void setCellStartEnd(int* cell_start, int* cell_end, const int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    //Reset vector content to -1
    cell_start[i] = -1;
    cell_end[i] = -1;
};

// Kernel that calculates the hashes for particles given grid and cell size
__global__ 
void calculateParticleHashes(particle *particles, int*hashes, const int grid_size, const float cell_size, const int N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    particle &p = particles[idx];
    int cell_x = static_cast<int>(p.pos.x / cell_size);
    int cell_y = static_cast<int>(p.pos.y / cell_size);
    int cell_z = static_cast<int>(p.pos.z / cell_size);
    hashes[idx] = cell_x + grid_size * (cell_y + grid_size * cell_z);
}

// Kernel that calculates the particle indices according to the chain order
__global__ 
void calculateParticleChainIndices(particle *particles, unsigned int*indices, const int N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    particle &p = particles[idx];
    indices[idx] = p.idx_chain;
}

// Kernel that calculates in one call the lower and upper bound position of cells
__global__ 
void findCellStartEnd(const int* hashes, int* cell_start, int* cell_end, const int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    //Reset vector content to -1
    cell_start[i] = -1;
    cell_end[i] = -1;
    int current = hashes[i];
    int prev = (i == 0) ? -1 : hashes[i - 1];
    int next = (i == N - 1) ? -1 : hashes[i + 1];

    if (i == 0 || current != prev) cell_start[current] = i;
    if (i == N - 1 || current != next) cell_end[current] = i;
}


// Initialize particles in a Ring
__global__
void InitParticlesRing(particle *particles, int N, float L){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    particle &p = particles[idx];
    float x0 = box_size/2;
    float y0 = box_size/2;
    float z0 = box_size/2;
    float dTheta = 2*ppi/N;
    float R = 0.005/dTheta;
    p.pos.x = x0+R*cos(2*ppi/N*(idx));
    p.pos.y = y0+R*sin(2*ppi/N*(idx));
    p.pos.z = z0;
    p.force.x = 0.0f;
    p.force.y = 0.0f;
    p.force.z = 0.0f;
    p.forceel.x = 0.0f;
    p.forceel.y = 0.0f;
    p.forceel.z = 0.0f;
    p.m = 0.0f;
    p.forcem = 0.0f;
    p.idx_chain = idx;
}

// Initialize epigenetic field to fixed configuration
__global__
void InitEpiAvg(particle *particles, const int N, const float avg){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    particle &p = particles[idx];
    p.m = avg;
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

__device__
inline float force_module(const float A, const float distance, const float lambda, const float exponent, const float shift){
    
    if distance > 0.05
        float inverse = 1/(distance+shift);
        inverse = inverse*inverse-b;
        // inverse = inverse;
        return inverse;
    else
        float inverse = 500.0;
        return inverse;
}

//Function to reset forces
__global__
void reset_forces(particle *particles, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;

    particle& p = particles[idx];
    
    //Reset forces
    p.forcem = 0.0f;
    p.force.x = 0.0f;
    p.force.y = 0.0f;
    p.force.z = 0.0f;
}

//Search kernel function
__global__
void neighbor_search_kernel(particle *particles, float *forces_x, int *cell_start, int *cell_end, int *particle_hashes,int *num_neighbors,  const int N, const int grid_size, const float cutoff_squared, const float ds, float rep, const float A, const float lambda, const float shiftrep, const int expn, const float gamma) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;

    // // Get particle index from list of indices
    // int p_idx = particles_idxs[idx];
    // Get particle
    particle& p = particles[idx];

    //Reset forces
    p.forcem = 0.0f;
    p.force.x = 0.0f;
    p.force.y = 0.0f;
    p.force.z = 0.0f;
    
    // Get the current particle's cell
    // int hash = particle_hashes[idx];

    // Get integer coordinates of particle
    int c_x = static_cast<int>(p.pos.x / cell_size);
    int c_y = static_cast<int>(p.pos.y / cell_size);
    int c_z = static_cast<int>(p.pos.z / cell_size);

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

                int idx_neigh = 0;

                for (int j = start; j < end; j++) {
                    
                    
                    particle &neighbor = particles[j];
                    float distance = sqrtf(distance_squared(p, neighbor));
                    int p_idx = p.idx_chain;
                    int idx_neigh = neighbor.idx_chain;

                    if(distance != 0 && distance <= cutoff_squared){

                        // Forces moduli
                        float force_mod = 0.0f;
                        float repulsive_force_mod = 0.0f;
                        if(distance<cutoff_rep){
                            repulsive_force_mod = ds*A*force_module(A,distance,-6.0,-expn-1,shiftrep);
                        }
                        float attractive_force_mod = lambda*rep*ds*gamma*exp(lambda*distance)*neighbor.m*p.m; // Attractive force;

                        // Add attractive and repulsive forces only among non-nearest neighbors neighbors
                        if(p_idx != N-1 && p_idx != 0 && p_idx != (idx_neigh - 1) && (p_idx != idx_neigh + 1)){
                            force_mod = repulsive_force_mod+attractive_force_mod; 
                        }
                        else if((p_idx == N-1 && idx_neigh != 0 && idx_neigh != N-2)){
                            force_mod = repulsive_force_mod+attractive_force_mod;
                        }
                        else if((p_idx == 0 && idx_neigh != N-1 && idx_neigh != 1)){
                            force_mod = repulsive_force_mod+attractive_force_mod;
                        }

                        //Update force
                        p.force.x += force_mod*(p.pos.x-neighbor.pos.x)/distance;
                        p.force.y += force_mod*(p.pos.y-neighbor.pos.y)/distance;
                        p.force.z += force_mod*(p.pos.z-neighbor.pos.z)/distance;
                        p.forcem += ds*gamma*exp(lambda*distance)*neighbor.m;

                    }
                    
                };
            }
        }
    }

    // Save force
    forces_x[idx] = p.forcem;
};

//Search kernel function
__global__
void neighbor_search_kernel_noidxs(particle *particles, float *forces_x, int *cell_start, int *cell_end, int *particle_hashes,int *num_neighbors,  const int N, const int grid_size, const float cutoff_squared, const float ds, float rep, const float A, const float lambda, const float shiftrep, const int expn, const float gamma) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;

    // Get particle
    particle& p = particles[idx];
    
    // Get the current particle's cell
    int hash = particle_hashes[idx];
    int c_x = static_cast<int>(p.pos.x / cell_size);
    int c_y = static_cast<int>(p.pos.y / cell_size);
    int c_z = static_cast<int>(p.pos.z / cell_size);

    //Define distance, distance vector and force modulus
    float distance_sq = 0.0f;
    float distance = 0.0f;
    float3 distance_vec = make_float3(0.0f,0.0f,0.0f);
    float force_mod = 0.0f;
    float attractive_force_mod = 0.0f;
    float repulsive_force_mod = 0.0f;

    //Check nearby cells in a 3x3x3 neighborhood
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

                if (start == -1 || end == -1) continue;

                for (int j = start; j < end; j++) {
                    
                                            
                        //Get neighbor
                        particle& neighbor = particles[j];
                        float distance = sqrtf(distance_squared(p, neighbor));
                        if(distance!=0.0){
                        

                            if (distance <= cutoff) //Calculate only if in radius of interaction
                            {
                                // //Update distance vector
                                distance_vec.x = p.pos.x-neighbor.pos.x;
                                distance_vec.y = p.pos.y-neighbor.pos.y;
                                distance_vec.z = p.pos.z-neighbor.pos.z;

                                attractive_force_mod = lambda*rep*ds*gamma*exp(lambda*distance)*neighbor.m*p.m; // Attractive force;
                                repulsive_force_mod = ds*A*force_module(A,distance,-2.0,-expn-1,shiftrep);

                                // Add attractive force only among non-nearest neighbors neighbors
                                force_mod = repulsive_force_mod; // Repulsive force
                                force_mod += attractive_force_mod;
                                p.force.x += force_mod*distance_vec.x/distance;
                                p.force.y += force_mod*distance_vec.y/distance;
                                p.force.z += force_mod*distance_vec.z/distance;
                                p.forcem += ds*gamma*exp(lambda*distance)*neighbor.m;
                            }
                    }
                    
                }
            }
        }
    }
    forces_x[idx] = p.force.x;

};


__global__
void update_particles(particle *particles,  int N, thrust::random::default_random_engine* engines, thrust::random::default_random_engine* engines_m, thrust::random::normal_distribution<float>* ndists, thrust::random::normal_distribution<float>* ndists_m, const float dt, const float D, const float lambda, const float rd, const float rm, const float dtnoise, const float epiev)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;


    thrust::random::default_random_engine& eng = engines[idx];
    thrust::random::default_random_engine& engm = engines_m[idx];
    thrust::random::normal_distribution<float>& distr = ndists[idx];
    thrust::random::normal_distribution<float>& distrm = ndists_m[idx];

    particle& p = particles[idx];
    eng.discard(idx);
    engm.discard(idx);

    //Evolve according to Langevin equation
    p.pos.x += dt*(p.force.x+p.forceel.x) + dtnoise*distr(eng);
    p.pos.y += dt*(p.force.y+p.forceel.y) + dtnoise*distr(eng);
    p.pos.z += dt*(p.force.z+p.forceel.z) + dtnoise*distr(eng);

    p.pos.x = wrap_float(p.pos.x, box_size);
    p.pos.y = wrap_float(p.pos.y, box_size);
    p.pos.z = wrap_float(p.pos.z, box_size);

    p.m += epiev*(dt*(p.forcem - rd*p.m + rm - lambda*p.m*p.m*p.m-p.m*p.m*p.m*p.m*p.m) + dtnoise*distr(eng));

};

// Calculate elastic force
__global__
void elastic_force(particle *particles, const float kel, const float r0, const int N){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N)) return;
    
    //Define geometric variables
    particle& p = particles[idx];
    int idx_prev = 0;
    int idx_forw = 0;
    
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
    float3 vecprev = make_float3(p.pos.x-neigh_prev.pos.x,p.pos.y-neigh_prev.pos.y,p.pos.z-neigh_prev.pos.z);
    float3 vecforw = make_float3(p.pos.x-neigh_forw.pos.x,p.pos.y-neigh_forw.pos.y,p.pos.z-neigh_forw.pos.z);
    float distprev = sqrtf(distance_squared_v(vecprev));
    float distforw = sqrtf(distance_squared_v(vecforw));
    float coeffprev = (distprev-r0)/distprev;
    float coeffforw = (distforw-r0)/distforw;

    //Calculate elastic force
    p.forceel.x = -kel*(vecprev.x)*coeffprev;
    p.forceel.y = -kel*(vecprev.y)*coeffprev;
    p.forceel.z = -kel*(vecprev.z)*coeffprev;
    p.forceel.x -= kel*(vecforw.x)*coeffforw;
    p.forceel.y -= kel*(vecforw.y)*coeffforw;
    p.forceel.z -= kel*(vecforw.z)*coeffforw;


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