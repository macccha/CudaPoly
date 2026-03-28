#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <iostream>

#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>

//PI constant
const float ppi = 3.14159265358979323846;
//System geometry variables
const int grid_size = 10;
const float b = 0.0f; //3333333333333f;
const float cutoff_rep = 10.0f;//0.408248290464f;
const float cutoff = 10.0f;
const float cell_size = 10.0f;
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
    unsigned int seed;
    initialize_engines(unsigned int s) : seed(s) {}
    __device__ void operator()(thrust::random::default_random_engine &engine) {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        engine = thrust::random::default_random_engine(seed ^ (thread_id * 2654435761u));
    }
};


// Function to initialize normal distributions
struct initialize_distributions {
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


// Copy to save array
__global__
void CopyArrayToSave(float4 *save_vector, const float4 *positions, const unsigned int *chain_indices, const int start, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int index_copy = chain_indices[idx];
    save_vector[index_copy+start] = positions[idx];
}


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


// Cell list rebuild only: hashing, sorting, mapping
inline void RebuildCellList(
    thrust::device_vector<float4>         &positions,
    thrust::device_vector<float4>         &positions_dup,
    thrust::device_vector<unsigned int>   &chain_indices,
    thrust::device_vector<unsigned int>   &chain_indices_dup,
    thrust::device_vector<unsigned int>   &d_id_to_index,
    thrust::device_vector<unsigned int>   &sorting_idxs,
    thrust::device_vector<int>            &particle_hashes,
    thrust::device_vector<int>            &cell_start,
    thrust::device_vector<int>            &cell_end,
    const int N, const int num_blocks, const int threads_per_block)
{
    // Hash particles into grid cells
    calculateParticleHashes<<<num_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(positions.data()),
        thrust::raw_pointer_cast(particle_hashes.data()),
        grid_size, cell_size, N
    );

    // Sort particle indices and positions by hash
    thrust::sequence(sorting_idxs.begin(), sorting_idxs.end());
    thrust::sort_by_key(thrust::device,
                        particle_hashes.begin(), particle_hashes.end(),
                        sorting_idxs.begin());
    thrust::gather(thrust::device,
                   sorting_idxs.begin(), sorting_idxs.end(),
                   positions.begin(), positions_dup.begin());
    thrust::gather(thrust::device,
                   sorting_idxs.begin(), sorting_idxs.end(),
                   chain_indices.begin(), chain_indices_dup.begin());
    thrust::copy(positions_dup.begin(),     positions_dup.end(),     positions.begin());
    thrust::copy(chain_indices_dup.begin(), chain_indices_dup.end(), chain_indices.begin());

    // Map chain indices to array positions
    MapChainIDToIndex<<<num_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(chain_indices.data()),
        thrust::raw_pointer_cast(d_id_to_index.data()),
        N
    );

    // Reset and fill cell start/end arrays
    cudaMemset(thrust::raw_pointer_cast(cell_start.data()), 0xFF, cell_numbers * sizeof(int));
    cudaMemset(thrust::raw_pointer_cast(cell_end.data()),   0xFF, cell_numbers * sizeof(int));
    findCellStartEnd<<<num_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(particle_hashes.data()),
        thrust::raw_pointer_cast(cell_start.data()),
        thrust::raw_pointer_cast(cell_end.data()),
        N
    );
}
