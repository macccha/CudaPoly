#include "particleslist_test.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;
using json = nlohmann::json;


int main(int argc, char *argv[])
{

    //Read parameters
    json params;
    // Different json file depending on execution policy
    printf("Running %s mode. \n", argv[1]);
    if(!strcmp(argv[1], "test")){
        printf("Loading params_test.json. \n");
        std::ifstream file("./params_test.json");
        file >> params;
    } else if(!strcmp(argv[1], "run")){
        printf("Loading params.json. \n");
        std::ifstream file("./params.json");
        file >> params;
    }
    // Load parameters
    const std::string saveflag = params["saveflag"];
    const int N = params["N"]; //Bitshift definition, 2^bitshift particles 
    const float L = params["L"]; // Length of the polymer
    const float ds = L/N; // Space step resolution
    const float dt = params["dt"];
    const float T = params["T"];
    const int Tsteps = T/dt;
    const float D = params["D"]; //Diffusion constant
    // Define GPU threads and block variables
    const int threads_per_block = params["threads_per_block"];
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    const float dtnoise = sqrtf(2*dt*D); // Timestep for noise term
    
    // Parameters for dynamics of polymer
    const float A = params["A"]; 
    const float lambda = params["lambda"];
    const float shiftrep = params["shiftrep"];
    const int expn = params["expn"];
    const float kel = params["kel"];
    const float r0 = params["r0"];
    const float r0sq = r0*r0;
    const float r0threshold = r0sq-0.2*r0sq;
    const float req = params["req"];
    // Parameters for dynamics of epigenetic fields
    const float rd = params["rd"]; // De-methylation rate
    const float rm = params["rm"]; // Methylation rate
    const float lambdam = params["lambdam"]; // Quartic interaction
    const float gammam = params["gammam"]; // Polymer-epigenetic field interaction
    const float Dm = params["Dm"]; //Laplacian prefactor for scalar field
    const float Dmfactor = Dm/(ds*ds);
    // Parameters for evolution at constant epigenet field
    const float is_epi_dyn = params["is_epi_dyn"]; // Is the epigenetic field fixed?
    const float init_epi_value = params["init_epi_value"]; // Value of fixed epigenetic field
    //Saving paraneters
    const int t_save = params["t_save"]; //Save every t_save time steps
    const int Nt_save = Tsteps/t_save; //Number of time steps saved
    //PI constant
    const float pi = 3.14159265358979323846;

    //Devine stop counters for different phase of dynamics
    const int stop1 = Tsteps/4; // Stop for Gaussian dynamics
    const int stop2 = Tsteps/2; // Stop for SAW dynamics
    const int stop3 = 6*(Tsteps/10); // Stop for dynamics with constant field (if seleceted)

    printf("Stops are at %d, %d, %d. \n", stop1, stop2, stop3);
    printf("-------------------------------------- \n");

    //Create random engines
    thrust::device_vector<thrust::random::default_random_engine> random_engines(N);
    thrust::device_vector<thrust::random::default_random_engine> random_engines_m(N);
    thrust::device_vector<thrust::random::normal_distribution<float>> normal_distrs(N);
    thrust::device_vector<thrust::random::normal_distribution<float>> normal_distrs_m(N);

    //Initialize random engines
    thrust::for_each(random_engines.begin(), random_engines.end(), initialize_engines());
    thrust::for_each(normal_distrs.begin(), normal_distrs.end(), initialize_distributions());
    
    //Initialize vectors for particles, cells and hashes
    thrust::device_vector<particle> particles(N);
    thrust::device_vector<float4> positions(N);
    thrust::device_vector<float4> forces(N);
    thrust::device_vector<float4> elasticforces(N);
    thrust::device_vector<unsigned int> chain_indices(N);
    thrust::device_vector<unsigned int> d_id_to_index(N);
    thrust::device_vector<unsigned int> sorting_idxs(N);
    thrust::device_vector<int> particle_hashes(N);
    thrust::device_vector<int> cell_start(grid_size * grid_size * grid_size, -1);
    thrust::device_vector<int> cell_end(grid_size * grid_size * grid_size, -1);
    thrust::device_vector<int> num_neighbors(N);
    thrust::device_vector<float> forces_x(N);

    //Duplicate vectors for gather out-of-place operations
    thrust::device_vector<unsigned int> chain_indices_dup(N);
    thrust::device_vector<float4> positions_dup(N);

    //Initialize vectors to save particle positions, create counter to save
    thrust::device_vector<float4> save_vectors(Nt_save*N);
    
    //Counter for saving
    int counter = 0;

    //Define rep for activating attractive part of force
    float rep = 0.0f;
    
    //Initialize particle positions and hashes
    thrust::counting_iterator<unsigned int> index_sequence(1);
    //Initialize particles in ring formation
    InitParticlesRing<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            N, L
        );

    thrust::sequence(chain_indices_dup.begin(), chain_indices_dup.end());
    thrust::sequence(d_id_to_index.begin(), d_id_to_index.end());
    thrust::sequence(sorting_idxs.begin(), sorting_idxs.end());
    thrust::device_vector<int> unique_hashes(grid_size * grid_size * grid_size);
    thrust::sequence(unique_hashes.begin(), unique_hashes.end());

    // Print parameters
    printf("Parameters used are: \n");
    std::cout << params.dump(2) << '\n';
    printf("-------------------------------------- \n");


    //Stat timer
    clock_sim.start();

    printf("Saving for %d steps \n", Nt_save);

    printf("Evolve as Gaussian...\n");
    // First evolve as Gaussian chain
    // Note: at beginning, particles are ordered according to their position along the chain
    
    for(int step = 0; step < stop1; step++){

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            thrust::raw_pointer_cast(d_id_to_index.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            kel, req, Dmfactor, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm, dtnoise, 0.0
        );

        //Copy data
        if(step % t_save == 0){
            thrust::copy(positions.begin(), positions.end(), save_vectors.begin() + counter * N);
            counter += 1;   
        }
    };

    printf("Evolve as Self-avoiding walk...\n");
   
    // Define total force components
    float forcex = 0.0;
    
    // Evolve as self-avoiding walk
    
    for(int step = stop1; step < stop2; step++){
        
        if(step % 5 == 0){

            //Create vector of particle hashes
            calculateParticleHashes<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                grid_size,
                cell_size,
                N
            );


            // Order particle indices and position by by hashes
            thrust::sequence(sorting_idxs.begin(), sorting_idxs.end()); // Reindex vector of indices
            thrust::sort_by_key(thrust::device, particle_hashes.begin(), particle_hashes.end(), sorting_idxs.begin()); // Order indices by hashes
            thrust::gather(thrust::device, sorting_idxs.begin(), sorting_idxs.end(), positions.begin(), positions_dup.begin());
            thrust::gather(thrust::device, sorting_idxs.begin(), sorting_idxs.end(), chain_indices.begin(), chain_indices_dup.begin());
            thrust::copy(positions_dup.begin(), positions_dup.end(), positions.begin());
            thrust::copy(chain_indices_dup.begin(), chain_indices_dup.end(), chain_indices.begin());

            // Map particles chain indices to vector indices
            MapChainIDToIndex<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(chain_indices.data()), 
                thrust::raw_pointer_cast(d_id_to_index.data()), 
                N
            );


            // Reset cells
            cudaMemset(thrust::raw_pointer_cast(cell_start.data()), 0xFF, cell_numbers * sizeof(int));
            cudaMemset(thrust::raw_pointer_cast(cell_end.data()), 0xFF, cell_numbers * sizeof(int));

            // Get initial and final indexes of unique cell hashes in particles vector
            findCellStartEnd<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particle_hashes.data()), 
                thrust::raw_pointer_cast(cell_start.data()), 
                thrust::raw_pointer_cast(cell_end.data()), 
                N
            );

            //Do neighbor search and update forces according to two-body interactions
            neighbor_search_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(forces.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                thrust::raw_pointer_cast(forces_x.data()),
                thrust::raw_pointer_cast(cell_start.data()),
                thrust::raw_pointer_cast(cell_end.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                thrust::raw_pointer_cast(num_neighbors.data()),
                N, grid_size, cutoff_squared, ds, 0.0, 1.0, lambda, shiftrep, expn, gammam
            );

            // float totforcex = 0.0;
            // for(int i = 0; i < N; i ++){
            //     forcex = forces_x[i];
            //     totforcex += forcex;
            // }

            // printf("Total x force is %f. Last particle force is %f \n", totforcex, forcex);

        }

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            thrust::raw_pointer_cast(d_id_to_index.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            kel, req, Dmfactor, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm, dtnoise, 1.0
        );

        //Copy data. 
        if(step % t_save == 0){
            CopyArrayToSave<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(save_vectors.data()),
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                counter * N, N);
            counter += 1;
        }
    };

    if(is_epi_dyn == 1.0){
        printf("Starting polymer evolution...\n");
    }
    else{
        printf("Equilibrating polymer with constant field...\n");
    }
    

    //Set rep to correct value
    rep = 1.0;//pi*A;

    //Initialize epigentic field to certain average
    InitEpiAvg<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(positions.data()), N, init_epi_value);

    //Equilibrate polymer with certain fixed methylation average
    for(int step = stop2; step < stop3; step++){
        
        if(step % 5 == 0){

            //Create vector of particle hashes
            calculateParticleHashes<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                grid_size,
                cell_size,
                N
            );

            // Order particle indices and position by by hashes
            thrust::sequence(sorting_idxs.begin(), sorting_idxs.end()); // Reindex vector of indices
            thrust::sort_by_key(thrust::device, particle_hashes.begin(), particle_hashes.end(), sorting_idxs.begin()); // Order indices by hashes
            thrust::gather(thrust::device, sorting_idxs.begin(), sorting_idxs.end(), positions.begin(), positions_dup.begin());
            thrust::gather(thrust::device, sorting_idxs.begin(), sorting_idxs.end(), chain_indices.begin(), chain_indices_dup.begin());
            thrust::copy(positions_dup.begin(), positions_dup.end(), positions.begin());
            thrust::copy(chain_indices_dup.begin(), chain_indices_dup.end(), chain_indices.begin());

            // Map particles chain indices to vector indices
            MapChainIDToIndex<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(chain_indices.data()), 
                thrust::raw_pointer_cast(d_id_to_index.data()), 
                N
            );

            // Reset cells
            cudaMemset(thrust::raw_pointer_cast(cell_start.data()), 0xFF, cell_numbers * sizeof(int));
            cudaMemset(thrust::raw_pointer_cast(cell_end.data()),   0xFF, cell_numbers * sizeof(int));

            // Get initial and final indexes of unique cell hashes in particles vector
            findCellStartEnd<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particle_hashes.data()), 
                thrust::raw_pointer_cast(cell_start.data()), 
                thrust::raw_pointer_cast(cell_end.data()), 
                N
            );

            //Do neighbor search and update forces according to two-body interactions
            neighbor_search_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(forces.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                // thrust::raw_pointer_cast(particles_idxs.data()),
                thrust::raw_pointer_cast(forces_x.data()),
                thrust::raw_pointer_cast(cell_start.data()),
                thrust::raw_pointer_cast(cell_end.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                thrust::raw_pointer_cast(num_neighbors.data()),
                N, grid_size, cutoff_squared, ds, 1.0, 1.0, lambda, shiftrep, expn, gammam
            );

        }

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            thrust::raw_pointer_cast(d_id_to_index.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            kel, req, Dmfactor, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm, dtnoise, is_epi_dyn
        );

        //Copy data. Sort again before saving
        if(step % t_save == 0){
            CopyArrayToSave<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(save_vectors.data()),
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                counter * N, N);
            counter += 1;
        }
    };

    if(is_epi_dyn == 0){
        printf("Evolve. Counter is at %d...\n", counter);
    }

    //Evolve as interacting walk
    for(int step = stop3; step < Tsteps; step++){
        
        if(step % 5 == 0){

            //Create vector of particle hashes
            calculateParticleHashes<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                grid_size,
                cell_size,
                N
            );

            // Order particle indices and position by by hashes
            thrust::sequence(sorting_idxs.begin(), sorting_idxs.end()); // Reindex vector of indices
            thrust::sort_by_key(thrust::device, particle_hashes.begin(), particle_hashes.end(), sorting_idxs.begin()); // Order indices by hashes
            thrust::gather(thrust::device, sorting_idxs.begin(), sorting_idxs.end(), positions.begin(), positions_dup.begin());
            thrust::gather(thrust::device, sorting_idxs.begin(), sorting_idxs.end(), chain_indices.begin(), chain_indices_dup.begin());
            thrust::copy(positions_dup.begin(), positions_dup.end(), positions.begin());
            thrust::copy(chain_indices_dup.begin(), chain_indices_dup.end(), chain_indices.begin());

            // Map particles chain indices to vector indices
            MapChainIDToIndex<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(chain_indices.data()), 
                thrust::raw_pointer_cast(d_id_to_index.data()), 
                N
            );

            // Reset cells
            cudaMemset(thrust::raw_pointer_cast(cell_start.data()), 0xFF, cell_numbers * sizeof(int));
            cudaMemset(thrust::raw_pointer_cast(cell_end.data()),   0xFF, cell_numbers * sizeof(int));

            // Get initial and final indexes of unique cell hashes in particles vector
            findCellStartEnd<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particle_hashes.data()), 
                thrust::raw_pointer_cast(cell_start.data()), 
                thrust::raw_pointer_cast(cell_end.data()), 
                N
            );

            //Do neighbor search and update forces according to two-body interactions
            neighbor_search_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(forces.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                thrust::raw_pointer_cast(forces_x.data()),
                thrust::raw_pointer_cast(cell_start.data()),
                thrust::raw_pointer_cast(cell_end.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                thrust::raw_pointer_cast(num_neighbors.data()),
                N, grid_size, cutoff_squared, ds, 1.0, 1.0, lambda, shiftrep, expn, gammam
            );

        }

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            thrust::raw_pointer_cast(d_id_to_index.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            kel, req, Dmfactor, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm, dtnoise, 1.0
        );

        //Copy data. Sort again before saving
        if(step % t_save == 0){
            CopyArrayToSave<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(save_vectors.data()),
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                counter * N, N);
            counter += 1;
        }
    };



    printf("Counter is %d. \n", counter);

    cudaDeviceSynchronize();

    printf("Saving data...\n");
    //Particle vector to fetch from device
    std::vector<float4> save_particles(Nt_save*N);
    //Vector of particle positions to save
    std::vector<float> save_pos(Nt_save*N*4);

    printf("Copying data from device to host...\n");
    thrust::copy(save_vectors.begin(), save_vectors.end(), save_particles.begin());
    for(int ts = 0; ts < counter; ts++){
        for(int i = 0; i < N; i++){
            save_pos[ts*(N*4)+i*4] = save_particles[ts*(N)+i].x;
            save_pos[ts*(N*4)+i*4+1] = save_particles[ts*(N)+i].y;
            save_pos[ts*(N*4)+i*4+2] = save_particles[ts*(N)+i].z;
            save_pos[ts*(N*4)+i*4+3] = save_particles[ts*(N)+i].w;
        }
    }

    // Create string with gamma
    std::stringstream stream_gamma;
    stream_gamma << std::fixed << std::setprecision(2) << gammam;
    std::string gamma_str = stream_gamma.str();

    // Create string with external field
    std::stringstream stream_rm;
    stream_rm << std::fixed << std::setprecision(0) << rm;
    std::string rm_str = stream_rm.str();

    // Create string with number of particles
    std::stringstream stream_N;
    stream_N << std::fixed << std::setprecision(0) << N;
    std::string N_str = stream_N.str();

    // Create string with number of saving points
    std::stringstream stream_Ntsave;
    stream_Ntsave << std::fixed << std::setprecision(0) << Nt_save;
    std::string Ntsave_str = stream_Ntsave.str();


    // Save binary file
    std::ofstream ofs_particle("/data/others/ciarchi/PolymerDyn/DataSim/particles_"+saveflag+"_"+Ntsave_str+"_"+N_str+"_"+gamma_str+"_"+rm_str+".bin", std::ios::binary);

    // Write particle data
    size_t particle_count = save_pos.size();
    ofs_particle.write(reinterpret_cast<const char*>(save_pos.data()), sizeof(float) * particle_count);
    ofs_particle.close();


    //End timer
    clock_sim.stop();
    printf("Elapsed time: %f seconds\n", clock_sim.elapsedSeconds());
   
};

