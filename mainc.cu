#include "particleslist.h"
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
    const float L = params["L"];
    const float ds = L/N;
    const float dt = params["dt"];
    const float T = params["T"];
    const int Tsteps = T/dt;
    const float D = params["D"]; //Diffusion constant
    // Define GPU threads and block variables
    const int threads_per_block = params["threads_per_block"];
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    //Timestep for noise term
    const float dtnoise = sqrtf(2*dt*D);
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
    thrust::for_each(random_engines_m.begin(), random_engines_m.end(), initialize_engines_m());
    thrust::for_each(normal_distrs.begin(), normal_distrs.end(), initialize_distributions());
    thrust::for_each(normal_distrs_m.begin(), normal_distrs_m.end(), initialize_distributions_m());
    
    //Initialize vectors for particles, cells and hashes
    thrust::device_vector<particle> particles(N);
    thrust::device_vector<unsigned int> particles_idxs(N);
    thrust::device_vector<int> particle_hashes(N);
    thrust::device_vector<int> cell_start(grid_size * grid_size * grid_size, -1);
    thrust::device_vector<int> cell_end(grid_size * grid_size * grid_size, -1);
    thrust::device_vector<int> num_neighbors(N);
    thrust::device_vector<float> forces_x(N);


    //Initialize vectors to save particle positions, create counter to save
    thrust::device_vector<particle> save_vectors(Nt_save*N);
    
    //Counter for saving
    int counter = 0;

    //Define rep for activating attractive part of force
    float rep = 0.0f;
    
    //Initialize particle positions and hashes
    thrust::counting_iterator<unsigned int> index_sequence(1);
    //Initialize particles in ring formation
    InitParticlesRing<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            N, L
        );
    // thrust::transform(index_sequence, index_sequence + N, particles.begin(), particleInit());
    thrust::sequence(particles_idxs.begin(), particles_idxs.end());
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
            thrust::raw_pointer_cast(particles.data()),
            kel, req, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm, dtnoise, 1
        );

        cudaDeviceSynchronize();

        //Copy data
        if(step % t_save == 0){
            thrust::copy(particles.begin(), particles.end(), save_vectors.begin() + counter * N);
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
                thrust::raw_pointer_cast(particles.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                grid_size,
                cell_size,
                N
            );
            // Order particle indices by hash
            thrust::sort_by_key(thrust::device,particle_hashes.begin(), particle_hashes.end(), particles.begin());

            // Get initial and final indexes of unique cell hashes in particles vector
            // setCellStartEnd<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(cell_start.data()),
            //     thrust::raw_pointer_cast(cell_end.data()),N);
            findCellStartEnd<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particle_hashes.data()), 
                thrust::raw_pointer_cast(cell_start.data()), 
                thrust::raw_pointer_cast(cell_end.data()), 
                N
            );

            //Do neighbor search and update forces according to two-body interactions
            neighbor_search_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                // thrust::raw_pointer_cast(particles_idxs.data()),
                thrust::raw_pointer_cast(forces_x.data()),
                thrust::raw_pointer_cast(cell_start.data()),
                thrust::raw_pointer_cast(cell_end.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                thrust::raw_pointer_cast(num_neighbors.data()),
                N, grid_size, cutoff_squared, ds, 0.0, 1.0, lambda, shiftrep, expn, gammam
            );

            cudaDeviceSynchronize();

            // Reorder particles according to index chain for elastic force calculation and saving
            calculateParticleChainIndices<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                thrust::raw_pointer_cast(particles_idxs.data()),
                N
            );
            thrust::sort_by_key(thrust::device,particles_idxs.begin(), particles_idxs.end(), particles.begin());
        

            // float totforcex = 0.0;
            // for(int i = 0; i < N; i ++){
            //     forcex = forces_x[i];
            //     totforcex += forcex;
            // }

            // printf("Total x force is %f. Last particle force is %f \n", totforcex, forcex);

        }

        //Update elastic forces between two consecutive particles
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            kel, req, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm, dtnoise, 1
        );

        cudaDeviceSynchronize();

        //Copy data
        if(step % t_save == 0){
            thrust::copy(particles.begin(), particles.end(), save_vectors.begin() + counter * N);
            counter += 1;
        }
    };

    printf("Equilibrating polymer with constant field...\n");

    //Set rep to correct value
    rep = 1.0;//pi*A;

    //Initialize epigentic field to certain average
    InitEpiAvg<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(particles.data()), N, init_epi_value);

    cudaDeviceSynchronize();

    //Equilibrate polymer with certain fixed methylation average
    for(int step = stop2; step < stop3; step++){
        

        if(step % 5 == 0){
            
            //Create vector of particle hashes
            calculateParticleHashes<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                grid_size,
                cell_size,
                N
            );
            //Create vector of particle hashes
            // thrust::transform(particles.begin(), particles.end(), particle_hashes.begin(),
                            // particleHash{grid_size, cell_size});
            // Order particle indices by hash
            thrust::sort_by_key(thrust::device,particle_hashes.begin(), particle_hashes.end(), particles.begin());

            // Get initial and final indexes of unique cell hashes in particles vector
            // setCellStartEnd<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(cell_start.data()), 
            //     thrust::raw_pointer_cast(cell_end.data()),N);
            findCellStartEnd<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particle_hashes.data()), 
                thrust::raw_pointer_cast(cell_start.data()), 
                thrust::raw_pointer_cast(cell_end.data()), 
                N
            );

            //Do neighbor search and update forces according to two-body interactions
            neighbor_search_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                // thrust::raw_pointer_cast(particles_idxs.data()),
                thrust::raw_pointer_cast(forces_x.data()),
                thrust::raw_pointer_cast(cell_start.data()),
                thrust::raw_pointer_cast(cell_end.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                thrust::raw_pointer_cast(num_neighbors.data()),
                N, grid_size, cutoff_squared, ds, 1.0, 1.0, lambda, shiftrep, expn, gammam
            );


            // Reorder particles according to index chain for elastic force calculation and saving
            calculateParticleChainIndices<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                thrust::raw_pointer_cast(particles_idxs.data()),
                N
            );

            thrust::sort_by_key(thrust::device,particles_idxs.begin(), particles_idxs.end(), particles.begin());


        
        }

        //Update elastic forces between two consecutive particles
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            kel, req, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm, dtnoise, is_epi_dyn
        );

        cudaDeviceSynchronize();

        //Copy data
        if(step % t_save == 0){
            thrust::copy(particles.begin(), particles.end(), save_vectors.begin() + counter * N);
            counter += 1;
        }
    };


    printf("Evolve. Counter is at %d...\n", counter);


    //Evolve as interacting walk
    for(int step = stop3; step < Tsteps; step++){
        

        if(step % 5 == 0){
    
            //Create vector of particle hashes
            calculateParticleHashes<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                grid_size,
                cell_size,
                N
            );
            //Create vector of particle hashes
            // thrust::transform(particles.begin(), particles.end(), particle_hashes.begin(),
                            // particleHash{grid_size, cell_size});
            // Order particle indices by hash
            thrust::sort_by_key(thrust::device,particle_hashes.begin(), particle_hashes.end(), particles.begin());

            // Get initial and final indexes of unique cell hashes in particles vector
            // setCellStartEnd<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(cell_start.data()), 
            //     thrust::raw_pointer_cast(cell_end.data()),N);
            findCellStartEnd<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particle_hashes.data()), 
                thrust::raw_pointer_cast(cell_start.data()), 
                thrust::raw_pointer_cast(cell_end.data()), 
                N
            );
            
            //Do neighbor search and update forces according to two-body interactions
            neighbor_search_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                // thrust::raw_pointer_cast(particles_idxs.data()),
                thrust::raw_pointer_cast(forces_x.data()),
                thrust::raw_pointer_cast(cell_start.data()),
                thrust::raw_pointer_cast(cell_end.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                thrust::raw_pointer_cast(num_neighbors.data()),
                N, grid_size, cutoff_squared, ds, 1.0, 1.0, lambda, shiftrep, expn, gammam
            );

            cudaDeviceSynchronize();

            // Reorder particles according to index chain for elastic force calculation and saving
            calculateParticleChainIndices<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                thrust::raw_pointer_cast(particles_idxs.data()),
                N
            );
            thrust::sort_by_key(thrust::device,particles_idxs.begin(), particles_idxs.end(), particles.begin());
        
        }

        //Update elastic forces between two consecutive particles
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            kel, req, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm, dtnoise, 1
        );

        cudaDeviceSynchronize();

        //Copy data
        if(step % t_save == 0){
            thrust::copy(particles.begin(), particles.end(), save_vectors.begin() + counter * N);
            counter += 1;
            // for(p = 0; p < N; p++){
            //     save_vectors[counter*Ntsave+p] = particles[p];
            // }      
        }

    };



    printf("Counter is %d. \n", counter);

    printf("Saving data...\n");
    //Particle vector to fetch from device
    std::vector<particle> save_particles(Nt_save*N);
    //Vector of particle positions to save
    std::vector<float> save_pos(Nt_save*N*4);

    printf("Copying data from device to host...\n");
    thrust::copy(save_vectors.begin(), save_vectors.end(), save_particles.begin());
    for(int ts = 0; ts < counter; ts++){
        for(int i = 0; i < N; i++){
            save_pos[ts*(N*4)+i*4] = save_particles[ts*(N)+i].pos.x;
            save_pos[ts*(N*4)+i*4+1] = save_particles[ts*(N)+i].pos.y;
            save_pos[ts*(N*4)+i*4+2] = save_particles[ts*(N)+i].pos.z;
            save_pos[ts*(N*4)+i*4+3] = save_particles[ts*(N)+i].m;
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

