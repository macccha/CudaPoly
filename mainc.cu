#include "particleslist.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using json = nlohmann::json;


int main()
{


    //Read parameters
    std::ifstream file("params.json");
    json params;
    file >> params;
    const int N = params["N"]; //Bitshift definition, 2^bitshift particles 
    const float L = N/10.0;
    const float ds = L/N;
    const float dt = params["dt"];
    const float T = params["T"];
    const int Tsteps = T/dt;
    const float D = params["D"]; //Diffusion constant
    const int threads_per_block = params["threads_per_block"];
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    //Interaction parameters
    const float A = params["A"]; 
    const float lambda = params["lambda"];
    const float shiftrep = params["shiftrep"];
    const int expn = params["expn"];
    const float kel = params["kel"];
    const float r0 = params["r0"];
    // Parameters for dynamics of epigenetic fields
    const float rd = params["rd"];
    const float rm = params["rm"];
    const float lambdam = params["lambdam"];
    const float gammam = params["gammam"];
    //Saving paraneters
    const int t_save = params["t_save"]; //Save every t_save time steps
    const int Nt_save = Tsteps/t_save; //Number of time steps saved
    //PI constant
    const float pi = 3.14159265358979323846;

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

    //Initialize vectors to save particle positions, create counter to save
    thrust::device_vector<particle> save_vectors[Nt_save];
    for(int ts = 0; ts < Nt_save; ts++){
        save_vectors[ts].resize(N);
    }
    int counter = 0;

    cudaDeviceSynchronize();

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


    //Stat timer
    clock_sim.start();

    printf("Saving for %d steps \n", Nt_save);

    printf("Evolve as Gaussian...\n");
    //First evolve as Gaussian chain
    for(int step = 0; step < Tsteps/4; step++){

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            kel, r0, N
        );

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm
        );

        cudaDeviceSynchronize();

        if(step % t_save == 0){
            save_vectors[counter] = particles;
            counter += 1;
        }
    };

    printf("Evolve as Self-avoiding walk...\n");
    //Evolve as self-avoiding walk
    for(int step = Tsteps/4; step < Tsteps/2; step++){
        

        //Update elastic forces between two consecutive particles
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            kel, r0, N
        );

        // cudaDeviceSynchronize();

        if(step % 5 == 0){


            //Reset forces
            reset_forces<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                N
            );
            
            //Reindex the particle indices vector to normal sequence
            thrust::sequence(particles_idxs.begin(),particles_idxs.end());
            //Create vector of particle hashes
            thrust::transform(particles.begin(), particles.end(), particle_hashes.begin(),
                            particleHash{grid_size, cell_size});
            //Order particle indexes by hash
            thrust::sort_by_key(thrust::device,particle_hashes.begin(), particle_hashes.end(), particles_idxs.begin());
            
            //Get initial and final indexes of unique cell hashes in particles vector
            thrust::sequence(cell_start.begin(), cell_start.end(), -1);
            thrust::sequence(cell_end.begin(), cell_end.end(), -1);
            thrust::lower_bound(thrust::device,particle_hashes.begin(), particle_hashes.end(), unique_hashes.begin(), unique_hashes.end(), cell_start.begin());
            thrust::upper_bound(thrust::device,particle_hashes.begin(), particle_hashes.end(), unique_hashes.begin(), unique_hashes.end(), cell_end.begin());

            cudaDeviceSynchronize();

            //Do neighbor search and update forces according to two-body interactions
            neighbor_search_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                thrust::raw_pointer_cast(particles_idxs.data()),
                thrust::raw_pointer_cast(cell_start.data()),
                thrust::raw_pointer_cast(cell_end.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                thrust::raw_pointer_cast(num_neighbors.data()),
                N, grid_size, cutoff_squared, ds, rep, A, lambda, shiftrep, expn, gammam
            );
        }
            
        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm
        );

        cudaDeviceSynchronize();

        if(step % t_save == 0){
            save_vectors[counter] = particles;
            counter += 1;
        }
    };

    printf("Evolve...\n");

    //Set rep to correct value
    rep = 1.0;//pi*A;

    //Initialize epigentic field to certain average
    InitEpiAvg<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(particles.data()), N, 5.0);

    cudaDeviceSynchronize();


    //Evolve as interacting walk
    for(int step = Tsteps/2; step < Tsteps; step++){
        

        //Update elastic forces between two consecutive particles
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            kel, r0, N
        );

        // cudaDeviceSynchronize();


        if(step % 5 == 0){

            //Reset forces
            reset_forces<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                N
            );

            // cudaDeviceSynchronize();


            //Reindex the particle indices vector to normal sequence
            thrust::sequence(particles_idxs.begin(),particles_idxs.end());
            //Create vector of particle hashes
            thrust::transform(particles.begin(), particles.end(), particle_hashes.begin(),
                            particleHash{grid_size, cell_size});
            //Order particle indexes by hash
            thrust::sort_by_key(thrust::device, particle_hashes.begin(), particle_hashes.end(), particles_idxs.begin());
            
            //Get initial and final indexes of unique cell hashes in particles vector
            thrust::sequence(cell_start.begin(), cell_start.end(), -1);
            thrust::sequence(cell_end.begin(), cell_end.end(), -1);
            thrust::lower_bound(thrust::device,particle_hashes.begin(), particle_hashes.end(), unique_hashes.begin(), unique_hashes.end(), cell_start.begin());
            thrust::upper_bound(thrust::device,particle_hashes.begin(), particle_hashes.end(), unique_hashes.begin(), unique_hashes.end(), cell_end.begin());

            cudaDeviceSynchronize();

            //Do neighbor search and update forces according to two-body interactions
            neighbor_search_kernel<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(particles.data()),
                thrust::raw_pointer_cast(particles_idxs.data()),
                thrust::raw_pointer_cast(cell_start.data()),
                thrust::raw_pointer_cast(cell_end.data()),
                thrust::raw_pointer_cast(particle_hashes.data()),
                thrust::raw_pointer_cast(num_neighbors.data()),
                N, grid_size, cutoff_squared, ds, rep, A, lambda, shiftrep, expn, gammam
            );
        }

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(particles.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(random_engines_m.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            thrust::raw_pointer_cast(normal_distrs_m.data()),
            dt, D, lambdam, rd, rm
        );

        cudaDeviceSynchronize();

        if(step % t_save == 0){
            save_vectors[counter] = particles;
            counter += 1;
        }
    };



    printf("Saving data...\n");
    //Particle vector to fetch from device
    std::vector<std::vector<particle>> save_particles(Nt_save, vector<particle>(N));
    //Vector of particle positions to save
    std::vector<float> save_pos(Nt_save*N*4);

    for(int ts = 0; ts < Nt_save; ts++){
        thrust::copy(save_vectors[ts].begin(), save_vectors[ts].end(), save_particles[ts].begin());
    }
    for(int ts = 0; ts < Nt_save; ts++){
        for(int i = 0; i < N; i++){
            save_pos[ts*(N*4)+i*4] = save_particles[ts][i].pos.x;
            save_pos[ts*(N*4)+i*4+1] = save_particles[ts][i].pos.y;
            save_pos[ts*(N*4)+i*4+2] = save_particles[ts][i].pos.z;
            save_pos[ts*(N*4)+i*4+3] = save_particles[ts][i].m;
        }
    }

    std::ofstream ofs_particle("/data/others/ciarchi/PolymerDyn/particles.bin", std::ios::binary);

    // Write particle data
    size_t particle_count = save_pos.size();
    ofs_particle.write(reinterpret_cast<const char*>(save_pos.data()), sizeof(float) * particle_count);
    ofs_particle.close();


    //End timer
    clock_sim.stop();
    printf("Elapsed time: %f seconds\n", clock_sim.elapsedSeconds());
   
};

