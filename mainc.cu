#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "particleslist.h"
#include "forces.h"
#include "adaptive_time_step.h"
// #include "verlet_skin.h"   // must come after particleslist.h, relies on box_size
#include "perturbations.h"
#include "gaussian_ring_init.h"
#include <nlohmann/json.hpp>


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

    // Read seed for random number generator
    unsigned int rng_seed;
    unsigned int ext_seed; // Seed for initial random walk configuration
    if (argc >= 3) {
        rng_seed = static_cast<unsigned int>(std::stoul(argv[2]));
        ext_seed = static_cast<unsigned int>(std::stoul(argv[2]));
    } else {
        rng_seed = static_cast<unsigned int>(std::time(nullptr)); // fallback
        ext_seed = static_cast<unsigned int>(std::time(nullptr));
    }
    printf("RNG seed: %u\n", rng_seed);

    // Load parameters
    const std::string saveflag = params["saveflag"];
    const std::string init_conf = params["init_conf"];
    const int N = params["N"]; //Bitshift definition, 2^bitshift particles 
    const float L = params["L"]; // Length of the polymer
    const float ds = L/N; // Space step resolution
    const float dt = params["dt"];
    const float dx_thresh = params["dx_thresh"];
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
    const float expn = params["expn"];
    const float kel = params["kel"];
    const float r0 = params["r0"];
    const float r0sq = r0*r0;
    const float r0threshold = r0sq-0.2*r0sq;
    const float req = params["req"];
    // Parameters for dynamics of epigenetic fields
    const float rd = params["rd"]; // De-methylation rate
    float rm = params["rm"]; // Methylation rate
    const float lambdam = params["lambdam"]; // Quartic interaction
    const float gammam = params["gammam"]; // Polymer-epigenetic field interaction
    const float Dm = params["Dm"]; //Laplacian prefactor for scalar field
    const float Dmfactor = Dm/(ds*ds);
    // Parameters for evolution at constant epigenet field
    float is_epi_dyn = params["is_epi_dyn"]; // Is the epigenetic field fixed?
    const float epi_scale = params["epi_scale"];
    const float init_epi_value = params["init_epi_value"]; // Value of fixed epigenetic field
    //Saving paraneters
    const int t_save = params["t_save"]; //Save every t_save time steps
    const int Nt_save = Tsteps/t_save; //Number of time steps saved
    //PI constant
    const float pi = 3.14159265358979323846;

    //Define counter for particle interaction calculations
    const int steps_list_calc = params["steps_list_calc"];
    int step_list_cal = steps_list_calc-1;

    // Get skin parameter for Verlet Skin algorithm
    // const float skin = params["skin"]; 
    // const float skin_trigger_sq = (skin/2.0f) * (skin/2.0f);  // Rebuild if max disp > skin/2

    printf("-------------------------------------- \n");

    //Create random engines
    thrust::device_vector<thrust::random::default_random_engine> random_engines(N);
    thrust::device_vector<thrust::random::default_random_engine> andom_engines_m(N);
    thrust::device_vector<thrust::random::normal_distribution<float>> normal_distrs(N);

    //Initialize random engines
    thrust::for_each(random_engines.begin(), random_engines.end(), initialize_engines(rng_seed));
    thrust::for_each(normal_distrs.begin(), normal_distrs.end(), initialize_distributions());
    
    //Initialize vectors for particles, cells and hashes
    thrust::device_vector<float4> positions(N);
    thrust::device_vector<float4> positions_snapshot(N); // Position snapshots for Verlet Skin
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
    float t_save_time = dt*t_save;
    float next_save_time = 0.0f;

    //Counter for showing progress
    float t_show_progress = 0.1f*T; // Show every 10% of total time
    float next_show_time = 40.0f; // This is the time when the real dynamics start

    //Define rep for activating attractive part of force
    float rep = 0.0f;
    
    //Initialize particle positions and hashes
    thrust::counting_iterator<unsigned int> index_sequence(1);
    
    //Initialize particles in ring formation
    initialize_system(init_conf,
                    thrust::raw_pointer_cast(positions.data()),
                    thrust::raw_pointer_cast(forces.data()),
                    thrust::raw_pointer_cast(elasticforces.data()),
                    thrust::raw_pointer_cast(chain_indices.data()),
                    N, L, req, ext_seed,
                    num_blocks, threads_per_block);

    // Initialize vetors for hashes, chain indices and sorting containers
    thrust::sequence(chain_indices.begin(), chain_indices.end());
    thrust::sequence(chain_indices_dup.begin(), chain_indices_dup.end());
    thrust::sequence(d_id_to_index.begin(), d_id_to_index.end());
    thrust::sequence(sorting_idxs.begin(), sorting_idxs.end());
    thrust::device_vector<int> unique_hashes(grid_size * grid_size * grid_size);
    thrust::sequence(unique_hashes.begin(), unique_hashes.end());

    // Print parameters
    printf("Parameters used are: \n");
    std::cout << params.dump(2) << '\n';
    printf("-------------------------------------- \n");


    // Initialize perturbation parameters
    Perturbation perturb;

    // Load from file
    perturb.load_from_json("params_perturbation.json");

    // Set perturbation position at center of chain
    perturb.chain_position = N/2+10;
    perturb.print();

    // Initialize random state for perturbation
    curandState *d_rand_states;
    cudaMalloc(&d_rand_states, N * sizeof(curandState));

    if(perturb.is_random){
        init_rand_states<<<num_blocks, threads_per_block>>>(d_rand_states, N, time(NULL));
    }

    printf("-------------------------------------- \n");

    //Stat timer
    clock_sim.start();

    // Set time parameters
    float t_current = 0.0f;
    float dt_adaptive = 0.0f;

    // Read parameters for external field transition induction
    json params_externalinduction;
    std::ifstream file_induction("./params_induction.json");
    file_induction >> params_externalinduction;
    bool is_external_ind = params_externalinduction["act_flag"];
    // float ext_rm = 0.0f;
    // float start_t_induction = 0.0f;
    // float end_t_induction = 0.0f;
    // bool is_induction_active = false;
    // if(is_external_ind){
    //     ext_rm = params_externalinduction["ext_rm"];
    //     start_t_induction = params_externalinduction["start_t_induction"];
    //     end_t_induction = params_externalinduction["end_t_induction"];
    //     printf("External induction chosen. \n");
    //     printf("External field at %f, starting at %f, finishing at %f. \n", ext_rm, start_t_induction, end_t_induction);
    //     printf("-------------------------------------- \n");

    // }
    // else{
    //     printf("External induction not enabled. \n");
    //     printf("-------------------------------------- \n");
    // }

    printf("Evolve as Gaussian...\n");
    
    // First evolve as Gaussian chain
    // Note: at beginning, particles are ordered according to their position along the chain
    
    float T1 = 10.0f;

    while(t_current <= T1){

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            thrust::raw_pointer_cast(d_id_to_index.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            kel, req, Dmfactor, N
        );

        // Set time step
        dt_adaptive = calculate_dt_adaptive(thrust::raw_pointer_cast(forces.data()),
                    thrust::raw_pointer_cast(elasticforces.data()),
                    N, dx_thresh, dt);

        t_current += dt_adaptive;

        // Set noise time step
        float dtnoise_adaptive = sqrtf(2*D*dt_adaptive);

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            dt_adaptive, D, lambdam, rd, rm, dtnoise_adaptive, 0.0
        );

        //Copy data
        if(t_current >= next_save_time){
            thrust::copy(positions.begin(), positions.end(), save_vectors.begin() + counter * N);
            next_save_time += t_save_time;
            counter += 1;
        }
    };

    printf("Current time is %f. Evolve as Self-avoiding walk...\n", t_current);
   
    // Define total force components
    float forcex = 0.0;
    
    // Evolve as self-avoiding walk
    
    float T2 = T1+30.0f;

    while(t_current <= T2){
        
        // // Update snapshot to calculate displacement from last snapshot
        // float max_disp_sq = ComputeMaxDisplacementSqThrust(
        //     thrust::raw_pointer_cast(positions.data()),
        //     thrust::raw_pointer_cast(positions_snapshot.data()),
        //     N);

        step_list_cal += 1;

        if(step_list_cal % steps_list_calc == 0){
            
            step_list_cal = 0;
            
            // Take new snapshot
            // SnapshotPositions<<<num_blocks, threads_per_block>>>(
            //     thrust::raw_pointer_cast(positions.data()),
            //     thrust::raw_pointer_cast(positions_snapshot.data()),
            //     N);

            // Rebuild neighboring list
            RebuildCellList(positions,
                            positions_dup,
                            chain_indices,
                            chain_indices_dup,
                            d_id_to_index,
                            sorting_idxs,
                            particle_hashes,
                            cell_start,
                            cell_end,
                            N, num_blocks, threads_per_block);

            // Update non-bonded forces
            ComputeNonBondedForces(positions, forces, chain_indices, particle_hashes,
                       cell_start, cell_end, forces_x, num_neighbors,
                       N, num_blocks, threads_per_block,
                       ds, rep, A, lambda, shiftrep, expn, gammam, 1);

        }

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            thrust::raw_pointer_cast(d_id_to_index.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            kel, req, Dmfactor, N
        );

        // Set time step
        dt_adaptive = calculate_dt_adaptive(thrust::raw_pointer_cast(forces.data()),
                    thrust::raw_pointer_cast(elasticforces.data()),
                    N, dx_thresh, dt);

        t_current += dt_adaptive;

        // Set noise time step
        float dtnoise_adaptive = sqrtf(2*D*dt_adaptive);

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            dt_adaptive, D, lambdam, rd, rm, dtnoise_adaptive, 1.0
        );

        //Copy data. Sort again before saving
        if(t_current >= next_save_time){
            CopyArrayToSave<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(save_vectors.data()),
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                counter * N, N);
            next_save_time += t_save_time;
            counter += 1;
        }
    };

    if(is_epi_dyn == 1.0){
        printf("Current time is %f. Starting polymer evolution...\n", t_current);
        // rm = 20.0f; // Activate external field
    }
    else{
        printf("Current time is %f. Equilibrating polymer with constant field...\n", t_current);
        //Initialize epigentic field to certain average
        InitEpiAvg<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(positions.data()), N, init_epi_value);
    }
    

    //Set rep to correct value
    rep = 1.0;//pi*A;
    float T3 = T2 + 40.0f;

    //Equilibrate polymer with certain fixed methylation average
    while(t_current <= T3){
        
        // // Update snapshot to calculate displacement from last snapshot
        // float max_disp_sq = ComputeMaxDisplacementSqThrust(
        //     thrust::raw_pointer_cast(positions.data()),
        //     thrust::raw_pointer_cast(positions_snapshot.data()),
        //     N);

        step_list_cal += 1;

        if(step_list_cal % steps_list_calc == 0){
            
            step_list_cal = 0;
            
            // Take new snapshot
            // SnapshotPositions<<<num_blocks, threads_per_block>>>(
            //     thrust::raw_pointer_cast(positions.data()),
            //     thrust::raw_pointer_cast(positions_snapshot.data()),
            //     N);

            // Rebuild neighboring list
            RebuildCellList(positions,
                            positions_dup,
                            chain_indices,
                            chain_indices_dup,
                            d_id_to_index,
                            sorting_idxs,
                            particle_hashes,
                            cell_start,
                            cell_end,
                            N, num_blocks, threads_per_block);

            // Update non-bonded forces
            ComputeNonBondedForces(positions, forces, chain_indices, particle_hashes,
                       cell_start, cell_end, forces_x, num_neighbors,
                       N, num_blocks, threads_per_block,
                       ds, rep, A, lambda, shiftrep, expn, 4.0, 0); //Equilibrate collapse at low gamma to allow for contact changes

        }

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            thrust::raw_pointer_cast(d_id_to_index.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            kel, req, Dmfactor, N
        );

        // Set time step
        dt_adaptive = calculate_dt_adaptive(thrust::raw_pointer_cast(forces.data()),
                    thrust::raw_pointer_cast(elasticforces.data()),
                    N, dx_thresh, dt);
        

        t_current += dt_adaptive;

        // Set noise time step
        float dtnoise_adaptive = sqrtf(2*D*dt_adaptive);

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            dt_adaptive, D, lambdam, rd, rm, dtnoise_adaptive, is_epi_dyn*epi_scale
        );

        //Copy data. Sort again before saving
        if(t_current >= next_save_time){
            CopyArrayToSave<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(save_vectors.data()),
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                counter * N, N);
            next_save_time += t_save_time;
            counter += 1;
        }

        //Show progress
        if(t_current > next_show_time){
            printf("Current time is %f. \n", t_current);
            next_show_time += t_show_progress;
        }

    };

    if(is_epi_dyn == 0){
        printf("Evolve. Counter is at %d. Time is at %f...\n", counter, t_current);
        is_epi_dyn = 1.0f;
    }

    rm = 0.0f;

    //Evolve as interacting walk
    while(t_current <= T){
        
        // // Update snapshot to calculate displacement from last snapshot
        // float max_disp_sq = ComputeMaxDisplacementSqThrust(
        //     thrust::raw_pointer_cast(positions.data()),
        //     thrust::raw_pointer_cast(positions_snapshot.data()),
        //     N);

        step_list_cal += 1;

        if(step_list_cal % steps_list_calc == 0){
            
            step_list_cal = 0;
            
            // Take new snapshot
            // SnapshotPositions<<<num_blocks, threads_per_block>>>(
            //     thrust::raw_pointer_cast(positions.data()),
            //     thrust::raw_pointer_cast(positions_snapshot.data()),
            //     N);

            // Rebuild neighboring list
            RebuildCellList(positions,
                            positions_dup,
                            chain_indices,
                            chain_indices_dup,
                            d_id_to_index,
                            sorting_idxs,
                            particle_hashes,
                            cell_start,
                            cell_end,
                            N, num_blocks, threads_per_block);

            // Update non-bonded forces
            ComputeNonBondedForces(positions, forces, chain_indices, particle_hashes,
                       cell_start, cell_end, forces_x, num_neighbors,
                       N, num_blocks, threads_per_block,
                       ds, rep, A, lambda, shiftrep, expn, gammam, 0);

        }

        //Update elastic forces between two consecutive particles: first need to resort array
        elastic_force<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(chain_indices.data()),
            thrust::raw_pointer_cast(d_id_to_index.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            kel, req, Dmfactor, N
        );

        // Set time step
        dt_adaptive = calculate_dt_adaptive(thrust::raw_pointer_cast(forces.data()),
                    thrust::raw_pointer_cast(elasticforces.data()),
                    N, dx_thresh, dt);

        t_current += dt_adaptive;

        // Check status of perturbation
        if( perturb.enabled == true && perturb.is_perturb_finished==false){
            perturb.update(t_current);
            if(perturb.active == true && perturb.is_activated == false){
                printf("Perturbation started at %f. \n", t_current);
                is_epi_dyn = 0.0f;
                perturb_epifield<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(positions.data()),
                                thrust::raw_pointer_cast(d_id_to_index.data()),
                                N, perturb.chain_position, perturb.width, perturb.field_value,
                                perturb.is_random,          
                                perturb.std_random,
                                d_rand_states);
                perturb.is_activated = true;
            }
            if(perturb.is_perturb_finished==true){
                printf("Perturbation finished at %f. \n", t_current);
                is_epi_dyn = 1.0f;
            }
        }

        // // Check if external induction needs to be activated
        // if(is_external_ind && !is_induction_active && t_current > start_t_induction && t_current < end_t_induction){
        //     printf("External induction started at at %f. \n", t_current);
        //     rm = ext_rm;
        //     is_induction_active = !is_induction_active;
        // }
        // else if(is_external_ind && is_induction_active && t_current > end_t_induction){
        //     printf("External induction finished at at %f. \n", t_current);
        //     rm = 0.0f;
        //     is_induction_active = !is_induction_active;
        // }
        
        // Set noise time step
        float dtnoise_adaptive = sqrtf(2*D*dt_adaptive);

        //Update particles positions
        update_particles<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(elasticforces.data()),
            N,
            thrust::raw_pointer_cast(random_engines.data()),
            thrust::raw_pointer_cast(normal_distrs.data()),
            dt_adaptive, D, lambdam, rd, rm, dtnoise_adaptive, is_epi_dyn*epi_scale
        );

        //Copy data. Sort again before saving
        if(t_current >= next_save_time && counter < Nt_save){
            CopyArrayToSave<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(save_vectors.data()),
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(chain_indices.data()),
                counter * N, N);
            next_save_time += t_save_time;
            counter += 1;
        }

        //Show progress
        if(t_current > next_show_time){
            printf("Current time is %f. \n", t_current);
            next_show_time += t_show_progress;
        }
    };



    printf("Counter is %d. \n", counter);

    printf("Saving data...\n");
    //Particle vector to fetch from device
    std::vector<float4> save_particles(Nt_save*N);
    //Vector of particle positions to save
    std::vector<float> save_pos(Nt_save*N*4);

    printf("Copying data from device to host...\n");
    counter = (counter > Nt_save) ? Nt_save : counter;
    thrust::copy(save_vectors.begin(), save_vectors.end(), save_particles.begin());
    for(int ts = 0; ts < (counter); ts++){
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

    // Create string with temperature
    std::stringstream stream_D;
    stream_D << std::fixed << std::setprecision(2) << D;
    std::string D_str = stream_D.str();

    // Create string with external field
    std::stringstream stream_rm;
    stream_rm << std::fixed << std::setprecision(0) << rm;
    std::string rm_str = stream_rm.str();

    // Create string with number of particles
    std::stringstream stream_N;
    stream_N << std::fixed << std::setprecision(0) << N;
    std::string N_str = stream_N.str();

    // Create string with seed
    std::stringstream stream_seed;
    stream_seed << std::fixed << rng_seed;
    std::string seed_str = stream_seed.str();

    // Create string with number of saving points
    std::stringstream stream_Ntsave;
    stream_Ntsave << std::fixed << std::setprecision(0) << Nt_save;
    std::string Ntsave_str = stream_Ntsave.str();

    // Flag for perturbation
    std::string pert_flag = perturb.pert_flag;

    // Flag for induction
    std::string ind_flag = is_external_ind ? "ExtInd" : "NoExtInd";


    // Save binary file
    std::ofstream ofs_particle("/data/others/ciarchi/PolymerDyn/DataSim/particles_"+seed_str+"_"+saveflag+"_"+pert_flag+"_"+ind_flag+"_"+Ntsave_str+"_"+N_str+"_"+gamma_str+"_"+D_str+"_"+rm_str+".bin", std::ios::binary);

    // Write particle data
    size_t particle_count = save_pos.size();
    ofs_particle.write(reinterpret_cast<const char*>(save_pos.data()), sizeof(float) * particle_count);
    ofs_particle.close();


    //End timer
    clock_sim.stop();
    printf("Elapsed time: %f seconds\n", clock_sim.elapsedSeconds());
   
};

