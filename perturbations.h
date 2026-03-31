#ifndef PERTURBATION_H
#define PERTURBATION_H

#include <string>
#include <fstream>
#include <iostream>
#include <curand_kernel.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Perturbation
{
    bool enabled = false; // Flag for perturbation
    std::string pert_flag = "NoPert";
    bool is_random = false;
    float std_random = 1.0f;
    float start_time = 0.0f; // Start time of perturbation
    float duration = 0.0f; // Duration of perturbation

    float field_value = 0.0f; // Value of field to put in perturbation

    int chain_position = 0; // Initial position of perturbation
    int width = 1; // Width of perturbation

    // runtime flag
    bool active = false;
    // Flag for finished perturbation;
    bool is_perturb_finished = false;
    // Flag for activation completed
    bool is_activated = false;

    Perturbation() {}

    // Load parameters from json file
    void load_from_json(const std::string &filename)
    {
        std::ifstream f(filename);

        if (!f.is_open())
        {
            std::cerr << "Could not open perturbation JSON: " << filename << std::endl;
            exit(1);
        }

        json j;
        f >> j;

        if (j.contains("enabled"))
            enabled = j["enabled"];
        
        if (j.contains("pert_flag") && enabled == true)
            pert_flag = j["pert_flag"];

        if (j.contains("is_random"))
            is_random = j["is_random"];
        
        if (j.contains("std_random"))
            std_random = j["std_random"];
        
        if (j.contains("start_time"))
            start_time = j["start_time"];

        if (j.contains("duration"))
            duration = j["duration"];

        if (j.contains("field_value"))
            field_value = j["field_value"];

        if (j.contains("chain_position"))
            chain_position = j["chain_position"];

        if (j.contains("width"))
            width = j["width"];
    }

    // Update runtime state
    void update(float time)
    {
        if (!enabled)
        {
            active = false;
            return;
        }

        active = (time >= start_time && time <= start_time + duration);
        is_perturb_finished = (time >= start_time+duration);
    }

    void print() const
    {
        if(enabled == true && ! is_random){
            std::cout << "Perturbation parameters: \n";
            std::cout << "Enabled: " << enabled << "\n";
            std::cout << "Is random: " << is_random << "\n";
            std::cout << "Start time: " << start_time << "\n";
            std::cout << "Duration: " << duration << "\n";
            std::cout << "Field value: " << field_value << "\n";
            std::cout << "Chain_position: " << chain_position << "\n";
            std::cout << "Width: " << width << "\n";
        }
        else if(enabled == true && is_random){
            std::cout << "Perturbation parameters: \n";
            std::cout << "Enabled: " << enabled << "\n";
            std::cout << "Is random: " << is_random << "\n";
            std::cout << "Start time: " << start_time << "\n";
            std::cout << "Duration: " << duration << "\n";
            std::cout << "Noise STD: " << std_random << "\n";
        }
        else{
            printf("Perturbation not enabled. \n");
        }
        
    }
};

__global__
void init_rand_states(curandState *rand_states, const int N, const unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curand_init(seed, idx, 0, &rand_states[idx]);
}

__global__
void perturb_epifield(float4 *positions, const unsigned int *d_id_to_index, const int N, 
                      const int chain_position, const int width, const float field_value,
                      const bool use_random, const float noise_stddev,
                      curandState *rand_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    

    unsigned int particle_idx = d_id_to_index[idx];

    if (use_random) {
        curandState local_state = rand_states[idx];
        positions[particle_idx].w = positions[particle_idx].w + curand_normal(&local_state) * noise_stddev;
    } else {
        // if (idx < chain_position || idx > chain_position + width) return;
        // else positions[particle_idx].w = field_value;
        float field_at_pos = positions[particle_idx].w;
        positions[particle_idx].w = field_at_pos < 0 ? field_value : field_at_pos;
    }
}

#endif