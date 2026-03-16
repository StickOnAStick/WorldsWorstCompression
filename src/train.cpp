#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <functional>
#include <utility>
#include <any>
#include "models/autoencoder.hpp"
#include "Modern-Text-Tokenizer/Modern-Text-Tokenizer.hpp"

static constexpr size_t EPOCHS = 1000;
static constexpr size_t FILE_COUNT = 1000;
static constexpr int    INPUT_DIM = 1024;
static constexpr int    MEDIAL_DIM = 512;
static constexpr int    LATENT_DIM = 512;

static const std::string VOCAB_PATH = "data/vocab/english_words.txt";
static const std::string BASE_DATA_PATH = "data/random_words/";

torch::Tensor get_file_tokens(
    MecanikDev::TextTokenizer& tokenizer,
    std::ifstream& ifile
){
    std::vector<int> res;
    std::string word;
    while(ifile >> word) {
        std::vector<int> w_tokens = tokenizer.encode(word);
        res.insert(res.end(), w_tokens.begin(), w_tokens.end());
    }

    torch::Tensor token_tensor = torch::tensor(res, torch::dtype(torch::kInt64));

    return token_tensor;
}

// A generic template function to measure the execution time of any callable
template<typename Func, typename... Args>
auto time_function(Func&& func, Args&&... args)
{
    // Record the start time using the steady_clock (best for measuring elapsed time)
    auto start = std::chrono::steady_clock::now();

    // Call the function with perfect forwarding of arguments
    // Use 'auto result = ...' to handle functions with a return value
    // If the function returns void, this will still compile
    if constexpr (std::is_void_v<std::result_of_t<Func(Args...)>>) {
        std::forward<Func>(func)(std::forward<Args>(args)...);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start;
        return std::make_pair(duration, std::any{}); // Return duration and a placeholder
    } else {
        auto result = std::forward<Func>(func)(std::forward<Args>(args)...);
        // Record the end time
        auto end = std::chrono::steady_clock::now();

        // Calculate the duration
        std::chrono::duration<double> duration = end - start;
        
        // Return both the duration and the function's return value
        return std::make_pair(duration, std::any{result});
    }
}

int main() {

    // Hardware configuration
    torch::Device device = torch::mps::is_available() 
                        ? torch::Device(torch::kMPS)
                        : torch::Device(torch::kCPU);

    

    // Model configuration
    MecanikDev::TextTokenizer tokenizer;
    tokenizer.load_vocab(VOCAB_PATH);

    torch::nn::Embedding embedding(
        static_cast<int>(tokenizer.vocab_size()), 
        INPUT_DIM
    );
    embedding->to(device);

    AutoEncoder model(INPUT_DIM, MEDIAL_DIM, LATENT_DIM);
    model->to(device);
    model->train();

    torch::optim::Adam optimizer(
        torch::optim::AdamOptions(1e-3)
    );
    optimizer.add_parameters(model->parameters());
    optimizer.add_parameters(embedding->parameters());
    
    // Runtime stats
    double avg_time_us;
    size_t time_sample_count = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {

        for(size_t i = 0; i < FILE_COUNT; i++){
            std::ifstream input_file(BASE_DATA_PATH + std::to_string(i) + ".txt");
            if (input_file.fail() || !input_file.is_open()) {
                std::cerr << "Failed to open data file: " << i << std::endl;
                exit(1);
            }

            torch::Tensor tokens = get_file_tokens(tokenizer, input_file).to(device);
            torch::Tensor embeded_tokens = embedding(tokens);

            auto output = model->forward(embeded_tokens);
            auto loss = torch::mse_loss(output, embeded_tokens);

            auto start = std::chrono::steady_clock::now();
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            auto end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::micro> duration_us = end - start;
            avg_time_us += duration_us.count();
            time_sample_count++;
            double avg_time = avg_time_us / static_cast<double>(time_sample_count);
            
            std::cout << "Epoch " << epoch
                      << " Loss: " << loss.item<float>() 
                      << "\tTime: " << duration_us.count() 
                      << "\tAvg Time: "<< avg_time << std::endl;

            // Clean
            input_file.close();
        }
    }

    torch::save(model, "autoencoder.pt");
}