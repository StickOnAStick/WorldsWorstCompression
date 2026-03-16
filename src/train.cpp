#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "models/autoencoder.hpp"
#include "Modern-Text-Tokenizer/Modern-Text-Tokenizer.hpp"

static constexpr size_t EPOCHS = 1000;
static constexpr size_t FILE_COUNT = 100;
static constexpr int    input_dim = 100;
static constexpr int    latent_dim = 30;

static const std::string VOCAB_PATH = "./data/vocab/english_words.txt";
static const std::string BASE_DATA_PATH = "./data/random_words/";

std::vector<std::string> get_file_tokens(
    MecanikDev::TextTokenizer& tokenizer,
    std::ifstream& ifile
){
    std::vector<std::string> res;
    std::string word;
    while(ifile >> word) {
        std::vector<std::string> w_tokens = tokenizer.tokenize(word);
        res.insert(res.end(), w_tokens.begin(), w_tokens.end());
    }

    return res;
}


torch::Tensor

int main() {

    MecanikDev::TextTokenizer tokenizer;
    tokenizer.load_vocab(VOCAB_PATH);

    AutoEncoder model(input_dim, latent_dim);
    model->train();

    torch::optim::Adam optimizer(model->parameters(), 1e-3);


    for (int epoch = 0; epoch < EPOCHS; ++epoch) {

        for(size_t i = 0; i < FILE_COUNT; i++){
            std::ifstream input_file(BASE_DATA_PATH + std::to_string(i));
            if (input_file.fail() || !input_file.is_open()) {
                std::cerr << "Failed to open data file: " << i << std::endl;
                exit(1);
            }

            auto output = model->forward(data);
            auto loss = torch::mse_loss(output, data);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            std::cout << "Epoch " << epoch
                    << " Loss: " << loss.item<float>() << std::endl;
        }
    }

    torch::save(model, "autoencoder.pt");
}