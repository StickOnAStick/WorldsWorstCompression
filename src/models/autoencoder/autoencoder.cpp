#include "models/autoencoder.hpp"
#include <torch/torch.h>

AutoEncoderImpl::AutoEncoderImpl(int input_dim, int latent_dim) {

    encoder = torch::nn::Sequential(
        torch::nn::Linear(input_dim, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, latent_dim)
    );

    decoder = torch::nn::Sequential(
        torch::nn::Linear(latent_dim, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, input_dim)
    );

    register_module("encoder", encoder);
    register_module("decoder", decoder);
}

torch::Tensor AutoEncoderImpl::forward(torch::Tensor x) {
    auto z = encoder->forward(x);
    return decoder->forward(z);
}

torch::Tensor AutoEncoderImpl::encode(torch::Tensor x) {
    return encoder->forward(x);
}

torch::Tensor AutoEncoderImpl::decode(torch::Tensor z) {
    return decoder->forward(z);
}