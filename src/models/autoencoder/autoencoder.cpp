#include "models/autoencoder.hpp"
#include <torch/torch.h>

AutoEncoderImpl::AutoEncoderImpl(int input_dim, int medial_dim, int latent_dim) {

    encoder = torch::nn::Sequential(
        torch::nn::Linear(input_dim, medial_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(medial_dim, latent_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(latent_dim, latent_dim)
    );

    decoder = torch::nn::Sequential(
        torch::nn::Linear(latent_dim, latent_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(latent_dim, medial_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(medial_dim, input_dim)
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