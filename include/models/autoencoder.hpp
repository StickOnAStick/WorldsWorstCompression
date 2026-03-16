#pragma once
#include "torch/torch.h"

struct AutoEncoderImpl : torch::nn::Module {
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Sequential decoder{nullptr};

    AutoEncoderImpl(int input_dim, int latent_dim);

    torch::Tensor forward(torch::Tensor x);
    torch::Tensor encode(torch::Tensor x);
    torch::Tensor decode(torch::Tensor z);
};

TORCH_MODULE(AutoEncoder);
