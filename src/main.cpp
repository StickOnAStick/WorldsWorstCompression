#include <stdio.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <torch/torch.h>
#include <CLI/CLI.hpp>
#include "models/autoencoder.hpp"


std::string parse_input(CLI::App& app, int argc, char** argv) {
    std::string file_path;
    std::string raw_text;

    auto file_opt = app.add_option("-f,--file", file_path, "Input file");
    auto text_opt = app.add_option("-t,--text", raw_text, "Raw text input");

    file_opt->excludes(text_opt);
    text_opt->excludes(file_opt);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::exit(app.exit(e));
    }

    std::string input;

    if (!file_path.empty()) {
        std::ifstream file(file_path);

        input.assign(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        );
    } else {
        input = raw_text;
    }

    return input;
}

int main(int argc, char** argv) {

    CLI::App app{"AutoCompressor"};
    // Add the preparse callback
    app.preparse_callback([](size_t argCount)-> void{
        if (argCount == 0) {
            throw(CLI::CallForHelp());
        }
    });

    const std::string input = parse_input(app, argc, argv);


    std::cout << "Input:\n" << input << "\n";

    // Example: instantiate the AutoEncoder (input_dim, latent_dim)
    AutoEncoder model(128, 64, 32);
    (void)model;

    return 0;
}