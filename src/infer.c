#include <stdio.h>
#include <stdlib.h>
#include "catdog.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_path> [model_path]\n", argv[0]);
        return 1;
    }

    const char *image_path = argv[1];
    const char *model_path = (argc >= 3) ? argv[2] : "catdog.nn";

    NeuralNetwork nn = create_network();
    if (!nn.layer1.weights || !nn.layer2.weights || !nn.layer1.biases || !nn.layer2.biases) {
        fprintf(stderr, "Failed to allocate network\n");
        free_network(&nn);
        return 1;
    }

    if (!load_network(&nn, model_path)) {
        fprintf(stderr, "Could not load model from %s. Train and save a model first.\n", model_path);
        free_network(&nn);
        return 1;
    }

    Image img = load_and_resize_image(image_path, 0);
    if (!img.pixels) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        free_network(&nn);
        return 1;
    }

    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    forward(&nn, img.pixels, hidden, output);

    int pred = output[0] > output[1] ? 0 : 1;
    printf("Prediction for %s:\n", image_path);
    printf("  Cat probability: %.4f\n", output[0]);
    printf("  Dog probability: %.4f\n", output[1]);
    printf("  Predicted class: %s\n", pred == 0 ? "Cat" : "Dog");

    free(img.pixels);
    free_network(&nn);
    return 0;
}
