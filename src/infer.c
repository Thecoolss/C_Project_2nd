#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "catdog.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_path> [model_path]\n", argv[0]);
        return 1;
    }

    const char *image_path = argv[1];
    const char *model_path = (argc >= 3) ? argv[2] : "catdog.nn";

    FILE *test = fopen(model_path, "rb");
    if (!test) {
        fprintf(stderr, "ERROR: Cannot open model file '%s': %s\n", 
                model_path, strerror(errno));
        fprintf(stderr, "Current working directory: ");
        system("pwd");
        fprintf(stderr, "Looking for: %s\n", model_path);
        return 1;
    }
    fclose(test);
    fprintf(stderr, "[DEBUG] Found model file: %s\n", model_path);

    NeuralNetwork nn = create_network();
    if (!nn.layer1.weights || !nn.layer2.weights || !nn.layer1.biases || !nn.layer2.biases) {
        fprintf(stderr, "Failed to allocate network\n");
        free_network(&nn);
        return 1;
    }

    if (!load_network(&nn, model_path)) {
        fprintf(stderr, "ERROR: load_network() failed for '%s'\n", model_path);
        fprintf(stderr, "File exists but may be corrupted or have wrong format.\n");
        free_network(&nn);
        return 1;
    }
    fprintf(stderr, "[DEBUG] Successfully loaded network from %s\n", model_path);

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
