#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "catdog.h"

int main(void) {
    srand(SPLIT_SEED);

    printf("=== Cat vs Dog Classifier ===\n\n");

    Image *images = (Image*)malloc(MAX_IMAGES * sizeof(Image));
    if (!images) {
        fprintf(stderr, "Failed to allocate image buffer\n");
        return 1;
    }

    int cat_count = load_images_from_folder("PetImages/Cat", 0, images, MAX_IMAGES / 2);
    int dog_count = load_images_from_folder("PetImages/Dog", 1, images + cat_count, MAX_IMAGES / 2);
    int total_count = cat_count + dog_count;

    printf("\nLoaded %d cat images and %d dog images\n", cat_count, dog_count);
    printf("Total images: %d\n\n", total_count);

    if (total_count == 0) {
        printf("No images loaded! Please check your folder paths.\n");
        printf("Expected folders: './PetImages/Cat' and './PetImages/Dog'\n");
        free(images);
        return 1;
    }

    Image *train_set = NULL;
    Image *val_set = NULL;
    int train_count = 0;
    int val_count = 0;
    split_dataset(images, total_count, DATA_SPLIT_RATIO,
                  &train_set, &train_count, &val_set, &val_count);

    printf("Training samples: %d (%.2f%%)\n", train_count,
           total_count ? (100.0f * train_count / total_count) : 0.0f);
    printf("Validation samples: %d (%.2f%%)\n\n", val_count,
           total_count ? (100.0f * val_count / total_count) : 0.0f);

    NeuralNetwork nn = create_network();
    if (load_network(&nn, "catdog.nn")) {
        float loss = 0.0f;
        float acc = 0.0f;
        evaluate_network(&nn, val_set, val_count, &loss, &acc);
        printf("Loaded saved network — validation before training: Loss=%.4f Acc=%.2f%%\n", loss, acc);
    } else {
        printf("No saved network found — training from scratch.\n");
    }

    printf("Training neural network...\n\n");
    train(&nn, train_set, train_count, val_set, val_count, EPOCHS);

    printf("\nTraining complete!\n");

    if (val_count > 0) {
        float val_loss = 0.0f;
        float val_acc = 0.0f;
        evaluate_network(&nn, val_set, val_count, &val_loss, &val_acc);
        printf("Validation after training: Loss=%.4f Acc=%.2f%%\n", val_loss, val_acc);
    }

    if (save_network(&nn, "catdog.nn"))
        printf("Saved network to catdog.nn\n");
    else
        printf("Failed to save network!\n");

    for (int i = 0; i < total_count; i++) {
        free(images[i].pixels);
    }
    free(images);
    free_network(&nn);

    return 0;
}

