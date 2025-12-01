#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "catdog.h"

static void test_split_dataset_even_split(void) {
    Image images[4] = {0};
    Image *train_set = NULL;
    Image *val_set = NULL;
    int train_count = 0;
    int val_count = 0;

    split_dataset(images, 4, 0.5f, &train_set, &train_count, &val_set, &val_count);

    assert(train_set == images);
    assert(val_set == images + 2);
    assert(train_count == 2);
    assert(val_count == 2);
}

static void test_split_dataset_bounds(void) {
    Image images[1] = {0};
    Image *train_set = NULL;
    Image *val_set = NULL;
    int train_count = 0;
    int val_count = 0;

    split_dataset(images, 1, -1.0f, &train_set, &train_count, &val_set, &val_count);

    assert(train_set == images);
    assert(val_set == images);
    assert(train_count == 0);
    assert(val_count == 1);
}

static void test_network_shape(void) {
    NeuralNetwork nn = create_network();

    assert(nn.layer1.rows == HIDDEN_SIZE);
    assert(nn.layer1.cols == INPUT_SIZE);
    assert(nn.layer2.rows == OUTPUT_SIZE);
    assert(nn.layer2.cols == HIDDEN_SIZE);

    free_network(&nn);
}

int main(void) {
    srand(0);

    test_split_dataset_even_split();
    test_split_dataset_bounds();
    test_network_shape();

    printf("All catdog unit tests passed.\n");
    return 0;
}

