#ifndef CATDOG_H
#define CATDOG_H

#ifdef __cplusplus
extern "C" {
#endif

#define IMG_SIZE 64
#define INPUT_SIZE (IMG_SIZE * IMG_SIZE * 3)
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 2
#define LEARNING_RATE 0.001f
#define EPOCHS 50
#define MAX_IMAGES 1000
#define DATA_SPLIT_RATIO 0.8f

typedef struct {
    float **weights;
    float *biases;
    int rows;
    int cols;
} Layer;

typedef struct {
    Layer layer1;
    Layer layer2;
} NeuralNetwork;

typedef struct {
    float *pixels;
    int label;
} Image;

float randf(void);

Layer create_layer(int rows, int cols);
void free_layer(Layer *layer);

NeuralNetwork create_network(void);
void free_network(NeuralNetwork *nn);
void forward(NeuralNetwork *nn, float *input, float *hidden, float *output);
void backward(NeuralNetwork *nn, float *input, float *hidden, float *output,
              int true_label, float learning_rate);

Image load_and_resize_image(const char *filename, int label);
int load_images_from_folder(const char *folder_path, int label, Image *images, int max_count);
void shuffle_images(Image *images, int count);
void split_dataset(Image *images, int total_count, float train_ratio,
                   Image **train_set, int *train_count,
                   Image **val_set, int *val_count);
void train(NeuralNetwork *nn, Image *images, int count, int epochs);

int save_network(const NeuralNetwork *nn, const char *path);
int load_network(NeuralNetwork *nn, const char *path);
void evaluate_network(NeuralNetwork *nn, Image *images, int count,
                      float *out_loss, float *out_acc);

#ifdef __cplusplus
}
#endif

#endif

