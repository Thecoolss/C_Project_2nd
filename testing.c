#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

// Minimal STB image loading (you'll need stb_image.h)
// Download from: https://github.com/nothings/stb

#define IMG_SIZE 64  // Resize all images to 64x64
#define INPUT_SIZE (IMG_SIZE * IMG_SIZE * 3)  // RGB channels
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 2  // Cat or Dog
#define LEARNING_RATE 0.001
#define EPOCHS 50
#define MAX_IMAGES 1000

typedef struct {
    float *data;
    int size;
} Vector;

typedef struct {
    float **weights;
    float *biases;
    int rows;
    int cols;
} Layer;

typedef struct {
    Layer layer1;  // Input to hidden
    Layer layer2;  // Hidden to output
} NeuralNetwork;

typedef struct {
    float *pixels;  // Normalized pixel values [0, 1]
    int label;      // 0 for cat, 1 for dog
} Image;

// ============= Utility Functions =============

float randf() {
    return (float)rand() / (float)RAND_MAX;
}

Vector create_vector(int size) {
    Vector v;
    v.size = size;
    v.data = (float*)calloc(size, sizeof(float));
    return v;
}

void free_vector(Vector *v) {
    free(v->data);
}

Layer create_layer(int rows, int cols) {
    Layer layer;
    layer.rows = rows;
    layer.cols = cols;
    layer.weights = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        layer.weights[i] = (float*)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {
            layer.weights[i][j] = (randf() - 0.5f) * 0.1f;  // Xavier init
        }
    }
    layer.biases = (float*)calloc(rows, sizeof(float));
    return layer;
}

void free_layer(Layer *layer) {
    for (int i = 0; i < layer->rows; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
}

// ============= Activation Functions =============

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

void softmax(float *input, float *output, int size) {
    float max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// ============= Neural Network Functions =============

NeuralNetwork create_network() {
    NeuralNetwork nn;
    nn.layer1 = create_layer(HIDDEN_SIZE, INPUT_SIZE);
    nn.layer2 = create_layer(OUTPUT_SIZE, HIDDEN_SIZE);
    return nn;
}

void free_network(NeuralNetwork *nn) {
    free_layer(&nn->layer1);
    free_layer(&nn->layer2);
}

void forward(NeuralNetwork *nn, float *input, float *hidden, float *output) {
    // Layer 1: Input -> Hidden (with ReLU)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = nn->layer1.biases[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += nn->layer1.weights[i][j] * input[j];
        }
        hidden[i] = relu(sum);
    }
    
    // Layer 2: Hidden -> Output (with Softmax)
    float raw_output[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = nn->layer2.biases[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += nn->layer2.weights[i][j] * hidden[j];
        }
        raw_output[i] = sum;
    }
    softmax(raw_output, output, OUTPUT_SIZE);
}

void backward(NeuralNetwork *nn, float *input, float *hidden, float *output, 
              int true_label, float learning_rate) {
    // Output layer gradients
    float output_grad[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_grad[i] = output[i] - (i == true_label ? 1.0f : 0.0f);
    }
    
    // Hidden layer gradients
    float hidden_grad[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_grad[i] += output_grad[j] * nn->layer2.weights[j][i];
        }
        hidden_grad[i] *= relu_derivative(hidden[i]);
    }
    
    // Update layer 2 weights and biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->layer2.weights[i][j] -= learning_rate * output_grad[i] * hidden[j];
        }
        nn->layer2.biases[i] -= learning_rate * output_grad[i];
    }
    
    // Update layer 1 weights and biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->layer1.weights[i][j] -= learning_rate * hidden_grad[i] * input[j];
        }
        nn->layer1.biases[i] -= learning_rate * hidden_grad[i];
    }
}

// ============= Image Loading =============

Image load_and_resize_image(const char *filename, int label) {
    Image img;
    img.pixels = (float*)malloc(INPUT_SIZE * sizeof(float));
    img.label = label;
    
    int width, height, channels;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);
    if (!data) {
        printf("Cannot load %s: %s\n", filename, stbi_failure_reason());
        memset(img.pixels, 0, INPUT_SIZE * sizeof(float));
        return img;
    }
    
    unsigned char *resized = (unsigned char*)malloc(IMG_SIZE * IMG_SIZE * 3);
    if (!resized) {
        printf("Out of memory while resizing %s\n", filename);
        stbi_image_free(data);
        memset(img.pixels, 0, INPUT_SIZE * sizeof(float));
        return img;
    }
    
    if (!stbir_resize_uint8_linear(data, width, height, width * 3,
                                   resized, IMG_SIZE, IMG_SIZE, IMG_SIZE * 3,
                                   STBIR_RGB)) {
        printf("Resize failed for %s\n", filename);
        stbi_image_free(data);
        free(resized);
        memset(img.pixels, 0, INPUT_SIZE * sizeof(float));
        return img;
    }
    
    stbi_image_free(data);
    
    for (int i = 0; i < INPUT_SIZE; i++) {
        img.pixels[i] = (float)resized[i] / 255.0f;  // Normalize to [0, 1]
    }
    
    free(resized);
    return img;
}

int load_images_from_folder(const char *folder_path, int label, Image *images, int max_count) {
    DIR *dir = opendir(folder_path);
    if (!dir) {
        printf("Cannot open directory: %s\n", folder_path);
        return 0;
    }
    
    struct dirent *entry;
    int count = 0;
    
    while ((entry = readdir(dir)) != NULL && count < max_count) {
        if (entry->d_name[0] == '.') continue;
        
        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/%s", folder_path, entry->d_name);
        
        // Check for common image extensions
        char *ext = strrchr(entry->d_name, '.');
        if (ext && (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0 || 
                    strcmp(ext, ".png") == 0 || strcmp(ext, ".ppm") == 0)) {
            printf("Loading: %s (label=%d)\n", filepath, label);
            images[count] = load_and_resize_image(filepath, label);
            count++;
        }
    }
    
    closedir(dir);
    return count;
}

// ============= Training =============

void shuffle_images(Image *images, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Image temp = images[i];
        images[i] = images[j];
        images[j] = temp;
    }
}

void train(NeuralNetwork *nn, Image *images, int count, int epochs) {
    float *hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_images(images, count);
        
        float total_loss = 0.0f;
        int correct = 0;
        
        for (int i = 0; i < count; i++) {
            forward(nn, images[i].pixels, hidden, output);
            
            // Calculate loss (cross-entropy)
            total_loss -= logf(output[images[i].label] + 1e-10f);
            
            // Check accuracy
            int predicted = output[0] > output[1] ? 0 : 1;
            if (predicted == images[i].label) correct++;
            
            // Backpropagation
            backward(nn, images[i].pixels, hidden, output, images[i].label, LEARNING_RATE);
        }
        
        printf("Epoch %d/%d - Loss: %.4f - Accuracy: %.2f%%\n", 
               epoch + 1, epochs, total_loss / count, 100.0f * correct / count);
    }
    
    free(hidden);
    free(output);
}

// ============= Main =============

int main() {
    srand(time(NULL));
    
    printf("=== Cat vs Dog Classifier ===\n\n");
    
    // Load images
    Image *images = (Image*)malloc(MAX_IMAGES * sizeof(Image));
    int cat_count = load_images_from_folder("PetImages/Cat", 0, images, MAX_IMAGES / 2);
    int dog_count = load_images_from_folder("PetImages/Dog", 1, images + cat_count, MAX_IMAGES / 2);
    int total_count = cat_count + dog_count;
    
    printf("\nLoaded %d cat images and %d dog images\n", cat_count, dog_count);
    printf("Total images: %d\n\n", total_count);
    
    if (total_count == 0) {
        printf("No images loaded! Please check your folder paths.\n");
        printf("Expected folders: './cats' and './dogs'\n");
        free(images);
        return 1;
    }
    
    // Create and train network
    NeuralNetwork nn = create_network();
    printf("Training neural network...\n\n");
    train(&nn, images, total_count, EPOCHS);
    
    printf("\nTraining complete!\n");
    
    // Cleanup
    for (int i = 0; i < total_count; i++) {
        free(images[i].pixels);
    }
    free(images);
    free_network(&nn);
    
    return 0;
}
