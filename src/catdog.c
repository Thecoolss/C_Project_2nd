#include "catdog.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"
// keep this if we ned to deeply inspect the documentation of stb_image and stb_image_resize2: https://github.com/nothings/stb

//NOTE: we still need to review the erro handling more to make sure it is robust enough for production use
// ============= Utility Functions =============

float randf() {
    return (float)rand() / (float)RAND_MAX;
}

//Removed Vector type and related functions as they are not used in the current implementation

Layer create_layer(int rows, int cols) {
    Layer layer;
    layer.rows = rows;
    layer.cols = cols;
    layer.weights = (float**)malloc(rows * sizeof(float*));
    if (!layer.weights) {
        layer.rows = 0;
        layer.cols = 0;
        layer.biases = NULL;
        return layer;
    }
    for (int i = 0; i < rows; i++) {
        layer.weights[i] = (float*)malloc(cols * sizeof(float));
        if (!layer.weights[i]) {
            for (int k = 0; k < i; k++) free(layer.weights[k]);
            free(layer.weights);
            layer.weights = NULL;
            layer.rows = 0;
            layer.cols = 0;
            layer.biases = NULL;
            return layer;
        }
        for (int j = 0; j < cols; j++) {
            layer.weights[i][j] = (randf() - 0.5f) * 0.1f;  // Xavier init 
        } //Who is Xavier crying_emoji
        //some sh*t about initialization but idk why gpt commented that when it was reading through the code
    }
    layer.biases = (float*)calloc(rows, sizeof(float));
    if (!layer.biases) {
        for (int i = 0; i < rows; i++) free(layer.weights[i]);
        free(layer.weights);
        layer.weights = NULL;
        layer.rows = 0;
        layer.cols = 0;
    }
    return layer;
}

void free_layer(Layer *layer) {
    if (!layer) return;
    if (layer->weights) {
        for (int i = 0; i < layer->rows; i++) {
            free(layer->weights[i]);
        }
        free(layer->weights);
        layer->weights = NULL;
    }
    if (layer->biases) {
        free(layer->biases);
        layer->biases = NULL;
    }
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
    if (!nn) return;
    free_layer(&nn->layer1);
    free_layer(&nn->layer2);
}

void forward(NeuralNetwork *nn, float *input, float *hidden, float *output) {
    if (!nn || !input || !hidden || !output) return;
    if (!nn->layer1.weights || !nn->layer2.weights || !nn->layer1.biases || !nn->layer2.biases) return;
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
    if (!nn || !input || !hidden || !output) return;
    if (!nn->layer1.weights || !nn->layer2.weights || !nn->layer1.biases || !nn->layer2.biases) return;
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
    if (!img.pixels) {
        img.label = label;
        return img;
    }
    img.label = label;
    
    int width, height, channels;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);
    if (!data) {
        printf("Cannot load %s: %s\n", filename, stbi_failure_reason());
        memset(img.pixels, 0, INPUT_SIZE * sizeof(float));
        return img; // Returns a black image on failure, maybe it is better to handle differently
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
    if (!folder_path || !images || max_count <= 0) return 0;
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
    if (!images || count <= 0) return;
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Image temp = images[i];
        images[i] = images[j];
        images[j] = temp;
    }
}

void split_dataset(Image *images, int total_count, float train_ratio,
                   Image **train_set, int *train_count,
                   Image **val_set, int *val_count) {
    if (!train_set || !train_count || !val_set || !val_count) return;
    if (!images || total_count <= 0) {
        *train_set = *val_set = NULL;
        *train_count = *val_count = 0;
        return;
    }
    if (total_count <= 0) {
        *train_set = *val_set = NULL;
        *train_count = *val_count = 0;
        return;
    }

    if (train_ratio < 0.0f) train_ratio = 0.0f;
    if (train_ratio > 1.0f) train_ratio = 1.0f;

    shuffle_images(images, total_count);

    int train_samples = (int)(train_ratio * total_count);
    if (total_count > 1) {
        if (train_samples < 1) train_samples = 1;
        if (train_samples > total_count - 1) train_samples = total_count - 1;
    }

    *train_set = images;
    *train_count = train_samples;
    *val_set = images + train_samples;
    *val_count = total_count - train_samples;
}

void train(NeuralNetwork *nn, Image *images, int count, int epochs) {
    float *hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    if (!nn || !images || count <= 0 || epochs <= 0 || !hidden || !output) {
        if (hidden) free(hidden);
        if (output) free(output);
        return;
    }
    
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

int save_network(const NeuralNetwork *nn, const char *path) {
    if (!nn || !path) return 0;
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    if (!nn->layer1.weights || !nn->layer2.weights || !nn->layer1.biases || !nn->layer2.biases) {
        fclose(f);
        return 0;
    }

    fwrite(&nn->layer1.rows, sizeof(int), 1, f);
    fwrite(&nn->layer1.cols, sizeof(int), 1, f);
    for (int i = 0; i < nn->layer1.rows; ++i)
        fwrite(nn->layer1.weights[i], sizeof(float), nn->layer1.cols, f);
    fwrite(nn->layer1.biases, sizeof(float), nn->layer1.rows, f);

    fwrite(&nn->layer2.rows, sizeof(int), 1, f);
    fwrite(&nn->layer2.cols, sizeof(int), 1, f);
    for (int i = 0; i < nn->layer2.rows; ++i)
        fwrite(nn->layer2.weights[i], sizeof(float), nn->layer2.cols, f);
    fwrite(nn->layer2.biases, sizeof(float), nn->layer2.rows, f);

    fclose(f);
    return 1;
}

// Load a saved network file (catdog.nn)
int load_network(NeuralNetwork *nn, const char *path) {
    if (!nn || !path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    int r1, c1, r2, c2;
    if (fread(&r1, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&c1, sizeof(int), 1, f) != 1) { fclose(f); return 0; }

    // Free any existing layers to avoid leaks
    free_layer(&nn->layer1);

    // Create layer with same dims
    nn->layer1 = create_layer(r1, c1);
    if (!nn->layer1.weights || !nn->layer1.biases) { fclose(f); return 0; }
    for (int i = 0; i < r1; ++i) {
        if (fread(nn->layer1.weights[i], sizeof(float), c1, f) != (size_t)c1) { fclose(f); return 0; }
    }
    if (fread(nn->layer1.biases, sizeof(float), r1, f) != (size_t)r1) { fclose(f); return 0; }

    if (fread(&r2, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&c2, sizeof(int), 1, f) != 1) { fclose(f); return 0; }

    free_layer(&nn->layer2);
    nn->layer2 = create_layer(r2, c2);
    if (!nn->layer2.weights || !nn->layer2.biases) { fclose(f); return 0; }
    for (int i = 0; i < r2; ++i) {
        if (fread(nn->layer2.weights[i], sizeof(float), c2, f) != (size_t)c2) { fclose(f); return 0; }
    }
    if (fread(nn->layer2.biases, sizeof(float), r2, f) != (size_t)r2) { fclose(f); return 0; }

    fclose(f);
    return 1;
}

// Evaluate network on an image set (no weight updates). Returns loss and accuracy via pointers.
void evaluate_network(NeuralNetwork *nn, Image *images, int count, float *out_loss, float *out_acc) {
    if (!nn || !images || count <= 0) { if (out_loss) *out_loss = 0.0f; if (out_acc) *out_acc = 0.0f; return; }
    if (count <= 0) { if (out_loss) *out_loss = 0.0f; if (out_acc) *out_acc = 0.0f; return; }
    float *hidden = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    if (!hidden || !output) {
        if (out_loss) *out_loss = 0.0f;
        if (out_acc) *out_acc = 0.0f;
        if (hidden) free(hidden);
        if (output) free(output);
        return;
    }
    float total_loss = 0.0f;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        forward(nn, images[i].pixels, hidden, output);
        total_loss -= logf(output[images[i].label] + 1e-10f);
        int pred = output[0] > output[1] ? 0 : 1;
        if (pred == images[i].label) ++correct;
    }
    if (out_loss) *out_loss = total_loss / count;
    if (out_acc) *out_acc = 100.0f * ((float)correct / count);
    free(hidden);
    free(output);
}
