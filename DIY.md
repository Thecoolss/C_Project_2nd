# Build It Yourself: Cat vs Dog Classifier in Plain C

This guide is for someone who wants to **write their own version** of this project instead of copying files. It walks through what to create, what the code should do, and how to compile and run it. No prior AI experience needed; only basic C and loops.

## 1) Plan the project
- Language: C (C11).
- Dependencies: only the standard library plus two single-header helpers for images: `stb_image.h` (load) and `stb_image_resize2.h` (resize). Both are MIT/public-domain and can be dropped into your repo.
- Target: a tiny 2-layer neural network that classifies 64×64 RGB images as cat (label 0) or dog (label 1).

## 2) Create the folder structure
```
your_project/
  include/        # headers you write
  src/            # .c files you write
  third_party/    # third-party headers you drop in
  PetImages/
    Cat/
    Dog/
```
Put `stb_image.h` and `stb_image_resize2.h` into `third_party/`. Add your cat/dog photos under `PetImages/Cat` and `PetImages/Dog`.

## 3) Write the public header: `include/catdog.h`
Define the constants and data structures the rest of the code will use:
- Image sizes and hyperparameters: `IMG_SIZE`, `INPUT_SIZE`, `HIDDEN_SIZE`, `OUTPUT_SIZE`, `LEARNING_RATE`, `EPOCHS`, `MAX_IMAGES`, `DATA_SPLIT_RATIO`.
- Types:
  - `Layer` with `weights` (float**), `biases` (float*), `rows`, `cols`.
  - `NeuralNetwork` holding two `Layer`s.
  - `Image` with `pixels` (float*) and `label` (int).
- Function prototypes:
  - Random helper `randf()`.
  - Layer helpers: `create_layer`, `free_layer`.
  - Network helpers: `create_network`, `free_network`, `forward`, `backward`.
  - Data helpers: `load_and_resize_image`, `load_images_from_folder`, `shuffle_images`, `split_dataset`.
  - Training/eval: `train`, `evaluate_network`.
  - Save/load: `save_network`, `load_network`.

## 4) Implement the core: `src/catdog.c`
Key pieces to write:
1. **Random float**: return a float in [0,1].
2. **Layer creation**: allocate `weights` as `rows x cols` floats with small random values (e.g., `(randf()-0.5f)*0.1f`) and zero biases.
3. **Activation functions**:
   - `relu(x)` returns `x > 0 ? x : 0`.
   - `softmax` over a small array (subtract max for stability).
4. **Forward pass**:
   - Hidden layer: for each hidden neuron, dot input with weights, add bias, apply ReLU.
   - Output layer: dot hidden with weights, add bias, softmax to get two probabilities.
5. **Backward pass (training step)**:
   - Compute `output_grad[i] = output[i] - (i == true_label ? 1 : 0)`.
   - Backprop to hidden with ReLU derivative.
   - Update weights/biases for layer2 then layer1 using `learning_rate`.
6. **Image loading**:
   - Use `stbi_load` to force 3 channels.
   - Resize to `IMG_SIZE x IMG_SIZE` with `stbir_resize_uint8_linear`.
   - Normalize each byte to float [0,1] and store in a `float` array of size `INPUT_SIZE`.
7. **Dataset helpers**:
   - `load_images_from_folder`: iterate directory entries, accept `.jpg/.jpeg/.png/.ppm`, skip dotfiles, call `load_and_resize_image`.
   - `shuffle_images`: Fisher–Yates shuffle the array.
   - `split_dataset`: randomize then split by `DATA_SPLIT_RATIO` into train and validation pointers/counts.
8. **Training loop**:
   - Allocate `hidden` and `output` buffers.
   - For each epoch: shuffle, run forward/backward per image, accumulate loss (`-log(output[label])`), track accuracy, print per-epoch stats.
9. **Save/load**:
   - Write layer dimensions, then weights row-by-row, then biases, to a binary file (e.g., `catdog.nn`).
   - Load does the reverse, recreating layers with the stored dimensions.
10. **Evaluate**:
    - Forward each image, compute average loss and accuracy without updating weights.

## 5) Write the entry point: `src/main.c`
What `main()` should do:
1. `srand(time(NULL));`
2. Allocate an `Image` array sized `MAX_IMAGES`.
3. Load cats from `PetImages/Cat` with label 0 and dogs from `PetImages/Dog` with label 1.
4. If none loaded, print a helpful message and exit.
5. Split into train/validation using `split_dataset`.
6. Create the network via `create_network`.
7. Try `load_network("catdog.nn")`; if it succeeds, evaluate on validation and report; otherwise say you’re training from scratch.
8. Call `train` with the training set and `EPOCHS`.
9. After training, evaluate on validation (if available) and print metrics.
10. Save with `save_network("catdog.nn")`.
11. Free all allocated `pixels`, then `free_network`.

## 6) Write a tiny inference helper: `src/infer.c`
Purpose: load a saved model and classify one image from the command line.
- Accept args: `<image_path> [model_path]`, defaulting the model path to `catdog.nn`.
- Create the network and call `load_network`; if it fails, print a helpful message and exit.
- Load/resize the image with `load_and_resize_image`.
- Run `forward`, print the cat/dog probabilities, and state the predicted class.
- Free the image pixels and the network before exiting.

## 7) Add the third-party headers
- Download `stb_image.h` and `stb_image_resize2.h` from the official stb repository (https://github.com/nothings/stb) and place them in `third_party/`.
- In `src/catdog.c`, before including them, define:
  ```c
  #define STB_IMAGE_IMPLEMENTATION
  #define STB_IMAGE_RESIZE_IMPLEMENTATION
  #include "stb_image.h"
  #include "stb_image_resize2.h"
  ```

## 8) Optional: add a simple `CMakeLists.txt`
Minimal content:
```cmake
cmake_minimum_required(VERSION 3.16)
project(CatDogClassifier LANGUAGES C)
set(CMAKE_C_STANDARD 11)

add_library(catdog_lib STATIC src/catdog.c)
target_include_directories(catdog_lib PUBLIC include third_party)
if(UNIX) target_link_libraries(catdog_lib PUBLIC m) endif()

add_executable(catdog src/main.c)
target_link_libraries(catdog PRIVATE catdog_lib)

add_executable(catdog_infer src/infer.c)
target_link_libraries(catdog_infer PRIVATE catdog_lib)
```

## 9) Compile the project
- **Simple GCC/Clang command (no CMake):**
  ```bash
  gcc src/main.c  src/catdog.c -Iinclude -Ithird_party -std=c11 -lm -o catdog         # Linux/macOS/WSL
  gcc src/infer.c src/catdog.c -Iinclude -Ithird_party -std=c11 -lm -o catdog_infer   # Linux/macOS/WSL
  gcc src/main.c  src/catdog.c -Iinclude -Ithird_party -std=c11       -o catdog.exe       # Windows (MinGW/MSYS2)
  gcc src/infer.c src/catdog.c -Iinclude -Ithird_party -std=c11       -o catdog_infer.exe # Windows (MinGW/MSYS2)
  ```
- **With CMake:**
  ```bash
  cmake -S . -B build
  cmake --build build
  ```
  This produces both `catdog` and `catdog_infer` (or their `.exe` variants).

## 10) Run training and inference
From the project root (or `build/` if you used CMake):
```bash
./catdog          # or .\\catdog.exe on Windows
```
Expect logs like:
```
Epoch 1/50 - Loss: 0.69 - Accuracy: 52.00%
```
Higher accuracy means it is learning. A `catdog.nn` file will appear when it saves the trained weights.

After training, classify a single image with your saved model:
```bash
./catdog_infer path/to/photo.jpg [path/to/catdog.nn]
```
Omit the second argument to use `catdog.nn` in the current directory.

## 11) Extend or tweak
- Change hyperparameters in `catdog.h` (`IMG_SIZE`, `HIDDEN_SIZE`, `EPOCHS`, `LEARNING_RATE`, `DATA_SPLIT_RATIO`).
- Add early stopping: track validation loss and break if it worsens.
- Add a predict helper: load `catdog.nn`, forward one image, print the probabilities.
- Improve data handling: filter out unreadable files, add logging, or cap per-class counts.

## 12) Quick checklist (copy/paste to follow)
- [ ] Make folders: `include`, `src`, `third_party`, `PetImages/Cat`, `PetImages/Dog`.
- [ ] Download `stb_image.h` and `stb_image_resize2.h` into `third_party/`.
- [ ] Write `include/catdog.h` with constants, structs, and function prototypes.
- [ ] Write `src/catdog.c` implementing layers, forward, backward, image loading, training, save/load.
- [ ] Write `src/main.c` to wire everything: load data, split, train, evaluate, save.
- [ ] Write `src/infer.c` to load a saved model and classify one image.
- [ ] Put cat/dog photos into the right folders.
- [ ] Compile training and inference binaries (GCC/Clang or CMake).
- [ ] Run `./catdog` to train; run `./catdog_infer <image>` to predict.
