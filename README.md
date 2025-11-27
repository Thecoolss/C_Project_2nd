# Cat vs Dog Classifier (Plain C)

This project is a simple program that teaches a computer to guess whether a picture is a cat or a dog. It is written in plain C with no extra AI tools. The code shows every step in one file so you can see how it works.

## Repository Layout
- `testing.c` — the whole program: it loads pictures, shrinks them, learns from them, and prints progress.
- `stb_image.h`, `stb_image_resize2.h` — tiny helper libraries (already included) to open and resize images.
- `PetImages/` — folders for your pictures: put cats in `PetImages/Cat` and dogs in `PetImages/Dog`.
- `.vscode/` — editor/debug settings for VS Code (optional).

## What the Program Does
1) Reads all pictures in `PetImages/Cat` and `PetImages/Dog`.  
2) Shrinks each picture to 64×64 pixels and scales colors to a 0–1 range.  
3) Builds a very small “brain” made of two sets of numbers (layers).  
4) Trains that brain by showing every picture many times and nudging the numbers so cats look more “cat” and dogs more “dog.”  
5) Prints how well it is doing after each round of training.

## Prerequisites
- A C compiler (GCC or Clang). On Windows, MinGW/MSYS2 GCC works well.
- The math library (`-lm` on Linux/macOS).
- `dirent.h` (built in on Linux/macOS; comes with MinGW/MSYS2 on Windows).
- The STB headers are already here; you do not install anything else.

## Getting Pictures
Use any cat/dog photos. Keep the folder layout:
```
PetImages/
  Cat/
    0.jpg, 1.jpg, ...
  Dog/
    0.jpg, 1.jpg, ...
```
Files ending in `.jpg`, `.jpeg`, `.png`, or `.ppm` are loaded. Files starting with a dot are skipped.

## Build and Run
From the repo root:
```bash
# Linux/macOS/WSL
gcc testing.c -o catdog -std=c11 -lm

# Windows (MinGW/MSYS2)
gcc testing.c -o catdog.exe -std=c11

# Run
./catdog        # or .\\catdog.exe on Windows
```
The app will list images it loads, then print lines like:
```
Epoch 1/50 - Loss: 0.69 - Accuracy: 52.00%
```
Higher accuracy means it is guessing better.

## Plain-English Walkthrough of `testing.c`

### Setup and knobs
- Picks a target size: 64×64 pixels with 3 color channels (red, green, blue).
- Sets a small middle layer size (128 numbers) and two outputs (cat or dog).
- Chooses how fast to learn (`LEARNING_RATE 0.001`), how many passes to train (`EPOCHS 50`), and how many images to use (`MAX_IMAGES 1000`).

### How pictures become numbers
- `load_and_resize_image()` opens a file, forces it to 64×64, and divides every color value by 255 so all values are between 0 and 1. That gives one big array of numbers that describes the picture.
- `load_images_from_folder()` walks a folder, grabs allowed image files, and calls the resize function for each.

### The tiny “brain”
- Think of it as two layers of numbers:
  - Layer 1: Connects the 64×64×3 input numbers to 128 middle numbers. Each connection starts as a small random value. After adding them up, any negative result is set to zero (a simple “on/off” rule).
  - Layer 2: Connects those 128 middle numbers to 2 outputs (cat score, dog score). The two scores are then turned into percentages that add to 100%.
- `create_network()` builds these layers with random starting values.
- `forward()` runs one picture through both layers to get the two percentages.

### Learning loop
- `backward()` is the “learning” step. It looks at what the model guessed vs. the right answer and nudges the layer numbers a tiny bit so next time it is more likely to guess right. This happens for every picture.
- `shuffle_images()` randomizes the order each pass so the model does not see cats then dogs in the same order every time.
- `train()` repeats: shuffle, run forward on each picture, measure how wrong it was, adjust numbers, and print progress after the pass.

### main() ties it together
1) Seeds random numbers so the starting values differ each run.  
2) Loads cats and dogs from the two folders.  
3) Builds the tiny brain.  
4) Trains for the chosen number of passes.  
5) Cleans up memory and exits.

## Tweaks You Can Try
- Increase `EPOCHS` to train longer (slower but may improve results).
- Raise `HIDDEN_SIZE` to give the brain more middle numbers (uses more memory, may learn a bit better).
- Change `LEARNING_RATE` to make learning steps bigger or smaller (too big can make it unstable; too small can make it slow).
- If you resize to a different size, update `IMG_SIZE` and recalculate `INPUT_SIZE` as `size * size * 3`.

## Troubleshooting
- `dirent.h` missing on Windows: use MinGW/MSYS2 or WSL instead of plain MSVC.
- Math functions not found when linking: add `-lm` on Linux/macOS/WSL.
- If it runs but guesses poorly: this is a very small, simple model on raw pixels. Add more images, train longer, or increase `HIDDEN_SIZE`, but expect only basic accuracy.

## Where to Go Next
- Save the learned numbers to a file so you do not retrain every run.
- Split your images into “train” and “test” folders to check how well it generalizes.
- Use mini-batches (adjust multiple images at once) for faster training.
- Move to a small convolutional model (beyond this tutorial) for better picture results.

This README aims for plain language: load pictures, turn them into numbers, run them through two layers, nudge the layers each pass, and watch the accuracy climb. You can read `testing.c` alongside this to see every line that does each step. Happy tinkering!
