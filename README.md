# Cat vs Dog Classifier

This repo holds a tiny C program that learns to tell cats from dogs using only the C standard library plus two single-header helpers for loading and resizing images. Follow the steps below even if you have never touched AI and are only comfortable with basic loops.

## What you will end up with
- A command-line app called `catdog` (or `catdog.exe` on Windows).
- It reads your cat and dog photos, trains a very small neural network, prints progress, and saves the learned weights to `catdog.nn` so you can reuse them later.
- Everything is in plain C; no Python, no big frameworks.

## Step 0: What you need
- A laptop/desktop with ~2 GB free disk space. Works on Windows, macOS, or Linux (including WSL on Windows).
- Comfort running simple terminal commands you can copy/paste.
- A few dozen cat and dog images (JPG/PNG/PPM). More images help but are not required to get it running.

## Step 1: Install the C tools (one-time)
- **Windows (simplest):**
  1) Install MSYS2 from https://www.msys2.org.
  2) Open the "MSYS2 MinGW 64-bit" terminal.
  3) Run: `pacman -Sy --needed base-devel mingw-w64-x86_64-gcc git cmake`.
- **macOS:** Install the Xcode command line tools: `xcode-select --install`. If you want CMake, install Homebrew then run `brew install cmake`.
- **Linux/WSL (Ubuntu/Debian):** `sudo apt update && sudo apt install -y build-essential git cmake`.

## Step 2: Get this project onto your machine
- With Git: `git clone <this repo URL>` then `cd C_Project_2nd`.
- Without Git: click “Download ZIP” on the repo page, unzip it, and open a terminal inside the unzipped folder.

## Step 3: Know the important files
- `src/main.c` — the program entry: loads images, trains, evaluates, saves `catdog.nn`.
- `src/catdog.c` + `include/catdog.h` — the neural network, image loading, training loop, saving/loading.
- `third_party/stb_image*.h` — tiny headers that open/resize images; already included.
- `PetImages/Cat` and `PetImages/Dog` — where your training pictures go.
- `CMakeLists.txt` — build instructions for CMake; there is also a simple GCC/Clang one-liner below.

## Step 4: Put your images in the right folders
Create this layout inside the project folder (names are case-sensitive):
```
PetImages/
  Cat/
    cat1.jpg
    cat2.png
  Dog/
    dog1.jpg
    dog2.png
```
Tips:
- Use clear photos; avoid huge files (under ~1–2 MB is fine).
- Only `.jpg`, `.jpeg`, `.png`, `.ppm` are read. Files starting with a dot are skipped.
- More images = better learning. Start with at least 20 per class if you can.

## Step 5: Build the app

**B) CMake build:**
```bash
cmake -S . -B build
cmake --build build
```
The compiled app will be at `build/catdog` (or `build\\catdog.exe` on Windows).

## Step 6: Train and watch it learn
From the project folder (or `build/` if you used CMake):
```bash
./catdog          # or .\\catdog.exe on Windows
```
You will see it load images, then print lines like:
```
Epoch 1/50 - Loss: 0.69 - Accuracy: 52.00%
```
Accuracy climbing toward 100% means it is guessing better. At the end it writes `catdog.nn` so next run can start from the saved network.

## Step 7: Reuse or reset training
- To **reuse** the last trained model: just run `./catdog` again; it will load `catdog.nn` automatically and continue training.
- To **start over**: delete `catdog.nn` and rerun.

## Step 8: Optional: run the tiny tests
Only if you built with CMake:
```bash
ctest --test-dir build
```
You should see “All catdog unit tests passed.”

## Step 9: Tweak how it learns (all in `include/catdog.h`)
- `IMG_SIZE` (default 64): resize target for every image. Larger = more detail but slower.
- `HIDDEN_SIZE` (default 128): number of neurons in the middle layer.
- `EPOCHS` (default 50): how many passes over all images.
- `LEARNING_RATE` (default 0.001f): how big each adjustment step is.
- `MAX_IMAGES` (default 1000): cap on how many images to load in total.
- `DATA_SPLIT_RATIO` (default 0.8f): % of images used for training vs validation.
After changing these, rebuild and run again.

## If you want the 30-second mental model of the code
1) Each picture is resized to `IMG_SIZE x IMG_SIZE`, flattened into numbers 0–1.
2) Layer 1 mixes those numbers into `HIDDEN_SIZE` outputs, then ReLU turns negatives into zero.
3) Layer 2 turns those into two scores (cat vs dog); softmax makes them add to 1.0.
4) `backward()` compares the guess to the real label and nudges weights a tiny bit.
5) Repeat for all images, many times (epochs), shuffling each round.

## Common fixes
- Compiler says `dirent.h` missing on Windows: make sure you are using the MSYS2 MinGW shell, not plain PowerShell/CMD.
- Link errors about math functions on Linux/macOS/WSL: keep `-lm` in the compile command.
- Nothing loads: double-check the folder names `PetImages/Cat` and `PetImages/Dog` and file extensions.
- Accuracy stuck low: add more images, train longer (`EPOCHS`), or increase `HIDDEN_SIZE` a bit.
