# CIFAR-10 CNN Webcam Classifier

**A CNN trained from scratch on CIFAR-10, fine-tuned with a separate optimizer per layer, and run live on a webcam. 81.94% test accuracy on the 10,000-image CIFAR-10 test set.**

https://github.com/user-attachments/assets/975c6f2a-dc7b-408e-9c2e-06919c781299



[![Watch the 80-second webcam demo](assets/demo_thumbnail.jpg)](assets/cifar10_webcam_demo.mp4)

*Click the image to watch the 80-second demo (no audio): phone-screen photos held up to a laptop webcam, recorded before the "unknown" option below was added.*

## Problem

Classify an image into one of 10 everyday categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), then take the model out of the notebook and run it on live camera frames.

The goal was to build and understand every piece myself: a CNN without a pretrained backbone, a two-stage training process, and a real-time inference loop.

## Dataset

| | |
|---|---|
| Dataset | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009) |
| Images | 32×32 RGB, upscaled to 150×150 when loaded |
| Classes | 10, balanced |
| Split used | 40,000 train · 10,000 validation (both from the official training set) · 10,000 test (official test set, 1,000 per class) |
| License | No explicit license is published. Please cite: A. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 |

The images are not included in this repo. `src/prepare_data.py` downloads CIFAR-10 and writes the same folder layout and split sizes. The validation images it picks (seeded, 1,000 per class) may not be the exact ones I used.

## Model

**Stage 1: base CNN, trained from scratch** (`src/train.py`, 20 epochs, RMSprop lr 1e-3)

```
Input 150×150×3
→ Augmentation: RandomFlip(horizontal), RandomRotation(0.03), RandomZoom(0.1), RandomTranslation(0.1, 0.1)
→ Rescaling(1/255)
→ Conv(12) → MaxPool → Conv(34) → MaxPool → Conv(64) → MaxPool → Conv(64) → MaxPool → Conv(128)
→ GlobalAveragePooling → Dense(256, relu) → Dropout(0.2) → Dense(10, softmax)
```

**Stage 2: fine-tuning** (`src/finetune.py`, 10 epochs)

- Keep the augmentation, rescaling and all conv layers; replace the head with `GlobalAveragePooling → Dense(128, relu) → Dense(10, relu) → Dense(10, softmax)`.
- Freeze conv layers 1 and 3. Train conv layers 2, 4 and 5 plus the new head.
- Give each trainable group its own optimizer with Keras `MultiOptimizer`:

| Layer | Optimizer | Learning rate |
|---|---|---|
| Conv 2 (34 filters) | Adam | 0.0012 |
| Conv 4 (64 filters) | Adam | 0.001 |
| Conv 5 (128 filters) | RMSprop | 0.001 |
| New dense head | RMSprop | 0.0009 |

Final model: 152,386 parameters (132,402 trainable during fine-tuning), 2.1 MB on disk.

Augmentation and rescaling live inside the model, so inference only needs raw RGB pixels resized to 150×150.

Only the fine-tuned model is included. `src/train.py` rebuilds a base model if you want one.

## Results

Fine-tuned model (`models/final_model.keras`):

| Split | Accuracy | Loss |
|---|---|---|
| Training (last epoch) | 85.40% | 0.4276 |
| Validation (last epoch) | 82.47% | 0.5433 |
| **Test (10,000 images)** | **81.94%** | **0.5749** |

The test number is the honest one: those images were never used for training or for choosing settings. The 2.9-point gap between training and validation accuracy shows mild overfitting.

**In the live demo** the model gets clear photos right (dog 99.7%, truck, automobile, ship, cat), but it also gets things confidently wrong: a face comes out as "cat", an airplane photo as "frog", a supercar as "ship". Two reasons:

1. The model had no "none of these" option. Softmax always spreads 100% across the 10 classes, so a face has to be *something*.
2. Training images are 32×32 thumbnails. Webcam frames of a phone screen, with glare, blur and a hand in the shot, look very different.

### Added after the demo: an "unknown" answer

After recording the demo I added a confidence threshold to `webcam_demo.py`. If the top class is below 70% confidence, the model now answers `unknown (best guess cat 45%)` instead of guessing, shown in orange on the webcam.

What it fixes and what it doesn't:

- It catches the shaky guesses, like the airplane photo labelled "frog" at 60.2%.
- It does **not** catch confident mistakes. My face was labelled "cat" at 97.4%, well above any sensible threshold. A model can be very sure and still wrong; fixing that needs a real "other" class or out-of-distribution detection (see future work).

The 0.70 value is a starting point, not yet tuned on the validation set.

## Installation

Tested with Python 3.13.

```bash
git clone https://github.com/manibasiir-ux/CIFAR10_CNN_Webcam_Classifier.git
cd CIFAR10_CNN_Webcam_Classifier
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Run the trained model (no training needed)**

```bash
# Live webcam, press Q to quit. Hold a photo (or your phone screen) up to the camera.
python src/webcam_demo.py
```

To change the "unknown" cutoff, edit `unknown_threshold` at the top of `webcam_demo.py` (0 turns it off).

**Train from scratch**

```bash
python src/prepare_data.py     # download CIFAR-10 into data/cifar10/{train,validation,test}
python src/train.py            # stage 1 → models/my_cnn_First_model.keras
python src/finetune.py         # stage 2 → models/final_model.keras
```

Both training scripts end by evaluating on the 10,000-image test set and printing the test loss and accuracy.

Run the scripts from the repo folder: they read the data from `data/cifar10/` and the model from `models/`. If your data is somewhere else, change `Train_dir`, `Validation_dir` and `Test_dir` at the top of the script. On a CPU, expect training to take hours.

## Project structure

```
CIFAR10_CNN_Webcam_Classifier/
├── src/
│   ├── prepare_data.py    # CIFAR-10 → train/validation/test folders
│   ├── train.py           # stage 1: base CNN
│   ├── finetune.py        # stage 2: fine-tuning with per-layer optimizers
│   └── webcam_demo.py     # real-time webcam classification (with "unknown")
├── models/
│   ├── final_model.keras  # fine-tuned model (2.1 MB)
│   └── class_names.json
├── assets/                # demo video and its thumbnail
├── requirements.txt
└── LICENSE
```

## Limitations and future work

- [x] Add an "unknown" option via a confidence threshold
- [ ] Add a confusion matrix and per-class precision/recall (cat vs dog is the usual suspect)
- [ ] Tune the threshold on the validation set
- [ ] Catch confident mistakes too: train an "other" class on non-CIFAR images, or add out-of-distribution detection
- [ ] Compare against a pretrained backbone (e.g. MobileNetV3) to measure what training from scratch costs
- [ ] Train on higher-resolution images closer to what a webcam sees
- [ ] Export to TensorFlow Lite for an edge device

## Author

**Mani Basir**, ML/AI Engineer

[LinkedIn](https://www.linkedin.com/in/mani-basir) · [GitHub](https://github.com/manibasiir-ux)

## License

Code: [Apache License 2.0](LICENSE). The CIFAR-10 dataset is not included and is not covered by this license.
