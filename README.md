
# VAE Latent Space Explorer

This project uses a **Variational Autoencoder (VAE)** to train and generate images, combined with a **tkinter GUI** that allows real-time adjustment of the latent vector (z-vector) to see how it affects the generated images.

---

## Project Overview

- Custom **VariationalAutoencoder** model (based on convolutional layers).
- Train or load a pre-trained VAE model from disk.
- **tkinter** GUI for interactive latent vector manipulation and real-time image generation.
- Great for understanding **latent space** behavior and VAE-based image generation.

---

## Project Structure

```
your_project/
├── youmu2/                 # Dataset folder (subfolders for each class, containing .png images)
├── models/
│   └── VAE.py              # VariationalAutoencoder model definition
├── run/
│   └── vae_0002_faces/     # Training outputs (viz, images, weights)
├── images/
│   └── result.png          # Initial displayed image
├── main.py                 # Main application (the code you shared)
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

---

## Installation

1. **Install required packages**

```bash
pip install -r requirements.txt
```

Main dependencies include:
- TensorFlow 1.14
- Keras
- matplotlib
- tkinter (usually built-in with Python)
- Pillow
- numpy
- jupyter (for development)
- and others

2. **Prepare dataset**

Place your training images inside the `youmu2/` folder, organized in subfolders (one per class) — as required by `flow_from_directory`.

---

## How to Use

### 1. Train the VAE (optional)

If you want to retrain the model, set `mode = 'build'` in the script, then run:

```bash
python main.py
```

The trained model will be saved under `run/vae_0002_faces/weights/`.

### 2. Launch the Latent Space Explorer

Simply run:

```bash
python main.py
```

- Each slider corresponds to one latent dimension (25 total).
- Adjust the sliders to modify the latent vector and see the newly generated image.
- Clicking on the "Latent Vector" label resets all sliders to zero.
- Generated images are temporarily saved as `result2.png`.

---

## Model Details

- **Input Size**: 128x128x3
- **Encoder Layers**: Conv2D (32, 64, 64, 64 filters)
- **Decoder Layers**: Conv2DTranspose (64, 64, 32, 3 filters)
- **Latent Space Size (z_dim)**: 25
- **Batch Normalization and Dropout** enabled
- **Reconstruction Loss Multiplier (R_LOSS_FACTOR)**: 10000
- **Learning Rate**: 0.0004

---

## Special Features

- GPU memory usage limited to 1GB to avoid overflows.
- Uses matplotlib to generate and save new images which are displayed via tkinter.
- Easy switching between training and loading pre-trained weights.
- Minimalistic and intuitive UI — perfect for beginners exploring VAEs.

---

## Future Improvements
- Support batch image generation (currently one image at a time).
- Save/load custom latent vectors.
- Allow dynamic z_dim customization.
- Add animation to visualize smooth latent vector transitions.

---

## Credits
- **Keras**: High-level deep learning API.
- **TensorFlow**: Robust backend for model operations.
- **Matplotlib**: Image saving and plotting.
- **tkinter**: Lightweight GUI building for Python.

---

# Contributions and suggestions are welcome! Happy Exploring!

---
