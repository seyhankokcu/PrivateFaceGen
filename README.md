
# 🚀 Privacy-FaceGAN

> Local AI-powered GAN for generating **visually similar** synthetic faces — optimized for mobile or edge devices. Ensures data privacy by processing entirely on-device, never exposing original user photos to remote servers.

---

## 📸 Why This Project?

Big tech companies often rely on cloud-based pipelines that require uploading user images for AI training or transformation. **Privacy-FaceGAN** flips that approach:

- ✅ Runs locally on devices like **phones or Raspberry Pi**  
- ✅ Protects user privacy  
- ✅ Sends only **synthetic** images to remote servers if needed  
- ✅ Useful for **content personalization**, **AI data augmentation**, and **privacy-preserving personalization**

---

## 🧠 Core Features

- ⚡ Mixed precision and XLA acceleration for fast training  
- 🧬 Custom-trained GAN for realistic face generation  
- 🎨 Input-to-similar face synthesis using latent vector optimization  
- 🛡️ Entire process runs **on-device** — zero cloud dependency  

---

## 🗂️ Project Structure

```
privacy-facegan/
│
├── generator.py             # Generator model
├── discriminator.py         # Discriminator model
├── train.py                 # GAN training loop
├── dataset.py               # Data loading & preprocessing
├── generate_similar.py      # Image similarity generation script
├── utils.py                 # Plotting, helpers, etc.
│
├── generated_images/        # Generated sample outputs
├── Celebrity_Faces_Dataset/ # Training image dataset
│
├── requirements.txt         # Python dependencies
├── .gitignore               # Files to ignore in Git
├── LICENSE                  # MIT License
└── README.md                # This file
```

---

## 🚀 Getting Started

### 🔧 Installation

> Recommended: [Google Colab](https://colab.research.google.com/) or Python 3.10+ with TensorFlow >= 2.15

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 📦 Dataset

Download or place your face image dataset inside:

```
Celebrity_Faces_Dataset/
```

> Images will be automatically resized and normalized to 128×128 during loading.

---

### 🏋️ Train the GAN

```bash
python train.py
```

- Saves generator and discriminator as `.keras` files  
- Training runs for 50 epochs (can be adjusted)

---

### 🎨 Generate a Similar Face

Place a reference image in the root (e.g., `uzo.jpg`) and run:

```bash
python generate_similar.py
```

- Optimizes a latent vector to resemble the input  
- Outputs side-by-side visualization of real vs. synthetic face

---

## 🧠 Technologies Used

- **TensorFlow / Keras**  
- **Mixed Precision Training**  
- **TTUR (Two Time-Scale Update Rule)**  
- **Latent Vector Inversion**  
- **Matplotlib**

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
If you use this repo in your project, please consider crediting or linking back.

---

## ⚠️ Disclaimer

This project is for **ethical and educational purposes only**.  
Do **not** use generated faces for impersonation, fraud, or deceptive applications.  
Our goal is to **enhance privacy**, not violate it.
