## Annotation

Photo restoration, as a computer vision field, has always required intensive manual labor
in traditional restoration tasks, never satisfying scalability in modern applications. In
recent years, with the rapid development of AI, especially the invention of deep learning
methods such as CNNs and GANs, the quality of the whole restoration process has
gone to a new level. This project aims to devise with solution in photo restoration,
by developing a system that utilizes the advantages of AI models to enhance images
and remove visible defects, such as noise, blur, and compression artifacts. And also
presents dockerized solution with API and web-based application to demonstrate the
capabilities of the model.

Keywords: Photo Restoration, Deep Learning, Neural Networks, Image defects

## Introduction

Introduction
Photo restoration has always been a difficult task that demanded considerable efforts
from experts to restore damaged or lost image details. These days, after neural networks became commonly used with deep
learning techniques, new horizons have
opened for this area, and a great advance has been achieved. Modern deep learning-based approaches can be fully
automated, and they therefore present high-quality
results and huge savings on processing time.
Today, such approaches are already in use or are planned to be deployed in many areas where the demands on image
reconstruction accuracy and speed are high.
Medical imaging is one of the areas where AI-based image restoration methods are applied to clean medical images,
allowing for better diagnosis and treatment of patients.
For space and aerial images, superior algorithms result in sharper and more detailed
views of the Earth’s surface for mapping, environmental monitoring, and even military
applications.
Image restoration technologies in the forensic and security areas help improve facial recognition and evidence analysis,
increasing the efficiency of investigations.
Enhanced AI will be able to be used to create more powerful visual systems for autonomous vehicles, which will provide
more reliable recognition of objects and traffic conditions.
In this manner, the development of artificial intelligence-based photo restoration methods has solved not only the
problem of image restoration but also opened wide perspectives.

## Project structure

```
├── application/                                <-- Full-stack application source code folder, divided into modules
│   ├── apps/                                   
│   │   ├── defects-applying/                   <-- Python module for applying defects to images
│   │   └── web/                                <-- Next.js front-end application
│   ├── models/
│   │   └── Real-ESRGAN_x4plus-finetuned/       <-- Python module for hoisting the fine-tuned model 
│   │       ├── models/
│   │       │   ├─ net_g_latest_*.pth*          <-- Fine-tuned model weights in PyTorch format
│   │       │   └─ net_g_latest_*.bin*/.param*  <-- Fine-tuned model weights in NCNN format
│   │       ├── realesrgan-ncnn-vulkan*         <-- Linux executable for running the model
│   │       └── realesrgan-ncnn-vulkan.exe*     <-- Windows executable for running the model
│   ├── docker-compose.yml*                     <-- Docker Compose file for running the application
├── latex/                                      <-- LaTeX files for the paper with all assets and scripts for draws
├── projects/
│   ├── denoising/                              <-- Several projects for learning and practice in image restoration
│   ├── digit-classification-scratch/           <-- .
│   ├── mnist-denoising/                        <-- .
│   ├── mri-brain-denoising-nn/                 <-- .
│   ├── simple-autoencoder/                     <-- .
│   └── super-resolution-nn/                    <-- Main codes for the image restoration model
│       ├── degrade_image.py*                   <-- Python script for degrading images 
│       ├── degraded_dataset.py*                <-- Dataset class that uses degradation function
│       ├── pexels_lazy_dataset.py*             <-- Handles lazy loading of images from Pexels 
│       ├── quality_filter.py*                  <-- Python script for filtering images based on several metrics
│       ├── real-esrgan/                        <-- Real-ESRGAN Git submodule, fork of open-source repository
│       ├── requirements.txt*                   <-- Python requirements file
│       ├── v1.main.ipynb*                      <-- Jupyter notebook that was used to train the first model
│       ├── v2.main.ipynb*                      <-- Jupyter notebook that was used to train the second model
│       └── v3.main.ipynb*                      <-- Jupyter notebook that was used to fine-tune the final model

├── datasets/
│   ├── fine-tune/                              <-- The final dataset used for fine-tuning the Real-ESRGAN model
```