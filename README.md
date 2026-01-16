# HMPDM: Historical Motion Priors-Informed Diffusion Model for Driving Video Prediction

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)


**HMPDM** is a diffusion-based **driving video prediction** framework that forecasts how real-world driving scenes evolve in future frames.  
It leverages **historical motion priors** to improve **temporal consistency** and **visual quality**, supporting safer planning and reasoning for autonomous driving.

---

## ✨ Key Ideas

Existing video prediction models can struggle to model diverse motion patterns in real driving scenes, often leading to degraded temporal coherence and visual quality.  
HMPDM addresses this by explicitly injecting **historical motion priors** into a diffusion-based prediction pipeline.

---

## 🔧 Method Overview

HMPDM introduces three core designs:

- **TaLC (Temporal-aware Latent Conditioning)**  
  Injects implicit historical motion information into the latent space for better temporal understanding.

- **MaPE (Motion-aware Pyramid Encoder)**  
  Builds multi-scale motion representations to capture diverse motion patterns more effectively.

- **Self-Conditioning (SC)**  
  Stabilizes iterative denoising to produce more coherent and higher-quality future predictions.

---

## 📊 Results 



## 🚀 Quick Start



### Installation

