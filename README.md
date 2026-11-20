# DLCalciumImagingSuite (DL-CIS) 🧠⚡
<!-- Official Release Badges -->
[![Latest Release](https://img.shields.io/github/v/release/davidelattanzi/DLCalciumImagingSuite?color=blue&label=Download%20v1.0.5)](https://github.com/davidelattanzi/DLCalciumImagingSuite/releases/latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24392022.svg)](https://doi.org/10.5281/zenodo.24392022)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

### 🚀 Quick Download & Assets
Download the standalone Windows application (`.exe`), sample `.nd2` datasets, and demonstration videos:

👉 **[Download Latest Release (v1.0.5)](https://github.com/davidelattanzi/DLCalciumImagingSuite/releases/latest)**

---

An open-source, standalone GUI application for calcium imaging analysis of spontaneous transients, kinetics, and ensemble network synchrony across diverse cell preparations.

---

## 👥 Authors & Collaborators

- **Lead Developer & Software Architect:** Davide Lattanzi
- **Co-Authors & Scientific Collaborators:**
  - Stefano Sartini
  - Ludovica Montesi
  - Alice Sartini
  - Giulia Api
  - Irene Zanchini
  - Rossana Rauti

---

## ✨ What's New in v1.0.5

- **3-Channel & RGB Composite Support:** Load up to 3 independent fluorescence channels (Red, Green, and Blue) simultaneously with real-time RGB Composite overlay visualization and simultaneous trace extraction across shared ROIs.
- **Advanced Photobleaching Correction:** Expanded detrending models including Gentle Linear Detrend, 1-Exp, Bi-Exponential, and Tri-Exponential (`Fast + Mid + Slow`) decay fitting.
- **Improved UI Layout & High-DPI Scaling:** Widened the left parameter panel, enabled adaptive horizontal scrolling, and prevented long dropdown labels from clipping numeric spinbox controls on Windows displays.

---

## 🚀 Features

- **Multi-Format Video Loader:** Direct support for `.tif`, `.nd2`, `.avi`, and `.mp4` image stacks.
- **3-Channel & RGB Composite Support:** Load up to 3 independent fluorescence channels (R, G, B) simultaneously for co-localization, RGB overlay visualization, and multi-channel trace extraction across shared ROIs.
- **Versatile Analysis:** Designed for spontaneous and evoked calcium transients across primary neurons, brain slices, cell lines, and non-neuronal models.
- **Signal Pre-Processing:** Background subtraction and multi-model photobleaching correction (Linear, 1-Exp, 2-Exp, and 3-Exp decay fitting).
- **Robust Peak Detection:** Median Absolute Deviation (MAD) segmentation for accurate transient identification.
- **Kinetic Parameter Fitting:** Single-transient extraction of linear Rate of Rise, Rise time, decay constant ($\tau$), T50%, T90%, FWHM, and Area Under the Curve (AUC).
- **Network Synchrony Mapping:** Pearson correlation and Event Coincidence Rate (ECR) matrices.
- **Spatial Topology & Hubs:** Automatic identification of top HUB units and Convex Hull cluster boundary overlays.
- **Automated Reporting:** Executive PDF reports and structured multi-sheet Excel export.

---

## 📦 Installation & Usage

### Option 1: Standalone Windows Executable (Recommended)
1. Go to the **[Releases](https://github.com/davidelattanzi/DLCalciumImagingSuite/releases/latest)** page and download the `.zip` archive (`v1.0.5`).
2. Extract the archive and run `DLCalciumImagingSuiteV5.exe`.

### Option 2: Run from Source (Python)
1. Clone the repository:
```bash
git clone [https://github.com/davidelattanzi/DLCalciumImagingSuite.git](https://github.com/davidelattanzi/DLCalciumImagingSuite.git)
cd DLCalciumImagingSuite
