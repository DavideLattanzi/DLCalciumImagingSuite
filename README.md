# DLCalciumImagingSuite (DL-CIS) 🧠⚡

An open-source, standalone GUI application for calcium imaging analysis of spontaneous transients, kinetics, and ensemble network synchrony across diverse cell preparations.

\---

## 👥 Authors \& Collaborators

* **Lead Developer \& Software Architect:** Davide Lattanzi
* **Co-Authors \& Scientific Collaborators:**

  * Stefano Sartini
  * Ludovica Montesi
  * Alice Sartini
  * Giulia Api
  * Irene Zanchini
  * Rossana Rauti

\---

## 🚀 Features

* **Multi-Format Video Loader:** Direct support for `.tif`, `.nd2`, `.avi`, and `.mp4` image stacks.
* **Versatile Analysis:** Designed for spontaneous and evoked calcium transients across primary neurons, brain slices, cell lines, and non-neuronal models.
* **Signal Pre-Processing:** Background subtraction and photobleaching correction.
* **Robust Peak Detection:** Median Absolute Deviation (MAD) segmentation for accurate transient identification.
* **Kinetic Parameter Fitting:** Single-transient extraction of linear Rate of Rise, Rise time, decay constant ($\\tau$), T50%, T90% FWHM, and Area Under the Curve (AUC).
* **Network Synchrony Mapping:** Pearson correlation and Event Coincidence Rate (ECR) matrices.
* **Spatial Topology \& Hubs:** Automatic identification of top HUB units and Convex Hull cluster boundary overlays.
* **Automated Reporting:** Executive PDF reports and structured multi-sheet Excel export.

\---

## 📦 Installation \& Usage

1. Clone the repository or download the source code:

```bash
   git clone \\\[https://github.com/DavideLattanzi/DLCalciumImagingSuite.git](https://github.com/tuo-username/DLCalciumImagingSuite.git)
   cd DLCalciumImagingSuite


