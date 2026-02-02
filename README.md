# Heightmap Terrain Classification AI

Author: Ethan Mullins  
Class: CS 470

---
# Abstract
This project explores the application of deep learning for terrain segmentation using synthetically generated labels derived from heightmap features. The study introduces a preprocessing pipeline that computes elevation, slope, and aspect from real heightmap data, followed by a rule-based classifier that assigns land-use categories relevant to agricultural and ecological planning. A U-Net convolutional neural network is trained on these features using a combination of weighted cross-entropy loss and Dice loss to address class imbalance. The model is evaluated on an unseen test set, producing segmentation outputs that are compared visually and quantitatively against the rule-based ground truth. Results show that the U-Net learns meaningful spatial patterns from the handcrafted rules, demonstrating moderate to strong agreement depending on label frequency. This project highlights how synthetic labels can bootstrap segmentation models in domains where real annotations are costly.

# 1. Introduction

Terrain analysis plays a crucial role in agriculture, forestry, hydrology, and ecological land-use planning. Traditional geographic information systems (GIS) rely heavily on manual annotation or expert-driven classification, which is time-consuming and expensive. The goal of this project is to automate terrain classification using deep learning, specifically a U-Net architecture, trained on labels generated from a custom rule-based system.  
This approach eliminates the need for manual data labeling while providing a repeatable and scalable preprocessing pipeline. The central research question explored in this project is:

**Can a deep learning model learn meaningful segmentation from synthetic rule-derived terrain labels?**

# 2. Methodologies

Each heightmap undergoes a multi-stage preprocessing pipeline that converts raw elevation values into a structured, multi-channel representation suitable for training a segmentation model. This process is handled primarily by the functions `process_heightmap_features()`, `process_heightmap_labels()`, and the `TerrainDataset` class.

## 2.1. Feature Extraction

For every image, the system first loads and resizes the heightmap to a fixed 256 × 256 resolution. The function `process_heightmap_features()` computes four core feature channels:

**1. Normalized Elevation**  
Raw 16-bit elevation values are min-max normalized to the range [0, 1].
   
**2. Smoothed Slope**    
* Gradients are calculated using `np.gradient`
* Slope magnitude is computed as:
    - √((δx)² + (δy)²)
* A 3x3 mean filter (`scipy.ndimage.convolve`) smooths local noise.
* The smoothed slope is normalized to [0, 1].

**3. Aspect (Sin and Cos components)**  
Aspect (0-360°) is derived using arctan2 and converted to radians. To avoid angular discontinuities in training, aspect is encoded as:  
* aspect_sin = sin(aspect)
* aspect_cos = cos(aspect)

The output of this step is a 4-channel tensor representing:  
* `[elevation, slope, aspect_sin, aspect_cos]`

TODO: Add image: Elevation, Slope, Aspect

## 2.2. Rule-Based Labels  

# 3. Dataset

# 4. Results

# 5. Conclusion
