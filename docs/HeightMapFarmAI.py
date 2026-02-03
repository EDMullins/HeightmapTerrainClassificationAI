import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import convolve
from sklearn.metrics import f1_score

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#-----------------------------------------------------------------
# Process Heightmap Features 
# Synthetic label based on terrain features and agricultural rules
# Input: image filepath
# Output: elevation, slope, aspect
def process_heightmap_features(filepath):
    #"""Return normalized elevation, slope, aspect arrays"""
    img = Image.open(filepath).resize((256, 256))
    elev = np.array(img, dtype=np.uint16)
    elev_norm = (elev - elev.min()) / (elev.max() - elev.min())
    
    # Compute gradient
    gy, gx = np.gradient(elev_norm)
    # Compute initial slope magnitude
    slope = np.sqrt(gx**2 + gy**2)
    # Smooth slope by averaging over neighbors (3x3 mean filter)
    kernel = np.ones((3, 3)) / 9.0
    slope_smooth = convolve(slope, kernel, mode='reflect')
    # Normalize smoothed slope
    slope_norm = (slope_smooth - slope_smooth.min()) / (slope_smooth.max() - slope_smooth.min())

    aspect = np.degrees(np.arctan2(-gy, gx))
    aspect_norm = (aspect + 360) % 360
    aspect_rad = np.deg2rad(aspect_norm)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)
    
    #print the feature maps
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.title("Elevation")
    # plt.imshow(elev_norm, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1, 3, 2)
    # plt.title("Slope")
    # plt.imshow(slope_norm, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1, 3, 3)
    # plt.title("Aspect")
    # plt.imshow(aspect_norm, cmap='hsv')
    # plt.axis('off')
    # plt.show()

    return elev_norm, slope_norm, aspect_sin, aspect_cos

#-----------------------------------------------------------------
# Classify Pixel
# Synthetic label based on terrain features and agricultural rules
# Input: Elev, Slope, Aspect
# Output: Label (string)
def classify_pixel(elev, slope, aspect_sin, aspect_cos):
    if elev < 0.2: elev_zone = "low"
    elif elev > 0.8: elev_zone = "high"
    else: elev_zone = "mid"

    # Reconstruct aspect angle in degrees (0–360)
    aspect = np.degrees(np.arctan2(aspect_sin, aspect_cos)) % 360

    if 150 <= aspect <= 210: aspect_cat = "south"
    elif 90 <= aspect < 150: aspect_cat = "southeast"
    elif 210 < aspect <= 270: aspect_cat = "southwest"
    elif (aspect >= 315) or (aspect <= 45): aspect_cat = "north"
    elif (aspect > 45) and (aspect < 135): aspect_cat = "east"
    else: aspect_cat = "west"

    # Flat ground
    if slope < 0.03:
        if elev_zone == "high": return "building/solar"
        elif elev_zone == "mid":
            if aspect_cat == "southeast": return "annuals_south"
            elif aspect_cat == "southwest": return "drought_tolerant_annuals"
            elif aspect_cat == "north": return "annuals_north"
            else: return "annuals_open"
        else: return "pond_wetland"
    if slope < 0.08:
        if elev_zone == "high": return "perennial_hillside"
        elif elev_zone == "mid": return "orchard" if aspect_cat in ["south","north"] else "perennial"
        else: return "riparian_buffer"
    if slope < 0.18:
        if elev_zone == "high": return "windbreak"
        elif elev_zone == "mid": return "silvopasture" if aspect_cat in ["south","southeast"] else "perennials_contour"
        else: return "erosion_control"
    if slope < 0.35:
        return "windbreak" if aspect_cat in ["east","west"] else "silvopasture"
    if slope < 0.5: return "forest"
    return "unsuitable"

#-----------------------------------------------------------------
# Process Heightmap Labels
# for every image gets a np array with labels for each pixel
# Input: image filepath,
# Output: elevation, slope, aspect
def process_heightmap_labels(filepath, label_to_idx):
    elev, slope, aspect_sin, aspect_cos = process_heightmap_features(filepath)
    shape = elev.shape
    labels = np.empty(shape, dtype=np.int64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            lbl = classify_pixel(elev[i,j], slope[i,j], aspect_sin[i,j], aspect_cos[i,j])
            labels[i,j] = label_to_idx[lbl]
    return labels

#-----------------------------------------------------------------
# TerrainDataset
# A Class that defines the dataset of Terrain
class TerrainDataset(Dataset):
    # Sets basic values for dataset like img directory files and labels
    def __init__(self, img_dir, labels):
        self.img_dir = img_dir
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.labels = labels
        self.label_to_idx = {l:i for i,l in enumerate(labels)}

    # Number of files
    def __len__(self):
        return len(self.files)

    # Gets the data for one image
    def __getitem__(self, idx):
        filepath = os.path.join(self.img_dir, self.files[idx])
        elev, slope, aspect_sin, aspect_cos = process_heightmap_features(filepath)
        img_tensor = torch.tensor(np.stack([elev, slope, aspect_sin, aspect_cos], axis=0), dtype=torch.float32)
        label_map = process_heightmap_labels(filepath, self.label_to_idx)
        label_tensor = torch.tensor(label_map, dtype=torch.long)
        return img_tensor, label_tensor

#-----------------------------------------------------------------
# UNet
# A class that defines a UNet architecture neural network
class UNet(nn.Module):
    # initilizes cnn architecture
    # Input: Number of channels in input images, Number of output channels (labels)
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        # CBR = Convolution -> BatchNorm -> ReLU
        # Input: Input channel num, Output channel num
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True))
        
        # ----- Encoder (downsampling) -----
        # 3 -> 32 channels
        self.enc1 = CBR(in_channels, 32)
        # 32 -> 64 channels
        self.enc2 = CBR(32, 64)
        # 64 -> 128
        self.enc3 = CBR(64, 128)
        # Halves height and width (Resolution /2)
        self.pool = nn.MaxPool2d(2)
        # Doubles height and width (Resolution 2x)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ----- Decoder (Upsampling) -----
        # Combine e3(128) with e2(64) and reduces channels back to 64
        self.dec2 = CBR(128+64, 64)
        # Combine d2(64) with e1(32) and reduce channels to 32
        self.dec1 = CBR(64+32, 32)
        # Final convolution maps the 32 channel features labels
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # --- Encoder Forward Pass ---
        # Extract low-level features (edges) (32, H, W)
        e1 = self.enc1(x)
        # Pool e1 then extract mid-level features (64, H/2, W/2)
        e2 = self.enc2(self.pool(e1))
        # Pool e2 then extract deep features (context) (128, H/4, W/4)
        e3 = self.enc3(self.pool(e2))
        
        # --- Decoder Foward Pass ---
        # Upsample e3 (H/2, W/2), concatenate with e2 (skip connection)
        # Combines context(e3) with spatial detail(e2) (64, H/2, W/2)
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
        # Upsample d2 (H, W), concatenate with e1 (skip connection)
        # combines fine detail(e1) with reconstructed features(d2) (32, H, W)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        # Produces per-pixel output (16, H, W)
        out = self.final(d1)
        return out

#-----------------------------------------------------------------
# Dice Loss
# A class that defines the Dice Loss function for segmentation tasks
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        # logits: (B, C, H, W)
        # true:   (B, H, W)
        
        num_classes = logits.size(1)

        # Softmax probabilities
        probs = torch.softmax(logits, dim=1)

        # One-hot encode ground truth
        true_1hot = torch.zeros_like(probs)
        true_1hot.scatter_(1, true.unsqueeze(1), 1)

        # Flatten over batch, H, W
        dims = (0, 2, 3)

        intersection = torch.sum(probs * true_1hot, dims)
        sum_probs = torch.sum(probs, dims)
        sum_true = torch.sum(true_1hot, dims)

        dice = (2 * intersection + self.smooth) / (sum_probs + sum_true + self.smooth)

        # Dice loss = 1 - mean dice over classes
        loss = 1 - dice.mean()
        return loss

#-----------------------------------------------------------------
# Train UNet
# This function initializes a U-Net, loads training data from the given directory,
# computes class weights to handle class imbalance, and trains the model for a 
# specified number of epochs using Adam optimization and cross-entropy loss.
# Input: train_dir (string), labels(list), epochs(int, opt), batch_size(int, opt), learning-rate(float, opt)
# Output: trained U-Net model (torch.nn.module), Training dataset object (Dataset)
def train_unet(train_dir, labels, epochs=20, batch_size=2, lr=1e-3):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load training dataset (images + labels)
    train_data = TerrainDataset(train_dir, labels)
    # Dataloader for batching and shuffling training data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # Initialize UNet Model (4 input channels = RGB, Output channels = # of labels)
    model = UNet(in_channels=4, out_channels=len(labels)).to(device)

    # compute class weights
    # helps balance over labeled/trained classes
    class_counts = np.zeros(len(labels))
    # Count Label occurences
    for _,lbl in train_data:
        for i in range(len(labels)):
            class_counts[i] += np.sum(lbl.numpy() == i)
    # Inverse frequency weighting (rare classes get higher weight)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Cross-entropy Loss with class weights
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    # Dice Loss
    criterion_dice = DiceLoss()
    # Adam optimizer to accelerate the gradient descent (faster learning)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train() # Training mode
        total_loss = 0 # Track loss per epoch
        # iterate over each batch of images and labels
        for imgs, lbls in train_loader:
            # Move batch to GPU/CPU
            imgs, lbls = imgs.to(device), lbls.to(device)
            # Reset gradient before backpropagation
            optimizer.zero_grad()
            # Forward pass (predict segmentation map)
            outputs = model(imgs)
            # Compute loss
            ce_loss = criterion_ce(outputs, lbls)
            dice_loss = criterion_dice(outputs, lbls)
            loss = ce_loss + dice_loss
            # Backward pass (compute gradients)
            loss.backward()
            # Update weights
            optimizer.step()
            # Accumulate loss
            total_loss += loss.item()
        # Print avg loss for the epoch
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
    
    print("Training complete.")
    return model, train_data

#-----------------------------------------------------------------
# Visualize Prediction
# for element in dataset, shows the predicted classified image and the original heightmap
# Input: model, dataset, label_colors, 
# Output: elevation, slope, aspect
def visualize_prediction(model, dataset, label_colors, img_dir):
    device = next(model.parameters()).device
    model.eval()

    for i in range(len(dataset)):
        img_tensor, _ = dataset[i]
        img = img_tensor.unsqueeze(0).to(device)

        # --- Model Prediction ---
        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).squeeze(0).cpu().numpy()

        # --- Original Heightmap ---
        img_path = os.path.join(img_dir, dataset.files[i])
        orig_img = np.array(Image.open(img_path).resize((256, 256)))

        # --- Rule-Based Ground Truth ---
        elev, slope, aspect_sin, aspect_cos = process_heightmap_features(img_path)
        h, w = elev.shape
        rule_based_labels = np.zeros((h, w), dtype=np.int32)
        for y in range(h):
            for x in range(w):
                label = classify_pixel(elev[y,x], slope[y,x], aspect_sin[y,x], aspect_cos[y,x])
                rule_based_labels[y, x] = dataset.label_to_idx[label]

        # --- Convert Masks to RGB ---
        def colorize(mask):
            h, w = mask.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for label, color in label_colors.items():
                idx = dataset.label_to_idx[label]
                rgb[mask == idx] = color
            return rgb

        gt_img = colorize(rule_based_labels)
        pred_img = colorize(pred)

        # --- Visualization ---
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Heightmap")
        plt.imshow(orig_img, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Rule-Based Ground Truth")
        plt.imshow(gt_img)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Model Prediction")
        plt.imshow(pred_img)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

#-----------------------------------------------------------------
# Evaluate Model Accuracy
# Runs the test dataset through the trained model and compares predictions
# against the rule-based ground truth generated by classify_pixel.
# Computes per-label accuracy, label counts, and total accuracy.
def evaluate_model_accuracy(model, dataset):
    device = next(model.parameters()).device
    model.eval()

    label_names = dataset.labels
    num_labels = len(label_names)

    all_preds = []
    all_labels = []

    # Track per-label accuracy
    total_per_label = np.zeros(num_labels, dtype=np.int64)
    correct_per_label = np.zeros(num_labels, dtype=np.int64)

    with torch.no_grad():
        for i in range(len(dataset)):
            img_tensor, true_label_tensor = dataset[i]
            img = img_tensor.unsqueeze(0).to(device)
            true_labels = true_label_tensor.numpy()

            pred = torch.argmax(model(img), dim=1).squeeze(0).cpu().numpy()

            all_preds.append(pred.flatten())
            all_labels.append(true_labels.flatten())

            for idx in range(num_labels):
                mask = (true_labels == idx)
                total_per_label[idx] += np.sum(mask)
                correct_per_label[idx] += np.sum(pred[mask] == idx)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracies = correct_per_label / np.maximum(total_per_label, 1)
    total_accuracy = np.sum(correct_per_label) / np.maximum(np.sum(total_per_label), 1)

    # Compute F1 scores
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Print results
    print("\n--- Model Evaluation ---")
    for name, acc, total, correct in zip(label_names, accuracies, total_per_label, correct_per_label):
        print(f"{name:25s}: {acc*100:6.2f}%  ({correct}/{total})")

    print(f"\nTotal Accuracy: {total_accuracy*100:.2f}%")
    print(f"F1 (macro): {f1_macro:.3f}")
    print(f"F1 (weighted): {f1_weighted:.3f}")

# ----------------- Main -----------------
label_colors = {
    "building/solar": (255, 212, 0),
    "annuals_south": (255, 27, 225),
    "drought_tolerant_annuals": (217, 3, 104),
    "annuals_north": (178, 0, 255),
    "annuals_open": (255, 185, 240),
    "pond_wetland": (89, 143, 219),
    "perennial_hillside": (201, 142, 223),
    "perennials_contour": (5, 200, 21),
    "perennial": (33, 184, 124),
    "riparian_buffer": (39, 120, 86),
    "windbreak": (183, 255, 254),
    "silvopasture": (102, 131, 63),
    "orchard": (101, 3, 88),
    "erosion_control": (252, 207, 136),
    "forest": (5, 75, 23),
    "unsuitable": (43, 1, 1)
}

train_dir = "Train"
test_dir = "Test"
labels = list(label_colors.keys())

model, train_dataset = train_unet(train_dir, labels, epochs=30, batch_size=2)
test_dataset = TerrainDataset(test_dir, labels)

visualize_prediction(model, test_dataset, label_colors, test_dir)
evaluate_model_accuracy(model, test_dataset)