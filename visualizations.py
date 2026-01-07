"""
Visualization Utilities
Includes Grad-CAM visualization and model explainability tools.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

class GradCAMVisualizer:
    """
    Grad-CAM visualization for ResNet50 embeddings.
    Shows which parts of satellite images the CNN focuses on.
    """
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load full ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Target layer for Grad-CAM (last conv block)
        self.target_layer = self.model.layer4[-1]
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Hook to capture forward activations."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_gradcam(self, img_path):
        """
        Generate Grad-CAM heatmap for an image.
        
        Parameters:
        -----------
        img_path : str
            Path to input image
        
        Returns:
        --------
        tuple
            (original_image, heatmap)
        """
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Global average pooling of gradients
        pooled_grads = torch.mean(self.gradients, dim=(0, 2, 3))
        
        # Weight activations by gradients
        weighted_activations = self.activations[0].clone()
        for i in range(weighted_activations.shape[0]):
            weighted_activations[i] *= pooled_grads[i]
        
        # Generate heatmap
        heatmap = torch.mean(weighted_activations, dim=0)
        heatmap = F.relu(heatmap)
        heatmap = heatmap.cpu().numpy()
        
        # Normalize
        heatmap /= heatmap.max() + 1e-8
        heatmap = cv2.resize(heatmap, img.size)
        
        return np.array(img), heatmap
    
    def visualize(self, img_path, alpha=0.4, save_path=None):
        """
        Visualize Grad-CAM overlay on image.
        
        Parameters:
        -----------
        img_path : str
            Path to input image
        alpha : float
            Transparency of heatmap overlay
        save_path : str, optional
            Path to save visualization
        """
        img, heatmap = self.generate_gradcam(img_path)
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )
        
        # Overlay
        overlay = cv2.addWeighted(
            img, 1 - alpha, heatmap_color, alpha, 0
        )
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Grad-CAM visualization saved to {save_path}")
        else:
            plt.show()
    
    def batch_visualize(self, image_dir, sample_ids, output_dir="gradcam_outputs"):
        """
        Generate Grad-CAM visualizations for multiple images.
        
        Parameters:
        -----------
        image_dir : str
            Directory containing images
        sample_ids : list
            List of property IDs to visualize
        output_dir : str
            Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create image lookup
        image_files = {
            f.split("_")[0]: f
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        }
        
        for pid in sample_ids:
            pid = str(pid)
            
            if pid not in image_files:
                print(f"⚠️ Image not found for ID: {pid}")
                continue
            
            img_path = os.path.join(image_dir, image_files[pid])
            save_path = os.path.join(output_dir, f"gradcam_{pid}.png")
            
            print(f"Generating Grad-CAM for property {pid}...")
            self.visualize(img_path, save_path=save_path)


def plot_price_distribution(df, save_path=None):
    """
    Plot distribution of house prices.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'price' column
    save_path : str, optional
        Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original scale
    axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Price Distribution')
    axes[0].grid(alpha=0.3)
    
    # Log scale
    axes[1].hist(np.log1p(df['price']), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Log(Price + 1)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log Price Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Price distribution plot saved to {save_path}")
    else:
        plt.show()


def plot_correlation_heatmap(df, features, save_path=None):
    """
    Plot correlation heatmap for selected features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    features : list
        List of feature names to include
    save_path : str, optional
        Path to save plot
    """
    import seaborn as sns
    
    corr = df[features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Correlation heatmap saved to {save_path}")
    else:
        plt.show()


def plot_spatial_distribution(df, save_path=None):
    """
    Plot spatial distribution of properties.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'lat', 'long', 'price' columns
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['long'], df['lat'], 
                         c=np.log1p(df['price']), 
                         cmap='viridis', 
                         s=10, 
                         alpha=0.5)
    plt.colorbar(scatter, label='Log(Price + 1)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Distribution of Properties (colored by price)')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Spatial distribution plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    
    # 1. Grad-CAM visualization
    print("Generating Grad-CAM visualizations...")
    visualizer = GradCAMVisualizer()
    
    IMAGE_DIR = "C:\\Users\\anshr\\Downloads\\multimodal_price_prediction\\images"  #put the directory in which images are
    SAMPLE_IDS = ['3585300445', '6600220380', '7781600100']
    
    visualizer.batch_visualize(IMAGE_DIR, SAMPLE_IDS, "outputs/gradcam")
    
    # 2. Data visualizations
    print("\nGenerating data visualizations...")
    df = pd.read_csv("/kaggle/working/processed_data.csv") #put processed_data path
    
    plot_price_distribution(df, "outputs/price_distribution.png")
    
    key_features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 
                   'grade', 'condition', 'waterfront']
    plot_correlation_heatmap(df, key_features, "outputs/correlation_heatmap.png")
    
    plot_spatial_distribution(df, "outputs/spatial_distribution.png")
    
    print("\n✅ All visualizations complete!")
#heelo