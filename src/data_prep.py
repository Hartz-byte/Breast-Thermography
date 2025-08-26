import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, Callable, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml
from typing import Dict, List, Tuple, Optional, Callable, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class BreastThermographyDataset(Dataset):
    """PyTorch dataset for breast thermography images with class-based folder structure."""
    
    def __init__(self, data_df: pd.DataFrame, transform: Optional[Callable] = None, 
                 config: Optional[Dict] = None, data_preprocessor=None):
        """Initialize dataset.
        
        Args:
            data_df: DataFrame containing image paths and labels
            transform: Optional transform to be applied on a sample
            config: Configuration dictionary
            data_preprocessor: Reference to parent DataPreprocessor for class weights
        """
        self.data_df = data_df.reset_index(drop=True)
        self.transform = transform
        self.config = config or {}
        self.views = self.config.get('data', {}).get('views', ['anterior', 'oblleft', 'oblright'])
        self.data_preprocessor = data_preprocessor
        
        # Map class names to folder names
        self.class_to_folder = {
            0: 'Normal',
            1: 'Benign',
            2: 'Malignant'
        }
        
    def __len__(self) -> int:
        return len(self.data_df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data_df.iloc[idx]
        patient_id = row['Image']
        label = row['label']
        
        # Get class folder name
        class_folder = self.class_to_folder[label]
        
        # Load all three views
        images = []
        for view in self.views:
            img_path = os.path.join(
                self.config['data']['raw_path'],
                'Breast-Thermography-Raw',
                class_folder,
                patient_id,
                f"{patient_id}_{view}.jpg"
            )
            
            try:
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    if image is None:
                        raise ValueError(f"Failed to load image at {img_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # If image is missing, use a black image and log a warning
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
                    if self.transform is not None:  # Only log during training
                        logging.warning(f"Missing image at {img_path}, using zero tensor")
                
                if self.transform is not None:
                    # Apply class-specific transformations if available
                    if hasattr(self, 'data_preprocessor'):
                        # Get class-specific transform
                        class_transform = self.data_preprocessor.create_transforms(
                            is_train=True,
                            label=label
                        )
                        augmented = class_transform(image=image)
                    else:
                        # Fallback to default transform
                        augmented = self.transform(image=image)
                        
                    image = augmented['image']
                    
                images.append(image)
                
            except Exception as e:
                logging.error(f"Error loading {img_path}: {str(e)}")
                # Return a zero tensor of the expected shape if there's an error
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                images.append(image)
        
        # Stack views along channel dimension (3 views × 3 channels = 9 channels)
        try:
            combined_image = np.concatenate(images, axis=-1)
            combined_image = torch.from_numpy(combined_image).float()
            combined_image = combined_image.permute(2, 0, 1)  # HWC to CHW
        except Exception as e:
            logging.error(f"Error processing images for patient {patient_id}: {str(e)}")
            # Return zero tensor of expected shape if concatenation fails
            combined_image = torch.zeros((9, 224, 224), dtype=torch.float32)
        
        return {
            'image': combined_image,
            'label': label,
            'patient_id': patient_id,
            'temp': torch.tensor(row['Temp(°C)'], dtype=torch.float32) if 'Temp(°C)' in row else torch.tensor(0.0, dtype=torch.float32)
        }

class DataPreprocessor:
    """Data preprocessing and loading utilities."""
    
    def __init__(self, config_path: str = "../configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.label_encoder = LabelEncoder()
        self.left_encoder = LabelEncoder()
        self.right_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)
        
        # Class balancing parameters
        self.class_weights = None
        self.class_counts = None
        self.label_mapping = {'N': 0, 'PB': 1, 'PM': 2}
        self.label_names = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
        
    def load_diagnostics(self):
        """Load and preprocess the diagnostics Excel file."""
        try:
            # Construct the full path to the Excel file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_path = os.path.join(base_dir, self.config['data']['raw_path'].replace('/', os.sep))
            excel_path = os.path.join(raw_path, 'Breast-Thermography-Raw', self.config['data']['excel_file'])
            
            # Check if file exists
            if not os.path.exists(excel_path):
                raise FileNotFoundError(f"Excel file not found at: {excel_path}")
                
            df = pd.read_excel(excel_path)
            
            # Clean up column names
            df.columns = [col.strip() for col in df.columns]
            
            # Map labels to numerical values
            label_mapping = {'N': 0, 'PB': 1, 'PM': 2}
            
            # Create target labels (multi-label format)
            df['left_label'] = df['Left'].map(label_mapping)
            df['right_label'] = df['Right'].map(label_mapping)
            
            # Combine labels (using the more severe condition if both sides have issues)
            df['label'] = df[['left_label', 'right_label']].max(axis=1)
            
            # Map numerical labels back to string labels for readability
            label_names = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
            df['combined_label'] = df['label'].map(label_names)
            
            # Store class distribution
            self.class_counts = df['label'].value_counts().sort_index()
            total_samples = len(df)
            num_classes = len(self.class_counts)
            
            # Get weighting method from config or use default
            method = self.config['training'].get('class_weighting', 'inverse_freq')
            
            # Calculate class weights
            if method == 'inverse_freq':
                weights = total_samples / (num_classes * self.class_counts.values)
            elif method == 'sqrt':
                weights = np.sqrt(total_samples / self.class_counts.values)
            elif method == 'balanced':
                weights = (1 / self.class_counts.values) * (total_samples / num_classes)
            elif method == 'effective':
                beta = self.config['training'].get('beta', 0.999)
                effective_num = 1.0 - np.power(beta, self.class_counts.values)
                weights = (1.0 - beta) / effective_num
                weights = weights / np.sum(weights) * num_classes
            else:
                weights = np.ones(num_classes)
                
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            
            self.logger.info(f"Class distribution: {self.class_counts.to_dict()}")
            self.logger.info(f"Using {method} class weights: {self.class_weights.tolist()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading diagnostics file: {str(e)}")
            raise
    
    def _get_augmentation_strength(self, label: Optional[int] = None) -> float:
        """Get augmentation strength multiplier based on class frequency."""
        if label is None or self.class_counts is None:
            return 1.0
            
        # Get class frequency (inverse of count, normalized)
        freq = 1.0 / self.class_counts[label]
        max_freq = 1.0 / self.class_counts.min()
        
        # Scale between base_strength and max_strength
        base_strength = self.config['augmentation'].get('base_strength', 0.5)
        max_strength = self.config['augmentation'].get('max_strength', 2.0)
        
        # Linear scaling based on inverse frequency
        strength = base_strength + (max_strength - base_strength) * (freq / max_freq)
        return min(max(strength, base_strength), max_strength)
    
    def create_transforms(self, is_train: bool = True, label: Optional[int] = None) -> A.Compose:
        """
        Create data augmentation transforms with class-specific augmentation strength.
        
        Args:
            is_train: Whether to create training transforms (if False, only basic transforms are applied)
            label: Class label (used to determine augmentation strength for training)
            
        Returns:
            A.Compose: Albumentations composition of transforms
        """
        # Base resize and normalize transforms
        base_transforms = [
            A.Resize(
                self.config['data']['image_size'][0], 
                self.config['data']['image_size'][1]
            ),
            A.Normalize(
                mean=self.config['data']['mean'],
                std=self.config['data']['std']
            ),
            ToTensorV2()
        ]
        
        if not is_train:
            return A.Compose(base_transforms)
            
        # Get augmentation strength based on class frequency
        strength = self._get_augmentation_strength(label)
        
        # Base augmentation probabilities
        p_hflip = 0.5 * strength
        p_vflip = 0.5 * strength
        p_rotate = 0.5 * strength
        p_brightness = 0.2 * strength
        p_affine = 0.3 * strength
        
        # Define augmentation pipeline
        augmentations = [
            # Geometric transforms
            A.HorizontalFlip(p=p_hflip),
            A.VerticalFlip(p=p_vflip),
            A.Rotate(limit=30, p=p_rotate),
            A.ShiftScaleRotate(
                shift_limit=0.1 * strength,
                scale_limit=0.1 * strength,
                rotate_limit=15 * strength,
                p=p_affine
            ),
            
            # Color transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.2 * strength,
                contrast_limit=0.2 * strength,
                p=p_brightness
            ),
            A.HueSaturationValue(
                hue_shift_limit=20 * strength,
                sat_shift_limit=30 * strength,
                val_shift_limit=20 * strength,
                p=p_brightness
            ),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0 * strength, 50.0 * strength), p=0.1 * strength),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1 * strength),
            
            # Advanced augmentations
            A.CoarseDropout(
                max_holes=8,
                max_height=32 * strength,
                max_width=32 * strength,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.1 * strength
            ),
            A.RandomGridShuffle(grid=(2, 2), p=0.1 * strength)
        ]
        
        # Combine all transforms
        transform = A.Compose([
            *augmentations,
            *base_transforms
        ])
        
        return transform
    
    def save_processed_data(self, df: pd.DataFrame, output_dir: str = "../data/processed") -> None:
        """Save preprocessed data to disk with progress tracking."""
        from tqdm import tqdm
        import time
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create reverse mapping from label to class name
        label_to_class = {v: k for k, v in self.class_to_folder.items()}
        
        # Create subdirectories for each class
        for cls in self.class_to_folder.values():
            os.makedirs(os.path.join(output_dir, str(cls)), exist_ok=True)
        
        # Initialize counters
        total_samples = len(df)
        processed = 0
        errors = 0
        
        print(f"\nProcessing {total_samples} samples...")
        start_time = time.time()
        
        # Process and save each sample
        for idx, row in tqdm(df.iterrows(), total=total_samples, desc="Processing"):
            try:
                patient_id = row['Image']
                label = row['label']
                
                # Map numeric label to class name
                class_name = label_to_class.get(label, str(label))
                output_path = os.path.join(output_dir, class_name, f"{patient_id}.pt")
                
                # Skip if already processed
                if os.path.exists(output_path):
                    processed += 1
                    continue
                    
                # Get the sample and save
                sample = self.dataset[idx]
                torch.save({
                    'image': sample['image'],
                    'label': label,
                    'patient_id': patient_id
                }, output_path)
                
                processed += 1
                
            except Exception as e:
                errors += 1
                print(f"\nError processing {patient_id if 'patient_id' in locals() else 'unknown'}: {str(e)}")
                # Add a small delay to prevent error message flooding
                time.sleep(0.1)
                continue
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\nProcessing complete!")
        print(f"  - Processed: {processed}/{total_samples} samples")
        print(f"  - Errors: {errors}")
        print(f"  - Time taken: {elapsed:.2f} seconds")
        if errors > 0:
            print("\nWarning: Some samples could not be processed. Check the error messages above.")
        
        print(f"\nProcessed data saved to: {os.path.abspath(output_dir)}")
    
    def create_datasets(self, save_processed: bool = True) -> Tuple[DataLoader, DataLoader, pd.DataFrame]:
        """Create train and validation datasets.
        
        Args:
            save_processed: If True, save processed data to disk
        """
        df = self.load_diagnostics()
        
        # Get unique classes in the data
        unique_classes = df['label'].unique()
        num_classes = len(unique_classes)
        
        # Update class_to_folder mapping based on actual data
        self.class_to_folder = {i: cls for i, cls in enumerate(unique_classes)}
        
        # Create train/validation split
        train_df, val_df = train_test_split(
            df,
            test_size=self.config['validation']['split_ratio'],
            random_state=self.config['validation']['random_state'],
            stratify=df['label'] if self.config['validation']['stratify'] else None
        )
        
        # Create datasets
        train_dataset = BreastThermographyDataset(
            train_df, 
            transform=self.create_transforms(is_train=True),
            config=self.config,
            data_preprocessor=self
        )
        
        val_dataset = BreastThermographyDataset(
            val_df,
            transform=self.create_transforms(is_train=False),
            config=self.config,
            data_preprocessor=self
        )
        
        # Save processed data if requested
        if save_processed:
            self.dataset = train_dataset  # Store reference for saving
            self.save_processed_data(train_df)
            
            self.dataset = val_dataset
            self.save_processed_data(val_df)
        
        # Calculate weights for weighted random sampler
        train_labels = train_df['label'].values
        class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[np.where(np.unique(train_labels) == t)[0][0]] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        
        # Create sampler
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, 
            len(samples_weight),
            replacement=True
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=sampler,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            drop_last=True,
            persistent_workers=True if self.config['hardware']['num_workers'] > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            persistent_workers=True if self.config['hardware']['num_workers'] > 0 else False
        )
        
        return train_loader, val_loader, df
    
    def visualize_data_distribution(self, df: pd.DataFrame, save_path: str = "outputs/plots/data_distribution.png"):
        """Visualize data distribution."""
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Label distribution
        axes[0, 0].pie(df['combined_label'].value_counts().values, 
                      labels=df['combined_label'].value_counts().index,
                      autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Label Distribution')
        
        # Age distribution by label
        sns.boxplot(data=df, x='combined_label', y='Age(years)', ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Label')
        
        # Weight distribution by label
        sns.boxplot(data=df, x='combined_label', y='Weight (Kg)', ax=axes[0, 2])
        axes[0, 2].set_title('Weight Distribution by Label')
        
        # Height distribution by label
        sns.boxplot(data=df, x='combined_label', y='Height(cm)', ax=axes[1, 0])
        axes[1, 0].set_title('Height Distribution by Label')
        
        # Temperature distribution by label
        sns.boxplot(data=df, x='combined_label', y='Temp(°C)', ax=axes[1, 1])
        axes[1, 1].set_title('Temperature Distribution by Label')
        
        # Correlation heatmap
        numeric_cols = ['Age(years)', 'Weight (Kg)', 'Height(cm)', 'Temp(°C)', 'label']
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 2])
        axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print("Dataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Label distribution:\n{df['combined_label'].value_counts()}")
        print(f"\nLeft breast status:\n{df['Left'].value_counts()}")
        print(f"\nRight breast status:\n{df['Right'].value_counts()}")
        
    def get_label_mappings(self) -> Dict:
        """Get label encoding mappings."""
        return {
            'combined': dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))),
            'left': dict(zip(self.left_encoder.classes_, range(len(self.left_encoder.classes_)))),
            'right': dict(zip(self.right_encoder.classes_, range(len(self.right_encoder.classes_))))
        }

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    train_loader, val_loader, df = preprocessor.create_dataloaders()
    preprocessor.visualize_data_distribution(df)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Label mappings: {preprocessor.get_label_mappings()}")
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
