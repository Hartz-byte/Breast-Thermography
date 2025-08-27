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
        
        # Map class names to folder names (only Benign and Malignant)
        self.class_to_folder = {
            0: 'Benign',
            1: 'Malignant'
        }
        
    def __len__(self) -> int:
        return len(self.data_df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        processed_images = []
        row = self.data_df.iloc[idx]
        # Ensure patient_id is a string and strip any whitespace
        patient_id = str(row['Image']).strip()
        label = int(row['label'])  # Ensure label is an integer (0 or 1)
        
        # Validate label is within expected range
        if label not in self.class_to_folder:
            logging.warning(f"Invalid label {label} for patient {patient_id}, defaulting to Benign (0)")
            label = 0  # Default to Benign for any invalid labels
        
        # Get class folder name
        class_folder = self.class_to_folder[label]
        
        # Load all three views
        images = []
        missing_views = 0
        
        target_size = self.config.get('data', {}).get('image_size', [224, 224])
        for view in self.views:
            # Construct filename
            filename = f"{patient_id}_{view}.jpg"
            # Build path components as strings
            path_components = [
                str(self.config.get('data', {}).get('raw_path', '')).strip(),
                'Breast-Thermography-Raw',
                class_folder,
                patient_id,
                filename
            ]
            
            # Filter out any empty path components
            path_components = [str(pc) for pc in path_components if pc]
            # Join path components
            img_path = os.path.join(*path_components)
            
            try:
                if os.path.exists(img_path) and os.access(img_path, os.R_OK):
                    image = cv2.imread(img_path)
                    if image is None:
                        raise ValueError(f"Failed to read image at {img_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image.shape != target_size or image.shape != target_size:
                        image = cv2.resize(image, (target_size, target_size))
                else:
                    missing_views += 1
                    image = np.zeros((*target_size, 3), dtype=np.uint8)
                    
                # Apply transforms to each view individually
                if self.transform:
                    if isinstance(self.transform, dict):
                        transform = self.transform.get(label, self.transform.get(0))
                        if transform:
                            augmented = transform(image=image)
                            image = augmented['image']
                    else:
                        augmented = self.transform(image=image)
                        image = augmented['image']
                
                # Ensure image is a tensor with correct shape (3, 224, 224)
                if not isinstance(image, torch.Tensor):
                    image = torch.from_numpy(image).float()
                    if len(image.shape) == 3 and image.shape == 3:  # HWC format
                        image = image.permute(2, 0, 1)  # Convert to CHW
                
                # Verify shape is correct
                if image.shape != (3, 224, 224):
                    logging.warning(f"Unexpected image shape {image.shape} for {patient_id}_{view}")
                    image = torch.zeros((3, 224, 224), dtype=torch.float32)
                    
                processed_images.append(image)
                
            except Exception as e:
                missing_views += 1
                logging.error(f"Error loading {img_path}: {str(e)}")
                processed_images.append(torch.zeros((3, 224, 224), dtype=torch.float32))
        
        # Log warning for missing views
        if missing_views > 0:
            logging.warning(f"Patient {patient_id} is missing {missing_views} view(s) out of {len(self.views)}")
        
        # Concatenate along channel dimension: (3, 224, 224) x 3 -> (9, 224, 224)
        try:
            combined_image = torch.cat(processed_images, dim=0)  # Concatenate along channel dimension
            
            # Verify final shape
            expected_shape = (9, 224, 224)
            if combined_image.shape != expected_shape:
                logging.error(f"Incorrect combined shape {combined_image.shape}, expected {expected_shape}")
                combined_image = torch.zeros(expected_shape, dtype=torch.float32)
                
        except Exception as e:
            logging.error(f"Error concatenating images for patient {patient_id}: {str(e)}")
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
        # Updated to only include Benign (0) and Malignant (1)
        self.label_mapping = {'N': 0, 'PB': 0, 'PM': 1}
        self.label_names = {0: 'Benign', 1: 'Malignant'}
        
        # Add class_to_folder mapping for saving processed data
        self.class_to_folder = {
            0: 'Benign',
            1: 'Malignant'
        }
        
    def load_diagnostics(self):
        """
        Load and preprocess the diagnostics Excel file.
        Handles only Benign (0) and Malignant (1) classes.
        """
        try:
            # Get the base directory (project root)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Get paths from config
            rel_raw_path = self.config['data']['raw_path']
            excel_file = self.config['data']['excel_file']
            
            # Construct possible paths where the Excel file might be located
            possible_paths = [
                # Try the direct path from config
                os.path.normpath(os.path.join(base_dir, rel_raw_path, excel_file)),
                # Try with just data/raw/Breast-Thermography-Raw
                os.path.normpath(os.path.join(base_dir, 'data', 'raw', 'Breast-Thermography-Raw', excel_file)),
                # Try in data/raw directly
                os.path.normpath(os.path.join(base_dir, 'data', 'raw', excel_file)),
                # Try the exact path we know exists
                os.path.normpath(os.path.join(base_dir, 'data', 'raw', 'Breast-Thermography-Raw', 'Diagnostics.xlsx'))
            ]
            
            # Try each possible path
            excel_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    excel_path = path
                    break
            
            if not excel_path:
                # List directory contents for debugging
                print("Could not find Excel file. Searched in:")
                for path in possible_paths:
                    parent_dir = os.path.dirname(path)
                    print(f"- {path} (exists: {os.path.exists(path)})")
                    if os.path.exists(parent_dir):
                        print(f"  Directory contents: {os.listdir(parent_dir)}")
                
                raise FileNotFoundError(
                    f"Excel file '{excel_file}' not found in any of the expected locations. "
                    f"Please ensure the file exists in one of these paths:\n"
                    + "\n".join(f"- {path}" for path in possible_paths)
                )
            
            print(f"Loading diagnostics from: {excel_path}")
            df = pd.read_excel(excel_path)
            
            # Clean up column names
            df.columns = [col.strip() for col in df.columns]
            
            # Map labels to numerical values
            # N (Normal) and PB (Probably Benign) are mapped to 0 (Benign)
            # PM (Probably Malignant) is mapped to 1 (Malignant)
            left_mapping = {'N': 0, 'PB': 0, 'PM': 1}
            right_mapping = {'N': 0, 'PB': 0, 'PM': 1}
            
            # Create target labels for each breast
            df['left_label'] = df['Left'].map(left_mapping)
            df['right_label'] = df['Right'].map(right_mapping)
            
            # Combine labels (using the more severe condition if both sides have issues)
            df['label'] = df[['left_label', 'right_label']].max(axis=1)
            
            # Filter out any rows with invalid labels (shouldn't happen with our mapping)
            df = df[df['label'].isin([0, 1])].copy()
            
            # Map numerical labels back to string labels for readability
            label_names = {0: 'Benign', 1: 'Malignant'}
            df['combined_label'] = df['label'].map(label_names)
            
            # Store class distribution
            self.class_counts = df['label'].value_counts().sort_index()
            total_samples = len(df)
            
            # Ensure we have exactly 2 classes
            num_classes = 2
            
            # Fill in missing classes with 0 count
            for i in range(num_classes):
                if i not in self.class_counts:
                    self.class_counts[i] = 0
            
            # Sort the class counts to ensure consistent ordering
            self.class_counts = self.class_counts.sort_index()
            
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
            
            # Convert to tensor and ensure we have weights for both classes
            if len(weights) < num_classes:
                # If we're missing weights for any class, use 1.0 as default
                full_weights = np.ones(num_classes)
                for i in range(min(len(weights), num_classes)):
                    full_weights[i] = weights[i]
                weights = full_weights
                
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            
            self.logger.info(f"Class distribution: {self.class_counts.to_dict()}")
            self.logger.info(f"Using {method} class weights: {self.class_weights.tolist()}")
            
            # Log any potential issues
            if len(df) == 0:
                self.logger.warning("No valid samples found after filtering!")
            elif any(count == 0 for count in self.class_counts):
                self.logger.warning(f"Some classes have zero samples: {self.class_counts.to_dict()}")
            
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
    
    def visualize_data_distribution(self, df: pd.DataFrame, save_path: str = None) -> None:
        """
        Visualize the distribution of data across classes.
        
        Args:
            df: DataFrame containing the data
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))
        
            # Create a copy to avoid modifying the original
            plot_df = df.copy()
            
            # Map numeric labels to names
            plot_df['class_name'] = plot_df['label'].map(self.label_names)
            
            # Count plot
            ax = sns.countplot(x='class_name', data=plot_df, 
                            order=self.label_names.values())
            
            # Add counts on top of bars
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points')
            
            plt.title('Distribution of Classes')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.tight_layout()
            
            # Save the figure if path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
                
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error in visualize_data_distribution: {str(e)}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, output_dir: str = None) -> None:
        """Save processed data with proper directory structure."""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                self.config['data']['processed_path'].replace('/', os.sep)
            )
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for each class
        for cls_name in self.class_to_folder.values():
            os.makedirs(os.path.join(output_dir, str(cls_name)), exist_ok=True)
            
        # Save CSV with metadata
        df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
        
        from tqdm import tqdm
        import time
        
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
                
                # Get class name from label
                class_name = self.class_to_folder.get(label, str(label))
                output_path = os.path.join(output_dir, str(class_name), f"{patient_id}.pt")
                
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
        """Create train and validation datasets for binary classification.
        
        Args:
            save_processed: If True, save processed data to disk
            
        Returns:
            Tuple containing train_loader, val_loader, and the processed DataFrame
        """
        try:
            # Load and preprocess the data
            df = self.load_diagnostics()
            
            # Ensure we have data to work with
            if df is None or len(df) == 0:
                raise ValueError("No data available after loading diagnostics")
                
            # Ensure required columns exist
            required_columns = ['Image', 'label']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns in dataframe: {missing}")
            
            # Ensure we have both classes present
            unique_labels = df['label'].unique()
            if len(unique_labels) < 2:
                raise ValueError(f"Expected at least 2 classes, found only: {unique_labels}")
                
            # Log data distribution
            self.logger.info("\n" + "="*50)
            self.logger.info("DATASET SUMMARY")
            self.logger.info("="*50)
            self.logger.info(f"Total samples: {len(df)}")
            self.logger.info("Class distribution:")
            for label, count in df['label'].value_counts().sort_index().items():
                self.logger.info(f"  - {self.label_names.get(label, f'Class {label}')}: {count} samples")
            
            # Split data into train and validation sets
            split_ratio = self.config['validation'].get('split_ratio', 0.2)
            stratify = df['label'] if self.config['validation'].get('stratify', True) else None
            random_state = self.config['validation'].get('random_state', 42)
            
            # Ensure we have enough samples for the split
            min_samples_per_class = 2  # Minimum samples per class for train/test split
            for label in unique_labels:
                if (df['label'] == label).sum() < min_samples_per_class * 2:  # Need at least 2 samples per class
                    raise ValueError(f"Class {self.label_names.get(label, f'Class {label}')} has too few samples for splitting")
            
            train_df, val_df = train_test_split(
                df,
                test_size=split_ratio,
                stratify=stratify,
                random_state=random_state
            )
            
            self.logger.info("\n" + "-"*50)
            self.logger.info(f"Training samples: {len(train_df)}")
            self.logger.info(f"Validation samples: {len(val_df)}")
            self.logger.info("-"*50 + "\n")
            
            # Create class-specific transforms for training data
            train_transforms = {}
            for label in [0, 1]:  # Only Benign (0) and Malignant (1)
                train_transforms[label] = self.create_transforms(is_train=True, label=label)
            
            # Create datasets
            train_dataset = BreastThermographyDataset(
                train_df, 
                transform=train_transforms,  # Use class-specific transforms for training
                config=self.config,
                data_preprocessor=self
            )
            
            # For validation, we don't need augmentation, just basic transforms
            val_transforms = self.create_transforms(is_train=False)
            val_dataset = BreastThermographyDataset(
                val_df,
                transform=val_transforms,
                config=self.config,
                data_preprocessor=self
            )
            
            # Log dataset details
            self.logger.info("\nDATASET DETAILS:")
            self.logger.info(f"Training samples: {len(train_dataset)}")
            self.logger.info(f"Validation samples: {len(val_dataset)}")
            self.logger.info(f"Number of classes: {len(self.class_to_folder)}")
            self.logger.info(f"Class to folder mapping: {self.class_to_folder}")
            
            # Save processed data if requested
            if save_processed:
                self.logger.info("\nSaving processed data...")
                self.dataset = train_dataset  # Store reference for saving
                self.save_processed_data(train_df)
                self.logger.info("Processed data saved successfully.")
            
            # Create data loaders with appropriate settings
            batch_size = self.config['training']['batch_size']
            num_workers = self.config['data'].get('num_workers', 4)
            pin_memory = self.config['data'].get('pin_memory', True)
            
            # Create weighted sampler for training
            train_labels = train_df['label'].values
            class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[np.where(np.unique(train_labels) == t)[0][0]] for t in train_labels])
            samples_weight = torch.from_numpy(samples_weight).double()
            
            sampler = torch.utils.data.WeightedRandomSampler(
                samples_weight,
                len(samples_weight),
                replacement=True
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                persistent_workers=num_workers > 0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0
            )
            
            # Log data loader details
            self.logger.info(f"\nDATA LOADER CONFIGURATION:")
            self.logger.info(f"Batch size: {batch_size}")
            self.logger.info(f"Number of workers: {num_workers}")
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            self.logger.info("\n" + "="*50)
            self.logger.info("DATASET CREATION COMPLETE")
            self.logger.info("="*50 + "\n")
            
            return train_loader, val_loader, df
            
        except Exception as e:
            self.logger.error(f"Error creating datasets: {str(e)}")
            raise

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
