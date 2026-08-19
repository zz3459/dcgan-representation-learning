# ------------------------------------------------------------------------------
# File: data_utils.py
# Author: Natel Putinati Diego (UNI: dn2659)
# ------------------------------------------------------------------------------

"""
data_utils.py

- Load and preprocess image datasets for DCGAN training
- Normalize images to [-1, 1] range for tanh activation
- Support for training datasets (LSUN, ImageNet, Faces) and evaluation datasets (CIFAR-10, SVHN, MNIST)
- Batch size: 128 for all datasets
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
import hashlib
import os


### Weight Initialization Configuration
## All weights from N(0, 0.02) as per DCGAN paper

WEIGHT_INIT_CONFIG = {
    'distribution': 'normal',
    'mean': 0.0,
    'stddev': 0.02  
}


### Normalization Functions
def normalize_to_tanh_range(image):
    # Normalize image from [0, 255] to [-1, 1] for tanh activation
    if isinstance(image, tf.Tensor):
        image = tf.cast(image, tf.float32)
        return (image / 127.5) - 1.0
    else:
        return (image.astype(np.float32) / 127.5) - 1.0


def denormalize_from_tanh_range(tensor):
    # Inverse operation for visualization: maps [-1, 1] back to [0, 255]
    if isinstance(tensor, tf.Tensor):
        denorm = (tensor + 1.0) * 127.5
        return tf.cast(tf.clip_by_value(denorm, 0, 255), tf.uint8)
    else:
        denorm = ((tensor + 1.0) * 127.5).astype(np.float32)
        return np.clip(denorm, 0, 255).astype(np.uint8)

### LSUN Bedrooms Loader
## Output: 64×64 per DCGAN paper
## Includes deduplication step as per section 4.1.1

def _compute_semantic_hash_numpy(image_np):
    """Compute Average Hash (aHash) of image for semantic deduplication.
    
    This serves as a lightweight alternative to the Autoencoder-based 
    semantic hashing described in the DCGAN paper (Section 4.1.1).
    It detects near-duplicates by comparing structural features on a 
    downsampled 32x32 grayscale version of the image.
    """
    # Ensure image is 32x32 (input size per paper spec)
    # If input is not 32x32, we rely on the caller to have resized it
    # or we resize here (but for efficiency we assume 32x32 input from caller)
    
    # 1. Convert to grayscale (simple average of channels)
    if image_np.shape[-1] == 3: 
        gray = np.mean(image_np, axis=-1)
    else:
        gray = image_np.squeeze()
        
    # 2. Compute mean value
    avg = gray.mean()
    
    # 3. Threshold to create binary hash (1 if > avg, 0 otherwise)
    # This captures low-frequency structural information
    binary_map = (gray > avg).astype(np.uint8)
    
    # 4. Pack bits into hash string
    # Flatten and pack into bytes for efficient storage/comparison
    hash_bytes = np.packbits(binary_map.flatten()).tobytes()
    
    # Return as hex string
    return hashlib.md5(hash_bytes).hexdigest()


def _default_lsun_dedup_metadata_path(category: str, target_size) -> Path:
    """
    Default on-disk cache for LSUN deduplication keep-indices.

    We store a list of dataset indices (after resize, before normalization) that should be kept.
    This avoids expensive runtime hashing on every run and makes deduplication reproducible
    across machines, assuming the underlying TFDS example order is stable.
    """
    repo_root = Path(__file__).resolve().parent
    h, w = int(target_size[0]), int(target_size[1])
    return repo_root / "task1" / "data" / "metadata" / f"lsun_{category}_{h}x{w}_keep_indices.npy"


def _filter_by_keep_indices(dataset: tf.data.Dataset, keep_indices: np.ndarray) -> tf.data.Dataset:
    """
    Filter an *unbatched* dataset by keeping only elements whose enumerate()-index is in keep_indices.

    This is the fast-path used when dedup metadata is available.
    """
    keep_indices = np.asarray(keep_indices, dtype=np.int64).reshape([-1])
    keys = tf.constant(keep_indices, dtype=tf.int64)
    vals = tf.ones_like(keys, dtype=tf.bool)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
        default_value=False,
    )

    ds = dataset.enumerate()
    ds = ds.filter(lambda i, x: table.lookup(i))
    ds = ds.map(lambda i, x: x, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def _compute_dedup_keep_indices(dataset_uint8: tf.data.Dataset, target_size) -> np.ndarray:
    """
    One-time (still expensive) computation of dedup keep-indices.

    This iterates deterministically over the dataset and records the first index for each unique
    semantic hash. The resulting keep-indices can be saved and reused.
    """
    seen = set()
    keep = []
    for idx, image in dataset_uint8.enumerate():
        image_np = image.numpy()
        # Create 32x32 version for semantic hashing (per paper spec)
        if image_np.shape[0] != 32 or image_np.shape[1] != 32:
            if image_np.shape[0] == 64 and image_np.shape[1] == 64:
                image_small = image_np[::2, ::2, :]
            else:
                image_small = image_np
        else:
            image_small = image_np

        h = _compute_semantic_hash_numpy(image_small)
        if h not in seen:
            seen.add(h)
            keep.append(int(idx.numpy()))
    return np.asarray(keep, dtype=np.int64)


def _deduplicate_dataset(dataset, target_size):
    """Remove duplicate images from dataset based on semantic hash.
    
    Fixed to use a streaming generator to prevent OOM errors on large datasets (LSUN).
    Maintains a set of seen hashes in memory (~100MB for 3M images) but streams 
    the images themselves.
    """
    
    def generator():
        # Keep track of seen hashes (RAM usage: ~32 bytes * 3M approx 100MB)
        seen_hashes = set()
        
        # Iterate over the dataset one by one
        for image in dataset:
            # Convert to numpy for hashing
            image_np = image.numpy()
            
            # Create 32x32 version for semantic hashing (per paper spec)
            if image_np.shape[0] != 32 or image_np.shape[1] != 32:
                # Simple subsampling for efficiency (64->32 is 2x)
                if image_np.shape[0] == 64 and image_np.shape[1] == 64:
                     image_small = image_np[::2, ::2, :]
                else:
                     image_small = image_np
            else:
                image_small = image_np
                
            # Compute semantic hash
            hash_str = _compute_semantic_hash_numpy(image_small)
            
            # Yield only if new
            if hash_str not in seen_hashes:
                seen_hashes.add(hash_str)
                yield image_np

    # Create a new dataset from the generator
    # We use the element_spec from the input dataset to ensure types match
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=dataset.element_spec
    )


def load_lsun_images(
    category='bedroom',
    shuffle_buffer=None,
    target_size=(64, 64),
    shuffle=True,
    seed=None,
    reshuffle_each_iteration=True,
    deduplicate=True,
    data_dir=None,
    dedup_metadata_path=None,
    dedup_cache_write=True,
):
    """
    Load LSUN images as an *unbatched* tf.data.Dataset of float32 tensors normalized to [-1, 1].

    This is useful when you need to take a deterministic subset by example count (e.g., scaling studies)
    before batching. For standard training, use `load_lsun()` which returns a batched dataset.
    """
    assert target_size == (64, 64), "LSUN output must be 64x64 per specification"

    try:
        ds = tfds.load(f'lsun/{category}', split='train', as_supervised=False)

        def extract_image(example):
            return example['image']

        dataset = ds.map(extract_image, num_parallel_calls=tf.data.AUTOTUNE)
        source = f"tfds:lsun/{category}"
    except Exception as e:
        # Simple fallback: load LSUN from a local folder if TFDS LSUN isn't available.
        # Expected layout:
        #   <LSUN_DIR>/<category>/**/image.(jpg|png|...)
        # Where <LSUN_DIR> can be:
        # - passed explicitly via data_dir
        # - set via env var LSUN_DIR
        # - defaulted to repo_root/data/lsun
        repo_root = Path(__file__).resolve().parent
        base = Path(data_dir) if data_dir else Path(os.environ.get("LSUN_DIR", ""))
        if str(base) == "":
            base = repo_root / "data" / "lsun"
        candidates = [
            base / category,
            repo_root / "task1" / "data" / "lsun" / category,
        ]
        local_category_dir = next((p for p in candidates if p.exists()), None)
        if local_category_dir is None:
            raise ValueError(
                "Error loading dataset. TFDS could not load LSUN and no local LSUN directory was found.\n"
                f"- TFDS error: {type(e).__name__}: {e}\n"
                "- To use local LSUN, set env var LSUN_DIR to a folder containing '<category>/' (e.g. 'bedroom/')\n"
                f"- Tried: {', '.join(str(p) for p in candidates)}"
            )

        # image_dataset_from_directory returns batches; unbatch to match the rest of our pipeline.
        local_ds = tf.keras.preprocessing.image_dataset_from_directory(
            str(local_category_dir),
            label_mode=None,
            image_size=target_size,
            batch_size=64,
            shuffle=False,  # deterministic file order; we shuffle later in tf.data
        ).unbatch()
        dataset = local_ds
        source = f"local_dir:{local_category_dir}"

    # Helpful one-line trace so users know what backend they are using.
    try:
        tf.print("[LSUN] loaded", source)
    except Exception:
        pass

    # Resize to 64x64
    def resize_image(image):
        return tf.image.resize(image, target_size, method='bilinear')

    # Important: ensure deterministic ordering for any index-based or "first occurrence wins" logic.
    # We keep the deterministic settings only through the dedup stage.
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    dataset = dataset.map(resize_image, num_parallel_calls=1)

    # Deduplication step (as per section 4.1.1)
    if deduplicate:
        # Convert to uint8 for consistent hashing before deduplication
        def to_uint8(image):
            image = tf.clip_by_value(image, 0.0, 255.0)
            return tf.cast(tf.round(image), tf.uint8)

        dataset_uint8 = dataset.map(to_uint8, num_parallel_calls=1)

        metadata_path = Path(dedup_metadata_path) if dedup_metadata_path else _default_lsun_dedup_metadata_path(category, target_size)

        if metadata_path.exists():
            keep_indices = np.load(metadata_path)
            dataset = _filter_by_keep_indices(dataset_uint8, keep_indices)
        else:
            if dedup_cache_write:
                # One-time cost: compute keep indices and save them for future runs.
                print(f"[dedup] Metadata not found at {metadata_path}. Computing keep-indices once and caching...")
                keep_indices = _compute_dedup_keep_indices(dataset_uint8, target_size)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(metadata_path, keep_indices)
                dataset = _filter_by_keep_indices(dataset_uint8, keep_indices)
                print(f"[dedup] Cached keep-indices to {metadata_path} (kept {len(keep_indices)} samples).")
            else:
                # Fallback to legacy runtime hashing without caching.
                print(f"[dedup][WARN] Metadata not found at {metadata_path}. Falling back to runtime hashing (slow).")
                dataset = _deduplicate_dataset(dataset_uint8, target_size)

        dataset = dataset.map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=tf.data.AUTOTUNE)

    # Normalize to [-1, 1]
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)

    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000
        dataset = dataset.shuffle(
            shuffle_buffer,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_lsun(category='bedroom', batch_size=128, shuffle_buffer=None, 
              target_size=(64, 64), shuffle=True, deduplicate=True,
              dedup_metadata_path=None, dedup_cache_write=True):

    # Pipeline: Load → Resize to 64×64 → Deduplicate → Normalize to [-1, 1] → Batch of 128
    assert batch_size == 128, "Batch size must be 128 per specification"
    assert target_size == (64, 64), "LSUN output must be 64x64 per specification"

    dataset = load_lsun_images(
        category=category,
        shuffle_buffer=shuffle_buffer,
        target_size=target_size,
        shuffle=shuffle,
        seed=None,
        reshuffle_each_iteration=True,
        deduplicate=deduplicate,
        dedup_metadata_path=dedup_metadata_path,
        dedup_cache_write=dedup_cache_write,
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset










# Task3 part
# use the miniimagenet dataset with image size 64*64

### ImageNet-1k Loader
## Output: 32×32 with min-resize + center crop

def imagenet_min_resize_center_crop(image, target_size=(32, 32)):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    
    # Min-resize: preserve aspect ratio, resize so min dimension = 32
    min_dim = tf.minimum(h, w)
    scale = target_size[0] / min_dim
    
    new_h = tf.cast(h * scale, tf.int32)
    new_w = tf.cast(w * scale, tf.int32)
    
    resized = tf.image.resize(image, [new_h, new_w], method='bilinear')
    
    crop_size = target_size[0]
    offset_h = (new_h - crop_size) // 2
    offset_w = (new_w - crop_size) // 2
    
    cropped = tf.image.crop_to_bounding_box(
        resized, offset_h, offset_w, crop_size, crop_size
    )
    
    return cropped


# def load_imagenet(batch_size=128, shuffle_buffer=None, target_size=(32, 32), 
#                   subset='train', shuffle=True):
#     # Pipeline: Load → Min-resize (preserve aspect ratio) → Center crop to 32×32 → Normalize → Batch of 128
#     assert batch_size == 128, "Batch size must be 128 per specification"
#     assert target_size == (32, 32), "ImageNet output must be 32x32 per specification"
    
#     try:
#         ds = tfds.load('imagenet_resized/32x32', split=subset, as_supervised=False, 
#                       download=False)
        
#         def extract_image(example):
#             image = example['image']
#             return image
        
#         dataset = ds.map(extract_image)
        
#     except Exception:
#         raise ValueError("Error loading dataset")
    
#     # Apply min-resize and center crop
#     dataset = dataset.map(
#         lambda x: imagenet_min_resize_center_crop(x, target_size),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )
    
#     # Normalize to [-1, 1]
#     def normalize_image(image):
#         image = tf.cast(image, tf.float32)
#         return normalize_to_tanh_range(image)
    
#     dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
#     # Shuffle and batch
#     if shuffle:
#         if shuffle_buffer is None:
#             shuffle_buffer = 10000 
#         dataset = dataset.shuffle(shuffle_buffer)
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
#     return dataset


'''
Date: <09/11/2025>
Written by: <Ziyi Zhao> <zz3459@columbia.edu>

The code in this file was fully implemented by the student.
It has not been generated by AI tools, and it has not been copied 
from any external resource.
'''


def load_local_imagenet(data_dir, batch_size=128, shuffle_buffer=10000, 
                        target_size=(32, 32), shuffle=True):
    """
    Modified loader: Specifically for loading local ImageNet folders.
    Pipeline: Load -> Resize -> Center Crop -> Normalize -> Batch
    """
    print(f"Loading local dataset from: {data_dir}")

    try:
        # Set image_size to (64, 64)
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            label_mode=None,       # Unsupervised learning
            image_size=(64, 64),   
            batch_size=batch_size,      
            shuffle=shuffle
        )
    except Exception as e:
        raise ValueError(f"Error reading directory: {e}")

    # We must unbatch to process images individually to meet the resize function.
    dataset = dataset.unbatch()
    
    # tf.image.resize will work correctly now that input has a fixed shape
    dataset = dataset.map(
        lambda x: imagenet_min_resize_center_crop(x, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Normalize to [-1, 1]
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and Batch
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset








'''
Date: <09/11/2025>
Written by: <Ziyi Zhao> <zz3459@columbia.edu>

The code in this file was fully implemented by the student.
It has not been generated by AI tools, and it has not been copied 
from any external resource.
'''





# Task3 part
# use cifar10 dataset to test accuracy
# Here we use simple coding because cifar10 dataset with original size 32,32 and we do only need to normalize and flatten labels
def load_cifar10_task3(target_size=(32, 32)):
    """
    Load CIFAR-10 dataset specifically for Section 5.1 Evaluation.
    Returns normalized Numpy arrays with labels, ready for SVM classification.
    
    Returns:
        (x_train, y_train), (x_test, y_test)
        - x data: normalized to [-1, 1] for Tanh compatibility
        - y data: flattened 1D arrays
    """
    import tensorflow as tf
    import numpy as np

    print("Loading CIFAR-10 for Supervised Linear Classification (Evaluation only)")
    
    # Load Data with Labels
    # [cite_start]We need labels (y) to train the Linear SVM classifier [cite: 161]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize to [-1, 1]
    # The Discriminator expects inputs in [-1, 1] range.
    x_train = (x_train / 127.5) - 1.0
    x_test = (x_test / 127.5) - 1.0

    # Flatten Labels
    # Scikit-learn expects 1D arrays for labels, not (N, 1)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print(f"Data Loaded: Train {x_train.shape}, Test {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)




def load_svhn_task3(target_size=(32, 32)):
    """
    Load SVHN dataset specifically for Section 5.2 Evaluation.
    Uses TFDS 'svhn_cropped' dataset (32x32 color digits).
    
    Returns normalized Numpy arrays with labels, ready for SVM classification.

    Returns:
        (x_train, y_train), (x_test, y_test)
        - x data: normalized to [-1, 1] for DCGAN Discriminator compatibility
        - y data: flattened 1D arrays (0–9)
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import numpy as np

    print("Loading SVHN (svhn_cropped) for Supervised Linear Classification.")

    # Load TFDS dataset (images already 32x32)
    ds_train = tfds.load("svhn_cropped", split="train", as_supervised=True)
    ds_test = tfds.load("svhn_cropped", split="test", as_supervised=True)

    # Convert to numpy arrays
    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    for img, label in tfds.as_numpy(ds_train):
        x_train_list.append(img)
        y_train_list.append(label)

    for img, label in tfds.as_numpy(ds_test):
        x_test_list.append(img)
        y_test_list.append(label)

    x_train = np.array(x_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)

    x_test = np.array(x_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    # Normalize to [-1, 1]
    x_train = (x_train / 127.5) - 1.0
    x_test = (x_test / 127.5) - 1.0

    # Flatten labels to 1D array
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print(f"SVHN Loaded: Train {x_train.shape}, Test {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)







### Faces Dataset Loader (CelebA)
## Output: 64×64 (CelebA images are already aligned/cropped faces)

def load_faces_dataset(data_dir, batch_size=128, shuffle_buffer=None, 
                       target_size=(64, 64), shuffle=True):

    # Simplified pipeline for CelebA: Load image → Resize to 64×64 → Normalize → Batch of 128
    assert batch_size == 128, "Batch size must be 128 per specification"
    assert target_size == (64, 64), "Faces output must be 64x64 per specification"
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError("Error loading dataset")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_path.glob(ext)))
        image_files.extend(list(data_path.glob(ext.upper())))
    
    if len(image_files) == 0:
        raise ValueError("Error loading dataset")
    
    # Create dataset from file paths
    file_paths = [str(f) for f in image_files]
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    def load_and_preprocess_image(file_path):
        # Read image
        image = tf.io.read_file(file_path)
        # Decode image (CelebA is JPEG)
        image = tf.image.decode_jpeg(image, channels=3)
        # Resize to 64x64
        image = tf.image.resize(image, target_size, method='bilinear')
        # Convert to uint8 (resize outputs float32)
        image = tf.cast(image, tf.uint8)
        return image
    
    # Load and preprocess images
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Normalize to [-1, 1]
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


### Evaluation Dataset Loaders
## These datasets are for evaluation only, not for training DCGAN (Important!)

def load_cifar10(batch_size=128, shuffle_buffer=None, target_size=(32, 32), shuffle=True):
    # Load and preprocess CIFAR-10 dataset for evaluation (feature extraction)
    # Standard CIFAR-10, NOT for training. Used for Table 1 results. Shape: (32, 32, 3)
    assert batch_size == 128, "Batch size must be 128 per specification"
    assert target_size == (32, 32), "CIFAR-10 output must be 32x32 per specification"
    
    # Load CIFAR-10 dataset
    (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def load_svhn(batch_size=128, shuffle_buffer=None, target_size=(32, 32), 
              split='train', num_labeled=None, shuffle=True):
    # Load and preprocess SVHN dataset for semi-supervised evaluation
    # Paper spec: Validation 10k from non-extra set, Labeled samples 1k uniformly distributed
    assert batch_size == 128, "Batch size must be 128 per specification"
    assert target_size == (32, 32), "SVHN output must be 32x32 per specification"
    
    # Load SVHN dataset using tensorflow_datasets
    try:
        if split == 'validation':
            # Take 10,000 from non-extra set (train split, first 10k)
            ds = tfds.load('svhn_cropped', split='train[:10000]', as_supervised=True)
        elif split == 'train':
            # Full training set (non-extra)
            ds = tfds.load('svhn_cropped', split='train', as_supervised=True)
        elif split == 'test':
            ds = tfds.load('svhn_cropped', split='test', as_supervised=True)
        else:
            raise ValueError("Error loading dataset")
        
        # Extract images and labels
        def extract_image_label(example):
            image, label = example
            return image, label
        
        dataset = ds.map(extract_image_label)
        
        if num_labeled is not None:
            # Group by label and sample uniformly
            # This requires collecting all data first (for small labeled sets)
            images_list = []
            labels_list = []
            
            for image, label in dataset:
                images_list.append(image.numpy() if hasattr(image, 'numpy') else image)
                labels_list.append(label.numpy() if hasattr(label, 'numpy') else label)
            
            images_array = np.array(images_list)
            labels_array = np.array(labels_list)
            
            # Sample uniformly across classes (10 classes for SVHN: digits 0-9)
            num_classes = 10
            samples_per_class = num_labeled // num_classes
            selected_indices = []
            
            for class_id in range(num_classes):
                class_indices = np.where(labels_array == class_id)[0]
                if len(class_indices) > 0:
                    
                    np.random.seed(42)  
                    selected = np.random.choice(
                        class_indices, 
                        size=min(samples_per_class, len(class_indices)),
                        replace=False
                    )
                    selected_indices.extend(selected)
            
            
            selected_images = images_array[selected_indices]
            dataset = tf.data.Dataset.from_tensor_slices(selected_images)
        else:
            # Extract only images (no labels needed for unsupervised training)
            dataset = dataset.map(lambda img, lbl: img)
        
    except Exception:
        raise ValueError("Error loading dataset")
    
    
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def load_mnist(batch_size=128, shuffle_buffer=None, target_size=(28, 28), shuffle=True):
    # Load and preprocess MNIST dataset for conditional GAN evaluation
    # Converts grayscale to RGB (3 channels) and normalizes to [-1, 1]
    assert batch_size == 128, "Batch size must be 128 per specification"
    
    # Load MNIST dataset
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    
    # Preprocess: Convert to 3 channels and normalize
    def preprocess_mnist(image):
        # Expand to 3 channels (grayscale to RGB)
        image = tf.expand_dims(image, axis=-1)
        image = tf.repeat(image, 3, axis=-1)
        # Normalize to [-1, 1]
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


### Convenience Function

def get_dataset(dataset_name='cifar10', **kwargs):
    # Convenience function to get dataset by name
    # Training: 'lsun', 'imagenet', 'faces'
    # Evaluation: 'cifar10', 'svhn', 'mnist'
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == 'lsun':
        return load_lsun(**kwargs)
    elif dataset_name_lower == 'imagenet':
        return load_imagenet(**kwargs)
    elif dataset_name_lower == 'faces':
        if 'data_dir' not in kwargs:
            raise ValueError("Error loading dataset")
        return load_faces_dataset(**kwargs)
    elif dataset_name_lower == 'cifar10':
        return load_cifar10(**kwargs)
    elif dataset_name_lower == 'svhn':
        return load_svhn(**kwargs)
    elif dataset_name_lower == 'mnist':
        return load_mnist(**kwargs)
    else:
        raise ValueError("Error loading dataset")


if __name__ == "__main__":
    # Test CIFAR-10 loading
    dataset = load_cifar10(batch_size=128)
