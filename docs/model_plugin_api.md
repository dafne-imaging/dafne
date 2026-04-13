# Dafne Model Plugin API Reference

This document describes the API for creating model plugins for the Dafne segmentation framework. Models are Python objects that follow a specific interface and are serialized along with their functions. The framework supports Keras-based (`DynamicDLModel`), PyTorch-based (`DynamicTorchModel`), PyTorch ensemble (`DynamicEnsembleModel`), and FLARE-style (`DynamicTorchFLAREModel`) models, operating on either 2D or 3D data.

## Architecture Overview

A model plugin consists of **top-level functions** that are passed as arguments to a model class constructor. These functions are serialized (via `dill`) along with the model weights, so they must define all their imports locally within the function body. The model object is then saved as a `.model` file and distributed to users.

Models are registered and packaged via the `generate_convert()` utility function in `dafne-models`.

## Model Classes

| Class | Backend | `self.model` type | Default dimensionality | Source |
|---|---|---|---|---|
| `DynamicDLModel` | Keras/TF | Single Keras model | 2 | `dafne_dl/DynamicDLModel.py` |
| `DynamicTorchModel` | PyTorch | Single `nn.Module` | 2 | `dafne_dl/DynamicTorchModel.py` |
| `DynamicEnsembleModel` | PyTorch | List of `nn.Module` | 3 | `dafne_dl/DynamicEnsembleModel.py` |
| `DynamicTorchFLAREModel` | PyTorch | Single `nn.Module` | 2 | `dafne_dl/DynamicTorchFLAREModel.py` |

All classes inherit from the abstract base class `DeepLearningClass` (defined in `dafne_dl/interfaces.py`).

## Required Functions

When creating a model plugin, you must provide at minimum three functions:

1. **`init_model_function`** -- Creates and returns the model architecture
2. **`apply_model_function`** -- Runs inference on input data
3. **`incremental_learn_function`** (optional) -- Performs incremental learning from user corrections

### Important: Imports must be local

All top-level functions are serialized as source code. **All imports must be inside the function body**, not at the module top level:

```python
# CORRECT
def model_apply(modelObj, data):
    import numpy as np
    from scipy.ndimage import zoom
    # ...

# WRONG - will fail on deserialization
import numpy as np
def model_apply(modelObj, data):
    # np won't be available when loaded
```

---

## Function Signatures

### `init_model_function() -> model`

Creates and returns the neural network model. Takes no parameters.

- For `DynamicDLModel`: returns a Keras model
- For `DynamicTorchModel` / `DynamicTorchFLAREModel`: returns a single `torch.nn.Module` (will be automatically moved to the appropriate device via `.to(device)`)
- For `DynamicEnsembleModel`: returns a **tuple or list** of `torch.nn.Module` objects (each will be moved to device)

```python
def init_unet():
    from monai.networks.nets import UNet
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
    )
    return model
```

---

### `apply_model_function(modelObj, data: dict) -> dict`

Runs inference. This is the core function of the plugin.

**Parameters:**
- `modelObj` -- The model wrapper object. Access the underlying neural network via `modelObj.model`. For PyTorch models, `modelObj.device` gives the torch device.
- `data` -- A dictionary containing the input data (see below for keys).

**Return value:**
- A dictionary mapping **label names** (str) to **binary masks** (numpy arrays, dtype `uint8`, same spatial dimensions as the input image).

#### `data` dictionary keys -- 2D segmentation models

| Key | Type | Description |
|---|---|---|
| `'image'` | `np.ndarray` (H, W) | 2D grayscale image slice |
| `'resolution'` | array-like, length 2 | Pixel spacing in mm: `[spacing_x, spacing_y]` |
| `'split_laterality'` | `bool` | Whether to split left/right structures |
| `'classification'` | `str` | Body region classification label (e.g. `'Thigh'`, `'Leg'`) |

#### `data` dictionary keys -- 3D segmentation models

| Key | Type | Description |
|---|---|---|
| `'image'` | `np.ndarray` (H, W, D), dtype `float32` | 3D image volume |
| `'affine'` | `np.ndarray` (4, 4), dtype `float32` | Affine transformation matrix (NIfTI-style) |
| `'resolution'` | array-like, length 3 | Voxel spacing in mm: `[spacing_x, spacing_y, spacing_z]` |
| `'split_laterality'` | `bool` | Always `False` for 3D models |
| `'classification'` | `str` | Body region classification label |

#### Additional optional `data` dictionary keys (all model types)

Any model (2D or 3D) may additionally receive the following keys in the `data` dictionary. Currently, multi-contrast images are only used by FLARE models and model options are not yet exposed in the Dafne user interface, but the API supports them for all model types.

| Key | Type | Description |
|---|---|---|
| `'image2'` | `np.ndarray` | Second contrast/channel (same spatial shape as `'image'`) |
| `'image3'` | `np.ndarray` | Third contrast/channel, and so on (`'imageN'`) |
| `'options'` | `dict` | Model-specific options (e.g., `{'threshold': 0.4, 'min_lesion_size': 3}`) |

Models that require multiple contrasts should check for the presence of the additional `'imageN'` keys and raise an informative error if they are missing. The available options should be declared in the model's metadata under `'options'`.

#### Example -- 2D apply function

```python
def model_apply(modelObj, data: dict):
    import numpy as np

    LABELS_DICT = {1: 'LK', 2: 'RK'}
    MODEL_RESOLUTION = [1.25, 1.25]

    image = data['image']
    resolution = data['resolution']

    # Resample to model resolution
    zoom_factors = [resolution[i] / MODEL_RESOLUTION[i] for i in range(2)]
    from scipy.ndimage import zoom
    image_resampled = zoom(image, zoom_factors, order=3)

    # Normalize
    image_resampled = (image_resampled - image_resampled.mean()) / (image_resampled.std() + 1e-8)

    # Run inference
    model = modelObj.model
    # ... prepare input tensor, run model ...
    prediction = model.predict(input_tensor)

    # Convert predictions to label masks and resample back
    outputLabels = {}
    for label_idx, label_name in LABELS_DICT.items():
        mask = (prediction == label_idx).astype(np.uint8)
        mask = zoom(mask, [1.0/z for z in zoom_factors], order=0)
        outputLabels[label_name] = mask

    return outputLabels
```

---

### `incremental_learn_function(modelObj, trainingData: dict, trainingOutputs: list, bs: int, minTrainImages: int)`

Performs incremental (online) learning from user-corrected segmentations. Optional -- if not provided, the model's `can_incremental_learn()` returns `False`.

**Parameters:**
- `modelObj` -- The model wrapper object.
- `trainingData` -- A dictionary (see below).
- `trainingOutputs` -- A list of dictionaries, each mapping label names (str) to binary masks (numpy arrays, `uint8`). One entry per training image.
- `bs` -- Batch size (default: `5` for 2D, `1` for 3D ensembles).
- `minTrainImages` -- Minimum number of training images required before learning is performed.

#### `trainingData` keys -- 2D models

| Key | Type | Description |
|---|---|---|
| `'image_list'` | `list[np.ndarray]` | List of 2D images |
| `'resolution'` | array-like, length 2 | Pixel spacing in mm |
| `'classification'` | `str` | Body region classification label |

#### `trainingData` keys -- 3D models

| Key | Type | Description |
|---|---|---|
| `'image_list'` | `list[np.ndarray]` | List of 3D volumes (float32) |
| `'affine'` | `list[np.ndarray]` | List of 4x4 affine matrices (float32), one per volume |
| `'resolution'` | array-like, length 3 | Voxel spacing in mm |
| `'classification'` | `str` | Body region classification label |

---

## FLARE Model Additional Functions

`DynamicTorchFLAREModel` extends `DynamicTorchModel` with three additional optional functions and a different learning interface:

### `data_preprocess_function(modelObj, data: dict) -> tensor`

Preprocesses the input data dictionary into a tensor suitable for the model. Called before inference.

### `data_postprocess_function(modelObj, data)`

Postprocesses the model output (currently optional, not used in existing models).

### `label_preprocess_function(modelObj, label_dictionary: dict) -> tensor`

Converts a label dictionary (label name -> mask) into a tensor for training.

### `learn(train_dataset, validation_dataset, options=None)`

FLARE models use a `learn()` method instead of `incremental_learn()`. This takes dataset objects and an options dictionary rather than raw image lists. The `incremental_learn_function` provided to the constructor is called with `(modelObj, train_dataset, validation_dataset, options)`.

---

## Model Registration with `generate_convert()`

Models are packaged using the `generate_convert()` function from `dafne_models.common`:

```python
from dafne_models.common import generate_convert
from dafne_dl.DynamicTorchModel import DynamicTorchModel

generate_convert(
    model_id='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx',  # Unique UUID
    default_weights_path='weights/',                    # Path to weight files
    model_name_prefix='MyModel',                        # Used for output filename
    model_create_function=init_model,                   # init_model_function
    model_apply_function=model_apply,                   # apply_model_function
    model_learn_function=model_learn,                   # incremental_learn_function (or None)
    dimensionality=2,                                   # 2 or 3
    model_type=DynamicTorchModel,                       # Model class to use
    metadata=metadata                                   # Metadata dictionary
)
```

For FLARE models, additional keyword arguments are available:

```python
generate_convert(
    ...,
    model_type=DynamicTorchFLAREModel,
    data_preprocess_function=preprocess,
    data_postprocess_function=None,
    label_preprocess_function=label_preprocess,
)
```

---

## Metadata Dictionary

The metadata dictionary provides information about the model for the UI and dependency management:

```python
metadata = {
    'categories': [['MSK', 'Muscle', 'Lower limbs']],  # Hierarchical category path
    'variants': ['', 'Left', 'Right'],                  # Model variants
    'dimensionality': '3',                               # '2' or '3' (string)
    'model_name': 'MyModel',                             # Display name
    'model_type': 'DynamicTorchModel',                   # Class name as string
    'orientation': 'Axial',                              # Expected orientation (or '')
    'info': {
        'Description': 'Model description',
        'Author': 'Author name',
        'Modality': 'MRI',
    },
    'dependencies': {                                    # pip package requirements
        'monai': 'monai >= 1.3.0',
        'einops': 'einops',
    },
    'options': {                                         # For FLARE models: user-configurable options
        'threshold': 'float',
        'min_lesion_size': 'int',
    },
    'description': 'Free-text description of the model.',
}
```

The `get_metadata()` method on the base class automatically adds `dimensionality`, `type`, and `can_incremental_learn` fields, so these are available at runtime even if not set in the initial metadata.

---

## Summary of Model Types

### 2D Segmentation Model (DynamicDLModel or DynamicTorchModel)

- `data_dimensionality = 2`
- `apply` receives: `{'image': (H,W), 'resolution': [sx, sy], 'split_laterality': bool, 'classification': str}`
- `apply` returns: `{'LabelName': np.ndarray(H,W, uint8), ...}`
- `incremental_learn` receives: `{'image_list': [...], 'resolution': [sx, sy], 'classification': str}`

### 3D Segmentation Model (DynamicTorchModel or DynamicEnsembleModel)

- `data_dimensionality = 3`
- `apply` receives: `{'image': (H,W,D) float32, 'affine': (4,4) float32, 'resolution': [sx, sy, sz], 'split_laterality': False, 'classification': str}`
- `apply` returns: `{'LabelName': np.ndarray(H,W,D, uint8), ...}`
- `incremental_learn` receives: `{'image_list': [...], 'affine': [...], 'resolution': [sx, sy, sz], 'classification': str}`

### FLARE Model (DynamicTorchFLAREModel)

- Extends `DynamicTorchModel` with `data_preprocess_function`, `data_postprocess_function`, `label_preprocess_function`
- `apply` receives: standard dict (may include additional contrasts `'image2'`, `'image3'`, ... and `'options'`)
- `apply` returns: `{'LabelName': np.ndarray(H,W,D, uint8), ...}`
- Training via `learn(train_dataset, validation_dataset, options)` instead of `incremental_learn`

**Note:** All model types may receive additional contrasts (`'image2'`, `'image3'`, ...) and an `'options'` dict. Currently, multi-contrast input is only implemented for FLARE models, and model options are not yet exposed in the Dafne user interface.