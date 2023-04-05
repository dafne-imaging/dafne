[![PyPI version](https://badge.fury.io/py/dafne.svg)](https://badge.fury.io/py/dafne)
[![PDF Documentation](https://img.shields.io/badge/Docs-pdf-brightgreen)](https://www.dafne.network/files/documentation.pdf)
[![HTML Documentation](https://img.shields.io/badge/Docs-html-brightgreen)](https://www.dafne.network/documentation/)

# Dafne
Deep Anatomical Federated Network is a program for the segmentation of medical images. It relies on a server to provide deep learning models to aid the segmentation, and incremental learning is used to improve the performance. See https://www.dafne.network/ for documentation and user information.

## Windows binary installation
Please install the Visual Studio Redistributable Package under windows: https://aka.ms/vs/16/release/vc_redist.x64.exe
Then, run the provided installer

## Mac binary installation
Install the Dafne App from the downloaded .dmg file as usual. Make sure to download the archive appropriate for your architecture (x86 or arm).

## Linux binary installation
The Linux distribution is a self-contained executable file. Simply download it, make it executable, and run it.

## pip installation
Dafne can also be installed with pip
`pip install dafne`

# Citing
If you are writing a scientific paper, and you used Dafne for your data evaluation, please cite the following paper:

> Santini F, Wasserthal J, Agosti A, et al. *Deep Anatomical Federated Network (Dafne): an open client/server framework for the continuous collaborative improvement of deep-learning-based medical image segmentation*. 2023 doi: [10.48550/arXiv.2302.06352](https://doi.org/10.48550/arXiv.2302.06352).


# Notes for developers

## dafne

Run: 
`python dafne.py <path_to_dicom_img>`


## Notes for the DL models

### Apply functions
The input of the apply function is:
```
dict({
    'image': np.array (2D image)
    'resolution': sequence with two elements (image resolution in mm)
    'split_laterality': True/False (indicates whether the ROIs should be split in L/R if applicable)
    'classification': str - The classification tag of the image (optional, to identify model variants)
})
```

The output of the classifier is a string.
The output of the segmenters is:
```
dict({
    roi_name_1: np.array (2D mask),
    roi_name_2: ...
})
``` 

### Incremental learn functions
The input of the incremental learn functions are:
```
training data: dict({
    'resolution': sequence (see above)
    'classification': str (see above)
    'image_list': list([
        - np.array (2D image)
        - np.array (2D image)
        - ...
    ])
})

training outputs: list([
    - dict({
        roi_name_1: np.array (2D mask)
        roi_name_2: ...
    })
    - dict...
```

Every entry in the training outputs list corresponds to an entry in the image_list inside the training data.
So `len(training_data['image_list']) == len(training_outputs)`.

# Acknowledgments
Input/Output is based on [DOSMA](https://github.com/ad12/DOSMA) - GPLv3 license

Other packages required for this project are listed in requirements.txt
