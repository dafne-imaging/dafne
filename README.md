# Dafne
Deep Anatomical Federated Network is a program for the segmentation of medical images. It relies on a server to provide deep learning models to aid the segmentation, and incremental learning is used to improve the performance. See https://www.dafne.network/ for documentation and user information.

## Windows binary installation
Please install the Visual Studio Redistributable Package under windows: https://aka.ms/vs/16/release/vc_redist.x64.exe
Then, run the provided installer

## Mac binary installation
Decompress the .zip file and run the `dafne` program from the unzipped folder.

**Important note for Mac users:** if you download the zip file from github, the system will ask to enter security exceptions for every binary file included in the distribution, because the binaries are not signed. This is too much to do by hand. Either install Dafne from source, or temporarily disable the access control with the following procedure:
1. Open a terminal window.
2. Run the command `sudo spctl --master-disable` (it will ask for your password).
3. Run Dafne once by executing the `dafne` file.
4. Run the `calc_transforms` command as well.
4. Re-enable the protection by running the following command in a terminal: `sudo spctl --master-enable`

# Notes for developers

Remember to `git push --recurse-submodules` if you change the dl folder!

Install dependencies for pypotrace  
`sudo apt-get install build-essential python-dev libagg-dev libpotrace-dev pkg-config`

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

This repository contains a precompiled version of [potrace](http://potrace.sourceforge.net/) and [pypotrace](https://github.com/flupke/pypotrace), with parts of [AGG](http://agg.sourceforge.net/antigrain.com/index.html) - Licensed under GPL.

Other packages required for this project are listed in requirements.txt
