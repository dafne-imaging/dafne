# Notes for developers

Remember to `git push --recurse-submodules` if you change the dl folder!

Install dependencies for pypotrace  
`sudo apt-get install build-essential python-dev libagg-dev libpotrace-dev pkg-config`

# dafne

Run: 
`python dafne.py <path_to_dicom_img>`

To run with server backend:  
Edit `config.txt` to use the right server. Then run dafne with RemoteModelProvider:  
`python dafne.py <path_to_dicom_img> -rm`


# Notes for the DL models

## Apply functions
The input of the apply function is:
```
dict({
    'image': np.array (2D image)
    'resolution': sequence with two elements (image resolution in mm)
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

## Incremental learn functions
The input of the incremental learn functions are:
```
training data: dict({
    'resolution': sequence (see above)
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
Code for converting dicom to nifti is based on https://github.com/icometrix/dicom2nifti, copyright Icometrix, MIT License.