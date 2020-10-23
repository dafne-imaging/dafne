from .dicomUtils.dicom3D import save3dDicom
from scipy.spatial.distance import dice
import os
import numpy as np
import nibabel as nib


def calc_dice_score(mask1, mask2):
    return dice(mask1.ravel(), mask2.ravel())


def save_dicom_masks(base_path: str, mask_dict: dict, dicom_headers: list):
    for name, mask in mask_dict.items():
        dicom_path = os.path.join(base_path, name)
        try:
            os.mkdirs(dicom_path)
        except OSError:
            pass
        save3dDicom(mask, dicom_headers, dicom_path)


def save_npy_masks(base_path, mask_dict):
    for name, mask in mask_dict.items():
        npy_name = os.path.join(base_path, name + '.npy')
        np.save(npy_name, mask)


def save_npz_masks(filename, mask_dict):
    np.savez(filename, **mask_dict)


def save_nifti_masks(base_path, mask_dict, affine):
    for name, mask in mask_dict.items():
        nifti_name = os.path.join(base_path, name + '.nii.gz')
        niimg = nib.Nifti2Image(mask, affine)
        niimg.to_filename(nifti_name)
