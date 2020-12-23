from .dicomUtils.dicom3D import save3dDicom
import os
import numpy as np
import nibabel as nib


def calc_dice_score(mask1, mask2):
    a = 2 * np.sum(np.logical_and(mask1, mask2))
    b = np.sum(mask1) + np.sum(mask2)
    if b == 0:
        return 1
    else:
        return a/b


def save_dicom_masks(base_path: str, mask_dict: dict, dicom_headers: list):
    for name, mask in mask_dict.items():
        dicom_path = os.path.join(base_path, name)
        try:
            os.makedirs(dicom_path)
        except OSError:
            pass
        save3dDicom(mask, dicom_headers, dicom_path)


def save_npy_masks(base_path, mask_dict):
    for name, mask in mask_dict.items():
        npy_name = os.path.join(base_path, name + '.npy')
        np.save(npy_name, mask)


def save_npz_masks(filename, mask_dict):
    np.savez_compressed(filename, **mask_dict)


def save_nifti_masks(base_path, mask_dict, affine, transpose=None):
    for name, mask in mask_dict.items():
        nifti_name = os.path.join(base_path, name + '.nii.gz')
        if transpose is not None:
            signs = np.sign(transpose)
            transpose = np.abs(transpose) - 1
            for ax in range(3):
                if signs[ax] < 0:
                    mask = np.flip(mask, axis=ax)
            mask = np.transpose(mask, np.argsort(transpose)) #invert the original transposition
        niimg = nib.Nifti2Image(mask, affine)
        niimg.to_filename(nifti_name)