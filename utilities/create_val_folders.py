import os
import shutil
from os.path import join
import json

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


def create_validation_folders(dataset_name):
    splits_file_path = join(nnUNet_preprocessed, dataset_name, 'splits_final.json')
    with open(splits_file_path, 'r') as f:
        data = json.load(f)

    for i, fold in enumerate(data):
        fold_val_folder = join(nnUNet_raw, dataset_name, f'imagesVal_fold_{i}')
        os.makedirs(fold_val_folder, exist_ok=True)

        for img_name in fold['val']:
            src_path = join(nnUNet_raw, dataset_name, 'imagesTr', f'{img_name}_0000.nii.gz')
            dst_path = join(fold_val_folder, f'{img_name}_0000.nii.gz')

            shutil.copy(src_path, dst_path)
            print(f'Copied {img_name} to {dst_path}.')
        print(f'Done copying to fold {i}')


if __name__ == '__main__':
    create_validation_folders('Dataset027_ACDC')
