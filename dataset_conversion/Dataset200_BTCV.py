import re

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_btcv(btcv_base_dir: str, nnunet_dataset_id: int = 200):
    task_name = "BTCV"
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_folder = join(btcv_base_dir, 'Training')

    number_pattern = re.compile(r'(\d+)')

    train_case_ids = os.listdir(os.path.join(train_folder, 'img'))
    tr_ctr = 0
    for tr in train_case_ids:
        match = number_pattern.search(tr)
        if match:
            number = match.group()
            tr_ctr += 1
            shutil.copy(join(train_folder, 'img', 'img' + number + '.nii.gz'), join(imagestr, f'{number}_0000.nii.gz'))
            shutil.copy(join(train_folder, 'label', 'label' + number + '.nii.gz'), join(labelstr, f'{number}.nii.gz'))

    test_folder = join(btcv_base_dir, 'Testing')
    test_identifiers = os.listdir(os.path.join(test_folder, 'img'))
    for ts in test_identifiers:
        match = number_pattern.search(ts)
        if match:
            number = match.group()
            shutil.copy(join(test_folder, 'img', 'img' + number + '.nii.gz'), join(imagests, f'{number}_0000.nii.gz'))
            shutil.copy(join(test_folder, 'label', 'label' + number + '.nii.gz'), join(labelsts, f'{number}.nii.gz'))

    generate_dataset_json(out_base, {0: "CT"}, labels=
    {
        'background': 0,
        'spleen': 1,
        'right kidney': 2,
        'left kidney': 3,
        'gallbladder': 4,
        'esophagus': 5,
        'liver': 6,
        'stomach': 7,
        'aorta': 8,
        'inferior vena cava': 9,
        'portal vein and splenic vein': 10,
        'pancreas': 11,
        'right adrenal gland': 12,
        'left adrenal gland': 13
    },
                          num_training_cases=tr_ctr, file_ending='.nii.gz', dataset_name=task_name)


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input_folder', type=str,
    #                   help="The downloaded and extracted AMOS2022 (https://amos22.grand-challenge.org/) data. "
    #                         "Use this link: https://zenodo.org/record/7262581."
    #                        "You need to specify the folder with the imagesTr, imagesVal, labelsTr etc subfolders here!")
    # parser.add_argument('-d', required=False, type=int, default=218, help='nnU-Net Dataset ID, default: 218')
    # args = parser.parse_args()
    base_dir = "/mnt/hdd2/home/danielionita/data/Abdomen/RawData"
    convert_btcv(base_dir, 200)
