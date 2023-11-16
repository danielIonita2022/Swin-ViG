import multiprocessing
from multiprocessing import Pool
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from skimage import io

from nnunetv2.paths import nnUNet_raw


def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str):
    seg = io.imread(input_seg)
    # Assuming blood vessels are marked in white (255) and the background is black (0)
    seg[seg == 0] = 0  # Set non-vessel pixels to 0
    seg[seg > 0] = 255  # Set vessel pixels to 1
    image = io.imread(input_image)

    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)

if __name__ == "__main__":
    source = '/mnt/hdd1/home/danielionita/data/RetinaVesselSegmentation/'  # Update this path to your dataset location

    dataset_name = 'Dataset500_RetinaVesselSegmentation'
    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'training')
    test_source = join(source, 'testing')

    with multiprocessing.get_context("spawn").Pool(8) as p:
        # Training set
        valid_ids = subfiles(join(train_source, 'images'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(train_source, 'images', v),
                         join(train_source, 'labels', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                     ),)
                )
            )

        # Test set
        valid_ids = subfiles(join(test_source, 'images'), join=False, suffix='png')
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(test_source, 'images', v),
                         join(test_source, 'labels', v),
                         join(imagests, v[:-4] + '_0000.png'),
                         join(labelsts, v),
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'vessel': 1},
                          num_train, '.png', dataset_name=dataset_name)



