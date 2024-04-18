import io
import os
from os.path import join
import nibabel as nib
import tempfile
import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from scipy import ndimage
import shutil
from nnunetv2.paths import nnUNet_results, nnUNet_raw

import inference.predict_from_raw_data
from inference.predict_from_raw_data import nnUNetPredictor


def make_inference(input_file_path, fold=0):
    trained_model_path = join(nnUNet_results, 'Dataset218_AMOS2022_postChallenge_task1', 'nnUNetTrainer_NexToU_BTI_Synapse__nnUNetPlans__3d_fullres')
    # prediction_folder_path = join(trained_model_path, "demo_app")
    # if not os.path.exists(prediction_folder_path):
    #     os.makedirs(prediction_folder_path)
    # shutil.move(input_file_path, prediction_folder_path)
    inference.predict_from_raw_data.make_prediction_app(input_file_path, trained_model_path, fold)

def display_image(nifti_image):
    """ Display slices from the NIfTI image with adjusted aspect ratios and sizes """

    # Calculate the aspect ratios for coronal and sagittal views
    voxel_dims = nifti_image.header.get_zooms()
    coronal_aspect_ratio = voxel_dims[2] / voxel_dims[0]
    sagittal_aspect_ratio = voxel_dims[2] / voxel_dims[1]

    # Get the middle slice number in each dimension
    slices = [nifti_image.shape[0] // 2, nifti_image.shape[1] // 2, nifti_image.shape[2] // 2]

    # Create a figure with custom subplot sizes
    fig = plt.figure(figsize=(8, 10))  # Adjust the figure size as needed
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])

    # Axial view - top span two columns
    ax_axial = fig.add_subplot(gs[0, :])
    ax_axial.imshow(nifti_image.dataobj[:, :, slices[2]].T, cmap='gray', origin='lower')
    ax_axial.set_title('Axial')
    ax_axial.axis('off')

    # Coronal view - middle left
    ax_coronal = fig.add_subplot(gs[1, 0])
    ax_coronal.imshow(nifti_image.dataobj[:, slices[1], :].T, cmap='gray', origin='lower', aspect=coronal_aspect_ratio)
    ax_coronal.set_title('Coronal')
    ax_coronal.axis('off')

    # Sagittal view - middle right
    ax_sagittal = fig.add_subplot(gs[1, 1])
    ax_sagittal.imshow(nifti_image.dataobj[slices[0], :, :].T, cmap='gray', origin='lower',
                       aspect=sagittal_aspect_ratio)
    ax_sagittal.set_title('Sagittal')
    ax_sagittal.axis('off')

    # Adjust layout spacing manually to avoid cropping, if necessary
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing as needed

    # Save to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    # Use PyWebIO to display the image
    put_image(buffer.read())


def main():
    file_data = file_upload("Upload your NIfTI file:", accept='.nii,.nii.gz')
    if not file_data:
        put_text("No file uploaded.")
        return
    gt_file_data = file_upload("Upload your ground truth file: ", accept='.nii,.nii.gz')
    if not gt_file_data:
        put_text("No ground truth file uploaded.")
        return

    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file_data['filename'])

    with open(temp_file_path, 'wb') as f:
        f.write(file_data['content'])

    try:
        nifti_image = nib.load(temp_file_path)
        put_text("File loaded successfully!")
        put_text("Image shape:", nifti_image.shape)

        display_image(nifti_image)
        predicted_seg = nib.load(make_inference(temp_file_path))
        display_image(predicted_seg)
    except Exception as e:
        put_text("Failed to load or display NIfTI file:", str(e))
    finally:
        os.remove(temp_file_path)


# Run the app
if __name__ == "__main__":
    start_server(main, port=8080)
