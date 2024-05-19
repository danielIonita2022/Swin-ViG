import io
import os
from os.path import join
import nibabel as nib
import tempfile
import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from pywebio import start_server, pin
from pywebio.input import *
from pywebio.output import *
from scipy import ndimage
import shutil
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import inference.predict_from_raw_data
from inference.predict_from_raw_data import nnUNetPredictor
from pyngrok import ngrok


def make_inference(input_file_path, fold=0):
    trained_model_path = join(nnUNet_results, 'Dataset200_BTCV',
                              'nnUNetTrainer_NexToU_BTI_Synapse__nnUNetPlans__3d_fullres')
    # prediction_folder_path = join(trained_model_path, "demo_app")
    # if not os.path.exists(prediction_folder_path):
    #     os.makedirs(prediction_folder_path)
    # shutil.move(input_file_path, prediction_folder_path)
    return inference.predict_from_raw_data.make_prediction_app(input_file_path, trained_model_path, fold)


def display_images(nifti_image1, nifti_image2, slice_indices):
    # Assume nifti_image1 and nifti_image2 are loaded and are instances of nibabel.Nifti1Image

    # Calculate the aspect ratios for coronal and sagittal views
    voxel_dims = nifti_image1.header.get_zooms()
    coronal_aspect_ratio = voxel_dims[2] / voxel_dims[0]
    sagittal_aspect_ratio = voxel_dims[2] / voxel_dims[1]

    fig = plt.figure(figsize=(18, 8))  # Adjust the figure size as needed
    # Define the gridspec to include empty space in the middle
    gs = gridspec.GridSpec(3, 7, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1, 1])

    # Display axial slices
    for i, (img, slices) in enumerate(zip([nifti_image1, nifti_image2], [slice_indices, slice_indices])):
        ax = fig.add_subplot(gs[0, i * 3:i * 3 + 2])
        ax.imshow(img.dataobj[:, :, slices[2]].T, cmap='gray', origin='lower')
        ax.set_title(f'Axial {"Original" if i == 0 else "Segmentation"}')
        ax.axis('off')

    # Display coronal slices
    for i, (img, slices) in enumerate(zip([nifti_image1, nifti_image2], [slice_indices, slice_indices])):
        ax = fig.add_subplot(gs[1, i * 3])
        ax.imshow(img.dataobj[:, slices[1], :].T, cmap='gray', origin='lower', aspect=coronal_aspect_ratio)
        ax.set_title(f'Coronal {"Original" if i == 0 else "Segmentation"}')
        ax.axis('off')

    # Display sagittal slices
    for i, (img, slices) in enumerate(zip([nifti_image1, nifti_image2], [slice_indices, slice_indices])):
        ax = fig.add_subplot(gs[1, i * 3 + 1])
        ax.imshow(img.dataobj[slices[0], :, :].T, cmap='gray', origin='lower', aspect=sagittal_aspect_ratio)
        ax.set_title(f'Sagittal {"Original" if i == 0 else "Segmentation"}')
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Save to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    # Use PyWebIO to display the image
    return buffer.read()


def main():
    file_data = file_upload(label="Upload your NIfTI file:", accept='.nii,.nii.gz')
    if not file_data:
        put_text("No file uploaded.")
        return
    gt_file_data = file_upload(label="Upload your ground truth file:", accept='.nii,.nii.gz')
    if not gt_file_data:
        put_text("No ground truth file uploaded.")
        return

    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            # If it's a file, delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            # If it's a directory, delete it and all its contents
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    if not os.path.exists(join(temp_dir, 'predict')):
        os.mkdir(join(temp_dir, 'predict'))
    original_file_folder = join(temp_dir, 'predict')
    original_file_path = join(original_file_folder, file_data['filename'])
    gt_file_path = join(temp_dir, gt_file_data['filename'])

    with open(original_file_path, 'wb') as f:
        f.write(file_data['content'])
    with open(gt_file_path, 'wb') as f:
        f.write(gt_file_data['content'])

    try:
        nifti_image = nib.load(original_file_path)
        gt_nifti_image = nib.load(gt_file_path)

        put_text("Waiting on model to finish prediction...")
        predicted_image_path = make_inference(original_file_folder, 0)
        predicted_image = nib.load(predicted_image_path)
        clear()
        max_slices = nifti_image.shape

        slice_indices = [max_slices[i] // 2 for i in range(3)]

        def update_images(slice_index, axis):
            # Update the corresponding slice index before redrawing the image
            slice_indices[axis] = slice_index
            image_data = display_images(gt_nifti_image, predicted_image, slice_indices)
            clear(scope='image_display')
            with use_scope('image_display'):
                put_image(image_data).show()

        image1_data = display_images(gt_nifti_image, predicted_image, slice_indices)
        put_text("CT segmentation visualizer").style(
            'text-align: center; margin-bottom: 5px; padding: 10px; font-size: 24px; font-weight: bold;')
        with use_scope('image_display'):
            put_image(image1_data).show()

        put_text("Adjust the slices using the sliders below:").style('margin-bottom: 5px; padding: 10px')

        # Display the sliders inside a yellow box

        with use_scope('sliders', clear=True):
            pin.put_slider(name="axial", label="Axial Slice", min_value=0, max_value=max_slices[2] - 1, step=1,
                           value=slice_indices[2])
            pin.put_slider(name="coronal", label="Coronal Slice", min_value=0, max_value=max_slices[1] - 1, step=1,
                           value=slice_indices[1])
            pin.put_slider(name="sagittal", label="Sagittal Slice", min_value=0, max_value=max_slices[0] - 1, step=1,
                           value=slice_indices[0])

        pin.pin_on_change(name="axial", clear=True, onchange=lambda value: update_images(value, 2))
        pin.pin_on_change(name="coronal", clear=True, onchange=lambda value: update_images(value, 1))
        pin.pin_on_change(name="sagittal", clear=True, onchange=lambda value: update_images(value, 0))

    except Exception as e:
        put_text("Failed to load or display NIfTI file:", str(e))


# Run the app
if __name__ == "__main__":
    public_url = ngrok.connect('8080')
    print(f"Public URL: {public_url}")
    start_server(main, port=8080)
