import mlflow
import torch
import gc
import nnunetv2.run.run_training as run_training

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('BraTS21_nnUNet')
    dataset_id = '137'
    run_training.run_training(dataset_id, '3d_fullres', 0, trainer_class_name='nnUNetTrainer')
