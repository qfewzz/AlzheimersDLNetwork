"""Unified home for training and evaluation. Imports model and dataloader."""

import os
import sys
import multiprocessing
from . import const
from . import utils

import numpy as np
import matplotlib.pyplot as plt


import gc
import json
import time
import math
import traceback
import torch
import torch.nn as nn

# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# from torch.utils.data.sampler import SubsetRandomSampler

import random


# sys.path.insert(1, './model')

# import model.network
from .model.network import Network
from .model.network2 import NetworkWithViT_V2
from .model.data_loader import MRIData
from .model.data_loader_utils import cache_all_multiprocess


# import argparse


# parser = argparse.ArgumentParser(description='Train and validate network.')
# parser.add_argument(
#     '--disable-cuda', action='store_true', default=False, help='Disable CUDA'
# )
# args = parser.parse_args()
# args.device = None
# print(args.disable_cuda)
# if torch.cuda.is_available():
#     print("Using CUDA. : )")
#     # torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     args.device = torch.device('cuda')
# else:
#     print("We aren't using CUDA.")
#     args.device = torch.device('cpu')

# For reproducibility for testing purposes. Delete during actual training.
# torch.manual_seed(1)
# random.seed(1)

## Hyperparameters


## Training Function
def train(model, training_data, optimizer, criterion, device, data_shape):
    """takes (model, training data, optimizer, loss function)"""
    # Activate training mode
    model.train()
    # Initialize the per epoch loss
    epoch_loss = 0
    epoch_length = len(training_data)
    correct_predictions = 0
    total_predictions = 0
    for i, patient_data in enumerate(training_data):
        print(f'\t** batch {i+1}/{epoch_length}', end='\n')
        # if i % (math.floor(epoch_length / 5) + 1) == 0:
        # print(f"\t\tTraining Progress:{i / len(training_data) * 100}%")
        # Clear gradients
        model.zero_grad()
        utils.clear()
        batch_loss = torch.tensor(0.0).to(device)

        # Clear the LSTM hidden state after each patient
        model.hidden = model.init_hidden()

        # Get the MRI's and classifications for the current patient
        patient_markers = patient_data['num_images']
        current_batch_patients_MRIs = patient_data["images"].to(device)

        patient_classifications = patient_data["label"]
        # print("Patient batch classes ", patient_classifications)

        for index_patient_mri in range(len(current_batch_patients_MRIs)):
            try:
                # Clear hidden states to give each patient a clean slate
                model.hidden = model.init_hidden()
                patient_real_MRIs_ignore_padding = current_batch_patients_MRIs[
                    index_patient_mri
                ][: patient_markers[index_patient_mri]].view(
                    -1, 1, data_shape[0], data_shape[1], data_shape[2]
                )

                patient_diagnosis = patient_classifications[index_patient_mri]
                patient_endstate = (
                    torch.ones(patient_real_MRIs_ignore_padding.size(0))
                    * patient_diagnosis
                )
                patient_endstate = patient_endstate.long().to(device)

                out = model(patient_real_MRIs_ignore_padding)

                if len(out.shape) == 1:
                    out = out[
                        None, ...
                    ]  # In the case of a single input, we need padding

                # print("model predictions are ", out)
                # print("patient endstate is ", patient_endstate)
                model_predictions = out

                loss = criterion(model_predictions, patient_endstate)
                batch_loss += loss

                # Calculate accuracy
                predicted_classes = torch.argmax(model_predictions, dim=1)
                correct_predictions += (
                    (predicted_classes == patient_endstate).sum().item()
                )
                total_predictions += patient_endstate.size(0)

            except Exception as e:
                print("\nEXCEPTION CAUGHT:", e)
                traceback.print_exc()

        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
        # print("\tbatch_loss: ", batch_loss)
    print()
    # print("\tepoch_loss: ", epoch_loss)

    # print("\tepoch_loss: ", epoch_loss)

    if epoch_length == 0:
        epoch_length = 0.000001

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return utils.number_to_cpu(epoch_loss / epoch_length), accuracy


## Testing Function
def test(model, test_data, criterion, device, data_shape):
    """takes (model, test_data, loss function) and returns the epoch loss."""
    model.eval()
    epoch_loss = 0
    # epoch_loss = torch.tensor(0.0).to(device)
    epoch_length = len(test_data)
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for i, patient_data in enumerate(test_data):
            print(f'\t** batch {i+1}/{epoch_length}', end='\n')
            # if i % (math.floor(epoch_length / 5) + 1) == 0:
            #     print(f"\t\tTesting Progress:{i / len(test_data) * 100}%")
            # Clear gradients
            model.zero_grad()
            # utils.clear()
            batch_loss = torch.tensor(0.0).to(device)

            # Clear the LSTM hidden state after each patient
            model.hidden = model.init_hidden()

            # Get the MRI's and classifications for the current patient
            patient_markers = patient_data['num_images']
            current_batch_patients_MRIs = patient_data["images"].to(device)

            patient_classifications = patient_data["label"]
            # print("Patient batch classes ", patient_classifications)

            for index_patient_mri in range(len(current_batch_patients_MRIs)):
                try:
                    # Clear hidden states to give each patient a clean slate
                    model.hidden = model.init_hidden()
                    patient_real_MRIs_ignore_padding = current_batch_patients_MRIs[
                        index_patient_mri
                    ][: patient_markers[index_patient_mri]].view(
                        -1, 1, data_shape[0], data_shape[1], data_shape[2]
                    )
                    patient_diagnosis = patient_classifications[index_patient_mri]
                    patient_endstate = (
                        torch.ones(patient_real_MRIs_ignore_padding.size(0))
                        * patient_diagnosis
                    )
                    patient_endstate = patient_endstate.long().to(device)

                    out = model(patient_real_MRIs_ignore_padding)

                    if len(out.shape) == 1:
                        out = out[
                            None, ...
                        ]  # In the case of a single input, we need padding

                    model_predictions = out

                    loss = criterion(model_predictions, patient_endstate)
                    batch_loss += loss.item()

                    # Calculate accuracy
                    predicted_classes = torch.argmax(model_predictions, dim=1)
                    correct_predictions += (
                        (predicted_classes == patient_endstate).sum().item()
                    )
                    total_predictions += patient_endstate.size(0)

                except Exception as e:
                    epoch_length -= 1
                    print("\nEXCEPTION CAUGHT:", e)
                    traceback.print_exc()

            epoch_loss += batch_loss
            utils.clear()
            # print("\tbatch_loss: ", batch_loss)
        print()
    # print("\tepoch_loss: ", epoch_loss)

    if epoch_length == 0:
        epoch_length = 0.000001

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return utils.number_to_cpu(epoch_loss / epoch_length), accuracy


def draw_plots(epoches, train_losses, test_losses, train_accuracies, test_accuracies):
    plt.plot(epoches, train_losses, label='train_losses')
    plt.plot(epoches, test_losses, label='test_losses')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.xticks(epoches)
    plt.grid()
    plt.show()
    
    plt.plot(epoches, train_accuracies, label='train_accuracies')
    plt.plot(epoches, test_accuracies, label='test_accuracies')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.xticks(epoches)
    plt.grid()
    plt.show()


def main(dataset_json_path=None, device=None, **args):
    config = const.Config()
    config_fields = const.Config.__dataclass_fields__.keys()
    
    for key, value in args.items():
        if key not in config_fields:
            raise Exception(f'the name {key} is not a valid config!')
        if key is not None:
            setattr(config, key, value)

    config.refresh()

    # Dimensionality of the data outputted by the LSTM,
    # forwarded to the final dense layer.
    input_size = 1  # Size of the processed MRI scans fed into the CNN.

    output_dimension = 2  # the number of predictions the model will make
    # 2 used for binary prediction for each image.
    # update the splicing used in train()

    ## Import Data
    # MRI_images_list = pickle.load(open("./Data/Combined_MRI_List.pkl", "rb"))

    # MRI_images_list = MRI_images_list[:5]
    # MRI_images_list = [
    #     [*data[: random.randint(2, min(len(data) - 1, 4))], data[-1]]
    #     for data in MRI_images_list
    # ]

    if dataset_json_path is None:
        dataset_json_path = (
            f'AlzheimersDLNetwork\\data_sample\\data_sample_image_paths.txt'
        )
    if device is None:
        device = 'cpu'

    MRI_images_list = json.loads(open(dataset_json_path, 'r', encoding='utf-8').read())
    random.shuffle(MRI_images_list)

    train_size = int(0.8 * len(MRI_images_list))

    # Split list
    training_list = MRI_images_list[:train_size]
    test_list = MRI_images_list[train_size:]

    DATA_ROOT_DIR = './'
    train_dataset = MRIData(DATA_ROOT_DIR, training_list)
    test_dataset = MRIData(DATA_ROOT_DIR, test_list)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    training_data = train_loader
    test_data = test_loader
    
    if config.CHECK_CACHE:
        cache_all_multiprocess(train_dataset.root_dir, train_dataset.data_array)
        cache_all_multiprocess(test_dataset.root_dir, test_dataset.data_array)

    ## Define Model
    if config.USE_VT:
        model = NetworkWithViT_V2(input_size, config.DIMESIONS, output_dimension).to(device)
    else:
        model = Network(input_size, config.DIMESIONS, output_dimension).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

    # Perform training and measure test accuracy. Save best performing model.
    best_test_accuracy = float('inf')

    epoches = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # This evaluation workflow below was adapted from Ben Trevett's design
    # on https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

    try:
        for epoch in range(config.EPOCHES):
            print(f'* starting epoch {epoch+1}/{config.EPOCHES}')
            start_time = time.time()
            print('start training...')

            train_loss, train_accuracy = train(
                model, training_data, optimizer, loss_function, device, config.DIMESIONS
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            utils.clear()

            print('start testing...')
            test_loss, test_accuracy = test(
                model, test_data, loss_function, device, config.DIMESIONS
            )
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            epoches.append(epoch + 1)

            utils.clear()
            end_time = time.time()

            epoch_mins = math.floor((end_time - start_time) / 60)
            epoch_secs = math.floor((end_time - start_time) % 60)

            print()
            print(
                f"epoch {epoch + 1}/{config.EPOCHES} done | Time: {epoch_mins}m {epoch_secs}s"
            )
            print(
                f"Train Loss:\t{train_loss:.3f} | Train Accuracy: {train_accuracy:.3f}"
            )
            print(f"Test Loss:\t{test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}")

            if test_loss < best_test_accuracy:
                print("that was our best test accuracy yet!")
                best_test_accuracy = test_loss
                torch.save(model.state_dict(), 'ad-model.pt')

            print('-' * 20)
            # utils.clear()
    except BaseException as e:
        print(f'* got exception: {e}')
    finally:
        return epoches, train_losses, test_losses, train_accuracies, test_accuracies


if __name__ == "__main__":
    main()
