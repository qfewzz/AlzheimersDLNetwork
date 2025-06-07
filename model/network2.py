"""Pytorch implementation of a hybrid CNN-ViT LSTM network (V2 - with resizing fix)."""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import torchvision.models as models
import torchvision.transforms as transforms  # <-- 1. IMPORT TRANSFORMS

# For reproducibility for testing purposes. Delete during actual training.
# torch.manual_seed(1)


class NetworkWithViT_V2(nn.Module):
    """
    Hybrid CNN and Vision Transformer (ViT) LSTM to classify ADNI data.
    V2: Includes a resizing transform to match the ViT's expected input size.
    """

    def __init__(self, input_channels, input_shape, output_size, lstm_layers=1):
        super(NetworkWithViT_V2, self).__init__()

        print("Initializing hybrid CNN-ViT-LSTM model (V2)...")

        self.input_shape = input_shape

        # ====================================================================
        # 1. CNN Branch (Original 3D Feature Extractor)
        # ====================================================================

        def dimensions_after_convolution(kernel, stride, padding, current_shape):
            # (Calculation helper function remains the same)
            output_depth = math.floor(
                (current_shape[0] + 2 * padding - kernel + 1) / stride
                + (stride - 1) / stride
            )
            output_height = math.floor(
                (current_shape[1] + 2 * padding - kernel + 1) / stride
                + (stride - 1) / stride
            )
            output_width = math.floor(
                (current_shape[2] + 2 * padding - kernel + 1) / stride
                + (stride - 1) / stride
            )
            return output_depth, output_height, output_width

        kernel_size = 4
        padding = 0

        self.convolution1 = nn.Conv3d(input_channels, 10, kernel_size, padding=padding)
        cnn_shape = dimensions_after_convolution(
            kernel_size, 1, padding, self.input_shape
        )
        self.pool1 = nn.MaxPool3d(kernel_size)
        cnn_shape = dimensions_after_convolution(
            kernel_size, kernel_size, padding, cnn_shape
        )
        self.convolution2 = nn.Conv3d(10, 5, kernel_size, padding=padding)
        cnn_shape = dimensions_after_convolution(kernel_size, 1, padding, cnn_shape)
        self.pool2 = nn.MaxPool3d(kernel_size)
        cnn_shape = dimensions_after_convolution(
            kernel_size, kernel_size, padding, cnn_shape
        )
        self.convolution3 = nn.Conv3d(5, 1, kernel_size, padding=padding)
        cnn_shape = dimensions_after_convolution(kernel_size, 1, padding, cnn_shape)

        self.cnn_feature_size = cnn_shape[0] * cnn_shape[1] * cnn_shape[2]
        print(f"CNN branch will produce features of size: {self.cnn_feature_size}")

        # ====================================================================
        # 2. Vision Transformer (ViT) Branch (New 2D Feature Extractor)
        # ====================================================================

        self.vision_transformer = models.vit_b_16(weights=None)

        # Modify ViT for 1-channel input
        vit_original_proj = self.vision_transformer.conv_proj
        self.vision_transformer.conv_proj = nn.Conv2d(
            1,
            vit_original_proj.out_channels,
            kernel_size=vit_original_proj.kernel_size,
            stride=vit_original_proj.stride,
            padding=vit_original_proj.padding,
        )

        # Get features, not classifications
        self.vit_feature_size = self.vision_transformer.hidden_dim
        self.vision_transformer.heads = nn.Identity()
        print(f"ViT branch will produce features of size: {self.vit_feature_size}")

        # <-- 2. DEFINE THE RESIZE TRANSFORM
        # The ViT model expects 224x224 images. We create a transform to resize our slices.
        self.resize_transform = transforms.Resize((224, 224), antialias=True)

        # ====================================================================
        # 3. Combined LSTM and Final Prediction Layer
        # ====================================================================

        combined_feature_size = self.cnn_feature_size + self.vit_feature_size
        print(f"Combined feature size for LSTM input: {combined_feature_size}")

        self.lstm = nn.LSTM(combined_feature_size, combined_feature_size, lstm_layers)
        self.prediction_converter = nn.Linear(combined_feature_size, output_size)

        self.num_layers = lstm_layers
        self.hidden_dimensions = combined_feature_size

    def init_hidden(self, batch_size=1):
        return (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dimensions,
                device=next(self.parameters()).device,
            ),
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dimensions,
                device=next(self.parameters()).device,
            ),
        )

    def forward(self, MRI):
        # --- 1. CNN Branch Forward Pass ---
        cnn_feature_space = self.convolution3(
            self.pool2(self.convolution2(self.pool1(self.convolution1(MRI))))
        )
        cnn_features = cnn_feature_space.view(MRI.size(0), -1)

        # --- 2. ViT Branch Forward Pass ---
        middle_slice_idx = self.input_shape[0] // 2
        mri_2d_slice = MRI[:, :, middle_slice_idx, :, :]

        # <-- 3. APPLY THE RESIZE TRANSFORM
        # Resize the 2D slice to the ViT's expected input size (224x224)
        mri_2d_slice_resized = self.resize_transform(mri_2d_slice)

        # Pass the correctly-sized slice to the ViT
        vit_features = self.vision_transformer(mri_2d_slice_resized)

        # --- 3. Combine Features and Pass to LSTM ---
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        lstm_in = combined_features.unsqueeze(1)
        lstm_out, self.hidden = self.lstm(lstm_in)
        dense_conversion = self.prediction_converter(lstm_out.squeeze(1))

        return dense_conversion


# Testing. Run random data through the network to ensure that everything checks out.
if __name__ == "__main__":
    # Parameters for the test
    seq_len = 4
    input_channels_test = 1
    input_shape_test = (64, 96, 96)  # Using a more realistic, smaller size
    output_size_test = 3

    # Create the model
    hybrid_net = NetworkWithViT(input_channels_test, input_shape_test, output_size_test)

    # Create some random input data
    # Shape: (SequenceLength, Channels, Depth, Height, Width)
    random_mri_sequence = torch.randn(seq_len, input_channels_test, *input_shape_test)

    # Run the data through the network
    preds = hybrid_net(random_mri_sequence)

    # Print the output shape to verify
    print("\nInput Shape:", random_mri_sequence.shape)
    print("Output Predictions Shape:", preds.shape)
    # Expected output shape: [4, 3] (Sequence Length, Output Size)
