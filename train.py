"""
Training script for the TS-MLP 2D DOA estimation network.

Usage:
    python train.py --K 2 --dataset datasets/dataset_2targets.h5
    python train.py --K 3 --dataset datasets/dataset_3targets.h5
    python train.py --K 4 --dataset datasets/dataset_4targets.h5

The script:
  1. Loads the pre-generated HDF5 dataset.
  2. Splits it 90 % training / 10 % validation (following the paper).
  3. Trains the TS-MLP for up to N_EPOCHS epochs with:
       - Mini-batch size BATCH_SIZE (= 1000)
       - Adam optimiser, initial LR = 1e-4
       - LR decay by factor 0.7 if validation loss plateaus for 10 epochs
       - Model checkpoint saved whenever validation loss improves
  4. Saves training history to a NumPy file.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    BATCH_SIZE, N_EPOCHS, LR_PATIENCE, LR_DECAY_FACTOR,
    MODEL_DIR,
)
from ts_mlp import build_ts_mlp


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

class HDF5DataGenerator(keras.utils.Sequence):
    """Keras data generator that streams batches from an HDF5 file.

    This avoids loading the entire (potentially hundreds of millions of
    samples) dataset into RAM at once.
    """

    def __init__(self, h5_path: str, indices: np.ndarray,
                 batch_size: int = BATCH_SIZE, shuffle: bool = True):
        self.h5_path = h5_path
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        with h5py.File(h5_path, 'r') as f:
            self.n_total = f['X'].shape[0]
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = sorted(self.indices[
            idx * self.batch_size: (idx + 1) * self.batch_size
        ])
        with h5py.File(self.h5_path, 'r') as f:
            X = f['X'][batch_idx]
            y = {
                'z1_az': f['z1_az'][batch_idx],
                'z1_el': f['z1_el'][batch_idx],
                'z2_az': f['z2_az'][batch_idx],
                'z2_el': f['z2_el'][batch_idx],
                'z3':    f['z3'][batch_idx],
            }
        return X, y


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(K: int, h5_path: str):
    print(f"\nTraining TS-MLP for K = {K} targets")
    print(f"Dataset : {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        n_total = f['X'].shape[0]
    print(f"Total samples : {n_total:,}")

    # 90/10 train/validation split
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)
    split = int(0.9 * n_total)
    train_idx = all_idx[:split]
    val_idx   = all_idx[split:]
    print(f"Train : {len(train_idx):,}  |  Validation : {len(val_idx):,}")

    train_gen = HDF5DataGenerator(h5_path, train_idx, BATCH_SIZE, shuffle=True)
    val_gen   = HDF5DataGenerator(h5_path, val_idx,   BATCH_SIZE, shuffle=False)

    model = build_ts_mlp(K)
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, f'ts_mlp_K{K}_best.h5')

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_DECAY_FACTOR,
            patience=LR_PATIENCE,
            verbose=1,
            min_lr=1e-7,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(MODEL_DIR, f'training_history_K{K}.csv')
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=N_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    hist_path = os.path.join(MODEL_DIR, f'history_K{K}.npy')
    np.save(hist_path, history.history)
    print(f"\nTraining history saved to {hist_path}")
    print(f"Best model checkpoint : {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train TS-MLP for 2D DOA estimation.')
    parser.add_argument('--K', type=int, required=True, choices=[2, 3, 4],
                        help='Number of targets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the HDF5 dataset file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device index (default: 0)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.dataset):
        print(f"ERROR: dataset file not found: {args.dataset}")
        sys.exit(1)

    train(args.K, args.dataset)


if __name__ == '__main__':
    main()
