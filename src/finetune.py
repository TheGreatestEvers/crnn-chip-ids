"""
Finetuning script for the CRNN model with Data Augmentation and TensorBoard logging.
"""
import os
import sys
# --- Set the MPS Fallback Environment Variable ---
# This must be done BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from torchvision import transforms

# <-- 1. IMPORT SummaryWriter -->
from torch.utils.tensorboard import SummaryWriter

try:
    from dataset import CustomOCRDataset, ocr_collate_fn
    from model import CRNN
    from evaluate import evaluate
except ImportError as e:
    print(e)
    print("Please make sure dataset.py, model.py, and evaluate.py are in the same directory.")
    exit(1)


def train_batch(crnn, data, optimizer, criterion, device):
    # This function remains unchanged
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]
    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5)
    optimizer.step()
    return loss.item()


def finetune():
    # =================================================================================
    # 1. CONFIGURATION
    # =================================================================================
    # --- Paths ---
    TRAIN_CSV_PATH = '../data_chip_ids/train_df.csv'
    VAL_CSV_PATH = '../data_chip_ids/val_df.csv'
    IMAGE_ROOT_DIR = '../data_chip_ids/images/'
    PRETRAINED_MODEL_PATH = '../checkpoints/crnn_synth90k.pt'
    FINETUNED_CHECKPOINTS_DIR = '../finetuned_checkpoints'
    TENSORBOARD_LOG_DIR = '../runs/finetune_crnn' # <-- Path for TensorBoard logs -->

    # --- Hyperparameters ---
    EPOCHS = 400
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4

    # --- Model Parameters ---
    IMG_HEIGHT = 32
    IMG_WIDTH = 100
    RNN_HIDDEN = 256
    LEAKY_RELU = False

    # --- Training Control ---
    CPU_WORKERS = 0
    # VALID_INTERVAL and SAVE_INTERVAL should ideally be the same or multiples
    VALID_INTERVAL = 40
    SAVE_INTERVAL = 40
    # =================================================================================

    # <-- 2. INITIALIZE SummaryWriter -->
    # The writer will create the log directory if it doesn't exist.
    # A subdirectory with the current timestamp will be created automatically,
    # which is great for comparing different runs.
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    os.makedirs(FINETUNED_CHECKPOINTS_DIR, exist_ok=True)

    # Updated device logic for M2 Mac
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple M1/M2 GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- Data Preparation & Augmentation --- (Unchanged)
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)

    all_text = ''.join(train_df['text'].tolist() + val_df['text'].tolist())
    unique_chars = sorted(list(set(all_text)))
    CHARS = "".join(unique_chars)
    print(f"Character set found ({len(CHARS)} chars): '{CHARS}'")

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])

    train_dataset = CustomOCRDataset(df=train_df, root_dir=IMAGE_ROOT_DIR, chars=CHARS,
                                     img_height=IMG_HEIGHT, img_width=IMG_WIDTH, transform=train_transform)
    val_dataset = CustomOCRDataset(df=val_df, root_dir=IMAGE_ROOT_DIR, chars=CHARS,
                                   img_height=IMG_HEIGHT, img_width=IMG_WIDTH, transform=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=CPU_WORKERS, collate_fn=ocr_collate_fn)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=CPU_WORKERS, collate_fn=ocr_collate_fn)

    # --- Model Initialization --- (Unchanged)
    num_class = len(CHARS) + 1
    crnn = CRNN(1, IMG_HEIGHT, IMG_WIDTH, num_class)
    print(f"Loading pre-trained model from {PRETRAINED_MODEL_PATH}")
    try:
        # Manually filter the dictionary ---
        print("Starting the correct loading process...")
        model_dict = crnn.state_dict()
        pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)

        # 1. Filter out weights that don't match in name or shape
        pretrained_dict_filtered = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }

        # 2. See what was left out (should be fc.weight and fc.bias)
        print("The following layers were not loaded from the pretrained model:")
        for k in pretrained_dict:
            if k not in pretrained_dict_filtered:
                print(k)

        # 3. Overwrite the state_dict of our new model with the filtered weights
        model_dict.update(pretrained_dict_filtered)

        # 4. Load the updated state_dict. This will work perfectly.
        #    We can even use strict=True now because we've made the dicts compatible.
        crnn.load_state_dict(model_dict, strict=True)

        print("\nSuccessfully loaded compatible weights!")
        #crnn.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device), strict=False)
    except Exception as e:
        print(f"Error when loading state dict: {e}")
    crnn.to(device)

    # --- Optimizer and Loss --- (Unchanged)
    optimizer = optim.Adam(crnn.parameters(), lr=LEARNING_RATE)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.to(device)

    # =================================================================================
    # 5. TRAINING LOOP with TensorBoard Logging
    # =================================================================================
    print("Starting fine-tuning...")
    # Use a global step counter for TensorBoard logging.
    # This ensures the x-axis of your plots is consistent across epochs.
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        print(f'--- Epoch {epoch}/{EPOCHS} ---')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            
            # <-- 3. LOG TRAINING LOSS -->
            # We log the per-sample loss for the current batch.
            # The 'global_step' is used as the x-axis value.
            writer.add_scalar('Loss/Train_Batch', loss / train_size, global_step)

            if global_step % VALID_INTERVAL == 0 and global_step > 0:
                print("Running validation...")
                # Put model in eval mode for validation
                crnn.eval()
                with torch.no_grad():
                    evaluation = evaluate(crnn, valid_loader, criterion, val_dataset.LABEL2CHAR,
                                          decode_method='beam_search', beam_size=10)
                # Put model back in train mode
                crnn.train()

                print(f'Validation: Loss={evaluation["loss"]:.4f}, Accuracy={evaluation["acc"]:.4f}')

                # <-- 4. LOG VALIDATION METRICS -->
                writer.add_scalar('Loss/Validation', evaluation['loss'], global_step)
                writer.add_scalar('Accuracy/Validation', evaluation['acc'], global_step)
                # You can also log hyperparameters to easily compare runs
                writer.add_hparams(
                    {'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE},
                    {'hparam/val_loss': evaluation['loss'], 'hparam/val_acc': evaluation['acc']},
                    run_name='.' # Log hparams to the same directory
                )

                if global_step % SAVE_INTERVAL == 0:
                    prefix = 'crnn_finetuned'
                    loss_val = evaluation['loss']
                    save_model_path = os.path.join(FINETUNED_CHECKPOINTS_DIR,
                                                   f'{prefix}_step{global_step}_loss{loss_val:.4f}.pth')
                    torch.save(crnn.state_dict(), save_model_path)
                    print(f'Model saved to {save_model_path}')
            
            global_step += 1

        # Log average training loss at the end of each epoch
        avg_train_loss = tot_train_loss / tot_train_count
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        print(f'End of Epoch {epoch}, Average Train Loss: {avg_train_loss:.4f}')

    # <-- 5. CLOSE THE WRITER -->
    writer.close()
    print("Fine-tuning finished. TensorBoard logs saved.")


if __name__ == '__main__':
    finetune()