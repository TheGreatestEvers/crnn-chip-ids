"""
Refactored evaluation script, compatible with CUDA, Apple MPS, and CPU.

This script contains a generic `evaluate` function that can be called from the 
fine-tuning script during validation. It also includes a `main` function that serves 
as a standalone script to evaluate a finetuned model on a test set.
"""
import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from tqdm import tqdm
import pandas as pd

from model import CRNN
from dataset import CustomOCRDataset, ocr_collate_fn
from ctc_decoder import ctc_decode

torch.backends.cudnn.enabled = False


def evaluate(crnn, dataloader, criterion, label2char_map,
             decode_method='beam_search', beam_size=10):
    """
    Evaluates the CRNN model on a given dataset. This function is device-agnostic
    as it infers the device from the model itself.
    """
    crnn.eval()
    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar = tqdm(total=len(dataloader), desc="Evaluate")

    with torch.no_grad():
        for data in dataloader:
            # This is the robust way to get the device within a function
            device = next(crnn.parameters()).device
            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char_map)
            
            reals_unpacked = []
            target_lengths_list = target_lengths.cpu().numpy().tolist()
            targets_list = targets.cpu().numpy().tolist()
            current_pos = 0
            for length in target_lengths_list:
                reals_unpacked.append(
                    "".join([label2char_map[c] for c in targets_list[current_pos:current_pos + length]])
                )
                current_pos += length

            tot_count += batch_size
            tot_loss += loss.item()
            for pred_text, real_text in zip(preds, reals_unpacked):
                pred_text = "".join(pred_text)
                if pred_text == real_text:
                    tot_correct += 1
                else:
                    wrong_cases.append((real_text, pred_text))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation


def main():
    """
    Standalone function to evaluate a finetuned model on a test set.
    """
    # --- CONFIGURATION ---
    TEST_CSV_PATH = '../data_chip_ids/test_df.csv'
    IMAGE_ROOT_DIR = '../data_chip_ids/images/'
    RELOAD_CHECKPOINT = '../finetuned_checkpoints/crnn_finetuned_..._.pth'
    EVAL_BATCH_SIZE = 32
    CPU_WORKERS = 4
    IMG_HEIGHT = 32
    IMG_WIDTH = 100
    RNN_HIDDEN = 256
    LEAKY_RELU = False

    if RELOAD_CHECKPOINT is None or '...' in RELOAD_CHECKPOINT:
        print("Error: Please specify the path to your finetuned model in RELOAD_CHECKPOINT.")
        return

    # --- Data Preparation ---
    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    all_text = ''.join(test_df['text'].tolist())
    unique_chars = sorted(list(set(all_text)))
    CHARS = "".join(unique_chars)
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    
    test_dataset = CustomOCRDataset(df=test_df, root_dir=IMAGE_ROOT_DIR, chars=CHARS,
                                    img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    test_loader = DataLoader(dataset=test_dataset, batch_size=EVAL_BATCH_SIZE,
                             shuffle=False, num_workers=CPU_WORKERS, collate_fn=ocr_collate_fn)

    # --- Model Initialization ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple M1/M2 GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    num_class = len(CHARS) + 1
    crnn = CRNN(1, IMG_HEIGHT, IMG_WIDTH, num_class, rnn_hidden=RNN_HIDDEN, leaky_relu=LEAKY_RELU)
    print(f"Loading finetuned model from {RELOAD_CHECKPOINT}")
    crnn.load_state_dict(torch.load(RELOAD_CHECKPOINT, map_location=device))
    crnn.to(device)

    # --- Loss Function ---
    criterion = CTCLoss(reduction='sum', zero_infinity=True)

    # --- Run Evaluation ---
    print("Starting evaluation on the test set...")
    results = evaluate(crnn, test_loader, criterion, LABEL2CHAR)

    print(f"\n--- Evaluation Results ---")
    print(f"  Test Loss: {results['loss']:.4f}")
    print(f"  Test Accuracy: {results['acc']:.4f}")
    print(f"  Total Correct: {int(results['acc'] * len(test_dataset))}/{len(test_dataset)}")
    
    print("\n--- Sample of Wrong Predictions (Ground Truth -> Prediction) ---")
    for i, (real, pred) in enumerate(results['wrong_cases']):
        if i >= 10: break
        print(f"  '{real}' -> '{pred}'")


if __name__ == '__main__':
    main()