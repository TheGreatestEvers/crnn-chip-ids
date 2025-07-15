# test_evaluation.py
import torch
from torch.nn import CTCLoss
import numpy as np
import pandas as pd

# Import the functions and classes we want to test
from evaluate import evaluate
from ctc_decoder import ctc_decode
from dataset import CustomOCRDataset, ocr_collate_fn

print("--- Running Unit Test for Evaluation Pipeline ---")

# ==============================================================================
# 1. SETUP THE ENVIRONMENT (like in finetune.py)
# ==============================================================================

# Define a simple character set for our test
CHARS = '0123456789FNS#'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
NUM_CLASSES = len(CHARS) + 1  # Plus 1 for the CTC blank token

# Define our ground truth labels for the test
GROUND_TRUTH_TEXTS = ['FNS#123', '#456']

# ==============================================================================
# 2. CREATE A "PERFECT" FAKE MODEL OUTPUT
# This simulates what a perfectly trained CRNN would output.
# ==============================================================================

# The CRNN model outputs a sequence of predictions. Let's say its sequence length is 24.
SEQ_LENGTH = 24
BATCH_SIZE = len(GROUND_TRUTH_TEXTS)

# Create a tensor to hold our fake `log_probs`
# Shape: (sequence_length, batch_size, num_classes)
log_probs = torch.full((SEQ_LENGTH, BATCH_SIZE, NUM_CLASSES), -100.0) # Fill with a very low log-probability
log_probs[:, :, 0] = 0 # Make the blank token the most likely by default (log(1.0) = 0)

# Now, for each text, "draw" the perfect prediction onto the log_probs tensor
for i, text in enumerate(GROUND_TRUTH_TEXTS):
    labels = [CHAR2LABEL[char] for char in text]
    for j, label in enumerate(labels):
        # At each time step `j`, for batch item `i`, set the probability of the correct
        # character `label` to be 1.0 (log-probability 0.0)
        log_probs[j, i, label] = 0.0
        # And make the blank token less likely at this position
        log_probs[j, i, 0] = -100.0

# `log_probs` is now a "perfect" prediction tensor.

# ==============================================================================
# 3. CREATE FAKE GROUND TRUTH DATA (like the DataLoader would)
# ==============================================================================

targets = []
target_lengths = []
for text in GROUND_TRUTH_TEXTS:
    labels = [CHAR2LABEL[char] for char in text]
    targets.extend(labels)
    target_lengths.append(len(labels))

targets = torch.LongTensor(targets)
target_lengths = torch.LongTensor(target_lengths)

# ==============================================================================
# 4. RUN THE CORE LOGIC OF THE EVALUATION FUNCTION
# We are testing the decoding and comparison part.
# ==============================================================================

try:
    print("\nStep 4a: Testing the ctc_decode function...")
    # Call the decoder with our perfect `log_probs`
    preds_as_labels = ctc_decode(log_probs, method='beam_search', beam_size=10)
    print(f"  -> Raw labels from decoder: {preds_as_labels}")

    print("\nStep 4b: Testing the text conversion and comparison logic...")
    # --- This block is copied directly from your evaluate.py ---
    preds_as_text = []
    for label_list in preds_as_labels:
        pred_text = "".join([LABEL2CHAR[l] for l in label_list])
        preds_as_text.append(pred_text)
    
    print(f"  -> Decoded text from model: {preds_as_text}")
    print(f"  -> Ground truth text:       {GROUND_TRUTH_TEXTS}")
    
    correct_count = 0
    for pred, real in zip(preds_as_text, GROUND_TRUTH_TEXTS):
        if pred == real:
            correct_count += 1

    accuracy = correct_count / BATCH_SIZE
    # --- End of copied block ---

    print("\n--- TEST RESULTS ---")
    if accuracy == 1.0:
        print("✅ SUCCESS: The evaluation pipeline is working correctly!")
        print("   The predicted text perfectly matched the ground truth.")
    else:
        print("❌ FAILURE: The evaluation pipeline is broken.")
        print(f"   Expected accuracy: 1.0, but got: {accuracy}")
        print("   This confirms the problem is in `ctc_decode.py` or `evaluate.py`, not the model training.")

except Exception as e:
    print(f"\n❌ CRITICAL FAILURE: The test script crashed with an error.")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()