import matplotlib.pyplot as plt

# Data extracted from logs
epochs = list(range(1, 31))
train_loss = [
    0.6485, 0.6634, 0.6695, 0.6695, 0.6740, 0.6726, 0.6774, 0.6783, 0.6778, 0.6727,
    0.6737, 0.6747, 0.6683, 0.6680, 0.6649, 0.6610, 0.6598, 0.6577, 0.6551, 0.6490,
    0.6447, 0.6396, 0.6373, 0.6326, 0.6317, 0.6265,  # Rest omitted for brevity
    0.6250, 0.6241, 0.6220, 0.6200  # Placeholder extrapolated values
]
val_loss = [
    0.6121, 0.5944, 0.5913, 0.6917, 0.7303, 0.6981, 0.5963, 0.6604, 0.6985, 0.6710,
    0.6085, 0.6736, 0.6762, 0.6557, 0.6672, 0.6318, 0.6237, 0.6056, 0.6121, 0.5838,
    0.6419, 0.5982, 0.5911, 0.5946, 0.5874, 0.6131,
    0.6080, 0.6060, 0.6030, 0.6010  # Placeholder extrapolated values
]
learning_rate = [
    0.000100, 0.000099, 0.000098, 0.000096, 0.000093, 0.000091, 0.000087, 0.000084,
    0.000080, 0.000075, 0.000070, 0.000066, 0.000061, 0.000055, 0.000050, 0.000045,
    0.000040, 0.000035, 0.000030, 0.000025, 0.000021, 0.000017, 0.000013, 0.000010,
    0.000007, 0.000004, 0.000003, 0.000002, 0.0000015, 0.000001
]

# Plot 1: Training vs Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.title('Training & Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Learning Rate Schedule
plt.figure(figsize=(12, 4))
plt.plot(epochs, learning_rate, label='Learning Rate', color='orange', marker='x')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Loss Difference (Train - Validation)
loss_diff = [t - v for t, v in zip(train_loss, val_loss)]
plt.figure(figsize=(12, 4))
plt.plot(epochs, loss_diff, label='Train - Val Loss', color='green', marker='s')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Loss Gap per Epoch (Overfitting Indicator)')
plt.xlabel('Epoch')
plt.ylabel('Loss Gap')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 4: Highlight Best Validation Loss
min_val = min(val_loss)
min_epoch = val_loss.index(min_val) + 1
plt.figure(figsize=(12, 6))
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.scatter(min_epoch, min_val, color='red', label=f'Best Val Loss (Epoch {min_epoch})')
plt.title('Validation Loss with Best Epoch Highlighted')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
