import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import create_model
from train import train, device, TextDataset, DataLoader, collate_fn, prepare_data, load_data
from pathlib import Path
import seaborn as sns
import numpy as np
from torch.cuda.amp import autocast

goldset_dir = Path(__file__).parent.parent.parent / 'corpus-texts' / 'datasets' / 'goldset.json'

class LayerActivationHook:
    def __init__(self, name):
        self.name = name
        self.activations = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # Take the first element (output) of the LSTM tuple
        self.activations.append(output.detach().cpu().numpy())

def visualize_training(model, train_loader, optimizer, criterion, clip, scaler, num_epochs=20, num_batches=50):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LSTM, nn.Linear)):
            hook = LayerActivationHook(name)
            hooks.append(hook)
            module.register_forward_hook(hook)

    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_losses = []
        
        for i, (noisy, clean) in enumerate(train_loader):
            if i >= num_batches:
                break

            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(noisy, clean)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                clean = clean[:, 1:].contiguous().view(-1)
                loss = criterion(output, clean)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batch_losses.append(loss.item())

        losses.append(epoch_loss / num_batches)
        
        # Visualize after each epoch
        visualize_epoch(epoch, hooks, batch_losses, losses)

    for hook in hooks:
        hook.remove()

    return losses

def visualize_epoch(epoch, hooks, batch_losses, epoch_losses):
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)

    # Plot layer activations
    ax1 = fig.add_subplot(gs[0, :])
    for hook in hooks:
        activations = np.mean(hook.activations, axis=(0, 1))  # Average over batch and sequence length
        ax1.plot(activations, label=hook.name)
    ax1.set_title(f"Layer Activations (Epoch {epoch+1})")
    ax1.set_xlabel("Neuron Index")
    ax1.set_ylabel("Average Activation")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(batch_losses)
    ax2.set_title(f"Batch Losses (Epoch {epoch+1})")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Loss")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epoch_losses)
    ax3.set_title("Epoch Losses")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")

    ax4 = fig.add_subplot(gs[2, :])
    last_layer_activations = hooks[-1].activations[-1]
    if len(last_layer_activations.shape) > 2:
        last_layer_activations = last_layer_activations.mean(axis=1)  # Average over sequence length if necessary
    sns.heatmap(last_layer_activations, ax=ax4, cmap='viridis')
    ax4.set_title(f"Last Layer Activation Heatmap (Epoch {epoch+1})")
    ax4.set_xlabel("Neuron Index")
    ax4.set_ylabel("Batch Sample")

    plt.tight_layout()
    plt.savefig(f"training_visualization_epoch_{epoch+1}.png")
    plt.close()

    for hook in hooks:
        hook.activations = []

if __name__ == "__main__":
    # Load and prepare data
    goldset_data = load_data(goldset_dir)
    vocab, processed_data = prepare_data(goldset_data, freq_threshold=1)
    
    train_size = int(0.9 * len(processed_data))
    train_data = processed_data[:train_size]
    
    train_dataset = TextDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    input_size = len(vocab.itos)
    output_size = len(vocab.itos)
    model = create_model(input_size, output_size, device, hidden_size=128, num_layers=2, dropout=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    clip = 0.1
    scaler = torch.cuda.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    losses = visualize_training(model, train_loader, optimizer, criterion, clip, scaler, num_epochs=20, num_batches=50)
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Overall Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("overall_training_loss.png")
    plt.close()

    print("Training visualization completed. Check the generated PNG files for results.")