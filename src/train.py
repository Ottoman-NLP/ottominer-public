import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data_utils import prepare_data, Vocabulary
from model import create_model
from preprocess import load_data, goldset_dir
from pathlib import Path
from utils import ProgressBar

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy, clean = self.data[idx]
        return torch.tensor(noisy, dtype=torch.long), torch.tensor(clean, dtype=torch.long)
    
def collate_fn(batch):
    noisy_batch, clean_batch = zip(*batch)
    
    # max length in the batch
    max_len = max(max(len(n) for n in noisy_batch), max(len(c) for c in clean_batch))
    
    # Pad sequences
    noisy_batch = [nn.functional.pad(torch.tensor(n), (0, max_len - len(n)), value=0) for n in noisy_batch]
    clean_batch = [nn.functional.pad(torch.tensor(c), (0, max_len - len(c)), value=0) for c in clean_batch]
    
    noisy_batch = torch.stack(noisy_batch)
    clean_batch = torch.stack(clean_batch)
    
    return noisy_batch, clean_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, iterator, optimizer, criterion, clip, scheduler):
    model.train()
    epoch_loss = 0
    
    with ProgressBar(len(iterator), prefix='Training:', suffix='Complete', length=50) as pbar:
        for i, (noisy, clean) in enumerate(iterator):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            
            try:
                output = model(noisy, clean)
                
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                clean = clean[:, 1:].contiguous().view(-1)
                
                loss = criterion(output, clean)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                
            except RuntimeError as e:
                print(f"\nError in batch {i}: {e}")
                print(f"noisy shape: {noisy.shape}, clean shape: {clean.shape}")
                continue
            
            pbar.print(i + 1)
    
    scheduler.step(epoch_loss)
    return epoch_loss / len(iterator)
if __name__ == "__main__":
    goldset_data = load_data(goldset_dir)
    aligned_data = load_data(Path(__file__).parent.parent.parent / 'corpus-texts' / 'datasets' / 'aligned_sentences.json')
    raw_data = goldset_data + aligned_data
    vocab, processed_data = prepare_data(raw_data, freq_threshold=2)
    subset_size = min(len(processed_data), 10000)
    processed_data = processed_data[:subset_size]
    # Split data into train and validation sets
    train_size = int(0.9 * len(processed_data))
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:]
    num_epochs = 20
    clip = 0.1
    batch_size = 64
    best_loss = float('inf')
    patience = 5
    no_improve = 0

    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = len(vocab.itos)
    output_size = len(vocab.itos)
    model = create_model(input_size, output_size, device, hidden_size=128, num_layers=2, dropout=0.3).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> index
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

with ProgressBar(num_epochs, prefix='Epochs:', suffix='Complete', length=50) as pbar:
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, clip, scheduler)
        
        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy, clean)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                clean = clean[:, 1:].contiguous().view(-1)
                loss = criterion(output, clean)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print("Early stopping")
            break
        
        pbar.print(epoch + 1)

print("Training completed")
torch.save(model.state_dict(), 'final_model.pth')
torch.save(vocab, 'vocab.pth')
print("Vocabulary saved to vocab.pth")