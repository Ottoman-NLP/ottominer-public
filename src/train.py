import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data_utils import prepare_data, Vocabulary
from model import create_model
from preprocess import load_data, goldset_dir
from pathlib import Path
from utils import ProgressBar
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.utils.validation import DataConversionWarning
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    max_len = max(max(len(n) for n in noisy_batch), max(len(c) for c in clean_batch))
    noisy_batch = [nn.functional.pad(torch.tensor(n), (0, max_len - len(n)), value=0) for n in noisy_batch]
    clean_batch = [nn.functional.pad(torch.tensor(c), (0, max_len - len(c)), value=0) for c in clean_batch]
    return torch.stack(noisy_batch), torch.stack(clean_batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, optimizer, criterion, clip, scaler, accumulation_steps=2):
    model.train()
    epoch_loss = 0
    
    with ProgressBar(len(iterator), prefix='Training:', suffix='Complete', length=50) as pbar:
        for i, (noisy, clean) in enumerate(iterator):
            noisy, clean = noisy.to(device), clean.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(noisy, clean)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                clean = clean[:, 1:].contiguous().view(-1)
                loss = criterion(output, clean) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            pbar.print(i + 1)
    
    return epoch_loss / len(iterator)

if __name__ == "__main__":
    goldset_data = load_data(goldset_dir)
    aligned_data = load_data(Path(__file__).parent.parent.parent / 'corpus-texts' / 'datasets' / 'aligned_sentences.json')
    raw_data = goldset_data + aligned_data
    vocab, processed_data = prepare_data(raw_data, freq_threshold=1)
    
    train_size = int(0.91 * len(processed_data))
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:]
    
    num_epochs = 10
    clip = 0.2
    batch_size = 256  # Increased batch size
    best_loss = float('inf')
    patience = 5
    no_improve = 0

    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    input_size = len(vocab.itos)
    output_size = len(vocab.itos)
    model = create_model(input_size, output_size, device, hidden_size=128, num_layers=3, dropout=0.3).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(train_loader))
    scaler = GradScaler('cuda')

    with ProgressBar(num_epochs, prefix='Epochs:', suffix='Complete', length=50) as pbar:
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion, clip, scaler)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
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



from data_utils import prepare_data
from preprocess import load_data