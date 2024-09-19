import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data_utils import prepare_data, Vocabulary
from model import create_model
from preprocess import load_data, goldset_dir
from tqdm import tqdm

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
    
    # Find max length in the batch
    max_len = max(max(len(n) for n in noisy_batch), max(len(c) for c in clean_batch))
    
    # Pad sequences
    noisy_batch = [nn.functional.pad(torch.tensor(n), (0, max_len - len(n)), value=0) for n in noisy_batch]
    clean_batch = [nn.functional.pad(torch.tensor(c), (0, max_len - len(c)), value=0) for c in clean_batch]
    
    noisy_batch = torch.stack(noisy_batch)
    clean_batch = torch.stack(clean_batch)
    
    return noisy_batch, clean_batch

def train(model, iterator, optimizer, criterion, clip, scheduler):
    model.train()
    epoch_loss = 0
    
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
            print(f"Error in batch {i}: {e}")
            print(f"noisy shape: {noisy.shape}, clean shape: {clean.shape}")
            continue
    
    scheduler.step(epoch_loss)
    return epoch_loss / len(iterator)
if __name__ == "__main__":
    # Load and prepare data
    raw_data = load_data(goldset_dir)
    vocab, processed_data = prepare_data(raw_data, freq_threshold=2)

    # Create dataset and dataloader
    dataset = TextDataset(processed_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    input_size = len(vocab.itos)
    output_size = len(vocab.itos)
    model = create_model(input_size, output_size, device).to(device)

    # Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> index
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Training parameters
    num_epochs = 20
    clip = 1  # Gradient clipping
    best_loss = float('inf')
    patience = 5
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, clip, scheduler)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print("Early stopping")
            break

    print("Training completed")
    torch.save(model.state_dict(), 'final_model.pth')