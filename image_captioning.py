import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import csv
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import kagglehub

print("Downloading dataset...")
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)
print("CUDA available:", torch.cuda.is_available())

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("Loading captions...")
captions_file = "/kaggle/input/flickr8k/captions.txt"
captions = []

with open(captions_file, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 2:
            img = row[0]
            caption = ",".join(row[1:]).strip()
            captions.append((img, caption))

print(f"Loaded {len(captions)} captions")

print("Building vocabulary...")
word_freq = Counter()
for _, caption in captions:
    tokens = nltk.word_tokenize(caption.lower())
    word_freq.update(tokens)

threshold = 5
words = [w for w in word_freq if word_freq[w] >= threshold]
special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
vocab = special_tokens + words

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

print(f"Vocabulary size: {len(vocab)}")

def encode_caption(caption, word2idx, max_len=20):
    tokens = nltk.word_tokenize(caption.lower())
    tokens = [w if w in word2idx else "<UNK>" for w in tokens]
    tokens = ["<SOS>"] + tokens + ["<EOS>"]
    encoded = [word2idx[w] for w in tokens]
    if len(encoded) < max_len:
        encoded += [word2idx["<PAD>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FlickrDataset(Dataset):
    def __init__(self, img_folder, captions, word2idx, transform=None, max_len=20):
        self.img_folder = img_folder
        self.captions = captions
        self.word2idx = word2idx
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_id, caption = self.captions[idx]
        img_path = os.path.join(self.img_folder, img_id)
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        encoded_caption = encode_caption(caption, self.word2idx, self.max_len)
        return image, torch.tensor(encoded_caption, dtype=torch.long)

class CNNEncoder(nn.Module):
    def __init__(self, embed_size=256):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embed_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        hiddens = self.dropout(hiddens)
        outputs = self.fc(hiddens)
        return outputs

print("Preparing datasets...")
img_dir = "/kaggle/input/flickr8k/Images"
train_caps, val_caps = train_test_split(captions, test_size=0.1, random_state=42)

train_dataset = FlickrDataset(img_folder=img_dir, captions=train_caps, word2idx=word2idx, transform=transform, max_len=20)
val_dataset = FlickrDataset(img_folder=img_dir, captions=val_caps, word2idx=word2idx, transform=transform, max_len=20)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train batches: {len(train_dataloader)} | Val batches: {len(val_dataloader)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

encoder = CNNEncoder(embed_size).to(device)
decoder = RNNDecoder(embed_size, hidden_size, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=1e-3)

print("Starting training...")
num_epochs = 3

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print("-" * 50)
    
    for batch_idx, (images, captions) in enumerate(train_dataloader):
        try:
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions)
            targets = captions[:, 1:]
            outputs = outputs[:, 1:, :]
            batch_size_curr, seq_len, vocab_size_curr = outputs.shape
            outputs_flat = outputs.reshape(-1, vocab_size_curr)
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5)
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"Images shape: {images.shape if 'images' in locals() else 'N/A'}")
            print(f"Captions shape: {captions.shape if 'captions' in locals() else 'N/A'}")
            print(f"Outputs shape: {outputs.shape if 'outputs' in locals() else 'N/A'}")
            print(f"Targets shape: {targets.shape if 'targets' in locals() else 'N/A'}")
            continue
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(val_dataloader):
            try:
                images, captions = images.to(device), captions.to(device)
                features = encoder(images)
                outputs = decoder(features, captions)
                targets = captions[:, 1:]
                outputs = outputs[:, 1:, :]
                batch_size_curr, seq_len, vocab_size_curr = outputs.shape
                outputs_flat = outputs.reshape(-1, vocab_size_curr)
                targets_flat = targets.reshape(-1)
                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()
            except Exception as e:
                print(f"Validation error in batch {batch_idx}: {e}")
                continue
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'vocab': vocab,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'epoch': epoch,
        'train_loss': avg_loss,
        'val_loss': avg_val_loss
    }, f"checkpoint_epoch_{epoch+1}.pth")
    
    print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

print("Training completed!")

def generate_caption(encoder, decoder, image_path, word2idx, idx2word, device, max_len=20):
    encoder.eval()
    decoder.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image)
        caption = [word2idx["<SOS>"]]
        for _ in range(max_len - 1):
            caption_tensor = torch.tensor([caption]).to(device)
            outputs = decoder(features, caption_tensor)
            predicted = outputs[0, -1, :].argmax().item()
            caption.append(predicted)
            if predicted == word2idx["<EOS>"]:
                break
        caption_words = [idx2word[idx] for idx in caption[1:]]
        if "<EOS>" in caption_words:
            caption_words = caption_words[:caption_words.index("<EOS>")]
        return " ".join(caption_words)
