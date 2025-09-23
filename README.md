Custom CNN encoder for extracting image features.
LSTM-based decoder for generating captions.
Vocabulary building with tokenization and special tokens (<PAD>, <SOS>, <EOS>, <UNK>).
Training and validation loops with loss calculation and checkpoint saving.
Simple inference function to generate captions for new images.
Dataset
The model uses the Flickr8k dataset, which contains:
8,000 images.
5 captions per image.
The dataset is downloaded using the kagglehub package.

License
This project is for educational purposes.