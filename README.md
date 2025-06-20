# X-Ray Image Captioning with ResNet-50 and GPT-2

This project implements an image captioning system specifically designed for chest X-ray images using a combination of ResNet-50 for feature extraction and GPT-2 for caption generation.

## Overview

The model combines computer vision and natural language processing to automatically generate descriptive captions for chest X-ray images. It uses:

- **ResNet-50**: Pre-trained image classification model for extracting visual features from X-ray images
- **GPT-2**: Pre-trained language model for generating natural language captions
- **Custom Architecture**: A feature projection layer that bridges visual features to text generation

## Dataset

The project uses the **Indiana University Chest X-rays Dataset** which includes:

- `indiana_projections.csv`: Image metadata and filenames
- `indiana_reports.csv`: Medical reports and findings
- Normalized chest X-ray images

## Model Architecture

### Components

1. **Feature Extraction**: ResNet-50 (frozen) extracts 2048-dimensional features from X-ray images
2. **Feature Projection**: Linear layer projects visual features to GPT-2's hidden dimension (768)
3. **Text Generation**: GPT-2 generates captions based on combined visual and textual embeddings

### Key Features

- Handles grayscale to RGB conversion for X-ray images
- Freezes ResNet weights to prevent overfitting
- Uses teacher forcing during training
- Supports beam search for inference
- Real-time training progress monitoring with tqdm
- Advanced mixed precision training with gradient scaling
- Cosine learning rate scheduling with warmup

## Installation

1. Clone the repository and create the conda env

    ```bash
    conda create -n xray-captioning python=3.10
    conda activate xray-captioning
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download NLTK data (required for BLEU evaluation):

    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

### Training

The notebook includes a complete training pipeline:

1. **Data Loading**: Loads and preprocesses the Indiana University dataset using kagglehub
2. **Chunked Processing**: Processes large datasets in chunks to prevent memory overflow
3. **Model Setup**: Initializes ResNet-50 and GPT-2 models with optimized configurations
4. **Advanced Training Loop**:
    - Real-time progress bars showing loss, learning rate, and batch progress
    - Mixed precision training for improved performance
    - Gradient clipping for training stability
    - Cosine learning rate scheduling with warmup
5. **Evaluation**: Generates captions and computes evaluation metrics (BLEU, CIDEr)

### Inference

To generate captions for new X-ray images:

```python
model.eval()
with torch.no_grad():
    # Load and preprocess image
    pixel_values = torch.tensor(sample["pixel_values"]).unsqueeze(0).to(device)
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
    
    # Extract features and generate caption
    # (see notebook for complete inference code)
```

## Evaluation Metrics

The project implements several evaluation metrics tailored for medical image captioning:

- **BLEU Score**: Measures n-gram overlap between generated and reference captions
- **ROUGE-L**: Measures longest common subsequence, good for medical terminology
- **BERTScore**: Uses contextual embeddings to measure semantic similarity
- **Medical Entity F1**: Evaluates accuracy of medical terms and findings
- **Clinical Accuracy**: Domain-specific metric for medical correctness

## Project Structure

```text
xray/
├── v1-unical.ipynb          # Main notebook with complete pipeline
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Key Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for pre-trained models
- **Datasets**: Data loading and processing
- **PIL/Pillow**: Image processing
- **pandas**: Data manipulation
- **sacrebleu**: BLEU score computation
- **NLTK**: Natural language processing utilities
- **pycocoevalcap**: Image captioning evaluation metrics

## Hardware Requirements

- **GPU Recommended**: The model automatically detects and uses CUDA if available
- **Memory**: At least 8GB RAM recommended for training
- **Storage**: Sufficient space for the Indiana University dataset

## Model Configuration

- **Input Image Size**: 224x224 (ResNet-50 standard)
- **Caption Max Length**: 32 tokens (configurable)
- **Batch Size**: 10 (adjustable based on available memory)
- **Learning Rate**: 1e-4 (AdamW optimizer)
- **Training Epochs**: 3 (configurable)

## Notes

- The model converts grayscale X-ray images to RGB format for ResNet-50 compatibility
- Training uses teacher forcing with the ground truth captions
- The feature projection approach is simple but effective for this domain
- Model checkpoints can be saved and loaded using PyTorch's standard methods

## Future Improvements

- Implement attention mechanisms for better visual-textual alignment
- Experiment with different pre-trained vision models (ViT, DeiT)
- Add more sophisticated evaluation metrics (METEOR, ROUGE-L)
- Implement cross-validation for robust evaluation
- Add data augmentation techniques for better generalization

## Citation

If you use this code or approach in your research, please cite the relevant papers for ResNet-50, GPT-2, and the Indiana University Chest X-rays Dataset.
