# sentiment_model.py
import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import logging

# Set up logging
logger = logging.getLogger(__name__)

class RegularizedClassifier(torch.nn.Module):
    def __init__(self, num_labels=5, hyperparams=None):
        super(RegularizedClassifier, self).__init__()
        self.hyperparams = hyperparams or {}
        self.dropout = torch.nn.Dropout(self.hyperparams.get("Dropout", 0.3))
        self.fc = torch.nn.Linear(768, num_labels)  # BERT output size is 768

    def forward(self, x, training=True):
        if training:
            x = self.dropout(x)
        return self.fc(x)

def predict_sentiment(text, model_path=None, max_length=512):
    """
    Load a serialized regular model and predict sentiment for a user-generated text.

    Args:
        text (str): User-generated text to predict sentiment for
        model_path (str, optional): Path to the model file. Defaults to final_model.pt if None
        max_length (int): Maximum sequence length for BERT tokenization (default: 512)

    Returns:
        dict: Contains predicted class (0-4), sentiment label, and probabilities for all classes
    """
    # Set default model path if not provided
    local_dir = os.getcwd()
    if model_path is None:
        model_path = os.path.join(local_dir, "final_model.pt")

    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Initialize device (CUDA if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Initialize model
    hyperparams = {
        "Dropout": 0.3,
        "DropConnect Prob": 0.2,
        "Mixout Prob": 0.2,
        "Stochastic Depth Prob": 0.1,
        "Use Gradient Checkpointing": False
    }
    model = RegularizedClassifier(num_labels=5, hyperparams=hyperparams)

    # Load model state with weights_only=True for safety and strict=False to ignore extra keys
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Preprocess the input text
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get BERT embeddings
    bert_model = BertModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment').to(device)
    bert_model.eval()

    with torch.no_grad():
        bert_outputs = bert_model(**inputs)
        embeddings = bert_outputs.last_hidden_state[:, 0, :]  # CLS token embedding

    # Make prediction with autocast for efficiency
    with torch.no_grad(), torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
        output = model(embeddings, training=False)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        all_probabilities = probabilities[0].cpu().tolist()

    # Map prediction to sentiment label
    sentiment_labels = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }

    # Clean up
    del bert_model, inputs, bert_outputs, embeddings
    torch.cuda.empty_cache()

    # Return results with probabilities for each class
    return {
        "predicted_class": predicted_class,
        "sentiment": sentiment_labels[predicted_class],
        "probabilities": dict(zip(sentiment_labels.values(), all_probabilities))
    }

if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    sample_text = "I absolutely love this product, it's amazing!"
    result = predict_sentiment(sample_text)
    print("\nRegular Model Prediction:")
    print(f"Class: {result['predicted_class']}")
    print(f"Sentiment: {result['sentiment']}")
    print("Probabilities for each class:")
    for label, prob in result['probabilities'].items():
        print(f"{label}: {prob:.4f}")