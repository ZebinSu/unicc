from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
from transformers import DistilBertTokenizer, pipeline
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.cuda.amp import GradScaler, autocast


app = FastAPI(title="Toxicity Prediction API")
from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    toxicity_score: float
    predictions: List[float]
    topic: str
    sentiment: str
    
# 定义输入和输出模型
class InputText(BaseModel):
    text: str

class DistilBERTMultiLabel(nn.Module):
    def __init__(self, bert_model_name, numerical_feature_dim, num_labels):
        super(DistilBERTMultiLabel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        self.fc_text = nn.Linear(768, 128)
        self.fc_numerical = nn.Linear(numerical_feature_dim, 128)
        self.fc_output = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, numerical_features):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # CLS token embedding

        text_out = torch.nn.ReLU()(self.fc_text(text_embeddings))
        num_out = torch.nn.ReLU()(self.fc_numerical(numerical_features))

        combined = torch.cat((text_out, num_out), dim=1)
        logits = self.fc_output(combined)
        return logits


# 工具初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBERTMultiLabel("distilbert-base-uncased", numerical_feature_dim=8, num_labels=6)
model.load_state_dict(torch.load("distilbert_multilabel_model.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
scaler = StandardScaler()

# 全局变量：Hate Words 列表
hate_words = [
    'alien', 'illegal alien', 'wetback', 'border hopper', 'invader', 'job stealer',
    'criminal immigrant', 'refugee', 'illegal', 'migrant scum', 'asylum seeker',
    'invaders', 'freeloader', 'drain on resources', 'third-worlder', 'outlander',
    'economic migrant', 'foreign parasite', 'country leech', 'boat people', 'expat',
    'border crosser', 'foreign invader', 'refugee trash', 'immigrant menace',
    'foreigner', 'outsider', 'non-native', 'gook', 'chinaman', 'ching chong',
    'paki', 'sand niggar', 'mexican invader', 'muslim terrorist', 'anchor baby',
    'cholo', 'beaner', 'turbanhead', 'terrorist', 'dirty foreigner', 'eastern european scum',
    'third-worlder', 'redneck', 'hillbilly', 'yank', 'limey', 'chink', 'kraut',
    'nip', 'wetback', 'border jumper', 'gringo', 'spic', 'cracker', 'honky'
]


# 数据预处理函数
def preprocess_text(text: str):
    """
    Process a single text input and return X_numerical and X_text.
    """
    # 特征工程
    text_length = len(text)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    repeated_characters = len(re.findall(r'(.)\1{2,}', text))
    special_char_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    capital_ratio = sum(1 for char in text if char.isupper()) / len(text) if len(text) > 0 else 0

    # Hateword 特征
    def preprocess_text_for_hatewords(text):
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def count_hate_words(text, hate_words_list):
        return sum(1 for word in text.split() if word in hate_words_list)

    cleaned_for_hatewords = preprocess_text_for_hatewords(text)
    hate_word_count = count_hate_words(cleaned_for_hatewords, hate_words)
    hate_word_ratio = hate_word_count / word_count if word_count > 0 else 0

    # 数值特征组装
    numerical_features = np.array([
        [text_length, word_count, sentence_count, repeated_characters,
         special_char_count, capital_ratio, hate_word_count, hate_word_ratio]
    ], dtype=np.float32)

    # 数值特征标准化
    scaled_numerical_features = scaler.fit_transform(numerical_features)

    # 文本清理
    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return ' '.join(text.split())

    clean_text_result = clean_text(text)
    return scaled_numerical_features, clean_text_result


# 主题生成函数
def generate_topics(X_text, batch_size=64, device=0):
    topics = ["crime", "employment", "social benefits", "immigration policies", "violence", "other"]
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    predictions = []
    for i in tqdm(range(0, len(X_text), batch_size), desc="Classifying Topics"):
        batch = X_text[i:i + batch_size]
        results = classifier(batch, candidate_labels=topics, multi_label=False)
        for result in results:
            predictions.append(result['labels'][0])
    return predictions


# 情感生成函数
def generate_sentiments(X_text, batch_size=64, device=0, max_len=512):
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device
    )

    sentiments = []
    for i in tqdm(range(0, len(X_text), batch_size), desc="Classifying Sentiments"):
        batch = [text[:max_len] for text in X_text[i:i + batch_size]]
        results = sentiment_analyzer(batch)
        sentiments.extend([
            "NEGATIVE" if res['label'] == "LABEL_0" else
            "NEUTRAL" if res['label'] == "LABEL_1" else
            "POSITIVE"
            for res in results
        ])
    return sentiments


# Toxicity Score 计算函数
def calculate_toxicity_score(predictions, weights=None):
    """
    Calculate the Toxicity Score based on the predictions of six labels.
    """
    predictions = np.array(predictions)
    if predictions.shape[1] != 6:
        raise ValueError("Predictions must have exactly 6 columns, one for each label.")
    if weights is None:
        weights = np.ones(6) / 6
    weights = np.array(weights)
    if len(weights) != 6:
        raise ValueError("Weights must have exactly 6 values.")
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    toxicity_scores = np.dot(predictions, weights)
    return toxicity_scores



# API endpoint for toxicity prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict_toxicity(input_text: InputText):
    try:
        # Step 1: Preprocess the text
        X_numerical, X_text = preprocess_text(input_text.text)

        # Step 2: Generate topics and sentiments
        topic = generate_topics([input_text.text], batch_size=1, device=-1)[0]
        sentiment = generate_sentiments([input_text.text], batch_size=1, device=-1)[0]

        # Step 3: Tokenize text for model input
        tokenized_data = tokenizer(
            [X_text],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = tokenized_data["input_ids"].to(device)
        attention_masks = tokenized_data["attention_mask"].to(device)
        numerical_features = torch.tensor(X_numerical, dtype=torch.float).to(device)

        # Step 4: Get predictions from the model
        with torch.no_grad():
            logits = model(input_ids, attention_masks, numerical_features)
            predictions = torch.sigmoid(logits).cpu().numpy()

        # Step 5: Calculate toxicity scores
        toxicity_score = calculate_toxicity_score(predictions)[0]

        # Return the result
        return PredictionResponse(
            toxicity_score=toxicity_score,
            predictions=predictions[0].tolist(),
            topic=topic,
            sentiment=sentiment
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Input error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

