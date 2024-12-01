
---

# **Toxic Language and Misinformation Detection API**

## **Overview**
This project provides an advanced API for detecting toxic language, analyzing sentiment, and categorizing misinformation or hate speech. The system accepts textual input and returns predictions across multiple toxicity categories, along with sentiment analysis and topic modeling.

---

## **API Endpoint**
The API is hosted at:
**[https://fastapi-app-280288253402.us-central1.run.app/predict](https://fastapi-app-280288253402.us-central1.run.app/predict)**

---

## **Input Format**
- **Type**: JSON
- **Field**:
  - `text`: The input text to analyze.

### **Example Input**
```json
{
    "text": "in certain cities, the influx of immigrants has coincided with increased competition for jobs, which leads some local residents to feel displaced or overlooked"
}
```

---

## **Output Format**
- **Type**: JSON
- **Fields**:
  - **`toxicity_score`**:
    - A single value measuring the overall toxicity level in the text.
    - Higher scores indicate higher toxicity and may signal extreme hate speech.
  - **`predictions`**:
    - An array of six probabilities corresponding to predictions for the following labels:
      1. Toxic
      2. Severe Toxic
      3. Obscene
      4. Threat
      5. Insult
      6. Identity Hate
  - **`topic`**:
    - The categorized subject of the text based on the following topics:
      - `crime`
      - `employment`
      - `social benefits`
      - `immigration policies`
      - `violence`
      - `other`
  - **`sentiment`**:
    - The sentiment of the text, categorized as:
      - `NEGATIVE`
      - `NEUTRAL`
      - `POSITIVE`

---

### **Example Output**
For the given input:
```json
{
    "text": "in certain cities, the influx of immigrants has coincided with increased competition for jobs, which leads some local residents to feel displaced or overlooked"
}
```

The API would return:
```json
{
    "toxicity_score": 3.33573908856503e-05,
    "predictions": [
        0.00011814564641099423,
        4.6940104425630125e-07,
        3.177278267685324e-05,
        1.2817497008654755e-05,
        2.3205793695524335e-05,
        1.3733224477618933e-05
    ],
    "topic": "employment",
    "sentiment": "NEGATIVE"
}
```

---

## **Input and Output Details**

### **Input Parameters**
| Field   | Type   | Description                                                                 | Required |
|---------|--------|-----------------------------------------------------------------------------|----------|
| `text`  | string | The input text to analyze. Maximum length is 512 characters (truncated if exceeded). | Yes      |

### **Output Fields**
| Field           | Type     | Description                                                                 |
|------------------|----------|-----------------------------------------------------------------------------|
| `toxicity_score`| float    | Measures the overall toxicity level in the text.                            |
| `predictions`   | array    | An array of six probabilities corresponding to toxicity labels.             |
| `topic`         | string   | The categorized subject of the text (e.g., `crime`, `employment`).          |
| `sentiment`     | string   | The sentiment of the text (`NEGATIVE`, `NEUTRAL`, or `POSITIVE`).            |

#### **Toxicity Labels**
The `predictions` array corresponds to the following labels in order:
1. **Toxic**: General toxic language.
2. **Severe Toxic**: Aggressive toxic language.
3. **Obscene**: Vulgar or obscene language.
4. **Threat**: Direct or indirect threatening language.
5. **Insult**: Language meant to insult or demean.
6. **Identity Hate**: Hate speech targeting identity groups.

#### **Topics**
The `topic` field classifies the text into one of the following categories:
- `crime`
- `employment`
- `social benefits`
- `immigration policies`
- `violence`
- `other`

#### **Sentiments**
The `sentiment` field identifies the sentiment of the text:
- `NEGATIVE`: Indicates hostile or negative sentiment.
- `NEUTRAL`: Indicates neither positive nor negative sentiment.
- `POSITIVE`: Indicates supportive or positive sentiment.

---

## **Dependencies**
- Python >= 3.7
- FastAPI
- Uvicorn
- Transformers
- PyTorch

### **Install Dependencies**
```bash
pip install fastapi uvicorn transformers torch
```

---

## **License**
This project is licensed under the MIT License and is for research purposes only. Avoid using the model in inappropriate scenarios.

---


