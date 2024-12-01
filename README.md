# EquiSpeechImm





# Toxic Language Detection API

## Overview
This project is a deep learning-based multi-label classification system for detecting toxic language in text. The model accepts textual input and outputs predictions for the following toxicity labels:
- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

---

## API Endpoint
The API is hosted at:
**[https://fastapi-app-280288253402.us-central1.run.app/preidct](https://fastapi-app-280288253402.us-central1.run.app/preidct)**

---

## Input Format
- **Type**: JSON
- **Field**:
  - `text`: The input text to classify.

### Example Input
```json
{
    "text": "You are absolutely the worst kind of human being. Nobody wants to listen to your idiotic opinions."
}
```


## Output Format
- **Type**: JSON
- **Fields**:
  - `predictions`: A dictionary of probabilities for each toxicity label.
  - `labels`: A dictionary of binary predictions for each label (0 = non-toxic, 1 = toxic).

### Example Output
```json
{
    "predictions": {
        "toxic": 0.95,
        "severe_toxic": 0.40,
        "obscene": 0.88,
        "threat": 0.12,
        "insult": 0.90,
        "identity_hate": 0.15
    },
    "labels": {
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 1,
        "threat": 0,
        "insult": 1,
        "identity_hate": 0
    }
}
```

---

## Input and Output Details

### Input Parameters
| Field   | Type   | Description                                                                 | Required |
|---------|--------|-----------------------------------------------------------------------------|----------|
| `text`  | string | Text to classify. Maximum length is 512 characters (truncated if exceeded).| Yes      |

### Output Fields
| Field         | Type     | Description                                                    |
|---------------|----------|----------------------------------------------------------------|
| `predictions` | dict     | Probabilities for each toxicity label. Values range from 0 to 1.|
| `labels`      | dict     | Binary predictions for each label (0 = non-toxic, 1 = toxic).  |

#### Toxicity Labels
| Label             | Description                                    |
|-------------------|------------------------------------------------|
| `toxic`           | Contains general toxic language.               |
| `severe_toxic`    | Contains severe toxic language.                |
| `obscene`         | Contains obscene or vulgar language.           |
| `threat`          | Contains threatening language.                 |
| `insult`          | Contains insulting language.                   |
| `identity_hate`   | Contains hate speech targeting identity groups.|

---

## How to Use
### Running the API
The API is hosted at the given endpoint. To test the API:
1. Use a tool like Postman or `curl`.
2. Send a POST request with a JSON body containing the `text` field.

### Example Request
POST request to `https://fastapi-app-280288253402.us-central1.run.app/preidct`:
- **Input**:
```json
{
    "text": "Your example toxic text here."
}
```

- **Output**:
```json
{
    "predictions": {
        "toxic": 0.87,
        "severe_toxic": 0.34,
        "obscene": 0.76,
        "threat": 0.12,
        "insult": 0.80,
        "identity_hate": 0.22
    },
    "labels": {
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 1,
        "threat": 0,
        "insult": 1,
        "identity_hate": 0
    }
}
```

---

## Dependencies
- Python >= 3.7
- FastAPI
- Uvicorn
- Transformers
- Torch

### Install Dependencies
```bash
pip install fastapi uvicorn transformers torch
```

---

## License
This project is licensed under the MIT License and is for research purposes only. Avoid using the model in inappropriate scenarios.
```

