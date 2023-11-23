from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,RobertaForSequenceClassification,RobertaTokenizer
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np
from datetime import datetime
import os

file_path = 'golden_data/golden_eva.json'

# Open the file and load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)
    
    
texts=data
labels=[]

data_dict = {}
for sublist in data:
    if sublist:  # Check if the sublist is not empty
        key = sublist[0].lower().replace(" ", "_")  
        labels.append("id_"+key)
        data_dict[f"id_{key}"] = sublist

# Specify the file path where you want to save the JSON file
file_path = 'train_eva.json'

# Save to a JSON file
with open(file_path, 'w') as file:
    json.dump(data_dict, file, indent=4)

print(texts,"text")
print(labels,len(labels),"label len")

# Load pre-trained BERT model and tokenizer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# dataset 
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Flatten the list of lists into a single list of strings
flattened_texts = [sentence for sublist in texts for sentence in sublist]

# Tokenize the flattened list of texts
encodings = tokenizer(flattened_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

flattened_labels = []
for sublist, label in zip(texts, labels):
    for sentence in sublist:
        flattened_texts.append(sentence)
        flattened_labels.append(label)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(flattened_labels)
dataset = SentimentDataset(encodings, encoded_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=8, ################
    per_device_train_batch_size=1,
    #per_device_eval_batch_size=1,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=validation_dataset  # GPT4 similar data + GPT4 irrelevant data
)
formatted_time = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
models_path="saved_model"
model_path = "saved_model/"+formatted_time

## Train (fine-tune) the model ################################
trainer.train()
model.save_pretrained(model_path)


## get the latest model
parent_dir = models_path
dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
# Sort directories by modification time, newest first
dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
newest_dir = dirs[0] if dirs else None
print(newest_dir)


# Load the model for testing
model = BertForSequenceClassification.from_pretrained(newest_dir)

def predict_with_confidence_filtering(texts, model, tokenizer, confidence_threshold):
    predictions = []
    all_probs = []

    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)

        max_prob, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().numpy()
        if predicted_class.ndim == 0:
            predicted_class = np.array([predicted_class])
        max_prob = max_prob.cpu().numpy()

        # Filter predictions based on confidence threshold
        for idx, prob in enumerate(max_prob):
            if prob >= confidence_threshold:
                predictions.append((text, predicted_class[idx], prob))
            else:
                predictions.append((text, None, prob))  

        all_probs.append(probabilities.cpu().numpy())

    return predictions, all_probs


# test:
text =["easy app","app easy","easy cow","dog easy","easy use","simple software"]
predictions, all_probs = predict_with_confidence_filtering(text, model, tokenizer, confidence_threshold=0.5)


for i, (text, pred_class, prob) in enumerate(predictions):
    if pred_class is not None:
        # Convert pred_class to a 1D array 
        pred_class_array = np.array([pred_class]) if np.isscalar(pred_class) else np.array(pred_class)
        predicted_label = label_encoder.inverse_transform(pred_class_array)

        # Retrieve the corresponding probabilities from all_probs
        prob_array = all_probs[i]
        
        # Convert to numpy array if it's a tensor
        if isinstance(prob_array, torch.Tensor):
            prob_array = prob_array.numpy()
        formatted_probs = [f"{prob_item:.2f}" for prob_item in prob_array.flatten()]

        print(f"Input text: {text}, Predicted class: {predicted_label[0]} with confidence {prob:.2f}, All probs: {formatted_probs}")
    else:
        print(f"Input text: {text}, Prediction confidence too low")
        
'''
traning data:
[
    [
        "worst company",
        "absolutely terrible experience",
        "horrible service"
    ],
    [
        "account information",
        "access account info"
    ],
    [
        "convenient and easy to use",
        "convenient",
        "app is easy",
        "easy access",
        "convenient app",
        "easy to use"
    ],
    [
        "login to the app",
        "open the app"
    ],
    [
        "mobile check",
        "mobile check deposit",
        "cash checks",
        "mobile deposit"
    ],
    [
        "great customer",
        "friendly service",
        "friendly staff",
        "great customer service",
        "great service",
        "polite n professional"
    ],
    [
        "cannot access my account",
        "login issue",
        "locked out of my account",
        "cannot login"
    ],
    [
        "view my account",
        "account activity",
        "manage my account"
    ],
    [
        "ional service",
        "great experience"
    ],
    [
        "works fine",
        "great tool",
        "useful"
    ],
    [
        "Mortgage calculator",
        "Financial calculators"
    ],
    [
        "Scheduled payments and transfers",
        "Fund transfers between accounts"
    ],
    [
        "login info",
        "access code"
    ]
]

args:
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=8, ################
    per_device_train_batch_size=1,
    #per_device_eval_batch_size=1,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir='./logs',
)

model output 
Input text: easy app, Predicted class: id_convenient_and_easy_to_use with confidence 0.94, All probs: ['0.01', '0.00', '0.94', '0.01', '0.00', '0.00', '0.01', '0.00', '0.00', '0.00', '0.00', '0.01', '0.00']
Input text: app easy, Predicted class: id_convenient_and_easy_to_use with confidence 0.94, All probs: ['0.01', '0.00', '0.94', '0.01', '0.00', '0.00', '0.01', '0.01', '0.00', '0.00', '0.00', '0.01', '0.00']
Input text: easy cow, Predicted class: id_convenient_and_easy_to_use with confidence 0.89, All probs: ['0.01', '0.00', '0.89', '0.02', '0.01', '0.01', '0.01', '0.01', '0.00', '0.01', '0.01', '0.02', '0.01']
Input text: dog easy, Predicted class: id_convenient_and_easy_to_use with confidence 0.62, All probs: ['0.02', '0.01', '0.62', '0.07', '0.03', '0.01', '0.02', '0.02', '0.01', '0.02', '0.02', '0.08', '0.06']
Input text: easy use, Predicted class: id_convenient_and_easy_to_use with confidence 0.94, All probs: ['0.01', '0.00', '0.94', '0.01', '0.00', '0.00', '0.01', '0.00', '0.00', '0.00', '0.00', '0.01', '0.00']
Input text: simple software, Predicted class: id_convenient_and_easy_to_use with confidence 0.91, All probs: ['0.01', '0.00', '0.91', '0.01', '0.01', '0.00', '0.01', '0.01', '0.00', '0.01', '0.01', '0.01', '0.01']

'''