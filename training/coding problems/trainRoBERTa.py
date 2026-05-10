import torch
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from evaluate import load
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels_dict = {
    "human":0,
    "machine":1,
}

LABELS_NUM = 2

def read_json_file(file_path):
    json_objects = []
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        json_strings = content.split('\n}\n{')
        
        if len(json_strings) > 1:
            json_strings[0] += '}'
            json_strings[-1] = '{' + json_strings[-1]
            for i in range(1, len(json_strings) - 1):
                json_strings[i] = '{' + json_strings[i] + '}'
        
        for json_str in json_strings:
            if json_str.strip():  # Ensure the string is not empty
                json_objects.append(json.loads(json_str))
    
    return json_objects

def read_json(file_name) -> list:
   jsons_list = []
   with open(file_name, 'r') as file:
    for line in file:
        data = json.loads(line)
        jsons_list.append(data)

   return jsons_list
   
def json_dataset_parser(jsons_list, labels_dict:dict) -> pd.DataFrame:
    data_dict = {"text":[], "labels":[]}

    for obj in jsons_list:
        for cat in labels_dict.keys():
            if cat in obj.keys():
                data_dict["text"].append(obj[cat])
                data_dict["labels"].append(labels_dict[cat])
    print(data_dict)
    return pd.DataFrame(data_dict)

jsons_list  = read_json("data/trainHumanLlama.jsonl")
train_list, test_list = train_test_split(jsons_list, test_size=0.2)

train_df = json_dataset_parser(train_list, labels_dict)
test_df = json_dataset_parser(test_list, labels_dict)

train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

print(train_df)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Model

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=LABELS_NUM)

accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5  
)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",  
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

trainer.train()

results = trainer.evaluate()
print(results)
predictions = trainer.predict(tokenized_datasets['test'])
preds = torch.argmax(torch.tensor(predictions.predictions), axis=-1).cpu().numpy()

true_labels = tokenized_datasets['test']['labels'].numpy()
cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2, 3])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_dict.keys())

plt.figure(figsize=(30,30))
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=60)  
plt.title("RoBERTa Fine Tuning")
plt.show()

model.save_pretrained("./fine-tuned-roberta_llama")
tokenizer.save_pretrained("./fine-tuned-roberta_llama")