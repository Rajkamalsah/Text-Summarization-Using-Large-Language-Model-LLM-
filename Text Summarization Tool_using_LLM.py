# -*- coding: utf-8 -*-
"""
Created on Sun July 13 00:11:52 2024

@author: Raj
"""

from datasets import load_dataset

dataset = load_dataset('cnn_dailymail', '3.0.0')
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']


from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess_data(data):
    inputs = [doc['article'] for doc in data]
    targets = [doc['highlights'] for doc in data]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    labels = tokenizer(targets, max_length=150, truncation=True, padding='max_length', return_tensors='pt')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = preprocess_data(train_data)
val_dataset = preprocess_data(val_data)
test_dataset = preprocess_data(test_data)


from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

model = T5ForConditionalGeneration.from_pretrained('t5-small')

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()



from datasets import load_metric

metric = load_metric('rouge')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

trainer.evaluate(test_dataset, metric_key_prefix='test')
