# Text Summarization Tool

This project is a text summarization tool that uses both extractive and abstractive summarization techniques to generate concise summaries of long articles and documents. The tool leverages open-source libraries such as Hugging Face Transformers, PyTorch, NLTK, and Flask.

## Features

- **Extractive Summarization**: Selects key sentences from the original text.
- **Abstractive Summarization**: Generates new sentences that convey the same meaning as the original text.
- **Web Interface**: User-friendly web interface for easy input and output of text summaries.
- **Deployment**: Deployed on Heroku for accessibility and scalability.

## Tools and Technologies

- **Python**
- **Hugging Face Transformers**
- **PyTorch**
- **NLTK**
- **Flask**
- **Heroku**

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/text-summarization-tool.git
    cd text-summarization-tool
    ```

2. Install the required libraries:
    ```bash
    pip install torch transformers nltk flask sentencepiece accelerate
    ```

3. Download NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Usage

1. **Training the Model**:
    ```python
    from datasets import load_dataset
    from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

    # Load the dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    # Initialize the tokenizer
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

    # Initialize the model
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Define training arguments
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

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()
    ```

2. **Evaluating the Model**:
    ```python
    from datasets import load_metric

    # Load the ROUGE metric
    metric = load_metric('rouge')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result

    # Evaluate the model
    trainer.evaluate(test_dataset, metric_key_prefix='test')
    ```

3. **Running the Flask Application**:
    ```python
    from flask import Flask, request, jsonify, render_template

    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/summarize', methods=['POST'])
    def summarize():
        text = request.form['text']
        summary_type = request.form['summary_type']
        
        if summary_type == 'extractive':
            summary = extractive_summary(text)
        else:
            summary = abstractive_summary(text)
        
        return jsonify({'summary': summary})

    if __name__ == '__main__':
        app.run(debug=True)
    ```

4. **HTML Template** (`templates/index.html`):
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Summarization Tool</title>
    </head>
    <body>
        <h1>Text Summarization Tool</h1>
        <form action="/summarize" method="post">
            <textarea name="text" rows="10" cols="50" placeholder="Enter text here..."></textarea><br>
            <label for="summary_type">Summary Type:</label>
            <select name="summary_type">
                <option value="extractive">Extractive</option>
                <option value="abstractive">Abstractive</option>
            </select><br>
            <button type="submit">Summarize</button>
        </form>
        <div id="summary"></div>
    </body>
    </html>
    ```

5. **Extractive Summarization Function**:
    ```python
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.probability import FreqDist
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    nltk.download('punkt')
    nltk.download('stopwords')

    def extractive_summary(text, num_sentences=3):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        freq_dist = FreqDist(words)
        sentences = sent_tokenize(text)
        sentence_scores = {}
        
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in freq_dist:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = freq_dist[word]
                    else:
                        sentence_scores[sentence] += freq_dist[word]
        
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        summary = TreebankWordDetokenizer().detokenize(summary_sentences)
        return summary
    ```

6. **Abstractive Summarization Function**:
    ```python
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    def abstractive_summary(text, model_name='t5-small', max_length=150, min_length=30):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
        return summary
    ```

