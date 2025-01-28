from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # using gpt2 as its open source and fairly lightweight for my hardware

# Load the pretrained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# set pad_token so it doesnt throw a hissy fit
tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset

# we loading dataset with this one :fire: :fire:
dataset = load_dataset('csv', data_files='sakura2.csv')

dataset = dataset['train'].train_test_split(test_size=0.1)  # 90% train, 10% test

def tokenize_function(examples):
    # Concatenate prompt and response with a separator token (e.g., EOS token)
    texts = [f"{prompt} {tokenizer.eos_token} {response}" for prompt, response in zip(examples['prompt'], examples['response'])]
    
    # Tokenize and shift labels for causal language modeling
    encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=512)
    
    # GPT-2 requires labels to be the same as input_ids but shifted by one token
    encodings['labels'] = encodings['input_ids'].copy()
    
    return encodings

# why did i pick up an AI related project
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets['train'][0])  # make sure i didnt mess it up

# training

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluation strategy to use
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size for training
    num_train_epochs=3,              # number of training epochs
    remove_unused_columns=False,     # keep unused training data, it will throw a huge hissy fit if i dont add this
)

trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=tokenized_datasets['train'],  # Training dataset
    eval_dataset=tokenized_datasets['test'],   # dataset to test if it is bad (it probably is but oh well)
)

# Start training
trainer.train()

# save tokenizer, dont forget it.
tokenizer.save_pretrained('./results') # i didnt test this, so feel free to comment out if it dont work
