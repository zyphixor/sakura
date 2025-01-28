# this has got to be some of the worst code i have ever written, im so sorry if you go blind
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./results/checkpoint-249', local_files_only=True)
model = GPT2LMHeadModel.from_pretrained('./results/checkpoint-249', local_files_only=True)

# Set pad_token to be the same as eos_token (if it's not set)
tokenizer.pad_token = tokenizer.eos_token

# Function to generate text based on a prompt
def generate_response(prompt, max_length=200):
   
    # Encode the prompt to input format
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Encode the full conversation history to input format
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']  # Get attention mask

    # Generate a response from the model
    output = model.generate(
        input_ids,				# input text to gen from
        attention_mask=attention_mask,          # which tokens are padding and input
        max_length=max_length,                  # max length of the response
        num_return_sequences=1,                 # num of responses to gen
        no_repeat_ngram_size=2,                 # dont repeat words or i will beat you
        top_k=50,                               # consider the top 50 tokens
        top_p=0.95,                             # consider tokens with probability of 50 or less
	 do_sample=True,			# sampling so output is more random
        temperature=0.7,                        # Control randomness in generation
        pad_token_id=tokenizer.eos_token_id     # Padding token
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# input ui, this is the worst written program i have ever made
# please dont use this professionally or she will ruin everything
if __name__ == "__main__":
	# Initialize conversation history
	conversation_history = ""  

	# while loop for infinite input, like a chatroom
	while True:
		prompt = input("User: ")  # Get input from user
		if prompt.lower() == "exit":  # Allow user to exit
			break
		
		response = generate_response(prompt)
		print("Sakura:", response + "\n")  # print her response and add a new line so it is easier to read
