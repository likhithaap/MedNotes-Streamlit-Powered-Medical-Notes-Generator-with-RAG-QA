import together
import os
from together import Together

client = Together(api_key="38a11d9280e22f5b8c2e38385f133672f06cd405ca1f2cbfd7216183c451a33e")




upload_response = client.files.upload(
    file="/Users/likhithaparuchuri/projects/nlpProject/formatted_train.jsonl"
)

# Get the file ID from the upload response
training_file_id = upload_response.id
# Fine-tuning configuration
response = client.fine_tuning.create(
    model="togethercomputer/llama-2-7b",
    training_file=training_file_id,
    learning_rate=5e-5,  # Changed from lr
    batch_size=8,
    n_epochs=3, 
    # max_seq_length=512,
    lora=True,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
)

# Monitor fine-tuning job
print(response)

summarization_fine_tuning_job_id = response.id
print(summarization_fine_tuning_job_id)
