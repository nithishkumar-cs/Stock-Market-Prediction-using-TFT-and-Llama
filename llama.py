import triton
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

model_id = "Llama-3.2-3B-Instruct"
data_file = "datasets/SP500-1h-news-long.csv"
output_file = "datasets/TSLA-1h-news.csv"

filtered_df = pd.read_csv(data_file, dtype=str).query("symbol == 'TSLA'")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
quantization_config = TorchAoConfig(quant_type="int4_weight_only", group_size=128)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.resize_token_embeddings(model.config.vocab_size + 1)
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = Autoencoder(input_dim=3072, bottleneck_dim=128).cuda()

def generate_output_vectors(texts):
    tokenized_inputs = tokenizer(
        texts,
        padding=True,
        max_length=3072,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        pooled_vectors = last_hidden_state.mean(dim=1)
        encoded_vectors = autoencoder.encoder(pooled_vectors.to(torch.float32))
    return encoded_vectors.cpu().numpy()

filtered_df['text'] = filtered_df.apply(lambda row: ";".join([str(value) for value in row.values]), axis=1)
filtered_df['token_length'] = filtered_df['text'].apply(lambda text: len(tokenizer.encode(text, padding=False, truncation=True, max_length=3072)))
filtered_df = filtered_df.sort_values(by='token_length', ascending=False)

batch_texts = []
current_batch = []
max_count = 0

for _, row in filtered_df.iterrows():
    max_count = max(max_count, row['token_length'])
    if max_count * (len(current_batch) + 1) <= 3072 * 2:
        current_batch.append(row['text'])
    else:
        batch_texts.append(current_batch)
        current_batch = [row['text']]
        max_count = row['token_length']

if current_batch:
    batch_texts.append(current_batch)

vectors = []
for batch in tqdm(batch_texts, desc="Processing batches"):
    vectors.append(generate_output_vectors(batch))

vectors_flat = [vec for batch in vectors for vec in batch]
vectors_df = pd.DataFrame({"news_vector": [str(vec.tolist()) for vec in vectors_flat]})

df_result = pd.concat([filtered_df.reset_index(drop=True), vectors_df.reset_index(drop=True)], axis=1)
df_result.drop(columns=["text", "token_length"])
df_result = df_result.sort_values(by='datetime')
df_result.to_csv(output_file, index=False)

print("Processing complete.")
