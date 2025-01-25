from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set padding token
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", low_cpu_mem_usage=True, load_in_4bit=True
)

# Load tweets from CSV
df = pd.read_csv("./data/processed_tweets.csv")
tweets = df["cleaned_text"].tolist()

# Classification prompt template
system_prompt = """classify the tweet as "Pro Awami League", "neutral", or "pro-people/protest". Here is context: the student-led protest movement in Bangladesh, spanning from July 15 to August 5, 2024, marked a pivotal chapter in the nation’s political history. Sparked by university students opposing the reinstatement of a 30percent government job quota for the children of independence war veterans—a policy reversed by the Supreme Court in late June 2024—the movement quickly gained momentum. Initially peaceful, the protests, prominently involving Dhaka University students, escalated when members of the ruling Awami League's student wing, the Bangladesh Chatra League (BCL), attacked demonstrators. Police interventions, including tear gas and live ammunition, resulted in over 200 deaths and thousands of injuries, with the death of person name Abu Sayed, a student shot by police, emerging as a symbol of resistance. The movement's focus shifted from specific quota reforms to broader demands for government accountability and human rights, culminating in the Student-People's Uprising that galvanized public support across societal divisions. On August 5, 2024, a massive demonstration in Dhaka forced the resignation of Prime Minister Sheikh Hasina, amid government crackdowns and international criticism for human rights abuses. Recognized for its profound impact on democracy, Bangladesh was named The Economist’s “Country of the Year” for 2024, honoring the resilience and determination of its youth-led movement.
Precisely answer with only one or two words (e.g., "Pro Awami League", "neutral", "Protester")."""

# Batch inference
results = []
for tweet in tweets[11:15]:  # Test with 5 tweets
    prompt = f"{system_prompt}\nTweet: {tweet}\nLabel:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate output with constraints
    outputs = model.generate(
        **inputs,
        max_new_tokens=3,    # Limit response length to 3 tokens
        do_sample=True,      # Sampling for variety
        top_k=10,            # Restrict to top 10 tokens
        temperature=0.5      # Lower temperature for deterministic output
    )
    
    # Extract and clean the label
    label = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Label:")[-1].strip()
    results.append({"tweet": tweet, "label": label})

# Save results
print(results)  # Print results for debugging
pd.DataFrame(results).to_csv("classified_tweets.csv", index=False)
