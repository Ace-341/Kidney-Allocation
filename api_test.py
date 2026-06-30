import time
from groq import Groq

client = Groq()

# Record start time
start_time = time.time()

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": "Explain latency."}]
)

# Calculate elapsed time
end_to_end_latency = time.time() - start_time

print(f"End-to-End Latency: {end_to_end_latency:.4f} seconds")

# Extract precise server-side latency metrics
metrics = response.usage
print(f"Groq Server Total Time: {metrics.total_time}s")
print(f"Queue Time: {metrics.queue_time}s")
