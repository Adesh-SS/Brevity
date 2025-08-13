from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from difflib import SequenceMatcher

app = Flask(__name__)

# Load model and tokenizer once at startup
model = AutoModelForSeq2SeqLM.from_pretrained("./brevity_small_stage5")
tokenizer = AutoTokenizer.from_pretrained("./brevity_small_stage5")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def is_similar(s1, s2, threshold = 0.85):
    return SequenceMatcher(None, s1.strip(), s2.strip()).ratio() > threshold

def filter_redundant(summaries):
    filtered = []

    for summary in summaries:
        if all(not is_similar(summary, existing) for existing in filtered):
            filtered.append(summary)
    return filtered

def chunk_text(text, chunk_size=512, overlap = 50):
    """ Function to split text into chunks smaller than the max token length. """
    tokens = tokenizer.encode(text, truncation=False)

    stride = chunk_size - overlap
    chunks = []

    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)

        # Stop if the chunk is smaller than chunk_size (last part)
        if i + chunk_size >= len(tokens):
            break
    return chunks

def summarize_chunks(chunks, length_control="Full Picture"):
    summaries = []

    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        
        chunk_size = 512 if len(chunk_text.split()) < 1000 else 1024
        target_length = 150 if len(chunk_text.split()) < 1000 else 250

        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            max_length=chunk_size,
            truncation=True,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=target_length,
                min_length=target_length // 2,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    summaries = filter_redundant(summaries)

    combined_summary = " ".join(summaries)

    if length_control == "Full Picture":
        return combined_summary
    
    snapshot_target_length = 80
    snapshot_min_len = 40

    inputs = tokenizer(
        combined_summary,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        snapshot_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=snapshot_target_length,
            min_length=snapshot_min_len,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
        )
    
    return tokenizer.decode(snapshot_ids[0], skip_special_tokens=True)

def summarize_text(text, length_control="Full Picture"):
    chunks = chunk_text(text)
    final_summary = summarize_chunks(chunks, length_control)

    # # Optionally, save to file
    # with open('summary_output.txt', 'w', encoding='utf-8') as f:
    #     f.write(final_summary)

    return final_summary

@app.route("/summarize", methods=["POST"])
def summarize_api():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' field in JSON body."}), 400

    input_text = data["text"]
    length_control = data.get("length", "Full Picture")
    summary = summarize_text(input_text, length_control)
    return jsonify({"summary": summary, "length": length_control})

@app.route("/health")   
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
