import os
import requests
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_CHARS_PER_CHUNK = 50_000

if not GEMINI_API_KEY:
    raise ValueError("❌ Please set GEMINI_API_KEY in your environment variables.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# === HELPERS ===
def chunk_text(text, max_len):
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split()).strip()

def clean_chunk(chunk, index):
    prompt = f"""
System: You are an expert at extracting information.

You will receive a chunk of text.

Your job is to carefully read the chunk and return only the meaningful content.

Remove anything unrelated or low-value such as advertisements, HTML or JavaScript code, styling instructions, or boilerplate text.

Preserve the full details of the remaining content.

If no meaningful information is found, output exactly: "No relevant information found."

---

User: You are an information extraction assistant.

You will receive a chunk of text.

Instructions:
- Extract all meaningful and informative content from the text chunk.
- Remove any unrelated or low-value parts such as advertisements, HTML or JavaScript code, styling instructions, boilerplate text, or navigation menus.
- Preserve full details and explanations from relevant sections — do not shorten unless necessary to remove junk.
- Return only the cleaned, relevant text.
- If the chunk contains no meaningful information, output exactly: "No relevant information found."

Text Chunk ({index + 1}):
{chunk}
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# === FLASK APP ===
app = Flask(__name__)

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.json
    input_url = data.get("url")

    if not input_url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    try:
        # Fetch page with ScraperAPI
        response = requests.get("https://api.scraperapi.com/", params={
            "api_key": SCRAPER_API_KEY,
            "url": input_url,
            "render": "true"
        })
        response.raise_for_status()

        # Extract text
        extracted_text = extract_text_from_html(response.text)

        # Chunk & clean
        chunks = chunk_text(extracted_text, MAX_CHARS_PER_CHUNK)
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            cleaned = clean_chunk(chunk, i)
            cleaned_chunks.append(cleaned)

        final_output = "\n".join(f"{i + 1}\n{c}" for i, c in enumerate(cleaned_chunks))

        return jsonify({
            "url": input_url,
            "chunks": len(chunks),
            "final_output": final_output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Render provides PORT env var
    app.run(host="0.0.0.0", port=port, debug=True)

