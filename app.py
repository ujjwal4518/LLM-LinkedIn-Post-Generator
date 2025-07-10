from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Flask app
app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LinkedIn Post Generator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f4f4f4; }
        h1 { color: #333; }
        form { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        textarea { width: 100%; height: 150px; margin-top: 10px; padding: 10px; font-size: 16px; }
        .btn { background-color: #0073b1; color: white; padding: 10px 20px; border: none; cursor: pointer; margin-top: 10px; }
        .output { background: white; padding: 20px; border-radius: 8px; margin-top: 20px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>🔗 LinkedIn Post Generator</h1>
    <form method="POST">
        <label for="theme">Enter a theme for your LinkedIn post:</label><br>
        <input type="text" id="theme" name="theme" required style="width:100%; padding:8px;"><br>
        <button type="submit" class="btn">Generate</button>
    </form>
    {% if post %}
    <div class="output">
        <h2>📝 Generated Post:</h2>
        <p>{{ post }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    post = None
    if request.method == "POST":
        theme = request.form.get("theme", "").strip()
        prompt = (
            f"Write a human-like, professional, and engaging LinkedIn post on the topic '{theme}'. "
            f"It should sound authentic, inspiring, and use a friendly tone. Include relevant hashtags.\n\n"
            f"LinkedIn Post:\n"
        )
        result = generator(
                prompt,
                max_length=250,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_return_sequences=1,
                truncation=True
              )       
        post = result[0]["generated_text"].replace(prompt, "").strip()
    return render_template_string(HTML_TEMPLATE, post=post)

if __name__ == "__main__":
    app.run(debug=True)
