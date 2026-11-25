"""
🚩 Red Flag Text Classifier — Web App

Usage:
    python app.py

Then open http://localhost:7860 in your browser.
Uses Gradio if installed, otherwise falls back to a built-in Python web server.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
from src.predict import load_predictor, format_result

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
USE_BERT  = os.path.exists("models/distilbert_redflag")
predictor = load_predictor(use_bert=USE_BERT)
model_tag = "DistilBERT 🤖" if USE_BERT else "TF-IDF + ComplementNB 📦"

# ─── EXAMPLE TEXTS ────────────────────────────────────────────────────────────
EXAMPLES = [
    ["You're imagining things, that never happened."],
    ["I've never felt this way about anyone in my life."],
    ["Who is that guy in your photo? Delete it now."],
    ["Fine, I guess I'm just not important to you."],
    ["If you loved me, you wouldn't go out tonight."],
    ["You're so stupid, how do you not understand this?"],
    ["I respect your boundaries and I won't push you on this."],
    ["Take all the time you need with your friends tonight!"],
    ["Let's talk this through calmly and figure it out together."],
    ["I trust you. I don't need to know every detail."],
]


def render_result_html(text: str) -> str:
    if not text.strip():
        return "<p style='color:#b45309'>⚠️ Please enter a message.</p>"

    label, prob = predictor.predict(text)
    result      = format_result(text, label, prob)

    verdict    = result["verdict"]
    category   = result["category"]
    confidence = result["confidence"]

    if label == 1:
        return f"""
        <div style='background:#fee2e2;border-left:5px solid #ef4444;
                    padding:16px;border-radius:8px;margin-top:8px;'>
            <h2 style='color:#b91c1c;margin:0'>{verdict}</h2>
            <p style='margin:4px 0;color:#374151'><b>Category:</b> {category}</p>
            <p style='margin:4px 0;color:#374151'><b>Confidence:</b> {confidence}</p>
        </div>
        """
    return f"""
    <div style='background:#dcfce7;border-left:5px solid #22c55e;
                padding:16px;border-radius:8px;margin-top:8px;'>
        <h2 style='color:#15803d;margin:0'>{verdict}</h2>
        <p style='margin:4px 0;color:#374151'><b>Category:</b> {category}</p>
        <p style='margin:4px 0;color:#374151'><b>Confidence:</b> {confidence}</p>
    </div>
    """


def launch_gradio():
    import gradio as gr

    css = """
    #main-title { text-align: center; }
    #subtitle   { text-align: center; color: #6b7280; }
    """

    with gr.Blocks(css=css, title="🚩 Red Flag Classifier") as demo:
        gr.Markdown("# 🚩 Red Flag Text Classifier", elem_id="main-title")
        gr.Markdown(
            f"Paste a text message below to find out if it's a 🚩 **Red Flag** or ✅ **Green Flag**.  \n"
            f"Model: **{model_tag}**",
            elem_id="subtitle"
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="📩 Enter a message",
                    placeholder="e.g. 'You're overreacting, that never happened.'",
                    lines=4,
                )
                with gr.Row():
                    clear_btn  = gr.Button("🗑 Clear", variant="secondary")
                    submit_btn = gr.Button("🔍 Classify", variant="primary")

            with gr.Column(scale=2):
                result_html = gr.HTML(label="Result")

        gr.Examples(
            examples=EXAMPLES,
            inputs=[text_input],
            label="📋 Try these examples",
        )

        with gr.Accordion("ℹ️ About this model", open=False):
            gr.Markdown("""
**Red Flag categories detected:**
- 🔴 **Gaslighting** — making someone question their own reality
- 💜 **Love Bombing** — overwhelming affection used as manipulation
- 🟠 **Controlling** — monitoring, restricting, or demanding behaviour
- 🟡 **Guilt Tripping** — making someone feel responsible for your emotions
- 🔵 **Manipulation** — coercive tactics to get what you want
- ⚫ **Disrespect** — belittling, insulting, or demeaning language

**Training data:** Synthetically generated examples curated by domain logic.
            """)

        submit_btn.click(fn=render_result_html, inputs=[text_input], outputs=[result_html])
        clear_btn.click(fn=lambda: ("", ""), outputs=[text_input, result_html])

    demo.launch(share=False, server_port=7860)


def launch_builtin_server(port: int = 7860):
    examples_json = json.dumps([ex[0] for ex in EXAMPLES])
    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Red Flag Text Classifier</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; color: #111827; background: #f9fafb; }}
    .card {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    h1 {{ text-align: center; margin-bottom: 4px; }}
    .sub {{ text-align: center; color: #6b7280; margin-bottom: 24px; }}
    textarea {{ width: 100%; box-sizing: border-box; padding: 12px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 15px; resize: vertical; }}
    .btns {{ display: flex; gap: 10px; margin-top: 12px; }}
    button {{ padding: 10px 18px; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; font-weight: 600; }}
    .primary {{ background: #ef4444; color: #fff; }}
    .secondary {{ background: #e5e7eb; color: #374151; }}
    .examples {{ margin-top: 20px; }}
    .ex-btn {{ background: #f3f4f6; color: #1f2937; margin: 4px; padding: 6px 10px; font-weight: 400; font-size: 13px; }}
    #result {{ margin-top: 16px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>🚩 Red Flag Text Classifier</h1>
    <p class="sub">Model: <b>{model_tag}</b></p>
    <textarea id="msg" rows="4" placeholder="e.g. 'You're overreacting, that never happened.'"></textarea>
    <div class="btns">
      <button class="primary" onclick="classify()">🔍 Classify</button>
      <button class="secondary" onclick="clearAll()">🗑 Clear</button>
    </div>
    <div id="result"></div>
    <div class="examples">
      <p><b>📋 Try these examples:</b></p>
      <div id="ex-list"></div>
    </div>
  </div>
  <script>
    const examples = {examples_json};
    const exList = document.getElementById('ex-list');
    examples.forEach(t => {{
      const b = document.createElement('button');
      b.className = 'ex-btn';
      b.textContent = t;
      b.onclick = () => {{ document.getElementById('msg').value = t; classify(); }};
      exList.appendChild(b);
    }});
    async function classify() {{
      const text = document.getElementById('msg').value;
      const res = await fetch('/predict', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ text }})
      }});
      const data = await res.json();
      document.getElementById('result').innerHTML = data.html;
    }}
    function clearAll() {{
      document.getElementById('msg').value = '';
      document.getElementById('result').innerHTML = '';
    }}
  </script>
</body>
</html>"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page_html.encode("utf-8"))

        def do_POST(self):
            if self.path != "/predict":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            html = render_result_html(body.get("text", ""))
            payload = json.dumps({"html": html}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    print(f"🌐 Gradio not installed — running built-in web UI at http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    try:
        import gradio  # noqa: F401
        launch_gradio()
    except ImportError:
        launch_builtin_server(7860)
