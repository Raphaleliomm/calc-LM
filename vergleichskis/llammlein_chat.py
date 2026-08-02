import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")

# (display_name, model_id, type)
MODELS = [
    ("LLaMmlein 120M", "LSX-UniWue/LLaMmlein_120M", "base"),
    ("SmolLM2 135M Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "chatml"),
    ("Leviathan 100M", "ShiningSon/Leviathan-MLGRU-100M-TinyStories-Instruct-v08a", "base"),
    ("LLaMmlein2Vec 120M (Embed)", "LSX-UniWue/LLaMmlein2Vec_120M", "embed"),
]


class LlammleinChat:
    def __init__(self, root):
        self.root = root
        self.root.title("LLaMmlein Chat")
        self.root.geometry("700x600")

        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.current_model_type = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chat_history = []

        self.create_widgets()

    # ----- UI -----
    def create_widgets(self):
        # Top bar: model dropdown + load button
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(top, text="Model:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(
            top, textvariable=self.model_var,
            values=[m[0] for m in MODELS], state="readonly", width=30,
        )
        self.model_menu.current(0)
        self.model_menu.pack(side="left", padx=(4, 8))

        self.load_btn = tk.Button(top, text="Load", command=self.load_model)
        self.load_btn.pack(side="left")

        # Status line
        self.status_var = tk.StringVar(value="No model loaded")
        tk.Label(self.root, textvariable=self.status_var).pack(fill="x", padx=8, anchor="w")

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(self.root, wrap="word")
        self.chat_display.pack(fill="both", expand=True, padx=8, pady=4)

        # Input area (swapped depending on model type)
        self.input_container = tk.Frame(self.root)
        self.input_container.pack(fill="x", padx=8, pady=(4, 8))
        self.setup_chat_input()

    def setup_chat_input(self):
        for w in self.input_container.winfo_children():
            w.destroy()

        self.input_entry = tk.Text(self.input_container, height=3, wrap="word")
        self.input_entry.pack(side="left", fill="x", expand=True)
        self.input_entry.bind("<Return>", self.on_enter)
        self.input_entry.bind("<Shift-Return>", lambda e: None)

        btn_frame = tk.Frame(self.input_container)
        btn_frame.pack(side="right", padx=(4, 0))
        self.send_btn = tk.Button(btn_frame, text="Send", command=self.send_message)
        self.send_btn.pack()
        tk.Button(btn_frame, text="Clear", command=self.clear_chat).pack(pady=(2, 0))

    def setup_embed_input(self):
        for w in self.input_container.winfo_children():
            w.destroy()

        tk.Label(self.input_container, text="Text A:").pack(anchor="w")
        self.text1_entry = tk.Text(self.input_container, height=2, wrap="word")
        self.text1_entry.pack(fill="x")

        tk.Label(self.input_container, text="Text B:").pack(anchor="w", pady=(4, 0))
        self.text2_entry = tk.Text(self.input_container, height=2, wrap="word")
        self.text2_entry.pack(fill="x")

        self.compare_btn = tk.Button(self.input_container, text="Compare", command=self.compare_texts)
        self.compare_btn.pack(pady=(4, 0))

    # ----- Model loading -----
    def load_model(self):
        idx = self.model_menu.current()
        if idx < 0:
            return
        name, model_id, model_type = MODELS[idx]
        self.current_model_type = model_type
        self.model_loaded = False
        self.chat_history = []
        self.status_var.set(f"Loading {name}...")
        self.load_btn.config(state="disabled")

        if model_type == "embed":
            self.setup_embed_input()
        else:
            self.setup_chat_input()

        threading.Thread(target=self._load_model, args=(model_id, model_type), daemon=True).start()

    def _load_model(self, model_id, model_type):
        try:
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if model_type == "embed":
                self.model = AutoModel.from_pretrained(
                    model_id, cache_dir=MODEL_CACHE_DIR,
                    torch_dtype=torch.float32, trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, cache_dir=MODEL_CACHE_DIR,
                    torch_dtype=torch.float32, low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )

            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self.root.after(0, self._on_loaded, model_id)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _on_loaded(self, model_id):
        short = model_id.split("/")[-1]
        self.status_var.set(f"Ready: {short} ({self.device})")
        self.load_btn.config(state="normal")
        self.chat_display.insert("end", f"[Loaded {model_id} on {self.device}]\n\n")
        self.chat_display.see("end")

    def _on_error(self, error):
        self.status_var.set(f"Error: {error[:80]}")
        self.load_btn.config(state="normal")
        self.chat_display.insert("end", f"[Error: {error}]\n\n")
        self.chat_display.see("end")
        # Safely re-enable buttons that exist
        if hasattr(self, "send_btn"):
            self.send_btn.config(state="normal")
        if hasattr(self, "compare_btn"):
            self.compare_btn.config(state="normal")

    # ----- Chat -----
    def on_enter(self, event):
        if not (event.state & 0x0001):  # Shift not pressed
            self.send_message()
            return "break"

    def send_message(self):
        if not self.model_loaded:
            messagebox.showwarning("Loading", "Model is still loading!")
            return

        text = self.input_entry.get("1.0", "end-1c").strip()
        if not text:
            return

        self.input_entry.delete("1.0", "end")
        self.chat_display.insert("end", f"You: {text}\n")
        self.chat_display.see("end")
        self.send_btn.config(state="disabled")
        self.status_var.set("Generating...")

        if self.current_model_type == "chatml":
            threading.Thread(target=self._generate_chatml, args=(text,), daemon=True).start()
        else:
            threading.Thread(target=self._generate_base, args=(text,), daemon=True).start()

    def _generate_chatml(self, user_input):
        """SmolLM2 uses ChatML format."""
        try:
            messages = []
            for msg in self.chat_history[-4:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["assistant"]})
            messages.append({"role": "user", "content": user_input})

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=200,
                    temperature=0.7, top_p=0.9, top_k=50,
                    repetition_penalty=1.1, do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            if not response:
                response = "..."

            self.chat_history.append({"user": user_input, "assistant": response})
            self.root.after(0, self._show_response, response)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _generate_base(self, user_input):
        """Few-shot prompting for base models (LLaMmlein, Leviathan)."""
        try:
            few_shot = (
                "Frage: Hallo, wie geht es dir?\n"
                "Antwort: Mir geht es gut! Wie kann ich dir helfen?\n\n"
                "Frage: Was ist die Hauptstadt von Deutschland?\n"
                "Antwort: Die Hauptstadt von Deutschland ist Berlin.\n\n"
            )

            # BUG FIX: history was computed but never included in the prompt.
            history = ""
            for msg in self.chat_history[-2:]:
                history += f"Frage: {msg['user']}\nAntwort: {msg['assistant']}\n\n"

            # BUG FIX: few_shot no longer ends with "Frage: " — prompt is
            # assembled cleanly so history and current question don't merge.
            prompt = few_shot + history + f"Frage: {user_input}\nAntwort:"

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=150,
                    temperature=0.8, top_p=0.9, top_k=40,
                    repetition_penalty=1.2, do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            if "Frage:" in response:
                response = response.split("Frage:")[0].strip()
            if not response or len(response) < 3:
                response = "..."

            self.chat_history.append({"user": user_input, "assistant": response})
            self.root.after(0, self._show_response, response)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _show_response(self, response):
        self.chat_display.insert("end", f"AI: {response}\n\n")
        self.chat_display.see("end")
        self.send_btn.config(state="normal")
        self.status_var.set("Ready")
        self.input_entry.focus_set()

    # ----- Embedding mode -----
    def compare_texts(self):
        if not self.model_loaded:
            messagebox.showwarning("Loading", "Model is still loading!")
            return

        t1 = self.text1_entry.get("1.0", "end-1c").strip()
        t2 = self.text2_entry.get("1.0", "end-1c").strip()
        if not t1 or not t2:
            messagebox.showwarning("Input", "Please enter both texts.")
            return

        self.compare_btn.config(state="disabled")
        self.status_var.set("Computing similarity...")
        threading.Thread(target=self._calc_similarity, args=(t1, t2), daemon=True).start()

    def _calc_similarity(self, t1, t2):
        try:
            encoded = self.tokenizer(
                [t1, t2], padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

            # BUG FIX: use last_hidden_state instead of outputs[0] for
            # reliable access across model output types.
            emb = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            mask = encoded["attention_mask"].unsqueeze(-1).expand(emb.size()).float()
            embeddings = torch.sum(emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            score = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

            result = f"Similarity: {score:.4f} ({score*100:.1f}%)"
            self.root.after(0, self._show_sim_result, t1, t2, result)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _show_sim_result(self, t1, t2, result):
        self.chat_display.insert("end", f"A: {t1}\nB: {t2}\n{result}\n\n")
        self.chat_display.see("end")
        self.compare_btn.config(state="normal")
        self.status_var.set("Ready")

    # ----- Clear -----
    def clear_chat(self):
        self.chat_display.delete("1.0", "end")
        self.chat_history = []
        self.chat_display.insert("end", "[Cleared]\n\n")
        self.chat_display.see("end")


def main():
    root = tk.Tk()
    app = LlammleinChat(root)
    root.mainloop()


if __name__ == "__main__":
    main()