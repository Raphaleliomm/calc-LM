import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import os
import sys
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")

# Model definitions: (display_name, model_id, type, description, color)
MODELS = {
    "chat": [
        ("💬 LLaMmlein 120M", "LSX-UniWue/LLaMmlein_120M", "base", "Deutsches Llama-Basismodell (Few-Shot)", "#b388ff"),
        ("🤖 SmolLM2 135M Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "chatml", "Chat-optimiertes Modell (Englisch)", "#4fc3f7"),
        ("🧪 Leviathan 100M Instruct", "ShiningSon/Leviathan-MLGRU-100M-TinyStories-Instruct-v08a", "base", "GRU-basiertes Story-Modell (Englisch)", "#ffb74d"),
    ],
    "embed": [
        ("🔍 LLaMmlein2Vec 120M", "LSX-UniWue/LLaMmlein2Vec_120M", "embed", "Bidirektionaler Embedding-Vergleich", "#4fc3f7"),
    ]
}

class LlammleinChat:
    def __init__(self, root):
        self.root = root
        self.root.title("🦙 LLaMmlein Model Hub")
        self.root.geometry("800x700")
        self.root.configure(bg="#1a1a1a")
        self.root.minsize(600, 500)
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.current_model_id = None
        self.mode = "chat"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chat_history = []
        
        self.create_widgets()
        self.show_model_selection()
    
    def create_widgets(self):
        bg = "#1a1a1a"
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Willkommen beim LLaMmlein Model Hub!")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            bg="#111111", fg="#888888", anchor="w",
            padx=12, pady=4, font=("Segoe UI", 9)
        )
        status_bar.pack(fill="x", side="bottom")
        
        self.main_frame = tk.Frame(self.root, bg=bg)
        self.main_frame.pack(fill="both", expand=True, padx=12, pady=(10, 5))
        
        # Header with model selector
        self.header_frame = tk.Frame(self.main_frame, bg=bg)
        self.header_frame.pack(fill="x", pady=(0, 8))
        
        self.title_label = tk.Label(
            self.header_frame, text="🦙 LLaMmlein Model Hub",
            bg=bg, fg="#ffffff", font=("Segoe UI", 16, "bold")
        )
        self.title_label.pack(side="left")
        
        self.model_badge = tk.Label(
            self.header_frame, text="Kein Modell geladen",
            bg="#333333", fg="#888888",
            font=("Segoe UI", 8), padx=8, pady=2
        )
        self.model_badge.pack(side="left", padx=(10, 0), pady=(4, 0))
        
        self.change_btn = tk.Button(
            self.header_frame, text="🔄 Modell wechseln",
            command=self.change_model,
            bg="#333333", fg="#aaaaaa",
            font=("Segoe UI", 8), borderwidth=0,
            cursor="hand2", padx=8, pady=2
        )
        self.change_btn.pack(side="right", pady=(4, 0))
        
        # Separator
        sep = tk.Frame(self.main_frame, bg="#333333", height=1)
        sep.pack(fill="x", pady=(0, 8))
        
        # Chat/display area
        chat_frame = tk.Frame(self.main_frame, bg="#141414")
        chat_frame.pack(fill="both", expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap="word", state="disabled",
            bg="#141414", fg="#e0e0e0", font=("Segoe UI", 11),
            insertbackground="#ffffff", padx=14, pady=12,
            borderwidth=0, highlightthickness=0,
            relief="flat"
        )
        self.chat_display.pack(fill="both", expand=True)
        
        # Text tags
        self.chat_display.tag_config("system", foreground="#666666", font=("Segoe UI", 9, "italic"))
        self.chat_display.tag_config("user", foreground="#4fc3f7", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground="#b388ff", font=("Segoe UI", 11))
        self.chat_display.tag_config("error", foreground="#ef5350", font=("Segoe UI", 11))
        self.chat_display.tag_config("info", foreground="#66bb6a", font=("Segoe UI", 10))
        
        # Input container - will be populated by setup methods
        self.input_frame = tk.Frame(self.main_frame, bg=bg)
        self.input_frame.pack(fill="x", pady=(8, 0))
    
    def setup_chat_ui(self):
        """Single-line chat input"""
        for w in self.input_frame.winfo_children():
            w.destroy()
        
        entry_frame = tk.Frame(self.input_frame, bg="#2a2a2a", bd=1, relief="flat")
        entry_frame.pack(fill="x")
        
        self.input_entry = tk.Text(
            entry_frame, height=2, wrap="word",
            bg="#2a2a2a", fg="#e0e0e0",
            font=("Segoe UI", 11), insertbackground="#ffffff",
            padx=12, pady=8, borderwidth=0, relief="flat"
        )
        self.input_entry.pack(side="left", fill="both", expand=True)
        self.input_entry.bind("<Return>", self.on_enter)
        self.input_entry.bind("<Shift-Return>", lambda e: None)
        self.input_entry.focus_set()
        
        btn_frame = tk.Frame(entry_frame, bg="#2a2a2a")
        btn_frame.pack(side="right", padx=(0, 4), pady=4)
        
        self.send_btn = tk.Button(
            btn_frame, text="➤", command=self.send_message,
            bg="#5a3a8a", fg="#ffffff",
            font=("Segoe UI", 14), borderwidth=0,
            cursor="hand2", padx=16, pady=6,
            activebackground="#7a5aaa"
        )
        self.send_btn.pack()
        
        clear_btn = tk.Button(
            btn_frame, text="↺", command=self.clear_chat,
            bg="#3a3a3a", fg="#888888",
            font=("Segoe UI", 12), borderwidth=0,
            cursor="hand2", padx=12, pady=4,
            activebackground="#555555"
        )
        clear_btn.pack(pady=(3, 0))
    
    def setup_embed_ui(self):
        """Two text fields for embedding comparison"""
        for w in self.input_frame.winfo_children():
            w.destroy()
        
        inner = tk.Frame(self.input_frame, bg="#1a1a1a")
        inner.pack(fill="x")
        
        t1_frame = tk.Frame(inner, bg="#1a1a1a")
        t1_frame.pack(fill="x", pady=(0, 4))
        tk.Label(t1_frame, text="Text A:", bg="#1a1a1a", fg="#4fc3f7",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.text1_entry = tk.Text(
            t1_frame, height=2, wrap="word",
            bg="#2a2a2a", fg="#e0e0e0",
            font=("Segoe UI", 11), insertbackground="#ffffff",
            padx=10, pady=6, borderwidth=0, relief="flat"
        )
        self.text1_entry.pack(fill="x")
        
        t2_frame = tk.Frame(inner, bg="#1a1a1a")
        t2_frame.pack(fill="x", pady=(4, 6))
        tk.Label(t2_frame, text="Text B:", bg="#1a1a1a", fg="#4fc3f7",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.text2_entry = tk.Text(
            t2_frame, height=2, wrap="word",
            bg="#2a2a2a", fg="#e0e0e0",
            font=("Segoe UI", 11), insertbackground="#ffffff",
            padx=10, pady=6, borderwidth=0, relief="flat"
        )
        self.text2_entry.pack(fill="x")
        
        btn_frame = tk.Frame(inner, bg="#1a1a1a")
        btn_frame.pack(fill="x", pady=(4, 0))
        
        self.compare_btn = tk.Button(
            btn_frame, text="🔍 Ähnlichkeit berechnen",
            command=self.compare_texts,
            bg="#5a3a8a", fg="#ffffff",
            font=("Segoe UI", 10, "bold"), borderwidth=0,
            cursor="hand2", padx=16, pady=6
        )
        self.compare_btn.pack(side="left")
        
        self.text1_entry.focus_set()
    
    def show_model_selection(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("🦙 Modell auswählen")
        dialog.geometry("520x520")
        dialog.configure(bg="#1a1a1a")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center on parent
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 520) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 520) // 2
        dialog.geometry(f"+{max(0,x)}+{max(0,y)}")
        
        frame = tk.Frame(dialog, bg="#1a1a1a", padx=20, pady=20)
        frame.pack(fill="both", expand=True)
        
        tk.Label(frame, text="🦙 Wähle ein Modell", bg="#1a1a1a", fg="#ffffff",
                 font=("Segoe UI", 18, "bold")).pack(pady=(0, 16))
        
        selected = tk.StringVar(value="")
        
        # Chat models section
        tk.Label(frame, text="— Chat-Modelle —", bg="#1a1a1a", fg="#888888",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        
        for name, model_id, mtype, desc, color in MODELS["chat"]:
            card = tk.Frame(frame, bg="#2a2a2a", bd=1, relief="flat", padx=14, pady=10)
            card.pack(fill="x", pady=3)
            
            rb = tk.Radiobutton(
                card, text=f"{name} ({model_id.split('/')[-1]})",
                variable=selected, value=f"chat|{model_id}|{mtype}",
                bg="#2a2a2a", fg=color, selectcolor="#1a1a1a",
                font=("Segoe UI", 11, "bold"),
                activebackground="#2a2a2a", activeforeground=color
            )
            rb.pack(anchor="w")
            
            tk.Label(card, text=f"  {desc}  |  {model_id}",
                     bg="#2a2a2a", fg="#888888",
                     font=("Segoe UI", 8), justify="left"
            ).pack(anchor="w", padx=(26, 0))
        
        # Embedding model
        tk.Label(frame, text="— Analyse-Modell —", bg="#1a1a1a", fg="#888888",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(16, 8))
        
        for name, model_id, mtype, desc, color in MODELS["embed"]:
            card = tk.Frame(frame, bg="#2a2a2a", bd=1, relief="flat", padx=14, pady=10)
            card.pack(fill="x", pady=3)
            
            rb = tk.Radiobutton(
                card, text=f"{name}",
                variable=selected, value=f"embed|{model_id}|{mtype}",
                bg="#2a2a2a", fg=color, selectcolor="#1a1a1a",
                font=("Segoe UI", 11, "bold"),
                activebackground="#2a2a2a", activeforeground=color
            )
            rb.pack(anchor="w")
            
            tk.Label(card, text=f"  {desc}  |  {model_id}",
                     bg="#2a2a2a", fg="#888888",
                     font=("Segoe UI", 8), justify="left"
            ).pack(anchor="w", padx=(26, 0))
        
        # Select first by default
        selected.set("chat|LSX-UniWue/LLaMmlein_120M|base")
        
        def confirm():
            val = selected.get()
            if val:
                parts = val.split("|")
                if len(parts) == 3:
                    self.mode = parts[0]
                    model_id = parts[1]
                    model_type = parts[2]
                    dialog.destroy()
                    self.load_selected_model(model_id, model_type)
        
        btn_frame = tk.Frame(frame, bg="#1a1a1a")
        btn_frame.pack(fill="x", pady=(12, 0))
        
        tk.Button(btn_frame, text="✅ Auswählen & Laden",
                  command=confirm, bg="#5a3a8a", fg="#ffffff",
                  font=("Segoe UI", 11, "bold"), borderwidth=0,
                  cursor="hand2", padx=20, pady=8
        ).pack(side="right")
    
    def load_selected_model(self, model_id, model_type):
        self.current_model_id = model_id
        self.current_model_type = model_type
        self.chat_history = []
        self.model_loaded = False
        
        # Update UI
        short_name = model_id.split("/")[-1]
        if self.mode == "chat":
            self.title_label.config(text=f"💬 {short_name}")
            self.setup_chat_ui()
            self.model_badge.config(text="Lade...", bg="#5a3a8a", fg="#ffffff")
        else:
            self.title_label.config(text=f"🔍 {short_name}")
            self.setup_embed_ui()
            self.model_badge.config(text="Lade...", bg="#5a3a8a", fg="#ffffff")
        
        self.update_display("system", f"Lade {model_id}...\nGerät: {self.device.upper()}\n\n")
        self.status_var.set(f"📥 Lade {short_name} ({model_id})...")
        
        threading.Thread(target=self._load_model, args=(model_id, model_type), daemon=True).start()
    
    def _load_model(self, model_id, model_type):
        try:
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.status_var.set(f"📥 Lade Gewichte in RAM...")
            
            if model_type == "embed":
                self.model = AutoModel.from_pretrained(
                    model_id, cache_dir=MODEL_CACHE_DIR,
                    torch_dtype=torch.float32, trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, cache_dir=MODEL_CACHE_DIR,
                    torch_dtype=torch.float32, low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            self.root.after(0, self._on_model_loaded, model_id)
            
        except Exception as e:
            self.root.after(0, self._on_model_error, str(e))
    
    def _on_model_loaded(self, model_id):
        short = model_id.split("/")[-1]
        self.model_badge.config(text=f"✅ {short}", bg="#2e7d32", fg="#ffffff")
        self.status_var.set(f"✅ Bereit: {short}")
        
        if self.mode == "chat":
            self.send_btn.config(state="normal")
            self.update_display("info", f"✅ {model_id} geladen!\n")
            self.update_display("system", "💬 Schreibe eine Nachricht und drücke Enter\n\n")
        else:
            self.compare_btn.config(state="normal")
            self.update_display("info", f"✅ {model_id} geladen!\n")
            self.update_display("system", "🔍 Vergleiche zwei Texte auf semantische Ähnlichkeit\n\n")
    
    def _on_model_error(self, error):
        self.model_badge.config(text="❌ Fehler", bg="#c62828", fg="#ffffff")
        self.status_var.set(f"❌ Fehler: {error[:50]}")
        self.update_display("error", f"Fehler beim Laden: {error}\n")
    
    def change_model(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.chat_history = []
        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.config(state="disabled")
        self.title_label.config(text="🦙 LLaMmlein Model Hub")
        self.model_badge.config(text="Kein Modell", bg="#333333", fg="#888888")
        self.show_model_selection()
    
    def update_display(self, tag, text):
        self.chat_display.config(state="normal")
        self.chat_display.insert("end", text, tag)
        self.chat_display.see("end")
        self.chat_display.config(state="disabled")
    
    def on_enter(self, event):
        if not (event.state & 0x0001):  # Shift not pressed
            self.send_message()
            return "break"
    
    # ----- Chat Mode -----
    def send_message(self):
        if not self.model_loaded:
            messagebox.showwarning("⏳", "Modul wird noch geladen!")
            return
        
        text = self.input_entry.get("1.0", "end-1c").strip()
        if not text:
            return
        
        self.input_entry.delete("1.0", "end")
        self.update_display("user", f"Du: {text}\n")
        self.send_btn.config(state="disabled", text="⏳")
        self.status_var.set("💭 Generiere...")
        
        if self.current_model_type == "chatml":
            threading.Thread(target=self._generate_chatml, args=(text,), daemon=True).start()
        else:
            threading.Thread(target=self._generate_base, args=(text,), daemon=True).start()
    
    def _generate_chatml(self, user_input):
        """SmolLM2 uses ChatML format"""
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
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            
            if not response:
                response = "I'm not sure how to respond to that."
            
            self.chat_history.append({"user": user_input, "assistant": response})
            self.root.after(0, self._show_response, response)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _generate_base(self, user_input):
        """Few-shot prompting for base models (LLaMmlein, Leviathan)"""
        try:
            examples = """Frage: Hallo, wie geht es dir?
Antwort: Mir geht es gut! Wie kann ich dir helfen?

Frage: Was ist die Hauptstadt von Deutschland?
Antwort: Die Hauptstadt von Deutschland ist Berlin.

Frage: """
            
            history = ""
            for msg in self.chat_history[-2:]:
                history += f"Frage: {msg['user']}\nAntwort: {msg['assistant']}\n\n"
            
            prompt = examples + user_input + "\nAntwort:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=150,
                    temperature=0.8, top_p=0.9, top_k=40,
                    repetition_penalty=1.2, do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            
            if "Frage:" in response:
                response = response.split("Frage:")[0].strip()
            
            if not response or len(response) < 3:
                response = "Interessante Frage! Kannst du mehr Details geben?"
            
            self.chat_history.append({"user": user_input, "assistant": response})
            self.root.after(0, self._show_response, response)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _show_response(self, response):
        self.update_display("assistant", f"KI: {response}\n\n")
        self.send_btn.config(state="normal", text="➤")
        self.status_var.set("✅ Bereit")
        self.input_entry.focus_set()
    
    def _show_error(self, error):
        self.update_display("error", f"Fehler: {error}\n\n")
        if self.mode == "chat":
            self.send_btn.config(state="normal", text="➤")
        else:
            self.compare_btn.config(state="normal", text="🔍 Ähnlichkeit berechnen")
        self.status_var.set("❌ Fehler")
    
    # ----- Embedding Mode -----
    def compare_texts(self):
        if not self.model_loaded:
            messagebox.showwarning("⏳", "Modul wird noch geladen!")
            return
        
        t1 = self.text1_entry.get("1.0", "end-1c").strip()
        t2 = self.text2_entry.get("1.0", "end-1c").strip()
        if not t1 or not t2:
            messagebox.showwarning("⚠️", "Bitte beide Texte eingeben.")
            return
        
        self.compare_btn.config(state="disabled", text="⏳")
        self.status_var.set("🔍 Berechne Ähnlichkeit...")
        threading.Thread(target=self._calc_similarity, args=(t1, t2), daemon=True).start()
    
    def _calc_similarity(self, t1, t2):
        try:
            def mean_pooling(out, mask):
                emb = out[0]
                expanded = mask.unsqueeze(-1).expand(emb.size()).float()
                return torch.sum(emb * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)
            
            encoded = self.tokenizer(
                [t1, t2], padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
            
            embeddings = mean_pooling(outputs, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            score = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
            
            if score > 0.9:
                label = "🔵 Sehr ähnlich (fast identisch)"
            elif score > 0.7:
                label = "🟢 Ähnlich"
            elif score > 0.5:
                label = "🟡 Teilweise ähnlich"
            elif score > 0.3:
                label = "🟠 Kaum ähnlich"
            else:
                label = "🔴 Sehr unterschiedlich"
            
            result = f"Ähnlichkeit: {score:.4f} ({score*100:.1f}%)\n{label}"
            self.root.after(0, self._show_sim_result, t1, t2, result)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _show_sim_result(self, t1, t2, result):
        self.update_display("user", f"A: {t1}\nB: {t2}\n")
        self.update_display("info", f"🔍 {result}\n\n")
        self.compare_btn.config(state="normal", text="🔍 Ähnlichkeit berechnen")
        self.status_var.set("✅ Bereit")
    
    # ----- Clear -----
    def clear_chat(self):
        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.config(state="disabled")
        self.chat_history = []
        self.update_display("system", "🔄 Zurückgesetzt!\n\n")

def main():
    root = tk.Tk()
    app = LlammleinChat(root)
    root.mainloop()

if __name__ == "__main__":
    main()