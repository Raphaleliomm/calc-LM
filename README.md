# calc-LM

Zwei winzige Language Models, von Grund auf trainiert auf einem No-World-Knowledge-Datensatz (16.021 Chat-Beispiele, Logik-Puzzles/Reasoning). Beide nutzen **1.58-bit ternäre Gewichte** (BitNet b1.58, STE-Quantisierung), 64 Token Context und ein 2048-Token BPE-Vokabular (trainiert nur auf den Daten). Nur Assistant-Tokens werden als LM-Ziel maskiert (mit korrektem Target-Shift).

## Direkter Vergleich

|                              | `Calc_LMs/firsttry`                       | `Calc_LMs/secondtry`                              |
|------------------------------|-------------------------------------------|---------------------------------------------------|
| **Architektur**              | Standard Transformer                       | Recurrent Transformer + Attention Residuals       |
| **Schichten**                | 8 einzigartige                            | 8 einzigartige, davon 2 geloopt ×4 → effektive Tiefe 14 |
| **Residual**                 | fixer +1-Skip                             | Attention über Tiefe (AttnRes): gelernte Pseudo-Query attendiert über alle vorherigen Layer-Outputs |
| **Parameter**                | ~5.26M                                    | ~6.84M                                            |
| **Val-Loss (best.pt)**       | **3.8686**                                | **3.9543** (nach 8000 Steps)                      |
| **Perplexity**               | **47.88**                                 | **52.16**                                         |
| **Token-Accuracy**           | **25.77%**                                | **25.26%**                                        |
| **Trainingsschritte**        | 4000                                      | 4000 + 4000 Resume = 8000                        |

## Nutzung

```powershell
# Tokenizer (einmal je Projekt)
python Calc_LMs/firsttry/train_tokenizer.py
python Calc_LMs/secondtry/train_tokenizer.py

# Training (secondtry: Resume vom Checkpoint)
$env:TRAIN_STEPS="8000"; $env:RESUME_CKPT="final.pt"; python Calc_LMs/secondtry/train.py

# Inferenz
python Calc_LMs/firsttry/inference.py "Dein Prompt" --ckpt best.pt
python Calc_LMs/secondtry/inference.py "Dein Prompt" --ckpt best.pt

# Eval
python Calc_LMs/firsttry/eval.py --ckpt best.pt --num-examples 200
python Calc_LMs/secondtry/eval.py --ckpt best.pt --num-examples 200
```

Weitere Details zu Architektur und Dateien: siehe Kommentare in `model.py`/`train.py` je Projekt.

## Git

Trainingsgewichte (`Calc_LMs/**/checkpoints/*.pt`) sind git-ignored; `tokenizer.json` und `config.json` bleiben versioniert.