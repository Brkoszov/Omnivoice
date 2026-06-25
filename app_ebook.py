import os
import sys
import re
import numpy as np
import torch
import soundfile as sf
import gradio as gr
from ebooklib import epub
from bs4 import BeautifulSoup
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Cesta k modelu pro lokální Windows prostředí
CHECKPOINT = r"F:\AI_Projekt\omni\OmniVoice" 
DRIVE_OUTPUT_DIR = "./OmniVoice_Audiobooks"

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
print(f"Loading model from {CHECKPOINT} to cuda ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map="cuda",
    dtype=torch.float16,
    load_asr=True,
)
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# EPUB Parsing Logic
# ---------------------------------------------------------------------------
def extract_chapters_from_epub(epub_path):
    print(f"Parsing EPUB: {epub_path}")
    try:
        book = epub.read_epub(epub_path)
    except Exception as e:
        print(f"Error reading EPUB: {e}")
        return []

    chapters = []
    current_chapter_title = "Introduction"
    current_chapter_text = []

    # ITEM_DOCUMENT = 9
    for item in book.get_items_of_type(9): 
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        
        # Hledáme nadpisy pro definici kapitol a odstavce pro text
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Pokud už máme text z předchozí kapitoly, uložíme ho
                if current_chapter_text:
                    chapters.append((current_chapter_title, " ".join(current_chapter_text)))
                
                current_chapter_title = element.get_text().strip()
                current_chapter_text = []
            elif element.name == 'p':
                text = element.get_text().strip()
                if text:
                    current_chapter_text.append(text)
        
    # Poslední kapitolu přidáme po skončení cyklu
    if current_chapter_text:
        chapters.append((current_chapter_title, " ".join(current_chapter_text)))
    
    return chapters

def split_text_into_chunks(text, max_chars=200):
    """Dělí dlouhý text na menší části, aby nedošlo k přetečení VRAM."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# ---------------------------------------------------------------------------
# Generation Logic
# ---------------------------------------------------------------------------
def generate_audiobook(epub_path, ref_audio_path, language="Auto", instruct=""):
    # 1. Extrakce kapitol
    chapters = extract_chapters_from_epub(epub_path)
    if not chapters:
        return "No chapters found or error reading EPUB."

    status_log = f"Found {len(chapters)} chapters. Starting generation...\n"
    
    for i, (title, text) in enumerate(chapters):
        status_log += f"\n--- Chapter {i+1}: {title} ---\n"
        
        # Vytvoření adresáře pro danou kapitolu
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:30]
        chapter_dir = os.path.join(DRIVE_OUTPUT_DIR, f"Chapter_{i+1}_{safe_title}")
        os.makedirs(chapter_dir, exist_ok=True)

        # Konfigurace generování
        gen_config = OmniVoiceGenerationConfig(
            num_step=32,
            guidance_scale=2.0,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
        )

        # Voice Cloning Prompt
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=None 
        )

        # Procesování textu po čunkách
        chunks = split_text_into_chunks(text)
        
        chapter_waveforms = []
        for j, chunk in enumerate(chunks):
            kw = dict(
                text=chunk.strip(), 
                language=language if language != "Auto" else None, 
                generation_config=gen_config,
                voice_clone_prompt=voice_clone_prompt
            )
            
            if instruct:
                kw["instruct"] = instruct.strip()

            try:
                audio = model.generate(**kw)
                waveform = (audio[0] * 32767).astype(np.int16)
                chapter_waveforms.append(waveform)
                
                out_path = os.path.join(chapter_dir, f"chunk_{j+1:03d}.wav")
                sf.write(out_path, waveform, sampling_rate)
            except Exception as e:
                status_log += f"Error in chunk {j+1}: {e}\n"

        # Slučování celé kapitoly do jednoho souboru
        if chapter_waveforms:
            full_chapter_waveform = np.concatenate(chapter_waveforms)
            chapter_final_path = os.path.join(DRIVE_OUTPUT_DIR, f"Chapter_{i+1:02d}_{safe_title}.wav")
            sf.write(chapter_final_path, full_chapter_waveform, sampling_rate)
            status_log += f"Saved: {chapter_final_path}\n"
        else:
            status_log += f"No audio generated for Chapter {i+1}.\n"

    return status_log + "\n✅ Audiobook generation complete!"

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="OmniVoice Audiobook Generator") as demo:
        gr.Markdown("# 📚 OmniVoice Audiobook Generator")
        gr.Markdown("Nahrajte EPUB knihu a referenční audio pro automatickou syntézu celé knihy.")
        
        with gr.Row():
            with gr.Column():
                epub_input = gr.File(label="EPUB soubor", file_types=[".epub"])
                ref_audio_input = gr.Audio(label="Referenční audio (klonování)", type="filepath")
                lang_input = gr.Textbox(label="Jazyk", value="Auto", placeholder="Auto nebo např. English")
                instr_input = gr.Textbox(label="Instrukce (volitelně)", placeholder="Např. 'Slowly, calm voice'")
                generate_btn = gr.Button("Generovat audioknihu", variant="primary")
            
            with gr.Column():
                output_log = gr.Textbox(label="Průběh generování", lines=20, interactive=False)
        
        generate_btn.click(
            fn=generate_audiobook,
            inputs=[epub_input, ref_audio_input, lang_input, instr_input],
            outputs=output_log
        )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(share=True)
