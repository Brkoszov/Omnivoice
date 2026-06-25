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
def generate_audiobook(selected_chapters, all_chapters, ref_audio_path, language="Auto", instruct=""):
    if not all_chapters:
        return "No chapters data available. Please upload an EPUB first."
    
    if not selected_chapters:
        return "Please select at least one chapter to generate."

    status_log = f"Starting generation for {len(selected_chapters)} selected chapters...\n"
    
    # Create a map of title -> text from all_chapters
    chapter_map = {title: text for title, text in all_chapters}
    
    for title in selected_chapters:
        text = chapter_map.get(title)
        if not text:
            status_log += f"Error: Could not find text for chapter {title}\n"
            continue
            
        status_log += f"\n--- Generating: {title} ---\n"
        
        # Vytvoření adresáře pro danou kapitolu
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:30]
        chapter_dir = os.path.join(DRIVE_OUTPUT_DIR, f"Chapter_{safe_title}")
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
            chapter_final_path = os.path.join(DRIVE_OUTPUT_DIR, f"{safe_title}.wav")
            sf.write(chapter_final_path, full_chapter_waveform, sampling_rate)
            status_log += f"Saved: {chapter_final_path}\n"
        else:
            status_log += f"No audio generated for {title}.\n"

    return status_log + "\n✅ Selected chapters generation complete!"

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="OmniVoice Audiobook Generator") as demo:
        gr.Markdown("# 📚 OmniVoice Audiobook Generator")
        gr.Markdown("Nahrajte EPUB knihu a referenční audio pro automatickou syntézu vybraných kapitol.")
        
        # State to store parsed chapters: [(title, text), ...]
        chapters_state = gr.State([])

        with gr.Row():
            with gr.Column():
                epub_input = gr.File(label="EPUB soubor", file_types=[".epub"])
                
                # Chapter selection UI
                chapter_selection = gr.CheckboxGroup(
                    label="Vyberte kapitoly pro generování", 
                    choices=[], 
                    value=[]
                )
                
                ref_audio_input = gr.Audio(label="Referenční audio (klonování)", type="filepath")
                lang_input = gr.Textbox(label="Jazyk", value="Auto", placeholder="Auto nebo např. English")
                instr_input = gr.Textbox(label="Instrukce (volitelně)", placeholder="Např. 'Slowly, calm voice'")
                generate_btn = gr.Button("Generovat vybrané kapitoly", variant="primary")
            
            with gr.Column():
                output_log = gr.Textbox(label="Průběh generování", lines=20, interactive=False)
        
        # Function to handle EPUB upload and update chapter list
        def on_epub_upload(file_path):
            if not file_path:
                return [], []
            
            chapters = extract_chapters_from_epub(file_path)
            if not chapters:
                return gr.update(value=[], info="Error parsing EPUB"), []
            
            titles = [title for title, text in chapters]
            return gr.update(choices=titles, value=titles), chapters

        epub_input.change(
            fn=on_epub_upload,
            inputs=[epub_input],
            outputs=[chapter_selection, chapters_state]
        )
        
        generate_btn.click(
            fn=generate_audiobook,
            inputs=[chapter_selection, chapters_state, ref_audio_input, lang_input, instr_input],
            outputs=output_log
        )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(share=True)
