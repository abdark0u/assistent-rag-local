# Assistent documental local (RAG)

Assistent de preguntes i respostes sobre documentació tècnica interna (BCN Serveis TIC) que s'executa **completament en local**: cap document ni pregunta surt de la màquina.

Respon **només** amb la informació dels documents carregats, cita els fragments utilitzats i es nega a respondre quan la documentació no conté la resposta.

## Arquitectura

| Peça | Eina |
|------|------|
| Interfície | Streamlit |
| Orquestració RAG | LlamaIndex |
| Base vectorial | Chroma (persistent a `data/chroma`) |
| LLM | Ollama `qwen3:8b` |
| Embeddings | Ollama `qwen3-embedding:0.6b` |
| Extracció PDF | PyMuPDF / pymupdf4llm + Tesseract OCR per a pàgines sense text |

Flux: PDF/Markdown → text (OCR si cal) → fragments de ~250 paraules → embeddings → Chroma. En preguntar, es recuperen els 6 fragments més semblants; si cap supera el llindar de similitud (0.50) es respon directament que no hi ha informació, sense cridar el model.

## Funcionalitats

- Pujada de PDF o Markdown (màx. 10 documents, 5 MB, 100 pàgines) i eliminació des de la barra lateral.
- OCR automàtic (català + castellà) per a PDF escanejats.
- Respostes en l'idioma de la pregunta (català o castellà), amb streaming.
- Cites `[1]`, `[2]`... i panell amb les fonts utilitzades.
- Negativa determinista per a preguntes fora de la documentació i protecció bàsica contra injecció de prompts.
- Accés protegit amb contrasenya (variable d'entorn `APP_PASSWORD`).

## Requisits

- Python 3.10+
- [Ollama](https://ollama.com)
- Tesseract amb idiomes `cat` i `spa` (opcional, només per a PDF sense text)

macOS (Homebrew):

```bash
brew install ollama tesseract tesseract-lang
```

## Instal·lació

```bash
git clone https://github.com/abdark0u/assistent-rag-local.git
cd assistent-rag-local

python3 -m venv venv
venv/bin/pip install -r requirements.txt

ollama serve &            # si no està ja en marxa
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

## Ús

```bash
./iniciar.sh
```

Demana la contrasenya de l'aplicació i obre Streamlit a `http://127.0.0.1:8501`. Després, puja documents des de la barra lateral i prem **Processar documents**.

## Documents i preguntes d'exemple

- `docs_exemple/`: 10 PDF de mostra sobre serveis de xarxa (Apache, DHCP, DNS, FTP, NFS, Samba, SSH...).
- `preguntes_demo.txt`: preguntes de prova amb la resposta esperada, incloses preguntes que s'han de negar (fora d'àmbit, sense sentit i intents d'injecció).

## Configuració

Els paràmetres són a l'inici d'`app.py` (`LLM_MODEL`, `EMBED_MODEL`, `CHUNK_SIZE`, `TOP_K`, `THRESHOLD`, límits de pujada, idiomes OCR).
