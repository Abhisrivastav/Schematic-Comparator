# Schematic Comparator

AI-powered tool to compare two board schematics (Reference vs Customer) based on the same SoC.

## Features

- **Upload two PDFs** — Reference board + Customer board (drag-and-drop)
- **AI extraction** — GPT-4o Vision parses each schematic page-by-page (works on image-based and text-based PDFs)
- **Interface-by-interface diff report** — USB, PCIe, DDR, I2C, SPI, UART, HDMI, Power, GPIO, and more
- **Downloadable HTML report** — formatted report with match scores per interface
- **Chatbot** — ask natural-language questions about both schematics
- **Password-protected PDF support** — prompts for password if required

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Abhisrivastav/Schematic-Comparator.git
cd Schematic-Comparator

# 2. Setup (creates .venv and installs packages)
setup.bat

# 3. Configure
copy .env.example .env
# Edit .env — add your EXPERGPT_API_KEY

# 4. Launch
start.bat
```

Open **http://127.0.0.1:5000** in your browser.

## Workflow

```
Step 1 — Upload    →  Drop Reference PDF + Customer PDF
Step 2 — Extract   →  AI reads each schematic (Vision mode for image-based)
Step 3 — Compare   →  AI generates interface-by-interface diff + HTML report
Step 4 — Chatbot   →  Ask questions about both schematics
```

## Configuration

| Variable | Value |
|---|---|
| `EXPERGPT_BASE_URL` | `https://expertgpt.intel.com/v1` |
| `EXPERGPT_API_KEY` | Your `pak_...` key |

> `expertgpt.intel.com` is an internal Intel address — requires Intel VPN or on-site network.

## Tech Stack

| Component | Library |
|---|---|
| Web framework | Flask 3.x |
| PDF text extraction | PyMuPDF (fitz), pdfplumber |
| AI backend | OpenAI SDK → Intel ExpertGPT (GPT-4o) |
| HTTP client | httpx (`trust_env=False` bypasses corporate proxy for internal addresses) |
| Environment | python-dotenv |
