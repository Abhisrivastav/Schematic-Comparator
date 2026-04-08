"""
Schematic Comparator — Flask application
=========================================
Upload two board schematics (Reference + Customer) based on the same SoC.

Features:
  1. Interface-by-interface diff report (USB, PCIe, DDR, I2C, SPI, Power, …)
  2. Downloadable HTML report
  3. Chatbot to query both schematics

AI backend  : Intel ExpertGPT (https://expertgpt.intel.com/v1) — GPT-4o Vision
PDF modes   : Text extraction (text-based PDFs) OR Vision (image-based PDFs)
"""

import os
import re
import json
import base64
import uuid
import shutil
import io
import hashlib
from pathlib import Path
from datetime import datetime

import fitz                        # PyMuPDF
import pdfplumber
import httpx
from flask import (Flask, request, render_template, jsonify,
                   session, send_file, make_response)
from dotenv import load_dotenv
from openai import OpenAI

# ── Bootstrap ──────────────────────────────────────────────────────────────
load_dotenv()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# SSL cert bundle (Intel TLS-inspection CA)
_local_cert  = os.path.join(BASE_DIR, "intel_certs.pem")
_cert_bundle = _local_cert if os.path.exists(_local_cert) else True
if isinstance(_cert_bundle, str):
    os.environ.setdefault("SSL_CERT_FILE",      _cert_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _cert_bundle)
    print(f"[SSL]  Cert bundle: {os.path.basename(_cert_bundle)}")

EXPERGPT_KEY  = os.getenv("EXPERGPT_API_KEY",  "")
EXPERGPT_BASE = os.getenv("EXPERGPT_BASE_URL",  "https://expertgpt.intel.com/v1")
USE_AI        = bool(EXPERGPT_KEY)

# PDF extraction settings
TEXT_THRESHOLD  = 80    # avg chars/page — below this → Vision mode
RENDER_DPI      = 150   # DPI for page-to-image rendering
MAX_PAGES_VISION = 30   # max pages to send to Vision API

# In-memory chat history store  { session_id: [ {role, content}, ... ] }
_chat_histories: dict[str, list] = {}

# ── AI Client Factory ────────────────────────────────────────────────────────
def get_ai_client() -> OpenAI:
    cert = (_cert_bundle
            if isinstance(_cert_bundle, str) and os.path.exists(_cert_bundle)
            else True)
    return OpenAI(
        api_key=EXPERGPT_KEY,
        base_url=EXPERGPT_BASE,
        http_client=httpx.Client(verify=cert, timeout=120, trust_env=False),
    )


# ── Session helpers ──────────────────────────────────────────────────────────
def get_session_dir() -> str:
    """Return (and create) a per-session upload directory."""
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    d = os.path.join(UPLOAD_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return d


def session_file(name: str) -> str:
    return os.path.join(get_session_dir(), name)


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── PDF helpers ───────────────────────────────────────────────────────────────
def is_pdf_password_protected(path: str) -> dict:
    try:
        doc = fitz.open(path)
        enc = doc.is_encrypted
        needs = (enc and doc.authenticate("") == 0)
        doc.close()
        return {"encrypted": enc, "needs_password": needs}
    except Exception as e:
        return {"encrypted": False, "needs_password": False, "error": str(e)}


def open_pdf(path: str, password: str = "") -> fitz.Document:
    doc = fitz.open(path)
    if doc.is_encrypted:
        if doc.authenticate(password) == 0:
            doc.close()
            raise ValueError("Incorrect PDF password.")
    return doc


def extract_text_from_pdf(path: str, password: str = "") -> tuple[str, int]:
    """Return (full_text, page_count)."""
    doc = open_pdf(path, password)
    pages = doc.page_count
    chunks = []
    for page in doc:
        chunks.append(page.get_text("text"))
    doc.close()
    # Fallback: pdfplumber for tables
    if sum(len(c) for c in chunks) < pages * TEXT_THRESHOLD:
        try:
            with pdfplumber.open(path) as pdf:
                if password:
                    pdf = pdfplumber.open(path, password=password)
                for i, pg in enumerate(pdf.pages):
                    t = pg.extract_text() or ""
                    if t and len(t) > len(chunks[i]):
                        chunks[i] = t
        except Exception:
            pass
    return "\n".join(chunks), pages


def render_pages_as_b64(path: str, password: str = "",
                         max_pages: int = MAX_PAGES_VISION,
                         dpi: int = RENDER_DPI) -> list[str]:
    """Render PDF pages to base64-encoded PNG strings."""
    doc = open_pdf(path, password)
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        images.append(base64.b64encode(pix.tobytes("png")).decode())
    doc.close()
    return images


def is_image_based(path: str, password: str = "") -> bool:
    text, pages = extract_text_from_pdf(path, password)
    if pages == 0:
        return True
    return (len(text) / pages) < TEXT_THRESHOLD


# ── PDF Content Extraction (AI) ───────────────────────────────────────────────
_EXTRACT_SYSTEM = """You are an expert electronics/hardware engineer analyzing board schematics.
Extract ALL signal/pin/interface information from this schematic.
Return ONLY a valid JSON object (no markdown fences) with this structure:
{
  "board_name": "string – inferred board/product name if visible",
  "soc": "string – SoC/processor name if visible",
  "interfaces": {
    "USB":   [ { "signal": "USB0_D+", "direction": "IO", "voltage": "3.3V", "pin": "A1", "note": "" }, ... ],
    "PCIe":  [ ... ],
    "DDR":   [ ... ],
    "I2C":   [ ... ],
    "SPI":   [ ... ],
    "UART":  [ ... ],
    "GPIO":  [ ... ],
    "HDMI":  [ ... ],
    "MIPI":  [ ... ],
    "Power": [ ... ],
    "Audio": [ ... ],
    "Ethernet": [ ... ],
    "Other": [ ... ]
  },
  "total_signals": 0
}
Include every signal you can find. Use "Other" for signals that don't fit standard interfaces.
If a field is unknown, use an empty string "".
"""

def extract_schematic_content(path: str, password: str = "",
                               label: str = "schematic") -> dict:
    """
    Use AI to extract all interface/signal data from a schematic PDF.
    Returns the parsed JSON dict from the AI, or a fallback structure.
    """
    client = get_ai_client()
    use_vision = is_image_based(path, password)
    print(f"[{label}] Mode: {'Vision' if use_vision else 'Text'}")

    try:
        if use_vision:
            pages_b64 = render_pages_as_b64(path, password)
            content_parts = [
                {"type": "text",
                 "text": f"This is a board schematic PDF ({label}). "
                         "Extract all signal/interface information as instructed."}
            ]
            for b64 in pages_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}",
                                  "detail": "high"}
                })
            messages = [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user",   "content": content_parts}
            ]
        else:
            text, _ = extract_text_from_pdf(path, password)
            # Split into chunks if very long
            chunk_size = 12000
            chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 80000), chunk_size)]
            combined_text = "\n...\n".join(chunks[:6])
            messages = [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user",   "content":
                 f"Board schematic text ({label}):\n\n{combined_text}"}
            ]

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        result = json.loads(raw)
        result["_mode"]  = "vision" if use_vision else "text"
        result["_label"] = label
        return result

    except json.JSONDecodeError as e:
        print(f"[{label}] JSON parse error: {e} — raw: {raw[:300]}")
        return {"board_name": "", "soc": "", "interfaces": {}, "total_signals": 0,
                "_mode": "error", "_label": label, "_error": str(e)}
    except Exception as e:
        print(f"[{label}] AI extraction error: {e}")
        return {"board_name": "", "soc": "", "interfaces": {}, "total_signals": 0,
                "_mode": "error", "_label": label, "_error": str(e)}


# ── Schematic Comparison ──────────────────────────────────────────────────────
_COMPARE_SYSTEM = """You are a senior hardware engineer comparing two board schematics based on the same SoC.
Given extracted signal/interface data from a REFERENCE board and a CUSTOMER board, produce a detailed comparison report.

Return ONLY a valid JSON object (no markdown fences) with this structure:
{
  "summary": "2-3 sentence executive summary of the key differences",
  "ref_board": "reference board name",
  "cust_board": "customer board name",
  "soc": "SoC name",
  "overall_match_pct": 85,
  "interfaces": [
    {
      "name": "USB",
      "ref_signal_count": 12,
      "cust_signal_count": 10,
      "match_pct": 83,
      "status": "DIFFERENCES_FOUND",
      "differences": [
        {
          "type": "MISSING_IN_CUSTOMER",
          "signal": "USB0_D+",
          "ref_detail": "IO, 3.3V, pin A1",
          "cust_detail": "",
          "notes": "USB port 0 full differential pair missing"
        },
        {
          "type": "EXTRA_IN_CUSTOMER",
          "signal": "USB3_ID",
          "ref_detail": "",
          "cust_detail": "Input, 1.8V",
          "notes": "Additional USB OTG ID pin added"
        },
        {
          "type": "CONFIG_DIFF",
          "signal": "USB_VBUS",
          "ref_detail": "3.3V power rail",
          "cust_detail": "5.0V power rail",
          "notes": "Voltage level difference — verify compatibility"
        }
      ],
      "matching_signals": ["USB1_D+", "USB1_D-", "USB_GND"]
    }
  ]
}

status options: "IDENTICAL", "MINOR_DIFFERENCES", "DIFFERENCES_FOUND", "MAJOR_DIFFERENCES", "MISSING_INTERFACE"
type options: "MISSING_IN_CUSTOMER", "EXTRA_IN_CUSTOMER", "CONFIG_DIFF"

Be thorough — analyse every interface present in either schematic.
"""

def compare_schematics(ref: dict, cust: dict) -> dict:
    """Call AI to compare two extracted schematic dicts. Returns comparison JSON."""
    client = get_ai_client()
    prompt = (
        f"REFERENCE BOARD extracted data:\n{json.dumps(ref, indent=2)}\n\n"
        f"CUSTOMER BOARD extracted data:\n{json.dumps(cust, indent=2)}\n\n"
        "Produce the detailed interface-by-interface comparison report as instructed."
    )
    messages = [
        {"role": "system", "content": _COMPARE_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=6000,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        result = json.loads(raw)
        result["_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result
    except json.JSONDecodeError as e:
        print(f"[compare] JSON parse error: {e}")
        return {"summary": "Comparison failed — JSON parse error.",
                "interfaces": [], "_error": str(e)}
    except Exception as e:
        print(f"[compare] AI error: {e}")
        return {"summary": f"Comparison failed — {e}",
                "interfaces": [], "_error": str(e)}


# ── HTML Report Generator ─────────────────────────────────────────────────────
_STATUS_COLOR = {
    "IDENTICAL":          "#22c55e",
    "MINOR_DIFFERENCES":  "#f59e0b",
    "DIFFERENCES_FOUND":  "#f97316",
    "MAJOR_DIFFERENCES":  "#ef4444",
    "MISSING_INTERFACE":  "#6b7280",
}
_TYPE_LABEL = {
    "MISSING_IN_CUSTOMER": ("Missing in Customer", "#fef2f2", "#dc2626"),
    "EXTRA_IN_CUSTOMER":   ("Extra in Customer",   "#f0fdf4", "#16a34a"),
    "CONFIG_DIFF":         ("Config Difference",   "#fffbeb", "#d97706"),
}

def build_html_report(comparison: dict, ref_name: str, cust_name: str) -> str:
    ts = comparison.get("_generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    summary = comparison.get("summary", "")
    soc = comparison.get("soc", "")
    overall = comparison.get("overall_match_pct", "N/A")
    interfaces = comparison.get("interfaces", [])

    # Build nav links
    nav_links = "".join(
        f'<a href="#{ifc["name"].replace(" ","_")}">{ifc["name"]}</a>'
        for ifc in interfaces
    )

    # Build interface sections
    ifc_sections = []
    for ifc in interfaces:
        name      = ifc.get("name", "Unknown")
        anchor    = name.replace(" ", "_")
        status    = ifc.get("status", "DIFFERENCES_FOUND")
        color     = _STATUS_COLOR.get(status, "#6b7280")
        match_pct = ifc.get("match_pct", 0)
        ref_cnt   = ifc.get("ref_signal_count", 0)
        cust_cnt  = ifc.get("cust_signal_count", 0)
        diffs     = ifc.get("differences", [])
        matching  = ifc.get("matching_signals", [])

        # Differences table rows
        diff_rows = ""
        for d in diffs:
            dtype = d.get("type", "")
            label, bg, fg = _TYPE_LABEL.get(dtype, ("Unknown", "#f8fafc", "#374151"))
            diff_rows += f"""
            <tr style="background:{bg}">
              <td><span class="badge" style="background:{fg};color:#fff">{label}</span></td>
              <td><code>{d.get("signal","")}</code></td>
              <td>{d.get("ref_detail","—")}</td>
              <td>{d.get("cust_detail","—")}</td>
              <td>{d.get("notes","")}</td>
            </tr>"""

        # Matching signals
        matching_pills = "".join(
            f'<span class="pill">{s}</span>' for s in matching[:30]
        )
        if len(matching) > 30:
            matching_pills += f'<span class="pill muted">+{len(matching)-30} more</span>'

        ifc_sections.append(f"""
      <section class="ifc-card" id="{anchor}">
        <div class="ifc-header">
          <div>
            <h2>{name}</h2>
            <div class="ifc-meta">
              Reference: <strong>{ref_cnt}</strong> signals &nbsp;|&nbsp;
              Customer: <strong>{cust_cnt}</strong> signals &nbsp;|&nbsp;
              Differences: <strong>{len(diffs)}</strong>
            </div>
          </div>
          <div class="ifc-status-block">
            <div class="match-ring" style="--pct:{match_pct};--clr:{color}">
              <span>{match_pct}%</span>
            </div>
            <div class="status-badge" style="background:{color}20;color:{color};border:1px solid {color}">
              {status.replace("_"," ")}
            </div>
          </div>
        </div>
        {"" if not diffs else f"""
        <h3 style="margin:1.2rem 0 .6rem">Differences</h3>
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th>Type</th><th>Signal</th>
              <th>Reference</th><th>Customer</th><th>Notes</th>
            </tr></thead>
            <tbody>{diff_rows}</tbody>
          </table>
        </div>"""}
        {"" if not matching else f"""
        <h3 style="margin:1.2rem 0 .6rem">Matching Signals ({len(matching)})</h3>
        <div class="pills">{matching_pills}</div>"""}
      </section>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Schematic Comparison Report</title>
<style>
  :root{{--bg:#f8fafc;--card:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}
  a{{color:#3b82f6;text-decoration:none}}
  a:hover{{text-decoration:underline}}
  code{{font-family:'Consolas','Courier New',monospace;font-size:.85em;background:#f1f5f9;padding:1px 5px;border-radius:4px}}
  .page-header{{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;padding:2.5rem 2rem 2rem}}
  .page-header h1{{font-size:1.8rem;font-weight:700;margin-bottom:.4rem}}
  .page-header .sub{{opacity:.8;font-size:.95rem}}
  .meta-row{{display:flex;gap:2rem;margin-top:1.2rem;flex-wrap:wrap}}
  .meta-item{{background:rgba(255,255,255,.12);border-radius:8px;padding:.5rem 1rem;font-size:.88rem}}
  .meta-item strong{{display:block;font-size:1rem}}
  .overall-badge{{background:#fff;color:#1e3a5f;padding:.4rem 1.2rem;border-radius:20px;font-weight:700;font-size:1.1rem;align-self:center}}
  .summary-card{{background:var(--card);border-left:4px solid #2563eb;margin:1.5rem 2rem;padding:1rem 1.4rem;border-radius:0 8px 8px 0;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
  .summary-card h2{{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.5rem}}
  nav{{background:var(--card);border-bottom:1px solid var(--border);padding:.6rem 2rem;position:sticky;top:0;z-index:10;display:flex;gap:.5rem;flex-wrap:wrap}}
  nav a{{font-size:.82rem;padding:.3rem .7rem;border-radius:4px;background:#f1f5f9;color:var(--text);border:1px solid var(--border)}}
  nav a:hover{{background:#dbeafe;border-color:#93c5fd;text-decoration:none}}
  .content{{padding:0 2rem 4rem}}
  .ifc-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;margin:1.5rem 0;padding:1.5rem;box-shadow:0 1px 4px rgba(0,0,0,.05)}}
  .ifc-header{{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}}
  .ifc-header h2{{font-size:1.25rem;color:var(--text)}}
  .ifc-meta{{font-size:.85rem;color:var(--muted);margin-top:.25rem}}
  .ifc-status-block{{display:flex;flex-direction:column;align-items:center;gap:.5rem}}
  .match-ring{{position:relative;width:70px;height:70px}}
  .match-ring::before{{content:'';position:absolute;inset:0;border-radius:50%;background:conic-gradient(var(--clr) calc(var(--pct)*1%),#e2e8f0 0);mask:radial-gradient(farthest-side,transparent calc(100% - 10px),#000 0)}}
  .match-ring span{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.95rem;color:var(--text)}}
  .status-badge{{font-size:.72rem;font-weight:600;padding:.2rem .7rem;border-radius:20px;text-align:center;white-space:nowrap}}
  h3{{font-size:.95rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.05em}}
  .table-wrap{{overflow-x:auto;border-radius:8px;border:1px solid var(--border)}}
  table{{width:100%;border-collapse:collapse;font-size:.88rem}}
  thead{{background:#f8fafc}}
  th{{padding:.6rem 1rem;text-align:left;font-weight:600;color:var(--muted);border-bottom:1px solid var(--border);font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}}
  td{{padding:.6rem 1rem;border-bottom:1px solid #f1f5f9;vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  .badge{{font-size:.72rem;font-weight:700;padding:.18rem .6rem;border-radius:4px;white-space:nowrap}}
  .pills{{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.3rem}}
  .pill{{font-size:.78rem;background:#f1f5f9;border:1px solid var(--border);border-radius:4px;padding:.15rem .55rem;font-family:'Consolas',monospace}}
  .pill.muted{{color:var(--muted);background:#fff}}
  .footer{{text-align:center;padding:2rem;color:var(--muted);font-size:.82rem;border-top:1px solid var(--border)}}
  @media print{{nav{{display:none}}.page-header{{break-inside:avoid}}}}
</style>
</head>
<body>
<header class="page-header">
  <h1>&#9889; Schematic Comparison Report</h1>
  <div class="sub">Interface-by-interface analysis &mdash; {ts}</div>
  <div class="meta-row">
    <div class="meta-item"><strong>{comparison.get("ref_board","Reference Board")}</strong>Reference Board</div>
    <div class="meta-item"><strong>{comparison.get("cust_board","Customer Board")}</strong>Customer Board</div>
    {"" if not soc else f'<div class="meta-item"><strong>{soc}</strong>SoC Platform</div>'}
    <div class="meta-item"><strong>{len(interfaces)}</strong>Interfaces</div>
    <div class="overall-badge">&#x2713; {overall}% Overall Match</div>
  </div>
</header>

<div class="summary-card">
  <h2>Executive Summary</h2>
  <p>{summary}</p>
</div>

<nav>{nav_links}</nav>
<div class="content">
  {"".join(ifc_sections)}
</div>
<div class="footer">
  Generated by Schematic Comparator &nbsp;&bull;&nbsp; {ts}
</div>
</body>
</html>"""


# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", os.urandom(24).hex())
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024   # 128 MB


@app.route("/")
def index():
    return render_template("index.html", use_ai=USE_AI)


@app.route("/check-password", methods=["POST"])
def check_password():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    tmp = os.path.join(UPLOAD_DIR, f"tmp_{uuid.uuid4().hex}.pdf")
    f.save(tmp)
    try:
        result = is_pdf_password_protected(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return jsonify(result)


@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload one PDF (reference or customer).
    Form fields: file, type (ref|cust), password (optional)
    """
    f        = request.files.get("file")
    pdf_type = request.form.get("type", "ref")    # "ref" or "cust"
    password = request.form.get("password", "")

    if not f:
        return jsonify({"success": False, "error": "No file provided"}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "Only PDF files are accepted"}), 400

    sdir = get_session_dir()
    dest = os.path.join(sdir, f"{pdf_type}.pdf")
    f.save(dest)

    # Verify password if encrypted
    chk = is_pdf_password_protected(dest)
    if chk.get("needs_password") and not password:
        return jsonify({"success": False, "needs_password": True,
                        "filename": f.filename})
    if password:
        try:
            open_pdf(dest, password).close()
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400

    # Store password (for later extraction)
    pwd_path = os.path.join(sdir, f"{pdf_type}_pwd.txt")
    with open(pwd_path, "w") as fp:
        fp.write(password)

    # Save metadata
    meta_path = os.path.join(sdir, f"{pdf_type}_meta.json")
    save_json(meta_path, {
        "filename": f.filename,
        "type": pdf_type,
        "uploaded_at": datetime.now().isoformat(),
    })

    # Invalidate any cached extraction / comparison
    for stale in [f"{pdf_type}_content.json", "comparison.json", "report.html"]:
        sp = os.path.join(sdir, stale)
        if os.path.exists(sp):
            os.remove(sp)

    return jsonify({
        "success": True,
        "filename": f.filename,
        "type": pdf_type,
        "image_based": is_image_based(dest, password),
    })


@app.route("/extract", methods=["POST"])
def extract():
    """
    Run AI extraction on one uploaded PDF.
    Body JSON: { "type": "ref"|"cust" }
    """
    if not USE_AI:
        return jsonify({"success": False, "error": "No AI key configured"}), 503

    data     = request.get_json(force=True, silent=True) or {}
    pdf_type = data.get("type", "ref")

    sdir = get_session_dir()
    pdf_path = os.path.join(sdir, f"{pdf_type}.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({"success": False, "error": f"No {pdf_type} PDF uploaded yet"}), 400

    pwd_path = os.path.join(sdir, f"{pdf_type}_pwd.txt")
    password = open(pwd_path).read().strip() if os.path.exists(pwd_path) else ""

    meta_path = os.path.join(sdir, f"{pdf_type}_meta.json")
    meta      = load_json(meta_path) or {}
    label     = f"{pdf_type.upper()} ({meta.get('filename','')})"

    content = extract_schematic_content(pdf_path, password, label)
    save_json(os.path.join(sdir, f"{pdf_type}_content.json"), content)

    total = content.get("total_signals", 0)
    if not total:
        total = sum(len(v) for v in content.get("interfaces", {}).values())

    return jsonify({
        "success":      True,
        "board_name":   content.get("board_name", ""),
        "soc":          content.get("soc", ""),
        "interfaces":   list(content.get("interfaces", {}).keys()),
        "total_signals": total,
        "mode":         content.get("_mode", ""),
        "error":        content.get("_error", ""),
    })


@app.route("/compare", methods=["POST"])
def compare():
    """
    Run comparison between already-extracted ref and cust content.
    Caches result in session dir.
    """
    if not USE_AI:
        return jsonify({"success": False, "error": "No AI key configured"}), 503

    sdir = get_session_dir()
    ref_content  = load_json(os.path.join(sdir, "ref_content.json"))
    cust_content = load_json(os.path.join(sdir, "cust_content.json"))

    if not ref_content:
        return jsonify({"success": False, "error": "Reference schematic not extracted yet"}), 400
    if not cust_content:
        return jsonify({"success": False, "error": "Customer schematic not extracted yet"}), 400

    if ref_content.get("_error") or cust_content.get("_error"):
        return jsonify({"success": False,
                        "error": "One or both extractions failed. Re-extract before comparing."}), 400

    comparison = compare_schematics(ref_content, cust_content)
    save_json(os.path.join(sdir, "comparison.json"), comparison)

    # Build and cache HTML report
    ref_meta  = load_json(os.path.join(sdir, "ref_meta.json"))  or {}
    cust_meta = load_json(os.path.join(sdir, "cust_meta.json")) or {}
    html = build_html_report(
        comparison,
        ref_meta.get("filename", "Reference"),
        cust_meta.get("filename", "Customer"),
    )
    with open(os.path.join(sdir, "report.html"), "w", encoding="utf-8") as fp:
        fp.write(html)

    return jsonify({
        "success":      True,
        "summary":      comparison.get("summary", ""),
        "overall_match_pct": comparison.get("overall_match_pct", 0),
        "interface_count": len(comparison.get("interfaces", [])),
        "interfaces":   [
            {
                "name":    i.get("name"),
                "status":  i.get("status"),
                "match_pct": i.get("match_pct"),
                "diff_count": len(i.get("differences", [])),
            }
            for i in comparison.get("interfaces", [])
        ],
        "error": comparison.get("_error", ""),
    })


@app.route("/download-report")
def download_report():
    sdir = get_session_dir()
    path = os.path.join(sdir, "report.html")
    if not os.path.exists(path):
        return "No report generated yet.", 404
    ref_meta  = load_json(os.path.join(sdir, "ref_meta.json")) or {}
    fname = f"schematic_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return send_file(path, as_attachment=True,
                     download_name=fname, mimetype="text/html")


def _build_chat_context(sdir: str) -> str:
    """
    Build a rich, structured system-prompt context from all available
    session data: extracted ref/cust signal lists + full comparison detail.
    Avoids hard mid-JSON truncation by serialising each interface separately
    and stopping cleanly when we approach the token budget.
    """
    ref_content  = load_json(os.path.join(sdir, "ref_content.json"))
    cust_content = load_json(os.path.join(sdir, "cust_content.json"))
    comparison   = load_json(os.path.join(sdir, "comparison.json"))

    MAX_CHARS = 18_000   # safe limit well within GPT-4o's context window
    parts: list[str] = []

    # ── 1. Board identity ──────────────────────────────────────────────────
    if ref_content or cust_content:
        ref_name  = (ref_content  or {}).get("board_name", "Reference Board")
        cust_name = (cust_content or {}).get("board_name", "Customer Board")
        soc       = (ref_content  or cust_content or {}).get("soc", "")
        identity  = f"BOARDS UNDER ANALYSIS:\n  Reference : {ref_name}\n  Customer  : {cust_name}"
        if soc:
            identity += f"\n  SoC       : {soc}"
        parts.append(identity)

    # ── 2. Full comparison detail (most useful for the chatbot) ───────────
    if comparison:
        cmp_lines = [
            f"COMPARISON SUMMARY: {comparison.get('summary', '')}",
            f"Overall match: {comparison.get('overall_match_pct', 'N/A')}%",
            "",
            "INTERFACE-BY-INTERFACE DIFFERENCES:",
        ]
        for ifc in comparison.get("interfaces", []):
            ifc_hdr = (
                f"\n  [{ifc.get('name','?')}]  "
                f"Match={ifc.get('match_pct','?')}%  "
                f"Status={ifc.get('status','?')}  "
                f"Ref signals={ifc.get('ref_signal_count','?')}  "
                f"Cust signals={ifc.get('cust_signal_count','?')}"
            )
            cmp_lines.append(ifc_hdr)
            for d in ifc.get("differences", []):
                tag = d.get("type","")
                sig = d.get("signal","")
                ref_d  = d.get("ref_detail","")
                cust_d = d.get("cust_detail","")
                note   = d.get("notes","")
                cmp_lines.append(
                    f"    • [{tag}] {sig}"
                    + (f"  REF={ref_d}"  if ref_d  else "")
                    + (f"  CUST={cust_d}" if cust_d else "")
                    + (f"  NOTE: {note}"  if note   else "")
                )
            matching = ifc.get("matching_signals", [])
            if matching:
                preview = ", ".join(matching[:15])
                cmp_lines.append(
                    f"    Matching ({len(matching)}): {preview}"
                    + (" …" if len(matching) > 15 else "")
                )
        parts.append("\n".join(cmp_lines))

    # ── 3. Raw signal lists per board (add as many interfaces as budget allows)
    for label, content in [("REFERENCE BOARD SIGNALS", ref_content),
                            ("CUSTOMER BOARD SIGNALS",  cust_content)]:
        if not content:
            continue
        remaining = MAX_CHARS - sum(len(p) for p in parts)
        if remaining < 500:
            break
        board_lines = [f"{label} ({content.get('board_name', '')}):"]
        for ifc_name, signals in content.get("interfaces", {}).items():
            chunk = f"\n  {ifc_name} ({len(signals)} signals):\n"
            for s in signals:
                sig_str = (
                    f"    {s.get('signal','?')}"
                    + (f"  dir={s.get('direction','')}"  if s.get('direction') else "")
                    + (f"  V={s.get('voltage','')}"       if s.get('voltage')   else "")
                    + (f"  pin={s.get('pin','')}"          if s.get('pin')       else "")
                    + (f"  | {s.get('note','')}"           if s.get('note')      else "")
                )
                chunk += sig_str + "\n"
            if sum(len(p) for p in board_lines) + len(chunk) > remaining:
                board_lines.append(f"\n  … (remaining interfaces omitted — budget reached)")
                break
            board_lines.append(chunk)
        parts.append("".join(board_lines))

    if not parts:
        return (
            "No schematic data has been loaded yet. "
            "Tell the user to upload and extract schematics first."
        )
    return "\n\n".join(parts)


@app.route("/chat", methods=["POST"])
def chat():
    """Chatbot: answer questions using context from both schematics."""
    if not USE_AI:
        return jsonify({"success": False, "error": "No AI key configured"}), 503

    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"success": False, "error": "Empty message"}), 400

    sdir = get_session_dir()
    sid  = session.get("sid", "")

    # Build rich, complete context from all available session data
    schematic_context = _build_chat_context(sdir)

    system_prompt = f"""You are an expert hardware/electronics engineering assistant specialising in board schematics and SoC bring-up.

STRICT RULES — follow these exactly:
1. Answer ONLY based on the schematic data provided below. Do NOT invent, assume, or extrapolate signal names, voltages, pin numbers, or interface details that are not explicitly in the data.
2. If the data does not contain enough information to answer a question, say "That information is not available in the extracted schematic data." Do not guess.
3. When listing signals or differences, quote the exact signal names from the data.
4. If asked about a specific interface (e.g. USB, PCIe), look up that exact interface in the data before answering.
5. For comparison questions, use the INTERFACE-BY-INTERFACE DIFFERENCES section — it contains the authoritative diff.
6. Be concise but complete. Use bullet points for lists.

SCHEMATIC DATA:
{schematic_context}"""

    # Retrieve / update chat history (conversation turns only, no system prompt in history)
    history = _chat_histories.get(sid, [])
    history.append({"role": "user", "content": user_msg})

    # System prompt always first, then last 16 conversation turns
    messages = [{"role": "system", "content": system_prompt}] + history[-16:]

    try:
        client = get_ai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0,      # zero temperature = deterministic, grounded answers
        )
        assistant_msg = resp.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": assistant_msg})
        _chat_histories[sid] = history[-40:]   # keep last 40 turns

        return jsonify({
            "success": True,
            "message": assistant_msg,
            "history_length": len(history),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/clear-session", methods=["POST"])
def clear_session():
    sdir = get_session_dir()
    sid  = session.get("sid")
    if sid and os.path.exists(sdir):
        shutil.rmtree(sdir, ignore_errors=True)
    if sid and sid in _chat_histories:
        del _chat_histories[sid]
    session.clear()
    return jsonify({"success": True})


@app.route("/session-status")
def session_status():
    """Return what has been uploaded/extracted so far."""
    sdir = get_session_dir()
    def _meta(t):
        m = load_json(os.path.join(sdir, f"{t}_meta.json"))
        c = load_json(os.path.join(sdir, f"{t}_content.json"))
        return {
            "uploaded":  os.path.exists(os.path.join(sdir, f"{t}.pdf")),
            "filename":  (m or {}).get("filename", ""),
            "extracted": c is not None,
            "board_name": (c or {}).get("board_name", ""),
            "error":     (c or {}).get("_error", ""),
        }
    cmp = load_json(os.path.join(sdir, "comparison.json"))
    return jsonify({
        "ref":  _meta("ref"),
        "cust": _meta("cust"),
        "compared": cmp is not None,
        "overall_match_pct": (cmp or {}).get("overall_match_pct"),
    })


if __name__ == "__main__":
    print("=" * 65)
    print("  Schematic Comparator")
    print("=" * 65)
    print(f"  AI Mode  : {'ENABLED — Intel ExpertGPT (GPT-4o)' if USE_AI else 'DISABLED — no API key'}")
    print(f"  Endpoint : {EXPERGPT_BASE}")
    print(f"  URL      : http://127.0.0.1:5000")
    print("=" * 65)
    app.run(debug=True, host="127.0.0.1", port=5000)
