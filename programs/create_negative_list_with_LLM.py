#!/usr/bin/env python3
# segment_screen.py
import os, sys, csv, json, time, argparse
import concurrent.futures
from typing import List, Dict
import pandas as pd

# ---- OpenAI client ----
try:
    from openai import OpenAI
except ImportError:
    print("Please `pip install openai` first.")
    sys.exit(1)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # fast/cheap; swap if you prefer
BATCH_SIZE = 80  # keep conservative to avoid super long JSON responses
SLEEP_BETWEEN = 0.7  # seconds between calls, tweak if you hit rate limits
MAX_WORKERS_DEFAULT = int(os.getenv("MAX_WORKERS", "8"))

SYSTEM_PROMPT = """You are a forensic reviewer of corporate segment disclosures in oil & gas.
Your task: decide whether each segment NAME is clearly NOT upstream E&P (exploration, development, and production of oil & gas).
Be STRICT and CONSERVATIVE:

Label `is_non_e_and_p = true` ONLY when the name clearly denotes a non-E&P activity such as:
• Downstream (refining, retail marketing, fuel stations)
• Midstream & logistics (pipelines, gathering, processing, fractionation, compression, storage, terminals)
• Utilities & power (electricity generation, gas distribution, retail power)
• Trading/marketing (energy trading, optimization)
• Oilfield services / drilling contractors (supplier side)
• Chemicals & lubricants, petrochemicals
• LNG (liquefaction, regas terminals), gas & power portfolios
• Mining OTHER than oil & gas (coal, metals, etc.), real estate, agriculture
• Corporate & other, eliminations, headquarters (non-operating)
• Renewables (wind, solar, hydro) and CCS, when not described as upstream E&P

If the name contains E&P cues (e.g., exploration, development, production, upstream, onshore/offshore, field/basin names) and NOTHING else, treat it as E&P → `is_non_e_and_p = false`.

If a name mixes E&P with non-E&P terms (e.g., "Exploration & Marketing", "Upstream and Midstream"), treat as AMBIGUOUS; default to `is_non_e_and_p = false`.

If the name is vague (e.g., "Operations", geographic labels, "North America", "International"), default to `false`.

OUTPUT STRICT JSON with this shape:
{
  "decisions": [
    {
      "segment": "<original>",
      "is_non_e_and_p": true|false,
      "category": "<one of: Downstream|Midstream|Utilities/Power|Trading/Marketing|Oilfield Services|Chemicals|LNG/Gas&Power|Mining/Other|Real Estate|Corporate/Other|Renewables|Unknown>",
      "rationale": "<one sentence explaining why you flagged or not>",
      "confidence": <0.0-1.0>
    }, ...
  ]
}
Ensure decisions are in the same order as input.
"""

USER_TEMPLATE = """Classify these segment names. Remember: ONLY mark true when clearly not E&P; ambiguous defaults to false.

Segments:
{payload}
"""

# ---- Pretty printing & colors ----
USE_COLOR = False

class C:
    RESET = ""
    GREEN = ""
    RED = ""
    YELLOW = ""
    CYAN = ""
    MAGENTA = ""
    BLUE = ""
    GRAY = ""

def init_colors(disable: bool) -> None:
    global USE_COLOR
    if disable:
        USE_COLOR = False
        return
    try:
        from colorama import Fore, Style, init as colorama_init  # type: ignore
        colorama_init()
        C.GREEN = Fore.GREEN
        C.RED = Fore.RED
        C.YELLOW = Fore.YELLOW
        C.CYAN = Fore.CYAN
        C.MAGENTA = Fore.MAGENTA
        C.BLUE = Fore.BLUE
        C.GRAY = Fore.LIGHTBLACK_EX
        C.RESET = Style.RESET_ALL
        USE_COLOR = True
        return
    except Exception:
        pass
    # Fallback to ANSI on non-Windows TTYs
    if sys.platform != "win32" and sys.stdout.isatty():
        C.GREEN = "\x1b[32m"
        C.RED = "\x1b[31m"
        C.YELLOW = "\x1b[33m"
        C.CYAN = "\x1b[36m"
        C.MAGENTA = "\x1b[35m"
        C.BLUE = "\x1b[34m"
        C.GRAY = "\x1b[90m"
        C.RESET = "\x1b[0m"
        USE_COLOR = True

def fmt(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}" if USE_COLOR and color else text

def load_env_from_dotenv() -> None:
    """
    Load environment variables from a .env file if present.
    Tries python-dotenv first, then falls back to a minimal parser.
    Does not override existing environment variables.
    """
    loaded = False
    try:
        # Prefer python-dotenv if available
        from dotenv import load_dotenv, find_dotenv  # type: ignore
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path, override=False)
            loaded = True
        # Also try script dir and its parent to cover running from subdirs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for p in (script_dir, os.path.dirname(script_dir)):
            env_path = os.path.join(p, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path, override=False)
                loaded = True
    except Exception:
        pass

    if loaded:
        return

    # Fallback: simple .env parser
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [os.getcwd(), script_dir, os.path.dirname(script_dir)]
        for base in candidates:
            env_path = os.path.join(base, ".env")
            if not os.path.exists(env_path):
                continue
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    key, val = s.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            break
    except Exception:
        # If fallback fails, proceed without raising; main() will report missing key
        pass

def read_segments(path: str) -> List[str]:
    """
    Accepts a .txt (one segment per line) or .csv with a 'segment_name' column.
    Returns a de-duplicated, order-preserving list.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    segs = []
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    segs.append(s)
    else:
        df = pd.read_csv(path)
        col = None
        for c in df.columns:
            if c.lower() in {"segment", "segment_name", "name"}:
                col = c
                break
        if col is None:
            raise ValueError("CSV must have a 'segment_name' (or 'segment'/'name') column.")
        segs = df[col].astype(str).fillna("").tolist()
    # de-duplicate preserving order
    seen, uniq = set(), []
    for s in segs:
        key = s.strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq

def chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def classify_batch(client: OpenAI, model: str, segments: List[str]) -> List[Dict]:
    """
    Calls Chat Completions with JSON mode to get conservative flags.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(payload=json.dumps(segments, ensure_ascii=False))}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    decisions = data.get("decisions", [])
    # light validation
    out = []
    for i, seg in enumerate(segments):
        item = decisions[i] if i < len(decisions) else {}
        out.append({
            "segment": seg,
            "is_non_e_and_p": bool(item.get("is_non_e_and_p", False)),
            "category": item.get("category", "Unknown"),
            "rationale": item.get("rationale", ""),
            "confidence": float(item.get("confidence", 0.0))
        })
    return out

def main():
    ap = argparse.ArgumentParser(description="Conservatively flag clearly non-E&P segments via OpenAI.")
    ap.add_argument("-i", "--input_path", default="output/segment_names.txt", help="Path to segment_names.txt or CSV with column 'segment_name'.")
    ap.add_argument("-o", "--output_csv", default="output/segment_flags.csv", help="Output CSV path.")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help="OpenAI model (default: %(default)s).")
    ap.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE, help="Segments per API call.")
    ap.add_argument("-w", "--max_workers", type=int, default=MAX_WORKERS_DEFAULT, help="Number of concurrent API calls (default: %(default)s).")
    ap.add_argument("--no_live", action="store_true", help="Disable live per-segment printing.")
    ap.add_argument("--no_color", action="store_true", help="Disable colored output.")
    args = ap.parse_args()

    init_colors(disable=args.no_color)

    load_env_from_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in your environment.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    segments = read_segments(args.input_path)
    if not segments:
        print("No segments found.")
        return

    all_batches = list(enumerate(chunk(segments, args.batch_size)))
    total = len(segments)
    print(fmt(f"Classifying {total} segments in {len(all_batches)} batches of {args.batch_size} with up to {args.max_workers} workers...", C.CYAN))

    def process_batch(index: int, seg_batch: List[str]):
        try:
            return index, classify_batch(client, args.model, seg_batch)
        except Exception as e:
            sys.stderr.write(f"\nBatch {index+1} failed: {e}\nRetrying after a short pause...\n")
            time.sleep(max(2.0, SLEEP_BETWEEN))
            # second attempt
            return index, classify_batch(client, args.model, seg_batch)

    completed = 0
    results_by_index = {}
    if all_batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(process_batch, idx, batch) for idx, batch in all_batches]
            for fut in concurrent.futures.as_completed(futures):
                idx, res = fut.result()
                results_by_index[idx] = res
                # Live print and progress update
                if args.no_live:
                    completed += sum(1 for _ in res)
                    print(f"Progress: {completed}/{total}\r", end="")
                else:
                    for r in res:
                        completed += 1
                        status = fmt("NON-E&P", C.GREEN) if r["is_non_e_and_p"] else fmt("E&P/Unknown", C.YELLOW)
                        name = r["segment"]
                        cat = r.get("category", "Unknown")
                        conf = r.get("confidence", 0.0)
                        print(f"[{completed:>5}/{total}] {status}  {fmt(cat, C.BLUE)}  {name}  {fmt(f'({conf:.2f})', C.GRAY)}")

    rows = []
    for i in range(len(all_batches)):
        rows.extend(results_by_index.get(i, []))

    # If live was disabled, finish the progress line
    if args.no_live:
        print("")

    # write CSV
    fieldnames = ["segment", "is_non_e_and_p", "category", "confidence", "rationale"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # quick summary
    df = pd.DataFrame(rows)
    flagged = df[df["is_non_e_and_p"] == True]
    print(f"\nDone. Flagged {len(flagged)}/{len(df)} ({len(flagged)/max(1,len(df)):.1%}) as clearly non-E&P.")
    print(f"Output → {args.output_csv}")

if __name__ == "__main__":
    main()
