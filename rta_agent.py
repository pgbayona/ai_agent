#In VS Code: Ctrl+Shift+P → Python: Select Interpreter → choose
#C:\Users\Pamela Bayona\Documents\Python\TPRD rag\.venv\Scripts\python.exe

#Fresh install 
# open cmd: cd "C:\Users\Pamela Bayona\Documents\Python\ai_agent" 
# #py -3 -m venv .venv 
# #.\.venv\Scripts\activate #python -m pip install -r requirements.txt 
#type the below in cmd to install the packages; NOT IN PYTHON! 
#python -m pip install -U langgraph langchain langchain-openai openai #pip install openai
#pip install openai

#Generate requirements file
#pip freeze > requirements.txt

#run on stream
#.\.venv\Scripts\activate
# streamlit run rta_agent.py

#push to git
#git init
#git add -A
#git commit -m "Initial commit"
#git branch -M main
#git remote add origin https://github.com/pgbayona/ai_agent.git
#git push -u origin main

# wto_app.py
import os, re, io, unicodedata
from typing import Optional, Dict, List, Tuple

import streamlit as st
import pandas as pd

# ===== Azure OpenAI (optional – only to polish the on-screen report) =====
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None  # app still runs without LLM polish

AZURE_ENDPOINT = "https://oaishrp01.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4o-mini-bayona-1dx6p"  # pass as model=
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_API_KEY = os.getenv("OPENAI_API_KEY")  # set this in your env

def llm_polish(markdown: str) -> str:
    if not (AzureOpenAI and AZURE_API_KEY):
        return markdown
    try:
        client = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{
                "role": "user",
                "content": (
                    "Polish the following Markdown report for clarity and concision. "
                    "Keep headings, tables, and bullet lists intact. Do not invent facts.\n\n" + markdown
                ),
            }],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return markdown

# ===== Helpers =====
def normalize_header(s: str) -> str:
    s = str(s or "").strip().replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", s)

def fold_ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    )

def fix_numeric_code_year_headers(df: pd.DataFrame, start_year: Optional[int]):
    """
    Map sequences like 4045..4050 -> 2023..2028 using the inferred start_year.
    Returns: (df, renames_list)
    """
    renames = []
    if start_year is None:
        return df, renames

    # find 4-digit numeric headers that are NOT 20xx (e.g., 4045)
    bad_nums = []
    for c in df.columns:
        s = str(c).strip()
        if re.fullmatch(r"\d{4}", s):
            n = int(s)
            if not (2000 <= n <= 2099):
                bad_nums.append(n)

    if len(bad_nums) < 2:
        return df, renames  # need at least a small run to be confident

    bad_nums = sorted(set(bad_nums))
    # require they be (almost) consecutive
    if bad_nums[-1] - bad_nums[0] > (len(bad_nums) - 1):
        return df, renames

    base = bad_nums[0]
    # rename each matching header: new_year = start_year + (n - base)
    new_cols = {}
    for c in list(df.columns):
        s = str(c).strip()
        if re.fullmatch(r"\d{4}", s):
            n = int(s)
            if n in bad_nums:
                new = str(start_year + (n - base))
                if s != new:
                    new_cols[c] = new

    # apply (drop pre-existing dup targets; coalesce will still run later)
    for old, new in new_cols.items():
        if new in df.columns:
            df.drop(columns=[new], inplace=True)
        df.rename(columns={old: new}, inplace=True)
        renames.append((str(old), new, f"numeric code sequence → year (base={base}, start_year={start_year})"))

    return df, renames

# Fuzzy (optional)
try:
    from rapidfuzz import fuzz, process
    def best_match(q: str, choices: List[str]) -> Tuple[str, int]:
        m = process.extractOne(q, choices, scorer=fuzz.token_set_ratio)
        return (m[0], int(m[1])) if m else ("", 0)
except Exception:
    import difflib
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
    def best_match(q: str, choices: List[str]) -> Tuple[str, int]:
        qn = _norm(q)
        cn = [_norm(c) for c in choices]
        m = difflib.get_close_matches(qn, cn, n=1, cutoff=0.0)
        if not m:
            return ("", 0)
        idx = cn.index(m[0])
        score = int(difflib.SequenceMatcher(None, qn, cn[idx]).ratio() * 100)
        return (choices[idx], score)

# Canonical
SYNONYMS = {
    "TL": [
        "TL", "tariff line", "tariff-line", "product code", "tariff code",
        "hs", "hs code", "hs6", "hs 6-digit", "subheading", "hscode",
        "national line", "tariffline", "code"
    ],
    "Description": [
        "description", "goods description", "product description", "desc",
        "tl description", "article description"
    ],
    "MFN": [
        "mfn", "mfn rate", "applied mfn", "base rate", "bound mfn",
        "mfn tariff", "mfn (%)", "mfn percent", "mfn ad valorem",
        "ad valorem mfn", "mfn adval"
    ],
    "Quota": [
        "quota", "trq", "tariff rate quota", "quota quantity", "quota (trq)", "trq quantity"
    ],
}

# Explicit vs relative year detection (globals)
YEAR_ABS_PAT = re.compile(r"\b(20\d{2})\b")  # explicit 4-digit year anywhere

YEAR_REL_PATTERNS = [
    re.compile(r"\by(?:ear)?\s*0*([1-9]\d?)\b"),  # y1, year 1
    re.compile(r"\bphase\s*0*([1-9]\d?)\b"),      # phase 1
    re.compile(r"\bp\s*0*([1-9]\d?)\b"),          # p1
]

META_YEAR_HINT_KEYS = [
    "start year", "base year", "entry into force", "eif", "implementation year",
    "start", "baseline year"
]

NOTE_LIKE_PAT = re.compile(
    r"(remark|notes?|comment|footnote|uae.?s.*schedule.*to.*turkiye)", re.I
)

def detect_header_row(raw: pd.DataFrame) -> int:
    best_idx, best_score = 0, -1
    max_scan = min(15, len(raw))
    for i in range(max_scan):
        row = raw.iloc[i].tolist()
        texts = [str(x) for x in row if pd.notna(x)]
        n_text = sum(bool(re.search(r"[A-Za-z]", t)) for t in texts)
        n_unique = len(set(t.lower() for t in texts))
        token_bonus = sum(
            1 for t in texts for key in ["code", "desc", "mfn", "quota", "trq", "note", "remarks", "year", "202", "201"]
            if key in str(t).lower()
        )
        score = n_text + 0.2 * n_unique + 0.8 * token_bonus
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

def find_start_year(raw: pd.DataFrame) -> Optional[int]:
    scan_rows = min(12, len(raw))
    scan_cols = min(10, raw.shape[1])
    for i in range(scan_rows):
        for j in range(scan_cols):
            cell = str(raw.iat[i, j]).strip().lower()
            if not cell or cell in ("nan", "none"):
                continue
            if any(k in cell for k in META_YEAR_HINT_KEYS):
                m = re.search(r"(20\d{2})", cell)
                if m:
                    return int(m.group(1))
    for i in range(scan_rows):
        row_vals = [str(x).lower() for x in raw.iloc[i, :scan_cols].tolist()]
        for j, v in enumerate(row_vals):
            if any(k in v for k in META_YEAR_HINT_KEYS):
                for k in range(j + 1, min(j + 3, scan_cols)):
                    m = re.search(r"(20\d{2})", row_vals[k])
                    if m:
                        return int(m.group(1))
    return None

def build_header_mapping(cols: List[str], start_year: Optional[int]) -> Dict[str, Tuple[str, str]]:
    """
    Return mapping: orig -> (new_name, reason)
    Handles TL/Description/MFN/Quota + year mapping.
    Notes & imports are handled elsewhere.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    for c in cols:
        orig = c
        c_norm = normalize_header(c)
        low = c_norm.lower()

        # 1) Explicit calendar year anywhere (e.g., "2024 - 2nd installment")
        m_abs = YEAR_ABS_PAT.search(low)
        if m_abs:
            y = m_abs.group(1)
            mapping[orig] = (y, "explicit year label")
            continue

        # 2) Relative year patterns (Year 1, Phase 2, P3, ...)
        rel = None
        for pat in YEAR_REL_PATTERNS:
            m = pat.search(low)
            if m:
                rel = int(m.group(1))
                break
        if rel:
            new = str(start_year + (rel - 1)) if start_year else f"Year{rel}"
            mapping[orig] = (new, f"relative year → {new}")
            continue

        # 3) Fuzzy canonical (excluding Notes here)
        best_target, best_score = None, -1
        for target, syns in SYNONYMS.items():
            bm, sc = best_match(low, [s.lower() for s in syns])
            if sc > best_score:
                best_target, best_score = target, sc
        if best_target and best_score >= 80:
            mapping[orig] = (best_target, f"synonym→{best_target}")
        else:
            mapping[orig] = (c_norm, "unchanged")

    return mapping

def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    result = pd.DataFrame(index=df.index)
    for col in pd.unique(df.columns):
        block = df.loc[:, df.columns == col]
        if block.shape[1] == 1:
            result[col] = block.iloc[:, 0]
        else:
            result[col] = block.bfill(axis=1).iloc[:, 0]
    return result

def as_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    return obj if isinstance(obj, pd.Series) else obj.iloc[:, 0]

def assign_notes_anywhere(df: pd.DataFrame, rename_log: List[Tuple[str,str,str]]) -> pd.DataFrame:
    """
    Find all note-like columns (Remarks/Notes/Comments/UAE schedule …) in their current order
    and rename to: Notes, Notes 1, Notes 2, Notes 3, ...
    """
    cols = list(df.columns)
    cand_idxs = []
    for i, c in enumerate(cols):
        low = fold_ascii(str(c).lower())
        if NOTE_LIKE_PAT.search(low):
            cand_idxs.append(i)
    names_to_use = []
    for k, i in enumerate(cand_idxs):
        names_to_use.append("Notes" if k == 0 else f"Notes {k}")
    for i, newname in zip(cand_idxs, names_to_use):
        old = df.columns[i]
        if old != newname:
            if newname in df.columns:
                df.drop(columns=[newname], inplace=True)
            df.rename(columns={old: newname}, inplace=True)
            rename_log.append((str(old), newname, "note-like column"))
    return df

def tag_imports_anywhere(df: pd.DataFrame, rename_log: List[Tuple[str,str,str]]) -> pd.DataFrame:
    """
    Rename any exact 2020/2021/2022 (or with a trailing *) to imp_YYYY, regardless of position.
    """
    pat = re.compile(r"^(20(20|21|22))(\*?)$")
    cols = list(df.columns)
    for c in cols:
        m = pat.fullmatch(str(c).strip())
        if m:
            year = m.group(1)
            new = f"imp_{year}"
            if c != new:
                if new in df.columns:
                    df.drop(columns=[new], inplace=True)
                df.rename(columns={c: new}, inplace=True)
                rename_log.append((str(c), new, "import value year"))
            # numeric
            df[new] = _to_numeric_series(as_series(df, new))
    return df

def ensure_mfn(df: pd.DataFrame, rename_log: List[Tuple[str,str,str]]) -> pd.DataFrame:
    """
    If 'MFN' not present, map the exact bare '2023' column to MFN (not '2023 ...installment').
    """
    if "MFN" in df.columns:
        return df
    # exact bare 2023 columns
    bare_2023 = [c for c in df.columns if str(c).strip() == "2023"]
    if bare_2023:
        c = bare_2023[0]
        df.rename(columns={c: "MFN"}, inplace=True)
        rename_log.append((str(c), "MFN", "MFN Applied Rate (exact 2023)"))
        return df
    # otherwise, if any column already contains 'mfn' (unlikely after prior mapping), leave it
    return df

def harmonize_excel(file_like, sheet_name: Optional[str] = None):
    # 1) Read raw (no header) to detect row
    raw = pd.read_excel(file_like, header=None, sheet_name=sheet_name, dtype=object, engine="openpyxl")
    if isinstance(raw, dict):
        first_name = sheet_name or list(raw.keys())[0]
        raw = raw[first_name]

    header_row = detect_header_row(raw)
    if hasattr(file_like, "seek"):
        file_like.seek(0)

    # 2) Read with detected header
    df = pd.read_excel(file_like, header=header_row, sheet_name=sheet_name, dtype=object, engine="openpyxl")
    if isinstance(df, dict):
        first_name = sheet_name or list(df.keys())[0]
        df = df[first_name]

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [normalize_header(c) for c in df.columns]
    df = df.dropna(how="all")

    # 3) Initial mapping (TL/Description/MFN/Quota + years, but NOT notes/imports)
    start_year = find_start_year(raw)
    initial_map = build_header_mapping(list(df.columns), start_year)
    applied_map = {k: v[0] for k, v in initial_map.items()}
    df = df.rename(columns=applied_map)

    rename_log: List[Tuple[str, str, str]] = [(k, v[0], v[1]) for k, v in initial_map.items() if k != v[0]]

    # >>> NEW: repair 4045..4050 -> 2023.. using inferred start_year
    df, _extra = fix_numeric_code_year_headers(df, start_year)
    rename_log.extend(_extra)

    # 4) Order-agnostic special handling (log renames)
    rename_log: List[Tuple[str,str,str]] = [(k, v[0], v[1]) for k, v in initial_map.items() if k != v[0]]

    # 4a) MFN from bare 2023 if needed
    df = ensure_mfn(df, rename_log)

    # 4b) Imports anywhere → imp_2020/21/22
    df = tag_imports_anywhere(df, rename_log)

    # 4c) Notes anywhere → Notes, Notes 1, Notes 2, ...
    df = assign_notes_anywhere(df, rename_log)

    # 5) Coalesce any accidental duplicates after all renames
    df = coalesce_duplicate_columns(df)

    # 6) Validate and coerce types
    issues, warnings = [], []

    # Ensure TL
    if "TL" not in df.columns:
        first_col = df.columns[0]
        try:
            match_ratio = as_series(df, first_col).astype(str).str.match(r"^\d{4,10}$").mean()
        except Exception:
            match_ratio = 0
        if match_ratio > 0.6:
            df.rename(columns={first_col: "TL"}, inplace=True)
            rename_log.append((str(first_col), "TL", "heuristic code column"))
            warnings.append(f"'TL' not found; used first column '{first_col}' as TL.")
        else:
            issues.append("Missing required 'TL' column after harmonization.")

    # Ensure Description
    if "Description" not in df.columns:
        try:
            textiness = {c: as_series(df, c).astype(str).str.len().mean() for c in df.columns}
            cand = max(textiness, key=textiness.get)
        except Exception:
            cand = None
        if cand and as_series(df, cand).astype(str).str.len().mean() >= 10:
            df.rename(columns={cand: "Description"}, inplace=True)
            rename_log.append((str(cand), "Description", "heuristic text column"))
            warnings.append(f"'Description' not found; used '{cand}' as Description.")
        else:
            issues.append("Missing required 'Description' column after harmonization.")

    # MFN numeric
    if "MFN" in df.columns:
        df["MFN"] = _to_numeric_series(as_series(df, "MFN"))
        if df["MFN"].isna().mean() > 0.2:
            warnings.append("More than 20% of 'MFN' values could not be parsed as numeric.")
    else:
        issues.append("Missing required 'MFN' column after harmonization.")

    # Quota normalization
    if "Quota" in df.columns:
        tmp = _to_numeric_series(as_series(df, "Quota"))
        if tmp.notna().mean() >= 0.5:
            df["Quota"] = tmp
        else:
            warnings.append("Most 'Quota' values are non-numeric; left as text.")

    # Notes cleanup (any Notes*, including base 'Notes')
    for c in list(df.columns):
        if str(c) == "Notes" or re.fullmatch(r"Notes \d+", str(c)):
            s = as_series(df, c)
            df[c] = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Year columns (preferential rates) = exact 20xx (but exclude imports)
    year_cols = [c for c in df.columns if re.fullmatch(r"(20\d{2})", str(c))]
    for yc in year_cols:
        df[yc] = _to_numeric_series(as_series(df, yc))

    # TL / Description hygiene
    if "TL" in df.columns:
        df["TL"] = as_series(df, "TL").astype(str).str.strip()
        if df["TL"].duplicated().any():
            warnings.append("Duplicate 'TL' values detected.")
        if df["TL"].isna().any() or (df["TL"] == "").any():
            issues.append("Empty 'TL' codes found.")
    if "Description" in df.columns:
        df["Description"] = as_series(df, "Description").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Empty columns info
    empty_rate_cols = [c for c in year_cols if as_series(df, c).notna().mean() == 0]
    if empty_rate_cols:
        warnings.append(f"Year columns with no data: {', '.join(empty_rate_cols)}.")

    # Build meta
    meta = {
        "header_row": header_row,
        "start_year": start_year,
        "issues": issues,
        "warnings": warnings,
        "year_columns": year_cols,
        "import_columns": [c for c in df.columns if str(c).startswith("imp_")],
        "renames": rename_log,
    }
    return df, meta

def save_output_to_cwd(df: pd.DataFrame, uploaded_name: str) -> str:
    base, _ = os.path.splitext(os.path.basename(uploaded_name or "submission.xlsx"))
    out_name = f"{base}_harmonized.xlsx"
    out_path = os.path.join(os.getcwd(), out_name)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return out_path

# ===== Streamlit UI =====
st.set_page_config(page_title="RTA Data Harmonizer", layout="wide")
st.title("RTA Data Harmonizer")
st.caption("Upload an RTA Excel submission. Order-agnostic header detection; columns harmonized to TL, Description, MFN, Quota, Notes(+ numbered); imports tagged as imp_2020/21/22.")

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
sheet_choice = None

if uploaded:
    try:
        xls = pd.ExcelFile(uploaded)
        if len(xls.sheet_names) > 1:
            sheet_choice = st.selectbox("Select sheet", xls.sheet_names)
        uploaded.seek(0)
    except Exception as e:
        st.error(f"Could not read workbook: {e}")

if uploaded and st.button("Process", type="primary"):
    with st.spinner("Harmonizing..."):
        try:
            original_name = getattr(uploaded, "name", "submission.xlsx")
            df_clean, meta = harmonize_excel(uploaded, sheet_choice)

            header_row = meta.get("header_row", 0)
            start_year = meta.get("start_year")
            year_cols = meta.get("year_columns", [])
            import_cols = meta.get("import_columns", [])
            issues = meta.get("issues", [])
            warnings = meta.get("warnings", [])
            renames = meta.get("renames", [])

            md = f"""# Harmonization Summary

**Header row detected:** {header_row + 1} (1-based)  
**Inferred start year:** {start_year if start_year else "Not found"}  
**Preferential rate years:** {", ".join(year_cols) if year_cols else "None"}  
**Import value columns:** {", ".join(import_cols) if import_cols else "None"}

## Issues
- {(chr(10) + "- ").join(issues) if issues else "None detected"}

## Warnings
- {(chr(10) + "- ").join(warnings) if warnings else "None"}
"""
            st.markdown(llm_polish(md))

            # Renaming Summary
            if renames:
                st.subheader("Renaming Summary")
                ren_df = pd.DataFrame(renames, columns=["Original Header", "Renamed To", "Reason"])
                st.dataframe(ren_df, use_container_width=True, height=260)

            st.subheader("Cleaned Data Preview")
            st.dataframe(df_clean.head(50), use_container_width=True)

            out_path = save_output_to_cwd(df_clean, uploaded_name=original_name)
            st.success(f"Cleaned file saved to working directory:\n`{out_path}`")

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_clean.to_excel(writer, index=False, sheet_name="Data")
            buf.seek(0)
            st.download_button(
                label="Download cleaned Excel",
                data=buf,
                file_name=os.path.basename(out_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Processing failed: {e}")
