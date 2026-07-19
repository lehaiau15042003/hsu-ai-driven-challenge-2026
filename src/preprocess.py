"""
Data preprocessing script for HSU AI Driven Challenge 2026.
Processes raw CSV files into clean, deduplicated, normalized processed CSVs.
"""

import pandas as pd
import re
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def process_toxic_chat():
    """
    Process toxic_chat_raw.csv -> toxic_chat_clean.csv

    Raw schema:
        conv_id, user_input, model_output, human_annotation (bool),
        toxicity (int), jailbreaking (int), openai_moderation

    Processed schema:
        conv_id, prompt, model_output, label_unsafe (int), toxicity (int), jailbreaking (int)

    Pipeline (per spec):
        Step 2 - Rename 'user_input' -> 'prompt'
                 label_unsafe = 1 if toxicity == 1 OR jailbreaking == 1 else 0
        Step 3 - Drop null/empty 'prompt'
                 Deduplicate by 'prompt'
                 Normalize whitespace with RegEx (NO removal of special chars)
                 Filter out prompts with <= 1 word
        Step 4 - Print % distribution of Safe vs Unsafe
                 Export to toxic_chat_clean.csv
    """
    print("=" * 60)
    print("Processing toxic_chat (following pipeline spec)...")

    raw_path = os.path.join(RAW_DIR, 'toxic_chat_raw.csv')
    out_path = os.path.join(PROCESSED_DIR, 'toxic_chat_clean.csv')

    df = pd.read_csv(raw_path)
    print(f"  Raw shape          : {df.shape}")
    print(f"  Raw duplicates     : {df.duplicated().sum()}")
    print(f"  Raw null values    :\n{df.isnull().sum().to_string()}")

    # ── Bước 2: Chuẩn hóa cột ────────────────────────────────────────
    # Đổi tên user_input -> prompt
    df = df.rename(columns={'user_input': 'prompt'})

    # Đảm bảo kiểu int cho toxicity và jailbreaking
    df['toxicity']     = df['toxicity'].astype(int)
    df['jailbreaking'] = df['jailbreaking'].astype(int)

    # Tạo label_unsafe: = 1 nếu toxicity == 1 HOẶC jailbreaking == 1
    df['label_unsafe'] = ((df['toxicity'] == 1) | (df['jailbreaking'] == 1)).astype(int)

    # ── Bước 3: Làm sạch văn bản ─────────────────────────────────────
    # Missing values: xóa dòng null/empty ở cột prompt
    before = len(df)
    df = df.dropna(subset=['prompt'])
    df = df[df['prompt'].astype(str).str.strip().str.len() > 0]
    print(f"  Dropped null/empty : {before - len(df)} rows")

    # Duplicates: xóa dòng có prompt giống hệt nhau
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['prompt'], keep='first')
    print(f"  Dedup by prompt    : removed {before_dedup - len(df)} rows")

    # Whitespace: chuẩn hóa khoảng trắng thừa và newline liên tiếp bằng RegEx
    # KHÔNG xóa ký tự đặc biệt ({, [, <, /, \, %, $, ...)
    df['prompt'] = (
        df['prompt']
        .astype(str)
        .str.strip()
        .apply(lambda x: re.sub(r'\n{2,}', '\n', x))   # Gộp nhiều newline liên tiếp
        .apply(lambda x: re.sub(r' {2,}', ' ', x))      # Gộp nhiều khoảng trắng liên tiếp
        .apply(lambda x: re.sub(r'\t+', ' ', x))        # Thay tab bằng 1 khoảng trắng
    )

    # Length filter: lọc bỏ prompt chỉ có <= 1 từ
    before_len = len(df)
    df = df[df['prompt'].str.split().str.len() > 1]
    print(f"  Length filter (<= 1 word): removed {before_len - len(df)} rows")

    # Chuẩn hóa các cột text còn lại
    df['conv_id']      = df['conv_id'].astype(str).str.strip()
    df['model_output'] = df['model_output'].astype(str).str.strip()

    # ── Chọn và sắp xếp cột ─────────────────────────────────────────
    df = df[['conv_id', 'prompt', 'model_output', 'label_unsafe', 'toxicity', 'jailbreaking']]
    df = df.reset_index(drop=True)

    # ── Bước 4: Đánh giá & Xuất ─────────────────────────────────────
    total        = len(df)
    safe_count   = (df['label_unsafe'] == 0).sum()
    unsafe_count = (df['label_unsafe'] == 1).sum()
    print(f"\n  Final shape        : {df.shape}")
    print(f"  Remaining dupes    : {df.duplicated().sum()}")
    print(f"\n  === Phân phối nhãn (Label Distribution) ===")
    print(f"  Safe   (0) : {safe_count:>6} rows  ({safe_count / total * 100:.2f}%)")
    print(f"  Unsafe (1) : {unsafe_count:>6} rows  ({unsafe_count / total * 100:.2f}%)")
    print(f"  Total      : {total:>6} rows")

    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return df


def process_safetybench():
    """
    Process safetybench_raw.csv -> safetybench_processed.csv  (theo notebook 1.3)

    Raw schema:
        text, label (0=safe, 1=borderline, 2=unsafe), response_refusal_label,
        final_turn_role, turn_type, topic, subtopic, source, prompt

    Processed schema:
        prompt, label_unsafe (int), label_original (int),
        topic (int), subtopic (str), source (str)

    Pipeline (per notebook 1.3-safetybench-pipeline.ipynb):
        Step 1 - Load data
        Step 2 - Extract prompt từ cột 'text' (khớp cả 'user' và 'human' role)
        Step 3 - Map nhãn: label_unsafe = 1 if label != 0 else 0
        Step 4 - Balance classes: oversample minority để cân bằng
        Step 5 - clean_text() + is_garbage() để lọc & làm sạch
        Step 6 - Re-balance sau cleaning (downsample majority)
        Step 7 - Chuẩn hóa schema & export
    """
    from src import clean_text, is_garbage

    RANDOM_STATE = 42

    print("=" * 60)
    print("Processing safetybench (theo notebook 1.3)...")

    raw_path = os.path.join(RAW_DIR, 'safetybench_raw.csv')
    out_path = os.path.join(PROCESSED_DIR, 'safetybench_processed.csv')

    df = pd.read_csv(raw_path, low_memory=False)
    print(f"  Raw shape          : {df.shape}")
    print(f"  Raw duplicates     : {df.duplicated().sum()}")
    print(f"  Raw null values    :\n{df.isnull().sum().to_string()}")

    # ── Bước 2: Extract prompt từ cột 'text' ────────────────────────
    # Regex khớp cả role 'user' và 'human' (theo notebook)
    _USER_RE = re.compile(
        r"role': '(?:user|human)', 'content': \"([^\"]+)\"",
        re.DOTALL,
    )

    def extract_user_prompt(text_str) -> str:
        """Extract the first user/human turn content from the conversation string."""
        if not isinstance(text_str, str):
            return ""
        m = _USER_RE.search(text_str)
        return m.group(1) if m else ""

    # Luôn extract từ 'text' (đây là nguồn gốc đáng tin cậy nhất)
    df["prompt"] = df["text"].apply(extract_user_prompt)
    empty_after = (df["prompt"] == "").sum()
    print(f"  Extracted from text: empty={empty_after:,}  OK={len(df) - empty_after:,}")

    # ── Bước 3: Map nhãn → label_unsafe ─────────────────────────────
    df['label_original'] = df['label'].astype(int)          # 0=safe, 1=borderline, 2=unsafe
    df['label_unsafe']   = (df['label'] != 0).astype(int)   # binary: 0=safe, 1=unsafe/borderline

    # Ensure topic & subtopic/source
    df['topic']    = df['topic'].astype(int)
    df['subtopic'] = df['subtopic'].fillna('unknown').astype(str).str.strip()
    df['source']   = df['source'].astype(str).str.strip()

    # Drop empty prompt trước khi balance
    before = len(df)
    df = df[df['prompt'].str.len() > 0].reset_index(drop=True)
    print(f"  Dropped empty prompt: {before - len(df)} rows")

    # ── Bước 4: Balance classes (oversample minority) ─────────────
    df_safe   = df[df['label_unsafe'] == 0]
    df_unsafe = df[df['label_unsafe'] == 1]
    print(f"\n  Before balance: safe={len(df_safe):,}  unsafe={len(df_unsafe):,}")

    majority_n = max(len(df_safe), len(df_unsafe))
    if len(df_safe) < majority_n:
        df_safe = df_safe.sample(n=majority_n, replace=True, random_state=RANDOM_STATE)
    else:
        df_unsafe = df_unsafe.sample(n=majority_n, replace=True, random_state=RANDOM_STATE)

    df_balanced = (
        pd.concat([df_safe, df_unsafe])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    print(f"  After balance : {df_balanced['label_unsafe'].value_counts().to_dict()}")
    print(f"  Total rows    : {len(df_balanced):,}")

    # ── Bước 5: clean_text() & is_garbage() ─────────────────────────
    df_balanced['prompt'] = df_balanced['prompt'].apply(clean_text)

    n_before     = len(df_balanced)
    mask_garbage = df_balanced['prompt'].apply(is_garbage)
    df_balanced  = df_balanced[~mask_garbage].reset_index(drop=True)
    print(f"\n  Dropped garbage  : {n_before - len(df_balanced):,} rows")
    print(f"  Remaining        : {len(df_balanced):,}")
    print(f"\n  label_unsafe before re-balance:")
    print(f"  {df_balanced['label_unsafe'].value_counts().to_dict()}")

    # ── Bước 6: Re-balance sau cleaning (downsample majority) ───────
    df_s2 = df_balanced[df_balanced['label_unsafe'] == 0]
    df_u2 = df_balanced[df_balanced['label_unsafe'] == 1]
    n2 = min(len(df_s2), len(df_u2))
    df_s2 = df_s2.sample(n=n2, random_state=RANDOM_STATE)
    df_u2 = df_u2.sample(n=n2, random_state=RANDOM_STATE)
    df_balanced = (
        pd.concat([df_s2, df_u2])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    print(f"\n  After re-balance : {df_balanced['label_unsafe'].value_counts().to_dict()}")
    print(f"  Final total      : {len(df_balanced):,}")

    # ── Bước 7: Chuẩn hóa schema & export ───────────────────────────
    COLS_OUT = ['prompt', 'label_unsafe', 'label_original', 'topic', 'subtopic', 'source']
    COLS_OUT = [c for c in COLS_OUT if c in df_balanced.columns]
    df_out = df_balanced[COLS_OUT].copy()

    total        = len(df_out)
    safe_count   = (df_out['label_unsafe'] == 0).sum()
    unsafe_count = (df_out['label_unsafe'] == 1).sum()
    print(f"\n  Final shape        : {df_out.shape}")
    print(f"  Remaining dupes    : {df_out.duplicated().sum()}")
    print(f"\n  === Phân phối nhãn (Label Distribution) ===")
    print(f"  Safe   (0) : {safe_count:>6} rows  ({safe_count / total * 100:.2f}%)")
    print(f"  Unsafe (1) : {unsafe_count:>6} rows  ({unsafe_count / total * 100:.2f}%)")
    print(f"  Total      : {total:>6} rows")

    df_out.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n  Saved -> {out_path}")
    return df_out


if __name__ == '__main__':
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    tc_df = process_toxic_chat()
    print()
    sb_df = process_safetybench()
    print()
    print("=" * 60)
    print("All preprocessing complete.")
    print(f"  toxic_chat_processed     : {tc_df.shape}")
    print(f"  safetybench_processed: {sb_df.shape}")
