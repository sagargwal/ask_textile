from langchain_text_splitters import CharacterTextSplitter
import json

def chunk_record(record, source_type):
    text = record.get("transcript") or record.get("content", "")
    if not text:
        return []
    
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1600,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = splitter.split_text(text)
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "course_id":    record["course_id"],
            "course_title": record["course_title"],
            "professor":    record["professor"],
            "institute":    record["institute"],
            "lecture_name": record["lecture_name"],
            "source_type":  source_type,
            "source_url":   record.get("youtube_url") or record.get("lecture_url", ""),
            "chunk_index":  i,
            "chunk_total":  len(chunks),
            "chunk_text":   chunk,
            "char_count":   len(chunk)
        })
    return results

all_chunks = []

# ── Source 1: HTML lectures ────────────────────────────────
with open("../RAG/textile_html_lectures_v2.json") as f:
    html_lectures = json.load(f)

for record in html_lectures:
    all_chunks.extend(chunk_record(record, "html"))

html_count = len(all_chunks)
print(f"HTML chunks:   {html_count}")

# ── Source 2: Video transcripts ───────────────────────────
with open("../RAG/textile_transcripts.json") as f:
    video_lectures = json.load(f)

for record in video_lectures:
    all_chunks.extend(chunk_record(record, "video"))

video_count = len(all_chunks) - html_count
print(f"Video chunks:  {video_count}")

# ── Summary ───────────────────────────────────────────────
print(f"\nTotal chunks:     {len(all_chunks)}")
print(f"Avg chunk size:   {sum(c['char_count'] for c in all_chunks) // len(all_chunks)} chars")

# ── Save ──────────────────────────────────────────────────
with open("textile_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Saved → textile_chunks.json")