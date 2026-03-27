import requests
from bs4 import BeautifulSoup
import json
import time
import re

# For PDF extraction
import pymupdf  # pip install pymupdf

headers = {"User-Agent": "Mozilla/5.0"}

def scrape_html(url):
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_pdf(url):
    r = requests.get(url, headers=headers, timeout=15)
    # Write bytes to temp file
    with open("/tmp/temp_lecture.pdf", "wb") as f:
        f.write(r.content)
    # Extract text using pymupdf
    doc = pymupdf.open("/tmp/temp_lecture.pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_lecture(url):
    """Detect content type and scrape accordingly"""
    r = requests.head(url, headers=headers, timeout=10)
    content_type = r.headers.get("content-type", "").lower()
    
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        print(f"      📄 PDF detected")
        return scrape_pdf(url), "pdf"
    
    elif "html" in content_type or url.lower().endswith((".htm", ".html")):
        print(f"      🌐 HTML detected")
        return scrape_html(url), "html"
    
    else:
        print(f"      ❓ Unknown type: {content_type}, trying HTML")
        return scrape_html(url), "html"

def decode_data_array(data_array):
    """Extract lecture links from SvelteKit __data.json"""
    lectures = []
    for item in data_array:
        if isinstance(item, dict) and "lecturelink" in item:
            link_idx = item["lecturelink"]
            name_idx = item.get("name")
            link = data_array[link_idx] if isinstance(link_idx, int) else link_idx
            name = data_array[name_idx] if isinstance(name_idx, int) else name_idx
            if link and isinstance(link, str) and link.startswith("http"):
                lectures.append({"name": name, "url": link})
    return lectures

# ── Load courses ───────────────────────────────────────────
with open("textile_courses.json") as f:
    all_courses = json.load(f)

with open("textile_courses_with_youtube.json") as f:
    video_courses = json.load(f)

video_ids = {c["course_id"] for c in video_courses}
html_courses = [c for c in all_courses if c["course_id"] not in video_ids]

print(f"Courses to scrape: {len(html_courses)}")

# ── Scrape ─────────────────────────────────────────────────
all_lectures = []
failed = []
stats = {"html": 0, "pdf": 0, "failed": 0}

for course in html_courses:
    course_id = course["course_id"]
    print(f"\n📘 {course['title']}")
    
    # Get lecture list
    try:
        r = requests.get(
            f"https://nptel.ac.in/courses/{course_id}/__data.json",
            headers=headers, timeout=10
        )
        data_array = r.json()["nodes"][1]["data"]
        lectures = decode_data_array(data_array)
        print(f"   Found {len(lectures)} lectures")
    except Exception as e:
        print(f"   ❌ Could not get lecture list: {e}")
        continue
    
    for lecture in lectures:
        try:
            text, content_type = scrape_lecture(lecture["url"])
            
            if not text or len(text) < 50:
                raise Exception("Empty or too short content")
            
            record = {
                "course_id":    course_id,
                "course_title": course["title"],
                "professor":    course["professor"],
                "institute":    course["institute"],
                "lecture_name": lecture["name"],
                "lecture_url":  lecture["url"],
                "content_type": content_type,  # ← "html" or "pdf"
                "content":      text,
                "word_count":   len(text.split())
            }
            all_lectures.append(record)
            stats[content_type] += 1
            print(f"   ✅ {lecture['name'][:50]} ({len(text.split())} words)")
        
        except Exception as e:
            print(f"   ❌ {lecture['name'][:50]}: {e}")
            failed.append({
                "course_id":    course_id,
                "lecture_name": lecture["name"],
                "url":          lecture["url"],
                "reason":       str(e)
            })
            stats["failed"] += 1
        
        time.sleep(0.5)

# ── Save ───────────────────────────────────────────────────
with open("textile_html_lectures_v2.json", "w", encoding="utf-8") as f:
    json.dump(all_lectures, f, indent=2, ensure_ascii=False)

with open("html_failed_v2.json", "w", encoding="utf-8") as f:
    json.dump(failed, f, indent=2, ensure_ascii=False)

total_words = sum(r["word_count"] for r in all_lectures)
print(f"\n{'='*50}")
print(f"✅ HTML lectures:  {stats['html']}")
print(f"✅ PDF lectures:   {stats['pdf']}")
print(f"❌ Failed:         {stats['failed']}")
print(f"📝 Total words:    {total_words:,}")
print(f"📁 Saved → textile_html_lectures_v2.json")