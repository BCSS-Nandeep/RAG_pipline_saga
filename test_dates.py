import re
from datetime import datetime, timezone, timedelta
from dateutil import parser
import traceback

def extract_dates(question: str):
    """
    Returns (date_from, date_to) as YYYY-MM-DD strings.
    """
    q = question.lower()
    now = datetime.now(timezone.utc)
    today = now.date()
    yesterday = today - timedelta(days=1)
    
    # Check explicit format: "from September 10 to September 15"
    m_range = re.search(r'from\s+([a-z]+ \d{1,2}(?:st|nd|rd|th)?)\s+to\s+([a-z]+ \d{1,2}(?:st|nd|rd|th)?)', q)
    if m_range:
        try:
            d1 = parser.parse(m_range.group(1)).date()
            d2 = parser.parse(m_range.group(2)).date()
            if d1.month > today.month and d1.year == today.year:
                d1 = d1.replace(year=today.year - 1)
            if d2.month > today.month and d2.year == today.year:
                d2 = d2.replace(year=today.year - 1)
            return (d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d"))
        except Exception:
            pass

    # Check explicit date: "on september 10" or "september 10th"
    m_single = re.search(r'(?:on\s+)?([a-z]+ \d{1,2}(?:st|nd|rd|th)?)', q)
    if m_single:
        try:
            d = parser.parse(m_single.group(1)).date()
            if d.month > today.month and d.year == today.year:
                d = d.replace(year=today.year - 1)
            return (d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"))
        except Exception:
            pass

    # YYYY-MM-DD
    m_iso = re.search(r'(\d{4}-\d{2}-\d{2})(?:\s+to\s+(\d{4}-\d{2}-\d{2}))?', q)
    if m_iso:
        if m_iso.group(2):
            return (m_iso.group(1), m_iso.group(2))
        return (m_iso.group(1), m_iso.group(1))

    if "today" in q:
        return (today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        
    if "yesterday" in q:
        return (yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d"))

    return None, None

print(extract_dates("What happened today?"))
print(extract_dates("Show alerts from today"))
print(extract_dates("What happened yesterday?"))
print(extract_dates("Show alerts on September 10"))
print(extract_dates("Show alerts from September 10 to September 15"))
print(extract_dates("2026-09-10 to 2026-09-15"))

