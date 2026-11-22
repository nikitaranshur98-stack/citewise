import time


def call_with_retry(fn, tries=4, base_delay=5):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            temporary = any(code in str(e) for code in ("503", "UNAVAILABLE", "429"))
            if not temporary or attempt == tries:
                raise
            wait = base_delay * 2 ** (attempt - 1)
            print(f"Temporary API error, retrying in {wait}s (attempt {attempt}/{tries})...")
            time.sleep(wait)