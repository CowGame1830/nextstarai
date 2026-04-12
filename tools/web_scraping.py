import csv
import time
import random
import os
import unicodedata
import re

from playwright.sync_api import sync_playwright

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "player_data")

INPUT_CSV   = os.path.join(DATA_DIR, "player_names_unique.csv")
OUTPUT_CSV  = os.path.join(DATA_DIR, "player_names_unique_scraped.csv")
SESSION_DIR = os.path.join(BASE_DIR, "..", "browser_session")

# match_status values:
#   "exact"        – CSV name matches page name
#   "partial"      – names differ but we still saved the data (review recommended)
#   "first_result" – no exact search result found; clicked first item
#   "not_found"    – profile page never appeared
CSV_COLUMNS = [
    "Player", "matched_name", "match_status",
    "Stamina", "Pace", "Acceleration", "Work Rate", "source_url",
]
# ──────────────────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lowercase, strip accents, remove non-alphanumeric for loose comparison."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def names_match(csv_name: str, page_name: str) -> bool:
    """Return True if the two names are the same after normalization."""
    return normalize(csv_name) == normalize(page_name)


def load_player_names() -> list[str]:
    names = []
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found!")
        return names
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Player", "").strip()
            if name:
                names.append(name)
    print(f"Loaded {len(names)} player names from input CSV.")
    return names


def load_checkpoint() -> dict[str, dict]:
    """
    Read OUTPUT_CSV; return {player_name: row} for rows that already have
    a match_status set (meaning they were processed).
    """
    done = {}
    if not os.path.exists(OUTPUT_CSV):
        return done
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name   = row.get("Player", "").strip()
            status = row.get("match_status", "").strip()
            if name and status:        # any status means already processed
                done[name] = row
    print(f"Checkpoint: {len(done)} players already processed.")
    return done


def write_checkpoint(all_names: list[str], results: dict[str, dict]) -> None:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for name in all_names:
            entry = results.get(name, {})
            writer.writerow({
                "Player":        name,
                "matched_name":  entry.get("matched_name", ""),
                "match_status":  entry.get("match_status", ""),
                "Stamina":       entry.get("Stamina", ""),
                "Pace":          entry.get("Pace", ""),
                "Acceleration":  entry.get("Acceleration", ""),
                "Work Rate":     entry.get("Work Rate", ""),
                "source_url":    entry.get("source_url", ""),
            })


def get_page_player_name(page) -> str:
    """
    Try to read the player's displayed name from the profile page.
    Returns empty string if not found.
    """
    # fmplayer.net typically shows the name in an <h1> or a heading element
    for selector in ["h1", "h1.player-name", ".player-name", "[class*='player-name']"]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=2000):
                text = el.text_content().strip()
                if text:
                    return text
        except Exception:
            pass
    return ""


def extract_attributes(page) -> dict:
    attrs = {}
    for key in ["Stamina", "Pace", "Acceleration", "Work Rate"]:
        try:
            selector = f'li.player-attr-list-item:has-text("{key}") span.player-attr-list-item-value'
            el = page.locator(selector).first
            if el.is_visible(timeout=5000):
                val = el.text_content().strip()
                attrs[key] = int(val) if val.isdigit() else val
            else:
                attrs[key] = None
        except Exception:
            attrs[key] = None
    return attrs


def run_scraping():
    all_names = load_player_names()
    if not all_names:
        return

    results = load_checkpoint()
    pending = [n for n in all_names if n not in results]
    total   = len(all_names)
    print(f"Total: {total} | Done: {total - len(pending)} | Remaining: {len(pending)}\n")

    if not pending:
        print("All players already processed!")
        write_checkpoint(all_names, results)
        return

    with sync_playwright() as p:
        print(f"Opening browser (session: {SESSION_DIR}) …")
        context = p.chromium.launch_persistent_context(
            user_data_dir=SESSION_DIR,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )

        page = context.pages[0]

        try:
            page.goto("https://fmplayer.net/", wait_until="load")
        except Exception as e:
            print(f"Navigation error: {e}")

        # ── Cloudflare gate ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("ACTION REQUIRED: SOLVE THE CLOUDFLARE CHALLENGE NOW.")
        print("Script resumes automatically once the home page loads.")
        print("=" * 60 + "\n")

        search_trigger = 'span.cursor-text:has-text("search player or club")'
        try:
            page.wait_for_selector(search_trigger, timeout=0)
            print("Site verified. Starting automation …\n")
        except Exception as e:
            print(f"Error: {e}")
            context.close()
            return

        modal = "#search-modal.show"

        for idx, player_name in enumerate(pending, start=1):
            done_so_far = total - len(pending) + idx
            print(f"[{done_so_far}/{total}] {player_name}")

            try:
                # ── Open search modal ─────────────────────────────────────────
                if not page.is_visible(modal):
                    page.click(search_trigger, timeout=10000)
                page.wait_for_selector(modal, timeout=10000)

                # ── Type name ─────────────────────────────────────────────────
                inp = page.locator("#search-modal input").first
                inp.click()
                page.keyboard.press("Control+A")
                page.keyboard.press("Backspace")
                inp.fill(player_name)
                time.sleep(random.uniform(0.6, 1.2))
                inp.press("Enter")

                # ── Navigate to profile ───────────────────────────────────────
                clicked_status = "exact"   # assume best case
                on_profile = page.is_visible("li.player-attr-list-item", timeout=3000)

                if not on_profile:
                    page.wait_for_timeout(2000)

                    # Try exact name match in results list
                    exact_link = page.locator(
                        f'a.search-item-link:has(p.fw-bold:has-text("{player_name}"))'
                    ).first

                    if exact_link.is_visible(timeout=2000):
                        exact_link.click()
                        clicked_status = "exact"
                    else:
                        # Fallback: first available result
                        first_link = page.locator("a.search-item-link").first
                        if first_link.is_visible(timeout=5000):
                            print(f"  ! No exact result — clicking first result")
                            first_link.click()
                            clicked_status = "first_result"
                        else:
                            print(f"  ✗ No search results at all")
                            results[player_name] = {"match_status": "not_found"}
                            write_checkpoint(all_names, results)
                            # reset
                            page.keyboard.press("Escape")
                            page.wait_for_timeout(500)
                            page.goto("https://fmplayer.net/", wait_until="domcontentloaded")
                            page.wait_for_selector(search_trigger, timeout=15000)
                            time.sleep(random.uniform(2.0, 4.5))
                            continue

                    page.wait_for_timeout(3000)

                # ── Extract data + check name match ───────────────────────────
                try:
                    page.wait_for_selector("li.player-attr-list-item", timeout=10000)

                    page_name = get_page_player_name(page)
                    attrs     = extract_attributes(page)
                    attrs["source_url"]   = page.url
                    attrs["matched_name"] = page_name

                    # Determine final match_status
                    if clicked_status == "first_result":
                        attrs["match_status"] = "first_result"
                    elif page_name and not names_match(player_name, page_name):
                        attrs["match_status"] = "partial"
                        print(f"  ⚠ Name mismatch — CSV: '{player_name}' | Page: '{page_name}'")
                    else:
                        attrs["match_status"] = "exact"

                    print(f"  ✓ [{attrs['match_status']}] {attrs}")
                    results[player_name] = attrs
                    write_checkpoint(all_names, results)

                except Exception:
                    print(f"  ✗ Profile page not found — skipping")
                    results[player_name] = {"match_status": "not_found"}
                    write_checkpoint(all_names, results)
                    if page.is_visible(modal):
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(500)

                # ── Go home for next player ───────────────────────────────────
                page.goto("https://fmplayer.net/", wait_until="domcontentloaded")
                page.wait_for_selector(search_trigger, timeout=15000)
                time.sleep(random.uniform(2.0, 4.5))

            except Exception as e:
                print(f"  ✗ Error: {e}")
                try:
                    if page.is_visible("#search-modal.show"):
                        page.keyboard.press("Escape")
                    else:
                        page.goto("https://fmplayer.net/")
                        page.wait_for_selector(search_trigger, timeout=15000)
                except Exception:
                    pass
                continue

        # ── Summary ───────────────────────────────────────────────────────────
        counts = {"exact": 0, "partial": 0, "first_result": 0, "not_found": 0}
        for v in results.values():
            s = v.get("match_status", "")
            if s in counts:
                counts[s] += 1

        print(f"\nFinished! Results:")
        print(f"  ✓ Exact match   : {counts['exact']}")
        print(f"  ⚠ Partial match : {counts['partial']}")
        print(f"  ? First result  : {counts['first_result']}")
        print(f"  ✗ Not found     : {counts['not_found']}")
        print(f"\nCSV → {OUTPUT_CSV}")
        input("\nPress Enter to close the browser …")
        context.close()


if __name__ == "__main__":
    run_scraping()
