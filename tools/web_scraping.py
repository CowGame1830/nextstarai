import json
import time
import random
import os
from playwright.sync_api import sync_playwright

# Configuration
PLAYER_NAMES_JSON = "player_data/player.json"
OUTPUT_JSON = "player_data/fm_scraped_attributes.json"
SESSION_DIR = "browser_session" # Stores your Cloudflare tokens/cookies

def load_players():
    # Adjusted path to check locally for player_data
    if not os.path.exists(PLAYER_NAMES_JSON):
        # Try one level up if not found locally
        alt_path = os.path.join("..", PLAYER_NAMES_JSON)
        if os.path.exists(alt_path):
            with open(alt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"Error: {PLAYER_NAMES_JSON} not found!")
        return []
    with open(PLAYER_NAMES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_attributes(page):
    """
    Extracts the 4 target attributes from the current player page.
    """
    target_keys = ["Stamina", "Pace", "Acceleration", "Work Rate"]
    extracted = {}
    
    for key in target_keys:
        try:
            # Selector logic: Find the <li> containing the key name, then get the value span
            selector = f'li.player-attr-list-item:has-text("{key}") span.player-attr-list-item-value'
            element = page.locator(selector).first
            if element.is_visible(timeout=5000):
                value = element.text_content().strip()
                extracted[key] = int(value) if value.isdigit() else value
            else:
                extracted[key] = None
        except Exception:
            extracted[key] = None
            
    return extracted

def run_scraping():
    players = load_players()
    if not players:
        print("No players found to process.")
        return
        
    print(f"Loaded {len(players)} players.")
    # Load existing results for checkpointing
    scraped_results = []
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                scraped_results = json.load(f)
            print(f"Loaded {len(scraped_results)} existing results from checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            scraped_results = []

    already_scraped = {res["player_name"] for res in scraped_results if "player_name" in res}


    with sync_playwright() as p:
        # 1. Open browser with persistent context
        print(f"Opening browser... (Session data: {SESSION_DIR})")
        context = p.chromium.launch_persistent_context(
            user_data_dir=SESSION_DIR,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        page = context.pages[0]
        
        print("Navigating to https://fmplayer.net/...")
        try:
            page.goto("https://fmplayer.net/", wait_until="load")
        except Exception as e:
            print(f"Initial navigation error: {e}")

        # 2. Hybrid Manual Step: Wait for you to solve Cloudflare
        print("\n" + "="*60)
        print("ACTION REQUIRED: PLEASE SOLVE THE CLOUDFLARE CHALLENGE NOW.")
        print("Once the home page loads, the script will resume automatically.")
        print("="*60 + "\n")
        
        search_trigger_selector = 'span.cursor-text:has-text("search player or club")'
        try:
            page.wait_for_selector(search_trigger_selector, timeout=0) 
            print("\nVerification passed! Site detected. Starting automation...")
        except Exception as e:
            print(f"An error occurred while waiting for site: {e}")
            context.close()
            return

        # 3. Automated Search Loop
        for i, player_name in enumerate(players):
            if player_name in already_scraped:
                continue
            
            print(f"\n[{i+1}/{len(players)}] Processing: {player_name}")
            
            try:
                # 3a. Ensure we are ready to search
                modal_selector = "#search-modal.show"
                is_modal_open = page.is_visible(modal_selector)
                
                if not is_modal_open:
                    # Only click the trigger if the modal isn't already open
                    print("Opening search modal...")
                    page.click(search_trigger_selector, timeout=10000)
                
                # Wait for the search modal to definitely be visible
                page.wait_for_selector(modal_selector, timeout=10000)
                
                # 3b. Find input, clear it, and type
                input_field = page.locator("#search-modal input").first
                input_field.click() # Focus
                # Select all and backspace to clear any previous text
                page.keyboard.press("Control+A")
                page.keyboard.press("Backspace")
                
                input_field.fill(player_name)
                time.sleep(random.uniform(0.6, 1.2))
                input_field.press("Enter")
                
                # 3d. Check if we ARE on a player page or if we should click a result
                is_on_profile = page.is_visible("li.player-attr-list-item", timeout=3000)
                
                if not is_on_profile:
                    # 3c. Wait for search Results or direct Redirect
                    try:
                        # Give it a moment to load potential results in the modal
                        page.wait_for_timeout(2000) 
                        
                        # Target the specific 'search-item-link'
                        # Try exact match first
                        result_link = page.locator(f'a.search-item-link:has(p.fw-bold:has-text("{player_name}"))').first
                        
                        if not result_link.is_visible(timeout=2000):
                            # Fallback: Just click the first result if no exact match found
                            print(f"Precise match not found for '{player_name}'. Clicking first available result...")
                            result_link = page.locator('a.search-item-link').first

                        if result_link.is_visible(timeout=5000):
                            print(f"Clicking link for {player_name}...")
                            result_link.click()
                            # Wait for navigation/load
                            page.wait_for_timeout(3000)
                    except Exception as e:
                        print(f"Note: Search selection step skipped or failed: {e}")

                # Final check: Are we on the profile page now?
                try:
                    page.wait_for_selector("li.player-attr-list-item", timeout=10000)
                    attributes = extract_attributes(page)
                    attributes["player_name"] = player_name
                    attributes["source_url"] = page.url
                    print(f"Extracted: {attributes}")
                    scraped_results.append(attributes)
                    
                    # Reset state: Go back home for the next search
                    page.goto("https://fmplayer.net/", wait_until="domcontentloaded")
                    page.wait_for_selector(search_trigger_selector)
                    
                    # Periodic save every 5 players
                    if len(scraped_results) % 5 == 0:
                        print(f"Saving progress... ({len(scraped_results)} players)")
                        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                            json.dump(scraped_results, f, indent=4)
                    
                    # Human-like delay between players
                    time.sleep(random.uniform(2.0, 5.0))

                except Exception:
                    print(f"Could not find a profile page for {player_name} (timed out waiting for attributes).")
                    # Close modal if it's still stuck open
                    if page.is_visible(modal_selector):
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(1000)
                
            except Exception as e:
                print(f"Error processing {player_name}: {e}")
                # Emergency reset: close modal and/or go home
                try:
                    if page.is_visible("#search-modal.show"):
                        page.keyboard.press("Escape")
                    else:
                        page.goto("https://fmplayer.net/")
                except:
                    pass
                continue

        # 4. Save results to JSON
        print(f"\nSaving results to {OUTPUT_JSON}...")
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(scraped_results, f, indent=4)

        print("\nAll tasks complete!")
        input("Press Enter to close the browser...")
        context.close()

if __name__ == "__main__":
    run_scraping()
