import asyncio
import base64
import json
import random
import time
from datetime import datetime, timedelta
import sys

# Import colorama for colored console output
from colorama import init, Fore, Style

# Initialize Colorama
init(autoreset=True)

# Import pandas and pandas_ta for technical indicators
import pandas as pd
import pandas_ta as ta

from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException, \
    ElementClickInterceptedException

# Assuming driver.py contains get_driver() setup for undetected_chromedriver
try:
    from driver import get_driver, companies

    if not isinstance(companies, dict):
        companies = {}
        print(
            f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} 'companies' variable in driver.py is not a dictionary. Initialized as empty dict.")
except ImportError:
    print(
        f"{Fore.RED}[ERROR]{Style.RESET_ALL} Could not import 'driver.py'. Make sure it exists in the same directory and 'get_driver' and 'companies' are defined within it.")
    print("Example 'driver.py' content:")
    print(f"{Fore.CYAN}```python{Style.RESET_ALL}")
    print("import undetected_chromedriver as uc")
    print("def get_driver():")
    print("    options = uc.ChromeOptions()")
    print("    # options.add_argument('--headless')")
    print("    return uc.Chrome(options=options)")
    print("companies = {} # You can fill this with actual mappings if needed, e.g., {'EUR/USD': 'EURUSD'}")
    print(f"{Fore.CYAN}```{Style.RESET_ALL}")
    sys.exit(1)

# --- Global Configuration and Variables ---
LENGTH_STACK_MIN = 460
LENGTH_STACK_MAX = 1000
PERIOD = 0
STACK = []
ACTIONS = {}
MAX_ACTIONS = 1
ACTIONS_SECONDS = 10
LAST_REFRESH = datetime.now()
CURRENCY = None
CURRENCY_CHANGE = False
CURRENCY_CHANGE_DATE = datetime.now()
HISTORY_TAKEN = False

MODEL = None
SCALER = None

INIT_DEPOSIT = None
NUMBERS = {
    '0': '11', '1': '7', '2': '8', '3': '9', '4': '4',
    '5': '5', '6': '6', '7': '1', '8': '2', '9': '3',
}
IS_AMOUNT_SET = True
AMOUNTS = []
MARTINGALE_COEFFICIENT = 2.0
MAX_CONSECUTIVE_LOSSES = 5

# New: Trading Limits & Performance Tracking (Amount-based)
DAILY_PROFIT_LIMIT_AMOUNT = 50.0
DAILY_LOSS_LIMIT_AMOUNT = 20.0
TOTAL_PROFIT_TODAY = 0.0
TOTAL_LOSS_TODAY = 0.0
TODAY_START_DATE = datetime.now().date()
ASSET_PERFORMANCE = {}

# Asset Management
FAVORITE_ASSETS_UI_SELECTOR = ".assets-list__item[data-id]"
FALLBACK_SPECIFIC_SELECTOR = "//li[contains(@class, 'assets-list__item') and contains(., '92%')]"
CURRENT_FAVORITE_ASSET_INDEX = 0
FAVORITE_ASSETS = []

# Strategy Confluence Setting
REQUIRED_CONFLUENCE = 3

# --- Initialize WebDriver ---
driver = get_driver()


# --- Helper Functions (using Selenium) ---

def print_naif_bot_banner():
    """Prints a colorful 'NAIF BOT' banner."""
    print(f"{Fore.BLUE}{Style.BRIGHT}" + "=" * 50)
    print(f"{Fore.BLUE}{Style.BRIGHT}        ███╗   ██╗ █████╗ ██╗ ███████╗  ██████╗ {Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}        ████╗  ██║██╔══██╗██║██╔════╝ ██╔════╝ {Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}        ██╔██╗ ██║███████║██║█████╗   ██║     {Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}        ██║╚██╗██║██╔══██║██║██╔══╝   ██║     {Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}        ██║ ╚████║██║  ██║██║███████╗ ███████╗{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}        ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚══════╝ ╚══════╝{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}              Automated Trading Bot{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}" + "=" * 50 + Style.RESET_ALL)


def close_active_modals_if_any(wait_time=5):
    """
    Attempts to close any active modals/overlays by clicking outside them or pressing ESC.
    This function can be called before any critical UI interaction.
    Returns True if a modal was found and closed, False otherwise.
    """
    modal_closed = False
    common_modal_selectors = [
        'div.drop-down-modal-wrap.active',
        '.mfp-wrap.mfp-close-btn-in.mfp-auto-cursor.mfp-ready',
        '.modal-dialog',
        '.overlay.active',
        '.notification-panel__wrapper',
        'div[data-modal-id]'
    ]

    for selector in common_modal_selectors:
        try:
            modal_element = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            if modal_element.is_displayed():
                print(
                    f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Found active modal/overlay with selector '{selector}'. Attempting to close.")

                ActionChains(driver).move_to_element_with_offset(driver.find_element(By.TAG_NAME, 'body'), 0,
                                                                 0).click().perform()
                time.sleep(0.5)
                ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(0.5)

                WebDriverWait(driver, wait_time).until(
                    EC.invisibility_of_element_located((By.CSS_SELECTOR, selector)),
                    message=f"Modal with selector '{selector}' did not disappear after closing attempt."
                )
                print(
                    f"[{Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Modal with selector '{selector}' closed successfully.")
                modal_closed = True
                break
        except TimeoutException:
            pass
        except Exception as e:
            print(
                f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error closing modal '{selector}': {e}.")

    if not modal_closed:
        print(
            f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No active modals found or they closed themselves. Proceeding.")
    return modal_closed


# Helper to click an element, handling intercepts and retries
def click_element_robustly(by_method, selector, wait_time=15,
                           retry_attempts=10):  # Increased wait_time and retry_attempts
    """
    Attempts to click an element, handling ElementClickInterceptedException by closing modals.
    Returns the clicked element if successful, None otherwise.
    """
    for attempt in range(retry_attempts):
        try:
            # Ensure no modals intercept before clicking
            close_active_modals_if_any()
            time.sleep(1)  # Increased buffer after closing modals

            element = WebDriverWait(driver, wait_time).until(
                EC.element_to_be_clickable((by_method, selector))
            )
            element.click()
            print(
                f"[{Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Clicked element {selector} on attempt {attempt + 1}.")
            return element
        except ElementClickInterceptedException as e:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Click intercepted for {selector}: {e}. Retrying after closing modals (Attempt {attempt + 1}).")
            time.sleep(random.uniform(2, 5))  # Longer random delay before next retry
            close_active_modals_if_any(wait_time=5)  # Aggressively close modals
            time.sleep(random.uniform(2, 5))  # Extra buffer
        except TimeoutException as e:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Timeout waiting for {selector} to be clickable. (Attempt {attempt + 1}). Retrying.")
            time.sleep(random.uniform(3, 7))  # Longer random delay
        except StaleElementReferenceException:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Stale element reference for {selector}. (Attempt {attempt + 1}). Retrying.")
            time.sleep(random.uniform(3, 7))
        except Exception as e:
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error clicking {selector}: {e}. (Attempt {attempt + 1}). Retrying.")
            time.sleep(random.uniform(5, 10))  # Even longer delay for unexpected errors

    print(
        f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to click element {selector} after {retry_attempts} attempts.")
    return None


def load_web_driver():
    """Loads the trading platform URL in the browser and waits for key elements. Retries indefinitely."""
    global driver
    url = 'https://u.shortink.io/cabinet/demo-quick-high-low?utm_campaign=806509&utm_source=affiliate&utm_medium=sr&a=ovlztqbPkiBiOt&ac=github'

    while True:  # Infinite retry loop for page load
        try:
            driver.get(url)
            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Web driver loaded successfully. Waiting for page elements...")

            time.sleep(20)  # Very long initial sleep to ensure page fully renders

            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Checking for initial overlays...")
            for _ in range(10):  # Try closing modals even more times
                if not close_active_modals_if_any(wait_time=10):  # Increased wait time here
                    break
                time.sleep(3)  # Longer buffer after closing modals

            # Use robust click for currency symbol
            if not WebDriverWait(driver, 120).until(  # Increased to 120 seconds
                    EC.presence_of_element_located((By.CLASS_NAME, 'current-symbol'))  # Only check presence here
            ): raise TimeoutException("Currency symbol element not found after long wait.")
            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Currency symbol loaded.")

            header_balance_selector = 'div.right-block__item.js-drop-down-modal-open'
            if not WebDriverWait(driver, 120).until(  # Increased to 120 seconds
                    EC.presence_of_element_located((By.CSS_SELECTOR, header_balance_selector))
            ): raise TimeoutException(
                f"Header balance element with selector '{header_balance_selector}' not found after long wait.")
            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Header balance element loaded.")

            if not WebDriverWait(driver, 120).until(  # Increased to 120 seconds
                    EC.presence_of_element_located((By.CLASS_NAME, 'btn-call'))
            ): raise TimeoutException("Call button not found after long wait.")
            print(f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Trade buttons loaded.")

            # Load favorite assets. This will also retry internally.
            # No sys.exit() here. If it fails, it will raise an exception
            # which will be caught by the outer loop, leading to a full page reload retry.
            load_favorite_assets_from_ui()
            if not FAVORITE_ASSETS:
                raise Exception("Failed to load favorite assets after all attempts. Retrying page load.")

            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] All initial page elements are loaded.")
            time.sleep(2)
            return  # Success, exit the infinite loop
        except Exception as e:
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load web driver or page elements: {e}. Retrying full page load in 30-60 seconds...")
            # If an error occurs, quit the current driver instance and get a new one for the next attempt
            try:
                driver.quit()
            except Exception as quit_e:
                print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Error quitting driver during failed load: {quit_e}")
            driver = get_driver()  # Get a fresh driver for the next retry
            time.sleep(random.uniform(30, 60))  # Longer random delay before next full retry


def load_favorite_assets_from_ui():
    """
    Identifies favorite tradable assets from the UI based on a specific selector.
    This will populate the FAVORITE_ASSETS global list. Retries until successful or max attempts.
    Returns True on success, False on persistent failure.
    """
    global FAVORITE_ASSETS
    max_total_retries = 15  # Allow even more retries for this critical step
    for attempt in range(max_total_retries):
        try:
            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Attempt {attempt + 1}/{max_total_retries} to load favorite assets.")

            # Close modals before attempting to click current symbol
            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Checking for overlays before clicking currency symbol...")
            for _ in range(5):
                if not close_active_modals_if_any(wait_time=5):
                    break
                time.sleep(1)

                # Click current symbol robustly (this is the problematic click)
            current_symbol_element = click_element_robustly(By.CLASS_NAME, 'current-symbol', wait_time=20,
                                                            retry_attempts=5)  # Increased wait/retry
            if not current_symbol_element:
                raise Exception("Failed to click current symbol to open asset list for loading assets.")
            time.sleep(4)  # Increased sleep after click for dropdown to fully appear

            # Find asset elements
            asset_elements = driver.find_elements(By.CSS_SELECTOR, FAVORITE_ASSETS_UI_SELECTOR)

            if not asset_elements:
                asset_elements = driver.find_elements(By.XPATH, FALLBACK_SPECIFIC_SELECTOR)
                print(
                    f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Falling back to specific payout asset selector.")

            if asset_elements:
                FAVORITE_ASSETS = []
                for element in asset_elements:
                    asset_id = element.get_attribute('data-id')
                    if asset_id:
                        FAVORITE_ASSETS.append(asset_id)
                    else:
                        text_content = element.text.strip()
                        if text_content and '/' in text_content:
                            FAVORITE_ASSETS.append(text_content.split('\n')[0].strip())

                FAVORITE_ASSETS = list(set(FAVORITE_ASSETS))
                FAVORITE_ASSETS = [asset for asset in FAVORITE_ASSETS if asset]

                if not FAVORITE_ASSETS:
                    print(
                        f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No valid assets found after parsing UI elements in this attempt.")
                else:
                    print(
                        f"[{Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Loaded {len(FAVORITE_ASSETS)} favorite assets from UI: {FAVORITE_ASSETS}")
                    # Close the dropdown after successful load
                    click_element_robustly(By.CLASS_NAME, 'current-symbol', wait_time=5,
                                           retry_attempts=3)  # Click again to close
                    time.sleep(1)
                    return True  # Success
            else:
                print(
                    f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No asset elements found using any specified selector in this attempt. Retrying loading assets.")

            # If assets not loaded, close dropdown before next retry
            try:
                current_symbol_element_check = driver.find_element(By.CLASS_NAME, 'current-symbol')
                if "ddm_open" in current_symbol_element_check.get_attribute("class"):
                    click_element_robustly(By.CLASS_NAME, 'current-symbol', wait_time=5, retry_attempts=3)
                    time.sleep(0.5)
            except:
                pass

        except Exception as e:  # Catch any exception, including ElementClickInterceptedException from click_element_robustly
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading favorite assets from UI in attempt {attempt + 1}: {e}. Retrying in 5-10 seconds...")
            time.sleep(random.uniform(5, 10))
            # No need to call close_active_modals_if_any here, click_element_robustly already handles it.

    print(
        f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load favorite assets after {max_total_retries} attempts. This might require a full page reload.")
    return False  # Failed after all retries


# ... (rest of the code remains the same as the previous version) ...

def get_next_favorite_asset():
    """Rotates through the list of favorite assets."""
    global CURRENT_FAVORITE_ASSET_INDEX, FAVORITE_ASSETS, CURRENCY

    if not FAVORITE_ASSETS:
        print(
            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No favorite assets loaded. Cannot switch.")
        return None

    next_asset = FAVORITE_ASSETS[CURRENT_FAVORITE_ASSET_INDEX]
    CURRENT_FAVORITE_ASSET_INDEX = (CURRENT_FAVORITE_ASSET_INDEX + 1) % len(FAVORITE_ASSETS)

    if next_asset == CURRENCY:
        if len(FAVORITE_ASSETS) > 1:
            next_asset = FAVORITE_ASSETS[CURRENT_FAVORITE_ASSET_INDEX]
            CURRENT_FAVORITE_ASSET_INDEX = (CURRENT_FAVORITE_ASSET_INDEX + 1) % len(FAVORITE_ASSETS)
        else:
            return None

    return next_asset


def switch_to_asset_ui(asset_name):
    """
    Switches the trading asset in the UI by clicking elements. Retries until successful.
    """
    global CURRENCY, CURRENCY_CHANGE, CURRENCY_CHANGE_DATE, HISTORY_TAKEN, STACK, MODEL, INIT_DEPOSIT
    if asset_name == CURRENCY:
        print(
            f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Already on asset {asset_name}. No switch needed.")
        return True

    print(
        f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Attempting to switch to asset: {asset_name}")
    max_retries = 5  # Retries for switching asset
    for attempt in range(max_retries):
        try:
            # Ensure no modal is intercepting before clicking
            close_active_modals_if_any()
            time.sleep(0.5)

            # Click current symbol robustly
            current_symbol_element = click_element_robustly(By.CLASS_NAME, 'current-symbol', wait_time=10)
            if not current_symbol_element:
                raise Exception("Failed to click current symbol to open asset list for switch.")
            time.sleep(random.uniform(0.5, 1.0))

            asset_to_click = None
            try:
                asset_to_click = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, f'.assets-list__item[data-id="{asset_name}"]'))
                )
            except TimeoutException:
                asset_to_click = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, f"//li[contains(@class, 'assets-list__item') and contains(., '{asset_name}')]"))
                )

            if asset_to_click:
                asset_to_click.click()
                CURRENCY_CHANGE = True
                CURRENCY_CHANGE_DATE = datetime.now()
                STACK = []
                HISTORY_TAKEN = False

                print(
                    f"[{Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Switched UI to {asset_name}. Awaiting data for new asset.")
                return True  # Success
            else:
                print(
                    f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Failed to find asset '{asset_name}' in the UI dropdown. Retrying.")
                # Close dropdown before next retry
                click_element_robustly(By.CLASS_NAME, 'current-symbol', wait_time=5)  # Click again to close
                time.sleep(2)  # Wait before retry
        except ElementClickInterceptedException as e:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Click intercepted during asset switch: {e}. Retrying.")
            time.sleep(random.uniform(1, 3))
            close_active_modals_if_any(wait_time=5)
            time.sleep(random.uniform(1, 2))
        except Exception as e:
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error switching to asset {asset_name} via UI in attempt {attempt + 1}: {e}. Retrying.")
            time.sleep(random.uniform(2, 5))

    print(
        f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to switch to asset '{asset_name}' after {max_retries} attempts.")
    return False


def do_action(signal):
    """
    Executes a 'call' or 'put' trade action by clicking UI elements.
    Includes basic rate limiting and anti-reversal logic based on previous actions.
    Returns True if action was successfully initiated, False otherwise.
    """
    action_allowed = True
    if not STACK:
        print(
            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] STACK is empty. Cannot perform action.")
        return False

    last_value = STACK[-1]['close']

    global ACTIONS, IS_AMOUNT_SET
    for dat in list(ACTIONS.keys()):
        if dat < datetime.now() - timedelta(seconds=ACTIONS_SECONDS):
            del ACTIONS[dat]

    if len(ACTIONS) >= MAX_ACTIONS:
        print(
            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Max concurrent actions reached ({MAX_ACTIONS}). Cannot do a {signal} action.")
        action_allowed = False

    if action_allowed and ACTIONS:
        last_action_value = list(ACTIONS.values())[-1]
        if signal == 'call' and last_value <= last_action_value:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Prevented CALL: current_value ({last_value:.4f}) is not above previous action value ({last_action_value:.4f}).")
            action_allowed = False
        elif signal == 'put' and last_value >= last_action_value:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Prevented PUT: current_value ({last_value:.4f}) is not below previous action value ({last_action_value:.4f}).")
            action_allowed = False

    if action_allowed:
        try:
            # Ensure no modal is intercepting before clicking trade button
            close_active_modals_if_any()
            time.sleep(0.5)

            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Executing {signal.upper()}, currency: {CURRENCY}, last_value: {last_value:.4f}")
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, f'btn-{signal}'))
            ).click()
            ACTIONS[datetime.now()] = last_value
            IS_AMOUNT_SET = False
            return True
        except TimeoutException:
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Timeout waiting for {signal} button to be clickable.")
            return False
        except ElementClickInterceptedException as e:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Click intercepted during trade execution: {e}. Trade not placed.")
            return False
        except Exception as e:
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error executing action '{signal}': {e}")
            return False
    return False


def hand_delay():
    """Introduces a small random delay to mimic human interaction."""
    time.sleep(random.choice([0.2, 0.3, 0.4, 0.5, 0.6]))


def get_amounts(current_balance):
    """Calculates a martingale sequence based on current balance and coefficient and MAX_CONSECUTIVE_LOSSES."""
    global AMOUNTS, MARTINGALE_COEFFICIENT, MAX_CONSECUTIVE_LOSSES

    if current_balance <= 0:
        print(
            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Current balance is 0 or less. Cannot calculate martingale amounts dynamically. Using default: [1, 2, 4].")
        return [1, 2, 4]

    base_amount = AMOUNTS[0] if AMOUNTS and AMOUNTS[0] > 0 else 1.0
    amounts = [base_amount]

    for i in range(1, MAX_CONSECUTIVE_LOSSES + 1):
        next_amount = amounts[-1] * MARTINGALE_COEFFICIENT
        amounts.append(round(next_amount))
        if amounts[-1] > current_balance * 0.20:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Martingale amount {amounts[-1]:.2f} exceeds 20% of balance. Capping martingale stack.")
            amounts.pop()
            break

    amounts = sorted(list(set(amounts)))
    if not amounts or amounts[0] < 1: amounts = [1]

    print(
        f"[{Fore.MAGENTA}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Martingale stack calculated: {amounts}, Init deposit: {INIT_DEPOSIT:.2f}")
    return amounts


def get_deposit_value_selenium():
    """Retrieves the current balance from the UI."""
    try:
        # Before clicking the header balance, ensure no modal is intercepting
        close_active_modals_if_any()
        time.sleep(0.5)

        header_balance_element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.right-block__item.js-drop-down-modal-open'))
        )
        header_balance_element.click()
        print(
            f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Clicked header balance to open modal.")
        time.sleep(1)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'ddm_balance')),
            message="Balance modal (ddm_balance) not found."
        )
        print(f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Balance modal is visible.")

        demo_balance_selector = 'div.balance-item--demo .js-balance-demo'
        deposit_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, demo_balance_selector)),
            message=f"QT Demo balance element with selector '{demo_balance_selector}' not found inside modal."
        )
        balance_text = deposit_element.get_attribute("data-hd-show") or deposit_element.text
        balance_value = float(balance_text.replace(',', '').replace(' ', ''))

        ActionChains(driver).move_to_element_with_offset(driver.find_element(By.TAG_NAME, 'body'), 0,
                                                         0).click().perform()
        print(
            f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Clicked outside modal to close it.")
        time.sleep(1)

        return balance_value

    except TimeoutException as te:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Timeout during balance retrieval: {te}. Element not found or clickable. Returning 0.0")
        return 0.0
    except Exception as e:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error getting deposit value from UI (modal method): {e}. Returning 0.0")
        return 0.0


def set_trade_amount_selenium(amount_to_set):
    """Sets the trade amount in the UI using the virtual keyboard."""
    try:
        # Before interacting with amount input, close any active modals
        close_active_modals_if_any()
        time.sleep(0.5)

        amount_input = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR,
                                        '#put-call-buttons-chart-1 > div > div.blocks-wrap > div.block.block--bet-amount > div.block__control.control > div.control__value.value.value--several-items > div > input[type=text]'))
        )
        amount_input.click()
        hand_delay()

        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#modal-root div.virtual-keyboard')),
            message="Virtual keyboard did not appear."
        )

        base_keyboard_selector = '#modal-root > div > div > div > div > div.trading-panel-modal__in > div.virtual-keyboard > div > div:nth-child(%s) > div'

        try:
            zero_key_element = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, base_keyboard_selector % NUMBERS['0'])))
            zero_key_element.click()
            hand_delay()
        except Exception as e:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Could not click '0' key on virtual keyboard: {e}. Proceeding to type.")
            pass

        for number_char in str(int(amount_to_set)):
            key_selector = base_keyboard_selector % NUMBERS[number_char]
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, key_selector))
            ).click()
            hand_delay()

        time.sleep(1)
        print(
            f"[{Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Trade amount set to {amount_to_set}.")
        return True
    except TimeoutException as te:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Timeout waiting for amount input or virtual keyboard key: {te}.")
        return False
    except Exception as e:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error setting trade amount via UI: {e}")
        return False


def check_trade_results_and_set_amount(current_stack):
    """
    Checks recent closed trades on the UI and adjusts the next trade amount
    based on win/loss using the martingale strategy. Also updates daily profit/loss.
    """
    global IS_AMOUNT_SET, AMOUNTS, INIT_DEPOSIT, PERIOD, TOTAL_WINS_TODAY, TOTAL_LOSSES_TODAY, TRADING_ALLOWED, ASSET_PERFORMANCE, TOTAL_PROFIT_TODAY, TOTAL_LOSS_TODAY

    if not IS_AMOUNT_SET:
        if ACTIONS and list(ACTIONS.keys())[-1] + timedelta(seconds=(PERIOD + 5)) > datetime.now():
            return

        try:
            # Before interacting with tabs, close any active modals
            close_active_modals_if_any()
            time.sleep(0.5)

            closed_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR,
                                            '#bar-chart > div > div > div.right-widget-container > div > div.widget-slot__header > div.divider > ul > li:nth-child(2) > a'))
            )
            closed_tab_parent = closed_tab.find_element(by=By.XPATH, value='..')

            if 'active' not in closed_tab_parent.get_attribute('class'):
                closed_tab.click()
                time.sleep(1)

            closed_trades_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'deals-list__item'))
            )

            if closed_trades_elements:
                last_trade_element = closed_trades_elements[0]

                trade_asset = "Unknown"
                investment_amount = 0.0
                payout_amount = 0.0
                is_win, is_draw, is_loss = False, False, False

                try:
                    trade_info_lines = last_trade_element.text.split('\n')
                    if len(trade_info_lines) >= 5:
                        trade_asset = trade_info_lines[1].strip()
                        investment_str = trade_info_lines[3].replace('$', '').replace(' ', '').replace(',', '').strip()
                        payout_str = trade_info_lines[4].replace('$', '').replace(' ', '').replace(',', '').strip()

                        investment_amount = float(investment_str)
                        payout_amount = float(payout_str)

                        if payout_amount > investment_amount:
                            is_win = True
                        elif payout_amount == investment_amount and investment_amount > 0:
                            is_draw = True
                        else:
                            is_loss = True
                    else:
                        print(
                            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Trade info lines (last_split) not as expected length. Content: {trade_info_lines}. Skipping.")

                except (ValueError, IndexError) as e:
                    print(
                        f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Could not parse trade result amounts or asset: {e}. Skipping update for this trade.")

                # --- Update Global Stats ---
                if is_win:
                    TOTAL_WINS_TODAY += 1
                    profit_from_trade = payout_amount - investment_amount
                    TOTAL_PROFIT_TODAY += profit_from_trade
                    print(
                        f"[{Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Last trade WON! Profit: {profit_from_trade:.2f}. Total wins today: {TOTAL_WINS_TODAY}. Daily Profit: {TOTAL_PROFIT_TODAY:.2f}")
                    if trade_asset not in ASSET_PERFORMANCE: ASSET_PERFORMANCE[trade_asset] = {'wins': 0, 'losses': 0,
                                                                                               'payout': 0.0}
                    ASSET_PERFORMANCE[trade_asset]['wins'] += 1
                    ASSET_PERFORMANCE[trade_asset]['payout'] += profit_from_trade
                elif is_draw:
                    print(
                        f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Last trade was a DRAW. Daily Profit: {TOTAL_PROFIT_TODAY:.2f}")
                    if trade_asset not in ASSET_PERFORMANCE: ASSET_PERFORMANCE[trade_asset] = {'wins': 0, 'losses': 0,
                                                                                               'payout': 0.0}
                elif is_loss:
                    TOTAL_LOSSES_TODAY += 1
                    loss_from_trade = investment_amount
                    TOTAL_LOSS_TODAY += loss_from_trade
                    print(
                        f"[{Fore.RED}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Last trade LOST! Loss: {loss_from_trade:.2f}. Total losses today: {TOTAL_LOSSES_TODAY}. Daily Loss: {TOTAL_LOSS_TODAY:.2f}")
                    if trade_asset not in ASSET_PERFORMANCE: ASSET_PERFORMANCE[trade_asset] = {'wins': 0, 'losses': 0,
                                                                                               'payout': 0.0}
                    ASSET_PERFORMANCE[trade_asset]['losses'] += 1
                    ASSET_PERFORMANCE[trade_asset]['payout'] -= loss_from_trade
                else:
                    print(
                        f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Unknown trade result for last trade. Cannot update stats.")

                # --- Determine Next Martingale Amount ---
                current_trade_amount_value = int(WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR,
                                                    '#put-call-buttons-chart-1 > div > div.blocks-wrap > div.block.block--bet-amount > div.block__control.control > div.control__value.value.value--several-items > div > input[type=text]'))
                ).get_attribute('value').replace(',', '').replace(' ', ''))

                next_amount_to_set = AMOUNTS[0]

                if is_win or is_draw:
                    next_amount_to_set = AMOUNTS[0]
                elif is_loss:
                    if current_trade_amount_value in AMOUNTS:
                        current_index = AMOUNTS.index(current_trade_amount_value)
                        if current_index + 1 < len(AMOUNTS):
                            next_amount_to_set = AMOUNTS[current_index + 1]
                        else:
                            print(
                                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Martingale stack exhausted. Resetting to base amount.")
                            next_amount_to_set = AMOUNTS[0]
                    else:
                        print(
                            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Current trade amount ({current_trade_amount_value}) not in martingale stack. Resetting to base amount.")
                        next_amount_to_set = AMOUNTS[0]

                success = set_trade_amount_selenium(next_amount_to_set)
                if success:
                    chart_tab = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR,
                                                    '#bar-chart > div > div > div.right-widget-container > div > div.widget-slot__header > div.divider > ul > li:nth-child(1) > a'))
                    )
                    if 'active' not in chart_tab.find_element(by=By.XPATH, value='..').get_attribute('class'):
                        chart_tab.click()
                        time.sleep(1)
                    print(
                        f"[{Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Next trade amount set to: {next_amount_to_set}")
                else:
                    print(
                        f"[{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to set next trade amount.")

            else:
                print(
                    f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No closed trades found.")

            IS_AMOUNT_SET = True

            if TOTAL_PROFIT_TODAY >= DAILY_PROFIT_LIMIT_AMOUNT and DAILY_PROFIT_LIMIT_AMOUNT > 0:
                print(
                    f"[{Fore.GREEN}{Style.BRIGHT}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Daily profit target of ${DAILY_PROFIT_LIMIT_AMOUNT:.2f} reached. Stopping trading.")
                TRADING_ALLOWED = False
            elif TOTAL_LOSS_TODAY >= DAILY_LOSS_LIMIT_AMOUNT and DAILY_LOSS_LIMIT_AMOUNT > 0:
                print(
                    f"[{Fore.RED}{Style.BRIGHT}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Daily loss limit of ${DAILY_LOSS_LIMIT_AMOUNT:.2f} reached. Stopping trading.")
                TRADING_ALLOWED = False

        except (NoSuchElementException, StaleElementReferenceException, TimeoutException) as e:
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] UI element not found/stale/timeout during trade result check: {e}")
        except Exception as e:
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error during trade result check: {e}")
    else:
        pass


def websocket_log(stack):
    """
    Intercepts WebSocket messages from the browser's performance log.
    Extracts real-time price data and historical data to update the STACK.
    """
    global CURRENCY, CURRENCY_CHANGE, CURRENCY_CHANGE_DATE, HISTORY_TAKEN, PERIOD, ACTIONS_SECONDS, MODEL, INIT_DEPOSIT

    global TODAY_START_DATE, TOTAL_WINS_TODAY, TOTAL_LOSSES_TODAY, TOTAL_PROFIT_TODAY, TOTAL_LOSS_TODAY
    if datetime.now().date() != TODAY_START_DATE:
        TODAY_START_DATE = datetime.now().date()
        TOTAL_WINS_TODAY = 0
        TOTAL_LOSSES_TODAY = 0
        TOTAL_PROFIT_TODAY = 0.0
        TOTAL_LOSS_TODAY = 0.0
        print(
            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] New day detected. Resetting daily profit/loss counters.")

    try:
        current_symbol_ui = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'current-symbol'))
        ).text

        if current_symbol_ui != CURRENCY:
            CURRENCY = current_symbol_ui
            CURRENCY_CHANGE = True
            CURRENCY_CHANGE_DATE = datetime.now()
            print(
                f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] UI Currency changed to: {CURRENCY}. Initiating reset.")
    except (NoSuchElementException, StaleElementReferenceException, TimeoutException):
        pass
    except Exception as e:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error checking UI currency: {e}")

    if CURRENCY_CHANGE and CURRENCY_CHANGE_DATE < datetime.now() - timedelta(seconds=5):
        print(
            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Performing full reset due to currency change.")
        stack = []
        HISTORY_TAKEN = False
        driver.refresh()
        time.sleep(7)
        CURRENCY_CHANGE = False
        MODEL = None
        INIT_DEPOSIT = None
        return stack

    try:
        logs = driver.get_log('performance')
        for wsData in logs:
            message = json.loads(wsData['message'])['message']
            response = message.get('params', {}).get('response', {})

            if response.get('opcode', 0) == 2:
                try:
                    payload_str = base64.b64decode(response['payloadData']).decode('utf-8')
                    data = json.loads(payload_str)

                    if not HISTORY_TAKEN and 'history' in data:
                        PERIOD = data.get('period', PERIOD)
                        ACTIONS_SECONDS = PERIOD * 60 - 1 if PERIOD > 0 else 10
                        stack = [{'timestamp': int(d[0]), 'open': float(d[1]), 'high': float(d[1]), 'low': float(d[1]),
                                  'close': float(d[1]), 'volume': 0} for d in data['history']]
                        HISTORY_TAKEN = True
                        print(
                            f"[{Fore.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] History taken for asset: {data.get('asset', 'Unknown')}, period: {data.get('period', 'Unknown')}, len_history: {len(data['history'])}, len_stack: {len(stack)}")
                        continue

                    if isinstance(data, list) and len(data) == 3:
                        symbol, timestamp_raw, value_raw = data[0], data[1], data[2]
                        timestamp = int(timestamp_raw)
                        value = float(value_raw)

                        current_symbol_clean = CURRENCY.replace('/', '').replace(' ', '') if CURRENCY else ''
                        websocket_symbol_clean = symbol.replace('_', '').upper()

                        if current_symbol_clean != websocket_symbol_clean and companies.get(
                                current_symbol_clean) != websocket_symbol_clean:
                            continue

                        new_data_point = {'timestamp': timestamp, 'close': value, 'open': value, 'high': value,
                                          'low': value, 'volume': 0}

                        if stack and stack[-1]['timestamp'] == timestamp:
                            stack[-1]['close'] = value
                            if value > stack[-1]['high']: stack[-1]['high'] = value
                            if value < stack[-1]['low']: stack[-1]['low'] = value
                        else:
                            if len(stack) >= LENGTH_STACK_MAX:
                                stack.pop(0)
                            stack.append(new_data_point)

                        if len(stack) > LENGTH_STACK_MAX + 5:
                            print(
                                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Warning: STACK length ({len(stack)}) exceeded MAX_LENGTH_MAX + buffer. Resetting stack to prevent issues.")
                            stack = []
                            HISTORY_TAKEN = False

                except (json.JSONDecodeError, ValueError) as e:
                    continue
                except Exception as e:
                    continue
    except Exception as e:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error accessing performance logs (outer): {e}")

    if len(stack) >= LENGTH_STACK_MIN:
        check_trade_results_and_set_amount(stack)
        if IS_AMOUNT_SET:
            run_trading_strategy(stack)

    return stack


class TradingStrategies:
    """
    Contains implementations for various trading strategies.
    Each method returns a signal ('call', 'put') or None.
    """

    @staticmethod
    def _prepare_dataframe(candles_data):
        """Helper to convert list of dicts to DataFrame and ensure numeric types."""
        df = pd.DataFrame(candles_data)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = df['close']
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.fillna(0)
        return df

    # Strategy 1: Moving Averages Cross (SMA 5/10)
    @staticmethod
    def sma_cross(candles_data, fast_length=5, slow_length=10):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < slow_length: return None

        fast_ma = ta.sma(df['close'], length=fast_length)
        slow_ma = ta.sma(df['close'], length=slow_length)

        if fast_ma.empty or slow_ma.empty or len(fast_ma) < 2 or len(slow_ma) < 2: return None

        if fast_ma.iloc[-2] < slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
            return 'call'
        elif fast_ma.iloc[-2] > slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
            return 'put'
        return None

    # Strategy 2: Price Bounce from Support/Resistance
    @staticmethod
    def support_resistance_bounce(candles_data, lookback_period=10, bounce_percent_threshold=0.005):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < lookback_period: return None

        recent_period = df.iloc[-lookback_period:]
        resistance = recent_period['high'].max()
        support = recent_period['low'].min()
        current_close = df['close'].iloc[-1]

        if abs(current_close - resistance) / resistance < bounce_percent_threshold and df['close'].iloc[
            -2] > current_close:
            return 'put'
        elif abs(current_close - support) / support < bounce_percent_threshold and df['close'].iloc[-2] < current_close:
            return 'call'
        return None

    # Strategy 3: Bollinger Bands Squeeze Breakout
    @staticmethod
    def bollinger_squeeze_breakout(candles_data, length=20, std=2, bandwidth_threshold=0.01):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < length: return None

        bbands = ta.bbands(df['close'], length=length, std=std)
        if bbands.empty or len(bbands) < 2: return None

        middle_band = bbands[f'BBM_{length}_{std}.0'].iloc[-1]
        upper_band = bbands[f'BBU_{length}_{std}.0'].iloc[-1]
        lower_band = bbands[f'BBL_{length}_{std}.0'].iloc[-1]

        if middle_band == 0: return None

        current_bandwidth = (upper_band - lower_band) / middle_band
        prev_bandwidth = (bbands[f'BBU_{length}_{std}.0'].iloc[-2] - bbands[f'BBL_{length}_{std}.0'].iloc[-2]) / \
                         bbands[f'BBM_{length}_{std}.0'].iloc[-2]

        current_close = df['close'].iloc[-1]

        if current_bandwidth < bandwidth_threshold and current_bandwidth < prev_bandwidth:
            if current_close > upper_band:
                return 'call'
            elif current_close < lower_band:
                return 'put'
        return None

    # Strategy 4: RSI with Fibonacci Levels (simulated using fixed levels)
    @staticmethod
    def rsi_fibonacci(candles_data, rsi_length=14, overbought_level=70, oversold_level=30):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < rsi_length: return None

        rsi = ta.rsi(df['close'], length=rsi_length)
        if rsi.empty: return None

        current_rsi = rsi.iloc[-1]

        if current_rsi > overbought_level and rsi.iloc[-2] <= overbought_level:
            return 'put'
        elif current_rsi < oversold_level and rsi.iloc[-2] >= oversold_level:
            return 'call'
        return None

    # Strategy 5: Strong Candle Pattern (Big Body, Small Shadows)
    @staticmethod
    def strong_candle_pattern(candles_data, min_body_percent=0.8, max_shadow_ratio=0.1):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < 1: return None

        current_candle = df.iloc[-1]

        body = abs(current_candle['close'] - current_candle['open'])
        candle_range = current_candle['high'] - current_candle['low']

        if candle_range == 0: return None

        body_ratio_to_range = body / candle_range

        if body_ratio_to_range < min_body_percent: return None

        if current_candle['close'] > current_candle['open']:
            upper_shadow = current_candle['high'] - current_candle['close']
            lower_shadow = current_candle['open'] - current_candle['low']
            if body > 0 and upper_shadow / body < max_shadow_ratio and lower_shadow / body < max_shadow_ratio:
                return 'call'
        else:
            upper_shadow = current_candle['high'] - current_candle['open']
            lower_shadow = current_candle['close'] - current_candle['low']
            if body > 0 and upper_shadow / body < max_shadow_ratio and lower_shadow / body < max_shadow_ratio:
                return 'put'
        return None

    # Strategy 6: Instantaneous Momentum (Rate of Change)
    @staticmethod
    def instantaneous_momentum(candles_data, length=1):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < length + 1: return None

        roc = ta.roc(df['close'], length=length)
        if roc.empty: return None

        current_roc = roc.iloc[-1]

        if current_roc > 0.1:
            return 'call'
        elif current_roc < -0.1:
            return 'put'
        return None

    # Strategy 7: RSI and MACD Confluence
    @staticmethod
    def rsi_macd_confluence(candles_data, rsi_length=14, macd_fast=12, macd_slow=26, macd_signal=9):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < max(rsi_length, macd_slow + macd_signal + 1): return None

        rsi = ta.rsi(df['close'], length=rsi_length)
        macd_data = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)

        if rsi.empty or macd_data.empty or len(rsi) < 2 or len(macd_data) < 2: return None

        current_rsi = rsi.iloc[-1]
        current_macd = macd_data[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'].iloc[-1]
        current_macd_signal = macd_data[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'].iloc[-1]
        prev_macd = macd_data[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'].iloc[-2]
        prev_macd_signal = macd_data[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'].iloc[-2]

        if current_rsi < 40 and prev_macd < prev_macd_signal and current_macd > current_macd_signal:
            return 'call'
        elif current_rsi > 60 and prev_macd > prev_macd_signal and current_macd < current_macd_signal:
            return 'put'
        return None

    # Strategy 8: Support/Resistance Breakout with Volume Spike
    @staticmethod
    def sr_volume_breakout(candles_data, lookback_period=20, volume_multiplier=2.0):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < lookback_period + 1: return None

        recent_df = df.iloc[-lookback_period - 1:]

        historical_volumes = df['volume'].iloc[-(lookback_period + 1):-1]
        if historical_volumes.empty: return None
        avg_volume = historical_volumes.mean()

        current_close = recent_df['close'].iloc[-1]
        current_volume = recent_df['volume'].iloc[-1]

        resistance = df['high'].iloc[-lookback_period:-1].max()
        support = df['low'].iloc[-lookback_period:-1].min()

        if current_volume > avg_volume * volume_multiplier:
            if current_close > resistance and df['close'].iloc[-2] <= resistance:
                return 'call'
            elif current_close < support and df['close'].iloc[-2] >= support:
                return 'put'
        return None

    # Strategy 9: Engulfing Pattern (Bullish/Bearish)
    @staticmethod
    def engulfing_pattern(candles_data):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < 2: return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        if prev['close'] < prev['open'] and current['close'] > current['open'] and \
                current['open'] <= prev['close'] and current['close'] >= prev['open']:
            return 'call'

        elif prev['close'] > prev['open'] and current['close'] < current['open'] and \
                current['open'] >= prev['close'] and current['close'] <= prev['open']:
            return 'put'
        return None

    # Strategy 10: Strong Reversal Pattern (Simplified: Hammer/Hanging Man type)
    @staticmethod
    def strong_reversal_pattern(candles_data, hammer_ratio=2.0):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < 2: return None

        candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        body = abs(candle['close'] - candle['open'])
        if body == 0: return None

        if candle['close'] > candle['open']:
            lower_shadow = candle['open'] - candle['low']
            upper_shadow = candle['high'] - candle['close']
            if body > 0 and upper_shadow / body < max_shadow_ratio and lower_shadow / body < max_shadow_ratio:
                return 'call'
        else:
            lower_shadow = candle['close'] - candle['low']
            upper_shadow = candle['high'] - candle['open']
            if body > 0 and upper_shadow / body < max_shadow_ratio and lower_shadow / body < max_shadow_ratio:
                return 'put'
        return None

    # Strategy 11: Stochastic and RSI Reversals
    @staticmethod
    def stochastic_rsi_reversal(candles_data, stoch_k=14, stoch_d=3, rsi_length=14, overbought_stoch=80,
                                oversold_stoch=20, overbought_rsi=70, oversold_rsi=30):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < max(stoch_k, rsi_length, stoch_d + 1): return None

        stoch = ta.stoch(df['high'], df['low'], df['close'], k=stoch_k, d=stoch_d)
        rsi = ta.rsi(df['close'], length=rsi_length)

        if stoch.empty or rsi.empty or len(stoch) < 2 or len(rsi) < 2: return None

        current_k = stoch[f'STOCHk_{stoch_k}_{stoch_d}'].iloc[-1]
        current_d = stoch[f'STOCHd_{stoch_k}_{stoch_d}'].iloc[-1]
        prev_k = stoch[f'STOCHk_{stoch_k}_{stoch_d}'].iloc[-2]
        prev_d = stoch[f'STOCHd_{stoch_k}_{stoch_d}'].iloc[-2]
        current_rsi = rsi.iloc[-1]

        if current_k > overbought_stoch and current_d > overbought_stoch and \
                current_k < prev_k and current_d < prev_d and current_rsi > overbought_rsi:
            return 'put'
        elif current_k < oversold_stoch and current_d < oversold_stoch and \
                current_k > prev_k and current_d > prev_d and current_rsi < oversold_rsi:
            return 'call'
        return None

    # Strategy 12: ADX Strong Trend Confirmation
    @staticmethod
    def adx_trend_confirmation(candles_data, adx_length=14, adx_threshold=25):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < adx_length * 2: return None

        adx_data = ta.adx(df['high'], df['low'], df['close'], length=adx_length)

        if adx_data.empty or len(adx_data) < 1: return None

        current_adx = adx_data[f'ADX_{adx_length}'].iloc[-1]
        current_plus_di = adx_data[f'DMP_{adx_length}'].iloc[-1]
        current_minus_di = adx_data[f'DMN_{adx_length}'].iloc[-1]

        if current_adx > adx_threshold:
            if current_plus_di > current_minus_di:
                return 'call'
            elif current_minus_di > current_plus_di:
                return 'put'
        return None

    # Strategy 13: Fibonacci Retracement Bounce
    @staticmethod
    def fibonacci_retracement_bounce(candles_data, lookback_period=30, retracement_levels=[0.382, 0.5, 0.618],
                                     tolerance=0.005):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < lookback_period: return None

        swing_df = df.iloc[-lookback_period:]
        swing_high = swing_df['high'].max()
        swing_low = swing_df['low'].min()

        if swing_high == swing_low: return None

        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]

        is_uptrend_in_swing = swing_df['close'].iloc[0] < swing_df['close'].iloc[-1]

        for level in retracement_levels:
            if is_uptrend_in_swing:
                retrace_price = swing_high - (swing_high - swing_low) * level
                if abs(current_close - retrace_price) / retrace_price < tolerance and current_close > prev_close:
                    return 'call'
            else:
                retrace_price = swing_low + (swing_high - swing_low) * level
                if abs(current_close - retrace_price) / retrace_price < tolerance and current_close < prev_close:
                    return 'put'
        return None

    # Strategy 14: Price Action Swing (Higher Highs/Lower Lows)
    @staticmethod
    def price_action_swing(candles_data, min_swing_percent=0.5):
        df = TradingStrategies._prepare_dataframe(candles_data)
        if len(df) < 5: return None

        if df['high'].iloc[-3] > df['high'].iloc[-4] and df['high'].iloc[-3] > df['high'].iloc[-2]:
            if (df['high'].iloc[-3] - df['low'].iloc[-3]) / df['low'].iloc[-3] * 100 > min_swing_percent:
                if df['close'].iloc[-1] < df['low'].iloc[-3]:
                    return 'put'

        elif df['low'].iloc[-3] < df['low'].iloc[-4] and df['low'].iloc[-3] < df['low'].iloc[-2]:
            if (df['high'].iloc[-3] - df['low'].iloc[-3]) / df['low'].iloc[-3] * 100 > min_swing_percent:
                if df['close'].iloc[-1] > df['high'].iloc[-3]:
                    return 'call'
        return None


def run_trading_strategy(candles_data):
    """
    Applies all defined trading strategies based on the collected candle data.
    Decides whether to issue a 'call' or 'put' signal based on confluence.
    Returns True if an action was attempted, False otherwise.
    """
    global REQUIRED_CONFLUENCE

    if not candles_data or len(candles_data) < 40:
        print(
            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Not enough candle data ({len(candles_data)}) to run complex strategies. Need at least 40.")
        return False

    signals_count = {'call': 0, 'put': 0}
    active_strategies_signals = []

    strategies_to_check = [
        ("SMA_Cross", TradingStrategies.sma_cross, {'fast_length': 5, 'slow_length': 10}),
        ("SR_Bounce", TradingStrategies.support_resistance_bounce,
         {'lookback_period': 15, 'bounce_percent_threshold': 0.003}),
        ("BB_Squeeze_Breakout", TradingStrategies.bollinger_squeeze_breakout,
         {'length': 20, 'std': 2, 'bandwidth_threshold': 0.01}),
        ("RSI_Fibo", TradingStrategies.rsi_fibonacci, {'rsi_length': 14, 'overbought_level': 65, 'oversold_level': 35}),
        ("Strong_Candle", TradingStrategies.strong_candle_pattern, {'min_body_percent': 0.6, 'max_shadow_ratio': 0.2}),
        ("Momentum", TradingStrategies.instantaneous_momentum, {'length': 1}),
        ("RSI_MACD_Confluence", TradingStrategies.rsi_macd_confluence,
         {'rsi_length': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9}),
        ("SR_Volume_Breakout", TradingStrategies.sr_volume_breakout, {'lookback_period': 20, 'volume_multiplier': 1.5}),
        ("Engulfing_Pattern", TradingStrategies.engulfing_pattern, {}),
        ("Strong_Reversal", TradingStrategies.strong_reversal_pattern, {'hammer_ratio': 2.0}),
        ("Stoch_RSI_Reversal", TradingStrategies.stochastic_rsi_reversal,
         {'stoch_k': 14, 'stoch_d': 3, 'rsi_length': 14, 'overbought_stoch': 75, 'oversold_stoch': 25,
          'overbought_rsi': 65, 'oversold_rsi': 35}),
        ("ADX_Trend", TradingStrategies.adx_trend_confirmation, {'adx_length': 14, 'adx_threshold': 20}),
        ("Fibo_Retracement", TradingStrategies.fibonacci_retracement_bounce,
         {'lookback_period': 40, 'retracement_levels': [0.382, 0.5, 0.618]}),
        ("Price_Action_Swing", TradingStrategies.price_action_swing, {'min_swing_percent': 0.5})
    ]

    for name, strategy_func, params in strategies_to_check:
        try:
            signal = strategy_func(candles_data, **params)
            if signal:
                signals_count[signal] += 1
                active_strategies_signals.append(f"{name}_{signal.upper()}")
        except Exception as e:
            print(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in strategy '{name}': {e}")

    final_signal = None
    if signals_count['call'] >= REQUIRED_CONFLUENCE:
        final_signal = 'call'
    elif signals_count['put'] >= REQUIRED_CONFLUENCE:
        final_signal = 'put'

    if final_signal:
        print(
            f"{Fore.MAGENTA}{Style.BRIGHT}[STRATEGY]{Style.RESET_ALL} Final Confluence Signal: {final_signal.upper()} (Votes: {signals_count[final_signal]}, Strategies: {', '.join(active_strategies_signals)})")
        return do_action(final_signal)
    else:
        print(
            f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No strong confluence signal. Call votes: {signals_count['call']}, Put votes: {signals_count['put']}. Active Strategy Signals: {', '.join(active_strategies_signals) if active_strategies_signals else 'None'}")
        return False

    # --- Interactive CLI Menu Functions ---


async def display_main_menu():
    """Displays the main interactive menu for the bot, with colors and banner."""
    print_naif_bot_banner()
    print(f"{Fore.BLUE}{Style.BRIGHT}" + "=" * 40 + Style.RESET_ALL)
    print(f"{Fore.CYAN}{Style.BRIGHT}        NAIF BOT - Main Menu{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}" + "=" * 40 + Style.RESET_ALL)
    print(f"{Fore.YELLOW}1.{Style.RESET_ALL} View Current Status (Balance, Currency, Stack, Daily Stats, Assets)")
    print(f"{Fore.YELLOW}2.{Style.RESET_ALL} Change Trading Currency (Manual Select)")
    print(f"{Fore.YELLOW}3.{Style.RESET_ALL} Auto-Switch to Next Favorite Asset")
    print(f"{Fore.YELLOW}4.{Style.RESET_ALL} Manual Trade (Call/Put)")
    print(f"{Fore.YELLOW}5.{Style.RESET_ALL} Adjust Bot Settings (Trade Amount, Martingale, Daily Limits, Confluence)")
    print(f"{Fore.YELLOW}6.{Style.RESET_ALL} Restart Browser/Bot")
    print(f"{Fore.YELLOW}7.{Style.RESET_ALL} Exit Bot")
    print(f"{Fore.BLUE}{Style.BRIGHT}" + "=" * 40 + Style.RESET_ALL)


async def manual_trade_action():
    """Allows user to manually execute a trade (Call/Put)."""
    global IS_AMOUNT_SET, AMOUNTS
    print(f"\n{Fore.MAGENTA}--- Manual Trade ---{Style.RESET_ALL}")
    trade_type = await asyncio.to_thread(input,
                                         f"{Fore.CYAN}Enter trade type ('call' or 'put'): {Style.RESET_ALL}").lower()
    if trade_type not in ['call', 'put']:
        print(f"{Fore.RED}Invalid trade type. Please enter 'call' or 'put'.{Style.RESET_ALL}")
        return

    trade_amount = AMOUNTS[0] if AMOUNTS else 1.0
    print(f"Attempting manual trade of {trade_amount} for {trade_type.upper()}.")
    if not set_trade_amount_selenium(trade_amount):
        print(f"{Fore.RED}Failed to set trade amount for manual trade.{Style.RESET_ALL}")
        return

    if do_action(trade_type):
        print(f"{Fore.GREEN}Manual {trade_type.upper()} trade initiated with amount {trade_amount}.{Style.RESET_ALL}")
        IS_AMOUNT_SET = False
    else:
        print(f"{Fore.RED}Failed to initiate manual {trade_type.upper()} trade.{Style.RESET_ALL}")


async def adjust_bot_settings():
    """Allows user to adjust various bot settings."""
    global AMOUNTS, MARTINGALE_COEFFICIENT, MAX_CONSECUTIVE_LOSSES, DAILY_PROFIT_LIMIT_AMOUNT, DAILY_LOSS_LIMIT_AMOUNT, REQUIRED_CONFLUENCE

    print(f"\n{Fore.MAGENTA}--- Adjust Bot Settings ---{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}1.{Style.RESET_ALL} Set Base Trade Amount (Current: {AMOUNTS[0]})")
    print(f"{Fore.YELLOW}2.{Style.RESET_ALL} Set Martingale Coefficient (Current: {MARTINGALE_COEFFICIENT})")
    print(
        f"{Fore.YELLOW}3.{Style.RESET_ALL} Set Max Consecutive Losses for Martingale (Current: {MAX_CONSECUTIVE_LOSSES})")
    print(f"{Fore.YELLOW}4.{Style.RESET_ALL} Set Daily Profit Limit Amount (Current: ${DAILY_PROFIT_LIMIT_AMOUNT:.2f})")
    print(f"{Fore.YELLOW}5.{Style.RESET_ALL} Set Daily Loss Limit Amount (Current: ${DAILY_LOSS_LIMIT_AMOUNT:.2f})")
    print(
        f"{Fore.YELLOW}6.{Style.RESET_ALL} Set Required Strategy Confluence (Current: {REQUIRED_CONFLUENCE} strategies)")
    print(f"{Fore.YELLOW}B.{Style.RESET_ALL} Back to Main Menu")

    setting_choice = await asyncio.to_thread(input(f"{Fore.CYAN}Enter setting to adjust: {Style.RESET_ALL}"))

    if setting_choice == '1':
        try:
            new_base_amount = float(
                await asyncio.to_thread(input(f"{Fore.CYAN}Enter new base trade amount: {Style.RESET_ALL}")))
            if new_base_amount > 0:
                AMOUNTS[0] = round(new_base_amount)
                print(f"{Fore.GREEN}Base trade amount updated to {AMOUNTS[0]}.{Style.RESET_ALL}")
                current_balance = get_deposit_value_selenium()
                if current_balance > 0:
                    global INIT_DEPOSIT
                    AMOUNTS[:] = get_amounts(current_balance)
            else:
                print(f"{Fore.RED}Invalid amount. Must be greater than 0.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
    elif setting_choice == '2':
        try:
            new_coeff = float(await asyncio.to_thread(
                input(f"{Fore.CYAN}Enter new Martingale Coefficient (e.g., 2.0): {Style.RESET_ALL}")))
            if new_coeff >= 1.0:
                MARTINGALE_COEFFICIENT = new_coeff
                print(f"{Fore.GREEN}Martingale Coefficient updated to {MARTINGALE_COEFFICIENT}.{Style.RESET_ALL}")
                current_balance = get_deposit_value_selenium()
                if current_balance > 0:
                    global INIT_DEPOSIT
                    AMOUNTS[:] = get_amounts(current_balance)
            else:
                print(f"{Fore.RED}Coefficient must be 1.0 or greater.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
    elif setting_choice == '3':
        try:
            new_max_losses = int(await asyncio.to_thread(
                input(f"{Fore.CYAN}Enter new Max Consecutive Losses (e.g., 5): {Style.RESET_ALL}")))
            if new_max_losses >= 0:
                MAX_CONSECUTIVE_LOSSES = new_max_losses
                print(f"{Fore.GREEN}Max Consecutive Losses updated to {MAX_CONSECUTIVE_LOSSES}.{Style.RESET_ALL}")
                current_balance = get_deposit_value_selenium()
                if current_balance > 0:
                    global INIT_DEPOSIT
                    AMOUNTS[:] = get_amounts(current_balance)
            else:
                print(f"{Fore.RED}Value must be 0 or greater.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter an integer.{Style.RESET_ALL}")
    elif setting_choice == '4':
        try:
            new_profit_limit = float(await asyncio.to_thread(
                input(f"{Fore.CYAN}Enter new Daily Profit Limit Amount (e.g., 50.0): {Style.RESET_ALL}")))
            if new_profit_limit >= 0:
                DAILY_PROFIT_LIMIT_AMOUNT = new_profit_limit
                print(
                    f"{Fore.GREEN}Daily Profit Limit Amount updated to ${DAILY_PROFIT_LIMIT_AMOUNT:.2f}.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Limit must be 0 or greater.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
    elif setting_choice == '5':
        try:
            new_loss_limit = float(await asyncio.to_thread(
                input(f"{Fore.CYAN}Enter new Daily Loss Limit Amount (e.g., 20.0): {Style.RESET_ALL}")))
            if new_loss_limit >= 0:
                DAILY_LOSS_LIMIT_AMOUNT = new_loss_limit
                print(
                    f"{Fore.GREEN}Daily Loss Limit Amount updated to ${DAILY_LOSS_LIMIT_AMOUNT:.2f}.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Limit must be 0 or greater.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
    elif setting_choice == '6':
        try:
            new_confluence = int(await asyncio.to_thread(
                input(f"{Fore.CYAN}Enter new Required Strategy Confluence (e.g., 3): {Style.RESET_ALL}")))
            if new_confluence >= 1:
                REQUIRED_CONFLUENCE = new_confluence
                print(f"{Fore.GREEN}Required Strategy Confluence updated to {REQUIRED_CONFLUENCE}.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Confluence must be 1 or greater.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter an integer.{Style.RESET_ALL}")
    elif setting_choice.upper() == 'B':
        print(f"{Fore.CYAN}Returning to main menu.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")

    await asyncio.sleep(1)


async def handle_user_input(choice):
    """Handles user choices from the interactive menu."""
    global TRADING_ALLOWED, driver, STACK, HISTORY_TAKEN, CURRENCY_CHANGE, CURRENCY, MODEL, INIT_DEPOSIT, AMOUNTS, IS_AMOUNT_SET, TOTAL_WINS_TODAY, TOTAL_LOSSES_TODAY, TODAY_START_DATE, ASSET_PERFORMANCE, MARTINGALE_COEFFICIENT, MAX_CONSECUTIVE_LOSSES, DAILY_PROFIT_LIMIT_AMOUNT, DAILY_LOSS_LIMIT_AMOUNT, TOTAL_PROFIT_TODAY, TOTAL_LOSS_TODAY, REQUIRED_CONFLUENCE, FAVORITE_ASSETS, CURRENT_FAVORITE_ASSET_INDEX

    if choice == '1':
        print(f"\n{Fore.BLUE}--- Current Bot Status ---{Style.RESET_ALL}")
        print(f"Trading Allowed: {TRADING_ALLOWED}")
        print(f"Current Currency: {CURRENCY if CURRENCY else 'Not set'}")
        print(f"Current Balance: {get_deposit_value_selenium():.2f}")
        print(f"Stack Size: {len(STACK)}")
        if STACK:
            print(f"Last Price in Stack: {STACK[-1]['close']:.4f}")
            print(f"Period (Candle Duration): {PERIOD} seconds")
        print(f"History Taken: {HISTORY_TAKEN}")
        print(f"Martingale Amounts: {AMOUNTS}")
        print(f"Daily Wins (Count): {TOTAL_WINS_TODAY} (Target: {DAILY_PROFIT_TARGET_COUNT})")
        print(f"Daily Losses (Count): {TOTAL_LOSSES_TODAY} (Limit: {DAILY_LOSS_LIMIT_COUNT})")
        print(f"Daily Profit (Amount): ${TOTAL_PROFIT_TODAY:.2f} (Target: ${DAILY_PROFIT_LIMIT_AMOUNT:.2f})")
        print(f"Daily Loss (Amount): ${TOTAL_LOSS_TODAY:.2f} (Limit: ${DAILY_LOSS_LIMIT_AMOUNT:.2f})")
        print(f"Martingale Coeff: {MARTINGALE_COEFFICIENT}, Max Cons. Losses: {MAX_CONSECUTIVE_LOSSES}")
        print(f"Required Strategy Confluence: {REQUIRED_CONFLUENCE}")

        print(f"\n{Fore.BLUE}--- Asset Performance ---{Style.RESET_ALL}")
        if ASSET_PERFORMANCE:
            for asset, stats in ASSET_PERFORMANCE.items():
                total_trades = stats['wins'] + stats['losses']
                win_rate = (stats['wins'] / total_trades * 100) if total_trades > 0 else 0
                print(
                    f"  {Fore.GREEN}{asset}{Style.RESET_ALL}: Wins={stats['wins']}, Losses={stats['losses']}, Net Payout={stats['payout']:.2f}, Win Rate={win_rate:.2f}%")
        else:
            print(f"{Fore.YELLOW}  No asset performance data yet.{Style.RESET_ALL}")
        print(f"\n{Fore.BLUE}--- Favorite Assets ---{Style.RESET_ALL}")
        if FAVORITE_ASSETS:
            print(f"  {FAVORITE_ASSETS}")
        else:
            print(f"{Fore.YELLOW}  No favorite assets loaded.{Style.RESET_ALL}")
        print(f"{Fore.BLUE}--------------------------{Style.RESET_ALL}")

    elif choice == '2':
        print(f"\n{Fore.MAGENTA}--- Changing Trading Currency (Manual Select) ---{Style.RESET_ALL}")
        try:
            close_active_modals_if_any()
            time.sleep(0.5)

            current_symbol_element = click_element_robustly(By.CLASS_NAME, 'current-symbol',
                                                            wait_time=15)  # Longer wait
            if not current_symbol_element:
                print(
                    f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to click current symbol to open asset list for manual selection.")
                return

            time.sleep(1)
            all_asset_elements = driver.find_elements(By.CSS_SELECTOR, '.assets-list__item[data-id]')
            if not all_asset_elements:
                all_asset_elements = driver.find_elements(By.XPATH, FALLBACK_SPECIFIC_SELECTOR)

            available_assets = [el.get_attribute('data-id') or el.text.strip().split('\n')[0].strip() for el in
                                all_asset_elements if el.get_attribute('data-id') or el.text.strip()]
            available_assets = list(set([a for a in available_assets if a]))

            print(f"{Fore.CYAN}Available Assets:{Style.RESET_ALL}")
            for i, asset_name in enumerate(available_assets):
                print(f"{Fore.YELLOW}{i + 1}.{Style.RESET_ALL} {asset_name}")

            click_element_robustly(By.CLASS_NAME, 'current-symbol', wait_time=5)  # Click again to close

            asset_index = await asyncio.to_thread(
                input(f"{Fore.CYAN}Enter number of asset to switch to (1-{len(available_assets)}): {Style.RESET_ALL}"))
            if asset_index.isdigit() and 1 <= int(asset_index) <= len(available_assets):
                selected_asset = available_assets[int(asset_index) - 1]
                switch_to_asset_ui(selected_asset)
            else:
                print(f"{Fore.RED}Invalid asset selection.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Error listing assets for manual selection: {e}")

    elif choice == '3':
        print(f"\n{Fore.MAGENTA}--- Auto-Switching to Next Favorite Asset ---{Style.RESET_ALL}")
        next_asset = get_next_favorite_asset()
        if next_asset:
            switch_to_asset_ui(next_asset)
        else:
            print(f"{Fore.YELLOW}No favorite assets to switch to.{Style.RESET_ALL}")

    elif choice == '4':
        await manual_trade_action()

    elif choice == '5':
        await adjust_bot_settings()

    elif choice == '6':
        print(f"\n{Fore.MAGENTA}--- Restarting Browser/Bot ---{Style.RESET_ALL}")
        TRADING_ALLOWED = False
        if driver:
            try:
                driver.quit()
                print(f"{Fore.GREEN}Browser closed.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Error closing browser: {e}")

        STACK = []
        ACTIONS = {}
        HISTORY_TAKEN = False
        CURRENCY = None
        CURRENCY_CHANGE = False
        CURRENCY_CHANGE_DATE = datetime.now()

        TOTAL_WINS_TODAY = 0
        TOTAL_LOSSES_TODAY = 0
        TOTAL_PROFIT_TODAY = 0.0
        TOTAL_LOSS_TODAY = 0.0
        TODAY_START_DATE = datetime.now().date()
        ASSET_PERFORMANCE = {}
        CURRENT_FAVORITE_ASSET_INDEX = 0
        FAVORITE_ASSETS = []

        driver = get_driver()
        load_web_driver()
        TRADING_ALLOWED = True
        print(f"{Fore.GREEN}Bot restarted.{Style.RESET_ALL}")

        INIT_DEPOSIT = None
        AMOUNTS = []
        IS_AMOUNT_SET = True

    elif choice == '7':
        print(f"\n{Fore.RED}Exiting Bot...{Style.RESET_ALL}")
        TRADING_ALLOWED = False
        if driver:
            driver.quit()
        return False
    else:
        print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")
    return True


# --- Main Bot Execution Loop ---
async def main():
    global STACK, TRADING_ALLOWED, INIT_DEPOSIT, AMOUNTS, IS_AMOUNT_SET

    TRADING_ALLOWED = True

    load_web_driver()

    INIT_DEPOSIT = get_deposit_value_selenium()
    if INIT_DEPOSIT is not None and INIT_DEPOSIT > 0:
        AMOUNTS = get_amounts(INIT_DEPOSIT)
    else:
        print(
            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Warning: Initial deposit is 0 or could not be read. Cannot calculate martingale amounts. Using default AMOUNTS.")
        AMOUNTS = [1, 2, 4, 8]

    websocket_task = asyncio.create_task(background_websocket_processor())

    if FAVORITE_ASSETS:
        first_asset = FAVORITE_ASSETS[0]
        print(
            f"[{Fore.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Auto-switching to initial favorite asset: {first_asset}")
        switch_to_asset_ui(first_asset)
    else:
        print(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No favorite assets available after initialization. Please ensure assets are loading correctly. Trading will be stopped.")
        TRADING_ALLOWED = False

    while TRADING_ALLOWED:
        await display_main_menu()
        user_choice = await asyncio.to_thread(input, f"{Fore.GREEN}Enter your choice: {Style.RESET_ALL}")

        should_continue = await handle_user_input(user_choice)
        if not should_continue:
            break

        await asyncio.sleep(1)

    websocket_task.cancel()
    try:
        await websocket_task
    except asyncio.CancelledError:
        print(f"{Fore.YELLOW}Background WebSocket processor cancelled.{Style.RESET_ALL}")

    print(f"{Fore.GREEN}Bot gracefully stopped.{Style.RESET_ALL}")


async def background_websocket_processor():
    """
    Runs in the background to continuously process WebSocket data and trigger strategies.
    Also handles automatic asset rotation if trading is active and no signal found.
    """
    global STACK, TRADING_ALLOWED, CURRENCY, CURRENCY_CHANGE, CURRENCY_CHANGE_DATE, HISTORY_TAKEN, PERIOD, ACTIONS_SECONDS, MODEL, INIT_DEPOSIT, AMOUNTS, IS_AMOUNT_SET, TOTAL_WINS_TODAY, TOTAL_LOSSES_TODAY, TODAY_START_DATE, ASSET_PERFORMANCE, CURRENT_FAVORITE_ASSET_INDEX, TOTAL_PROFIT_TODAY, TOTAL_LOSS_TODAY

    NO_SIGNAL_ASSET_SWITCH_DELAY = 30  # seconds (Switch asset if no signal for this duration)

    last_signal_time = datetime.now()

    while TRADING_ALLOWED:
        close_active_modals_if_any()

        if not CURRENCY:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No current currency. Attempting to switch to first favorite asset.")
            if FAVORITE_ASSETS:
                switch_to_asset_ui(FAVORITE_ASSETS[0])
            else:
                print(
                    f"[{Fore.RED}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No favorite assets defined. Cannot proceed without an asset.")
                TRADING_ALLOWED = False
                break
            await asyncio.sleep(5)
            continue

        STACK = websocket_log(STACK)

        if not TRADING_ALLOWED:
            print(
                f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] Trading stopped by daily limits. Exiting background processor.")
            break

        if IS_AMOUNT_SET and len(STACK) >= LENGTH_STACK_MIN:
            signal_was_generated = run_trading_strategy(STACK)

            if signal_was_generated:
                last_signal_time = datetime.now()
            else:
                if (datetime.now() - last_signal_time).total_seconds() > NO_SIGNAL_ASSET_SWITCH_DELAY:
                    print(
                        f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No signal for {NO_SIGNAL_ASSET_SWITCH_DELAY} seconds on {CURRENCY}. Attempting to switch asset.")
                    next_asset_to_try = get_next_favorite_asset()
                    if next_asset_to_try and next_asset_to_try != CURRENCY:
                        switch_to_asset_ui(next_asset_to_try)
                        last_signal_time = datetime.now()
                    else:
                        print(
                            f"[{Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}] No new favorite asset to switch to or only one asset available. Waiting.")
                        last_signal_time = datetime.now()

        await asyncio.sleep(0.1)

    # --- Run the Bot ---


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Bot interrupted by user (Ctrl+C). Shutting down.{Style.RESET_ALL}")
        if driver:
            driver.quit()
    except Exception as e:
        print(f"{Fore.RED}\nAn unexpected error occurred: {e}{Style.RESET_ALL}")
        if driver:
            driver.quit()
