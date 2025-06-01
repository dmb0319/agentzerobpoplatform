#!/usr/bin/env python3
"""
Enhanced Playwright setup with anti-detection measures and robust error handling.

This module provides configurable Playwright browser instances with enhanced
Cloudflare resilience capabilities. It includes functions for:

1. Creating and configuring browser instances with stealth mode
2. Implementing human-like interaction patterns (delays, mouse movements)
3. Detecting and handling Cloudflare challenges
4. Implementing intelligent retry mechanisms

The configuration is loaded from bpo_rules_and_settings.yaml and can be
overridden with function parameters.
"""

import os
import sys
import time
import random
import logging
import asyncio
import select
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
import yaml
from datetime import datetime

from playwright.sync_api import (
    sync_playwright,
    Page,
    Browser,
    BrowserContext,
    TimeoutError as PlaywrightTimeoutError,
)
from playwright.async_api import (
    async_playwright,
    Page as AsyncPage,
    Browser as AsyncBrowser,
    BrowserContext as AsyncBrowserContext,
    TimeoutError as AsyncPlaywrightTimeoutError,
)

print(f"[DEBUG IMPORT] playwright_setup.py loaded from: {__file__}")
print(f"[DEBUG IMPORT] perform_action_with_retry signature: {perform_action_with_retry.__code__.co_varnames}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exceptions
class MFACodeTimeoutError(Exception):
    """Raised when MFA code input times out."""
    pass

class MFACodeValidationError(Exception):
    """Raised when MFA code validation fails."""
    pass

class NetworkError(Exception):
    """Raised when network-related errors occur."""
    pass

class SessionError(Exception):
    """Raised when session-related errors occur."""
    pass

# Configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = os.getenv('BPO_CONFIG_PATH', 'bpo_agent_config')
    config_file = Path(config_path) / 'bpo_rules_and_settings.yaml'
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {
            'web_interaction': {
                'browser': {
                    'headless': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
                    'viewport_width': 1920,
                    'viewport_height': 1080,
                    'locale': 'en-US',
                    'timezone_id': 'America/New_York',
                    'geolocation': {'latitude': 40.7128, 'longitude': -74.0060}
                },
                'mfa': {
                    'timeout_seconds': 30
                }
            }
        }

# Browser Context Creation
def create_browser_context(playwright) -> Tuple[Browser, BrowserContext]:
    """Create a browser context with anti-detection measures."""
    config = load_config()
    browser_config = config['web_interaction']['browser']
    
    # Launch browser with enhanced options
    browser = playwright.chromium.launch(
        headless=browser_config['headless'],
        args=[
            '--disable-blink-features=AutomationControlled',
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-site-isolation-trials'
        ]
    )
    
    # Create context with anti-detection measures
    context = browser.new_context(
        viewport={'width': browser_config['viewport_width'], 'height': browser_config['viewport_height']},
        user_agent=browser_config['user_agent'],
        locale=browser_config['locale'],
        timezone_id=browser_config['timezone_id'],
        geolocation=browser_config['geolocation'],
        permissions=['geolocation', 'notifications'],
        ignore_https_errors=True,
        java_script_enabled=True,
        has_touch=False,
        is_mobile=False,
        device_scale_factor=1,
        color_scheme='light',
        reduced_motion='no-preference',
        forced_colors='none',
        accept_downloads=True,
        bypass_csp=True,
        service_workers='block',
        extra_http_headers={
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'DNT': '1'
        }
    )
    
    return browser, context

# Stealth Mode
def apply_stealth_mode(page: Union[Page, AsyncPage]) -> None:
    """Apply stealth mode to prevent detection."""
    try:
        # Add stealth scripts (each as a separate script)
        stealth_scripts = [
            # Navigator.webdriver
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});",
            # Navigator.languages
            "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});",
            # Navigator.plugins
            "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});",
            # Window.chrome
            "Object.defineProperty(window, 'chrome', {get: () => ({runtime: {}, loadTimes: function() {}, csi: function() {}, app: {}})});"
        ]
        for script in stealth_scripts:
            page.add_init_script(script)
    except Exception as e:
        logger.error(f"Failed to apply stealth mode: {e}")

# Human-like Behavior
def human_delay(min_ms: int = 500, max_ms: int = 2000) -> None:
    """Add random delay to simulate human behavior."""
    delay = random.uniform(min_ms / 1000, max_ms / 1000)
    time.sleep(delay)

def human_typing(page: Page, selector: str, text: str) -> None:
    """Type text with human-like delays."""
    page.click(selector)
    for char in text:
        page.type(selector, char)
        human_delay(50, 150)

async def async_human_typing(page: AsyncPage, selector: str, text: str) -> None:
    """Async version of human typing."""
    await page.click(selector)
    for char in text:
        await page.type(selector, char)
        await asyncio.sleep(random.uniform(0.05, 0.15))

# 2FA/MFA Handling
def prompt_for_2fa_code(page: Page, platform: str, timeout_seconds: int = 30) -> str:
    """Prompt user for 2FA code with timeout."""
    print(f"\n2FA code required for {platform}")
    print("Please enter the code (timeout in {timeout_seconds} seconds): ", end='', flush=True)
    
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            code = sys.stdin.readline().strip()
            if validate_2fa_code(code):
                return code
            print("Invalid code format. Please try again: ", end='', flush=True)
    
    raise MFACodeTimeoutError(f"2FA code input timed out after {timeout_seconds} seconds")

async def async_prompt_for_2fa_code(page: AsyncPage, platform: str, timeout_seconds: int = 30) -> str:
    """Async version of 2FA code prompt."""
    print(f"\n2FA code required for {platform}")
    print("Please enter the code (timeout in {timeout_seconds} seconds): ", end='', flush=True)
    
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            code = sys.stdin.readline().strip()
            if validate_2fa_code(code):
                return code
            print("Invalid code format. Please try again: ", end='', flush=True)
        await asyncio.sleep(0.1)
    
    raise MFACodeTimeoutError(f"2FA code input timed out after {timeout_seconds} seconds")

def validate_2fa_code(code: str) -> bool:
    """Validate 2FA code format."""
    return bool(code and code.isdigit() and len(code) == 6)

def handle_2fa_with_retry(page: Page, platform: str, max_retries: int = 3) -> Optional[str]:
    """Handle 2FA with retry logic."""
    for attempt in range(max_retries):
        try:
            return prompt_for_2fa_code(page, platform)
        except MFACodeTimeoutError:
            if attempt < max_retries - 1:
                print(f"Retrying 2FA code input (attempt {attempt + 2}/{max_retries})...")
            else:
                logger.error("Max retries exceeded for 2FA code input")
                return None
    return None

async def async_handle_2fa_with_retry(page: AsyncPage, platform: str, max_retries: int = 3) -> Optional[str]:
    """Async version of 2FA handling with retry."""
    for attempt in range(max_retries):
        try:
            return await async_prompt_for_2fa_code(page, platform)
        except MFACodeTimeoutError:
            if attempt < max_retries - 1:
                print(f"Retrying 2FA code input (attempt {attempt + 2}/{max_retries})...")
            else:
                logger.error("Max retries exceeded for 2FA code input")
                return None
    return None

# Error Handling
def handle_network_error(page: Page, error: Exception) -> bool:
    """Handle network-related errors."""
    if page.is_closed():
        return False
    
    try:
        page.reload()
        return True
    except Exception as e:
        logger.error(f"Failed to handle network error: {e}")
        return False

async def async_handle_network_error(page: AsyncPage, error: Exception) -> bool:
    """Async version of network error handling."""
    if page.is_closed():
        return False
    
    try:
        await page.reload()
        return True
    except Exception as e:
        logger.error(f"Failed to handle network error: {e}")
        return False

def recover_session(page: Page, context: BrowserContext) -> bool:
    """Attempt to recover from session errors."""
    try:
        # Get current context properties
        viewport = context.viewport_size
        user_agent = context.user_agent
        locale = context.locale
        timezone_id = context.timezone_id
        geolocation = context.geolocation
        permissions = context.permissions
        ignore_https_errors = context.ignore_https_errors
        java_script_enabled = context.java_script_enabled
        has_touch = context.has_touch
        is_mobile = context.is_mobile
        device_scale_factor = context.device_scale_factor
        color_scheme = context.color_scheme
        reduced_motion = context.reduced_motion
        forced_colors = context.forced_colors
        accept_downloads = context.accept_downloads
        bypass_csp = context.bypass_csp
        service_workers = context.service_workers
        extra_http_headers = context.extra_http_headers
        cookies = context.cookies()
        
        # Close current context
        context.close()
        
        # Create new context with same properties
        new_context = context.browser.new_context(
            viewport=viewport,
            user_agent=user_agent,
            locale=locale,
            timezone_id=timezone_id,
            geolocation=geolocation,
            permissions=permissions,
            ignore_https_errors=ignore_https_errors,
            java_script_enabled=java_script_enabled,
            has_touch=has_touch,
            is_mobile=is_mobile,
            device_scale_factor=device_scale_factor,
            color_scheme=color_scheme,
            reduced_motion=reduced_motion,
            forced_colors=forced_colors,
            accept_downloads=accept_downloads,
            bypass_csp=bypass_csp,
            service_workers=service_workers,
            extra_http_headers=extra_http_headers
        )
        
        # Restore cookies
        new_context.add_cookies(cookies)
        
        # Create new page
        new_page = new_context.new_page()
        
        return True
    except Exception as e:
        logger.error(f"Failed to recover session: {e}")
        return False

async def async_recover_session(page: AsyncPage, context: AsyncBrowserContext) -> bool:
    """Async version of session recovery."""
    try:
        # Get current context properties
        viewport = await context.viewport_size
        user_agent = context.user_agent
        locale = context.locale
        timezone_id = context.timezone_id
        geolocation = context.geolocation
        permissions = context.permissions
        ignore_https_errors = context.ignore_https_errors
        java_script_enabled = context.java_script_enabled
        has_touch = context.has_touch
        is_mobile = context.is_mobile
        device_scale_factor = context.device_scale_factor
        color_scheme = context.color_scheme
        reduced_motion = context.reduced_motion
        forced_colors = context.forced_colors
        accept_downloads = context.accept_downloads
        bypass_csp = context.bypass_csp
        service_workers = context.service_workers
        extra_http_headers = context.extra_http_headers
        cookies = await context.cookies()
        
        # Close current context
        await context.close()
        
        # Create new context with same properties
        new_context = await context.browser.new_context(
            viewport=viewport,
            user_agent=user_agent,
            locale=locale,
            timezone_id=timezone_id,
            geolocation=geolocation,
            permissions=permissions,
            ignore_https_errors=ignore_https_errors,
            java_script_enabled=java_script_enabled,
            has_touch=has_touch,
            is_mobile=is_mobile,
            device_scale_factor=device_scale_factor,
            color_scheme=color_scheme,
            reduced_motion=reduced_motion,
            forced_colors=forced_colors,
            accept_downloads=accept_downloads,
            bypass_csp=bypass_csp,
            service_workers=service_workers,
            extra_http_headers=extra_http_headers
        )
        
        # Restore cookies
        await new_context.add_cookies(cookies)
        
        # Create new page
        new_page = await new_context.new_page()
        
        return True
    except Exception as e:
        logger.error(f"Failed to recover session: {e}")
        return False

# Cloudflare Challenge Handling
def detect_cloudflare_challenge(page: Page) -> bool:
    """Detect if page is showing Cloudflare challenge."""
    try:
        # Check for common Cloudflare challenge elements
        return bool(
            page.locator('text="Checking your browser"').count() > 0 or
            page.locator('text="Just a moment"').count() > 0 or
            page.locator('text="Please wait"').count() > 0 or
            page.locator('text="Security check"').count() > 0
        )
    except Exception:
        return False

def handle_cloudflare_challenge(page: Page, max_wait_time: int = 30) -> bool:
    """Handle Cloudflare challenge with timeout."""
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if not detect_cloudflare_challenge(page):
            return True
        time.sleep(1)
    return False

# Navigation and Action Helpers
def navigate_with_retry(page: Page, url: str, max_retries: int = 3) -> bool:
    """Navigate to URL with retry logic."""
    for attempt in range(max_retries):
        try:
            page.goto(url)
            if detect_cloudflare_challenge(page):
                if handle_cloudflare_challenge(page):
                    return True
            else:
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Navigation attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Navigation failed after {max_retries} attempts: {e}")
                return False
    return False

def perform_action_with_retry(action: callable, action_name: str, max_retries: int = 3) -> bool:
    print(f"[DEBUG] perform_action_with_retry called with action={action}, action_name={action_name}, max_retries={max_retries}")
    for attempt in range(max_retries):
        try:
            result = action()
            print(f"[DEBUG] Attempt {attempt+1}: action result = {result}")
            if result == "success":
                return True
            # If not success, treat as failure and retry
            if attempt < max_retries - 1:
                logger.warning(f"{action_name} attempt {attempt + 1} failed: {result}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"{action_name} failed after {max_retries} attempts: {result}")
                return False
        except Exception as e:
            print(f"[DEBUG] Exception: {e}")
            if attempt < max_retries - 1:
                logger.warning(f"{action_name} attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"{action_name} failed after {max_retries} attempts: {e}")
                return False
    return False

# Screenshot and Debugging
async def save_screenshot(page, name: str, directory: str = "screenshots") -> str:
    print(f"[DEBUG] save_screenshot called with page={page}, name={name}, directory={directory}")
    print(f"[DEBUG] page.screenshot: {getattr(page, 'screenshot', None)}")
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}.png"
        os.makedirs(directory, exist_ok=True)
        screenshot_path = os.path.join(directory, filename)
        print(f"[DEBUG] Calling page.screenshot with path: {screenshot_path}")
        await page.screenshot(path=screenshot_path)
        return screenshot_path
    except Exception as e:
        logger.error(f"Failed to save screenshot: {e}")
        raise

# Browser Management
def create_playwright_browser() -> Tuple[Any, Browser, BrowserContext, Page]:
    """Create and configure Playwright browser instance."""
    playwright = sync_playwright().start()
    browser, context = create_browser_context(playwright)
    page = context.new_page()
    apply_stealth_mode(page)
    return playwright, browser, context, page

def close_playwright_browser(playwright: Any, browser: Browser, context: BrowserContext, page: Page) -> None:
    """Clean up Playwright resources."""
    try:
        page.close()
        context.close()
        browser.close()
        playwright.stop()
    except Exception as e:
        logger.error(f"Error during browser cleanup: {e}")

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example: Navigate to a website with Cloudflare protection
    try:
        playwright, browser, context, page = create_playwright_browser()
        
        # Example navigation with retry
        success = navigate_with_retry(page, "https://www.example.com")
        
        if success:
            # Example action with retry (click a button)
            def click_button():
                page.click("button#example")
                return True
            
            perform_action_with_retry(click_button, "click_example_button")
        
    except Exception as e:
        logging.error(f"Error in example: {e}")
    finally:
        # Always close the browser
        close_playwright_browser(playwright, browser, context, page)
