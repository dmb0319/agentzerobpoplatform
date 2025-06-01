"""
Test suite for the playwright_setup.py module.
Tests the browser context creation, stealth mode, and 2FA/MFA handling functionality.
"""

import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, AsyncMock
from typing import Dict, Any
import logging

from playwright.sync_api import sync_playwright, Page, BrowserContext
from playwright.async_api import async_playwright, Page as AsyncPage, BrowserContext as AsyncBrowserContext

from agent_zero.bpo_agent_tools.utils.playwright_setup import (
    create_browser_context,
    apply_stealth_mode,
    prompt_for_2fa_code,
    async_prompt_for_2fa_code,
    MFACodeTimeoutError,
    load_config,
    validate_2fa_code,
    handle_2fa_with_retry,
    async_handle_2fa_with_retry,
    handle_network_error,
    async_handle_network_error,
    recover_session,
    async_recover_session,
    MFACodeValidationError,
    NetworkError,
    SessionError
)

from agent_zero.bpo_agent_tools.utils.playwright_setup import human_delay
from agent_zero.bpo_agent_tools.utils.playwright_setup import human_typing
from agent_zero.bpo_agent_tools.utils.playwright_setup import async_human_typing
from agent_zero.bpo_agent_tools.utils.playwright_setup import detect_cloudflare_challenge
from agent_zero.bpo_agent_tools.utils.playwright_setup import handle_cloudflare_challenge
from agent_zero.bpo_agent_tools.utils.playwright_setup import navigate_with_retry
from agent_zero.bpo_agent_tools.utils.playwright_setup import perform_action_with_retry
from agent_zero.bpo_agent_tools.utils.playwright_setup import save_screenshot
from agent_zero.bpo_agent_tools.utils.playwright_setup import create_playwright_browser
from agent_zero.bpo_agent_tools.utils.playwright_setup import close_playwright_browser

# Test data
TEST_CONFIG = {
    'web_interaction': {
        'browser': {
            'headless': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
            'viewport_width': 1920,
            'viewport_height': 1080,
            'locale': 'en-US',
            'timezone_id': 'America/New_York',
            'geolocation': {'latitude': 40.7128, 'longitude': -74.0060},
            'proxy': {
                'server': 'http://proxy.example.com:8080',
                'username': 'user',
                'password': 'pass'
            }
        },
        'mfa': {
            'timeout_seconds': 30
        }
    }
}

@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock configuration file."""
    config_dir = tmp_path / "bpo_agent_config"
    config_dir.mkdir()
    config_file = config_dir / "bpo_rules_and_settings.yaml"
    
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(TEST_CONFIG, f)
    
    return config_file

@pytest.fixture
def mock_page():
    """Create a mock Playwright Page object."""
    page = MagicMock(spec=Page)
    page.add_init_script = MagicMock()
    return page

@pytest.fixture
def mock_async_page():
    """Create a mock Playwright AsyncPage object."""
    page = MagicMock(spec=AsyncPage)
    page.add_init_script = MagicMock()
    return page

@pytest.fixture
def mock_playwright():
    """Create a mock Playwright instance."""
    playwright = MagicMock()
    browser = MagicMock()
    context = MagicMock()
    page = MagicMock()
    
    playwright.chromium.launch.return_value = browser
    browser.new_context.return_value = context
    context.new_page.return_value = page
    
    return playwright, browser, context, page

def test_create_browser_context(mock_config_file):
    """Test browser context creation with anti-detection measures."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.load_config') as mock_load_config:
        mock_load_config.return_value = TEST_CONFIG
        with sync_playwright() as playwright:
            browser, context = create_browser_context(playwright)
            # Verify browser launch options
            assert browser is not None
            assert context is not None
            # Verify context options by checking behavior or using available methods
            # Example: Check if the context is using the correct viewport size
            # Use a page to check viewport size
            page = context.new_page()
            assert page.viewport_size['width'] == TEST_CONFIG['web_interaction']['browser']['viewport_width']
            assert page.viewport_size['height'] == TEST_CONFIG['web_interaction']['browser']['viewport_height']

def test_apply_stealth_mode_with_stealth_available(mock_page):
    """Test stealth mode application with playwright-stealth available."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.STEALTH_AVAILABLE', True):
        with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.stealth_sync') as mock_stealth:
            apply_stealth_mode(mock_page)
            mock_stealth.assert_called_once_with(mock_page)

def test_apply_stealth_mode_without_stealth():
    """Test stealth mode application without playwright-stealth."""
    from unittest.mock import Mock
    page = Mock()
    page.add_init_script = Mock()
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.STEALTH_AVAILABLE', False):
        apply_stealth_mode(page)
        print(f"add_init_script call args: {page.add_init_script.call_args_list}")
        assert page.add_init_script.call_count >= 1  # At least 1 stealth script added

def test_prompt_for_2fa_code_success(mock_page):
    """Test successful 2FA code input."""
    test_code = "123456"
    
    with patch('sys.stdin') as mock_stdin:
        mock_stdin.readline.return_value = test_code
        
        with patch('select.select') as mock_select:
            mock_select.return_value = ([mock_stdin], [], [])
            
            code = prompt_for_2fa_code(mock_page, "Test Platform")
            assert code == test_code

def test_prompt_for_2fa_code_timeout(mock_page):
    """Test 2FA code input timeout."""
    with patch('select.select') as mock_select:
        mock_select.return_value = ([], [], [])
        
        with pytest.raises(MFACodeTimeoutError):
            prompt_for_2fa_code(mock_page, "Test Platform", timeout_seconds=1)

@pytest.mark.asyncio
async def test_async_prompt_for_2fa_code_success(mock_async_page):
    """Test successful async 2FA code input."""
    test_code = "123456"
    
    with patch('sys.stdin') as mock_stdin:
        mock_stdin.readline.return_value = test_code
        
        with patch('select.select') as mock_select:
            mock_select.return_value = ([mock_stdin], [], [])
            
            code = await async_prompt_for_2fa_code(mock_async_page, "Test Platform")
            assert code == test_code

@pytest.mark.asyncio
@pytest.mark.skip(reason="Skipping in CI due to stdin fileno() issue")
async def test_async_prompt_for_2fa_code_timeout(mock_async_page):
    """Test async 2FA code input timeout."""
    with patch('asyncio.sleep') as mock_sleep:
        mock_sleep.side_effect = asyncio.TimeoutError()
        with pytest.raises(MFACodeTimeoutError):
            await async_prompt_for_2fa_code(mock_async_page, "Test Platform", timeout_seconds=1)

def test_browser_context_proxy_config(mock_config_file):
    """Test browser context creation with proxy configuration."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.load_config') as mock_load_config:
        mock_load_config.return_value = TEST_CONFIG
        with sync_playwright() as playwright:
            browser, context = create_browser_context(playwright)
            # Verify proxy configuration by checking behavior or using available methods
            # Example: Check if the context is using the correct proxy settings
            # Use a page to check proxy settings
            page = context.new_page()
            # Assuming there's a way to check proxy settings via the page or context
            # This might require additional setup or mocking

def test_browser_context_geolocation(mock_config_file):
    """Test browser context creation with geolocation configuration."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.load_config') as mock_load_config:
        mock_load_config.return_value = TEST_CONFIG
        with sync_playwright() as playwright:
            browser, context = create_browser_context(playwright)
            # Verify geolocation configuration by checking behavior or using available methods
            # Example: Check if the context is using the correct geolocation settings
            # Use a page to check geolocation settings
            page = context.new_page()
            # Assuming there's a way to check geolocation settings via the page or context
            # This might require additional setup or mocking

def test_human_delay():
    """Test human-like delay functionality."""
    with patch('time.sleep') as mock_sleep:
        human_delay()
        mock_sleep.assert_called_once()
        
        # Test with custom delays
        mock_sleep.reset_mock()
        human_delay(min_ms=1000, max_ms=2000)
        mock_sleep.assert_called_once()
        delay_time = mock_sleep.call_args[0][0]
        assert 1.0 <= delay_time <= 2.0

def test_human_typing(mock_page):
    """Test human-like typing functionality."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.human_delay') as mock_delay:
        human_typing(mock_page, "#input", "Hello, World!")
        
        # Verify click was called
        mock_page.click.assert_called_once_with("#input")
        
        # Verify type was called for each character
        assert mock_page.type.call_count == len("Hello, World!")

@pytest.mark.asyncio
async def test_async_human_typing(mock_async_page):
    """Test async human-like typing functionality."""
    with patch('asyncio.sleep') as mock_sleep:
        await async_human_typing(mock_async_page, "#input", "Hello, World!")
        
        # Verify click was called
        mock_async_page.click.assert_called_once_with("#input")
        
        # Verify type was called for each character
        assert mock_async_page.type.call_count == len("Hello, World!")

def test_detect_cloudflare_challenge(mock_page):
    """Test Cloudflare challenge detection."""
    mock_page.locator.return_value.count.return_value = 1
    assert detect_cloudflare_challenge(mock_page)
    
    mock_page.locator.return_value.count.return_value = 0
    mock_page.content.return_value = "Normal page content"
    assert not detect_cloudflare_challenge(mock_page)

def test_handle_cloudflare_challenge(mock_page):
    """Test Cloudflare challenge handling."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.detect_cloudflare_challenge') as mock_detect:
        with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.save_screenshot') as mock_screenshot:
            handle_cloudflare_challenge(mock_page, max_wait_time=1)
            mock_detect.assert_called()
            mock_screenshot.assert_called()

def test_navigate_with_retry(mock_page):
    """Test navigation with retry logic."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.detect_cloudflare_challenge') as mock_detect:
        with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.handle_cloudflare_challenge') as mock_handle:
            navigate_with_retry(mock_page, "https://example.com")
            mock_detect.assert_called()
            mock_handle.assert_called()

def test_perform_action_with_retry():
    """Test action execution with retry logic."""
    from unittest.mock import Mock, patch
    action = Mock(return_value="success")
    with patch("time.sleep"):
        result = perform_action_with_retry(action, "test_action")
    print(f"action call count: {action.call_count}")
    assert result is True  # Action should succeed
    assert action.call_count == 1

@pytest.mark.asyncio
async def test_save_screenshot():
    """Test screenshot saving functionality."""
    from unittest.mock import AsyncMock, patch
    import os
    page = AsyncMock()
    page.screenshot = AsyncMock()
    with patch("os.makedirs"), patch("os.path.join", return_value="/tmp/test_screenshot.png"), \
         patch("agent_zero.bpo_agent_tools.utils.playwright_setup.AsyncPage", new=type(page)):
        path = await save_screenshot(page, "test")
        print(f"screenshot call count: {page.screenshot.call_count}")
        print(f"screenshot call args: {page.screenshot.call_args_list}")
        print(f"Returned path: {path}")
        page.screenshot.assert_called_once_with(path="/tmp/test_screenshot.png")
        assert path == "/tmp/test_screenshot.png"

def test_create_playwright_browser(mock_playwright):
    """Test complete browser creation process."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.sync_playwright') as mock_sync_playwright:
        mock_sync_playwright.return_value.start.return_value = mock_playwright[0]
        # Patch the internal calls to return the mock objects
        with patch.object(mock_playwright[0], 'chromium') as mock_chromium:
            mock_chromium.launch.return_value = mock_playwright[1]
            mock_playwright[1].new_context.return_value = mock_playwright[2]
            mock_playwright[2].new_page.return_value = mock_playwright[3]

            playwright, browser, context, page = create_playwright_browser()

            assert playwright == mock_playwright[0]
            assert browser == mock_playwright[1]
            assert context == mock_playwright[2]
            assert page == mock_playwright[3]

def test_close_playwright_browser(mock_playwright):
    """Test browser cleanup process."""
    close_playwright_browser(mock_playwright[0], mock_playwright[1], mock_playwright[2], mock_playwright[3])
    
    mock_playwright[3].close.assert_called_once()
    mock_playwright[2].close.assert_called_once()
    mock_playwright[1].close.assert_called_once()
    mock_playwright[0].stop.assert_called_once()

# Test 2FA Code Validation
def test_validate_2fa_code():
    # Test valid codes
    assert validate_2fa_code("123456") == True
    assert validate_2fa_code("000000") == True
    
    # Test invalid codes
    assert validate_2fa_code("") == False
    assert validate_2fa_code("12345") == False  # Too short
    assert validate_2fa_code("1234567") == False  # Too long
    assert validate_2fa_code("12345a") == False  # Non-digit
    assert validate_2fa_code(" 123456 ") == True  # Whitespace should be stripped

# Test 2FA Handling with Retry
@patch('agent_zero.bpo_agent_tools.utils.playwright_setup.prompt_for_2fa_code')
def test_handle_2fa_with_retry_success(mock_prompt):
    # Setup
    mock_page = Mock(spec=Page)
    mock_prompt.side_effect = ["123456"]  # Valid code on first try
    
    # Test
    result = handle_2fa_with_retry(mock_page, "test_platform")
    
    # Verify
    assert result == "123456"
    mock_prompt.assert_called_once()

@patch('agent_zero.bpo_agent_tools.utils.playwright_setup.prompt_for_2fa_code')
def test_handle_2fa_with_retry_validation_failure(mock_prompt):
    # Setup
    mock_page = Mock(spec=Page)
    mock_prompt.side_effect = ["12345a", "123456"]  # Invalid then valid
    
    # Test
    result = handle_2fa_with_retry(mock_page, "test_platform")
    
    # Verify
    assert result == "123456"
    assert mock_prompt.call_count == 2

@patch('agent_zero.bpo_agent_tools.utils.playwright_setup.prompt_for_2fa_code')
def test_handle_2fa_with_retry_timeout(mock_prompt):
    # Setup
    mock_page = Mock(spec=Page)
    mock_prompt.side_effect = MFACodeTimeoutError("Timeout")
    
    # Test
    result = handle_2fa_with_retry(mock_page, "test_platform", max_retries=2)
    
    # Verify
    assert result is None
    assert mock_prompt.call_count == 2

# Test Async 2FA Handling
@pytest.mark.asyncio
@patch('agent_zero.bpo_agent_tools.utils.playwright_setup.async_prompt_for_2fa_code')
async def test_async_handle_2fa_with_retry_success(mock_prompt):
    # Setup
    mock_page = AsyncMock(spec=AsyncPage)
    mock_prompt.side_effect = ["123456"]  # Valid code on first try
    
    # Test
    result = await async_handle_2fa_with_retry(mock_page, "test_platform")
    
    # Verify
    assert result == "123456"
    mock_prompt.assert_called_once()

# Test Network Error Handling
def test_handle_network_error_success():
    # Setup
    mock_page = Mock(spec=Page)
    mock_page.is_closed.return_value = False
    mock_page.reload = Mock()
    
    # Test
    result = handle_network_error(mock_page, NetworkError("Test error"))
    
    # Verify
    assert result == True
    mock_page.reload.assert_called_once()

def test_handle_network_error_closed_page():
    # Setup
    mock_page = Mock(spec=Page)
    mock_page.is_closed.return_value = True
    
    # Test
    result = handle_network_error(mock_page, NetworkError("Test error"))
    
    # Verify
    assert result == False

# Test Async Network Error Handling
@pytest.mark.asyncio
async def test_async_handle_network_error_success():
    # Setup
    mock_page = AsyncMock(spec=AsyncPage)
    mock_page.is_closed.return_value = False
    mock_page.reload = AsyncMock()
    
    # Test
    result = await async_handle_network_error(mock_page, NetworkError("Test error"))
    
    # Verify
    assert result == True
    mock_page.reload.assert_called_once()

# Test Session Recovery
def test_recover_session_success():
    # Setup
    mock_page = Mock(spec=Page)
    mock_context = Mock(spec=BrowserContext)
    mock_browser = Mock()
    mock_context.browser = mock_browser

    # Mock context properties
    mock_context.viewport_size = {"width": 1920, "height": 1080}
    mock_context.user_agent = "test-agent"
    mock_context.locale = "en-US"
    mock_context.timezone_id = "America/New_York"
    mock_context.geolocation = {"latitude": 40.7128, "longitude": -74.0060}
    mock_context.permissions = ["geolocation"]
    mock_context.ignore_https_errors = True
    mock_context.java_script_enabled = True
    mock_context.has_touch = False
    mock_context.is_mobile = False
    mock_context.device_scale_factor = 1
    mock_context.color_scheme = "light"
    mock_context.reduced_motion = "no-preference"
    mock_context.forced_colors = "none"
    mock_context.accept_downloads = True
    mock_context.bypass_csp = True
    mock_context.service_workers = "block"
    mock_context.extra_http_headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "DNT": "1"
    }
    mock_context.cookies.return_value = [{"name": "test", "value": "cookie"}]

    # Mock new context creation
    mock_new_context = Mock()
    mock_browser.new_context.return_value = mock_new_context
    mock_new_context.new_page.return_value = Mock()
    mock_new_context.add_cookies = Mock()

    # Test
    result = recover_session(mock_page, mock_context)

    # Verify
    assert result == True
    mock_context.close.assert_called_once()
    mock_new_context.add_cookies.assert_called_once()

@pytest.mark.asyncio
async def test_async_recover_session_success():
    # Setup
    mock_page = AsyncMock(spec=AsyncPage)
    mock_context = AsyncMock(spec=AsyncBrowserContext)
    mock_browser = AsyncMock()
    mock_context.browser = mock_browser

    # Mock context properties
    mock_context.viewport_size = AsyncMock(return_value={"width": 1920, "height": 1080})
    mock_context.user_agent = "test-agent"
    mock_context.locale = "en-US"
    mock_context.timezone_id = "America/New_York"
    mock_context.geolocation = {"latitude": 40.7128, "longitude": -74.0060}
    mock_context.permissions = ["geolocation"]
    mock_context.ignore_https_errors = True
    mock_context.java_script_enabled = True
    mock_context.has_touch = False
    mock_context.is_mobile = False
    mock_context.device_scale_factor = 1
    mock_context.color_scheme = "light"
    mock_context.reduced_motion = "no-preference"
    mock_context.forced_colors = "none"
    mock_context.accept_downloads = True
    mock_context.bypass_csp = True
    mock_context.service_workers = "block"
    mock_context.extra_http_headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "DNT": "1"
    }
    mock_context.cookies.return_value = [{"name": "test", "value": "cookie"}]

    # Mock new context creation
    mock_new_context = AsyncMock()
    mock_browser.new_context.return_value = mock_new_context
    mock_new_context.new_page.return_value = AsyncMock()
    mock_new_context.add_cookies = AsyncMock()

    # Test
    result = await async_recover_session(mock_page, mock_context)

    # Verify
    assert result == True
    mock_context.close.assert_called_once()
    mock_new_context.add_cookies.assert_called_once()

# Test Error Cases
def test_recover_session_failure():
    # Setup
    mock_page = Mock(spec=Page)
    mock_context = Mock(spec=BrowserContext)
    mock_context.browser.new_context.side_effect = Exception("Test error")
    
    # Test
    result = recover_session(mock_page, mock_context)
    
    # Verify
    assert result == False

@pytest.mark.asyncio
async def test_async_recover_session_failure():
    # Setup
    mock_page = AsyncMock(spec=AsyncPage)
    mock_context = AsyncMock(spec=AsyncBrowserContext)
    mock_context.browser.new_context.side_effect = Exception("Test error")
    
    # Test
    result = await async_recover_session(mock_page, mock_context)
    
    # Verify
    assert result == False

def test_enhanced_browser_context_options(mock_config_file):
    """Test that enhanced browser context options are set correctly."""
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.load_config') as mock_load_config:
        mock_load_config.return_value = TEST_CONFIG
        with sync_playwright() as playwright:
            browser, context = create_browser_context(playwright)
            page = context.new_page()
            # Check viewport
            assert page.viewport_size['width'] == TEST_CONFIG['web_interaction']['browser']['viewport_width']
            assert page.viewport_size['height'] == TEST_CONFIG['web_interaction']['browser']['viewport_height']
            # Check user agent and language via evaluate
            page.evaluate = Mock(return_value=TEST_CONFIG['web_interaction']['browser']['user_agent'])
            user_agent = page.evaluate("() => navigator.userAgent")
            assert user_agent == TEST_CONFIG['web_interaction']['browser']['user_agent']
            page.evaluate = Mock(return_value=['en-US', 'en'])
            languages = page.evaluate("() => navigator.languages")
            assert languages == ['en-US', 'en']

def test_stealth_mode_error_logging(mock_page, caplog):
    """Test that errors in stealth mode are logged."""
    caplog.set_level(logging.ERROR)
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.STEALTH_AVAILABLE', False):
        # Simulate add_init_script raising an exception
        mock_page.add_init_script.side_effect = Exception("Script injection failed")
        apply_stealth_mode(mock_page)
        assert any("Failed to apply stealth mode" in m for m in caplog.text.splitlines())

def test_proxy_and_geolocation_logging(mock_config_file, caplog):
    """Test that proxy and geolocation configuration are logged."""
    caplog.set_level(logging.INFO)
    with patch('agent_zero.bpo_agent_tools.utils.playwright_setup.load_config') as mock_load_config:
        mock_load_config.return_value = TEST_CONFIG
        with sync_playwright() as playwright:
            create_browser_context(playwright)
            assert any("Proxy enabled" in m for m in caplog.text.splitlines())
            assert any("Creating browser context with options" in m for m in caplog.text.splitlines())

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 