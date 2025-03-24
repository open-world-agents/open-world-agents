# agent/infrastructure/desktop_actions/browser.py
import base64
import logging
import time
from io import BytesIO
from typing import Any, Dict

from agent.domain.services import TaskExecutionService
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from shared.protocol import TaskSpecification
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


def setup_browser_actions(service: TaskExecutionService):
    """Register browser-related task handlers."""
    service.register_task_handler("book_flight", book_flight_handler)
    service.register_task_handler("search_web", search_web_handler)
    service.register_task_handler("fill_form", fill_form_handler)


def book_flight_handler(task: TaskSpecification, service: TaskExecutionService) -> bool:
    """Task handler for booking a flight."""
    service.log("Starting flight booking process")

    # Extract task parameters
    params = task.steps[0].get("params", {})
    website = params.get("website", "https://demo.travel-website.com")
    from_location = params.get("from", "New York")
    to_location = params.get("to", "Los Angeles")
    departure_date = params.get("departure_date", "2023-12-15")
    return_date = params.get("return_date", "2023-12-22")

    try:
        # Setup Chrome WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service.log("Initializing web browser")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # Navigate to travel website
        service.log(f"Navigating to {website}")
        driver.get(website)

        # Take screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Fill out the search form
        service.log("Filling flight search form")
        driver.find_element(By.ID, "from").send_keys(from_location)
        driver.find_element(By.ID, "to").send_keys(to_location)
        driver.find_element(By.ID, "departure").send_keys(departure_date)
        driver.find_element(By.ID, "return").send_keys(return_date)

        # Take screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Submit search
        service.log("Submitting search")
        driver.find_element(By.ID, "search-button").click()

        # Wait for results
        service.log("Waiting for search results")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "flight-results")))

        # Take screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Select first flight
        service.log("Selecting flight")
        driver.find_element(By.CSS_SELECTOR, ".flight-results .select-button").click()

        # Wait for passenger form
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "passenger-details")))

        # Take screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Fill passenger details
        service.log("Filling passenger details")
        driver.find_element(By.ID, "first-name").send_keys("Test")
        driver.find_element(By.ID, "last-name").send_keys("User")
        driver.find_element(By.ID, "email").send_keys("test@example.com")

        # Take screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Submit booking
        service.log("Completing booking")
        driver.find_element(By.ID, "complete-booking").click()

        # Wait for confirmation
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "booking-confirmation")))

        # Take final screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Check for success criteria
        confirmation_element = driver.find_element(By.CLASS_NAME, "booking-confirmation")
        success = "confirmed" in confirmation_element.text.lower()

        if success:
            service.log("Flight booking successful")
        else:
            service.log("Flight booking did not complete successfully")

        # Close browser
        driver.quit()
        return success

    except Exception as e:
        service.log(f"Error during flight booking: {str(e)}")
        return False


def search_web_handler(task: TaskSpecification, service: TaskExecutionService) -> bool:
    """Task handler for web searching."""
    service.log("Starting web search task")

    # Extract parameters
    params = task.steps[0].get("params", {})
    search_engine = params.get("search_engine", "https://www.google.com")
    query = params.get("query", "python automation")

    try:
        # Setup Chrome WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service.log("Initializing web browser")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # Navigate to search engine
        service.log(f"Navigating to {search_engine}")
        driver.get(search_engine)

        # Take screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Perform search based on search engine
        service.log(f"Searching for: {query}")

        if "google.com" in search_engine:
            # Google search
            search_box = driver.find_element(By.NAME, "q")
            search_box.send_keys(query)
            search_box.submit()
        elif "bing.com" in search_engine:
            # Bing search
            search_box = driver.find_element(By.NAME, "q")
            search_box.send_keys(query)
            search_box.submit()
        else:
            # Generic search
            search_box = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
            search_box.send_keys(query)
            search_box.submit()

        # Wait for results
        service.log("Waiting for search results")
        time.sleep(2)  # Simple wait for results to load

        # Take screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Check for results
        page_source = driver.page_source.lower()
        search_terms = query.lower().split()

        # Check if search terms appear in the page
        found_terms = sum(1 for term in search_terms if term in page_source)
        success = found_terms > 0

        if success:
            service.log("Search completed successfully")
        else:
            service.log("Search did not return expected results")

        driver.quit()
        return success

    except Exception as e:
        service.log(f"Error during web search: {str(e)}")
        return False


def fill_form_handler(task: TaskSpecification, service: TaskExecutionService) -> bool:
    """Task handler for filling out a web form."""
    service.log("Starting form filling task")

    # Extract parameters
    params = task.steps[0].get("params", {})
    form_url = params.get("url", "https://example.com/form")
    form_fields = params.get("fields", {})
    submit_button = params.get("submit_button", "submit")

    try:
        # Setup Chrome WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service.log("Initializing web browser")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # Navigate to form
        service.log(f"Navigating to {form_url}")
        driver.get(form_url)

        # Take screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Fill each field
        service.log("Filling form fields")
        for field_id, value in form_fields.items():
            try:
                # First try to find by ID
                field = driver.find_element(By.ID, field_id)
            except:
                try:
                    # Then try to find by name
                    field = driver.find_element(By.NAME, field_id)
                except:
                    # Finally try to find by CSS selector
                    field = driver.find_element(By.CSS_SELECTOR, field_id)

            field.send_keys(value)

        # Take screenshot after filling
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Submit the form
        service.log("Submitting form")
        try:
            # First try to find by ID
            submit = driver.find_element(By.ID, submit_button)
        except:
            try:
                # Then try to find by name
                submit = driver.find_element(By.NAME, submit_button)
            except:
                try:
                    # Then try to find by class
                    submit = driver.find_element(By.CLASS_NAME, submit_button)
                except:
                    # Finally try to find by CSS selector
                    submit = driver.find_element(By.CSS_SELECTOR, submit_button)

        submit.click()

        # Wait for form submission
        service.log("Waiting for form submission")
        time.sleep(2)

        # Take final screenshot
        screenshot = _take_screenshot(driver)
        service.add_screenshot(screenshot)

        # Check success criteria
        current_url = driver.current_url
        success = current_url != form_url  # Assume success if URL changes

        if success:
            service.log("Form submitted successfully")
        else:
            service.log("Form submission did not complete as expected")

        driver.quit()
        return success

    except Exception as e:
        service.log(f"Error during form filling: {str(e)}")
        return False


def _take_screenshot(driver) -> str:
    """Take a screenshot and return as base64 string."""
    screenshot = driver.get_screenshot_as_png()
    return base64.b64encode(screenshot).decode("utf-8")
