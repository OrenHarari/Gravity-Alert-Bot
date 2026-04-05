import json
from playwright.sync_api import sync_playwright
import os

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.set_viewport_size({"width": 1280, "height": 800})
    
    # Path to dashboard.html
    html_path = f"file:///{os.path.abspath('dashboard.html')}"
    page.goto(html_path)
    
    # Wait for the page to render fully
    page.wait_for_timeout(2000)
    
    try:
        # Click on the Aggressive PNL Runner tab to show it in the screenshot
        page.click("text='Super Turtle Breakout (188%)'")
        page.wait_for_timeout(1000)
    except:
        pass
        
    page.screenshot(path="dashboard-preview.png")
    print("Screenshot saved to dashboard-preview.png")
    browser.close()
