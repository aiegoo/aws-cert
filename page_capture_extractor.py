#!/usr/bin/env python3
"""
Enhanced browser text and screenshot extractor
Incorporates kindleOCRer techniques: page flipping, screenshots, PDF generation
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pyperclip
import time
import argparse
from PIL import Image
import io
import os
from pathlib import Path


class PageCaptureExtractor:
    """Browser automation with screenshot capture and page flipping (kindleOCRer techniques)"""
    
    def __init__(self, headless=False, user_data_dir=None, profile_dir='Default'):
        """Initialize browser with session persistence
        
        Args:
            headless: Run without GUI
            user_data_dir: Chrome user data directory (preserves login sessions)
                         Linux: ~/.config/google-chrome
                         Mac: ~/Library/Application Support/Google/Chrome
                         Windows: C:\\Users\\<user>\\AppData\\Local\\Google\\Chrome\\User Data
            profile_dir: Chrome profile (Default, Profile 1, etc.)
        """
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Try to find Chrome binary (Linux and WSL/Windows)
        chrome_paths = [
            # WSL - Windows Chrome
            '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
            '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe',
            # Linux paths
            '/usr/bin/google-chrome',
            '/usr/bin/google-chrome-stable',
            '/usr/bin/chromium',
            '/usr/bin/chromium-browser',
            '/opt/google/chrome/google-chrome',
            '/snap/bin/chromium',
            '/var/lib/flatpak/exports/bin/com.google.Chrome',
            os.path.expanduser('~/.local/share/flatpak/exports/bin/com.google.Chrome')
        ]
        
        chrome_found = False
        for path in chrome_paths:
            if os.path.exists(path):
                chrome_options.binary_location = path
                print(f"Found Chrome at: {path}")
                chrome_found = True
                break
        
        if not chrome_found:
            print("Warning: Chrome binary not found, relying on system PATH")
        
        # Preserve login sessions (kindleOCRer technique)
        # For WSL, convert user_data_dir to Windows path if needed
        if user_data_dir:
            # Check if running in WSL and user_data_dir looks like Windows path
            if os.path.exists('/mnt/c') and not user_data_dir.startswith('/'):
                # Already a Windows-style path, use as-is
                chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
            elif user_data_dir.startswith('~/.config/google-chrome'):
                # Convert to Windows path for WSL
                # Typical Windows Chrome profile: C:\Users\<username>\AppData\Local\Google\Chrome\User Data
                try:
                    import subprocess
                    username = subprocess.check_output(['cmd.exe', '/c', 'echo', '%USERNAME%'], 
                                                      stderr=subprocess.DEVNULL).decode().strip()
                    windows_profile = f'C:\\Users\\{username}\\AppData\\Local\\Google\\Chrome\\User Data'
                    chrome_options.add_argument(f'--user-data-dir={windows_profile}')
                    print(f"Using Windows Chrome profile: {windows_profile}")
                except:
                    chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
            else:
                chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
            
            chrome_options.add_argument(f'--profile-directory={profile_dir}')
        
        # Use WebDriverManager to automatically handle ChromeDriver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 30)
        self.screenshots = []
    
    def open_url(self, url):
        """Open URL with extended wait time for login"""
        print(f"Opening: {url}")
        self.driver.get(url)
        self.driver.implicitly_wait(30)  # Wait for login if needed
        time.sleep(2)
    
    def fullscreen(self):
        """Set fullscreen mode"""
        self.driver.fullscreen_window()
        time.sleep(1)
    
    def take_screenshot(self, crop_header=0, crop_footer=0):
        """Capture screenshot with optional header/footer cropping
        
        Args:
            crop_header: Pixels to remove from top (remove navigation bars)
            crop_footer: Pixels to remove from bottom (remove footers)
        
        Returns:
            PIL Image object
        """
        screenshot_bytes = self.driver.get_screenshot_as_png()
        screenshot_io = io.BytesIO(screenshot_bytes)
        image = Image.open(screenshot_io)
        
        if crop_header > 0 or crop_footer > 0:
            width, height = image.size
            box = (0, crop_header, width, height - crop_footer)
            image = image.crop(box)
        
        return image
    
    def capture_pages_with_arrow(self, max_pages=100, arrow_key=Keys.RIGHT, 
                                  crop_header=0, crop_footer=0, wait_time=0.5):
        """Capture multiple pages by pressing arrow keys (kindleOCRer technique)
        
        Use cases:
        - E-readers (Kindle Cloud Reader, Google Books)
        - Slideshows (Google Slides, PowerPoint Online)
        - Paginated documents
        - Image galleries
        
        Args:
            max_pages: Maximum pages to capture
            arrow_key: Navigation key (Keys.RIGHT, Keys.DOWN, Keys.PAGE_DOWN)
            crop_header: Pixels to crop from top
            crop_footer: Pixels to crop from bottom
            wait_time: Seconds to wait after each keypress
        
        Returns:
            List of PIL Image objects
        """
        images = []
        actions = ActionChains(self.driver)
        duplicate_count = 0
        
        print(f"Capturing pages (max: {max_pages})...")
        
        for page in range(max_pages):
            # Capture current page
            img = self.take_screenshot(crop_header, crop_footer)
            
            # Check for duplicate (end of document detection)
            if images and self._images_similar(img, images[-1]):
                duplicate_count += 1
                if duplicate_count >= 3:
                    print(f"Reached end at page {page} (3 duplicate pages)")
                    break
            else:
                duplicate_count = 0
                images.append(img)
                print(f"  Captured page {len(images)}")
            
            # Navigate to next page
            actions.send_keys(arrow_key).perform()
            time.sleep(wait_time)
        
        print(f"✓ Captured {len(images)} pages")
        self.screenshots = images
        return images
    
    def capture_pages_with_click(self, next_button_selector, max_pages=100,
                                  crop_header=0, crop_footer=0, wait_time=0.5):
        """Capture pages by clicking 'Next' button
        
        Args:
            next_button_selector: CSS selector for next button
            max_pages: Maximum pages
            crop_header/crop_footer: Cropping pixels
            wait_time: Wait after click
        """
        images = []
        
        print(f"Capturing pages by clicking '{next_button_selector}'...")
        
        for page in range(max_pages):
            img = self.take_screenshot(crop_header, crop_footer)
            images.append(img)
            print(f"  Captured page {page + 1}")
            
            try:
                next_btn = self.driver.find_element(By.CSS_SELECTOR, next_button_selector)
                next_btn.click()
                time.sleep(wait_time)
            except:
                print(f"No more pages at {page + 1}")
                break
        
        print(f"✓ Captured {len(images)} pages")
        self.screenshots = images
        return images
    
    def _images_similar(self, img1, img2, threshold=0.95):
        """Check if two images are similar (for duplicate detection)"""
        if img1.size != img2.size:
            return False
        
        # Resize for faster comparison
        small1 = img1.resize((100, 100))
        small2 = img2.resize((100, 100))
        
        pixels1 = list(small1.getdata())
        pixels2 = list(small2.getdata())
        
        matches = sum(1 for p1, p2 in zip(pixels1, pixels2) if p1 == p2)
        similarity = matches / len(pixels1)
        
        return similarity >= threshold
    
    def save_as_pdf(self, output_file, resolution=100.0):
        """Save screenshots as PDF (kindleOCRer technique)"""
        if not self.screenshots:
            print("No screenshots to save")
            return
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.screenshots[0].save(
            output_file,
            "PDF",
            resolution=resolution,
            save_all=True,
            append_images=self.screenshots[1:] if len(self.screenshots) > 1 else []
        )
        
        print(f"✓ Saved PDF: {output_file} ({len(self.screenshots)} pages)")
    
    def save_as_images(self, output_dir, prefix='page'):
        """Save screenshots as individual PNG files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for idx, img in enumerate(self.screenshots):
            filename = f"{output_dir}/{prefix}_{idx + 1:03d}.png"
            img.save(filename)
        
        print(f"✓ Saved {len(self.screenshots)} images to {output_dir}/")
    
    def get_page_text(self):
        """Extract text from current page"""
        return self.driver.find_element(By.TAG_NAME, 'body').text
    
    def extract_text_from_all_pages(self, arrow_key=Keys.RIGHT, max_pages=100, wait_time=0.5):
        """Extract text by navigating through pages"""
        texts = []
        actions = ActionChains(self.driver)
        
        print(f"Extracting text from pages...")
        
        for page in range(max_pages):
            text = self.get_page_text()
            
            if texts and text == texts[-1]:
                print(f"Reached end at page {page + 1}")
                break
            
            texts.append(text)
            print(f"  Page {page + 1}: {len(text)} chars")
            
            actions.send_keys(arrow_key).perform()
            time.sleep(wait_time)
        
        combined = "\n\n=== PAGE BREAK ===\n\n".join(texts)
        print(f"✓ Extracted {len(texts)} pages, {len(combined)} total chars")
        return combined
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        pyperclip.copy(text)
        print(f"✓ Copied {len(text)} chars to clipboard")
    
    def close(self):
        """Close browser"""
        self.driver.quit()


def main():
    parser = argparse.ArgumentParser(
        description='Capture pages as screenshots or text using Selenium',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture Kindle Cloud Reader book as PDF
  python page_capture_extractor.py https://read.amazon.com \\
    --capture-screenshots --arrow-key right --max-pages 200 \\
    --crop-header 100 --crop-footer 50 --save-pdf book.pdf

  # Capture Google Slides presentation
  python page_capture_extractor.py <slides-url> \\
    --capture-screenshots --arrow-key down --save-pdf slides.pdf

  # Extract text from paginated document
  python page_capture_extractor.py <doc-url> \\
    --extract-text --arrow-key pagedown --output text.txt

  # Use existing Chrome profile (stay logged in)
  python page_capture_extractor.py https://read.amazon.com \\
    --user-data-dir ~/.config/google-chrome --profile-dir Default \\
    --capture-screenshots --save-pdf book.pdf
        """
    )
    
    parser.add_argument('url', help='URL to open')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    parser.add_argument('--user-data-dir', help='Chrome user data directory (preserve login)')
    parser.add_argument('--profile-dir', default='Default', help='Chrome profile name')
    
    # Capture mode
    parser.add_argument('--capture-screenshots', action='store_true', 
                       help='Capture pages as screenshots')
    parser.add_argument('--extract-text', action='store_true',
                       help='Extract text from pages')
    
    # Navigation
    parser.add_argument('--arrow-key', choices=['right', 'down', 'pagedown', 'up', 'left'],
                       default='right', help='Arrow key for navigation')
    parser.add_argument('--next-button', help='CSS selector for next button (alternative to arrow)')
    parser.add_argument('--max-pages', type=int, default=100, help='Maximum pages to capture')
    parser.add_argument('--wait-time', type=float, default=0.5, help='Seconds between pages')
    
    # Screenshot options
    parser.add_argument('--crop-header', type=int, default=0, help='Pixels to crop from top')
    parser.add_argument('--crop-footer', type=int, default=0, help='Pixels to crop from bottom')
    parser.add_argument('--fullscreen', action='store_true', help='Use fullscreen mode')
    
    # Output
    parser.add_argument('--save-pdf', help='Save screenshots as PDF')
    parser.add_argument('--save-images', help='Directory to save PNG images')
    parser.add_argument('--output', '-o', help='Save extracted text to file')
    parser.add_argument('--copy', action='store_true', help='Copy text to clipboard')
    
    args = parser.parse_args()
    
    # Validate
    if not (args.capture_screenshots or args.extract_text):
        parser.error("Must specify --capture-screenshots or --extract-text")
    
    # Map arrow key names
    arrow_keys = {
        'right': Keys.RIGHT,
        'left': Keys.LEFT,
        'down': Keys.DOWN,
        'up': Keys.UP,
        'pagedown': Keys.PAGE_DOWN
    }
    
    extractor = PageCaptureExtractor(
        headless=args.headless,
        user_data_dir=args.user_data_dir,
        profile_dir=args.profile_dir
    )
    
    try:
        extractor.open_url(args.url)
        
        if args.fullscreen:
            extractor.fullscreen()
        
        if args.capture_screenshots:
            if args.next_button:
                extractor.capture_pages_with_click(
                    next_button_selector=args.next_button,
                    max_pages=args.max_pages,
                    crop_header=args.crop_header,
                    crop_footer=args.crop_footer,
                    wait_time=args.wait_time
                )
            else:
                extractor.capture_pages_with_arrow(
                    arrow_key=arrow_keys[args.arrow_key],
                    max_pages=args.max_pages,
                    crop_header=args.crop_header,
                    crop_footer=args.crop_footer,
                    wait_time=args.wait_time
                )
            
            if args.save_pdf:
                extractor.save_as_pdf(args.save_pdf)
            
            if args.save_images:
                extractor.save_as_images(args.save_images)
        
        if args.extract_text:
            text = extractor.extract_text_from_all_pages(
                arrow_key=arrow_keys[args.arrow_key],
                max_pages=args.max_pages,
                wait_time=args.wait_time
            )
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"✓ Saved text to {args.output}")
            
            if args.copy:
                extractor.copy_to_clipboard(text)
    
    finally:
        extractor.close()


if __name__ == '__main__':
    main()
