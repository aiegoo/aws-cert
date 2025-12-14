#!/usr/bin/env python3
"""
Selenium-based browser text extractor
Reads and copies text content from web pages
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pyperclip
import time
import argparse


class BrowserTextExtractor:
    def __init__(self, headless=False):
        """Initialize the browser with options"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
    
    def open_url(self, url):
        """Open a URL in the browser"""
        print(f"Opening URL: {url}")
        self.driver.get(url)
        time.sleep(2)  # Wait for page load
    
    def get_page_text(self):
        """Get all text content from the current page"""
        body = self.driver.find_element(By.TAG_NAME, 'body')
        return body.text
    
    def get_element_text(self, selector, by=By.CSS_SELECTOR):
        """Get text from a specific element"""
        try:
            element = self.wait.until(
                EC.presence_of_element_located((by, selector))
            )
            return element.text
        except Exception as e:
            print(f"Error finding element {selector}: {e}")
            return None
    
    def get_multiple_elements_text(self, selector, by=By.CSS_SELECTOR):
        """Get text from multiple elements"""
        try:
            elements = self.driver.find_elements(by, selector)
            return [elem.text for elem in elements if elem.text.strip()]
        except Exception as e:
            print(f"Error finding elements {selector}: {e}")
            return []
    
    def copy_to_clipboard(self, text):
        """Copy text to system clipboard"""
        pyperclip.copy(text)
        print(f"Copied {len(text)} characters to clipboard")
    
    def select_all_and_copy(self):
        """Select all page content and copy to clipboard"""
        body = self.driver.find_element(By.TAG_NAME, 'body')
        
        # Use keyboard shortcuts
        actions = ActionChains(self.driver)
        actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
        time.sleep(0.5)
        actions.key_down(Keys.CONTROL).send_keys('c').key_up(Keys.CONTROL).perform()
        time.sleep(0.5)
        
        print("Selected all and copied to clipboard")
    
    def extract_headings(self):
        """Extract all headings (h1-h6) from the page"""
        headings = {}
        for level in range(1, 7):
            tag = f'h{level}'
            elements = self.driver.find_elements(By.TAG_NAME, tag)
            headings[tag] = [elem.text for elem in elements if elem.text.strip()]
        return headings
    
    def extract_paragraphs(self):
        """Extract all paragraph text"""
        paragraphs = self.driver.find_elements(By.TAG_NAME, 'p')
        return [p.text for p in paragraphs if p.text.strip()]
    
    def extract_links(self):
        """Extract all links with their text and href"""
        links = self.driver.find_elements(By.TAG_NAME, 'a')
        return [{'text': link.text, 'href': link.get_attribute('href')} 
                for link in links if link.get_attribute('href')]
    
    def extract_tables(self):
        """Extract table data"""
        tables = []
        table_elements = self.driver.find_elements(By.TAG_NAME, 'table')
        
        for table in table_elements:
            rows = table.find_elements(By.TAG_NAME, 'tr')
            table_data = []
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, 'td')
                if not cells:
                    cells = row.find_elements(By.TAG_NAME, 'th')
                table_data.append([cell.text for cell in cells])
            tables.append(table_data)
        
        return tables
    
    def extract_code_blocks(self):
        """Extract code blocks (pre, code tags)"""
        code_blocks = []
        
        # Extract <pre> blocks
        pre_elements = self.driver.find_elements(By.TAG_NAME, 'pre')
        code_blocks.extend([pre.text for pre in pre_elements if pre.text.strip()])
        
        # Extract <code> blocks not inside <pre>
        code_elements = self.driver.find_elements(By.TAG_NAME, 'code')
        for code in code_elements:
            if code.find_elements(By.XPATH, './ancestor::pre'):
                continue
            if code.text.strip():
                code_blocks.append(code.text)
        
        return code_blocks
    
    def extract_list_items(self):
        """Extract list items (ul, ol)"""
        list_items = []
        
        # Unordered lists
        ul_elements = self.driver.find_elements(By.TAG_NAME, 'ul')
        for ul in ul_elements:
            items = ul.find_elements(By.TAG_NAME, 'li')
            list_items.append({
                'type': 'unordered',
                'items': [item.text for item in items if item.text.strip()]
            })
        
        # Ordered lists
        ol_elements = self.driver.find_elements(By.TAG_NAME, 'ol')
        for ol in ol_elements:
            items = ol.find_elements(By.TAG_NAME, 'li')
            list_items.append({
                'type': 'ordered',
                'items': [item.text for item in items if item.text.strip()]
            })
        
        return list_items
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the page (for lazy-loaded content)"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        print("Scrolled to bottom of page")
    
    def save_to_file(self, content, filename):
        """Save extracted content to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved content to {filename}")
    
    def close(self):
        """Close the browser"""
        self.driver.quit()
        print("Browser closed")


def main():
    parser = argparse.ArgumentParser(description='Extract text from web pages using Selenium')
    parser.add_argument('url', help='URL to extract text from')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--output', '-o', help='Output file to save extracted text')
    parser.add_argument('--selector', '-s', help='CSS selector to extract specific element')
    parser.add_argument('--full-page', action='store_true', help='Extract full page text')
    parser.add_argument('--headings', action='store_true', help='Extract only headings')
    parser.add_argument('--paragraphs', action='store_true', help='Extract only paragraphs')
    parser.add_argument('--code', action='store_true', help='Extract code blocks')
    parser.add_argument('--links', action='store_true', help='Extract links')
    parser.add_argument('--tables', action='store_true', help='Extract tables')
    parser.add_argument('--copy', action='store_true', help='Copy to clipboard')
    parser.add_argument('--scroll', action='store_true', help='Scroll to bottom before extracting')
    
    args = parser.parse_args()
    
    extractor = BrowserTextExtractor(headless=args.headless)
    
    try:
        extractor.open_url(args.url)
        
        if args.scroll:
            extractor.scroll_to_bottom()
        
        content = ""
        
        if args.selector:
            # Extract specific element
            content = extractor.get_element_text(args.selector)
            print(f"\n=== Element Text ({args.selector}) ===")
            print(content)
        
        elif args.headings:
            # Extract headings
            headings = extractor.extract_headings()
            for tag, texts in headings.items():
                if texts:
                    content += f"\n=== {tag.upper()} ===\n"
                    content += '\n'.join(texts) + '\n'
            print(content)
        
        elif args.paragraphs:
            # Extract paragraphs
            paragraphs = extractor.extract_paragraphs()
            content = '\n\n'.join(paragraphs)
            print(f"\n=== Paragraphs ({len(paragraphs)}) ===")
            print(content)
        
        elif args.code:
            # Extract code blocks
            code_blocks = extractor.extract_code_blocks()
            for i, code in enumerate(code_blocks, 1):
                content += f"\n=== Code Block {i} ===\n{code}\n"
            print(content)
        
        elif args.links:
            # Extract links
            links = extractor.extract_links()
            for link in links:
                content += f"{link['text']}: {link['href']}\n"
            print(f"\n=== Links ({len(links)}) ===")
            print(content)
        
        elif args.tables:
            # Extract tables
            tables = extractor.extract_tables()
            for i, table in enumerate(tables, 1):
                content += f"\n=== Table {i} ===\n"
                for row in table:
                    content += ' | '.join(row) + '\n'
            print(content)
        
        else:
            # Extract full page text
            content = extractor.get_page_text()
            print(f"\n=== Full Page Text ({len(content)} chars) ===")
            print(content[:500] + "..." if len(content) > 500 else content)
        
        # Copy to clipboard if requested
        if args.copy:
            extractor.copy_to_clipboard(content)
        
        # Save to file if specified
        if args.output:
            extractor.save_to_file(content, args.output)
    
    finally:
        extractor.close()


if __name__ == '__main__':
    main()
