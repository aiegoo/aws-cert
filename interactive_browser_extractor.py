#!/usr/bin/env python3
"""
Interactive browser text extractor with GUI controls
Allows manual navigation and text extraction
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pyperclip
import time


class InteractiveBrowserExtractor:
    """Interactive browser for text extraction with user control"""
    
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--start-maximized')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        print("Browser opened. Navigate to your page manually.")
    
    def copy_current_page(self):
        """Copy all visible text from current page"""
        text = self.driver.find_element(By.TAG_NAME, 'body').text
        pyperclip.copy(text)
        print(f"✓ Copied {len(text)} characters to clipboard")
        return text
    
    def copy_element_by_id(self, element_id):
        """Copy text from element with specific ID"""
        try:
            element = self.driver.find_element(By.ID, element_id)
            text = element.text
            pyperclip.copy(text)
            print(f"✓ Copied element '{element_id}': {len(text)} characters")
            return text
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def copy_element_by_class(self, class_name):
        """Copy text from all elements with specific class"""
        try:
            elements = self.driver.find_elements(By.CLASS_NAME, class_name)
            text = '\n\n'.join([elem.text for elem in elements if elem.text.strip()])
            pyperclip.copy(text)
            print(f"✓ Copied {len(elements)} elements with class '{class_name}': {len(text)} characters")
            return text
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def copy_element_by_css(self, css_selector):
        """Copy text from element matching CSS selector"""
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, css_selector)
            text = element.text
            pyperclip.copy(text)
            print(f"✓ Copied element '{css_selector}': {len(text)} characters")
            return text
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def copy_all_headings(self):
        """Copy all headings (h1-h6) from the page"""
        headings = []
        for level in range(1, 7):
            elements = self.driver.find_elements(By.TAG_NAME, f'h{level}')
            for elem in elements:
                if elem.text.strip():
                    headings.append(f"{'#' * level} {elem.text}")
        
        text = '\n'.join(headings)
        pyperclip.copy(text)
        print(f"✓ Copied {len(headings)} headings to clipboard")
        return text
    
    def copy_main_content(self):
        """Copy main content (article, main tags, or largest text block)"""
        selectors = ['article', 'main', '[role="main"]', '.content', '#content']
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                text = element.text
                if text.strip():
                    pyperclip.copy(text)
                    print(f"✓ Copied main content from '{selector}': {len(text)} characters")
                    return text
            except:
                continue
        
        # Fallback to body
        return self.copy_current_page()
    
    def highlight_element(self, css_selector):
        """Highlight an element on the page for visual confirmation"""
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, css_selector)
            self.driver.execute_script(
                "arguments[0].style.border='3px solid red'",
                element
            )
            print(f"✓ Highlighted element: {css_selector}")
        except Exception as e:
            print(f"✗ Error highlighting: {e}")
    
    def save_page_source(self, filename='page_source.html'):
        """Save the full HTML source of current page"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.driver.page_source)
        print(f"✓ Saved page source to {filename}")
    
    def get_current_url(self):
        """Get current page URL"""
        url = self.driver.current_url
        print(f"Current URL: {url}")
        return url
    
    def interactive_mode(self):
        """Run in interactive mode with commands"""
        print("\n" + "="*60)
        print("INTERACTIVE BROWSER TEXT EXTRACTOR")
        print("="*60)
        print("\nCommands:")
        print("  copy / c          - Copy all page text")
        print("  main / m          - Copy main content")
        print("  headings / h      - Copy all headings")
        print("  id <id>           - Copy element by ID")
        print("  class <name>      - Copy elements by class")
        print("  css <selector>    - Copy element by CSS selector")
        print("  highlight <sel>   - Highlight element")
        print("  url               - Show current URL")
        print("  save              - Save page source")
        print("  goto <url>        - Navigate to URL")
        print("  back              - Go back")
        print("  forward           - Go forward")
        print("  refresh           - Refresh page")
        print("  quit / q          - Exit")
        print("\n" + "="*60 + "\n")
        
        while True:
            try:
                cmd = input(">>> ").strip().lower()
                
                if not cmd:
                    continue
                
                if cmd in ['quit', 'q', 'exit']:
                    break
                
                elif cmd in ['copy', 'c']:
                    self.copy_current_page()
                
                elif cmd in ['main', 'm']:
                    self.copy_main_content()
                
                elif cmd in ['headings', 'h']:
                    self.copy_all_headings()
                
                elif cmd.startswith('id '):
                    element_id = cmd[3:].strip()
                    self.copy_element_by_id(element_id)
                
                elif cmd.startswith('class '):
                    class_name = cmd[6:].strip()
                    self.copy_element_by_class(class_name)
                
                elif cmd.startswith('css '):
                    selector = cmd[4:].strip()
                    self.copy_element_by_css(selector)
                
                elif cmd.startswith('highlight '):
                    selector = cmd[10:].strip()
                    self.highlight_element(selector)
                
                elif cmd == 'url':
                    self.get_current_url()
                
                elif cmd == 'save':
                    self.save_page_source()
                
                elif cmd.startswith('goto '):
                    url = cmd[5:].strip()
                    self.driver.get(url)
                    print(f"✓ Navigated to {url}")
                
                elif cmd == 'back':
                    self.driver.back()
                    print("✓ Navigated back")
                
                elif cmd == 'forward':
                    self.driver.forward()
                    print("✓ Navigated forward")
                
                elif cmd == 'refresh':
                    self.driver.refresh()
                    print("✓ Page refreshed")
                
                else:
                    print(f"✗ Unknown command: {cmd}")
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
        
        self.close()
    
    def close(self):
        """Close the browser"""
        self.driver.quit()
        print("\nBrowser closed. Goodbye!")


def main():
    """Run the interactive browser extractor"""
    extractor = InteractiveBrowserExtractor()
    
    try:
        # Ask for initial URL
        url = input("Enter starting URL (or press Enter to skip): ").strip()
        if url:
            extractor.driver.get(url)
            print(f"Navigated to {url}\n")
        
        # Start interactive mode
        extractor.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        extractor.close()
    except Exception as e:
        print(f"Error: {e}")
        extractor.close()


if __name__ == '__main__':
    main()
