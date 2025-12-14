#!/usr/bin/env python3
"""
Generate flashcards from book text using AI-assisted extraction
"""

import re
import argparse
from pathlib import Path
import json


def extract_flashcards_from_text(text_file, output_html):
    """Parse text and create flashcards from key concepts
    
    Args:
        text_file: Input text file from OCR
        output_html: Output HTML flashcard file
    """
    print(f"Reading text from: {text_file}")
    text = Path(text_file).read_text(encoding='utf-8')
    
    # Split into sections/chapters
    pages = re.split(r'={60}\s*PAGE \d+\s*={60}', text)
    
    print(f"Found {len(pages)} pages")
    
    # Extract flashcards
    flashcards = []
    
    # Pattern matching for common flashcard formats:
    # 1. Bold terms followed by definitions
    # 2. Question/Answer pairs
    # 3. Lists with terms and descriptions
    # 4. Key concepts in headers
    
    for page_num, page_text in enumerate(pages, 1):
        if not page_text.strip():
            continue
        
        # Method 1: Find terms followed by colons or dashes
        term_pattern = r'^([A-Z][A-Za-z\s]+(?:[\w\s-]+)?)\s*[:\-]\s*(.+?)(?=\n[A-Z][A-Za-z\s]+[:\-]|\n\n|$)'
        matches = re.finditer(term_pattern, page_text, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            # Skip if too short or too long
            if len(term) < 3 or len(term) > 100:
                continue
            if len(definition) < 10 or len(definition) > 500:
                continue
            
            flashcards.append({
                'question': term,
                'answer': definition,
                'page': page_num
            })
        
        # Method 2: Find numbered or bulleted lists
        list_pattern = r'(?:^|\n)[\d\-\•]\s*\*\*([^*]+)\*\*[:\s]+([^\n]+(?:\n(?![\d\-\•])[^\n]+)*)'
        matches = re.finditer(list_pattern, page_text, re.MULTILINE)
        
        for match in matches:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            if len(term) < 3 or len(definition) < 10:
                continue
            
            flashcards.append({
                'question': term,
                'answer': definition,
                'page': page_num
            })
    
    print(f"Extracted {len(flashcards)} flashcards")
    
    # Generate HTML
    html_content = generate_flashcard_html(flashcards, text_file)
    
    # Save HTML
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')
    
    print(f"Saved flashcards to: {output_html}")
    print(f"Open in browser: file://{output_path.absolute()}")


def generate_flashcard_html(flashcards, source_file):
    """Generate interactive HTML flashcard interface"""
    
    # Convert flashcards to JSON for JavaScript
    flashcards_json = json.dumps(flashcards, indent=2)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kindle Book Flashcards</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        
        .container {{
            max-width: 800px;
            width: 100%;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .stats {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            display: flex;
            justify-content: space-around;
            color: white;
            margin-bottom: 20px;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .card-container {{
            perspective: 1000px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            min-height: 400px;
            position: relative;
            transform-style: preserve-3d;
            transition: transform 0.6s;
            cursor: pointer;
        }}
        
        .card.flipped {{
            transform: rotateY(180deg);
        }}
        
        .card-face {{
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            padding: 40px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}
        
        .card-front {{
            background: white;
            border-radius: 15px;
        }}
        
        .card-back {{
            background: #f8f9fa;
            transform: rotateY(180deg);
            border-radius: 15px;
        }}
        
        .card-label {{
            font-size: 0.9em;
            color: #667eea;
            font-weight: bold;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .card-content {{
            font-size: 1.3em;
            line-height: 1.6;
            color: #333;
        }}
        
        .page-indicator {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 20px;
        }}
        
        button {{
            background: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1em;
            font-weight: bold;
            color: #667eea;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        
        button:active {{
            transform: translateY(0);
        }}
        
        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .progress-bar {{
            background: rgba(255, 255, 255, 0.3);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            background: white;
            height: 100%;
            transition: width 0.3s;
        }}
        
        .hint {{
            text-align: center;
            color: white;
            margin-top: 20px;
            opacity: 0.8;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Kindle Book Flashcards</h1>
            <p>Source: {Path(source_file).name}</p>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="currentCard">1</div>
                <div class="stat-label">Current</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="totalCards">{len(flashcards)}</div>
                <div class="stat-label">Total Cards</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="progress">0%</div>
                <div class="stat-label">Progress</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progressBar"></div>
        </div>
        
        <div class="card-container">
            <div class="card" id="flashcard" onclick="flipCard()">
                <div class="card-face card-front">
                    <div class="page-indicator" id="pageIndicator">Page 1</div>
                    <div class="card-label">Question</div>
                    <div class="card-content" id="question"></div>
                </div>
                <div class="card-face card-back">
                    <div class="page-indicator" id="pageIndicatorBack">Page 1</div>
                    <div class="card-label">Answer</div>
                    <div class="card-content" id="answer"></div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="previousCard()" id="prevBtn">← Previous</button>
            <button onclick="shuffleCards()">🔀 Shuffle</button>
            <button onclick="nextCard()" id="nextBtn">Next →</button>
        </div>
        
        <div class="hint">
            Click card to flip • Use arrow keys to navigate • Space to flip • S to view source • ESC to close
        </div>
    </div>
    
    <script>
        const flashcards = {flashcards_json};
        let currentIndex = 0;
        let isFlipped = false;
        
        function updateCard() {{
            const card = flashcards[currentIndex];
            document.getElementById('question').textContent = card.question;
            document.getElementById('answer').textContent = card.answer;
            document.getElementById('pageIndicator').textContent = `Page ${{card.page}}`;
            document.getElementById('pageIndicatorBack').textContent = `Page ${{card.page}}`;
            document.getElementById('currentCard').textContent = currentIndex + 1;
            
            const progress = ((currentIndex + 1) / flashcards.length * 100).toFixed(0);
            document.getElementById('progress').textContent = `${{progress}}%`;
            document.getElementById('progressBar').style.width = `${{progress}}%`;
            
            // Update button states
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === flashcards.length - 1;
            
            // Reset flip
            if (isFlipped) {{
                flipCard();
            }}
        }}
        
        function flipCard() {{
            const card = document.getElementById('flashcard');
            card.classList.toggle('flipped');
            isFlipped = !isFlipped;
        }}
        
        function nextCard() {{
            if (currentIndex < flashcards.length - 1) {{
                currentIndex++;
                updateCard();
            }}
        }}
        
        function previousCard() {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateCard();
            }}
        }}
        
        function shuffleCards() {{
            for (let i = flashcards.length - 1; i > 0; i--) {{
                const j = Math.floor(Math.random() * (i + 1));
                [flashcards[i], flashcards[j]] = [flashcards[j], flashcards[i]];
            }}
            currentIndex = 0;
            updateCard();
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') previousCard();
            if (e.key === 'ArrowRight') nextCard();
            if (e.key === ' ') {{
                e.preventDefault();
                flipCard();
            }}
        }});
        
        // Initialize
        updateCard();
    </script>
</body>
</html>'''
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Generate flashcards from extracted text')
    parser.add_argument('--input', default='output/kindle_text.txt',
                       help='Input text file (default: output/kindle_text.txt)')
    parser.add_argument('--output', default='output/kindle_flashcards.html',
                       help='Output HTML file (default: output/kindle_flashcards.html)')
    
    args = parser.parse_args()
    
    extract_flashcards_from_text(args.input, args.output)


if __name__ == '__main__':
    main()
