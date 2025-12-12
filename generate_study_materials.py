#!/usr/bin/env python3
"""
Generate interactive study materials from OCR processed ML Engineering content
Creates: Interactive reader, flashcards, and practice quizzes
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class StudyMaterialGenerator:
    """Generate study materials from OCR processed content"""
    
    def __init__(self, ocr_json_path: str, output_dir: str = 'ml_study_materials'):
        self.ocr_data = self.load_ocr_data(ocr_json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_ocr_data(self, json_path: str) -> Dict:
        """Load consolidated OCR JSON data"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def escape_js_string(self, s: str) -> str:
        """Escape string for JavaScript"""
        # Replace backslashes first
        s = s.replace('\\', '\\\\')
        # Then other special characters
        s = s.replace('"', '\\"')
        s = s.replace("'", "\\'")
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        s = s.replace('</', '<\\/')
        return s
    
    def extract_sections(self) -> List[Dict]:
        """Extract pages from OCR content - each page is a section"""
        sections = []
        
        for chunk in self.ocr_data.get('chunks', []):
            chunk_data = chunk.get('data', {})
            pages = chunk_data.get('pages', [])
            chunk_num = chunk.get('chunk_number', 0)
            
            for page in pages:
                text = page.get('text', '').strip()
                page_num = page.get('page_number', 0)
                
                if not text:
                    continue
                
                # Calculate actual page number across all chunks
                # Each chunk has 20 pages, chunk 1 starts at page 1
                actual_page = ((chunk_num - 1) * 20) + page_num
                
                # Extract title from first line or use page number
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if lines:
                    # Use first non-empty line as title (up to 100 chars)
                    title = lines[0][:100]
                    if len(lines[0]) > 100:
                        title += '...'
                else:
                    title = f"Page {actual_page}"
                
                sections.append({
                    'title': title,
                    'content': text,
                    'page': actual_page,
                    'chunk': chunk_num
                })
        
        return sections
    
    def json_to_base64(self, data) -> str:
        """Convert JSON data to base64 to avoid escaping issues"""
        import base64
        json_str = json.dumps(data, ensure_ascii=False)
        return base64.b64encode(json_str.encode('utf-8')).decode('ascii')
    
    def generate_interactive_reader(self, sections: List[Dict]):
        """Generate HTML interactive reader with navigation"""
        
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Engineering on AWS - Interactive Reader</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Amazon Ember', 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #232f3e 0%, #1a242f 100%);
            color: #232f3e;
            min-height: 100vh;
        }
        
        .header {
            background: #232f3e;
            color: #fff;
            padding: 1.5rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
        
        .header .subtitle {
            color: #ff9900;
            font-size: 0.9rem;
        }
        
        .container {
            display: flex;
            max-width: 1400px;
            margin: 0 auto;
            min-height: calc(100vh - 100px);
        }
        
        .sidebar {
            width: 300px;
            background: #fff;
            padding: 1.5rem;
            overflow-y: auto;
            max-height: calc(100vh - 100px);
            box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        }
        
        .sidebar h2 {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #232f3e;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #ff9900;
        }
        
        .section-list {
            list-style: none;
        }
        
        .section-item {
            padding: 0.8rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.3s;
            border-left: 3px solid transparent;
        }
        
        .section-item:hover {
            background: #f7f7f7;
            border-left-color: #ff9900;
        }
        
        .section-item.active {
            background: #fef5e7;
            border-left-color: #ff9900;
            font-weight: 600;
        }
        
        .section-title {
            font-size: 0.9rem;
            color: #232f3e;
            margin-bottom: 0.3rem;
        }
        
        .section-meta {
            font-size: 0.75rem;
            color: #666;
        }
        
        .content-area {
            flex: 1;
            background: #fff;
            margin: 1rem;
            padding: 2rem 3rem;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow-y: auto;
            max-height: calc(100vh - 120px);
        }
        
        .content-header {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #ff9900;
        }
        
        .content-header h2 {
            font-size: 2rem;
            color: #232f3e;
            margin-bottom: 0.5rem;
        }
        
        .content-meta {
            color: #666;
            font-size: 0.9rem;
        }
        
        .content-body {
            line-height: 1.8;
            font-size: 1rem;
            color: #333;
        }
        
        .content-body p {
            margin-bottom: 1rem;
        }
        
        .navigation {
            display: flex;
            justify-content: space-between;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
        }
        
        .nav-button {
            padding: 0.8rem 1.5rem;
            background: #ff9900;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s;
        }
        
        .nav-button:hover {
            background: #ec7211;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .nav-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .search-box {
            width: 100%;
            padding: 0.8rem;
            margin-bottom: 1rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        
        .search-box:focus {
            outline: none;
            border-color: #ff9900;
        }
        
        .progress-bar {
            height: 4px;
            background: #f0f0f0;
            margin-bottom: 1rem;
        }
        
        .progress-fill {
            height: 100%;
            background: #ff9900;
            transition: width 0.3s;
        }
        
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
                max-height: 300px;
            }
            
            .content-area {
                margin: 0.5rem;
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎓 ML Engineering on AWS - Interactive Study Guide</h1>
        <div class="subtitle">520 pages • 26 chapters • MLA-C01 Exam Preparation</div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <input type="text" id="searchBox" class="search-box" placeholder="Search sections...">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <h2>Table of Contents</h2>
            <ul class="section-list" id="sectionList"></ul>
        </div>
        
        <div class="content-area">
            <div class="content-header">
                <h2 id="contentTitle">Select a section to begin</h2>
                <div class="content-meta" id="contentMeta"></div>
            </div>
            <div class="content-body" id="contentBody">
                <p>Welcome to the ML Engineering on AWS interactive study guide. Use the sidebar to navigate through the content.</p>
            </div>
            <div class="navigation">
                <button class="nav-button" id="prevButton" onclick="navigatePrev()">← Previous</button>
                <button class="nav-button" id="nextButton" onclick="navigateNext()">Next →</button>
            </div>
        </div>
    </div>
    
    <script>
        // Section data  
        const sections = JSON.parse(atob('''' + self.json_to_base64(sections) + ''''));
        
        let currentSectionIndex = 0;
        let filteredSections = [...sections];
        
        // Initialize
        function init() {
            renderSectionList();
            if (sections.length > 0) {
                loadSection(0);
            }
            updateProgress();
        }
        
        // Render section list
        function renderSectionList() {
            const listEl = document.getElementById('sectionList');
            listEl.innerHTML = '';
            
            filteredSections.forEach((section, index) => {
                const li = document.createElement('li');
                li.className = 'section-item';
                if (index === currentSectionIndex) {
                    li.classList.add('active');
                }
                
                li.innerHTML = `
                    <div class="section-title">${section.title}</div>
                    <div class="section-meta">Page ${section.page} • Chunk ${section.chunk}</div>
                `;
                
                li.onclick = () => loadSection(index);
                listEl.appendChild(li);
            });
        }
        
        // Load section content
        function loadSection(index) {
            if (index < 0 || index >= filteredSections.length) return;
            
            currentSectionIndex = index;
            const section = filteredSections[index];
            
            document.getElementById('contentTitle').textContent = section.title;
            document.getElementById('contentMeta').textContent = 
                `Page ${section.page} • Chunk ${section.chunk}`;
            
            // Format content
            const formattedContent = section.content
                .split('\n\n')
                .map(para => `<p>${para}</p>`)
                .join('');
            
            document.getElementById('contentBody').innerHTML = formattedContent || '<p>No content available.</p>';
            
            // Update UI
            renderSectionList();
            updateNavigationButtons();
            updateProgress();
            
            // Scroll to top
            document.querySelector('.content-area').scrollTop = 0;
        }
        
        // Navigation
        function navigatePrev() {
            if (currentSectionIndex > 0) {
                loadSection(currentSectionIndex - 1);
            }
        }
        
        function navigateNext() {
            if (currentSectionIndex < filteredSections.length - 1) {
                loadSection(currentSectionIndex + 1);
            }
        }
        
        function updateNavigationButtons() {
            document.getElementById('prevButton').disabled = currentSectionIndex === 0;
            document.getElementById('nextButton').disabled = 
                currentSectionIndex === filteredSections.length - 1;
        }
        
        // Search
        document.getElementById('searchBox').addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            
            if (!query) {
                filteredSections = [...sections];
            } else {
                filteredSections = sections.filter(section => 
                    section.title.toLowerCase().includes(query) ||
                    section.content.toLowerCase().includes(query)
                );
            }
            
            currentSectionIndex = 0;
            renderSectionList();
            if (filteredSections.length > 0) {
                loadSection(0);
            }
        });
        
        // Progress tracking
        function updateProgress() {
            const progress = ((currentSectionIndex + 1) / filteredSections.length) * 100;
            document.getElementById('progressFill').style.width = progress + '%';
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') navigatePrev();
            if (e.key === 'ArrowRight') navigateNext();
        });
        
        // Initialize on load
        init();
    </script>
</body>
</html>'''
        
        output_file = self.output_dir / 'ml_engineering_reader.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Interactive reader generated: {output_file}")
        print(f"  Sections: {len(sections)}")
    
    def generate_all_materials(self):
        """Generate all study materials"""
        print("Generating study materials from OCR data...")
        print(f"Total chunks: {len(self.ocr_data.get('chunks', []))}")
        print(f"Total pages: {self.ocr_data.get('total_pages', 0)}")
        
        # Extract sections
        print("\nExtracting sections...")
        sections = self.extract_sections()
        print(f"✓ Extracted {len(sections)} sections")
        
        # Generate interactive reader
        print("\nGenerating interactive reader...")
        self.generate_interactive_reader(sections)
        
        print(f"\n{'='*60}")
        print("Study materials generation complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate study materials from OCR data')
    parser.add_argument('--input', 
                        default='processed_materials/ml_engineering_full/ml_engineering_full_ocr.json',
                        help='Input OCR JSON file')
    parser.add_argument('--output', 
                        default='ml_study_materials',
                        help='Output directory for study materials')
    
    args = parser.parse_args()
    
    generator = StudyMaterialGenerator(args.input, args.output)
    generator.generate_all_materials()


if __name__ == '__main__':
    main()
