#!/usr/bin/env python3
"""
Generate flashcards from study_guide_readable.md
Creates interactive HTML flashcards for exam preparation
"""

import re
import json
from pathlib import Path


def parse_study_guide(file_path):
    """Parse study guide and extract flashcard content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    flashcards = []
    
    # Extract key concepts (heading + first paragraph pattern)
    sections = re.split(r'\n## ', content)
    
    for section in sections[1:]:  # Skip intro
        lines = section.split('\n')
        module_title = lines[0].strip()
        
        # Find subsections (###)
        subsections = re.split(r'\n### ', section)
        
        for subsection in subsections[1:]:
            sub_lines = subsection.split('\n')
            topic = sub_lines[0].strip()
            
            # Extract paragraphs
            paragraphs = []
            current_para = []
            
            for line in sub_lines[1:]:
                line = line.strip()
                if line.startswith('**') and line.endswith('**'):
                    # Bold header - potential card
                    if current_para:
                        paragraphs.append(' '.join(current_para))
                        current_para = []
                    paragraphs.append(line)
                elif line and not line.startswith('#'):
                    current_para.append(line)
                elif current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            
            # Create flashcards from bold headers
            for i, para in enumerate(paragraphs):
                if para.startswith('**') and para.endswith('**'):
                    question = para.strip('*')
                    # Next paragraph is the answer
                    if i + 1 < len(paragraphs):
                        answer = paragraphs[i + 1]
                        flashcards.append({
                            'module': module_title,
                            'topic': topic,
                            'question': question,
                            'answer': answer
                        })
    
    return flashcards


def extract_definition_cards(content):
    """Extract definition-style flashcards"""
    cards = []
    
    # Pattern: **Term** - Definition
    pattern = r'\*\*([^*]+)\*\*\s*[-–—]\s*([^.]+\.)'
    matches = re.findall(pattern, content)
    
    for term, definition in matches:
        cards.append({
            'question': f"What is {term}?",
            'answer': definition.strip()
        })
    
    return cards


def extract_service_cards(content):
    """Extract AWS service description cards"""
    cards = []
    
    # Pattern: **Service Name** - Description. Features: ...
    pattern = r'\*\*([^*]+)\*\*\s*[-–—]\s*([^.]+\.(?:[^.]+\.)*)'
    
    sections = content.split('\n## ')
    
    for section in sections:
        if 'AWS' in section or 'SageMaker' in section or 'Amazon' in section:
            matches = re.findall(pattern, section)
            for service, description in matches:
                if len(description) > 50:  # Meaningful descriptions only
                    cards.append({
                        'question': f"Describe {service}",
                        'answer': description.strip()
                    })
    
    return cards


def generate_html_flashcards(flashcards, output_file):
    """Generate interactive HTML flashcard interface"""
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AWS MLA-C01 Flashcards</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .stats {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px 30px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .stat-item {
            font-size: 1.1em;
        }
        
        .card-container {
            perspective: 1000px;
            width: 100%;
            max-width: 800px;
            height: 500px;
            margin-bottom: 30px;
        }
        
        .flashcard {
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            transition: transform 0.6s;
            cursor: pointer;
        }
        
        .flashcard.flipped {
            transform: rotateY(180deg);
        }
        
        .card-face {
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            border-radius: 20px;
            padding: 40px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .card-front {
            background: white;
        }
        
        .card-back {
            background: #f8f9fa;
            transform: rotateY(180deg);
        }
        
        .card-label {
            position: absolute;
            top: 20px;
            left: 20px;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .card-content {
            font-size: 1.4em;
            line-height: 1.6;
            color: #333;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .card-back .card-content {
            font-size: 1.1em;
            text-align: left;
        }
        
        .card-meta {
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            font-size: 0.9em;
            color: #666;
            text-align: left;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 20px;
        }
        
        button {
            padding: 15px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .btn-primary {
            background: #28a745;
            color: white;
        }
        
        .btn-primary:hover {
            background: #218838;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: #dc3545;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #c82333;
            transform: translateY(-2px);
        }
        
        .btn-nav {
            background: white;
            color: #667eea;
        }
        
        .btn-nav:hover {
            background: #f8f9fa;
            transform: translateY(-2px);
        }
        
        .btn-shuffle {
            background: #ffc107;
            color: #333;
        }
        
        .btn-shuffle:hover {
            background: #e0a800;
            transform: translateY(-2px);
        }
        
        .filter-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            width: 100%;
            max-width: 800px;
        }
        
        .filter-section h3 {
            margin-bottom: 15px;
            color: #667eea;
        }
        
        .filter-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .filter-btn {
            padding: 8px 16px;
            background: #e9ecef;
            border: 2px solid transparent;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
        }
        
        .filter-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .progress-bar {
            width: 100%;
            max-width: 800px;
            height: 10px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        
        .progress-fill {
            height: 100%;
            background: #28a745;
            transition: width 0.3s;
        }
        
        @media (max-width: 768px) {
            .card-container {
                height: 400px;
            }
            
            .card-content {
                font-size: 1.1em;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AWS MLA-C01 Flashcards</h1>
        <p>Machine Learning Engineer - Associate Certification</p>
    </div>
    
    <div class="stats">
        <div class="stat-item">📚 Total: <span id="total-cards">0</span></div>
        <div class="stat-item">✅ Mastered: <span id="mastered-count">0</span></div>
        <div class="stat-item">📖 Current: <span id="current-card">0</span></div>
    </div>
    
    <div class="progress-bar">
        <div class="progress-fill" id="progress"></div>
    </div>
    
    <div class="filter-section">
        <h3>Filter by Module</h3>
        <div class="filter-buttons" id="module-filters"></div>
    </div>
    
    <div class="card-container">
        <div class="flashcard" id="flashcard" onclick="flipCard()">
            <div class="card-face card-front">
                <div class="card-label">Question</div>
                <div class="card-content" id="question"></div>
                <div class="card-meta" id="meta-front"></div>
            </div>
            <div class="card-face card-back">
                <div class="card-label">Answer</div>
                <div class="card-content" id="answer"></div>
                <div class="card-meta" id="meta-back"></div>
            </div>
        </div>
    </div>
    
    <div class="controls">
        <button class="btn-nav" onclick="previousCard()">← Previous</button>
        <button class="btn-primary" onclick="markMastered()">✓ Know It</button>
        <button class="btn-secondary" onclick="markNeedReview()">✗ Review</button>
        <button class="btn-nav" onclick="nextCard()">Next →</button>
    </div>
    
    <div class="controls">
        <button class="btn-shuffle" onclick="shuffleCards()">🔀 Shuffle</button>
        <button class="btn-nav" onclick="resetProgress()">↻ Reset Progress</button>
    </div>

    <script>
        const allCards = """ + json.dumps(flashcards, indent=2) + """;
        
        let currentIndex = 0;
        let filteredCards = [...allCards];
        let masteredCards = new Set();
        let needReviewCards = new Set();
        
        // Load progress from localStorage
        function loadProgress() {
            const saved = localStorage.getItem('aws-mla-flashcards-progress');
            if (saved) {
                const data = JSON.parse(saved);
                masteredCards = new Set(data.mastered || []);
                needReviewCards = new Set(data.needReview || []);
                currentIndex = data.currentIndex || 0;
            }
        }
        
        function saveProgress() {
            localStorage.setItem('aws-mla-flashcards-progress', JSON.stringify({
                mastered: Array.from(masteredCards),
                needReview: Array.from(needReviewCards),
                currentIndex: currentIndex
            }));
        }
        
        function initFilters() {
            const modules = [...new Set(allCards.map(c => c.module))];
            const container = document.getElementById('module-filters');
            
            const allBtn = document.createElement('button');
            allBtn.className = 'filter-btn active';
            allBtn.textContent = 'All Modules';
            allBtn.onclick = () => filterByModule(null);
            container.appendChild(allBtn);
            
            modules.forEach(module => {
                const btn = document.createElement('button');
                btn.className = 'filter-btn';
                btn.textContent = module;
                btn.onclick = () => filterByModule(module);
                container.appendChild(btn);
            });
        }
        
        function filterByModule(module) {
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            if (module) {
                filteredCards = allCards.filter(c => c.module === module);
            } else {
                filteredCards = [...allCards];
            }
            
            currentIndex = 0;
            showCard();
        }
        
        function showCard() {
            if (filteredCards.length === 0) return;
            
            const card = filteredCards[currentIndex];
            const flashcard = document.getElementById('flashcard');
            flashcard.classList.remove('flipped');
            
            document.getElementById('question').textContent = card.question;
            document.getElementById('answer').textContent = card.answer;
            document.getElementById('meta-front').textContent = `${card.module} › ${card.topic}`;
            document.getElementById('meta-back').textContent = `${card.module} › ${card.topic}`;
            
            updateStats();
        }
        
        function updateStats() {
            document.getElementById('total-cards').textContent = filteredCards.length;
            document.getElementById('mastered-count').textContent = masteredCards.size;
            document.getElementById('current-card').textContent = currentIndex + 1;
            
            const progress = (masteredCards.size / allCards.length) * 100;
            document.getElementById('progress').style.width = progress + '%';
        }
        
        function flipCard() {
            document.getElementById('flashcard').classList.toggle('flipped');
        }
        
        function nextCard() {
            currentIndex = (currentIndex + 1) % filteredCards.length;
            showCard();
            saveProgress();
        }
        
        function previousCard() {
            currentIndex = (currentIndex - 1 + filteredCards.length) % filteredCards.length;
            showCard();
            saveProgress();
        }
        
        function markMastered() {
            const cardId = currentIndex;
            masteredCards.add(cardId);
            needReviewCards.delete(cardId);
            nextCard();
        }
        
        function markNeedReview() {
            const cardId = currentIndex;
            needReviewCards.add(cardId);
            masteredCards.delete(cardId);
            nextCard();
        }
        
        function shuffleCards() {
            filteredCards.sort(() => Math.random() - 0.5);
            currentIndex = 0;
            showCard();
        }
        
        function resetProgress() {
            if (confirm('Reset all progress?')) {
                masteredCards.clear();
                needReviewCards.clear();
                currentIndex = 0;
                localStorage.removeItem('aws-mla-flashcards-progress');
                showCard();
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === 'n') nextCard();
            if (e.key === 'ArrowLeft' || e.key === 'p') previousCard();
            if (e.key === ' ') { e.preventDefault(); flipCard(); }
            if (e.key === 'k') markMastered();
            if (e.key === 'r') markNeedReview();
            if (e.key === 's') shuffleCards();
        });
        
        // Initialize
        loadProgress();
        initFilters();
        showCard();
    </script>
</body>
</html>"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Generated {len(flashcards)} flashcards: {output_file}")


def main():
    # Read study guide
    study_guide_path = Path(__file__).parent / 'study_guide_readable.md'
    
    if not study_guide_path.exists():
        print(f"Error: {study_guide_path} not found")
        return
    
    print("Parsing study guide...")
    flashcards = parse_study_guide(study_guide_path)
    
    print(f"Extracted {len(flashcards)} flashcards")
    
    # Generate HTML
    output_file = Path(__file__).parent / 'flashcards_mla.html'
    generate_html_flashcards(flashcards, output_file)
    
    print(f"\n✓ Flashcards ready!")
    print(f"  Open: {output_file}")
    print(f"  Or visit: https://aiegoo.github.io/aws-cert/flashcards_mla.html (after deployment)")
    print(f"\nKeyboard shortcuts:")
    print(f"  Space - Flip card")
    print(f"  → or N - Next card")
    print(f"  ← or P - Previous card")
    print(f"  K - Mark as mastered")
    print(f"  R - Mark for review")
    print(f"  S - Shuffle")


if __name__ == '__main__':
    main()
