# CMS Workflow Guide: Adding Study Materials to Index

This guide explains how to add new study materials and flashcards to the study hub (index.html).

## Understanding the Card Pattern

The `index.html` file uses a card-based layout with different sections:

### 1. Exam Domains Section
Cards linking to filtered views of the ML engineering reader.

### 2. Flashcards & Quizzes Section
Cards linking to interactive flashcard and quiz pages.

### 3. ML Engineering Study Guides Section
Cards linking to study guide pages (HTML files that render markdown).

### 4. Other Resources Section
Cards for additional tools and resources.

## Card Structure

Each card follows this HTML pattern:

```html
<a href="page-url.html" class="card">
    <div class="card-icon">📚</div>
    <h3 class="card-title">Card Title</h3>
    <p class="card-description">Brief description of the resource.</p>
</a>
```

## Adding New Content - Workflow

### Option 1: Adding a New Flashcard Set

1. **Create the flashcard HTML file** (e.g., `my_flashcards.html`)
   - Use existing flashcard files as templates (e.g., `mla_study_cards.html`)
   - Follow the same styling and interactive flip card pattern

2. **Add a card to index.html** in the "Flashcards & Quizzes" section:
   ```html
   <a href="my_flashcards.html" class="card">
       <div class="card-icon">🎯</div>
       <h3 class="card-title">My New Flashcards</h3>
       <p class="card-description">Description of what the flashcards cover.</p>
   </a>
   ```

3. **Commit and push** to the master branch for deployment

### Option 2: Adding a New Study Guide

1. **Create markdown file** (e.g., `new_study_guide.md`)
   - Write your content in markdown format
   - Use clear headings and structure

2. **Create HTML wrapper** (e.g., `new_study_guide.html`)
   - Copy one of the existing study guide HTML files
   - Update the title and markdown file reference
   - Example:
   ```javascript
   fetch('new_study_guide.md')
       .then(response => response.text())
       .then(text => {
           document.getElementById('content').innerHTML = marked.parse(text);
       })
   ```

3. **Add a card to index.html** in the "ML Engineering Study Guides" section:
   ```html
   <a href="new_study_guide.html" class="card">
       <div class="card-icon">📚</div>
       <h3 class="card-title">New Study Guide</h3>
       <p class="card-description">What this guide covers.</p>
   </a>
   ```

4. **Commit and push** to the master branch

### Option 3: Adding Questions/Materials to Existing Resources

To add questions or materials to existing flashcards:

1. **Locate the flashcard HTML file** (e.g., `mla_study_cards.html`)

2. **Find the card data section** (usually at the bottom in a `<script>` tag)

3. **Add new card objects** following the existing pattern:
   ```javascript
   {
       domain: "Domain Name",
       question: "Your question here?",
       answer: "The answer with explanation.",
       source: "Source reference"
   }
   ```

4. **Commit and push**

## Icon Options

Common emojis used for cards:
- 📚 Study guides
- 🎯 Flashcards
- 💾 Storage/Data topics
- ⚡ DevOps/Performance
- ✅ Quizzes/Tests
- 📖 Readers
- 🧠 ML/AI content
- 🚀 Deployment
- 🛡️ Security

## Example: Full Workflow

### Scenario: You want to add new SageMaker flashcards

1. **Create the file**: `sagemaker_flashcards.html`
   - Copy structure from `mla_study_cards.html`
   - Add your SageMaker questions

2. **Edit index.html**:
   ```html
   <!-- Add this card in the Flashcards & Quizzes section -->
   <a href="sagemaker_flashcards.html" class="card">
       <div class="card-icon">🧠</div>
       <h3 class="card-title">SageMaker Deep Dive</h3>
       <p class="card-description">Advanced SageMaker features and best practices.</p>
   </a>
   ```

3. **Update README.md** (optional):
   Add the new resource to the deployment links section

4. **Commit**: 
   ```bash
   git add sagemaker_flashcards.html index.html README.md
   git commit -m "Add SageMaker flashcards"
   git push origin master
   ```

5. **Wait for deployment**: GitHub Actions will deploy automatically (2-3 minutes)

6. **Access**: Visit `https://aiegoo.github.io/aws-cert/sagemaker_flashcards.html`

## Tips

- **Consistency**: Keep card descriptions concise (1-2 sentences)
- **Icons**: Use relevant emojis that match the content type
- **Testing**: After pushing to master, verify the links work on GitHub Pages
- **Sections**: Keep cards organized in the appropriate section
- **Mobile**: The card grid is responsive and works on all devices

## Need Help?

If you want to add specific materials:
1. Provide the content/questions
2. Specify which section it belongs to
3. I'll create the files and add the cards to index.html following this pattern
