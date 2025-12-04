#!/bin/bash
# Batch process all AWS Korean study materials
# Runs OCR on all PDF files and generates English study guides

set -e

source venv/bin/activate

echo "================================================"
echo "AWS Study Materials - Batch OCR Processing"
echo "================================================"

# Create output directories
mkdir -p processed_materials/{mla,devops,saa,ai,general}
mkdir -p study_guides/{mla,devops,saa,ai,general}

# Function to process directory
process_dir() {
    local input_dir=$1
    local output_dir=$2
    local topic=$3
    local limit=$4
    
    echo ""
    echo "Processing: $input_dir"
    echo "Output: $output_dir"
    echo "Topic: $topic"
    
    if [ -z "$limit" ]; then
        python3 ocr_study_app.py \
            --input "$input_dir" \
            --output "$output_dir" \
            --pattern "*.pdf" \
            --dpi 200 \
            --chunk 2000
    else
        python3 ocr_study_app.py \
            --input "$input_dir" \
            --output "$output_dir" \
            --pattern "*.pdf" \
            --limit "$limit" \
            --dpi 200 \
            --chunk 2000
    fi
    
    echo "✓ OCR complete for $input_dir"
    
    # Generate English study guides
    if [ -d "$output_dir" ] && [ "$(ls -A $output_dir/*.json 2>/dev/null)" ]; then
        echo "Generating English study content..."
        python3 generate_study_content.py \
            --input "$output_dir" \
            --output "study_guides/$topic" \
            --topic "$topic"
        echo "✓ Study guides generated"
    fi
}

# Priority 1: DevOps (smaller files)
echo ""
echo "=== Phase 1: DevOps Materials ==="
process_dir "devops" "processed_materials/devops" "devops" ""

# Priority 2: AI/GenAI (single files)
echo ""
echo "=== Phase 2: AI Materials ==="
process_dir "ai" "processed_materials/ai" "ml" ""

# Priority 3: General (multiple files)
echo ""
echo "=== Phase 3: General AWS Materials ==="
process_dir "general" "processed_materials/general" "general" "5"

# Priority 4: SAA (exam materials)
echo ""
echo "=== Phase 4: SAA Exam Materials ==="
process_dir "saa" "processed_materials/saa" "saa" ""

# Priority 5: MLA (large file - process chunks)
echo ""
echo "=== Phase 5: ML Engineering (Chunked) ==="
if [ -d "chunks_ML Engineering on AWS" ]; then
    echo "Processing ML Engineering chunks..."
    python3 ocr_study_app.py \
        --input "chunks_ML Engineering on AWS" \
        --output "processed_materials/mla_chunks" \
        --pattern "*.pdf" \
        --limit 5 \
        --dpi 200
    
    echo "✓ ML Engineering chunks processed"
fi

echo ""
echo "================================================"
echo "Batch Processing Summary"
echo "================================================"
find processed_materials -name "*.json" -type f | wc -l | xargs echo "OCR outputs:"
find study_guides -name "*.md" -type f | wc -l | xargs echo "Study guides:"
echo ""
echo "✓ All processing complete!"
echo "================================================"
echo ""
echo "View study guides in: study_guides/"
echo "View OCR outputs in: processed_materials/"
