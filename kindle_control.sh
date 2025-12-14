#!/bin/bash
# Kindle Screenshot & OCR Control - Terminal Version
# Simple menu-driven workflow control

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
RAW_DIR="/mnt/c/Users/hsyyu/Documents/kindle_raw"
CLEAN_DIR="/mnt/c/Users/hsyyu/Documents/kindle_clean"
OUTPUT_DIR="output"

# Settings
PAGES=560
WAIT_TIME=2
LEFT_CROP=300
BOTTOM_CROP=100

show_menu() {
    clear
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Kindle Screenshot & OCR Control${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    echo -e "${GREEN}1.${NC} Start Screenshot Capture (${PAGES} pages, ${WAIT_TIME}s wait)"
    echo -e "${GREEN}2.${NC} Deduplicate & Crop Screenshots (left=${LEFT_CROP}, bottom=${BOTTOM_CROP})"
    echo -e "${GREEN}3.${NC} Run OCR on Clean Screenshots"
    echo -e "${GREEN}4.${NC} Generate Flashcards"
    echo -e "${GREEN}5.${NC} Run Complete Workflow (All Steps)"
    echo ""
    echo -e "${YELLOW}6.${NC} Check Screenshot Count"
    echo -e "${YELLOW}7.${NC} View OCR Output"
    echo -e "${YELLOW}8.${NC} Open Clean Screenshots Folder"
    echo -e "${YELLOW}9.${NC} Settings"
    echo ""
    echo -e "${RED}0.${NC} Exit"
    echo ""
    echo -n "Select option: "
}

check_count() {
    echo -e "\n${BLUE}Checking screenshot counts...${NC}"
    
    if [ -d "$RAW_DIR" ]; then
        raw_count=$(ls -1 "$RAW_DIR"/*.png 2>/dev/null | wc -l)
        echo -e "Raw screenshots: ${GREEN}${raw_count}${NC}"
    else
        echo -e "Raw directory not found"
    fi
    
    if [ -d "$CLEAN_DIR" ]; then
        clean_count=$(ls -1 "$CLEAN_DIR"/*.png 2>/dev/null | wc -l)
        echo -e "Clean screenshots: ${GREEN}${clean_count}${NC}"
    else
        echo -e "Clean directory not yet created"
    fi
    
    echo ""
    read -p "Press Enter to continue..."
}

start_capture() {
    echo -e "\n${GREEN}Starting Screenshot Capture...${NC}"
    echo -e "${YELLOW}Pages: ${PAGES}, Wait time: ${WAIT_TIME}s${NC}"
    echo ""
    echo -e "${RED}INSTRUCTIONS:${NC}"
    echo "1. Make sure Kindle is open in Chrome"
    echo "2. Press F11 to enter fullscreen"
    echo "3. Navigate to page 26 (start page)"
    echo "4. Press ENTER here to start"
    echo ""
    read -p "Ready? Press ENTER to start capture..."
    
    # Launch PowerShell script
    echo -e "\n${BLUE}Launching PowerShell...${NC}"
    powershell.exe -ExecutionPolicy Bypass -File ./capture_raw.ps1 -MaxPages "$PAGES" -WaitTime "$WAIT_TIME"
    
    echo -e "\n${GREEN}Capture complete!${NC}"
    read -p "Press Enter to continue..."
}

run_dedup() {
    echo -e "\n${GREEN}Deduplicating and Cropping Screenshots...${NC}"
    echo -e "${YELLOW}Left crop: ${LEFT_CROP}px, Bottom crop: ${BOTTOM_CROP}px${NC}"
    
    if [ ! -d "$RAW_DIR" ] || [ -z "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
        echo -e "${RED}Error: No raw screenshots found!${NC}"
        read -p "Press Enter to continue..."
        return
    fi
    
    python3 deduplicate_and_crop.py \
        --input "$RAW_DIR" \
        --output "$CLEAN_DIR" \
        --left "$LEFT_CROP" \
        --bottom "$BOTTOM_CROP"
    
    echo -e "\n${GREEN}Deduplication complete!${NC}"
    read -p "Press Enter to continue..."
}

run_ocr() {
    echo -e "\n${GREEN}Running OCR...${NC}"
    
    if [ ! -d "$CLEAN_DIR" ] || [ -z "$(ls -A "$CLEAN_DIR" 2>/dev/null)" ]; then
        echo -e "${RED}Error: No clean screenshots found!${NC}"
        echo -e "${YELLOW}Run deduplication first (option 2)${NC}"
        read -p "Press Enter to continue..."
        return
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    python3 extract_text_split_pages.py \
        --input-dir "$CLEAN_DIR" \
        --output "$OUTPUT_DIR/kindle_text_final.txt"
    
    echo -e "\n${GREEN}OCR complete!${NC}"
    read -p "Press Enter to continue..."
}

gen_flashcards() {
    echo -e "\n${GREEN}Generating Flashcards...${NC}"
    
    if [ ! -f "$OUTPUT_DIR/kindle_text_final.txt" ]; then
        echo -e "${RED}Error: OCR text not found!${NC}"
        echo -e "${YELLOW}Run OCR first (option 3)${NC}"
        read -p "Press Enter to continue..."
        return
    fi
    
    python3 generate_flashcards_from_text.py \
        --input "$OUTPUT_DIR/kindle_text_final.txt" \
        --output "$OUTPUT_DIR/kindle_flashcards.html"
    
    echo -e "\n${GREEN}Flashcards generated!${NC}"
    echo -e "${BLUE}File: ${OUTPUT_DIR}/kindle_flashcards.html${NC}"
    read -p "Press Enter to continue..."
}

run_all() {
    echo -e "\n${BLUE}Running Complete Workflow...${NC}"
    echo ""
    
    start_capture
    run_dedup
    run_ocr
    gen_flashcards
    
    echo -e "\n${GREEN}All steps complete!${NC}"
    read -p "Press Enter to continue..."
}

view_ocr() {
    if [ -f "$OUTPUT_DIR/kindle_text_final.txt" ]; then
        less "$OUTPUT_DIR/kindle_text_final.txt"
    else
        echo -e "${RED}OCR text file not found${NC}"
        read -p "Press Enter to continue..."
    fi
}

open_folder() {
    echo -e "\n${BLUE}Opening folder...${NC}"
    explorer.exe "$(wslpath -w "$CLEAN_DIR")" 2>/dev/null || echo "Folder not found"
    read -p "Press Enter to continue..."
}

settings_menu() {
    while true; do
        clear
        echo -e "${BLUE}Settings${NC}"
        echo ""
        echo "1. Pages to capture: ${GREEN}${PAGES}${NC}"
        echo "2. Wait time: ${GREEN}${WAIT_TIME}${NC} seconds"
        echo "3. Left crop: ${GREEN}${LEFT_CROP}${NC} pixels"
        echo "4. Bottom crop: ${GREEN}${BOTTOM_CROP}${NC} pixels"
        echo ""
        echo "0. Back to main menu"
        echo ""
        echo -n "Select option: "
        
        read choice
        case $choice in
            1) read -p "Enter pages to capture: " PAGES ;;
            2) read -p "Enter wait time (seconds): " WAIT_TIME ;;
            3) read -p "Enter left crop (pixels): " LEFT_CROP ;;
            4) read -p "Enter bottom crop (pixels): " BOTTOM_CROP ;;
            0) break ;;
            *) echo "Invalid option" ;;
        esac
    done
}

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1) start_capture ;;
        2) run_dedup ;;
        3) run_ocr ;;
        4) gen_flashcards ;;
        5) run_all ;;
        6) check_count ;;
        7) view_ocr ;;
        8) open_folder ;;
        9) settings_menu ;;
        0) 
            echo -e "\n${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            sleep 1
            ;;
    esac
done
