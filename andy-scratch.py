import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    EXCEL_FILE = Path('./reference/Data.xlsx')
    PROMPTS_DIR = Path('./prompts')
    GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
    GEMINI_TEMPERATURE = 0.1

def get_selected_columns(all_sheets: Dict[str, pd.DataFrame], sheet_number: str) -> str:
    """
    Select and process columns from a specific sheet in the Excel workbook.
    
    Args:
        all_sheets: Dictionary of DataFrames containing all Excel sheets
        sheet_number: Sheet number to process
        
    Returns:
        Processed data as a string
        
    Raises:
        KeyError: If sheet_number doesn't exist
        IndexError: If required columns are not present
    """
    try:
        sheet = all_sheets[str(sheet_number)]
        selected_columns = sheet.iloc[:, 1:3]
        selected_columns.iloc[:, 0] = selected_columns.iloc[:, 0].replace('I', 'P')
        return selected_columns.to_string(index=False)
    except KeyError:
        logging.error(f"Sheet {sheet_number} not found in workbook")
        raise
    except IndexError:
        logging.error(f"Required columns not found in sheet {sheet_number}")
        raise

def get_gemini_response(system_instruction: str, contents: str) -> Optional[str]:
    """
    Get a response from Gemini API using the provided system instruction and contents.
    
    Args:
        system_instruction: The system instruction to guide Gemini's behavior
        contents: The content to process
        
    Returns:
        Response text from Gemini or None if the request fails
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY not found in environment variables")
        return None
        
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=Config.GEMINI_TEMPERATURE,
            ),
        )
        return response.text
    except Exception as e:
        logging.error(f"Error getting Gemini response: {str(e)}")
        return None

def read_markdown_files(folder_path: str) -> List[str]:
    """
    Read all markdown files from a specified folder.
    
    Args:
        folder_path: Path to the folder containing markdown files
        
    Returns:
        List of strings, where each string is the content of a markdown file
        
    Raises:
        FileNotFoundError: If folder_path doesn't exist
    """
    folder = Path(folder_path)
    if not folder.exists():
        logging.error(f"Folder not found: {folder_path}")
        raise FileNotFoundError(f"Folder not found: {folder_path}")
        
    contents = []
    try:
        for file in sorted(folder.glob("*.md")):
            logging.info(f"Reading file: {file}")
            contents.append(file.read_text(encoding='utf-8'))
        return contents
    except Exception as e:
        logging.error(f"Error reading markdown files: {str(e)}")
        raise

def build_prompt(reference_prompt: str, transcript: str) -> str:
    return reference_prompt + "\n <transcript> \n" + transcript + "\n </transcript> \n"

def process_transcripts(all_sheets: Dict[str, pd.DataFrame], prompts: List[str]) -> None:
    """
    Process transcripts by generating responses using Gemini API.
    
    Args:
        all_sheets: Dictionary of DataFrames containing all Excel sheets
        prompts: List of prompts to use for generating responses
    """
    # Create output directory if it doesn't exist
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    system_prompt = prompts[0]
    
    # Process sheets 1 through 20 - !! This is hard-coded, which is not great !!
    for sheet_num in range(1, 21):
        transcript = get_selected_columns(all_sheets, str(sheet_num))
        
        # Process each reference prompt
        for i, reference_prompt in enumerate(prompts[1:], 1):
            prompt = build_prompt(reference_prompt, transcript)
            try:
                response = get_gemini_response(system_prompt, prompt)
                
                # Write to file
                output_file = output_dir / f'sheet_{sheet_num}_prompt_{i}.txt'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Sheet Number: {sheet_num}\n")
                    f.write(f"Prompt Number: {i}\n")
                    f.write("=" * 50 + "\n")
                    f.write(response)
                    
                logging.info(f"Processed sheet {sheet_num} with prompt {i}")
                
            except Exception as e:
                logging.error(f"Error processing sheet {sheet_num} with prompt {i}: {str(e)}")

def main():
    """Main execution function."""
    try:
        # Read Excel file
        logging.info(f"Reading Excel file: {Config.EXCEL_FILE}")
        all_sheets = pd.read_excel(Config.EXCEL_FILE, sheet_name=None)
        
        # Read prompts
        logging.info(f"Reading prompts from: {Config.PROMPTS_DIR}")
        prompts = read_markdown_files(Config.PROMPTS_DIR)
        system_prompt = prompts[0]
        
        # Process transcripts
        process_transcripts(all_sheets, prompts)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()