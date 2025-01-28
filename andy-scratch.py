import os
from google import genai
from google.genai import types
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Read all sheets into a dictionary of DataFrames
excel_file = './reference/Data.xlsx'
all_sheets = pd.read_excel(excel_file, sheet_name=None)

def get_selected_columns(all_sheets, sheet_number):
    # Select columns in a sheet and replace 'I' with 'P' in second column
    first_sheet = all_sheets[str(sheet_number)]
    selected_columns = first_sheet.iloc[:, 1:3]
    selected_columns.iloc[:, 0] = selected_columns.iloc[:, 0].replace('I', 'P')  # Replace in first column
    data_string = selected_columns.to_string(index=False)  # index=False removes the row numbers
    return data_string

def get_gemini_response(system_instruction: str, contents: str) -> str:
    """
    Get a response from Gemini API using the provided system instruction and contents.
    
    Args:
        system_instruction (str): The system instruction to guide Gemini's behavior
        contents (str): The content to process
        
    Returns:
        str: The response text from Gemini
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
        ),
    )
    return response.text

def read_markdown_files(folder_path: str) -> list[str]:
    """
    Read all markdown files from a specified folder and return their contents as a list of strings.
    
    Args:
        folder_path (str): Path to the folder containing markdown files
        
    Returns:
        list[str]: List of strings, where each string is the content of a markdown file
    """
    markdown_contents = []
    
    # Walk through the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    markdown_contents.append(f.read())
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
                
    return markdown_contents

prompts = read_markdown_files('./prompts')

system_prompt = prompts[0]

def build_prompt(reference_prompt: str, transcript: str) -> str:
    return reference_prompt + "\n <transcript> \n" + transcript + "\n </transcript> \n"

def process_transcripts():
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Process sheets 1 through 20
    for sheet_num in range(1, 21):
        transcript = get_selected_columns(all_sheets, sheet_num)
        
        # Process each reference prompt
        for i, reference_prompt in enumerate(prompts[1:], 1):
            prompt = build_prompt(reference_prompt, transcript)
            try:
                response = get_gemini_response(system_prompt, prompt)
                
                # Write to file
                output_file = f'outputs/sheet_{sheet_num}_prompt_{i}.txt'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Sheet Number: {sheet_num}\n")
                    f.write(f"Prompt Number: {i}\n")
                    f.write("=" * 50 + "\n")
                    f.write(response)
                    
                print(f"Processed sheet {sheet_num} with prompt {i}")
                
            except Exception as e:
                print(f"Error processing sheet {sheet_num} with prompt {i}: {str(e)}")

# Run the processing
process_transcripts()

##TODO: Read in each prompt, connect it with the contents of the interview, and generate a response