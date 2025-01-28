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

# Read the system prompt
with open('prompts/00-MITI-system-prompt.md', 'r') as f:
    system_prompt = f.read().strip()

response_text = get_gemini_response(system_prompt, "Add real prompt here.")
print(response_text)

##TODO: Read in each prompt, connect it with the contents of the interview, and generate a response