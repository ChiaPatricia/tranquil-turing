import os
from pathlib import Path

import google.generativeai as genai

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class Config:
    EXCEL_FILE = Path('./reference/Data.xlsx')
    PROMPTS_DIR = Path('./prompts')
    GEMINI_MODEL = 'models/gemini-2.0-flash-thinking-exp'
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


genai.configure(api_key=Config.GOOGLE_API_KEY)
model = genai.GenerativeModel(Config.GEMINI_MODEL)
with open(Config.PROMPTS_DIR / "miti_codes.md", "r", encoding="utf-8") as f:
    prompt = f.read()

output = []
for sheet in range(1, 21):
    df = pd.read_excel(Config.EXCEL_FILE, sheet_name=sheet)
    df.iloc[:,1].replace({'C': 'Client', 'P': 'Provider', 'I': 'Provider'}, inplace=True)
    
    dialogue_list = []
    context_window = 2 #how many utterances, 2 = pair of client+interviewer
    for idx, row in df.iterrows():
        if row[1].strip() != 'Client': #is interviewer
            dialogue = []
            for lookback in range(-context_window+1, 1):
                if idx + lookback >= 0:
                    dialogue.append(df.iloc[idx+lookback, 1] + ": " + df.iloc[idx+lookback, 2])
            dialogue_list.append(" \n ".join(dialogue))
            
    response_list = []
    for dialogue in dialogue_list:
        response = model.generate_content(prompt + " \n " + dialogue)
        response_list.append(response.text) 
        time.sleep(6) #keep under 10 RPM rate limit
            
    output.append(response_list)

def extract_code(text): #gets final coded abbreviation from LLM output
    if not pd.isna(text): 
        return text.replace('*','').replace('(','').replace(')','').replace('.','').split()[-1]
    else:
        return ''

predicted_codes = pd.concat(output).map(extract_code)

comparison = pd.DataFrame()
for sheet in range(1,21):
    df = pd.read_excel(Config.EXCEL_FILE, sheet_name=sheet)
    combined = pd.concat([predicted_codes.loc[sheet].dropna().rename('Predicted'), df.iloc[:,3].dropna().reset_index(drop=True)], axis=1)
    comparison = pd.concat([comparison, combined])

comparison['Code'] = comparison['Code'].str.upper().str.strip()
single_valued = ['CR', 'Q', 'NC', 'SR', 'CONFRONT', 'GI', 'SEEK', 'P', 'AF', 'EMPHASIZE', 'PW']
subset = comparison[comparison['Code'].isin(single_valued)]
name_standardization = {'CONFRONT':'Confront', 'EMPHASIZE':'Emphasize', 'P':'Persuade', 'PW':'Persuade with', 'SEEK':'Seek'}
subset['Code'].replace(name_standardization, inplace=True)

cmatrix = subset.groupby('Code').value_counts().unstack()
cmatrix.replace(np.nan, 0, inplace=True)
cmatrix.style.format("{:.0f}")

print(accuracy_score(subset['Code'],subset['Predicted']))
print(f1_score(subset['Code'],subset['Predicted'], average='macro'))