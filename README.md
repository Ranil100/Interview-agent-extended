# Interview Helper

A Streamlit app to compare a resume to a job description and generate:
- ATS compatibility score
- Interview preparation suggestions
- A professional follow-up email format

## Tech stack
- Python
- Streamlit
- LlamaIndex
- OpenAI API

## Setup
1. Open a terminal in `c:\Users\vaigh\OneDrive\Desktop\interview-helper`
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI key in environment variables or enter it in the sidebar:

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage
1. Upload your resume as PDF or paste the resume text.
2. Paste the job description.
3. Click `Analyze`.
4. Optionally use the prompt box to ask a custom question about the resume or job description, then click `Run prompt`.

The app will display the ATS score, interview suggestions, email format, and the custom prompt result.
