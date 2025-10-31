<pre>
  __  __ _           _                     _                  _______    _             
 |  \/  (_)         | |                   | |                |__   __|  | |            
 | \  / |_ _ __   __| |_ __ ___   __ _ ___| |_ ___ _ __ ___     | |_   _| |_ ___  _ __ 
 | |\/| | | '_ \ / _` | '_ ` _ \ / _` / __| __/ _ \ '__/ __|    | | | | | __/ _ \| '__|
 | |  | | | | | | (_| | | | | | | (_| \__ \ ||  __/ |  \__ \    | | |_| | || (_) | |   
 |_|  |_|_|_| |_|\__,_|_| |_| |_|\__,_|___/\__\___|_|  |___/    |_|\__,_|\__\___/|_|   
                                                                                       
</pre>

AI Math Tutor

FastAPI + Sympy backend that verifies algebraic reasoning, models how students think, 
and generates adaptive GPT hints in real time.

OVERVIEW

- Analyzes mathematical reasoning step-by-step.
- Models where logic breaks down and provides context-aware feedback
- Symbolic computation (Sympy) + large language model reasoning (GPT).

FEATURES

- Algebra step verification using Sympy
- AI-generated hints and next-step suggestions via OpenAI GPT
- Structured JSON responses for frontend integration
- Built with FastAPI, designed for extensibility

API ENDPOINTS

/health       →   GET   →  Server status  
/check-step   →   POST  →  Verify algebraic step  
/tutor        →   POST  →  Generate adaptive hint and next step  

EXAMPLE RESPONSE

{
  "verified": true,
  "hint": "Isolate the variable by subtracting 3 from both sides.",
  "proposed_next_step": "2x = 4"
}

TECH STACK

Python 3.11+
FastAPI
Sympy
OpenAI API
Uvicorn
python-dotenv

SETUP

1. Clone repository
   git clone https://github.com/<your-username>/math-tutor-backend.git

2. Create environment
   python -m venv .venv
   .venv\Scripts\activate    (Windows)
   source .venv/bin/activate (macOS/Linux)

3. Install dependencies
   pip install -r requirements.txt

4. Add environment variable
   OPENAI_API_KEY=your_api_key_here

5. Run server
   uvicorn main:app --reload
   Visit http://127.0.0.1:8000/docs

PURPOSE

Originally developed to support MindMasters students by providing real time adaptive feedback.

CONTACT

Maintainer: Srivath Kumaran  
Repository: https://github.com/srivathk/math-tutor-backend

LICENSE
MIT License
