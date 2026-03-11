# F1 Assistant

Ask any natural language question about F1 history (1950–2024).
No hallucinations — the LLM writes a pandas query, your CSV data answers it.

## How it works

```
Your question
    → LLM generates a pandas query (never sees your data)
    → pandas runs it on your CSVs (always accurate)
    → LLM formats the raw result as a sentence
    → Answer
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### File Structure
```
f1_assistant/
├── f1_qa.py
├── requirements.txt
├── README.md
└── data/
    ├── race_summaries.csv
    ├── race_details.csv
    ├── driver_standings.csv
    ├── driver_details.csv
    ├── constructor_standings.csv
    ├── fastest_laps.csv
    ├── fastestlaps_detailed.csv
    ├── pitstops.csv
    ├── practices.csv
    ├── qualifyings.csv
    ├── starting_grids.csv
    ├── sprint_results.csv
    ├── sprint_grid.csv
    └── team_details.csv
```

### Download and set up ollama and deepseek coder 
This version works on Deepseek_coder:later but you change it in the code, if you wanna use another agent.
Install ollama from https://ollama.com/
and run these commands in command line
ollama pull deepseek-coder
pip install pandas requests

### run ollama list and make sure the name there and in the code matches exactly


### 4. Run it

Interactive mode (recommended):
```bash
python f1_qa.py
```

Single question:
```bash
python f1_qa.py "who has the most race wins of all time?"
```

## Example questions

```
Who has the most race wins of all time?
Which team won the most constructors championships?
How many races did Ayrton Senna win?
Who had the most pole positions in 2023?
What was the fastest pitstop ever recorded?
Which driver has the most fastest laps?
How many different winners were there in 2021?
Who won the first ever Formula 1 race?
Which team scored the most points in 2023?
How many races has Fernando Alonso competed in?
```

Type `verbose` in interactive mode to see the generated pandas query alongside the answer.
Type `quit` to exit.

