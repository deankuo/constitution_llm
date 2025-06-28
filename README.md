# Constitution Analysis Pipeline

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Run
#### Use Shell script
```bash
# Make script executable
chmod +x run.sh

# Run with default settings (processes all data)
./run.sh

# Test with first 10 entries
./run.sh --test 10

# Custom configuration
./run.sh --models "gpt-3.5-turbo gpt-4" --delay 2.0 --batch-size 20
```
#### Use Python script
```bash
# Basic usage with defaults
python main.py

# Test mode
python main.py --test 5

# Custom configuration
python main.py --input ./Dataset/polity_level_data.csv \
               --output ./Dataset/results.csv \
               --models gpt-4.1-nano \
               --delay 1.5
```

### Input Format
CSV with: `territorynamehistorical`, `start_year`, `end_year`

### Output
CSV with constitution analysis results.
