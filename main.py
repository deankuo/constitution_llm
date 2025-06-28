import pandas as pd
import re
from openai import OpenAI
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
SYSTEM_PROMPT = """You are a professional political scientist, historian and constitutional expert specializing in constitutional history across different countries.
Your task is to determine whether a given polity had a constitution during its period of existence based on the country name and the time period provided. 
To your understanding, A constitution is understood as a set of rules setting forth how a polity is governed. 
Typically, these are encoded in written documents–statutes, treaties, charters, basic laws, legal codes, and constitutions. 
For your purposes, it is not enough for a polity to have a code of law; that code must stipulate something about how the polity is to be governed, e.g., how leaders are chosen, their scope of authority (and any limitations thereof), and/or the rights of citizens. 
You do not attempt to determine whether constitutional rules were adhered to. 
It is sufficient to measure their de jure existence. 
If, however, a constitution is clearly abrogated then it is no longer in force.

Please follow these requirements:
1. Base your judgment on historical facts
2. If a constitution existed during this period, provide the official name of the constitution
3. Provide a concise but clear explanation of your reasoning, considering the entire time period
4. Strictly follow the specified output format without adding extra content

You must answer each question with a professional, objective, and accurate attitude."""

USER_PROMPT_TEMPLATE = """Please analyze whether the following polity had a constitution during its period of existence:

Country/Polity: {country}
Start Year: {start_year}
End Year: {end_year}
Duration: {start_year}-{end_year}

Please answer strictly in the following format:
Constitution Existed: [Yes/No]
Constitution Name: [Constitution name, or "N/A" if none existed]
Explanation: [Brief explanation of your reasoning considering the entire time period]"""


def load_polity_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed polity data from CSV file
    
    Args:
        file_path: Path to the CSV file containing polity data
        
    Returns:
        DataFrame with polity information
    """
    print(f"Loading polity data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['territorynamehistorical', 'start_year', 'end_year']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Data loaded successfully! Total polities: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_prompt(country: str, start_year: int, end_year: int) -> Tuple[str, str]:
    """
    Create system prompt and user prompt for polity-level analysis
    
    Args:
        country: Country/polity name
        start_year: Start year of the polity
        end_year: End year of the polity
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        country=country,
        start_year=start_year,
        end_year=end_year
    )
    
    return SYSTEM_PROMPT, user_prompt

def parse_llm_response(response: str) -> Dict[str, str]:
    """
    Parse LLM response and extract structured information
    
    Args:
        response: Raw LLM response
        
    Returns:
        Dictionary containing parsed results
    """
    result = {
        'constitution_existed': None,
        'constitution_name': None,
        'explanation': None
    }
    
    try:
        # Use regex to extract information - handle both bracketed and non-bracketed formats
        # Pattern 1: Constitution Existed: [Yes/No] or Constitution Existed: Yes/No
        constitution_existed_match = re.search(r'Constitution Existed:\s*\[?([YesNo]+)\]?', response, re.IGNORECASE)
        
        # Pattern 2: Constitution Name: [Name] or Constitution Name: Name
        constitution_name_match = re.search(r'Constitution Name:\s*\[?([^\[\]\n]*?)\]?(?:\s*Explanation:|$)', response, re.IGNORECASE)
        
        # Pattern 3: Explanation: [Text] or Explanation: Text
        explanation_match = re.search(r'Explanation:\s*\[?([^\[\]]*?)\]?$', response, re.IGNORECASE | re.DOTALL)
        
        if constitution_existed_match:
            existed_text = constitution_existed_match.group(1).strip().lower()
            result['constitution_existed'] = 'yes' if 'yes' in existed_text else 'no'
        
        if constitution_name_match:
            name_text = constitution_name_match.group(1).strip()
            result['constitution_name'] = name_text if name_text else None
        
        if explanation_match:
            explanation_text = explanation_match.group(1).strip()
            result['explanation'] = explanation_text if explanation_text else None
            
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response}")
    
    return result

def query_openai_model(system_prompt: str, user_prompt: str, model: str = "gpt-4.1-nano", api_key: str = None) -> str:
    """
    Query OpenAI model using the new OpenAI Python client (>=1.0.0)
    
    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        model: Model name
        api_key: OpenAI API key
        
    Returns:
        Model response as string
    """
    try:
        client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Lower randomness for more consistent results
            max_tokens=800   # Increased for polity-level explanations
        )
        
        # Extract the actual text content from the response
        content = response.choices[0].message.content
        if content is not None:
            return content.strip()
        else:
            print("Warning: Received empty content from OpenAI API")
            return None
        
    except Exception as e:
        print(f"Error querying OpenAI model: {e}")
        return None


def process_single_polity(country: str, start_year: int, end_year: int, 
                         model_name: str = "gpt-4.1-nano", api_key: str = None) -> Dict:
    """
    Process a single polity
    
    Args:
        country: Country/polity name
        start_year: Start year
        end_year: End year
        model_name: Model name
        api_key: API key
        
    Returns:
        Processing result dictionary
    """
    # Create prompts
    system_prompt, user_prompt = create_prompt(country, start_year, end_year)
    
    # Query model based on model type
    response = query_openai_model(system_prompt, user_prompt, model_name, api_key)
    
    if response is None:
        return None
    
    # Parse response
    parsed_result = parse_llm_response(response)
    
    # Convert to final format
    model_suffix = model_name.replace("-", "_").replace(".", "_")
    result = {
        'territorynamehistorical': country,
        'start_year': start_year,
        'end_year': end_year,
        f'constitution_{model_suffix}': 1 if parsed_result['constitution_existed'] == 'yes' else 0,
        f'constitution_name_{model_suffix}': parsed_result['constitution_name'] if parsed_result['constitution_name'] else "N/A",
        f'explanation_{model_suffix}': parsed_result['explanation'] if parsed_result['explanation'] else "No explanation provided"
    }
    
    return result

def process_batch(df: pd.DataFrame, model_names: List[str] = ["gpt-4.1-nano"], 
                 batch_size: int = 10, delay: float = 1.0, api_key: str = None) -> pd.DataFrame:
    """
    Process polity data in batches
    
    Args:
        df: Input DataFrame with polity data
        model_names: List of model names to use
        batch_size: Batch size for temporary saves
        delay: Delay between requests (seconds)
        api_key: API key
        
    Returns:
        DataFrame containing results from all models
    """
    results = []
    total_polities = len(df)
    
    print(f"Starting to process {total_polities} polities using models: {model_names}")
    
    for idx, row in df.iterrows():
        country = row['territorynamehistorical']
        start_year = int(row['start_year'])
        end_year = int(row['end_year'])
        
        print(f"Processing: {idx + 1}/{total_polities} - {country} ({start_year}-{end_year})")
        
        entry_result = {
            'territorynamehistorical': country,
            'start_year': start_year,
            'end_year': end_year
        }
        
        # Query each model
        for model_name in model_names:
            print(f"  Using model: {model_name}")
            
            result = process_single_polity(country, start_year, end_year, model_name, api_key)
            
            if result:
                # Merge results (exclude duplicate basic fields)
                for key, value in result.items():
                    if key not in ['territorynamehistorical', 'start_year', 'end_year']:
                        entry_result[key] = value
            else:
                # If query fails, fill with default values
                model_suffix = model_name.replace("-", "_").replace(".", "_")
                entry_result[f'constitution_{model_suffix}'] = 0
                entry_result[f'constitution_name_{model_suffix}'] = "Query Failed"
                entry_result[f'explanation_{model_suffix}'] = "Failed to get response from model"
            
            # Add delay to avoid API limits
            time.sleep(delay)
        
        results.append(entry_result)
        
        # Save temporary results periodically
        if (idx + 1) % batch_size == 0:
            temp_df = pd.DataFrame(results)
            temp_filename = f'temp_polity_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            temp_df.to_csv(temp_filename, index=False)
            print(f"  Temporarily saved {len(results)} results to {temp_filename}")
    
    return pd.DataFrame(results)

def save_results(results_df: pd.DataFrame, output_path: str):
    """
    Save results to file with polity-level statistics
    
    Args:
        results_df: Results DataFrame
        output_path: Output file path
    """
    # Save as CSV
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to: {output_path}")
    
    # Display statistics
    print("\nPolity-Level Result Statistics:")
    print(f"Total polities processed: {len(results_df)}")
    
    # Show some examples
    print(f"\nSample results:")
    print(results_df[['territorynamehistorical', 'start_year', 'end_year']].head())

def main():
    """
    Main function for command line execution
    """
    parser = argparse.ArgumentParser(description='Constitution Analysis Pipeline - Polity Level')
    parser.add_argument('--input', '-i', default='./Dataset/polity_level_data.csv', help='Input CSV file path (preprocessed polity data)')
    parser.add_argument('--output', '-o', default='./Dataset/llm_predictions.csv', help='Output CSV file path')
    parser.add_argument('--user_prompt', '-u', type=str, help='User prompt for custom queries')
    parser.add_argument('--system_prompt', '-s', type=str, help='System prompt for custom queries')
    parser.add_argument('--models', '-m', nargs='+', default=['gpt-4.1-nano'], 
                       help='Model names to use (space-separated)')
    parser.add_argument('--api-key', '-k', help='API key (Currently use OpenAI, will be replace with AWS Bedrock in the future)')
    parser.add_argument('--batch-size', '-b', type=int, default=10, 
                       help='Batch size for temporary saves')
    parser.add_argument('--delay', '-d', type=float, default=1.0, 
                       help='Delay between API calls (seconds)')
    parser.add_argument('--test', '-t', type=int, default=None, 
                       help='Process only first N polities for testing')
    
    args = parser.parse_args()
    
    print(f"OPENAI_API_KEY loaded: {os.getenv('OPENAI_API_KEY')[:10] if os.getenv('OPENAI_API_KEY') else 'Not found'}...")
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: No API key provided. Set OPENAI_API_KEY environment variable or use --api-key argument")
    
    # Load preprocessed polity data
    df = load_polity_data(args.input)
    
    # Use subset for testing if specified
    if args.test:
        df = df.head(args.test)
        print(f"Testing mode: Processing only first {args.test} polities")
    
    # Prompt customization
    if args.system_prompt:
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = args.system_prompt
    if args.user_prompt:
        global USER_PROMPT_TEMPLATE
        USER_PROMPT_TEMPLATE = args.user_prompt
    
    # Process data
    results_df = process_batch(
        df, 
        model_names=args.models, 
        batch_size=args.batch_size, 
        delay=args.delay, 
        api_key=api_key
    )
    
    # Save results
    save_results(results_df, args.output)
    
    print("\nPolity-level processing completed successfully!")

if __name__ == "__main__":
    main()