import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
from pathlib import Path

class DatasetConstructor:
    def __init__(self, instruction_template_path: str, data_root_dir: str, 
                 reports_path: str, examples_path: Optional[str] = None,
                 report_examples_path: Optional[str] = None,
                 num_shots: int = 3, num_report_examples: int = 3):
        """
        Initialize the dataset constructor
        
        Args:
            instruction_template_path (str): Path to instruction template file
            data_root_dir (str): Root directory containing market subdirectories with data files
            reports_path (str): Path to reports.txt CSV file
            examples_path (str, optional): Path to example.csv file for few-shot learning
            report_examples_path (str, optional): Path to report examples CSV file
            num_shots (int): Number of few-shot examples to include (default: 3)
            num_report_examples (int): Number of report examples to include (default: 3)
        """
        self.instruction_template_path = instruction_template_path
        self.data_root_dir = data_root_dir
        self.reports_path = reports_path
        self.examples_path = examples_path
        self.report_examples_path = report_examples_path
        self.num_shots = num_shots
        self.num_report_examples = num_report_examples
        self.instruction = self._load_instruction()
        self.few_shot_examples = self._load_few_shot_examples() if examples_path else []
        self.report_examples = self._load_report_examples() if report_examples_path else []
        
    def _load_instruction(self) -> str:
        """Load instruction template from file"""
        with open(self.instruction_template_path, 'r') as f:
            return f.read().strip()
    
    def _load_market_data(self, market: str, source: str, date_str: str) -> Optional[pd.DataFrame]:
        """Load table data for specific market and date"""
        sub_dir = f"{'_'.join(market.split())}-{source}"
        file_path = os.path.join(self.data_root_dir, sub_dir, f"{self._format_date(date_str)}.csv")
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            return None

    def _load_reports(self) -> pd.DataFrame:
        """Load reports from CSV file"""
        return pd.read_csv(self.reports_path, encoding="utf-16", sep="\t")
    
    def _format_date(self, date_str: str) -> str:
        """Format date string to YYYY-MM-DD format"""
        return pd.to_datetime(date_str).strftime('%Y-%m-%d')

    def _format_data_example(self, example: Dict, index: int) -> str:
        """
        Format a single data example in a consistent way
        
        Args:
            example (dict): Example data
            index (int): Example number
            
        Returns:
            str: Formatted example string
        """
        return f"""Data Example {index + 1}:
Market: {example['market']}
Date: {example['date']}
Table Data:
{example['table_data']}

Report:
{example['report']}
---"""

    def _format_report_example(self, example: Dict, index: int) -> str:
        """
        Format a single report example in a consistent way
        
        Args:
            example (dict): Report example
            index (int): Example number
            
        Returns:
            str: Formatted report example string
        """
        return f"""Report Example {index + 1}:
Market: {example['market']}
Date: {example['date']}
Report:
{example['report']}
---"""

    def _format_input(self, table_data: str, market: str, date: str) -> str:
        """Format the input data in a consistent way"""
        return f"""Input:
Market: {market}
Date: {date}
Table Data:
{table_data}

Generate a report based on the table data above."""

    def _load_few_shot_examples(self) -> List[Dict]:
        """Load and process few-shot examples"""
        examples = defaultdict(list)
        if not self.examples_path or not os.path.exists(self.examples_path):
            return examples
            
        examples_df = pd.read_csv(self.examples_path, sep="\t", encoding="utf-16")
        
        for _, row in examples_df.iterrows():
            market = row['market']
            date_str = row['date']
            source = row['source']
            report = row['passage']
            
            df = self._load_market_data(market, source, date_str)
            if df is None:
                continue
                
            example = {
                'source': source,
                'market': market,
                'date': date_str,
                'table_data': df.to_string(index=False),
                'report': report
            }
            examples[f"{market}-{source}"].append(example)
            
        # Randomly sample if we have more examples than num_shots
        for market_source, market_examples in examples.items():
            if len(market_examples) > self.num_shots:
                import random
                examples[market_source] = random.sample(market_examples, self.num_shots)
            
        return examples

    def _load_report_examples(self) -> List[Dict]:
        """Load and process report examples"""
        examples = defaultdict(list)
        if not self.report_examples_path or not os.path.exists(self.report_examples_path):
            return examples
            
        examples_df = pd.read_csv(self.report_examples_path, encoding="utf-16", sep="\t")
        
        for _, row in examples_df.iterrows():
            market = row['market']
            date_str = row['date']
            source = row['source']
            report = row['passage']

            example = {
                'source': source,
                'market': market,
                'date': date_str,
                'report': report
            }
            examples[f"{market}-{source}"].append(example)

        examples[f"{market}-{source}"].append(example)
            
        # Randomly sample if we have more examples than specified
        for market_source, market_examples in examples.items():
            if len(market_examples) > self.num_report_examples:
                import random
                examples[market_source] = random.sample(market_examples, self.num_report_examples)
            
        return examples
    
    def _construct_prompt_with_examples(self, table_data: str, market: str, source: str, date: str) -> Dict:
        """
        Construct the full prompt including data examples, report examples, and formatted input
        """
        # Format data examples
        formatted_data_examples = [
            self._format_data_example(example, i) 
            for i, example in enumerate(self.few_shot_examples[f"{market}-{source}"])
        ]
        
        # Format report examples
        formatted_report_examples = [
            self._format_report_example(example, i)
            for i, example in enumerate(self.report_examples[f"{market}-{source}"])
        ]
        
        # Format the input
        formatted_input = self._format_input(table_data, market, date)
        
        # Combine everything into the final prompt
        sections = [self.instruction]
        
        if formatted_data_examples:
            sections.append("\nData Examples:")
            sections.extend(formatted_data_examples)
            
        if formatted_report_examples:
            sections.append("\nReport Style Examples:")
            sections.extend(formatted_report_examples)
            
        sections.append("\n" + formatted_input)
        
        full_prompt = "\n".join(sections)
        
        prompts = {
            'instruction': self.instruction,
            'few_shot_examples': self.few_shot_examples,
            'report_examples': self.report_examples,
            'formatted_prompt': full_prompt
        }
        return prompts
    
    def construct_dataset(self) -> List[Dict]:
        """Construct the dataset by matching reports with table data"""
        dataset = []
        reports_df = self._load_reports()
        
        for _, row in reports_df.iterrows():
            market = row['market']
            date_str = row['date']
            source = row['source']
            report = row['passage']
            
            df = self._load_market_data(market, source, date_str)
            if df is None:
                continue
            
            table_data = df.to_string(index=False)
            prompts = self._construct_prompt_with_examples(table_data, market, source, date_str)
                
            entry = {
                'source': source,
                'market': market,
                'date': date_str,
                'instruction': self.instruction,
                'table_data': table_data,
                'report': report,
                'prompts': prompts
            }
            
            dataset.append(entry)
            
        return dataset
    
    def save_dataset(self, output_dir: str, split: str, dataset: Optional[List[Dict]] = None):
        """Save the constructed dataset to a JSON file"""
        if dataset is None:
            dataset = self.construct_dataset()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{split}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

def main():
    # Configuration
    # history_span = "1day"
    history_span = '1day'
    config = {
        'instruction_template_path': 'prompts/data2text_generation_task_instruction.txt',
        'data_root_dir': f'data/tabular_data/report_table_data/{history_span}',
        'reports_path': 'data/reports/reports.tsv',
        'examples_path': 'data/reports/selected_sample_reports.tsv',  # New parameter
        'output_path': f'data/processed_dataset/{history_span}',
        'num_report_examples': 2,
        'num_shots': 0  # Number of few-shot examples to include
    }
    
    for split in ["train", "validate", "test"]:
        # Initialize and run the dataset constructor
        constructor = DatasetConstructor(
            instruction_template_path=config['instruction_template_path'],
            data_root_dir=os.path.join(config['data_root_dir'], split),
            reports_path=config['reports_path'],
            examples_path=config['examples_path'],
            report_examples_path=config['examples_path'],
            num_shots=config['num_shots'],
            num_report_examples=config['num_report_examples']
        )
        
        # Construct and save the dataset
        dataset = constructor.construct_dataset()
        constructor.save_dataset(config['output_path'], split, dataset)
        
        print(f"Dataset construction completed for {split} split. Total entries: {len(dataset)}")
    
    # Print a sample of the formatted prompt
    if dataset:
        print("\nSample formatted prompt:")
        print(dataset[0]['prompts']['formatted_prompt'])

if __name__ == "__main__":
    main()