import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
from pathlib import Path

class InstanceFactExtractor:
    THRESHOLDS = {
        'Treasury':  {'perf': 5.0,  'vol': 8.0},
        'Commodity': {'perf': 2.0,  'vol': 3.0},
        'Equity':    {'perf': 1.25, 'vol': 2.0},
        'Currency':  {'perf': 0.4,  'vol': 0.6}
    }

    def _classify_asset(self, symbol: str) -> str:
        """Accurately maps the asset based on ticker, ignoring folder contamination."""
        symbol = str(symbol).upper()
        if 'TMUBMUSD' in symbol: return 'Treasury'
        if '=X' in symbol: return 'Currency'
        if symbol in ['CL=F', 'GC=F', 'NG=F', 'ZC', 'ZW', 'ZS', 'LE', 'HE', 'GF', 'DC', 'DK'] or len(symbol) <= 3: 
            return 'Commodity'
        return 'Equity'

    def _map_market_to_class(self, market: str) -> str:
        """Map the DATATALES report target market to our volatility classes."""
        m = market.lower()
        if 'treasury' in m: return 'Treasury'
        if 'currency' in m: return 'Currency'
        if m in ['gold', 'oil', 'cattle', 'corn', 'dairy', 'lean hog', 'soybean', 'wheat']: return 'Commodity'
        return 'Equity'

    def extract_facts(self, df: pd.DataFrame, market: str) -> str:
        if df is None or df.empty: 
            return ""
            
        df_latest = df.sort_values('Date').groupby('Symbol').last().reset_index()
        
        req_cols = ['pct_change', 'weekly_change', 'intraday_range', 'dist_from_sma']
        if not all(col in df_latest.columns for col in req_cols):
            return "Note: Advanced metrics not pre-computed for this dataset."

        df_latest['Asset_Class'] = df_latest['Symbol'].apply(self._classify_asset)
        
        for ac, limits in self.THRESHOLDS.items():
            mask = df_latest['Asset_Class'] == ac
            df_latest.loc[mask, 'Daily_Performance'] = pd.cut(df_latest.loc[mask, 'pct_change'], bins=[-float('inf'), -limits['perf'], limits['perf'], float('inf')], labels=['Plunge', 'Flat', 'Surge'])
            df_latest.loc[mask, 'Weekly_Performance'] = pd.cut(df_latest.loc[mask, 'weekly_change'], bins=[-float('inf'), -limits['perf']*2.0, limits['perf']*2.0, float('inf')], labels=['Weekly Loss', 'Flat', 'Weekly Gain'])
            df_latest.loc[mask, 'Volatility_State'] = pd.cut(df_latest.loc[mask, 'intraday_range'], bins=[-float('inf'), limits['vol']/2, limits['vol'], float('inf')], labels=['Low Volatility', 'Normal', 'High Volatility'])
            df_latest.loc[mask, 'Trend_Status'] = pd.cut(df_latest.loc[mask, 'dist_from_sma'], bins=[-float('inf'), -2.0, 2.0, float('inf')], labels=['Below 20-SMA', 'Testing 20-SMA', 'Above 20-SMA'])

        facts = []
        
        target_ac = self._map_market_to_class(market)
        target_df = df_latest[df_latest['Asset_Class'] == target_ac]
        
        tradable = target_df.dropna(subset=['pct_change']).sort_values('pct_change', ascending=False)
        if len(tradable) > 0:
            best = tradable.iloc[0]
            facts.append(f"MARKET LEADER: Within {target_ac}s, the top performing asset was {best['Product Name']} ({best['Symbol']}) with a {best['pct_change']}% daily change ({best.get('weekly_change', 'N/A')}% weekly).")
            
            if len(tradable) > 1: 
                worst = tradable.iloc[-1]
                facts.append(f"MARKET LAGGARD: Within {target_ac}s, the worst performing asset was {worst['Product Name']} ({worst['Symbol']}) with a {worst['pct_change']}% daily change ({worst.get('weekly_change', 'N/A')}% weekly).")
        
        group_facts = []
        metrics = [
            ('Daily_Performance', 'pct_change', 'Performance'), 
            ('Weekly_Performance', 'weekly_change', 'Weekly'), 
            ('Trend_Status', 'dist_from_sma', 'Trend'), 
            ('Volatility_State', 'intraday_range', 'Volatility')
        ]
        
        market_size = len(target_df)
        if market_size >= 3: 
            for cat_col, val_col, f_type in metrics:
                dist = target_df.groupby(cat_col, observed=False).size()
                pcts = (dist / market_size) * 100
                means = target_df.groupby(cat_col, observed=False)[val_col].mean()
                
                for cat, pct in pcts.items():
                    if pct >= 60.0:
                        val = round(means[cat], 2) if pd.notna(means[cat]) else "N/A"
                        
                        if f_type == 'Performance':
                            text = f"GROUP FACT: A highly significant {round(pct,1)}% of {target_ac} assets exhibited a '{cat}' in daily closing price (Average daily change: {val}%)."
                        elif f_type == 'Weekly':
                            text = f"GROUP FACT: Over the 5-day horizon, {round(pct,1)}% of {target_ac} assets recorded a '{cat}' (Average weekly change: {val}%)."
                        elif f_type == 'Trend':
                            text = f"GROUP FACT: {round(pct,1)}% of {target_ac} assets are trending '{cat}', sitting an average of {val}% away from their 20-day moving average."
                        elif f_type == 'Volatility':
                            text = f"GROUP FACT: {round(pct,1)}% of {target_ac} assets showed '{cat}' intraday swings, with an average high-to-low range of {val}%."
                            
                        group_facts.append((pct, text))
        
        group_facts.sort(key=lambda x: x[0], reverse=True)
        facts.extend([f[1] for f in group_facts[:10]])
        
        return "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])

class DatasetConstructor:
    def __init__(self, instruction_template_path: str, data_root_dir: str, 
                 reports_path: str, examples_path: Optional[str] = None,
                 report_examples_path: Optional[str] = None,
                 num_shots: int = 3, num_report_examples: int = 3):
        
        self.instruction_template_path = instruction_template_path
        self.data_root_dir = data_root_dir
        self.reports_path = reports_path
        self.examples_path = examples_path
        self.report_examples_path = report_examples_path
        self.num_shots = num_shots
        self.num_report_examples = num_report_examples
        self.fact_extractor = InstanceFactExtractor()
        
        self.instruction = self._load_instruction()
        self.few_shot_examples = self._load_few_shot_examples() if examples_path else []
        self.report_examples = self._load_report_examples() if report_examples_path else []
        
    def _load_instruction(self) -> str:
        with open(self.instruction_template_path, 'r') as f: return f.read().strip()
    
    def _load_market_data(self, market: str, source: str, date_str: str) -> Optional[pd.DataFrame]:
        sub_dir = f"{'_'.join(market.split())}-{source}"
        file_path = os.path.join(self.data_root_dir, sub_dir, f"{pd.to_datetime(date_str).strftime('%Y-%m-%d')}.csv")
        try: return pd.read_csv(file_path)
        except FileNotFoundError: return None
        
    def _load_reports(self) -> pd.DataFrame:
        return pd.read_csv(self.reports_path, encoding="utf-16", sep="\t")
        
    def _format_data_example(self, example: Dict, index: int) -> str:
        return f"Data Example {index + 1}:\nMarket: {example['market']}\nDate: {example['date']}\nTable Data:\n{example['table_data']}\n\nReport:\n{example['report']}\n---"
        
    def _format_report_example(self, example: Dict, index: int) -> str:
        return f"Report Example {index + 1}:\nMarket: {example['market']}\nDate: {example['date']}\nReport:\n{example['report']}\n---"
        
    def _load_few_shot_examples(self) -> List[Dict]:
        return defaultdict(list) # Reference original function logic from DataTales
        
    def _load_report_examples(self) -> List[Dict]:
        return defaultdict(list) # Reference original function logic from DataTales

    def _format_input(self, table_data: str, market: str, date: str, facts: str) -> str:
        facts_block = f"\nExtracted Statistical Facts (Prioritize these insights):\n{facts}\n" if facts else ""
        return f"""Input:
Market: {market}
Date: {date}
Table Data:
{table_data}
{facts_block}
Generate a report based on the table data and extracted facts above."""

    def _construct_prompt_with_examples(self, table_data: str, market: str, source: str, date: str, facts: str) -> Dict:
        formatted_data_examples = [self._format_data_example(example, i) for i, example in enumerate(self.few_shot_examples.get(f"{market}-{source}", []))]
        formatted_report_examples = [self._format_report_example(example, i) for i, example in enumerate(self.report_examples.get(f"{market}-{source}", []))]
        
        formatted_input = self._format_input(table_data, market, date, facts)
        
        sections = [self.instruction]
        if formatted_data_examples:
            sections.append("\nData Examples:")
            sections.extend(formatted_data_examples)
        if formatted_report_examples:
            sections.append("\nReport Style Examples:")
            sections.extend(formatted_report_examples)
        sections.append("\n" + formatted_input)
        
        return {
            'instruction': self.instruction,
            'few_shot_examples': self.few_shot_examples,
            'report_examples': self.report_examples,
            'formatted_prompt': "\n".join(sections)
        }
    
    def construct_dataset(self) -> List[Dict]:
        dataset = []
        reports_df = self._load_reports()
        
        for _, row in reports_df.iterrows():
            market = row['market']
            date_str = row['date']
            source = row['source']
            report = row['passage']
            
            df = self._load_market_data(market, source, date_str)
            if df is None: continue
            
            facts_str = self.fact_extractor.extract_facts(df, market)
            
            table_data = df.to_string(index=False)
            
            prompts = self._construct_prompt_with_examples(table_data, market, source, date_str, facts_str)
                
            entry = {
                'source': source,
                'market': market,
                'date': date_str,
                'instruction': self.instruction,
                'table_data': table_data,
                'facts_data': facts_str,
                'report': report,
                'prompts': prompts
            }
            dataset.append(entry)
        return dataset
    
    def save_dataset(self, output_dir: str, split: str, dataset: Optional[List[Dict]] = None):
        if dataset is None: dataset = self.construct_dataset()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, f"{split}.json"), 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

def main():
    history_span = '1day'
    config = {
        'instruction_template_path': 'prompts/data2text_generation_task_instruction.txt',
        'data_root_dir': f'data/tabular_data/report_table_data/injected/{history_span}',
        'reports_path': 'data/reports/reports.tsv',
        'examples_path': 'data/reports/selected_sample_reports.tsv',
        'output_path': f'data/processed_dataset/injected/{history_span}',
        'num_report_examples': 2,
        'num_shots': 0
    }
    
    for split in ["train", "validate", "test"]:
        print(f"Constructing dataset for {split}...")
        
        constructor = DatasetConstructor(
            instruction_template_path=config['instruction_template_path'],
            data_root_dir=os.path.join(config['data_root_dir'], split),
            reports_path=config['reports_path'],
            examples_path=config['examples_path'],
            report_examples_path=config['examples_path'],
            num_shots=config['num_shots'],
            num_report_examples=config['num_report_examples']
        )
        
        dataset = constructor.construct_dataset()
        constructor.save_dataset(config['output_path'], split, dataset)
        print(f"Completed {split} split. Total entries: {len(dataset)}")

if __name__ == "__main__":
    main()