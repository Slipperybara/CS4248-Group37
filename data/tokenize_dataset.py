import datasets
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, List
import json
from dataclasses import dataclass
import pandas as pd
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizerFast,
    LlamaConfig,
    PreTrainedModel
)
import torch
from abc import ABC, abstractmethod

@dataclass
class TokenizationConfig:
    """Configuration for tokenization process"""
    model_name: str
    max_seq_length: int
    max_output_length: int
    context_length: Optional[int] = None
    skip_overlength: bool = True
    
    @property
    def max_context_length(self) -> int:
        return self.context_length or (self.max_seq_length - self.max_output_length)

class TokenizerFactory:
    """Factory class for creating tokenizers and configs"""
    @staticmethod
    def create(model_name: str) -> Tuple[PreTrainedTokenizer, AutoConfig]:
        if 'llama' in model_name.lower():
            return TokenizerFactory._create_llama_tokenizer(model_name)
        return TokenizerFactory._create_default_tokenizer(model_name)
    
    @staticmethod
    def _create_llama_tokenizer(model_name: str) -> Tuple[PreTrainedTokenizer, AutoConfig]:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        config = LlamaConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto'
        )
        return tokenizer, config
    
    @staticmethod
    def _create_default_tokenizer(model_name: str) -> Tuple[PreTrainedTokenizer, AutoConfig]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto'
        )
        return tokenizer, config

class DataFormatter:
    """Handles data formatting and preprocessing"""
    @staticmethod
    def format_context(instruction: str, date: str, table_data: Optional[str] = None) -> str:
        context = f"Instruction: {instruction}\n"
        if table_data:
            context += f"Generate report for date {date}\nTable:\n{table_data}\nReport:"
        return context
    
    @staticmethod
    def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
        df['context'] = df.apply(
            lambda row: DataFormatter.format_context(
                row['instruction'],
                row['date'],
                row.get('table_data')
            ),
            axis=1
        )
        df['target'] = df['report']
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')

class TokenizationProcessor:
    """Handles the tokenization of individual examples"""
    def __init__(self, tokenizer: PreTrainedTokenizer, max_context_length: int, max_output_length: int):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_output_length = max_output_length
    
    def process(self, context: str, target: str) -> Optional[Dict[str, torch.Tensor]]:
        context_ids = self.tokenizer.encode(
            context,
            truncation=True,
            max_length=self.max_context_length
        )
        
        target_ids = self.tokenizer.encode(
            target,
            truncation=True,
            max_length=self.max_output_length
        )
        
        if len(context_ids) + len(target_ids) > self.max_context_length + self.max_output_length:
            return None
            
        return {
            "input_ids": torch.tensor(context_ids),
            "labels": torch.tensor(target_ids),
            "attention_mask": torch.ones(len(context_ids), dtype=torch.long)
        }

class DatasetHandler:
    """Handles dataset operations including reading, processing and saving"""
    def __init__(self, tokenizer: PreTrainedTokenizer, config: TokenizationConfig):
        self.config = config
        self.processor = TokenizationProcessor(
            tokenizer,
            config.max_context_length,
            config.max_output_length
        )
    
    def process_dataset(self, input_path: Path, output_path: Path):
        df = self._read_dataset(input_path)
        dataset = self._create_dataset(df)
        dataset.save_to_disk(output_path)
    
    def _read_dataset(self, input_path: Path) -> pd.DataFrame:
        with open(input_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        df = pd.DataFrame(data_list)
        return DataFormatter.preprocess_dataset(df)
    
    def _create_dataset(self, df: pd.DataFrame) -> datasets.Dataset:
        return datasets.Dataset.from_generator(
            lambda: self._generate_examples(df),
            features=datasets.Features({
                "input_ids": datasets.Sequence(datasets.Value("int64")),
                "attention_mask": datasets.Sequence(datasets.Value("int64")),
                "labels": datasets.Sequence(datasets.Value("int64"))
            })
        )
    
    def _generate_examples(self, df: pd.DataFrame) -> Iterator[Optional[Dict]]:
        for _, row in df.iterrows():
            result = self.processor.process(row['context'], row['target'])
            if result is not None or not self.config.skip_overlength:
                yield result

class TokenizationPipeline:
    """Main pipeline for orchestrating the tokenization process"""
    def __init__(
        self,
        model_name: str,
        input_base_path: Path,
        output_base_path: Path,
        max_seq_length: int = 4096,
        max_output_length: int = 512
    ):
        self.config = TokenizationConfig(
            model_name=model_name,
            max_seq_length=max_seq_length,
            max_output_length=max_output_length
        )
        self.input_base_path = input_base_path
        self.output_base_path = self._setup_output_path(output_base_path)
        self.tokenizer, self.model_config = TokenizerFactory.create(model_name)
        self.dataset_handler = DatasetHandler(self.tokenizer, self.config)
    
    def _setup_output_path(self, base_path: Path) -> Path:
        model_name = self.config.model_name.split('/')[-1]
        output_path = base_path / model_name
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def process_splits(self, splits: List[str] = ['train', 'validate', 'test']):
        for split in splits:
            print(f"Processing {split} split...")
            input_path = self.input_base_path / f"{split}.json"
            output_path = self.output_base_path / split
            
            if input_path.exists():
                self.dataset_handler.process_dataset(input_path, output_path)
                print(f"Completed tokenization for {split} split")
            else:
                print(f"Warning: {split} split not found at {input_path}")

def main():
    # input_base_path = Path("data/processed_dataset/1day")
    # output_base_path = Path("data/tokenized_dataset/1day")
    input_base_path = Path("data/processed_dataset/1week")
    output_base_path = Path("data/tokenized_dataset/1week")
    
    pipeline = TokenizationPipeline(
        model_name="daryl149/llama-2-7b-chat-hf",
        input_base_path=input_base_path,
        output_base_path=output_base_path,
        max_seq_length=4096,
        max_output_length=512
    )
    
    pipeline.process_splits()

if __name__ == "__main__":
    main()