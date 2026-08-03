"""Dataset module for MLPR fine-tuning."""

import json
from typing import Dict, List, Optional
from datasets import load_dataset, DatasetDict


class MLPDataset:
    """
    Dataset class for loading and preparing KB triplets for MLPR training.
    
    The dataset consists of:
    - Memorization set (D_mem): Single-hop QA pairs for training
    - Generalization set (D_gen): Multi-hop QA pairs for evaluation only
    - Candidate set (A): Fixed set of entities for probe classification
    """
    
    def __init__(
        self,
        dataset_name: str,
        entity_vocab_path: Optional[str] = None,
        tokenizer=None,
        max_length: int = 512,
    ):
        """
        Initialize the MLPR dataset.
        
        Args:
            dataset_name: HuggingFace dataset name or local path
            entity_vocab_path: Path to entity vocabulary JSON file
            tokenizer: Tokenizer instance for preprocessing
            max_length: Maximum sequence length
        """
        self.dataset_name = dataset_name
        self.entity_vocab_path = entity_vocab_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.num_entities = 0
        
        # Load dataset from HuggingFace
        self.dataset = self._load_dataset()
        
        # Load entity vocabulary
        if entity_vocab_path:
            self._load_entity_vocab(entity_vocab_path)
    
    def _load_dataset(self) -> DatasetDict:
        """Load dataset from HuggingFace Hub."""
        try:
            dataset = load_dataset(self.dataset_name)
        except Exception:
            # If not available on HF, create a sample dataset for testing
            dataset = self._create_sample_dataset()
        return dataset
    
    def _create_sample_dataset(self) -> DatasetDict:
        """Create a sample dataset for testing purposes."""
        # Sample memorization data (single-hop QA)
        mem_data = [
            {
                "text": "Who is the regulator of Bank X? Answer: FinancialAuthority",
                "entity": "FinancialAuthority",
                "relation": "regulator_of",
                "head_entity": "Bank X",
                "entity_end_char_idx": len("Who is the regulator of Bank X? Answer: Financial"),
                "entity_class_id": 0,
                "type": "mem"
            },
            {
                "text": "What company issues Product Y? Answer: CorporationZ",
                "entity": "CorporationZ",
                "relation": "issues",
                "head_entity": "Product Y",
                "entity_end_char_idx": len("What company issues Product Y? Answer: Corporatio"),
                "entity_class_id": 1,
                "type": "mem"
            },
        ] * 500  # Repeat to get ~1000 samples
        
        # Sample generalization data (multi-hop QA)
        gen_data = [
            {
                "text": "Who regulates the issuer of Product Y? Answer: FinancialAuthority",
                "entity": "FinancialAuthority",
                "relations": ["issues", "regulator_of"],
                "head_entity": "Product Y",
                "entity_end_char_idx": len("Who regulates the issuer of Product Y? Answer: Financial"),
                "entity_class_id": 0,
                "type": "gen"
            },
            {
                "text": "Which products require checks A and B? Answer: ProductY",
                "entity": "ProductY",
                "relations": ["requires_check_A", "requires_check_B"],
                "head_entity": "checks A and B",
                "entity_end_char_idx": len("Which products require checks A and B? Answer: Produ"),
                "entity_class_id": 2,
                "type": "gen"
            },
        ] * 250  # Repeat to get ~500 samples
        
        from datasets import Dataset
        train_dataset = Dataset.from_list(mem_data)
        eval_dataset = Dataset.from_list(gen_data)
        
        return DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset,
            "mem": train_dataset,
            "gen": eval_dataset
        })
    
    def _load_entity_vocab(self, path: str) -> None:
        """Load entity vocabulary from JSON file."""
        try:
            with open(path, 'r') as f:
                self.entity2id = json.load(f)
            self.id2entity = {v: k for k, v in self.entity2id.items()}
            self.num_entities = len(self.entity2id)
        except FileNotFoundError:
            # Create a default vocabulary if file doesn't exist
            print(f"Entity vocab not found at {path}, creating default...")
            self.entity2id = {
                "FinancialAuthority": 0,
                "CorporationZ": 1,
                "ProductY": 2,
            }
            self.id2entity = {v: k for k, v in self.entity2id.items()}
            self.num_entities = len(self.entity2id)
    
    def get_train_dataset(self):
        """Get the memorization training dataset."""
        return self.dataset["train"]
    
    def get_eval_dataset(self):
        """Get the generalization evaluation dataset."""
        return self.dataset["eval"]
    
    def get_mem_dataset(self):
        """Get the memorization dataset."""
        return self.dataset.get("mem", self.dataset["train"])
    
    def get_gen_dataset(self):
        """Get the generalization dataset."""
        return self.dataset.get("gen", self.dataset["eval"])
    
    def prepare_dataset_for_tokenizer(self, dataset, tokenizer):
        """
        Prepare dataset with tokenization.
        
        Args:
            dataset: HuggingFace Dataset
            tokenizer: Tokenizer instance
            
        Returns:
            Tokenized dataset with input_ids, labels, entity_pos, entity_class_id
        """
        def tokenize_fn(example):
            # Tokenize the text
            encodings = tokenizer(
                example["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Calculate entity_pos (token index of anchor)
            # Map character index to token index
            char_idx = example.get("entity_end_char_idx", 0)
            token_idx = self._char_to_token_idx(encodings, char_idx)
            
            encodings["entity_pos"] = token_idx
            encodings["entity_class_id"] = example.get("entity_class_id", 0)
            encodings["labels"] = encodings["input_ids"].copy()
            
            return encodings
        
        return dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    
    def _char_to_token_idx(self, encodings: Dict, char_idx: int) -> int:
        """Convert character index to token index."""
        # Simple heuristic: find the token that contains the character position
        # In practice, you might want more sophisticated alignment
        word_ids = encodings.word_ids()
        if word_ids is None:
            return len(encodings["input_ids"]) // 2
        
        # Find which word/token contains the character position
        char_to_word = encodings.char_to_token
        if char_to_word:
            token_idx = char_to_word(char_idx)
            if token_idx is not None:
                return token_idx
        
        # Fallback: return middle token
        return len(encodings["input_ids"]) // 2
    
    @property
    def candidate_set_size(self) -> int:
        """Return the size of the candidate entity set."""
        return self.num_entities
