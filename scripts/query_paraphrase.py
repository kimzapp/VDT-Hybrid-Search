"""
Query Paraphrase Strategies for MS MARCO Passage Robustness Evaluation
======================================================================

This script generates paraphrased versions of MS MARCO passage queries
using 4 different strategies to evaluate retrieval system robustness:

1. Synonym Substitution (WordNet) — tests vocabulary sensitivity
2. Back-Translation (Helsinki NLP EN→DE→EN) — tests linguistic variation
3. T5 Paraphrase (Vamsi/T5_Paraphrase_Paws) — tests semantic rewriting
4. LLM Rewrite (Qwen3-4B) — tests complex reformulation

Each strategy saves queries as a TSV file loadable by ir_datasets.

Usage:
    python scripts/query_paraphrase.py --strategy all
    python scripts/query_paraphrase.py --strategy synonym
    python scripts/query_paraphrase.py --strategy backtranslation t5
"""

import argparse
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ir_datasets
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAPHRASE_DIR = PROJECT_ROOT / "paraphrased_queries"
DEFAULT_EVAL_ID = "msmarco-passage/dev/small"
DEFAULT_CORPUS_ID = "msmarco-passage"

STRATEGY_NAMES = ["synonym", "backtranslation", "t5", "llm_rewrite"]

STRATEGY_FILE_MAP = {
    "synonym": "synonym_queries.tsv",
    "backtranslation": "backtranslation_queries.tsv",
    "t5": "t5_paraphrase_queries.tsv",
    "llm_rewrite": "llm_rewrite_queries.tsv",
}


# =============================================================================
# Strategy 1: Synonym Substitution (WordNet)
# =============================================================================

class SynonymParaphraser:
    """
    Replace 1-2 content words with WordNet synonyms.
    Tests whether the retrieval system is sensitive to vocabulary choices.
    """

    def __init__(self, max_replacements: int = 2, seed: int = 42):
        import nltk
        from nltk.corpus import wordnet

        # Ensure required NLTK data is downloaded
        for resource in ["wordnet", "averaged_perceptron_tagger_eng", "punkt_tab"]:
            nltk.download(resource, quiet=True)

        self.wordnet = wordnet
        self.nltk = nltk
        self.max_replacements = max_replacements
        self.rng = random.Random(seed)
        # POS tag mapping: nltk POS -> WordNet POS
        self._pos_map = {
            "NN": wordnet.NOUN, "NNS": wordnet.NOUN,
            "NNP": wordnet.NOUN, "NNPS": wordnet.NOUN,
            "VB": wordnet.VERB, "VBD": wordnet.VERB,
            "VBG": wordnet.VERB, "VBN": wordnet.VERB,
            "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
            "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
            "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV,
        }

    def _get_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """Get synonyms for a word from WordNet."""
        wn_pos = self._pos_map.get(pos)
        if wn_pos:
            synsets = self.wordnet.synsets(word, pos=wn_pos)
        else:
            synsets = self.wordnet.synsets(word)

        synonyms = set()
        for synset in synsets:
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    synonyms.add(name)
        return list(synonyms)

    def paraphrase(self, text: str) -> str:
        """Replace up to max_replacements content words with synonyms."""
        tokens = self.nltk.word_tokenize(text)
        tagged = self.nltk.pos_tag(tokens)

        # Find candidate positions (content words with synonyms)
        candidates = []
        for i, (word, pos) in enumerate(tagged):
            if pos in self._pos_map and len(word) > 2:
                syns = self._get_synonyms(word, pos)
                if syns:
                    candidates.append((i, word, syns))

        if not candidates:
            return text

        # Randomly pick 1-max_replacements candidates
        n_replace = min(self.max_replacements, len(candidates))
        chosen = self.rng.sample(candidates, n_replace)

        result = list(tokens)
        for idx, original, syns in chosen:
            replacement = self.rng.choice(syns)
            # Preserve capitalization
            if original[0].isupper():
                replacement = replacement.capitalize()
            result[idx] = replacement

        return " ".join(result)

    def paraphrase_batch(self, queries: Dict[str, str]) -> Dict[str, str]:
        """Paraphrase a batch of queries."""
        paraphrased = OrderedDict()
        for qid, text in tqdm(queries.items(), desc="Synonym substitution"):
            paraphrased[qid] = self.paraphrase(text)
        return paraphrased


# =============================================================================
# Strategy 2: Back-Translation (Helsinki NLP)
# =============================================================================

class BackTranslationParaphraser:
    """
    Translate queries EN→DE→EN using Helsinki NLP MarianMT models.
    Tests robustness to natural linguistic variation.
    """

    def __init__(
        self,
        pivot_lang: str = "de",
        device: str = "cuda",
        batch_size: int = 64,
    ):
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device
        self.batch_size = batch_size

        # Load EN→pivot model
        en2pivot_name = f"Helsinki-NLP/opus-mt-en-{pivot_lang}"
        print(f"   Loading {en2pivot_name}...")
        self.en2pivot_tok = MarianTokenizer.from_pretrained(en2pivot_name)
        self.en2pivot_model = MarianMTModel.from_pretrained(en2pivot_name).to(device)
        self.en2pivot_model.eval()

        # Load pivot→EN model
        pivot2en_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-en"
        print(f"   Loading {pivot2en_name}...")
        self.pivot2en_tok = MarianTokenizer.from_pretrained(pivot2en_name)
        self.pivot2en_model = MarianMTModel.from_pretrained(pivot2en_name).to(device)
        self.pivot2en_model.eval()

        print(f"   Back-translation models loaded (pivot: {pivot_lang})")

    def _translate_batch(self, texts: List[str], tokenizer, model) -> List[str]:
        """Translate a batch of texts."""
        import torch

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(self.device)
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=128)
            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            results.extend(decoded)
        return results

    def paraphrase_batch(self, queries: Dict[str, str]) -> Dict[str, str]:
        """Back-translate a batch of queries EN→pivot→EN."""
        qids = list(queries.keys())
        texts = list(queries.values())

        print("   Step 1/2: Translating EN → pivot...")
        pivot_texts = self._translate_batch(texts, self.en2pivot_tok, self.en2pivot_model)

        print("   Step 2/2: Translating pivot → EN...")
        back_texts = self._translate_batch(pivot_texts, self.pivot2en_tok, self.pivot2en_model)

        paraphrased = OrderedDict()
        for qid, text in zip(qids, back_texts):
            paraphrased[qid] = text
        return paraphrased


# =============================================================================
# Strategy 3: T5 Paraphrase
# =============================================================================

class T5Paraphraser:
    """
    Use fine-tuned T5 model for high-quality paraphrase generation.
    Model: Vamsi/T5_Paraphrase_Paws (fine-tuned on PAWS dataset).
    Tests robustness to semantic rewriting.
    """

    def __init__(
        self,
        model_name: str = "Vamsi/T5_Paraphrase_Paws",
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 128,
        num_beams: int = 5,
    ):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        print(f"   Loading T5 paraphrase model: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        print(f"   T5 paraphrase model loaded")

    def paraphrase_batch(self, queries: Dict[str, str]) -> Dict[str, str]:
        """Generate paraphrases for a batch of queries using T5."""
        import torch

        qids = list(queries.keys())
        texts = [f"paraphrase: {q} </s>" for q in queries.values()]

        results = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="T5 paraphrase"):
            batch = texts[i : i + self.batch_size]
            encoding = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

        paraphrased = OrderedDict()
        for qid, text in zip(qids, results):
            paraphrased[qid] = text
        return paraphrased


# =============================================================================
# Strategy 4: LLM Rewrite (Qwen3-4B)
# =============================================================================

class LLMRewriteParaphraser:
    """
    Use Qwen3-4B to rewrite queries with complex reformulation.
    Tests robustness to extensive query reformulation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 64,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"   Loading LLM model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        print(f"   LLM model loaded")

    def _build_prompt(self, query: str) -> str:
        """Build a prompt for query rewriting."""
        return (
            f"Rewrite the following search query using different words while keeping "
            f"the exact same meaning. Output ONLY the rewritten query, nothing else.\n\n"
            f"Original: {query}\n"
            f"Rewritten:"
        )

    def _extract_rewritten(self, full_output: str, original_query: str) -> str:
        """Extract the rewritten query from model output."""
        # Try to find content after "Rewritten:"
        if "Rewritten:" in full_output:
            result = full_output.split("Rewritten:")[-1].strip()
        else:
            result = full_output.strip()

        # Clean up: take only the first line, remove quotes
        result = result.split("\n")[0].strip().strip('"').strip("'").strip()

        # Fallback if empty
        if not result:
            result = original_query
        return result

    def paraphrase_batch(self, queries: Dict[str, str]) -> Dict[str, str]:
        """Rewrite queries using LLM."""
        import torch

        qids = list(queries.keys())
        texts = list(queries.values())

        results = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="LLM rewrite"):
            batch_texts = texts[i : i + self.batch_size]
            batch_prompts = [self._build_prompt(q) for q in batch_texts]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )

            # Decode only the generated tokens (skip input tokens)
            input_length = inputs["input_ids"].shape[1]
            generated = outputs[:, input_length:]
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

            for text, original in zip(decoded, batch_texts):
                results.append(self._extract_rewritten(text, original))

        paraphrased = OrderedDict()
        for qid, text in zip(qids, results):
            paraphrased[qid] = text
        return paraphrased


# =============================================================================
# I/O: Save & Load TSV
# =============================================================================

def save_queries_tsv(queries: Dict[str, str], filepath: str):
    """Save queries as TSV file (query_id\ttext) — ir_datasets TsvQueries format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for qid, text in queries.items():
            # Ensure no tabs/newlines in text
            clean_text = text.replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{qid}\t{clean_text}\n")
    print(f"   Saved {len(queries):,} queries → {filepath}")


def load_queries_tsv(filepath: str) -> Dict[str, str]:
    """Load queries from TSV file."""
    queries = OrderedDict()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def save_metadata(strategies_run: List[str], eval_id: str, output_dir: str):
    """Save metadata about the paraphrase generation run."""
    meta = {
        "eval_id": eval_id,
        "strategies": strategies_run,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {s: STRATEGY_FILE_MAP[s] for s in strategies_run},
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"   Metadata saved → {meta_path}")


# =============================================================================
# ir_datasets Registration
# =============================================================================

class _InMemoryQueriesHandler:
    """
    A minimal queries handler that serves queries from a TSV file on disk.
    Compatible with ir_datasets Dataset constituent interface.
    """

    def __init__(self, tsv_path: str):
        self._tsv_path = tsv_path
        self._queries = None

    def _ensure_loaded(self):
        if self._queries is None:
            from ir_datasets.formats import GenericQuery
            self._queries = []
            with open(self._tsv_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        self._queries.append(GenericQuery(query_id=parts[0], text=parts[1]))

    def queries_iter(self):
        self._ensure_loaded()
        return iter(self._queries)

    def queries_count(self):
        self._ensure_loaded()
        return len(self._queries)

    def queries_cls(self):
        from ir_datasets.formats import GenericQuery
        return GenericQuery

    def queries_namespace(self):
        return None

    def queries_lang(self):
        return "en"

    def queries_handler(self):
        return self


def register_paraphrased_datasets(
    eval_id: str = DEFAULT_EVAL_ID,
    paraphrase_dir: Optional[str] = None,
):
    """
    Register all available paraphrased query subsets with ir_datasets.

    After calling this function, you can load paraphrased datasets like:
        ds = ir_datasets.load("msmarco-passage/dev/small/paraphrased/synonym")
        ds = ir_datasets.load("msmarco-passage/dev/small/paraphrased/t5")

    Each dataset shares docs and qrels from the original eval_id,
    but uses paraphrased queries instead.
    """
    from ir_datasets.datasets.base import Dataset

    if paraphrase_dir is None:
        paraphrase_dir = str(PARAPHRASE_DIR)

    # Load the original eval dataset to borrow docs and qrels handlers.
    # Note: original_ds inherits docs from the base corpus via ir_datasets,
    # so we don't need to separately load the corpus (which is very slow).
    original_ds = ir_datasets.load(eval_id)

    registered = []

    for strategy_name, tsv_filename in STRATEGY_FILE_MAP.items():
        tsv_path = os.path.join(paraphrase_dir, tsv_filename)
        if not os.path.exists(tsv_path):
            continue

        dataset_id = f"{eval_id}/paraphrased/{strategy_name}"

        # Build Dataset using ir_datasets' own Dataset class with constituents.
        # Dataset.__init__ accepts *constituents — objects that provide
        # *_handler() methods. It uses duck typing to discover capabilities.
        # We pass: our custom queries handler + original_ds (provides docs & qrels).
        queries_constituent = _InMemoryQueriesHandler(tsv_path)

        dataset = Dataset(
            queries_constituent,  # provides queries_handler
            original_ds,          # provides docs_handler & qrels_handler
        )

        ir_datasets.registry.register(dataset_id, dataset)
        registered.append(dataset_id)

    if registered:
        print(f"   Registered {len(registered)} paraphrased datasets:")
        for ds_id in registered:
            print(f"     - {ds_id}")
    else:
        print("   No paraphrased TSV files found. Run generation first.")

    return registered


# =============================================================================
# Main Generation Pipeline
# =============================================================================

def load_original_queries(eval_id: str) -> Dict[str, str]:
    """Load original queries from ir_datasets."""
    ds = ir_datasets.load(eval_id)
    queries = OrderedDict()
    for q in ds.queries_iter():
        queries[q.query_id] = q.text
    print(f"   Loaded {len(queries):,} original queries from {eval_id}")
    return queries


def generate_paraphrases(
    strategies: List[str],
    eval_id: str = DEFAULT_EVAL_ID,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 64,
    pivot_lang: str = "de",
):
    """Generate paraphrased queries for specified strategies."""
    if output_dir is None:
        output_dir = str(PARAPHRASE_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Load original queries
    print("\n" + "=" * 70)
    print("📥 Loading original queries")
    print("=" * 70)
    queries = load_original_queries(eval_id)

    # Save original queries as reference
    save_queries_tsv(queries, os.path.join(output_dir, "original_queries.tsv"))

    strategies_run = []

    for strategy in strategies:
        print("\n" + "=" * 70)
        print(f"🔄 Strategy: {strategy}")
        print("=" * 70)

        t_start = time.time()

        if strategy == "synonym":
            paraphraser = SynonymParaphraser(max_replacements=2, seed=seed)
            paraphrased = paraphraser.paraphrase_batch(queries)

        elif strategy == "backtranslation":
            paraphraser = BackTranslationParaphraser(
                pivot_lang=pivot_lang,
                device=device,
                batch_size=batch_size,
            )
            paraphrased = paraphraser.paraphrase_batch(queries)

        elif strategy == "t5":
            paraphraser = T5Paraphraser(
                device=device,
                batch_size=batch_size,
            )
            paraphrased = paraphraser.paraphrase_batch(queries)

        elif strategy == "llm_rewrite":
            paraphraser = LLMRewriteParaphraser(
                device=device,
                batch_size=min(batch_size, 8),  # LLM needs smaller batches
            )
            paraphrased = paraphraser.paraphrase_batch(queries)

        else:
            print(f"   ⚠️  Unknown strategy: {strategy}, skipping")
            continue

        elapsed = time.time() - t_start

        # Save to TSV
        tsv_path = os.path.join(output_dir, STRATEGY_FILE_MAP[strategy])
        save_queries_tsv(paraphrased, tsv_path)
        strategies_run.append(strategy)

        # Print sample comparisons
        print(f"\n   ⏱️  Time: {elapsed:.1f}s")
        print(f"\n   📋 Sample comparisons (first 5):")
        for i, qid in enumerate(list(queries.keys())[:5]):
            orig = queries[qid]
            para = paraphrased[qid]
            changed = "✅" if orig.lower() != para.lower() else "⬜"
            print(f"   {changed} [{qid}]")
            print(f"      Original:    {orig}")
            print(f"      Paraphrased: {para}")

    # Save metadata
    if strategies_run:
        save_metadata(strategies_run, eval_id, output_dir)

    return strategies_run


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paraphrased queries for MS MARCO passage robustness evaluation"
    )
    parser.add_argument(
        "--strategy",
        nargs="+",
        default=["all"],
        choices=STRATEGY_NAMES + ["all"],
        help="Paraphrase strategies to run. Use 'all' for all strategies.",
    )
    parser.add_argument(
        "--eval_id",
        type=str,
        default=DEFAULT_EVAL_ID,
        help="ir_datasets eval set ID (default: msmarco-passage/dev/small)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PARAPHRASE_DIR),
        help="Output directory for paraphrased query files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference (cuda or cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--pivot_lang",
        type=str,
        default="de",
        help="Pivot language for back-translation (default: de)",
    )
    parser.add_argument(
        "--register_only",
        action="store_true",
        help="Only register existing paraphrased datasets, don't generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.register_only:
        registered = register_paraphrased_datasets(
            eval_id=args.eval_id,
            paraphrase_dir=args.output_dir,
        )
        # Quick verification
        for ds_id in registered:
            ds = ir_datasets.load(ds_id)
            print(f"   ✅ {ds_id}: {ds.queries_count()} queries")
        return

    strategies = STRATEGY_NAMES if "all" in args.strategy else args.strategy

    print("=" * 70)
    print("🚀 QUERY PARAPHRASE GENERATION")
    print("=" * 70)
    print(f"   Eval set:    {args.eval_id}")
    print(f"   Strategies:  {strategies}")
    print(f"   Output dir:  {args.output_dir}")
    print(f"   Device:      {args.device}")
    print(f"   Batch size:  {args.batch_size}")
    print(f"   Pivot lang:  {args.pivot_lang}")

    strategies_run = generate_paraphrases(
        strategies=strategies,
        eval_id=args.eval_id,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        pivot_lang=args.pivot_lang,
    )

    # Register and verify
    if strategies_run:
        print("\n" + "=" * 70)
        print("📦 Registering paraphrased datasets with ir_datasets")
        print("=" * 70)
        registered = register_paraphrased_datasets(
            eval_id=args.eval_id,
            paraphrase_dir=args.output_dir,
        )

        print("\n" + "=" * 70)
        print("✅ VERIFICATION")
        print("=" * 70)
        for ds_id in registered:
            ds = ir_datasets.load(ds_id)
            count = ds.queries_count()
            print(f"   ✅ ir_datasets.load('{ds_id}') → {count:,} queries")
            # Show first query as sanity check
            first = next(ds.queries_iter())
            print(f"      First query: [{first.query_id}] {first.text[:80]}...")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
