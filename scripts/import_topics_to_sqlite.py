import argparse
import json
import sqlite3
import os
from tqdm import tqdm

def import_topics(jsonl_path, db_path, batch_size=50000):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Input file not found: {jsonl_path}")
        
    conn = sqlite3.connect(db_path)
    
    print("Setting up sqlite configurations and schema...")
    # Fast insertions
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    
    # Create table for topics
    conn.execute('''
        CREATE TABLE IF NOT EXISTS document_topics (
            doc_id TEXT,
            topic TEXT,
            score REAL,
            PRIMARY KEY (doc_id, topic)
        ) WITHOUT ROWID;
    ''')
    
    # Create index for fast topic querying
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_document_topics_topic
        ON document_topics(topic);
    ''')
    
    conn.commit()
    
    # Get total lines for tqdm if possible
    print("Counting lines for progress bar...")
    total_lines = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
            
    batch = []
    total_inserted = 0
    
    print(f"Importing data from {jsonl_path} to {db_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines):
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            doc_id = data.get('doc_id')
            if doc_id is None:
                continue
                
            topics = data.get('topics', {})
            
            for topic, score in topics.items():
                batch.append((str(doc_id), str(topic), float(score)))
                
            if len(batch) >= batch_size:
                conn.executemany('''
                    INSERT OR REPLACE INTO document_topics (doc_id, topic, score)
                    VALUES (?, ?, ?)
                ''', batch)
                conn.commit()
                total_inserted += len(batch)
                batch.clear()
                
        if batch:
            conn.executemany('''
                INSERT OR REPLACE INTO document_topics (doc_id, topic, score)
                VALUES (?, ?, ?)
            ''', batch)
            conn.commit()
            total_inserted += len(batch)
            
    conn.close()
    print(f"Successfully inserted/updated {total_inserted} topic records in {db_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import topic classification to sqlite")
    parser.add_argument("--input", type=str, default="cache/topic_classification.jsonl", help="Path to jsonl file")
    parser.add_argument("--db", type=str, default="metadata_sqlite", help="Path to sqlite db")
    parser.add_argument("--batch-size", type=int, default=100000, help="Batch size for insertion")
    
    args = parser.parse_args()
    import_topics(args.input, args.db, args.batch_size)
