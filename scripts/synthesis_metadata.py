from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import sqlite3
from typing import Iterator, Iterable


FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
    "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Margaret", "Betty", "Sandra",
    "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Carol", "Amanda",
    "Melissa", "Deborah", "Stephanie", "Rebecca", "Laura", "Sharon", "Cynthia",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Turner", "Phillips", "Parker", "Evans", "Edwards",
]


def stable_hash_u128(value: str, seed: int) -> int:
    """
    Hash ổn định, không phụ thuộc vào Python hash randomization.
    Cùng doc_id + seed sẽ luôn sinh cùng metadata.
    """
    h = hashlib.blake2b(digest_size=16, person=b"msmarco-meta")
    h.update((seed & ((1 << 64) - 1)).to_bytes(8, "little"))
    h.update(value.encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), "little")


def build_author_pool(num_authors: int) -> list[tuple[int, str]]:
    """
    Sinh danh sách author giả.
    author_id bắt đầu từ 1.
    """
    authors = []
    base_size = len(FIRST_NAMES) * len(LAST_NAMES)

    for idx in range(num_authors):
        first = FIRST_NAMES[idx % len(FIRST_NAMES)]
        last = LAST_NAMES[(idx // len(FIRST_NAMES)) % len(LAST_NAMES)]
        cycle = idx // base_size

        if cycle == 0:
            name = f"{first} {last}"
        else:
            # Thêm middle initial/suffix để giảm trùng tên khi num_authors lớn.
            middle = chr(ord("A") + (cycle % 26))
            suffix = cycle // 26
            if suffix == 0:
                name = f"{first} {middle}. {last}"
            else:
                name = f"{first} {middle}{suffix}. {last}"

        authors.append((idx + 1, name))

    return authors


def generate_metadata(
    doc_id: str,
    num_authors: int,
    start_date: dt.date,
    end_date: dt.date,
    seed: int,
) -> tuple[int, str]:
    """
    Trả về:
        author_id, written_date

    written_date được lưu dạng ISO: YYYY-MM-DD.
    """
    x = stable_hash_u128(doc_id, seed)

    author_id = int(x % num_authors) + 1

    start_ord = start_date.toordinal()
    end_ord = end_date.toordinal()
    num_days = end_ord - start_ord + 1

    date_offset = int((x >> 32) % num_days)
    written_date = dt.date.fromordinal(start_ord + date_offset).isoformat()

    return author_id, written_date


def iter_doc_ids_from_ir_datasets(dataset_name: str) -> Iterator[str]:
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)

    for doc in dataset.docs_iter():
        if hasattr(doc, "doc_id"):
            yield str(doc.doc_id)
        elif hasattr(doc, "docno"):
            yield str(doc.docno)
        elif hasattr(doc, "id"):
            yield str(doc.id)
        else:
            raise ValueError(f"Cannot find doc_id field in doc object: {doc}")


def iter_doc_ids_from_tsv(
    path: str,
    docid_col: int = 0,
    delimiter: str = "\t",
    has_header: bool = False,
) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            next(reader, None)

        for row in reader:
            if not row:
                continue
            yield str(row[docid_col])


def configure_sqlite(conn: sqlite3.Connection, fast_unsafe: bool = False) -> None:
    """
    fast_unsafe=True nhanh hơn nhưng kém an toàn nếu mất điện/crash giữa chừng.
    Mặc định dùng WAL + synchronous=NORMAL cho cân bằng tốc độ/an toàn.
    """
    if fast_unsafe:
        conn.execute("PRAGMA journal_mode=OFF;")
        conn.execute("PRAGMA synchronous=OFF;")
    else:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-262144;")  # khoảng 256MB cache
    conn.execute("PRAGMA foreign_keys=OFF;")


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS authors (
            author_id INTEGER PRIMARY KEY,
            author_name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS passage_metadata (
            doc_id TEXT PRIMARY KEY,
            author_id INTEGER NOT NULL,
            written_date TEXT NOT NULL,
            source_dataset TEXT NOT NULL
        ) WITHOUT ROWID;

        DROP VIEW IF EXISTS passage_metadata_full;

        CREATE VIEW passage_metadata_full AS
        SELECT
            m.doc_id,
            a.author_name,
            m.written_date,
            m.source_dataset
        FROM passage_metadata m
        JOIN authors a ON m.author_id = a.author_id;
        """
    )


def create_indexes(conn: sqlite3.Connection) -> None:
    """
    Tạo index sau khi insert xong sẽ nhanh hơn tạo trước rồi mới insert.
    """
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_passage_metadata_author_id
        ON passage_metadata(author_id);

        CREATE INDEX IF NOT EXISTS idx_passage_metadata_written_date
        ON passage_metadata(written_date);

        CREATE INDEX IF NOT EXISTS idx_passage_metadata_author_date
        ON passage_metadata(author_id, written_date);

        ANALYZE;
        """
    )


def populate_authors(conn: sqlite3.Connection, num_authors: int) -> None:
    authors = build_author_pool(num_authors)
    conn.executemany(
        """
        INSERT OR IGNORE INTO authors(author_id, author_name)
        VALUES (?, ?);
        """,
        authors,
    )


def insert_metadata(
    conn: sqlite3.Connection,
    doc_ids: Iterable[str],
    source_dataset: str,
    num_authors: int,
    start_date: dt.date,
    end_date: dt.date,
    seed: int,
    batch_size: int,
    commit_every_batches: int,
    insert_mode: str,
    limit: int | None,
) -> int:
    insert_mode = insert_mode.upper()
    if insert_mode not in {"IGNORE", "REPLACE"}:
        raise ValueError("insert_mode must be IGNORE or REPLACE")

    sql = f"""
        INSERT OR {insert_mode}
        INTO passage_metadata(doc_id, author_id, written_date, source_dataset)
        VALUES (?, ?, ?, ?);
    """

    total = 0
    batch = []
    batch_count_since_commit = 0

    conn.execute("BEGIN;")

    for doc_id in doc_ids:
        author_id, written_date = generate_metadata(
            doc_id=doc_id,
            num_authors=num_authors,
            start_date=start_date,
            end_date=end_date,
            seed=seed,
        )

        batch.append((doc_id, author_id, written_date, source_dataset))

        if len(batch) >= batch_size:
            conn.executemany(sql, batch)
            total += len(batch)
            batch.clear()

            batch_count_since_commit += 1
            if batch_count_since_commit >= commit_every_batches:
                conn.commit()
                print(f"Inserted {total:,} rows")
                conn.execute("BEGIN;")
                batch_count_since_commit = 0

            if limit is not None and total >= limit:
                break

    if batch and (limit is None or total < limit):
        if limit is not None:
            remaining = limit - total
            batch = batch[:remaining]

        conn.executemany(sql, batch)
        total += len(batch)

    conn.commit()
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fake metadata for MS MARCO Passage Ranking and store it in SQLite."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco-passage",
        help="ir_datasets dataset name. Default: msmarco-passage",
    )

    parser.add_argument(
        "--input-tsv",
        type=str,
        default=None,
        help="Optional TSV file. If provided, read doc_id from this file instead of ir_datasets.",
    )

    parser.add_argument(
        "--docid-col",
        type=int,
        default=0,
        help="Column index of doc_id when using --input-tsv. Default: 0",
    )

    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Use this if --input-tsv has header.",
    )

    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Output SQLite database path.",
    )

    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Optional source name stored in SQLite.",
    )

    parser.add_argument(
        "--num-authors",
        type=int,
        default=20_000,
        help="Number of fake authors. Default: 20000",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Minimum written_date. Format: YYYY-MM-DD",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="Maximum written_date. Format: YYYY-MM-DD",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic metadata generation.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Number of rows per executemany batch. Default: 50000",
    )

    parser.add_argument(
        "--commit-every-batches",
        type=int,
        default=10,
        help="Commit after this many batches. Default: 10",
    )

    parser.add_argument(
        "--insert-mode",
        type=str,
        default="IGNORE",
        choices=["IGNORE", "REPLACE"],
        help="IGNORE keeps existing metadata; REPLACE overwrites it.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N documents. Useful for testing.",
    )

    parser.add_argument(
        "--fast-unsafe",
        action="store_true",
        help="Faster SQLite pragmas, but less safe if interrupted.",
    )

    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip secondary indexes after insertion.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    if start_date > end_date:
        raise ValueError("--start-date must be <= --end-date")

    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)

    if args.input_tsv:
        doc_ids = iter_doc_ids_from_tsv(
            path=args.input_tsv,
            docid_col=args.docid_col,
            has_header=args.has_header,
        )
        source_dataset = args.source_name or f"tsv:{os.path.basename(args.input_tsv)}"
    else:
        doc_ids = iter_doc_ids_from_ir_datasets(args.dataset)
        source_dataset = args.source_name or args.dataset

    conn = sqlite3.connect(args.db)
    configure_sqlite(conn, fast_unsafe=args.fast_unsafe)

    print("Creating schema...")
    create_schema(conn)

    print(f"Populating {args.num_authors:,} fake authors...")
    populate_authors(conn, args.num_authors)
    conn.commit()

    print("Inserting passage metadata...")
    total = insert_metadata(
        conn=conn,
        doc_ids=doc_ids,
        source_dataset=source_dataset,
        num_authors=args.num_authors,
        start_date=start_date,
        end_date=end_date,
        seed=args.seed,
        batch_size=args.batch_size,
        commit_every_batches=args.commit_every_batches,
        insert_mode=args.insert_mode,
        limit=args.limit,
    )

    if not args.skip_indexes:
        print("Creating indexes...")
        create_indexes(conn)

    conn.close()

    print(f"Done. Inserted {total:,} metadata rows into {args.db}")


if __name__ == "__main__":
    main()