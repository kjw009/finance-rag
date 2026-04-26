"""
Finance RAG — command-line entry point.

Commands:
  python main.py index            Parse all docs, embed, and store in ChromaDB.
  python main.py query "..."      Answer a single question and exit.
  python main.py chat             Interactive REPL (Ctrl-C or 'exit' to quit).
"""

import os
import pickle
import argparse
from pathlib import Path
from openai import OpenAI

from src.parser    import parse_all
from src.retriever import Retriever, build_index
from src.generator import answer, answer_from_chunks

DATA_DIR     = Path("datasets/client")
CHROMA_DIR   = "chroma_db"
COLLECTION   = "finance_rag"
CACHE_PATH   = Path("chunks_cache.pkl")
LLM_MODEL    = "llama-3.3-70b-versatile"


def get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )


def load_retriever() -> Retriever:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No chunk cache at {CACHE_PATH}. Run `python main.py index` first."
        )
    with open(CACHE_PATH, "rb") as f:
        chunks = pickle.load(f)
    return Retriever.from_index(CHROMA_DIR, COLLECTION, chunks)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_index(_args) -> None:
    print(f"Parsing documents in {DATA_DIR} …")
    chunks = parse_all(DATA_DIR)

    print(f"\nSaving {len(chunks)} chunks to {CACHE_PATH} …")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("\nBuilding vector index …")
    build_index(chunks, CHROMA_DIR, COLLECTION)
    print("\nDone. Run `python main.py query \"...\"` to start asking questions.")


def cmd_query(args) -> None:
    retriever = load_retriever()
    client    = get_llm_client()
    response  = answer(args.question, retriever, client, LLM_MODEL, top_k=args.top_k)
    print(f"\nQ: {args.question}\n")
    print(response)


def cmd_chat(_args) -> None:
    print("Loading retriever …")
    retriever = load_retriever()
    client    = get_llm_client()
    print("Ready. Type a question, or 'exit' to quit.\n")

    while True:
        try:
            query = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        response = answer(query, retriever, client, LLM_MODEL)
        print(f"\n{response}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="Parse all documents and build the search index")

    q_parser = sub.add_parser("query", help="Answer a single question")
    q_parser.add_argument("question", help="The question to answer")
    q_parser.add_argument("--top-k", type=int, default=5, dest="top_k")

    sub.add_parser("chat", help="Start an interactive Q&A session")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
