"""
Finance RAG — command-line entry point.

Commands:
  python main.py index            Parse all docs, embed, and store in ChromaDB.
  python main.py query "..."      Answer a single question and exit.
  python main.py chat             Interactive REPL (Ctrl-C or 'exit' to quit).
  python main.py serve            Launch the Gradio web UI.
"""

import os
import pickle
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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


def cmd_serve(args) -> None:
    import gradio as gr
    from src.generator import answer_from_chunks

    print("Loading retriever …")
    retriever = load_retriever()
    client    = get_llm_client()
    print("Ready. Starting Gradio …")

    def chat_fn(message: str, history: list) -> str:
        chunks   = retriever.retrieve(message, top_k=5)
        response = answer_from_chunks(message, chunks, client, LLM_MODEL)
        sources  = sorted({c.source for c in chunks})
        return response + "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Finance RAG",
        description="Ask questions about SEC filings and earnings reports from the corpus.",
        examples=[
            "What were the key risks mentioned in Apple's latest 10-K filing?",
            "What are the main products highlighted in Google's latest 10-K?",
            "What was Meta's total revenue for the year ended December 31, 2025?"
        ],
    )
    demo.launch(share=args.share)


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

    serve_parser = sub.add_parser("serve", help="Launch the Gradio web UI")
    serve_parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio URL (72 h)"
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
