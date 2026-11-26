# QGIS2OllamaMCP Interactive Client
# RAG + Gemma 4 + MCP + QGIS

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import ollama

from fastmcp import Client as MCPClient


# ============================================================
# Configuration
# ============================================================

OLLAMA_MODEL = "gemma4:latest"
EMBED_MODEL = "nomic-embed-text:latest"

MCP_SERVER_URL = "http://127.0.0.1:9877/sse"

KNOWLEDGE_FILE = Path(
    "rag/knowledge/project_knowledge.txt"
)


# ============================================================
# RAG: Load knowledge base
# ============================================================

def load_knowledge():
    """
    Load project knowledge and split it using SECTION headings.
    """

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    text = KNOWLEDGE_FILE.read_text(
        encoding="utf-8"
    )

    chunks = []
    current_chunk = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("SECTION "):

            if current_chunk:
                chunks.append(
                    "\n".join(current_chunk).strip()
                )

            current_chunk = [line]

        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append(
            "\n".join(current_chunk).strip()
        )

    return chunks


# ============================================================
# RAG: Create embedding
# ============================================================

def embed(text):
    """
    Generate an embedding using nomic-embed-text through Ollama.
    """

    response = ollama.embed(
        model=EMBED_MODEL,
        input=text,
    )

    return np.array(
        response["embeddings"][0],
        dtype=float,
    )


# ============================================================
# RAG: Prepare knowledge embeddings
# ============================================================

def prepare_rag():
    """
    Load and embed all project knowledge chunks.
    """

    print("📚 Loading RAG knowledge base...")

    chunks = load_knowledge()

    embeddings = []

    for chunk in chunks:
        embeddings.append(
            embed(chunk)
        )

    print(
        f"✅ RAG loaded {len(chunks)} knowledge chunks.\n"
    )

    return chunks, embeddings


# ============================================================
# RAG: Retrieve top 3 chunks
# ============================================================

def retrieve_context(
    query,
    chunks,
    embeddings,
):
    """
    Retrieve the three chunks most semantically similar
    to the user query using cosine similarity.
    """

    query_embedding = embed(query)

    query_norm = np.linalg.norm(
        query_embedding
    )

    scores = []

    for i, chunk_embedding in enumerate(
        embeddings
    ):

        chunk_norm = np.linalg.norm(
            chunk_embedding
        )

        if query_norm == 0 or chunk_norm == 0:
            similarity = 0.0

        else:
            similarity = np.dot(
                query_embedding,
                chunk_embedding
            ) / (
                query_norm
                * chunk_norm
            )

        scores.append(
            (
                float(similarity),
                chunks[i],
            )
        )

    scores.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    top_results = scores[:3]

    retrieved_chunks = [
        chunk
        for score, chunk in top_results
    ]

    retrieved_context = "\n\n".join(
        retrieved_chunks
    )

    if top_results:
        best_score = top_results[0][0]
    else:
        best_score = 0.0

    return retrieved_context, best_score


# ============================================================
# MCP: Discover available tools
# ============================================================

async def load_mcp_tools():
    """
    Connect to MCP server and retrieve available QGIS tools.
    """

    try:

        async with MCPClient(
            MCP_SERVER_URL
        ) as mcp:

            tools_list = await mcp.list_tools()

            ollama_tools = []

            for tool in tools_list:

                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })

            return ollama_tools

    except Exception as e:

        print(
            f"❌ ERROR connecting to MCP server: {e}"
        )

        print(
            "\nMake sure QGIS and the MCP server are running."
        )

        sys.exit(1)


# ============================================================
# MCP: Execute tool
# ============================================================

async def execute_tool(
    tool_name,
    arguments,
):
    """
    Execute an MCP/QGIS tool.
    """

    try:

        async with MCPClient(
            MCP_SERVER_URL
        ) as mcp:

            result = await mcp.call_tool(
                tool_name,
                arguments,
            )

            return result

    except Exception as e:

        print(
            f"❌ ERROR executing tool {tool_name}: {e}"
        )

        return None


# ============================================================
# Convert MCP result to usable output
# ============================================================

def normalize_tool_result(
    tool_result,
    tool_name,
):

    if tool_result is None:

        return {
            "status": "error",
            "error": (
                f"Tool {tool_name} failed to execute."
            ),
        }

    structured = getattr(
        tool_result,
        "structured_content",
        None,
    )

    if structured:

        if isinstance(structured, dict):

            return structured.get(
                "result",
                structured,
            )

        return structured

    return str(tool_result)


# ============================================================
# Main interactive loop
# ============================================================

async def chat_loop():

    # --------------------------------------------------------
    # Load RAG first
    # --------------------------------------------------------

    try:

        chunks, embeddings = prepare_rag()

    except Exception as e:

        print(
            f"❌ ERROR loading RAG: {e}"
        )

        return

    # --------------------------------------------------------
    # Load MCP tools
    # --------------------------------------------------------

    print("🔍 Loading MCP tools...")

    tools = await load_mcp_tools()

    print(
        f"✅ Loaded {len(tools)} tools:"
    )

    for tool in tools:

        print(
            f"   - {tool['function']['name']}: "
            f"{tool['function']['description']}"
        )

    print()

    # --------------------------------------------------------
    # User interaction
    # --------------------------------------------------------

    while True:

        user_msg = input(
            "\n👤 You: "
        ).strip()

        if not user_msg:
            continue

        # Exit
        if user_msg.lower() in {
            "exit",
            "quit",
            "bye",
        }:

            print("👋 Goodbye!")
            return

        # Show MCP tools
        if user_msg.lower() in {
            "tools",
            "help",
        }:

            print("\nAvailable Tools:")

            for tool in tools:

                print(
                    f"   - {tool['function']['name']}: "
                    f"{tool['function']['description']}"
                )

            continue

        # ----------------------------------------------------
        # RAG retrieval
        # ----------------------------------------------------

        try:

            (
                retrieved_context,
                similarity_score,
            ) = retrieve_context(
                user_msg,
                chunks,
                embeddings,
            )

        except Exception as e:

            print(
                f"\n❌ RAG retrieval error: {e}"
            )

            retrieved_context = ""
            similarity_score = 0.0

        print("\n🔎 RAG retrieved:\n")

        print(
            retrieved_context
        )

        print(
            "\n📊 Similarity score: "
            f"{similarity_score:.4f}"
        )

        # ----------------------------------------------------
        # Give RAG context + user request to Gemma
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a GIS assistant controlling "
                    "QGIS through MCP tools.\n\n"

                    "RAG provides project-specific and GIS "
                    "knowledge. MCP provides live interaction "
                    "with QGIS. Do not confuse these roles.\n\n"

                    "Use the retrieved project knowledge when "
                    "it is relevant to the user's request.\n\n"

                    "For questions about the current QGIS "
                    "project, use MCP tools rather than assuming "
                    "stored RAG information is current.\n\n"

                    "You may call multiple MCP tools in sequence "
                    "when necessary.\n\n"

                    "If the user refers to a layer by name but "
                    "a tool requires a layer ID, first call "
                    "get_layers, identify the matching layer, "
                    "and then use its current layer ID.\n\n"

                    "Do not invent layer IDs, feature counts, "
                    "processing results, or successful tool "
                    "executions.\n\n"

                    "Do not claim an operation succeeded unless "
                    "the MCP tool result confirms success.\n\n"

                    "If an MCP tool returns an error, explain "
                    "the error instead of claiming success.\n\n"

                    "Retrieved project knowledge:\n"
                    f"{retrieved_context}"
                ),
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ]

        # ----------------------------------------------------
        # Multi-round Gemma -> MCP -> Gemma loop
        # ----------------------------------------------------

        max_tool_rounds = 8

        try:

            for _ in range(
                max_tool_rounds
            ):

                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    tools=tools,
                    think=False,
                    options={"num_ctx": 8192, "num_predict": 512},
                    stream=False,
                )

                print("OLLAMA STOP REASON:", response.done_reason)
                print("OLLAMA OUTPUT TOKENS:", response.eval_count)
                assistant_message = (
                    response["message"]
                )

                print("\nDEBUG — Gemma response:")
                print("Content:", repr(assistant_message.get("content")))
                print("Tool calls:", assistant_message.get("tool_calls"))
                print("Thinking:", repr(assistant_message.get("thinking")))

                messages.append(
                    assistant_message
                )

                tool_calls = (
                    assistant_message.get(
                        "tool_calls"
                    )
                )

                # ------------------------------------------------
                # No tool requested -> final response
                # ------------------------------------------------

                if not tool_calls:

                    print(
                        "\n🤖 AI response:"
                    )

                    content = (
                        assistant_message.get(
                            "content",
                            "",
                        )
                    )

                    if content:
                        print(content)
                    else:
                        print(
                            "(No text response returned.)"
                        )

                    break

                # ------------------------------------------------
                # Execute requested MCP tools
                # ------------------------------------------------

                for tool_call in tool_calls:

                    function_data = (
                        tool_call["function"]
                    )

                    tool_name = (
                        function_data["name"]
                    )

                    args = (
                        function_data["arguments"]
                    )

                    if isinstance(
                        args,
                        str,
                    ):
                        args = json.loads(args)

                    print(
                        f"\n🔧 Tool requested: {tool_name}"
                    )

                    print(
                        f"📝 Arguments: {args}"
                    )

                    tool_result = (
                        await execute_tool(
                            tool_name,
                            args,
                        )
                    )

                    tool_output = (
                        normalize_tool_result(
                            tool_result,
                            tool_name,
                        )
                    )

                    print(
                        f"✅ Tool result: {tool_output}\n"
                    )

                    if isinstance(
                        tool_output,
                        (dict, list),
                    ):

                        tool_content = (
                            json.dumps(
                                tool_output,
                                ensure_ascii=False,
                            )
                        )

                    else:

                        tool_content = str(
                            tool_output
                        )

                    messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": tool_content,
                    })

            else:

                print(
                    "\n⚠️ Maximum tool-call rounds reached."
                )

        except Exception as e:

            print(
                "\n❌ ERROR calling Gemma or "
                f"processing MCP tools: {e}"
            )

            print(
                "\nMake sure Ollama, MCP server, "
                "and QGIS are running."
            )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    asyncio.run(chat_loop())
