"""Dhakira with LangChain — integration example.

Shows how to use Dhakira as a memory backend alongside LangChain.
This is a conceptual example showing the integration pattern.
"""

from dhakira import Memory

# Initialize Dhakira memory
dhakira = Memory(config={
    "llm": {"provider": "openai", "model": "gpt-4.1-nano"},
    "arabic": {"detect_dialect": True},
})


def langchain_conversation_with_memory(user_id: str, user_message: str):
    """Example of using Dhakira memory in a LangChain-like flow."""

    # 1. Retrieve relevant memories
    memories = dhakira.search(
        query=user_message,
        user_id=user_id,
        limit=5,
    )

    # 2. Build context from memories
    memory_context = ""
    if memories:
        memory_lines = [f"- {m.text}" for m in memories]
        memory_context = (
            "Relevant memories about this user:\n" + "\n".join(memory_lines) + "\n\n"
        )

    # 3. Build prompt for LangChain (conceptual)
    prompt = f"""{memory_context}User message: {user_message}

Respond helpfully in Arabic, using the memory context if relevant."""

    print(f"Prompt with memory context:\n{prompt}\n")

    # 4. After getting LLM response, store the conversation
    assistant_response = "هذا مثال على الرد"  # Placeholder

    dhakira.add(
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ],
        user_id=user_id,
    )

    return assistant_response


# === Example usage ===
print("=== Conversation 1 ===")
langchain_conversation_with_memory(
    user_id="user_lang",
    user_message="اسمي سارة وأعمل طبيبة",
)

print("=== Conversation 2 ===")
langchain_conversation_with_memory(
    user_id="user_lang",
    user_message="ما هي أفضل الكتب الطبية؟",
)

print("Done!")
