"""Dhakira quickstart — basic chatbot memory usage.

Prerequisites:
    pip install dhakira
    export OPENAI_API_KEY=your_key_here

This example shows how to:
1. Create a memory instance
2. Add conversation memories
3. Search for relevant memories
4. Update and delete memories
"""

from dhakira import Memory

# Zero-config setup (uses defaults: OpenAI gpt-4.1-nano, GATE embeddings, Qdrant in-memory)
memory = Memory()

# === Add memories from a conversation ===
print("Adding memories from conversation...")
memory_ids = memory.add(
    messages=[
        {"role": "user", "content": "اسمي حسام وأحب القهوة العربية"},
        {"role": "assistant", "content": "أهلا حسام! القهوة العربية خيار رائع"},
    ],
    user_id="user_123",
    session_id="session_456",
)
print(f"Created {len(memory_ids)} memories: {memory_ids}")

# === Add more memories ===
memory.add(
    messages=[
        {"role": "user", "content": "أعمل كمهندس برمجيات في شركة تقنية"},
        {"role": "assistant", "content": "مجال رائع! ما هي لغات البرمجة المفضلة لديك؟"},
        {"role": "user", "content": "أفضل Python و Rust"},
    ],
    user_id="user_123",
)

# === Search memories (zero LLM calls!) ===
print("\nSearching for drink preferences...")
results = memory.search(
    query="ما هي المشروبات المفضلة؟",
    user_id="user_123",
    limit=5,
)
for r in results:
    print(f"  [{r.score:.3f}] {r.text} (category: {r.category})")

# === Get all memories ===
print("\nAll memories for user_123:")
all_memories = memory.get_all(user_id="user_123")
for m in all_memories:
    print(f"  - {m.text} (category: {m.category})")

# === Update a memory ===
if memory_ids:
    print(f"\nUpdating memory {memory_ids[0]}...")
    memory.update(
        memory_id=memory_ids[0],
        text="يفضل القهوة التركية بدلا من العربية",
    )

# === Delete a memory ===
if len(memory_ids) > 1:
    print(f"Deleting memory {memory_ids[1]}...")
    memory.delete(memory_id=memory_ids[1])

print("\nDone!")
