"""Dhakira dialect support — working with multiple Arabic dialects.

Shows how Dhakira handles different Arabic dialects (MSA, Gulf,
Egyptian, Levantine, Maghrebi) with dialect-aware normalization.
"""

from dhakira import Memory

memory = Memory(config={
    "arabic": {
        "detect_dialect": True,
        "normalize_dialect": False,  # Preserve dialectal forms
        "remove_diacritics": True,
    },
})

# === MSA (Modern Standard Arabic) ===
print("Adding MSA text...")
memory.add(
    messages=[
        {"role": "user", "content": "أريد أن أتعلم البرمجة بلغة بايثون"},
        {"role": "assistant", "content": "ممتاز! بايثون لغة رائعة للمبتدئين"},
    ],
    user_id="multilingual_user",
    metadata={"dialect": "MSA"},
)

# === Egyptian Arabic ===
print("Adding Egyptian Arabic text...")
memory.add(
    messages=[
        {"role": "user", "content": "عايز اتعلم برمجة، إيه أحسن لغة أبدأ بيها؟"},
        {"role": "assistant", "content": "ابدأ ببايثون، سهلة ومطلوبة في السوق"},
    ],
    user_id="multilingual_user",
    metadata={"dialect": "Egyptian"},
)

# === Gulf Arabic ===
print("Adding Gulf Arabic text...")
memory.add(
    messages=[
        {"role": "user", "content": "أبي أتعلم برمجة، وش أحسن لغة أبدأ فيها؟"},
        {"role": "assistant", "content": "جرب بايثون، حلوة ومطلوبة وايد"},
    ],
    user_id="multilingual_user",
    metadata={"dialect": "Gulf"},
)

# === Levantine Arabic ===
print("Adding Levantine Arabic text...")
memory.add(
    messages=[
        {"role": "user", "content": "بدي إتعلم برمجة، شو أحسن لغة أبلش فيها؟"},
        {"role": "assistant", "content": "ابلش ببايثون، كتير سهلة ومطلوبة"},
    ],
    user_id="multilingual_user",
    metadata={"dialect": "Levantine"},
)

# === Search across all dialects ===
print("\nSearching across all dialects for 'programming'...")
results = memory.search(
    query="تعلم البرمجة",
    user_id="multilingual_user",
    limit=10,
)
for r in results:
    dialect = r.metadata.get("dialect", "unknown")
    print(f"  [{r.score:.3f}] ({dialect}) {r.text[:80]}")

print("\nDone!")
