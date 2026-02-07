"""Dhakira agent memory — autonomous agent memory usage.

Shows how an AI agent can store and retrieve procedural knowledge,
task results, and learned patterns.
"""

from dhakira import Memory

memory = Memory()

# === Agent stores procedural knowledge ===
print("Agent storing procedural knowledge...")
memory.add(
    messages=[
        {
            "role": "assistant",
            "content": "وجدت أن أفضل طريقة لتحليل البيانات العربية هي استخدام مكتبة pandas "
                       "مع ضبط الترميز على UTF-8 واستخدام PyArabic للتطبيع قبل المعالجة",
        }
    ],
    agent_id="data_analyst_agent",
    metadata={"task": "data_analysis", "procedure": True},
)

# === Agent stores task results ===
memory.add(
    messages=[
        {
            "role": "assistant",
            "content": "تقرير تحليل المبيعات: المبيعات ارتفعت بنسبة 15% في الربع الأخير. "
                       "أكثر المنتجات مبيعا كان المنتج أ بنسبة 35% من إجمالي المبيعات",
        }
    ],
    agent_id="data_analyst_agent",
    metadata={"task": "sales_report", "quarter": "Q4_2024"},
)

# === Agent retrieves relevant knowledge for new task ===
print("\nAgent searching for data analysis procedures...")
results = memory.search(
    query="كيف أحلل البيانات العربية؟",
    agent_id="data_analyst_agent",
    limit=5,
)
for r in results:
    print(f"  [{r.score:.3f}] {r.text}")

print("\nAgent searching for sales data...")
results = memory.search(
    query="ما هي نتائج تحليل المبيعات؟",
    agent_id="data_analyst_agent",
    limit=5,
)
for r in results:
    print(f"  [{r.score:.3f}] {r.text}")

print("\nDone!")
