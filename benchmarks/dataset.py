"""Arabic conversation dataset for benchmarking.

12 conversations across 4 dialects (MSA, Egyptian, Gulf, Levantine),
each with 3 conversations covering different topics.
14 search queries with ground truth IDs for quality evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Conversation:
    id: str
    dialect: str
    topic: str
    messages: list[dict[str, str]]


@dataclass
class Query:
    id: str
    text: str
    ground_truth_ids: list[str]
    description: str = ""


# ---------------------------------------------------------------------------
# MSA conversations (3): personal facts, work info, daily routine
# ---------------------------------------------------------------------------

MSA_PERSONAL = Conversation(
    id="msa_personal",
    dialect="MSA",
    topic="personal facts",
    messages=[
        {"role": "user", "content": "اسمي محمد عبد الرحمن وأنا من مدينة القاهرة. عمري ثلاثون عاماً وأعمل كمهندس برمجيات."},
        {"role": "assistant", "content": "أهلاً محمد! يسعدني التعرف عليك. هل تعمل في مجال البرمجيات منذ فترة طويلة؟"},
        {"role": "user", "content": "نعم، أعمل في هذا المجال منذ ثماني سنوات. تخرجت من جامعة القاهرة في قسم هندسة الحاسوب عام ٢٠١٨."},
        {"role": "assistant", "content": "خبرة ممتازة! جامعة القاهرة من أعرق الجامعات. ما هي لغات البرمجة التي تتخصص فيها؟"},
        {"role": "user", "content": "أتقن بايثون وجافاسكريبت بشكل أساسي، وأستخدم TypeScript في المشاريع الكبيرة. كما أتعلم لغة Rust حالياً."},
    ],
)

MSA_WORK = Conversation(
    id="msa_work",
    dialect="MSA",
    topic="work info",
    messages=[
        {"role": "user", "content": "أعمل حالياً في شركة تقنية ناشئة متخصصة في الذكاء الاصطناعي. مقرها في دبي لكنني أعمل عن بُعد."},
        {"role": "assistant", "content": "العمل عن بُعد في شركة ذكاء اصطناعي يبدو مثيراً! ما هو دورك تحديداً في الشركة؟"},
        {"role": "user", "content": "أنا رئيس فريق تطوير الواجهة الخلفية. أقود فريقاً من خمسة مطورين ونعمل على منصة معالجة اللغة العربية."},
        {"role": "assistant", "content": "رائع! معالجة اللغة العربية مجال مهم جداً. ما هي أبرز التحديات التي تواجهونها؟"},
        {"role": "user", "content": "أكبر تحدٍّ هو التعامل مع اللهجات المختلفة. اللغة العربية غنية جداً والتشكيل يغير المعنى كثيراً. نستخدم نماذج BERT المعدلة للعربية."},
    ],
)

MSA_ROUTINE = Conversation(
    id="msa_routine",
    dialect="MSA",
    topic="daily routine",
    messages=[
        {"role": "user", "content": "أستيقظ يومياً في الساعة السادسة صباحاً. أبدأ يومي بالرياضة لمدة نصف ساعة ثم أتناول الفطور."},
        {"role": "assistant", "content": "روتين صحي ومنظم! ما هي الرياضة التي تمارسها عادةً؟"},
        {"role": "user", "content": "أمارس الجري ثلاث مرات في الأسبوع وتمارين القوة مرتين. أحب الجري في حديقة الأزهر القريبة من منزلي."},
        {"role": "assistant", "content": "حديقة الأزهر مكان جميل للجري! متى تبدأ العمل عادةً؟"},
        {"role": "user", "content": "أبدأ العمل في الثامنة والنصف وأنتهي في الخامسة. بعد العمل أقرأ كتاباً لمدة ساعة قبل النوم. حالياً أقرأ كتاب 'البؤساء' مترجماً."},
    ],
)

# ---------------------------------------------------------------------------
# Egyptian conversations (3): food preferences, family, hobbies
# ---------------------------------------------------------------------------

EGYPTIAN_FOOD = Conversation(
    id="egyptian_food",
    dialect="Egyptian",
    topic="food preferences",
    messages=[
        {"role": "user", "content": "أنا بحب الأكل المصري جداً، خصوصاً الكشري والفول. الكشري بتاع أبو طارق في التحرير مفيش زيه."},
        {"role": "assistant", "content": "الكشري المصري أكلة عظيمة فعلاً! بتحب تاكله بالشطة الكتير ولا عادي؟"},
        {"role": "user", "content": "لازم شطة كتير وكمان دقة. وبحب كمان الملوخية بالأرانب، أمي بتعملها أحسن حاجة. كل يوم جمعة عندنا ملوخية."},
        {"role": "assistant", "content": "ملوخية بالأرانب كل جمعة! دي عادة جميلة. وبالنسبة للحلويات، بتحب إيه؟"},
        {"role": "user", "content": "الكنافة النابلسية أحسن حاجة، وكمان أم علي. بس بحاول أقلل السكر عشان الدايت. باخد قهوة سادة من غير سكر بقالي سنة."},
    ],
)

EGYPTIAN_FAMILY = Conversation(
    id="egyptian_family",
    dialect="Egyptian",
    topic="family",
    messages=[
        {"role": "user", "content": "أنا متجوز ومعايا ولدين، أحمد عنده ست سنين وسارة عندها أربع سنين. عايشين في مدينة نصر."},
        {"role": "assistant", "content": "ماشاء الله! مدينة نصر منطقة حلوة. مراتك بتشتغل ولا ست بيت؟"},
        {"role": "user", "content": "مراتي دكتورة أسنان، عندها عيادة في مصر الجديدة. اسمها نورا وبتشتغل خمس أيام في الأسبوع."},
        {"role": "assistant", "content": "ماشاء الله عليها! والأولاد بيروحوا مدرسة فين؟"},
        {"role": "user", "content": "أحمد في مدرسة المستقبل الدولية في التجمع. سارة لسه في الحضانة بس هتدخل نفس المدرسة السنة الجاية إن شاء الله."},
    ],
)

EGYPTIAN_HOBBIES = Conversation(
    id="egyptian_hobbies",
    dialect="Egyptian",
    topic="hobbies",
    messages=[
        {"role": "user", "content": "أنا بحب التصوير الفوتوغرافي جداً. عندي كاميرا Canon R5 وبطلع أصور في الأماكن التاريخية."},
        {"role": "assistant", "content": "Canon R5 كاميرا ممتازة! بتصور إيه بالظبط؟ مناظر ولا بورتريه؟"},
        {"role": "user", "content": "بحب تصوير الشوارع والعمارة الإسلامية. صورت شارع المعز ومسجد ابن طولون وقلعة صلاح الدين. عندي حساب على انستجرام فيه كل صوري."},
        {"role": "assistant", "content": "شارع المعز من أجمل الأماكن للتصوير! بتصور كل أسبوع ولا إزاي؟"},
        {"role": "user", "content": "كل يوم سبت الصبح بطلع أصور لحد الضهر. وكمان بحب ألعب شطرنج أونلاين بالليل، مستوايا حوالي ١٥٠٠ على موقع chess.com."},
    ],
)

# ---------------------------------------------------------------------------
# Gulf conversations (3): travel, shopping, health
# ---------------------------------------------------------------------------

GULF_TRAVEL = Conversation(
    id="gulf_travel",
    dialect="Gulf",
    topic="travel",
    messages=[
        {"role": "user", "content": "أنا أحب السفر وايد. السنة اللي فاتت رحت تركيا واليابان وإسبانيا. أفضل رحلة كانت اليابان."},
        {"role": "assistant", "content": "اليابان بلد جميل! وش أكثر شي عجبك فيها؟"},
        {"role": "user", "content": "الثقافة والأكل والنظافة. قعدت في طوكيو أسبوع وكيوتو أسبوع. المعابد في كيوتو شي ما ينوصف. ودي أرجع مرة ثانية."},
        {"role": "assistant", "content": "كيوتو فعلاً مكان ساحر! وش خطتك للسفر هالسنة؟"},
        {"role": "user", "content": "إن شاء الله بروح سويسرا في الصيف وكوريا الجنوبية في الخريف. حجزت الفندق في زيوريخ من الحين. دايم أحجز من بدري عشان الأسعار."},
    ],
)

GULF_SHOPPING = Conversation(
    id="gulf_shopping",
    dialect="Gulf",
    topic="shopping",
    messages=[
        {"role": "user", "content": "أحب أتسوق من دبي مول، خصوصاً محلات الإلكترونيات. عندي آخر آيفون وآخر ساعة أبل."},
        {"role": "assistant", "content": "دبي مول فيه كل شي! غير الإلكترونيات، وش تحب تتسوق؟"},
        {"role": "user", "content": "أحب الثوب الإماراتي من محل البشت. وأشتري عطور من أجمل، عطر العود المخلط هو المفضل عندي. كل شهر تقريباً أشتري عطر ياديد."},
        {"role": "assistant", "content": "العود المخلط عطر فخم! ميزانيتك الشهرية للتسوق كم تقريباً؟"},
        {"role": "user", "content": "حوالي ثلاث آلاف درهم بالشهر للتسوق الشخصي. بس أوقات أزيد لو فيه عروض. أحب كمان أتسوق أونلاين من أمازون ونون."},
    ],
)

GULF_HEALTH = Conversation(
    id="gulf_health",
    dialect="Gulf",
    topic="health",
    messages=[
        {"role": "user", "content": "بديت حمية غذائية من شهرين. نزلت ثمان كيلو للحين. هدفي إني أنزل عشرين كيلو بالمجموع."},
        {"role": "assistant", "content": "ماشاء الله عليك! ثمان كيلو في شهرين إنجاز حلو. وش نوع الحمية اللي تتبعها؟"},
        {"role": "user", "content": "حمية كيتو مع صيام متقطع ١٦/٨. أكل بس من الساعة ١٢ للساعة ٨ بالليل. وأمشي ساعة كل يوم في الممشى."},
        {"role": "assistant", "content": "كيتو مع صيام متقطع كومبو قوي! عندك أي مشاكل صحية؟"},
        {"role": "user", "content": "كان عندي كولسترول عالي بس الحين نزل مع الحمية. وعندي حساسية من المكسرات لازم أنتبه. وأشرب ماي واجد، تقريباً ثلاث لتر باليوم."},
    ],
)

# ---------------------------------------------------------------------------
# Levantine conversations (3): education, technology, social
# ---------------------------------------------------------------------------

LEVANTINE_EDUCATION = Conversation(
    id="levantine_education",
    dialect="Levantine",
    topic="education",
    messages=[
        {"role": "user", "content": "أنا عم بدرس ماستر في علوم البيانات بجامعة الأردنية. هاد السمستر التاني والأخير إن شاء الله."},
        {"role": "assistant", "content": "علوم البيانات تخصص مطلوب كتير! شو موضوع رسالتك؟"},
        {"role": "user", "content": "رسالتي عن تحليل المشاعر في التغريدات العربية باستخدام التعلم العميق. جمعت ٥٠ ألف تغريدة وعم بدرب نموذج BERT مخصص."},
        {"role": "assistant", "content": "موضوع مهم كتير! مين المشرف تبعك؟"},
        {"role": "user", "content": "الدكتور خالد الحسيني، هو متخصص بمعالجة اللغات الطبيعية. الرسالة لازم تكون جاهزة بشهر حزيران. عم بشتغل عليها كل يوم تقريباً."},
    ],
)

LEVANTINE_TECH = Conversation(
    id="levantine_tech",
    dialect="Levantine",
    topic="technology",
    messages=[
        {"role": "user", "content": "أنا مبرمج ويب، بشتغل بشكل أساسي بـ React و Node.js. مشتغل فري لانسر من خمس سنين."},
        {"role": "assistant", "content": "فري لانسر خمس سنين! من وين بتجيب شغلك عادةً؟"},
        {"role": "user", "content": "أغلب شغلي من Upwork وكمان عندي زبائن ثابتين. بشتغل على مشروعين أو تلاتة بنفس الوقت عادةً. حالياً عم بعمل موقع تجارة إلكترونية لشركة سعودية."},
        {"role": "assistant", "content": "حلو! شو بتستخدم للباك إند للمشروع هاد؟"},
        {"role": "user", "content": "Next.js مع PostgreSQL و Prisma ORM. وعم بستخدم Stripe للدفع. المشروع بده ثلاث أشهر كمان تقريباً. ساعة الشغل عندي بخمسين دولار."},
    ],
)

LEVANTINE_SOCIAL = Conversation(
    id="levantine_social",
    dialect="Levantine",
    topic="social",
    messages=[
        {"role": "user", "content": "أنا بحب ألتقي بأصحابي كل يوم جمعة بمقهى في وسط البلد بعمان. اسم المقهى 'بيت الشاي' وهو مكاننا من عشر سنين."},
        {"role": "assistant", "content": "عشر سنين! هاد إشي حلو كتير. كم واحد بتكونوا عادةً؟"},
        {"role": "user", "content": "أربعة لخمسة أشخاص. أقرب صاحبي اسمه يزن، منعرف من أيام المدرسة. هو محامي ودايماً بنحكي بالسياسة والكتب."},
        {"role": "assistant", "content": "صداقة من أيام المدرسة إشي نادر! شو بتعملوا غير المقهى؟"},
        {"role": "user", "content": "مرة بالشهر بنروح نلعب كرة قدم بملعب في شفا بدران. وبالصيف بنرتب رحلات للعقبة أو البحر الميت. السنة اللي فاتت رحنا وادي رم وكان أحلى رحلة."},
    ],
)

# ---------------------------------------------------------------------------
# All conversations
# ---------------------------------------------------------------------------

ALL_CONVERSATIONS: list[Conversation] = [
    MSA_PERSONAL,
    MSA_WORK,
    MSA_ROUTINE,
    EGYPTIAN_FOOD,
    EGYPTIAN_FAMILY,
    EGYPTIAN_HOBBIES,
    GULF_TRAVEL,
    GULF_SHOPPING,
    GULF_HEALTH,
    LEVANTINE_EDUCATION,
    LEVANTINE_TECH,
    LEVANTINE_SOCIAL,
]

CONVERSATIONS_BY_ID: dict[str, Conversation] = {c.id: c for c in ALL_CONVERSATIONS}

# ---------------------------------------------------------------------------
# Search queries with ground truth
# ---------------------------------------------------------------------------

ALL_QUERIES: list[Query] = [
    Query(
        id="q1",
        text="ما اسمه وأين يسكن؟",
        ground_truth_ids=["msa_personal", "egyptian_family"],
        description="Personal name and location",
    ),
    Query(
        id="q2",
        text="ما هي لغات البرمجة التي يعرفها؟",
        ground_truth_ids=["msa_personal", "levantine_tech"],
        description="Programming languages",
    ),
    Query(
        id="q3",
        text="ماذا يعمل؟ ما هي وظيفته؟",
        ground_truth_ids=["msa_personal", "msa_work", "levantine_tech"],
        description="Job and occupation",
    ),
    Query(
        id="q4",
        text="ما هو الأكل المفضل عنده؟",
        ground_truth_ids=["egyptian_food"],
        description="Food preferences",
    ),
    Query(
        id="q5",
        text="كم عدد أولاده وما أسماؤهم؟",
        ground_truth_ids=["egyptian_family"],
        description="Children info",
    ),
    Query(
        id="q6",
        text="ما هي هواياته؟",
        ground_truth_ids=["egyptian_hobbies", "msa_routine", "levantine_social"],
        description="Hobbies",
    ),
    Query(
        id="q7",
        text="أين سافر وأين يريد يسافر؟",
        ground_truth_ids=["gulf_travel"],
        description="Travel history and plans",
    ),
    Query(
        id="q8",
        text="هل عنده مشاكل صحية؟",
        ground_truth_ids=["gulf_health"],
        description="Health issues",
    ),
    Query(
        id="q9",
        text="ماذا يدرس في الجامعة؟",
        ground_truth_ids=["levantine_education", "msa_personal"],
        description="University studies",
    ),
    Query(
        id="q10",
        text="كم يتقاضى على ساعة العمل؟",
        ground_truth_ids=["levantine_tech"],
        description="Hourly rate",
    ),
    Query(
        id="q11",
        text="ما هو روتينه اليومي والرياضة؟",
        ground_truth_ids=["msa_routine", "gulf_health"],
        description="Daily routine and exercise",
    ),
    Query(
        id="q12",
        text="ما هي العطور المفضلة عنده؟",
        ground_truth_ids=["gulf_shopping"],
        description="Perfume preferences",
    ),
    Query(
        id="q13",
        text="من هو أقرب أصدقائه؟",
        ground_truth_ids=["levantine_social"],
        description="Close friends",
    ),
    Query(
        id="q14",
        text="ما موضوع رسالة الماجستير؟",
        ground_truth_ids=["levantine_education"],
        description="Master's thesis topic",
    ),
]
