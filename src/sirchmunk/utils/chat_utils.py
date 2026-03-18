"""Chat intent detection and response utilities for conversational queries."""

import re

CHAT_QUERY_RE = re.compile(
    r"^("
    # Greetings (ZH / EN / pinyin / JA / KO)
    r"你好|您好|嗨|哈喽|喂|早上好|下午好|晚上好|早安|午安|晚安"
    r"|hello|hi|hey|howdy|greetings|yo"
    r"|nihao|ni\s*hao"
    r"|good\s*(morning|afternoon|evening|night)"
    r"|こんにちは|こんばんは|おはよう"
    r"|안녕하세요|안녕"
    # Identity / capability
    r"|who\s+are\s+you|what\s+are\s+you|你是谁|你是什么"
    r"|介绍.*你自己|tell\s+me\s+about\s+yourself"
    r"|what\s+can\s+you\s+do|你能做什么|你会什么"
    # Small talk
    r"|how\s+are\s+you|你好吗|你怎么样|what'?s\s+up"
    # Thanks
    r"|thank\s*you|thanks|谢谢|感谢|多谢"
    # Goodbye
    r"|bye|goodbye|再见|拜拜|see\s+you"
    # Ping / test
    r"|test(ing)?|ping"
    r")[\s!！？?。.，,~～…]*$",
    re.IGNORECASE,
)

CHAT_RESPONSE_SYSTEM = (
    "You are Sirchmunk, an intelligent document search and analysis assistant. "
    "The user sent a conversational message (greeting, identity question, etc.) "
    "rather than a search query. Respond naturally and helpfully in 1-3 sentences. "
    "Reply in the same language as the user's message."
)
