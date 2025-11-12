---
description:
---

This codebase implements a chatBot with a few Mistral and OpenAI LLMs
Its goals is to help the user realize the cost of the requests in terms of price and ecological impact.

It uses data directly imported from the ecologits library that we cannot use since we use LangChain

The displaying is handled with streamlit.
The app handles those models :
"gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14","mistral-medium-latest","magistral-medium-latest" and "codestral-latest"
