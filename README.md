# 10-K SEC Filing Analyzer

**Author:** Ved Kaura  
**Architecture:** Deterministic extraction with LLM assistance (zero hallucination for financial data)

## Quick Start (Kaggle)
```python
# Set your API key
import os
os.environ['GOOGLE_API_KEY'] = 'your-api-key-here'

# Run analyzer
!python /kaggle/working/tenk_analyzer.py
```

When prompted, enter ticker (e.g., MSFT, AAPL, GOOGL)

## Features

- Deterministic financial extraction (HTML + regex)
- Multi-layer validation (4 independent checks)
- Top 3 risk factors with evidence
- MD&A themes and sentiment analysis
- Outputs structured JSON

Full documentation in repository.
