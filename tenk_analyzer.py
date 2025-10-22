import re
import json
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import google.generativeai as genai

# ============================================================
# CONFIGURATION
# ============================================================

# Configure API key from environment variable
api_key = os.environ.get('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)

# ============================================================
# DATA STRUCTURES
# ============================================================

class ExtractionConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class RiskFactor:
    rank: int
    title: str
    description: str
    evidence_quote: str
    prominence_score: str
    confidence: ExtractionConfidence = ExtractionConfidence.MEDIUM
    source_chunk_index: Optional[int] = None

@dataclass
class Theme:
    theme: str
    explanation: str
    supporting_quote: str
    sentiment: str = "neutral"
    confidence: ExtractionConfidence = ExtractionConfidence.MEDIUM

@dataclass
class FinancialMetric:
    metric: str
    year_1: Optional[float]
    year_2: Optional[float]
    year_3: Optional[float]
    source_table: str
    verified: bool = False
    extraction_method: str = "unknown"
    verification_details: Dict = field(default_factory=dict)
    confidence_score: float = 0.0

@dataclass
class SentimentAnalysis:
    overall_sentiment: str
    confidence: str
    supporting_evidence: List[Dict[str, str]]
    positive_indicators: List[str]
    negative_indicators: List[str]
    neutral_indicators: List[str]

@dataclass
class TenKAnalysis:
    company_name: str
    filing_year: str
    risk_factors: List[RiskFactor]
    themes: List[Theme]
    financials: List[FinancialMetric]
    fiscal_years: List[str]
    sentiment: SentimentAnalysis
    extraction_metadata: Dict = field(default_factory=dict)

# ============================================================
# PROMPTS (Only for non-numeric extraction)
# ============================================================

RISK_PROMPT = """You are extracting risk factors from an SEC 10-K filing.

Extract ALL risk factors discussed. For each:
- Title: 5-10 words
- Description: 1 sentence
- Evidence: exact quote showing severity (max 150 words)
- Prominence: High/Medium/Low based on length and severity

OUTPUT JSON:
{
  "risks": [
    {
      "rank": 1,
      "title": "Risk title",
      "description": "One sentence",
      "evidence_quote": "Exact quote from text",
      "prominence_score": "High"
    }
  ]
}

Extract everything. Do not skip risks."""

MDA_COMBINED_PROMPT = """Analyze this MD&A section.

Extract:
1. 5-7 main themes (recurring topics like performance drivers, strategic initiatives, challenges)
2. For each theme, note sentiment (positive/negative/neutral)
3. Overall sentiment with evidence

OUTPUT JSON:
{
  "themes": [
    {
      "theme": "Theme title (max 10 words)",
      "explanation": "1-2 sentences",
      "supporting_quote": "Exact quote (max 100 words)",
      "sentiment": "positive|negative|neutral"
    }
  ],
  "overall_sentiment": "positive|negative|neutral",
  "confidence": "high|medium|low",
  "positive_indicators": ["growth in X", "expansion of Y"],
  "negative_indicators": ["decline in X", "challenges with Y"],
  "neutral_indicators": ["stable performance"]
}"""

# ============================================================
# LLM PROCESSOR (Only for text analysis, NOT numbers)
# ============================================================

class LLMProcessor:
    @staticmethod
    def call_llm(system_prompt: str, user_prompt: str, max_retries: int = 2) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                response = model.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': 0,
                        'max_output_tokens': 8192,
                    }
                )
                
                text = response.text.strip()
                
                if text.startswith('```json'):
                    text = text.split('```json')[1].split('```')[0].strip()
                elif text.startswith('```'):
                    text = text.split('```')[1].split('```')[0].strip()
                
                if '{' in text:
                    text = text[text.index('{'):]
                if '}' in text:
                    text = text[:text.rindex('}')+1]
                
                text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
                
                return json.loads(text)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                print(f"  LLM error after {max_retries} attempts: {e}")
                return None
        
        return None

# ============================================================
# SMART CHUNKER
# ============================================================

class SmartChunker:
    @staticmethod
    def chunk_with_overlap(text: str, chunk_size: int = 12000, 
                          overlap: int = 2000) -> List[Dict]:
        chunks = []
        position = 0
        chunk_index = 0
        
        while position < len(text):
            chunk_end = position + chunk_size
            
            if chunk_end < len(text):
                sentence_end = SmartChunker._find_sentence_boundary(
                    text, chunk_end, search_window=500
                )
                if sentence_end:
                    chunk_end = sentence_end
            
            chunk_text = text[position:chunk_end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'index': chunk_index,
                    'start_pos': position,
                    'end_pos': chunk_end,
                    'size': len(chunk_text)
                })
                chunk_index += 1
            
            position = chunk_end - overlap
            
            if chunks and position <= chunks[-1]['start_pos']:
                position = chunk_end
        
        return chunks
    
    @staticmethod
    def _find_sentence_boundary(text: str, position: int, 
                               search_window: int = 500) -> Optional[int]:
        search_text = text[position:position + search_window]
        
        for pattern in ['. ', '.\n', '? ', '! ']:
            idx = search_text.find(pattern)
            if idx != -1:
                return position + idx + len(pattern)
        
        return None

# ============================================================
# HTML TABLE PARSER
# ============================================================

class HTMLTableParser:
    """Parse HTML tables - deterministic extraction only"""
    
    @staticmethod
    def find_financial_tables(html_content: str, debug: bool = False) -> List[Dict]:
        """Find all tables that look like financial statements"""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        if debug:
            print(f"  Found {len(tables)} HTML tables total")
        
        financial_tables = []
        
        for i, table in enumerate(tables):
            table_info = HTMLTableParser._analyze_table(table, i)
            
            if table_info and HTMLTableParser._is_financial_table(table_info):
                financial_tables.append(table_info)
                if debug and table_info['type'] == 'income_statement':
                    print(f"  ✓ INCOME STATEMENT at Table {i}")
        
        if debug:
            print(f"  Identified {len(financial_tables)} financial tables")
            print(f"  Income statements: {sum(1 for t in financial_tables if t['type'] == 'income_statement')}")
        
        return financial_tables
    
    @staticmethod
    def _analyze_table(table: Tag, index: int) -> Optional[Dict]:
        """Extract structured data from HTML table"""
        rows = table.find_all('tr')
        
        if len(rows) < 3:
            return None
        
        table_data = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [HTMLTableParser._clean_cell_text(cell) for cell in cells]
            if row_data:
                table_data.append(row_data)
        
        if not table_data:
            return None
        
        years = HTMLTableParser._extract_years(table_data[:5])
        table_type = HTMLTableParser._identify_table_type_aggressive(table, table_data)
        
        return {
            'index': index,
            'type': table_type,
            'data': table_data,
            'years': years,
            'rows': len(table_data),
            'cols': max(len(row) for row in table_data) if table_data else 0,
            'raw_table': table
        }
    
    @staticmethod
    def _clean_cell_text(cell: Tag) -> str:
        """Extract clean text from table cell"""
        text = cell.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def _identify_table_type_aggressive(table: Tag, table_data: List[List[str]]) -> str:
        """AGGRESSIVE income statement detection"""
        
        first_column = []
        for row in table_data[:30]:
            if row:
                first_column.append(row[0].lower())
        
        first_column_text = " ".join(first_column)
        
        revenue_keywords = ['revenue', 'net sales', 'total revenue', 'sales']
        income_keywords = ['net income', 'net earnings', 'income']
        operating_keywords = ['operating income', 'income from operations', 'operating profit']
        eps_keywords = ['earnings per share', 'eps', 'diluted', 'basic']
        cost_keywords = ['cost of revenue', 'cost of sales', 'cost of goods']
        
        has_revenue = any(kw in first_column_text for kw in revenue_keywords)
        has_income = any(kw in first_column_text for kw in income_keywords)
        has_operating = any(kw in first_column_text for kw in operating_keywords)
        has_eps = any(kw in first_column_text for kw in eps_keywords)
        has_cost = any(kw in first_column_text for kw in cost_keywords)
        
        income_score = sum([has_revenue, has_income, has_operating, has_eps, has_cost])
        
        if income_score >= 3:
            return 'income_statement'
        
        context = ""
        prev_elements = table.find_all_previous(['p', 'div', 'span', 'b', 'strong'], limit=30)
        for elem in prev_elements:
            text = elem.get_text(strip=True)
            if len(text) < 200:
                context += text.lower() + " "
        
        income_titles = [
            'statement of operations',
            'statements of operations',
            'statement of income',
            'statements of income',
            'statement of earnings',
            'consolidated operations',
            'consolidatedoperations',
        ]
        
        if any(title in context for title in income_titles):
            return 'income_statement'
        
        balance_keywords = ['total assets', 'total liabilities', 'stockholders equity', 
                          'current assets', 'shareholders equity']
        balance_score = sum(1 for kw in balance_keywords if kw in first_column_text)
        
        if balance_score >= 2:
            return 'balance_sheet'
        
        cash_keywords = ['cash flows from operating', 'cash flows from investing',
                        'cash flows from financing', 'net cash provided']
        cash_score = sum(1 for kw in cash_keywords if kw in first_column_text)
        
        if cash_score >= 2:
            return 'cash_flow'
        
        return 'unknown'
    
    @staticmethod
    def _extract_years(header_rows: List[List[str]]) -> List[str]:
        """Extract fiscal years from header rows"""
        years = []
        
        for row in header_rows:
            for cell in row:
                year_matches = re.findall(r'\b(20\d{2})\b', cell)
                years.extend(year_matches)
        
        seen = set()
        unique_years = []
        for year in years:
            if year not in seen:
                seen.add(year)
                unique_years.append(year)
        
        return unique_years[:5]
    
    @staticmethod
    def _is_financial_table(table_info: Dict) -> bool:
        """Determine if this is a financial statement table"""
        if not table_info['years']:
            return False
        
        if table_info['rows'] < 3 or table_info['cols'] < 2:
            return False
        
        has_numbers = False
        for row in table_info['data'][:20]:
            for cell in row:
                if re.search(r'\d+', cell):
                    has_numbers = True
                    break
            if has_numbers:
                break
        
        return has_numbers
    
    @staticmethod
    def extract_metric(table_info: Dict, metric_keywords: List[str], debug: bool = False) -> Optional[Dict]:
        """Extract a specific metric from parsed table"""
        data = table_info['data']
        years = table_info['years']
        
        if not years:
            return None
        
        for row in data:
            if not row or not row[0]:
                continue
            
            label = row[0].lower()
            label_cleaned = re.sub(r'[^\w\s]', '', label)
            
            match_found = False
            for keyword in metric_keywords:
                keyword_cleaned = keyword.lower()
                if keyword_cleaned in label or keyword_cleaned in label_cleaned:
                    match_found = True
                    break
            
            if match_found:
                values = []
                for cell in row[1:]:
                    num = HTMLTableParser._parse_number(cell)
                    values.append(num)
                
                valid_values = [v for v in values if v is not None]
                
                if not valid_values:
                    continue
                
                result_values = valid_values[:len(years)]
                
                while len(result_values) < 3:
                    result_values.append(None)
                
                if debug:
                    print(f"    Matched '{row[0]}' with keywords {metric_keywords}")
                    print(f"    Extracted values: {result_values[:3]}")
                
                return {
                    'label': row[0],
                    'values': result_values[:3],
                    'years': years[:3]
                }
        
        if debug:
            print(f"    No match found for keywords: {metric_keywords}")
        
        return None
    
    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        """Parse a number from financial table cell"""
        if not text or text.strip() in ['—', '-', '–', '', 'N/A', 'N.A.', '$—', '$ —', '*', '**']:
            return None
        
        if any(word in text.lower() for word in ['year', 'ended', 'note', 'see', 'december', 'september', 'june']):
            return None
        
        is_negative = '(' in text and ')' in text
        
        cleaned = re.sub(r'[^\d.-]', '', text)
        
        if not cleaned or cleaned in ['-', '.', '-.']:
            return None
        
        try:
            value = float(cleaned)
            if 0 < abs(value) < 10:
                return None
            return -abs(value) if is_negative else value
        except ValueError:
            return None

# ============================================================
# PATTERN EXTRACTOR - Deterministic regex extraction
# ============================================================

class PatternExtractor:
    """Extract financials using deterministic regex patterns"""
    
    @staticmethod
    def extract_from_text(text_content: str, debug: bool = False) -> Tuple[List[FinancialMetric], List[str]]:
        """Extract financial metrics using regex patterns"""
        
        if debug:
            print("  Searching for financial statement section...")
        
        # Find income statement section
        statement_section = PatternExtractor._find_income_statement_section(text_content)
        
        if not statement_section:
            if debug:
                print("  ✗ Could not find income statement section")
            return [], []
        
        if debug:
            print(f"  ✓ Found statement section ({len(statement_section)} chars)")
        
        # Extract years
        years = PatternExtractor._extract_years(statement_section)
        
        if not years or len(years) < 2:
            if debug:
                print("  ✗ Could not extract fiscal years")
            return [], []
        
        if debug:
            print(f"  ✓ Extracted years: {years}")
        
        # Extract each metric
        metrics = []
        
        metric_definitions = {
            'Total Revenue': [
                r'Total\s+(?:net\s+)?(?:sales|revenue)s?\s+([^\n]{0,200})',
                r'Net\s+(?:sales|revenue)s?\s+([^\n]{0,200})',
                r'(?:Total\s+)?Revenue\s+([^\n]{0,200})',
            ],
            'Cost of Revenue': [
                r'Cost\s+of\s+(?:sales|revenue|goods\s+sold)\s+([^\n]{0,200})',
                r'Cost\s+of\s+products\s+sold\s+([^\n]{0,200})',
            ],
            'Operating Income': [
                r'Operating\s+income\s+([^\n]{0,200})',
                r'Income\s+from\s+operations\s+([^\n]{0,200})',
            ],
            'Net Income': [
                r'Net\s+(?:income|earnings)\s+([^\n]{0,200})',
                r'Consolidated\s+net\s+income\s+([^\n]{0,200})',
            ],
            'Earnings Per Share (Basic)': [
                r'Basic\s+earnings\s+per\s+share\s+([^\n]{0,200})',
                r'Earnings\s+per\s+share\s+[^\n]*?basic[^\n]*?\s+([^\n]{0,100})',
            ],
            'Earnings Per Share (Diluted)': [
                r'Diluted\s+earnings\s+per\s+share\s+([^\n]{0,200})',
                r'Earnings\s+per\s+share\s+[^\n]*?diluted[^\n]*?\s+([^\n]{0,100})',
            ],
        }
        
        for metric_name, patterns in metric_definitions.items():
            values = PatternExtractor._extract_metric_values(
                statement_section, patterns, years, debug=debug
            )
            
            if values:
                metric = FinancialMetric(
                    metric=metric_name,
                    year_1=values[0] if len(values) > 0 else None,
                    year_2=values[1] if len(values) > 1 else None,
                    year_3=values[2] if len(values) > 2 else None,
                    source_table='pattern_extraction',
                    verified=True,  # Pattern extraction is deterministic
                    extraction_method='regex_patterns',
                    confidence_score=0.8
                )
                metrics.append(metric)
                
                if debug:
                    print(f"  ✓ {metric_name}: {values}")
            else:
                if debug:
                    print(f"  ✗ {metric_name}: not found")
        
        return metrics, years
    
    @staticmethod
    def _find_income_statement_section(text: str) -> Optional[str]:
        """Find the income statement section in text"""
        text_lower = text.lower()
        
        # Look for income statement headers
        keywords = [
            'consolidated statements of operations',
            'consolidated statement of operations',
            'consolidated statements of income',
            'consolidated statement of income',
            'statements of operations',
            'statement of operations',
        ]
        
        best_position = None
        for keyword in keywords:
            pos = text_lower.find(keyword)
            if pos != -1:
                best_position = pos
                break
        
        if best_position is None:
            return None
        
        # Extract reasonable section (50k chars should cover full statement)
        end_position = min(best_position + 50000, len(text))
        
        return text[best_position:end_position]
    
    @staticmethod
    def _extract_years(text: str) -> List[str]:
        """Extract fiscal years from text"""
        year_pattern = r'\b(20\d{2})\b'
        matches = re.findall(year_pattern, text[:5000])
        
        seen = set()
        years = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                years.append(match)
        
        return years[:3]
    
    @staticmethod
    def _extract_metric_values(text: str, patterns: List[str], years: List[str], 
                              debug: bool = False) -> Optional[List[float]]:
        """Extract values for a metric using multiple patterns"""
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                line = match.group(1) if match.lastindex else match.group(0)
                
                # Extract all numbers from the line
                numbers = PatternExtractor._parse_numbers_from_line(line)
                
                if numbers and len(numbers) >= 2:
                    # Take first N numbers matching year count
                    values = numbers[:len(years)]
                    
                    # Pad if needed
                    while len(values) < 3:
                        values.append(None)
                    
                    return values[:3]
        
        return None
    
    @staticmethod
    def _parse_numbers_from_line(line: str) -> List[float]:
        """Parse all numbers from a text line"""
        # Look for number patterns: 123,456 or (123,456) or $123,456
        number_pattern = r'[\$]?\s*\(?\s*([\d,]+\.?\d*)\s*\)?'
        
        matches = re.findall(number_pattern, line)
        
        numbers = []
        for match in matches:
            try:
                # Remove commas
                cleaned = match.replace(',', '')
                value = float(cleaned)
                
                # Filter out unreasonably small values (likely not financial data)
                if abs(value) >= 10:
                    # Check if this was in parentheses (negative)
                    if f'({match})' in line:
                        value = -abs(value)
                    
                    numbers.append(value)
            except ValueError:
                continue
        
        return numbers

# ============================================================
# NUMBER VALIDATOR - Multi-layer verification
# ============================================================

class NumberValidator:
    """Comprehensive validation of extracted financial numbers"""
    
    @staticmethod
    def validate_metric(metric: FinancialMetric, source_text: str, 
                       context: Optional[str] = None) -> Dict:
        """Multi-layer validation of a metric"""
        
        validation_result = {
            'verified': False,
            'confidence': 0.0,
            'checks': {}
        }
        
        # Layer 1: Value existence
        existence_check = NumberValidator._check_value_existence(metric, source_text)
        validation_result['checks']['existence'] = existence_check
        
        # Layer 2: Magnitude sanity
        magnitude_check = NumberValidator._check_magnitude_sanity(metric)
        validation_result['checks']['magnitude'] = magnitude_check
        
        # Layer 3: Year-over-year sanity
        yoy_check = NumberValidator._check_yoy_sanity(metric)
        validation_result['checks']['yoy'] = yoy_check
        
        # Layer 4: Context validation (if provided)
        if context:
            context_check = NumberValidator._check_context_proximity(
                metric, source_text, context
            )
            validation_result['checks']['context'] = context_check
        
        # Calculate overall confidence
        passed_checks = sum(1 for check in validation_result['checks'].values() 
                          if check.get('passed', False))
        total_checks = len(validation_result['checks'])
        
        validation_result['confidence'] = passed_checks / total_checks
        validation_result['verified'] = validation_result['confidence'] >= 0.6
        
        return validation_result
    
    @staticmethod
    def _check_value_existence(metric: FinancialMetric, source_text: str) -> Dict:
        """Check if values exist in source"""
        values = [metric.year_1, metric.year_2, metric.year_3]
        found_count = 0
        
        for value in values:
            if value is None:
                continue
            
            patterns = NumberValidator._generate_patterns(value)
            
            for pattern in patterns:
                if pattern in source_text:
                    found_count += 1
                    break
        
        non_null_count = sum(1 for v in values if v is not None)
        
        return {
            'passed': found_count >= max(1, non_null_count * 0.5),
            'found': found_count,
            'total': non_null_count
        }
    
    @staticmethod
    def _generate_patterns(number: float) -> List[str]:
        """Generate all possible text representations"""
        patterns = []
        abs_num = abs(number)
        
        if abs_num == int(abs_num):
            num_int = int(abs_num)
            patterns.append(str(num_int))
            patterns.append(f"{num_int:,}")
            patterns.append(f"${num_int:,}")
            patterns.append(f"$ {num_int:,}")
            
            if number < 0:
                patterns.append(f"({num_int:,})")
                patterns.append(f"(${num_int:,})")
        else:
            patterns.append(f"{abs_num:.2f}")
            patterns.append(f"{abs_num:,.2f}")
            patterns.append(f"${abs_num:,.2f}")
            
            if number < 0:
                patterns.append(f"({abs_num:,.2f})")
                patterns.append(f"(${abs_num:,.2f})")
        
        return patterns
    
    @staticmethod
    def _check_magnitude_sanity(metric: FinancialMetric) -> Dict:
        """Check if magnitudes are reasonable"""
        values = [metric.year_1, metric.year_2, metric.year_3]
        values = [v for v in values if v is not None]
        
        if not values:
            return {'passed': False, 'reason': 'no_values'}
        
        # EPS can be small
        if 'Per Share' in metric.metric:
            min_reasonable = 0.01
            max_reasonable = 10000
        else:
            # Revenue, income in millions
            min_reasonable = 100
            max_reasonable = 10000000
        
        all_reasonable = all(min_reasonable <= abs(v) <= max_reasonable for v in values)
        
        return {
            'passed': all_reasonable,
            'values': values,
            'range': (min_reasonable, max_reasonable)
        }
    
    @staticmethod
    def _check_yoy_sanity(metric: FinancialMetric) -> Dict:
        """Check year-over-year changes are reasonable"""
        values = [metric.year_1, metric.year_2, metric.year_3]
        values = [v for v in values if v is not None and v != 0]
        
        if len(values) < 2:
            return {'passed': True, 'reason': 'insufficient_data'}
        
        # Calculate YoY changes
        changes = []
        for i in range(len(values) - 1):
            if values[i] != 0:
                pct_change = abs((values[i+1] - values[i]) / values[i])
                changes.append(pct_change)
        
        # No single year should have >500% change (10x would be suspicious)
        max_reasonable_change = 5.0  # 500%
        
        all_reasonable = all(change <= max_reasonable_change for change in changes)
        
        return {
            'passed': all_reasonable,
            'changes': changes,
            'max_change': max(changes) if changes else 0
        }
    
    @staticmethod
    def _check_context_proximity(metric: FinancialMetric, source_text: str, 
                                 context_keyword: str) -> Dict:
        """Check if numbers appear near expected context"""
        # Find all occurrences of context keyword
        keyword_lower = context_keyword.lower()
        text_lower = source_text.lower()
        
        keyword_positions = []
        start = 0
        while True:
            pos = text_lower.find(keyword_lower, start)
            if pos == -1:
                break
            keyword_positions.append(pos)
            start = pos + 1
        
        if not keyword_positions:
            return {'passed': False, 'reason': 'keyword_not_found'}
        
        # Check if any value appears within 500 chars of keyword
        values = [metric.year_1, metric.year_2, metric.year_3]
        proximity_window = 500
        
        found_near_keyword = False
        for value in values:
            if value is None:
                continue
            
            patterns = NumberValidator._generate_patterns(value)
            
            for pattern in patterns[:3]:  # Check top 3 patterns
                for kw_pos in keyword_positions:
                    search_start = max(0, kw_pos - proximity_window)
                    search_end = min(len(source_text), kw_pos + proximity_window)
                    search_region = source_text[search_start:search_end]
                    
                    if pattern in search_region:
                        found_near_keyword = True
                        break
                
                if found_near_keyword:
                    break
            
            if found_near_keyword:
                break
        
        return {
            'passed': found_near_keyword,
            'keyword_occurrences': len(keyword_positions)
        }

# ============================================================
# FINANCIAL EXTRACTOR - DETERMINISTIC ONLY
# ============================================================

class FinancialExtractor:
    """Extract and verify financial data - NO LLM HALLUCINATION"""
    
    def __init__(self, html_content: str, text_content: str):
        self.html = html_content
        self.text = text_content
        self.tables = []
    
    def extract_all_metrics(self) -> Tuple[List[FinancialMetric], List[str], str]:
        """Extract financials with deterministic methods only"""
        
        # Strategy 1: HTML table parsing
        print("  Strategy 1: HTML table parsing")
        try:
            html_result = self._extract_from_html_tables()
            if html_result and html_result[0]:
                html_metrics, html_years, html_company = html_result
                
                # Validate extracted metrics
                validated_metrics = self._validate_metrics(html_metrics)
                
                if len(validated_metrics) >= 3:
                    print(f"  ✓ HTML extraction successful: {len(validated_metrics)} validated metrics")
                    return validated_metrics, html_years, html_company
        except Exception as e:
            print(f"  ✗ HTML parsing failed: {e}")
        
        # Strategy 2: Regex pattern extraction
        print("  Strategy 2: Regex pattern extraction")
        try:
            pattern_result = self._extract_with_patterns()
            if pattern_result and pattern_result[0]:
                pattern_metrics, pattern_years = pattern_result
                
                validated_metrics = self._validate_metrics(pattern_metrics)
                
                if len(validated_metrics) >= 3:
                    print(f"  ✓ Pattern extraction successful: {len(validated_metrics)} validated metrics")
                    company = self._extract_company_name()
                    return validated_metrics, pattern_years, company
        except Exception as e:
            print(f"  ✗ Pattern extraction failed: {e}")
        
        # Both strategies failed - return empty
        print("  ✗ All extraction strategies failed")
        return [], [], "Unknown"
    
    def _extract_from_html_tables(self) -> Tuple[List[FinancialMetric], List[str], str]:
        """Extract from HTML tables"""
        self.tables = HTMLTableParser.find_financial_tables(self.html, debug=True)
        
        if not self.tables:
            raise ValueError("No HTML tables found")
        
        income_stmts = [t for t in self.tables if t['type'] == 'income_statement']
        
        if not income_stmts:
            raise ValueError("No income statement found")
        
        print(f"  ✓ Found {len(income_stmts)} income statement(s)")
        
        best_result = None
        best_score = 0
        
        for i, income_stmt in enumerate(income_stmts[:5]):
            print(f"\n  Trying income statement #{i+1}...")
            
            try:
                metrics = self._extract_metrics_from_table(income_stmt)
                
                # Score based on completeness
                score = sum(1 for m in metrics 
                          if m.year_1 is not None or m.year_2 is not None or m.year_3 is not None)
                
                print(f"    Result: {score}/6 metrics with data")
                
                if score > best_score:
                    best_result = (metrics, income_stmt['years'][:3], self._extract_company_name())
                    best_score = score
                
                if score >= 4:
                    break
                    
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        
        if best_result:
            return best_result
        
        raise ValueError("No usable income statement data")
    
    def _extract_metrics_from_table(self, income_stmt: Dict) -> List[FinancialMetric]:
        """Extract metrics from table"""
        
        metric_definitions = {
            'Total Revenue': [
                'total net sales', 'net sales', 'total revenue', 'revenue',
                'net revenues', 'total net revenue', 'sales'
            ],
            'Cost of Revenue': [
                'cost of sales', 'cost of revenue', 'cost of goods sold',
                'cost of products sold', 'cost of services'
            ],
            'Operating Income': [
                'operating income', 'income from operations', 'operating profit',
                'income operations'
            ],
            'Net Income': [
                'net income', 'net earnings', 'net loss',
                'consolidated net income', 'income attributable'
            ],
            'Earnings Per Share (Basic)': [
                'basic earnings per share', 'earnings per share basic', 'basic eps',
                'earnings per share', 'eps basic'
            ],
            'Earnings Per Share (Diluted)': [
                'diluted earnings per share', 'earnings per share diluted', 'diluted eps',
                'eps diluted'
            ]
        }
        
        metrics = []
        for metric_name, keywords in metric_definitions.items():
            result = HTMLTableParser.extract_metric(income_stmt, keywords, debug=False)
            
            if result:
                values = result['values']
                
                metric = FinancialMetric(
                    metric=metric_name,
                    year_1=values[0] if len(values) > 0 else None,
                    year_2=values[1] if len(values) > 1 else None,
                    year_3=values[2] if len(values) > 2 else None,
                    source_table='income_statement',
                    verified=False,  # Will verify later
                    extraction_method='html_table_parsing',
                    confidence_score=0.9
                )
                
                metrics.append(metric)
        
        return metrics
    
    def _extract_with_patterns(self) -> Tuple[List[FinancialMetric], List[str]]:
        """Extract using regex patterns"""
        metrics, years = PatternExtractor.extract_from_text(self.text, debug=True)
        
        if not metrics:
            raise ValueError("Pattern extraction returned no metrics")
        
        return metrics, years
    
    def _validate_metrics(self, metrics: List[FinancialMetric]) -> List[FinancialMetric]:
        """Validate all metrics"""
        validated = []
        
        for metric in metrics:
            validation = NumberValidator.validate_metric(metric, self.text)
            
            metric.verified = validation['verified']
            metric.confidence_score = validation['confidence']
            metric.verification_details = validation['checks']
            
            # Keep metrics with confidence >= 0.5
            if metric.confidence_score >= 0.5:
                validated.append(metric)
                status = "✓" if metric.verified else "~"
                print(f"  {status} {metric.metric}: confidence={metric.confidence_score:.2f}")
            else:
                print(f"  ✗ {metric.metric}: confidence={metric.confidence_score:.2f} (rejected)")
        
        return validated
    
    def _extract_company_name(self) -> str:
        """Extract company name"""
        search_text = self.text[:5000]
        
        patterns = [
            r'([A-Z][A-Za-z\s&.,Inc]+?)\s+(?:FORM 10-K|10-K)',
            r'COMPANY NAME:\s*([A-Z][A-Za-z\s&.,Inc]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, search_text)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                if 5 < len(name) < 100:
                    return name
        
        return "Unknown Company"

# ============================================================
# SECTION EXTRACTOR
# ============================================================

class SectionExtractor:
    """Extract sections using multiple strategies"""
    
    def __init__(self, raw_content: str, html_content: str):
        self.raw_text = raw_content
        self.html_content = html_content
        self.sections = {}
    
    def extract_all(self) -> Dict[str, str]:
        """Try multiple extraction strategies"""
        
        print("  Strategy 1: HTML structure parsing")
        try:
            if self._extract_from_html():
                print("  ✓ HTML parsing successful")
                return self.sections
        except Exception as e:
            print(f"  ✗ HTML parsing failed: {e}")
        
        print("  Strategy 2: Text pattern matching")
        try:
            if self._extract_from_text_improved():
                print("  ✓ Text parsing successful")
                return self.sections
        except Exception as e:
            print(f"  ✗ Text parsing failed: {e}")
        
        print("  Strategy 3: Keyword-based extraction")
        try:
            self._extract_by_keywords()
            print("  ✓ Keyword extraction successful")
            return self.sections
        except Exception as e:
            print(f"  ✗ Keyword extraction failed: {e}")
        
        raise ValueError("All extraction strategies failed")
    
    def _extract_from_html(self) -> bool:
        """Extract using HTML structure"""
        soup = BeautifulSoup(self.html_content, 'html.parser')
        
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div'])
        
        item_positions = []
        for i, header in enumerate(headers):
            text = header.get_text(strip=True)
            
            match = re.match(r'Item\s+(\d+[A-Z]?)[\.\s:]', text, re.IGNORECASE)
            if match:
                item_num = match.group(1).upper()
                item_positions.append({
                    'item': item_num,
                    'element': header,
                    'index': i,
                    'text': text
                })
        
        if len(item_positions) < 3:
            return False
        
        self.sections['risk_factors'] = self._extract_html_section(
            item_positions, '1A', ['1B', '2']
        )
        
        self.sections['mda'] = self._extract_html_section(
            item_positions, '7', ['7A', '8']
        )
        
        self.sections['financials'] = self._extract_html_section(
            item_positions, '8', ['9', '9A', '9B']
        )
        
        required = ['risk_factors', 'mda', 'financials']
        return all(self.sections.get(s) and len(self.sections[s]) > 1000 for s in required)
    
    def _extract_html_section(self, item_positions: List[Dict], 
                             start_item: str, end_items: List[str]) -> Optional[str]:
        """Extract section between items"""
        
        start_elem = None
        for item_info in reversed(item_positions):
            if item_info['item'] == start_item:
                start_elem = item_info['element']
                break
        
        if not start_elem:
            return None
        
        end_elem = None
        start_idx = next((i for i, item in enumerate(item_positions) 
                         if item['element'] == start_elem), None)
        
        if start_idx is not None:
            for i in range(start_idx + 1, len(item_positions)):
                if item_positions[i]['item'] in end_items:
                    end_elem = item_positions[i]['element']
                    break
        
        if not end_elem:
            return None
        
        content = []
        current = start_elem.next_sibling
        
        while current and current != end_elem:
            if isinstance(current, Tag):
                text = current.get_text(separator=' ', strip=True)
                if text:
                    content.append(text)
            
            current = current.next_sibling
            
            if len(content) > 10000:
                break
        
        result = ' '.join(content)
        return result if len(result) > 1000 else None
    
    def _extract_from_text_improved(self) -> bool:
        """Extract from text using last occurrence"""
        items = self._find_all_items()
        
        if len(items) < 10:
            return False
        
        item_last_positions = {}
        for item_num, pos in items:
            item_last_positions[item_num] = pos
        
        self.sections['risk_factors'] = self._extract_text_section(
            item_last_positions, '1A', ['1B', '2']
        )
        
        self.sections['mda'] = self._extract_text_section(
            item_last_positions, '7', ['7A', '8']
        )
        
        self.sections['financials'] = self._extract_text_section(
            item_last_positions, '8', ['9', '9A', '9B']
        )
        
        required = ['risk_factors', 'mda', 'financials']
        return all(self.sections.get(s) and len(self.sections[s]) > 1000 
                  for s in required)
    
    def _find_all_items(self) -> List[Tuple[str, int]]:
        """Find all Item X occurrences"""
        pattern = r'Item\s+(\d+[A-Z]?)[\.\s:]'
        matches = []
        
        for match in re.finditer(pattern, self.raw_text, re.IGNORECASE):
            item_num = match.group(1).upper()
            position = match.start()
            matches.append((item_num, position))
        
        return matches
    
    def _extract_text_section(self, item_positions: Dict[str, int],
                              start_item: str, end_items: List[str]) -> Optional[str]:
        """Extract section from text"""
        
        start_pos = item_positions.get(start_item)
        if start_pos is None:
            return None
        
        end_pos = None
        for end_item in end_items:
            pos = item_positions.get(end_item)
            if pos and pos > start_pos:
                end_pos = pos
                break
        
        if not end_pos:
            return None
        
        section = self.raw_text[start_pos:end_pos].strip()
        return section if len(section) > 1000 else None
    
    def _extract_by_keywords(self):
        """Fallback: keyword search"""
        
        risk_keywords = [
            'risk factors',
            'risks related to',
            'Item 1A'
        ]
        self.sections['risk_factors'] = self._search_section(
            risk_keywords, min_length=5000, max_length=200000
        )
        
        mda_keywords = [
            "management's discussion and analysis",
            "Item 7."
        ]
        self.sections['mda'] = self._search_section(
            mda_keywords, min_length=5000, max_length=100000
        )
        
        fin_keywords = [
            'financial statements and supplementary data',
            'Item 8.'
        ]
        self.sections['financials'] = self._search_section(
            fin_keywords, min_length=5000, max_length=300000
        )
    
    def _search_section(self, keywords: List[str], min_length: int, 
                       max_length: int) -> Optional[str]:
        """Search by keywords"""
        text_lower = self.raw_text.lower()
        
        for keyword in keywords:
            pos = text_lower.find(keyword.lower())
            if pos != -1:
                start = pos
                end = min(pos + max_length, len(self.raw_text))
                
                section = self.raw_text[start:end]
                
                if len(section) >= min_length:
                    return section
        
        return None

# ============================================================
# RISK EXTRACTOR
# ============================================================

class RiskExtractor:
    def __init__(self, risk_section: str):
        self.text = risk_section
    
    def extract_all_risks(self) -> List[RiskFactor]:
        if not self.text or len(self.text) < 1000:
            print("  ✗ Risk section too short")
            return []
        
        chunks = SmartChunker.chunk_with_overlap(self.text, chunk_size=12000, overlap=2000)
        print(f"  Processing {len(chunks)} chunks")
        
        all_risks = []
        for i, chunk in enumerate(chunks):
            chunk_risks = self._extract_from_chunk(chunk['text'], i)
            all_risks.extend(chunk_risks)
        
        print(f"  Total extracted: {len(all_risks)} risks")
        
        unique_risks = self._deduplicate_risks(all_risks)
        print(f"  After deduplication: {len(unique_risks)} risks")
        
        ranked = self._rank_by_prominence(unique_risks)
        return ranked[:3]
    
    def _extract_from_chunk(self, chunk_text: str, chunk_index: int) -> List[RiskFactor]:
        result = LLMProcessor.call_llm(
            "You are extracting risk factors.",
            RISK_PROMPT + "\n\nTEXT:\n" + chunk_text
        )
        
        if not result or 'risks' not in result:
            return []
        
        risks = []
        for r in result['risks']:
            risk = RiskFactor(
                rank=r.get('rank', 0),
                title=r.get('title', ''),
                description=r.get('description', ''),
                evidence_quote=r.get('evidence_quote', ''),
                prominence_score=r.get('prominence_score', 'Medium'),
                confidence=ExtractionConfidence.MEDIUM,
                source_chunk_index=chunk_index
            )
            risks.append(risk)
        
        return risks
    
    def _deduplicate_risks(self, risks: List[RiskFactor]) -> List[RiskFactor]:
        if not risks:
            return []
        
        unique = []
        seen_titles = set()
        sorted_risks = sorted(risks, key=lambda r: len(r.evidence_quote), reverse=True)
        
        for risk in sorted_risks:
            title_normalized = risk.title.lower().strip()
            title_normalized = re.sub(r'[^\w\s]', '', title_normalized)
            
            is_duplicate = False
            for seen in seen_titles:
                similarity = self._string_similarity(title_normalized, seen)
                if similarity > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(risk)
                seen_titles.add(title_normalized)
        
        return unique
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _rank_by_prominence(self, risks: List[RiskFactor]) -> List[RiskFactor]:
        def prominence_score(risk: RiskFactor) -> float:
            score = 0.0
            score += min(len(risk.evidence_quote) / 100, 10.0)
            
            if risk.prominence_score.lower() == 'high':
                score += 5.0
            elif risk.prominence_score.lower() == 'medium':
                score += 3.0
            else:
                score += 1.0
            
            severity_keywords = ['significant', 'material', 'substantial', 'critical', 
                               'major', 'severe', 'adversely', 'harm']
            text_combined = (risk.description + ' ' + risk.evidence_quote).lower()
            for keyword in severity_keywords:
                if keyword in text_combined:
                    score += 0.5
            
            return score
        
        ranked = sorted(risks, key=prominence_score, reverse=True)
        
        for i, risk in enumerate(ranked):
            risk.rank = i + 1
        
        return ranked

# ============================================================
# MDA ANALYZER
# ============================================================

class MDAAnalyzer:
    def __init__(self, mda_section: str):
        self.text = mda_section
    
    def analyze(self) -> Tuple[List[Theme], SentimentAnalysis]:
        if not self.text or len(self.text) < 1000:
            print("  ✗ MD&A section too short")
            return [], SentimentAnalysis("neutral", "low", [], [], [], [])
        
        if len(self.text) > 25000:
            sample_text = self._strategic_sample()
        else:
            sample_text = self.text
        
        result = LLMProcessor.call_llm(
            "You are analyzing MD&A content.",
            MDA_COMBINED_PROMPT + "\n\nTEXT:\n" + sample_text
        )
        
        if not result:
            return [], SentimentAnalysis("neutral", "low", [], [], [], [])
        
        themes = [Theme(
            theme=t['theme'],
            explanation=t['explanation'],
            supporting_quote=t['supporting_quote'],
            sentiment=t['sentiment']
        ) for t in result.get('themes', [])]
        
        sentiment = SentimentAnalysis(
            overall_sentiment=result.get('overall_sentiment', 'neutral'),
            confidence=result.get('confidence', 'medium'),
            supporting_evidence=[],
            positive_indicators=result.get('positive_indicators', []),
            negative_indicators=result.get('negative_indicators', []),
            neutral_indicators=result.get('neutral_indicators', [])
        )
        
        return themes, sentiment
    
    def _strategic_sample(self) -> str:
        chunk_size = 8000
        
        beginning = self.text[:chunk_size]
        middle_start = len(self.text) // 2 - chunk_size // 2
        middle = self.text[middle_start:middle_start + chunk_size]
        end = self.text[-chunk_size:]
        
        return (beginning + "\n\n[... middle section ...]\n\n" + 
                middle + "\n\n[... ending section ...]\n\n" + end)

# ============================================================
# SEC DOWNLOADER
# ============================================================

class SECDownloader:
    @staticmethod
    def download_filing(cik: str, accession: str) -> Tuple[str, str]:
        """Download both raw text and HTML"""
        headers = {'User-Agent': 'Research research@university.edu'}
        
        accession_nodash = accession.replace('-', '')
        archive_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"
        
        print(f"Fetching directory: {archive_url}")
        response = requests.get(archive_url, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Cannot access directory: HTTP {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        txt_files = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.txt'):
                parent_text = link.parent.get_text() if link.parent else ""
                size_match = re.search(r'(\d+)\s*$', parent_text)
                size = int(size_match.group(1)) if size_match else 0
                txt_files.append((href, size))
        
        if not txt_files:
            raise ValueError("No .txt files in directory")
        
        txt_files.sort(key=lambda x: x[1], reverse=True)
        txt_file = txt_files[0][0]
        
        if txt_file.startswith('/'):
            txt_url = f"https://www.sec.gov{txt_file}"
        else:
            txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{txt_file}"
        
        print(f"Downloading: {txt_url}")
        response = requests.get(txt_url, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Download failed: HTTP {response.status_code}")
        
        html_content = response.text
        
        if len(html_content) < 50000:
            raise ValueError(f"File too small: {len(html_content)} chars")
        
        print(f"✓ Downloaded {len(html_content):,} chars")
        
        text_content = re.sub(r'<[^>]+>', ' ', html_content)
        text_content = re.sub(r'\s+', ' ', text_content)
        
        return text_content, html_content

    @staticmethod
    def get_latest_10k(ticker: str) -> tuple:
        headers = {'User-Agent': 'Research research@university.edu'}
        
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=exclude&count=10"
        
        print(f"Searching: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Search failed: HTTP {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        cik = None
        cik_span = soup.find('span', {'class': 'companyName'})
        if cik_span:
            cik_match = re.search(r'(\d+)', cik_span.text)
            if cik_match:
                cik = cik_match.group(1).lstrip('0') or '0'
        
        if not cik:
            for text in soup.stripped_strings:
                if 'CIK' in text:
                    match = re.search(r'CIK.*?(\d+)', text)
                    if match:
                        cik = match.group(1).lstrip('0') or '0'
                        break
        
        print(f"CIK: {cik}")
        
        table = soup.find('table', {'class': 'tableFile2'})
        
        if not table:
            raise ValueError("Cannot find filings table")
        
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 2:
                filing_type = cols[0].text.strip()
                
                if filing_type == '10-K':
                    for link in row.find_all('a'):
                        href = link.get('href', '')
                        if 'accession' in href:
                            match = re.search(r'accession_number=([0-9-]+)', href)
                            if match:
                                accession = match.group(1)
                                print(f"✓ Found: CIK={cik}, Accession={accession}")
                                return (cik, accession)
        
        raise ValueError(f"No 10-K found for {ticker}")

# ============================================================
# MAIN ANALYZER
# ============================================================

class TenKAnalyzer:
    def __init__(self, text_content: str, html_content: str):
        self.text = text_content
        self.html = html_content
        self.sections = {}
        self.metadata = {}
    
    def analyze(self) -> TenKAnalysis:
        print("="*60)
        print("STAGE 1: SECTION EXTRACTION")
        print("="*60)
        self._extract_sections()
        
        print("\n" + "="*60)
        print("STAGE 2: FINANCIAL EXTRACTION (DETERMINISTIC)")
        print("="*60)
        financials, years, company = self._extract_financials()
        
        print("\n" + "="*60)
        print("STAGE 3: RISK ANALYSIS")
        print("="*60)
        risks = self._extract_risks()
        
        print("\n" + "="*60)
        print("STAGE 4: MD&A ANALYSIS")
        print("="*60)
        themes, sentiment = self._analyze_mda()
        
        return TenKAnalysis(
            company_name=company,
            filing_year=years[0] if years else "Unknown",
            risk_factors=risks,
            themes=themes,
            financials=financials,
            fiscal_years=years,
            sentiment=sentiment,
            extraction_metadata=self.metadata
        )
    
    def _extract_sections(self):
        extractor = SectionExtractor(self.text, self.html)
        self.sections = extractor.extract_all()
        
        for section_name in ['risk_factors', 'mda', 'financials']:
            section_content = self.sections.get(section_name)
            if section_content and len(section_content) > 1000:
                print(f"  ✓ {section_name.replace('_', ' ').title()}: {len(section_content):,} chars")
            else:
                print(f"  ✗ {section_name.replace('_', ' ').title()}: FAILED")
                self.sections[section_name] = ""
    
    def _extract_financials(self) -> Tuple[List[FinancialMetric], List[str], str]:
        extractor = FinancialExtractor(self.html, self.text)
        
        try:
            metrics, years, company = extractor.extract_all_metrics()
            
            verified_count = sum(1 for m in metrics if m.verified)
            avg_confidence = sum(m.confidence_score for m in metrics) / len(metrics) if metrics else 0
            
            print(f"  ✓ Extracted {len(metrics)} metrics")
            print(f"  ✓ Verified: {verified_count}/{len(metrics)}")
            print(f"  ✓ Average confidence: {avg_confidence:.2f}")
            
            self.metadata['financial_extraction'] = metrics[0].extraction_method if metrics else 'failed'
            self.metadata['verified_metrics'] = verified_count
            self.metadata['total_metrics'] = len(metrics)
            self.metadata['avg_confidence'] = avg_confidence
            
            return metrics, years, company
            
        except Exception as e:
            print(f"  ✗ All financial extraction failed: {e}")
            return [], [], "Unknown"
    
    def _extract_risks(self) -> List[RiskFactor]:
        extractor = RiskExtractor(self.sections.get('risk_factors', ''))
        
        try:
            risks = extractor.extract_all_risks()
            self.metadata['risk_extraction_method'] = 'multi_chunk'
            return risks
        except Exception as e:
            print(f"  ✗ Risk extraction failed: {e}")
            return []
    
    def _analyze_mda(self) -> Tuple[List[Theme], SentimentAnalysis]:
        analyzer = MDAAnalyzer(self.sections.get('mda', ''))
        
        try:
            themes, sentiment = analyzer.analyze()
            self.metadata['mda_analysis_method'] = 'combined'
            return themes, sentiment
        except Exception as e:
            print(f"  ✗ MD&A analysis failed: {e}")
            return [], SentimentAnalysis("neutral", "low", [], [], [], [])

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("10-K ANALYZER - DETERMINISTIC EXTRACTION")
    print("Author: Ved Kaura")
    print("Trust minimization: No LLM hallucination for numbers")
    print("="*60)
    
    # Prompt user for ticker
    ticker_input = input("\nCOMPANY TICKER: ").strip().upper()
    
    if not ticker_input:
        print("Error: No ticker provided")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {ticker_input}")
    print("="*60)
    
    try:
        # Download
        cik, accession = SECDownloader.get_latest_10k(ticker_input)
        text_content, html_content = SECDownloader.download_filing(cik, accession)
        
        # Analyze
        analyzer = TenKAnalyzer(text_content, html_content)
        analysis = analyzer.analyze()
        
        # Display results
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {ticker_input}")
        print("="*60)
        
        print(f"\nCompany: {analysis.company_name}")
        print(f"Filing Year: {analysis.filing_year}")
        print(f"Fiscal Years: {', '.join(analysis.fiscal_years)}")
        
        print(f"\n--- TOP 3 RISKS ---")
        for risk in analysis.risk_factors[:3]:
            print(f"\n{risk.rank}. {risk.title}")
            print(f"   {risk.description}")
        
        print(f"\n--- FINANCIALS ---")
        if analysis.financials:
            print(f"{'Metric':<30} | {'Year 1':>12} | {'Year 2':>12} | {'Year 3':>12} | Verified | Confidence")
            print("-" * 110)
            for metric in analysis.financials:
                status = "✓" if metric.verified else "✗"
                y1 = f"{metric.year_1:,.2f}" if metric.year_1 else "N/A"
                y2 = f"{metric.year_2:,.2f}" if metric.year_2 else "N/A"
                y3 = f"{metric.year_3:,.2f}" if metric.year_3 else "N/A"
                conf = f"{metric.confidence_score:.2f}"
                print(f"{metric.metric:<30} | {y1:>12} | {y2:>12} | {y3:>12} | {status:^8} | {conf:>10}")
        else:
            print("No financial metrics extracted")
        
        print(f"\n--- MD&A THEMES ---")
        for theme in analysis.themes[:5]:
            print(f"\n• {theme.theme} [{theme.sentiment}]")
            print(f"  {theme.explanation}")
        
        print(f"\n--- SENTIMENT ---")
        print(f"Overall: {analysis.sentiment.overall_sentiment} (confidence: {analysis.sentiment.confidence})")
        
        # Save output
        filename = f'tenk_analysis_{ticker_input.lower()}_deterministic.json'
        output = {
            "company": analysis.company_name,
            "filing_year": analysis.filing_year,
            "fiscal_years": analysis.fiscal_years,
            "key_risk_factors": [
                {
                    "rank": r.rank,
                    "title": r.title,
                    "description": r.description,
                    "evidence_quote": r.evidence_quote
                }
                for r in analysis.risk_factors[:3]
            ],
            "mda_highlights": [
                {
                    "theme": t.theme,
                    "explanation": t.explanation,
                    "sentiment": t.sentiment,
                    "supporting_quote": t.supporting_quote
                }
                for t in analysis.themes
            ],
            "financial_highlights": [
                {
                    "metric": m.metric,
                    "year_1": m.year_1,
                    "year_2": m.year_2,
                    "year_3": m.year_3,
                    "verified": m.verified,
                    "extraction_method": m.extraction_method,
                    "confidence_score": m.confidence_score
                }
                for m in analysis.financials
            ],
            "sentiment_analysis": {
                "overall_sentiment": analysis.sentiment.overall_sentiment,
                "confidence": analysis.sentiment.confidence,
                "positive_indicators": analysis.sentiment.positive_indicators,
                "negative_indicators": analysis.sentiment.negative_indicators
            },
            "extraction_metadata": analysis.extraction_metadata
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Saved to {filename}")
        
    except Exception as e:
        print(f"\n✗ Failed to analyze {ticker_input}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print("="*60)
