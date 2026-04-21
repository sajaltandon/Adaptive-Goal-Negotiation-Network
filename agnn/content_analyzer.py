"""
Tier-2 Week 2: Content Quality Analysis

Provides NLP-based content analysis for coherence, readability, and domain compliance.
Supports the artifact validator with detailed content assessment.
Enhanced with strategy domain support and critical fixes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import re
import math
from collections import Counter


@dataclass
class StructureScore:
    """Document structure quality assessment"""
    hierarchy_quality: float    # 0.0-1.0: Header hierarchy quality
    section_balance: float      # 0.0-1.0: Balance between sections
    formatting_quality: float  # 0.0-1.0: Use of lists, bold, etc.
    logical_flow: float        # 0.0-1.0: Logical progression
    overall: float             # 0.0-1.0: Weighted overall score


@dataclass
class CoherenceScore:
    """Content coherence assessment"""
    semantic_consistency: float  # 0.0-1.0: Consistent terminology
    cross_references: float     # 0.0-1.0: References between sections
    transition_quality: float  # 0.0-1.0: Quality of transitions
    topic_coherence: float     # 0.0-1.0: Staying on topic
    overall: float             # 0.0-1.0: Weighted overall score


@dataclass
class ReadabilityScore:
    """Content readability assessment"""
    sentence_complexity: float  # 0.0-1.0: Sentence length and structure
    vocabulary_level: float    # 0.0-1.0: Vocabulary complexity
    clarity: float            # 0.0-1.0: Overall clarity
    conciseness: float        # 0.0-1.0: Conciseness vs verbosity
    overall: float            # 0.0-1.0: Weighted overall score


class ContentAnalyzer:
    """
    Content quality analysis using NLP techniques and domain-specific rules.
    Provides detailed assessment of structure, coherence, and readability.
    Enhanced with strategy domain support and improved validation.
    """
    
    def __init__(self):
        """Initialize with analysis parameters"""
        self.transition_words = {
            "additive": ["additionally", "furthermore", "moreover", "also", "besides"],
            "adversative": ["however", "nevertheless", "nonetheless", "conversely", "although"],
            "causal": ["therefore", "consequently", "thus", "hence", "because"],
            "temporal": ["first", "next", "then", "finally", "subsequently", "meanwhile"],
            "exemplification": ["for example", "for instance", "specifically", "namely"],
            "summary": ["in conclusion", "to summarize", "overall", "in summary"]
        }
        
        self.domain_vocabularies = {
            "policy": {
                "formal": ["shall", "must", "required", "mandatory", "prohibited", "compliance"],
                "procedural": ["procedure", "process", "guideline", "standard", "protocol"],
                "authority": ["management", "supervisor", "approval", "authorization", "responsibility"]
            },
            "technical": {
                "implementation": ["configure", "install", "deploy", "integrate", "implement"],
                "specification": ["requirement", "specification", "parameter", "criteria"],
                "process": ["workflow", "pipeline", "methodology", "framework", "architecture"]
            },
            "business": {
                "strategy": ["objective", "goal", "target", "milestone", "deliverable"],
                "analysis": ["assessment", "evaluation", "analysis", "review", "audit"],
                "planning": ["timeline", "schedule", "resource", "budget", "allocation"]
            },
            "strategy": {
                "market": ["market", "segment", "target", "customer", "audience", "demographic"],
                "competitive": ["competitive", "competitor", "positioning", "advantage", "differentiation"],
                "pricing": ["pricing", "price", "cost", "revenue", "monetization", "subscription"],
                "marketing": ["marketing", "channel", "campaign", "promotion", "advertising", "outreach"],
                "go_to_market": ["launch", "rollout", "implementation", "execution", "strategy", "plan"]
            }
        }
    
    def analyze_structure(self, content: str) -> StructureScore:
        """
        Analyze document structure quality.
        
        Args:
            content: The content to analyze
            
        Returns:
            StructureScore with detailed structure assessment
        """
        hierarchy_quality = self._assess_hierarchy_quality(content)
        section_balance = self._assess_section_balance(content)
        formatting_quality = self._assess_formatting_quality(content)
        logical_flow = self._assess_logical_flow(content)
        
        # Weighted overall score
        overall = (
            0.30 * hierarchy_quality +
            0.25 * section_balance +
            0.25 * formatting_quality +
            0.20 * logical_flow
        )
        
        return StructureScore(
            hierarchy_quality=hierarchy_quality,
            section_balance=section_balance,
            formatting_quality=formatting_quality,
            logical_flow=logical_flow,
            overall=overall
        )
    
    def analyze_coherence(self, content: str) -> CoherenceScore:
        """
        Analyze content coherence and consistency.
        
        Args:
            content: The content to analyze
            
        Returns:
            CoherenceScore with detailed coherence assessment
        """
        semantic_consistency = self._assess_semantic_consistency(content)
        cross_references = self._assess_cross_references(content)
        transition_quality = self._assess_transition_quality(content)
        topic_coherence = self._assess_topic_coherence(content)
        
        # Weighted overall score
        overall = (
            0.30 * semantic_consistency +
            0.25 * cross_references +
            0.25 * transition_quality +
            0.20 * topic_coherence
        )
        
        return CoherenceScore(
            semantic_consistency=semantic_consistency,
            cross_references=cross_references,
            transition_quality=transition_quality,
            topic_coherence=topic_coherence,
            overall=overall
        )
    
    def analyze_readability(self, content: str) -> ReadabilityScore:
        """
        Analyze content readability and clarity.
        
        Args:
            content: The content to analyze
            
        Returns:
            ReadabilityScore with detailed readability assessment
        """
        sentence_complexity = self._assess_sentence_complexity(content)
        vocabulary_level = self._assess_vocabulary_level(content)
        clarity = self._assess_clarity(content)
        conciseness = self._assess_conciseness(content)
        
        # Weighted overall score
        overall = (
            0.25 * sentence_complexity +
            0.25 * vocabulary_level +
            0.30 * clarity +
            0.20 * conciseness
        )
        
        return ReadabilityScore(
            sentence_complexity=sentence_complexity,
            vocabulary_level=vocabulary_level,
            clarity=clarity,
            conciseness=conciseness,
            overall=overall
        )
    
    def check_domain_requirements(self, content: str, domain: str) -> Dict[str, float]:
        """
        Check domain-specific requirements and vocabulary usage.
        
        Args:
            content: The content to analyze
            domain: Domain type (policy, technical, business, strategy)
            
        Returns:
            Dictionary with domain compliance scores
        """
        if domain not in self.domain_vocabularies:
            return {"overall": 0.5, "vocabulary_usage": 0.5, "style_compliance": 0.5}
        
        vocab = self.domain_vocabularies[domain]
        content_lower = content.lower()
        
        # Vocabulary usage score
        vocab_scores = {}
        for category, words in vocab.items():
            found_words = sum(1 for word in words if word in content_lower)
            vocab_scores[category] = found_words / len(words)
        
        vocabulary_usage = sum(vocab_scores.values()) / len(vocab_scores)
        
        # Style compliance score (domain-specific patterns)
        style_compliance = self._assess_domain_style(content, domain)
        
        overall = (vocabulary_usage + style_compliance) / 2
        
        return {
            "overall": overall,
            "vocabulary_usage": vocabulary_usage,
            "style_compliance": style_compliance,
            "category_scores": vocab_scores
        }
    
    def _assess_hierarchy_quality(self, content: str) -> float:
        """Assess quality of header hierarchy"""
        # Count different header levels
        h1_count = len(re.findall(r'^#\s+', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s+', content, re.MULTILINE))
        h4_count = len(re.findall(r'^####\s+', content, re.MULTILINE))
        
        score = 0.0
        
        # Prefer documents with clear hierarchy
        if h2_count >= 1:  # Main title
            score += 0.3
        if h3_count >= 3:  # Multiple sections
            score += 0.4
        if h4_count > 0:   # Subsections
            score += 0.2
        
        # Penalize too many levels (confusing)
        if h4_count > h3_count:
            score -= 0.1
        
        # Check for logical progression
        if h1_count <= 1 and h2_count >= 1 and h3_count >= h2_count:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_section_balance(self, content: str) -> float:
        """Assess balance between sections"""
        sections = re.split(r'^###\s+', content, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]
        
        if len(sections) < 2:
            return 0.3  # Neutral for single section
        
        # Calculate section lengths
        section_lengths = [len(section.split()) for section in sections]
        
        if not section_lengths:
            return 0.0
        
        # Calculate coefficient of variation (lower is better)
        mean_length = sum(section_lengths) / len(section_lengths)
        if mean_length == 0:
            return 0.0
        
        variance = sum((length - mean_length) ** 2 for length in section_lengths) / len(section_lengths)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_length
        
        # Convert to score (lower CV = higher score)
        balance_score = max(0.0, 1.0 - cv)
        
        return min(1.0, balance_score)
    
    def _assess_formatting_quality(self, content: str) -> float:
        """Assess quality of formatting elements"""
        score = 0.0
        
        # Check for various formatting elements
        has_bold = '**' in content or '__' in content
        has_bullets = bool(re.search(r'^\s*[-*•]\s+', content, re.MULTILINE))
        has_numbers = bool(re.search(r'^\s*\d+\.\s+', content, re.MULTILINE))
        has_code = '`' in content or '```' in content
        has_tables = '|' in content and content.count('|') >= 6
        has_links = '[' in content and '](' in content
        
        # Score each element
        if has_bold: score += 0.15
        if has_bullets: score += 0.20
        if has_numbers: score += 0.20
        if has_code: score += 0.10
        if has_tables: score += 0.15
        if has_links: score += 0.10
        
        # Bonus for good combination
        format_count = sum([has_bold, has_bullets, has_numbers, has_code, has_tables, has_links])
        if format_count >= 3:
            score += 0.10
        
        return min(1.0, score)
    
    def _assess_logical_flow(self, content: str) -> float:
        """Assess logical flow and progression"""
        sections = re.split(r'^###\s+', content, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]
        
        if len(sections) < 2:
            return 0.5
        
        score = 0.0
        
        # Check for transition words
        transition_count = 0
        for category, words in self.transition_words.items():
            for word in words:
                transition_count += content.lower().count(word)
        
        # Normalize by content length
        word_count = len(content.split())
        transition_density = transition_count / max(1, word_count / 100)  # Per 100 words
        transition_score = min(1.0, transition_density / 2) * 0.4
        score += transition_score
        
        # Check for numbered sequences
        has_sequence = bool(re.search(r'(first|1\.|step 1)', content, re.IGNORECASE))
        if has_sequence:
            score += 0.3
        
        # Check for logical section ordering
        section_headers = re.findall(r'^###\s+(.+)$', content, re.MULTILINE)
        if self._has_logical_section_order(section_headers):
            score += 0.3
        
        return min(1.0, score)
    
    def _assess_semantic_consistency(self, content: str) -> float:
        """Assess consistency of terminology and concepts"""
        # Extract key terms (capitalized words, technical terms)
        key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        key_terms = [term.lower() for term in key_terms if len(term) > 3]
        
        if not key_terms:
            return 0.5
        
        # Count term frequencies
        term_counts = Counter(key_terms)
        
        # Terms should appear multiple times for consistency
        repeated_terms = sum(1 for count in term_counts.values() if count > 1)
        consistency_score = repeated_terms / len(set(key_terms))
        
        return min(1.0, consistency_score)
    
    def _assess_cross_references(self, content: str) -> float:
        """Assess cross-references between sections"""
        sections = re.split(r'^###\s+', content, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]
        
        if len(sections) < 2:
            return 0.5
        
        cross_refs = 0
        total_possible = len(sections) * (len(sections) - 1)
        
        for i, section in enumerate(sections):
            section_words = set(word.lower() for word in section.split() if len(word) > 3)
            
            for j, other_section in enumerate(sections):
                if i != j:
                    other_words = set(word.lower() for word in other_section.split()[:20])  # First 20 words
                    if section_words.intersection(other_words):
                        cross_refs += 1
        
        if total_possible == 0:
            return 0.5
        
        return min(1.0, cross_refs / total_possible)
    
    def _assess_transition_quality(self, content: str) -> float:
        """Assess quality of transitions between ideas"""
        total_transitions = 0
        quality_transitions = 0
        
        for category, words in self.transition_words.items():
            for word in words:
                count = content.lower().count(word)
                total_transitions += count
                
                # Higher quality for varied transition types
                if category in ["causal", "adversative", "exemplification"]:
                    quality_transitions += count * 1.5
                else:
                    quality_transitions += count
        
        if total_transitions == 0:
            return 0.3  # Neutral for no transitions
        
        # Normalize by content length
        word_count = len(content.split())
        transition_density = total_transitions / max(1, word_count / 100)
        quality_ratio = quality_transitions / total_transitions
        
        return min(1.0, (transition_density / 3) * quality_ratio)
    
    def _assess_topic_coherence(self, content: str) -> float:
        """Assess staying on topic throughout the document"""
        sections = re.split(r'^###\s+', content, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]
        
        if len(sections) < 2:
            return 0.7  # Assume coherent for single section
        
        # Extract key topics from each section
        section_topics = []
        for section in sections:
            words = re.findall(r'\b[a-z]{4,}\b', section.lower())
            topic_words = Counter(words).most_common(5)
            section_topics.append(set(word for word, count in topic_words))
        
        # Calculate topic overlap between sections
        total_overlap = 0
        comparisons = 0
        
        for i in range(len(section_topics)):
            for j in range(i + 1, len(section_topics)):
                overlap = len(section_topics[i].intersection(section_topics[j]))
                total_overlap += overlap
                comparisons += 1
        
        if comparisons == 0:
            return 0.7
        
        avg_overlap = total_overlap / comparisons
        coherence_score = min(1.0, avg_overlap / 3)  # Normalize
        
        return coherence_score
    
    def _assess_sentence_complexity(self, content: str) -> float:
        """Assess sentence complexity and readability"""
        # Remove markdown formatting for sentence analysis
        clean_content = re.sub(r'[#*`\[\]()|-]', '', content)
        sentences = re.split(r'[.!?]+', clean_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        # Calculate average sentence length
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # Optimal range is 15-20 words per sentence
        if 15 <= avg_length <= 20:
            complexity_score = 1.0
        elif 10 <= avg_length < 15 or 20 < avg_length <= 25:
            complexity_score = 0.8
        elif 5 <= avg_length < 10 or 25 < avg_length <= 30:
            complexity_score = 0.6
        else:
            complexity_score = 0.4
        
        return complexity_score
    
    def _assess_vocabulary_level(self, content: str) -> float:
        """Assess vocabulary complexity and appropriateness"""
        words = re.findall(r'\b[a-z]+\b', content.lower())
        
        if not words:
            return 0.5
        
        # Count syllables (rough approximation)
        def count_syllables(word):
            vowels = 'aeiouy'
            count = sum(1 for char in word if char in vowels)
            if word.endswith('e'):
                count -= 1
            return max(1, count)
        
        syllable_counts = [count_syllables(word) for word in words]
        avg_syllables = sum(syllable_counts) / len(syllable_counts)
        
        # Optimal range is 1.5-2.5 syllables per word
        if 1.5 <= avg_syllables <= 2.5:
            vocab_score = 1.0
        elif 1.0 <= avg_syllables < 1.5 or 2.5 < avg_syllables <= 3.0:
            vocab_score = 0.8
        else:
            vocab_score = 0.6
        
        return vocab_score
    
    def _assess_clarity(self, content: str) -> float:
        """Assess overall clarity of expression"""
        score = 0.0
        
        # Check for clear, direct language
        passive_voice = len(re.findall(r'\b(is|are|was|were|been|being)\s+\w+ed\b', content))
        active_voice = len(re.findall(r'\b\w+s\s+\w+', content))  # Rough approximation
        
        if active_voice > 0:
            active_ratio = active_voice / (active_voice + passive_voice)
            score += min(0.3, active_ratio * 0.3)
        
        # Check for concrete vs abstract language
        concrete_words = len(re.findall(r'\b(equipment|computer|office|desk|phone|software|system)\b', content.lower()))
        abstract_words = len(re.findall(r'\b(concept|idea|approach|methodology|framework|paradigm)\b', content.lower()))
        
        total_descriptive = concrete_words + abstract_words
        if total_descriptive > 0:
            concrete_ratio = concrete_words / total_descriptive
            score += min(0.3, concrete_ratio * 0.3)
        
        # Check for jargon density
        word_count = len(content.split())
        jargon_words = len(re.findall(r'\b\w{10,}\b', content))  # Very long words as proxy for jargon
        jargon_ratio = jargon_words / max(1, word_count)
        
        if jargon_ratio < 0.05:  # Less than 5% jargon
            score += 0.4
        elif jargon_ratio < 0.10:
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_conciseness(self, content: str) -> float:
        """Assess conciseness vs verbosity"""
        words = content.split()
        word_count = len(words)
        
        # Check for redundant phrases
        redundant_phrases = [
            "in order to", "due to the fact that", "it is important to note that",
            "please be aware that", "it should be noted that", "for the purpose of"
        ]
        
        redundancy_count = sum(content.lower().count(phrase) for phrase in redundant_phrases)
        
        # Check for filler words
        filler_words = ["very", "really", "quite", "rather", "somewhat", "fairly"]
        filler_count = sum(content.lower().count(word) for word in filler_words)
        
        # Calculate conciseness score
        total_fluff = redundancy_count + filler_count
        fluff_ratio = total_fluff / max(1, word_count / 100)  # Per 100 words
        
        conciseness_score = max(0.0, 1.0 - (fluff_ratio / 5))  # Penalize excessive fluff
        
        return conciseness_score
    
    def _assess_domain_style(self, content: str, domain: str) -> float:
        """Assess adherence to domain-specific style requirements"""
        content_lower = content.lower()
        
        if domain == "policy":
            # Policy documents should use formal, directive language
            directive_words = ["must", "shall", "required", "mandatory", "prohibited"]
            directive_count = sum(content_lower.count(word) for word in directive_words)
            
            # Should have clear structure with numbered or bulleted requirements
            has_structure = bool(re.search(r'^\s*\d+\.|^\s*[-*]', content, re.MULTILINE))
            
            style_score = min(1.0, (directive_count / 10) + (0.5 if has_structure else 0))
            
        elif domain == "technical":
            # Technical documents should be precise and specific
            technical_indicators = ["configure", "install", "parameter", "specification"]
            tech_count = sum(content_lower.count(word) for word in technical_indicators)
            
            # Should have code examples or specific values
            has_specifics = bool(re.search(r'`[^`]+`|\d+\.\d+|[A-Z_]{3,}', content))
            
            style_score = min(1.0, (tech_count / 8) + (0.4 if has_specifics else 0))
            
        elif domain == "business":
            # Business documents should focus on outcomes and metrics
            business_indicators = ["objective", "goal", "metric", "deliverable", "timeline"]
            business_count = sum(content_lower.count(word) for word in business_indicators)
            
            # Should have quantifiable elements
            has_metrics = bool(re.search(r'\d+%|\$\d+|\d+\s+(days|weeks|months)', content))
            
            style_score = min(1.0, (business_count / 6) + (0.3 if has_metrics else 0))
            
        elif domain == "strategy":
            # Strategy documents should be comprehensive and specific
            strategy_indicators = ["target", "segment", "competitive", "pricing", "marketing", "positioning"]
            strategy_count = sum(content_lower.count(word) for word in strategy_indicators)
            
            # Should have all key strategy components
            has_segments = any(word in content_lower for word in ["segment", "target", "audience"])
            has_competitive = any(word in content_lower for word in ["competitive", "competitor", "positioning"])
            has_pricing = any(word in content_lower for word in ["pricing", "price", "cost"])
            has_marketing = any(word in content_lower for word in ["marketing", "channel", "campaign"])
            
            component_score = sum([has_segments, has_competitive, has_pricing, has_marketing]) / 4
            
            style_score = min(1.0, (strategy_count / 8) + (component_score * 0.4))
            
        else:
            style_score = 0.5  # Neutral for unknown domains
        
        return style_score
    
    def _has_logical_section_order(self, headers: List[str]) -> bool:
        """Check if section headers follow a logical order"""
        if len(headers) < 3:
            return True  # Too few to judge
        
        # Common logical patterns
        logical_patterns = [
            ["introduction", "overview", "requirements", "implementation", "conclusion"],
            ["summary", "details", "analysis", "recommendations"],
            ["background", "methodology", "results", "discussion"],
            ["objectives", "approach", "execution", "evaluation"]
        ]
        
        headers_lower = [h.lower() for h in headers]
        
        # Check if headers match any logical pattern
        for pattern in logical_patterns:
            matches = 0
            for header in headers_lower:
                for pattern_word in pattern:
                    if pattern_word in header:
                        matches += 1
                        break
            
            if matches >= len(headers) * 0.6:  # 60% match
                return True
        
        return False
