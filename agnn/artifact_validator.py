"""
Tier-2 Week 2: Advanced Artifact Validation

Provides semantic validation and quality scoring for different types of artifacts.
Replaces basic heuristic detection with sophisticated content assessment.
Enhanced with critical fixes for repetition detection and completeness validation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import re
import math


@dataclass
class ArtifactScore:
    """Comprehensive artifact quality assessment"""
    completeness: float      # 0.0-1.0: How complete is the artifact
    coherence: float        # 0.0-1.0: Semantic coherence between sections
    structure: float        # 0.0-1.0: Quality of document structure
    domain_compliance: float # 0.0-1.0: Adherence to domain-specific requirements
    overall: float          # 0.0-1.0: Weighted overall score
    artifact_type: str      # Detected type: policy, plan, analysis, etc.
    missing_elements: List[str]  # What's missing for completeness
    quality_issues: List[str]    # Identified quality problems


@dataclass
class ArtifactRequirements:
    """Requirements for different artifact types"""
    min_sections: int
    min_words: int
    required_elements: List[str]
    optional_elements: List[str]
    section_patterns: List[str]
    quality_indicators: List[str]


class ArtifactValidator:
    """
    Advanced artifact validation with semantic analysis and quality scoring.
    Detects artifact type and assesses completeness, coherence, and quality.
    Enhanced with fixes for strategy documents and stricter validation.
    """
    
    def __init__(self):
        """Initialize with domain-specific requirements"""
        self.artifact_requirements = {
            "policy": ArtifactRequirements(
                min_sections=5,
                min_words=400,
                required_elements=["requirements", "procedures", "standards", "guidelines", "compliance"],
                optional_elements=["exceptions", "enforcement", "review", "updates"],
                section_patterns=[r"communication", r"equipment", r"security", r"work hours", r"performance"],
                quality_indicators=["must", "shall", "required", "mandatory", "prohibited", "allowed"]
            ),
            "strategy": ArtifactRequirements(
                min_sections=4,
                min_words=600,
                required_elements=["target segments", "pricing", "marketing channels", "competitive positioning"],
                optional_elements=["timeline", "metrics", "risks", "resources"],
                section_patterns=[r"target", r"segment", r"pricing", r"marketing", r"competitive", r"positioning"],
                quality_indicators=["strategy", "market", "competitive", "positioning", "pricing", "target"]
            ),
            "business": ArtifactRequirements(
                min_sections=4,
                min_words=500,
                required_elements=["market analysis", "strategy", "implementation", "metrics"],
                optional_elements=["risks", "timeline", "budget", "resources"],
                section_patterns=[r"market", r"strategy", r"implementation", r"business", r"plan"],
                quality_indicators=["business", "market", "strategy", "implementation", "analysis"]
            ),
            "plan": ArtifactRequirements(
                min_sections=4,
                min_words=300,
                required_elements=["timeline", "steps", "objectives", "deliverables"],
                optional_elements=["resources", "risks", "dependencies", "milestones"],
                section_patterns=[r"phase", r"step", r"stage", r"milestone", r"timeline"],
                quality_indicators=["by", "after", "before", "deadline", "complete", "deliver"]
            ),
            "analysis": ArtifactRequirements(
                min_sections=4,
                min_words=500,
                required_elements=["findings", "conclusions", "methodology", "data"],
                optional_elements=["recommendations", "limitations", "future work"],
                section_patterns=[r"analysis", r"results", r"findings", r"conclusion", r"summary"],
                quality_indicators=["shows", "indicates", "suggests", "concludes", "evidence", "data"]
            ),
            "report": ArtifactRequirements(
                min_sections=3,
                min_words=350,
                required_elements=["summary", "details", "conclusions"],
                optional_elements=["recommendations", "appendix", "references"],
                section_patterns=[r"summary", r"overview", r"details", r"results", r"conclusion"],
                quality_indicators=["report", "status", "progress", "update", "summary"]
            ),
            "guide": ArtifactRequirements(
                min_sections=3,
                min_words=300,
                required_elements=["instructions", "steps", "examples"],
                optional_elements=["troubleshooting", "tips", "best practices"],
                section_patterns=[r"how to", r"step", r"instruction", r"guide", r"tutorial"],
                quality_indicators=["first", "next", "then", "finally", "example", "note"]
            )
        }
    
    def validate_artifact(self, content: str, hint_type: Optional[str] = None) -> ArtifactScore:
        """
        Comprehensive artifact validation with quality scoring.
        
        Args:
            content: The artifact content to validate
            hint_type: Optional hint about expected artifact type
            
        Returns:
            ArtifactScore with detailed quality assessment
        """
        if not content or len(content.strip()) < 50:
            return ArtifactScore(0.0, 0.0, 0.0, 0.0, 0.0, "unknown", ["Content too short"], ["Insufficient content"])
        
        # Detect artifact type
        detected_type = hint_type or self.detect_artifact_type(content)
        requirements = self.artifact_requirements.get(detected_type, self.artifact_requirements["report"])
        
        # Calculate individual scores
        completeness = self._assess_completeness(content, requirements)
        coherence = self._assess_coherence(content)
        structure = self._assess_structure(content, requirements)
        domain_compliance = self._assess_domain_compliance(content, requirements)
        
        # Calculate weighted overall score
        overall = (
            0.30 * completeness +
            0.25 * coherence +
            0.25 * structure +
            0.20 * domain_compliance
        )
        
        # Identify missing elements and quality issues
        missing_elements = self._find_missing_elements(content, requirements)
        quality_issues = self._identify_quality_issues(content, requirements)
        
        return ArtifactScore(
            completeness=completeness,
            coherence=coherence,
            structure=structure,
            domain_compliance=domain_compliance,
            overall=overall,
            artifact_type=detected_type,
            missing_elements=missing_elements,
            quality_issues=quality_issues
        )
    
    def detect_artifact_type(self, content: str) -> str:
        """
        Auto-detect artifact type based on content analysis.
        
        Args:
            content: The content to analyze
            
        Returns:
            Detected artifact type (policy, plan, analysis, report, guide, strategy)
        """
        content_lower = content.lower()
        
        # Type detection based on keywords and patterns
        type_scores = {}
        
        for artifact_type, requirements in self.artifact_requirements.items():
            score = 0.0
            
            # Check for required elements
            for element in requirements.required_elements:
                if element in content_lower:
                    score += 2.0
            
            # Check for section patterns
            for pattern in requirements.section_patterns:
                if re.search(pattern, content_lower):
                    score += 1.5
            
            # Check for quality indicators
            for indicator in requirements.quality_indicators:
                score += content_lower.count(indicator) * 0.5
            
            type_scores[artifact_type] = score
        
        # Return type with highest score, default to "report"
        if not type_scores or max(type_scores.values()) == 0:
            return "report"
        
        return max(type_scores, key=type_scores.get)
    
    def _assess_completeness(self, content: str, requirements: ArtifactRequirements) -> float:
        """Assess how complete the artifact is based on requirements (enhanced)"""
        score = 0.0
        
        # Word count score (0.0-0.25) - reduced weight
        word_count = len(content.split())
        word_score = min(1.0, word_count / requirements.min_words) * 0.25
        score += word_score
        
        # Section count score (0.0-0.25) - reduced weight
        sections = self._count_sections(content)
        section_score = min(1.0, sections / requirements.min_sections) * 0.25
        score += section_score
        
        # Required elements score (0.0-0.5) - increased weight and stricter
        found_elements = 0
        content_lower = content.lower()
        
        for element in requirements.required_elements:
            # More strict matching - require substantial coverage of each element
            element_coverage = content_lower.count(element.lower())
            if element_coverage >= 2:  # Must appear at least twice
                found_elements += 1
            elif element_coverage == 1:
                # Partial credit only if element appears in a section header
                if any(pattern in content_lower for pattern in requirements.section_patterns 
                       if element.lower() in pattern):
                    found_elements += 0.5
        
        element_score = (found_elements / len(requirements.required_elements)) * 0.5
        score += element_score
        
        return min(1.0, score)
    
    def _assess_coherence(self, content: str) -> float:
        """Assess semantic coherence between sections"""
        sections = self._extract_sections(content)
        if len(sections) < 2:
            return 0.5  # Neutral score for single section
        
        coherence_score = 0.0
        
        # Check for logical flow indicators
        flow_indicators = ["first", "next", "then", "finally", "however", "therefore", "additionally", "furthermore"]
        flow_count = sum(1 for indicator in flow_indicators if indicator in content.lower())
        flow_score = min(1.0, flow_count / 5) * 0.4
        
        # Check for cross-references between sections
        cross_refs = 0
        for i, section in enumerate(sections):
            for j, other_section in enumerate(sections):
                if i != j and any(word in section.lower() for word in other_section.lower().split()[:10]):
                    cross_refs += 1
        
        ref_score = min(1.0, cross_refs / len(sections)) * 0.3
        
        # Check for consistent terminology
        key_terms = self._extract_key_terms(content)
        term_consistency = self._assess_term_consistency(sections, key_terms) * 0.3
        
        coherence_score = flow_score + ref_score + term_consistency
        return min(1.0, coherence_score)
    
    def _assess_structure(self, content: str, requirements: ArtifactRequirements) -> float:
        """Assess quality of document structure"""
        structure_score = 0.0
        
        # Header hierarchy score (0.0-0.4)
        has_title = bool(re.search(r'^##?\s+', content, re.MULTILINE))
        has_sections = bool(re.search(r'^###\s+', content, re.MULTILINE))
        has_subsections = bool(re.search(r'^####\s+', content, re.MULTILINE))
        
        hierarchy_score = 0.0
        if has_title: hierarchy_score += 0.15
        if has_sections: hierarchy_score += 0.15
        if has_subsections: hierarchy_score += 0.1
        structure_score += hierarchy_score
        
        # List and formatting score (0.0-0.3)
        has_bullets = bool(re.search(r'^\s*[-*•]\s+', content, re.MULTILINE))
        has_numbers = bool(re.search(r'^\s*\d+\.\s+', content, re.MULTILINE))
        has_bold = '**' in content
        has_tables = '|' in content and content.count('|') >= 6
        
        format_score = 0.0
        if has_bullets: format_score += 0.1
        if has_numbers: format_score += 0.1
        if has_bold: format_score += 0.05
        if has_tables: format_score += 0.05
        structure_score += format_score
        
        # Section balance score (0.0-0.3)
        sections = self._extract_sections(content)
        if sections:
            section_lengths = [len(section.split()) for section in sections]
            avg_length = sum(section_lengths) / len(section_lengths)
            variance = sum((length - avg_length) ** 2 for length in section_lengths) / len(section_lengths)
            balance_score = max(0.0, 1.0 - (variance / (avg_length ** 2))) * 0.3
            structure_score += balance_score
        
        return min(1.0, structure_score)
    
    def _assess_domain_compliance(self, content: str, requirements: ArtifactRequirements) -> float:
        """Assess adherence to domain-specific requirements"""
        compliance_score = 0.0
        content_lower = content.lower()
        
        # Quality indicators score (0.0-0.5)
        found_indicators = 0
        for indicator in requirements.quality_indicators:
            if indicator in content_lower:
                found_indicators += 1
        
        indicator_score = min(1.0, found_indicators / len(requirements.quality_indicators)) * 0.5
        compliance_score += indicator_score
        
        # Section pattern score (0.0-0.3)
        found_patterns = 0
        for pattern in requirements.section_patterns:
            if re.search(pattern, content_lower):
                found_patterns += 1
        
        pattern_score = min(1.0, found_patterns / len(requirements.section_patterns)) * 0.3
        compliance_score += pattern_score
        
        # Optional elements bonus (0.0-0.2)
        found_optional = 0
        for element in requirements.optional_elements:
            if element in content_lower:
                found_optional += 1
        
        optional_score = min(1.0, found_optional / len(requirements.optional_elements)) * 0.2
        compliance_score += optional_score
        
        return min(1.0, compliance_score)
    
    def _count_sections(self, content: str) -> int:
        """Count the number of sections in the content"""
        return len(re.findall(r'^###\s+', content, re.MULTILINE))
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract individual sections from the content"""
        sections = re.split(r'^###\s+', content, flags=re.MULTILINE)
        return [section.strip() for section in sections if section.strip()]
    
    def _extract_key_terms(self, content: str) -> Set[str]:
        """Extract key terms that should be used consistently"""
        # Simple extraction of capitalized terms and important words
        words = re.findall(r'\b[A-Z][a-z]+\b', content)
        return set(word.lower() for word in words if len(word) > 3)
    
    def _assess_term_consistency(self, sections: List[str], key_terms: Set[str]) -> float:
        """Assess consistency of key term usage across sections"""
        if not key_terms or len(sections) < 2:
            return 0.5
        
        term_usage = {}
        for term in key_terms:
            term_usage[term] = sum(1 for section in sections if term in section.lower())
        
        # Terms should appear in multiple sections for consistency
        consistent_terms = sum(1 for count in term_usage.values() if count > 1)
        return consistent_terms / len(key_terms) if key_terms else 0.5
    
    def _find_missing_elements(self, content: str, requirements: ArtifactRequirements) -> List[str]:
        """Identify missing required elements"""
        content_lower = content.lower()
        missing = []
        
        for element in requirements.required_elements:
            if element not in content_lower:
                missing.append(element)
        
        return missing
    
    def _identify_quality_issues(self, content: str, requirements: ArtifactRequirements) -> List[str]:
        """Identify potential quality issues"""
        issues = []
        
        # Check word count
        word_count = len(content.split())
        if word_count < requirements.min_words:
            issues.append(f"Content too short ({word_count} words, need {requirements.min_words})")
        
        # Check section count
        section_count = self._count_sections(content)
        if section_count < requirements.min_sections:
            issues.append(f"Too few sections ({section_count}, need {requirements.min_sections})")
        
        # Check for very short sections
        sections = self._extract_sections(content)
        short_sections = [i for i, section in enumerate(sections) if len(section.split()) < 20]
        if short_sections:
            issues.append(f"Sections too short: {short_sections}")
        
        # Check for missing formatting
        if '**' not in content:
            issues.append("No bold formatting found")
        
        if not re.search(r'^\s*[-*•]\s+', content, re.MULTILINE):
            issues.append("No bullet points found")
        
        return issues


def validate_artifact_content(content: str, artifact_type: Optional[str] = None) -> ArtifactScore:
    """
    Convenience function for artifact validation.
    
    Args:
        content: The artifact content to validate
        artifact_type: Optional hint about expected artifact type
        
    Returns:
        ArtifactScore with detailed quality assessment
    """
    validator = ArtifactValidator()
    return validator.validate_artifact(content, artifact_type)




