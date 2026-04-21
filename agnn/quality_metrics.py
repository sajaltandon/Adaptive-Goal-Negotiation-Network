"""
Tier-2 Week 2: Quality Metrics

Provides specialized quality scoring for different types of artifacts.
Integrates with artifact validator and content analyzer for comprehensive assessment.
Enhanced with strategy domain support and critical fixes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
from .artifact_validator import ArtifactScore, ArtifactValidator
from .content_analyzer import ContentAnalyzer, StructureScore, CoherenceScore, ReadabilityScore


@dataclass
class QualityScore:
    """Comprehensive quality assessment for artifacts"""
    completeness: float      # 0.0-1.0: How complete is the artifact
    coherence: float        # 0.0-1.0: Semantic coherence and consistency
    structure: float        # 0.0-1.0: Document structure quality
    readability: float      # 0.0-1.0: Clarity and readability
    domain_compliance: float # 0.0-1.0: Adherence to domain requirements
    overall: float          # 0.0-1.0: Weighted overall quality score
    
    # Detailed breakdowns
    artifact_score: Optional[ArtifactScore] = None
    structure_score: Optional[StructureScore] = None
    coherence_score: Optional[CoherenceScore] = None
    readability_score: Optional[ReadabilityScore] = None
    
    # Quality indicators
    strengths: List[str] = None
    weaknesses: List[str] = None
    recommendations: List[str] = None


@dataclass
class QualityThresholds:
    """Quality thresholds for different artifact types"""
    excellent: float = 0.9    # Excellent quality threshold
    good: float = 0.75        # Good quality threshold
    acceptable: float = 0.6   # Minimum acceptable quality
    poor: float = 0.4         # Below this is poor quality


class QualityMetrics:
    """
    Quality scoring metrics for different types of artifacts.
    Provides comprehensive quality assessment using multiple analysis techniques.
    Enhanced with strategy domain support and improved calibration.
    """
    
    def __init__(self):
        """Initialize with quality assessment components"""
        self.artifact_validator = ArtifactValidator()
        self.content_analyzer = ContentAnalyzer()
        
        # Quality weights for different artifact types (enhanced)
        self.quality_weights = {
            "policy": {
                "completeness": 0.35,
                "coherence": 0.20,
                "structure": 0.25,
                "readability": 0.10,
                "domain_compliance": 0.10
            },
            "plan": {
                "completeness": 0.30,
                "coherence": 0.25,
                "structure": 0.20,
                "readability": 0.15,
                "domain_compliance": 0.10
            },
            "analysis": {
                "completeness": 0.25,
                "coherence": 0.30,
                "structure": 0.20,
                "readability": 0.15,
                "domain_compliance": 0.10
            },
            "report": {
                "completeness": 0.30,
                "coherence": 0.25,
                "structure": 0.20,
                "readability": 0.20,
                "domain_compliance": 0.05
            },
            "guide": {
                "completeness": 0.25,
                "coherence": 0.20,
                "structure": 0.25,
                "readability": 0.25,
                "domain_compliance": 0.05
            },
            "strategy": {
                "completeness": 0.40,  # Critical for strategy documents
                "coherence": 0.25,     # Must be logical and connected
                "structure": 0.20,     # Good organization important
                "readability": 0.10,   # Less critical than completeness
                "domain_compliance": 0.05  # Strategy-specific elements
            }
        }
        
        # Default weights for unknown types
        self.default_weights = {
            "completeness": 0.30,
            "coherence": 0.25,
            "structure": 0.20,
            "readability": 0.15,
            "domain_compliance": 0.10
        }
    
    def assess_quality(self, content: str, artifact_type: Optional[str] = None, domain: Optional[str] = None) -> QualityScore:
        """
        Comprehensive quality assessment of an artifact.
        
        Args:
            content: The artifact content to assess
            artifact_type: Type of artifact (policy, plan, analysis, etc.)
            domain: Domain context (policy, technical, business, strategy)
            
        Returns:
            QualityScore with detailed quality assessment
        """
        if not content or len(content.strip()) < 20:
            return self._create_empty_quality_score("Content too short")
        
        # Get detailed assessments from each component
        artifact_score = self.artifact_validator.validate_artifact(content, artifact_type)
        structure_score = self.content_analyzer.analyze_structure(content)
        coherence_score = self.content_analyzer.analyze_coherence(content)
        readability_score = self.content_analyzer.analyze_readability(content)
        
        # Domain compliance assessment
        if domain:
            domain_results = self.content_analyzer.check_domain_requirements(content, domain)
            domain_compliance = domain_results["overall"]
        else:
            domain_compliance = artifact_score.domain_compliance
        
        # Get quality weights for this artifact type
        detected_type = artifact_score.artifact_type
        
        # Map business/strategy artifacts to strategy weights
        if detected_type in ["business", "plan"] and domain == "strategy":
            detected_type = "strategy"
        
        weights = self.quality_weights.get(detected_type, self.default_weights)
        
        # Calculate weighted overall score
        overall = (
            weights["completeness"] * artifact_score.completeness +
            weights["coherence"] * coherence_score.overall +
            weights["structure"] * structure_score.overall +
            weights["readability"] * readability_score.overall +
            weights["domain_compliance"] * domain_compliance
        )
        
        # Identify strengths, weaknesses, and recommendations
        strengths, weaknesses, recommendations = self._analyze_quality_aspects(
            artifact_score, structure_score, coherence_score, readability_score, domain_compliance
        )
        
        return QualityScore(
            completeness=artifact_score.completeness,
            coherence=coherence_score.overall,
            structure=structure_score.overall,
            readability=readability_score.overall,
            domain_compliance=domain_compliance,
            overall=overall,
            artifact_score=artifact_score,
            structure_score=structure_score,
            coherence_score=coherence_score,
            readability_score=readability_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def calculate_policy_quality(self, content: str) -> QualityScore:
        """Specialized quality assessment for policy documents"""
        return self.assess_quality(content, artifact_type="policy", domain="policy")
    
    def calculate_plan_quality(self, content: str) -> QualityScore:
        """Specialized quality assessment for plan documents"""
        return self.assess_quality(content, artifact_type="plan", domain="business")
    
    def calculate_strategy_quality(self, content: str) -> QualityScore:
        """Specialized quality assessment for strategy documents"""
        return self.assess_quality(content, artifact_type="strategy", domain="strategy")
    
    def calculate_analysis_quality(self, content: str) -> QualityScore:
        """Specialized quality assessment for analysis documents"""
        return self.assess_quality(content, artifact_type="analysis", domain="business")
    
    def calculate_technical_quality(self, content: str) -> QualityScore:
        """Specialized quality assessment for technical documents"""
        return self.assess_quality(content, artifact_type="guide", domain="technical")
    
    def get_quality_level(self, score: float, thresholds: Optional[QualityThresholds] = None) -> str:
        """Convert numeric quality score to descriptive level"""
        if thresholds is None:
            thresholds = QualityThresholds()
        
        if score >= thresholds.excellent:
            return "Excellent"
        elif score >= thresholds.good:
            return "Good"
        elif score >= thresholds.acceptable:
            return "Acceptable"
        elif score >= thresholds.poor:
            return "Poor"
        else:
            return "Very Poor"
    
    def is_acceptable_quality(self, quality_score: QualityScore, min_threshold: float = 0.6) -> bool:
        """Check if artifact meets minimum quality standards"""
        return quality_score.overall >= min_threshold
    
    def get_improvement_priority(self, quality_score: QualityScore) -> List[Tuple[str, float, str]]:
        """Get prioritized list of areas for improvement"""
        aspects = [
            ("completeness", quality_score.completeness, "Add missing sections or required elements"),
            ("coherence", quality_score.coherence, "Improve logical flow and consistency"),
            ("structure", quality_score.structure, "Enhance document organization and formatting"),
            ("readability", quality_score.readability, "Simplify language and improve clarity"),
            ("domain_compliance", quality_score.domain_compliance, "Follow domain-specific conventions")
        ]
        
        # Sort by score (lowest first = highest priority for improvement)
        aspects.sort(key=lambda x: x[1])
        
        # Only return aspects that need improvement (< 0.7)
        return [(aspect, score, rec) for aspect, score, rec in aspects if score < 0.7]
    
    def compare_quality(self, content1: str, content2: str, artifact_type: Optional[str] = None) -> Dict[str, float]:
        """Compare quality between two artifacts"""
        score1 = self.assess_quality(content1, artifact_type)
        score2 = self.assess_quality(content2, artifact_type)
        
        return {
            "content1_overall": score1.overall,
            "content2_overall": score2.overall,
            "difference": score2.overall - score1.overall,
            "better": "content2" if score2.overall > score1.overall else "content1",
            "completeness_diff": score2.completeness - score1.completeness,
            "coherence_diff": score2.coherence - score1.coherence,
            "structure_diff": score2.structure - score1.structure,
            "readability_diff": score2.readability - score1.readability,
            "domain_compliance_diff": score2.domain_compliance - score1.domain_compliance
        }
    
    def _create_empty_quality_score(self, reason: str) -> QualityScore:
        """Create a quality score for invalid/empty content"""
        return QualityScore(
            completeness=0.0,
            coherence=0.0,
            structure=0.0,
            readability=0.0,
            domain_compliance=0.0,
            overall=0.0,
            strengths=[],
            weaknesses=[reason],
            recommendations=["Provide substantial content for analysis"]
        )
    
    def _analyze_quality_aspects(self, artifact_score: ArtifactScore, structure_score: StructureScore, 
                                coherence_score: CoherenceScore, readability_score: ReadabilityScore,
                                domain_compliance: float) -> Tuple[List[str], List[str], List[str]]:
        """Analyze quality aspects to identify strengths, weaknesses, and recommendations"""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze completeness
        if artifact_score.completeness >= 0.8:
            strengths.append("Well-structured with all required elements")
        elif artifact_score.completeness < 0.6:
            weaknesses.append("Missing required sections or elements")
            recommendations.append("Add missing sections: " + ", ".join(artifact_score.missing_elements[:3]))
        
        # Analyze structure
        if structure_score.overall >= 0.8:
            strengths.append("Excellent document structure and formatting")
        elif structure_score.overall < 0.6:
            weaknesses.append("Poor document structure")
            if structure_score.hierarchy_quality < 0.6:
                recommendations.append("Improve header hierarchy with clear section levels")
            if structure_score.formatting_quality < 0.6:
                recommendations.append("Add more formatting elements (bullets, bold text, tables)")
        
        # Analyze coherence
        if coherence_score.overall >= 0.8:
            strengths.append("Highly coherent with good logical flow")
        elif coherence_score.overall < 0.6:
            weaknesses.append("Lacks coherence and logical flow")
            if coherence_score.transition_quality < 0.6:
                recommendations.append("Add transition words and phrases between sections")
            if coherence_score.semantic_consistency < 0.6:
                recommendations.append("Use consistent terminology throughout")
        
        # Analyze readability
        if readability_score.overall >= 0.8:
            strengths.append("Clear and readable content")
        elif readability_score.overall < 0.6:
            weaknesses.append("Difficult to read or understand")
            if readability_score.sentence_complexity < 0.6:
                recommendations.append("Simplify sentence structure and length")
            if readability_score.clarity < 0.6:
                recommendations.append("Use more concrete and direct language")
        
        # Analyze domain compliance
        if domain_compliance >= 0.8:
            strengths.append("Follows domain conventions well")
        elif domain_compliance < 0.6:
            weaknesses.append("Does not follow domain-specific requirements")
            recommendations.append("Include more domain-appropriate vocabulary and style")
        
        # Overall quality assessment
        overall = (artifact_score.completeness + structure_score.overall + 
                  coherence_score.overall + readability_score.overall + domain_compliance) / 5
        
        if overall >= 0.8:
            strengths.append("High overall quality")
        elif overall < 0.6:
            weaknesses.append("Overall quality needs improvement")
            recommendations.append("Focus on the highest priority improvements first")
        
        return strengths, weaknesses, recommendations


def assess_artifact_quality(content: str, artifact_type: Optional[str] = None, domain: Optional[str] = None) -> QualityScore:
    """Convenience function for comprehensive artifact quality assessment"""
    metrics = QualityMetrics()
    return metrics.assess_quality(content, artifact_type, domain)


def is_high_quality_artifact(content: str, min_threshold: float = 0.75) -> bool:
    """Quick check if an artifact meets high quality standards"""
    quality_score = assess_artifact_quality(content)
    return quality_score.overall >= min_threshold




