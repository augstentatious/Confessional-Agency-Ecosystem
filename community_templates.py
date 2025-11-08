"""
Community Templates and Governance System
Federated ethical template curation for CAE

Author: CAE Community & John Augustine Young
License: MIT
"""

import json
import time
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import requests
import threading
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Data Structures ====================

class TemplateStatus(Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"

class VoteType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"

@dataclass
class CommunityTemplate:
    """Community-contributed ethical template"""
    template_id: str
    name: str
    description: str
    category: str
    template_text: str
    author_id: str
    author_name: str
    created_at: datetime
    updated_at: datetime
    status: TemplateStatus
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 0.0
    average_rating: float = 0.0
    rating_count: int = 0
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if isinstance(self.status, str):
            self.status = TemplateStatus(self.status)

@dataclass
class TemplateVote:
    """Vote on community template"""
    vote_id: str
    template_id: str
    voter_id: str
    vote_type: VoteType
    confidence: float  # 0-1 confidence in vote
    rationale: str
    created_at: datetime
    voter_reputation: float = 1.0
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.vote_type, str):
            self.vote_type = VoteType(self.vote_type)

@dataclass
class TemplateUsage:
    """Record of template usage in CAE system"""
    usage_id: str
    template_id: str
    query_hash: str
    context_hash: str
    was_successful: bool
    user_rating: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

@dataclass
class CommunityMember:
    """Community member profile"""
    member_id: str
    name: str
    email: str
    reputation_score: float = 1.0
    join_date: datetime = field(default_factory=datetime.now)
    expertise_areas: List[str] = field(default_factory=list)
    total_votes: int = 0
    successful_templates: int = 0
    
    def __post_init__(self):
        if isinstance(self.join_date, str):
            self.join_date = datetime.fromisoformat(self.join_date)

# ==================== Database Layer ====================

class TemplateDatabase:
    """SQLite database for community templates"""
    
    def __init__(self, db_path: str = "community_templates.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS templates (
                    template_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    template_text TEXT NOT NULL,
                    author_id TEXT,
                    author_name TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT,
                    version TEXT,
                    tags TEXT,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    average_rating REAL DEFAULT 0.0,
                    rating_count INTEGER DEFAULT 0
                )
            ''')
            
            # Votes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS votes (
                    vote_id TEXT PRIMARY KEY,
                    template_id TEXT,
                    voter_id TEXT,
                    vote_type TEXT,
                    confidence REAL,
                    rationale TEXT,
                    created_at TEXT,
                    voter_reputation REAL,
                    FOREIGN KEY (template_id) REFERENCES templates (template_id)
                )
            ''')
            
            # Usage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage (
                    usage_id TEXT PRIMARY KEY,
                    template_id TEXT,
                    query_hash TEXT,
                    context_hash TEXT,
                    was_successful BOOLEAN,
                    user_rating INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (template_id) REFERENCES templates (template_id)
                )
            ''')
            
            # Members table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS members (
                    member_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    reputation_score REAL DEFAULT 1.0,
                    join_date TEXT,
                    expertise_areas TEXT,
                    total_votes INTEGER DEFAULT 0,
                    successful_templates INTEGER DEFAULT 0
                )
            ''')
            
            conn.commit()
    
    def add_template(self, template: CommunityTemplate):
        """Add new template to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO templates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template.template_id,
                template.name,
                template.description,
                template.category,
                template.template_text,
                template.author_id,
                template.author_name,
                template.created_at.isoformat(),
                template.updated_at.isoformat(),
                template.status.value,
                template.version,
                json.dumps(template.tags),
                template.usage_count,
                template.success_rate,
                template.average_rating,
                template.rating_count
            ))
            
            conn.commit()
    
    def get_template(self, template_id: str) -> Optional[CommunityTemplate]:
        """Get template by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM templates WHERE template_id = ?', (template_id,))
            row = cursor.fetchone()
            
            if row:
                return CommunityTemplate(*row)
            return None
    
    def get_approved_templates(self, category: Optional[str] = None) -> List[CommunityTemplate]:
        """Get all approved templates"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if category:
                cursor.execute('''
                    SELECT * FROM templates 
                    WHERE status = ? AND category = ?
                    ORDER BY average_rating DESC, usage_count DESC
                ''', (TemplateStatus.APPROVED.value, category))
            else:
                cursor.execute('''
                    SELECT * FROM templates 
                    WHERE status = ?
                    ORDER BY average_rating DESC, usage_count DESC
                ''', (TemplateStatus.APPROVED.value,))
            
            rows = cursor.fetchall()
            return [CommunityTemplate(*row) for row in rows]
    
    def add_vote(self, vote: TemplateVote):
        """Add vote for template"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO votes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                vote.vote_id,
                vote.template_id,
                vote.voter_id,
                vote.vote_type.value,
                vote.confidence,
                vote.rationale,
                vote.created_at.isoformat(),
                vote.voter_reputation
            ))
            
            conn.commit()
    
    def get_template_votes(self, template_id: str) -> List[TemplateVote]:
        """Get all votes for a template"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM votes WHERE template_id = ?', (template_id,))
            rows = cursor.fetchall()
            
            return [TemplateVote(*row) for row in rows]
    
    def add_usage(self, usage: TemplateUsage):
        """Record template usage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                usage.usage_id,
                usage.template_id,
                usage.query_hash,
                usage.context_hash,
                usage.was_successful,
                usage.user_rating,
                usage.created_at.isoformat()
            ))
            
            conn.commit()
    
    def update_template_stats(self, template_id: str):
        """Update template statistics based on usage and votes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get usage stats
            cursor.execute('''
                SELECT COUNT(*), SUM(CASE WHEN was_successful THEN 1 ELSE 0 END)
                FROM usage WHERE template_id = ?
            ''', (template_id,))
            total_usage, successful_usage = cursor.fetchone()
            
            # Get rating stats
            cursor.execute('''
                SELECT AVG(user_rating), COUNT(user_rating)
                FROM usage WHERE template_id = ? AND user_rating IS NOT NULL
            ''', (template_id,))
            avg_rating, rating_count = cursor.fetchone()
            
            # Update template
            success_rate = successful_usage / total_usage if total_usage > 0 else 0
            avg_rating = avg_rating or 0
            rating_count = rating_count or 0
            
            cursor.execute('''
                UPDATE templates 
                SET usage_count = ?, success_rate = ?, 
                    average_rating = ?, rating_count = ?
                WHERE template_id = ?
            ''', (total_usage, success_rate, avg_rating, rating_count, template_id))
            
            conn.commit()

# ==================== Template Validation ====================

class TemplateValidator:
    """Validate community templates for quality and safety"""
    
    def __init__(self):
        self.required_fields = ['name', 'description', 'category', 'template_text']
        self.prohibited_content = [
            'harmful_instruction',
            'illegal_activity',
            'hate_speech',
            'discrimination',
            'violence_promotion'
        ]
    
    def validate_template(self, template: CommunityTemplate) -> Tuple[bool, List[str]]:
        \"\"\"Validate template for quality and safety\"\"\"\n        \
        errors = []\n        \n        # Check required fields\n        for field in self.required_fields:\n            if not getattr(template, field):\n                errors.append(f\"Missing required field: {field}\")\n        \n        # Check content safety\n        template_lower = template.template_text.lower()\n        \n        # Basic safety checks\n        dangerous_keywords = [\n            'how to make explosives', 'how to hack', 'how to kill',\n            'how to steal', 'how to scam', 'how to poison',\n            'illegal activity', 'criminal behavior', 'violence against'\n        ]\n        \n        for keyword in dangerous_keywords:\n            if keyword in template_lower:\n                errors.append(f\"Potentially dangerous content detected: {keyword}\")\n        \n        # Check for hate speech patterns\n        hate_patterns = [\n            r'\\b(hate|kill|destroy)\\s+(all|every)\\s+\\w+\\b',\n            r'\\b\\w+\\s+(are|is)\\s+(inferior|subhuman|evil)\\b'\n        ]\n        \n        import re\n        for pattern in hate_patterns:\n            if re.search(pattern, template_lower, re.IGNORECASE):\n                errors.append(\"Potential hate speech pattern detected\")\n        \n        # Check template quality\n        if len(template.template_text) < 50:\n            errors.append(\"Template text too short (< 50 characters)\")\n        \n        if len(template.template_text) > 2000:\n            errors.append(\"Template text too long (> 2000 characters)\")\n        \n        # Check description quality\n        if len(template.description) < 20:\n            errors.append(\"Description too short (< 20 characters)\")\n        \n        return len(errors) == 0, errors\n    \n    def evaluate_template_quality(self, template: CommunityTemplate) -> Dict[str, float]:\n        \"\"\"Evaluate template quality on multiple dimensions\"\"\"\n        \n        quality_scores = {}\n        \n        # Completeness score\n        required_fields = ['name', 'description', 'category', 'template_text', 'tags']\n        completeness = sum(1 for field in required_fields if getattr(template, field)) / len(required_fields)\n        quality_scores['completeness'] = completeness\n        \n        # Description quality\n        desc_length = len(template.description)\n        if desc_length >= 50:\n            quality_scores['description_quality'] = 1.0\n        elif desc_length >= 20:\n            quality_scores['description_quality'] = 0.7\n        else:\n            quality_scores['description_quality'] = 0.3\n        \n        # Template sophistication\n        template_text = template.template_text\n        question_marks = template_text.count('?')\n        reflection_indicators = template_text.lower().count('consider') + template_text.lower().count('reflect')\n        \n        sophistication_score = min(1.0, (question_marks * 0.2 + reflection_indicators * 0.3))\n        quality_scores['sophistication'] = sophistication_score\n        \n        # Category appropriateness\n        valid_categories = [\n            'moral_reasoning', 'ethical_dilemma', 'harm_prevention', \n            'consent_boundary', 'trauma_informed', 'community_wisdom'\n        ]\n        \n        if template.category in valid_categories:\n            quality_scores['category_appropriateness'] = 1.0\n        else:\n            quality_scores['category_appropriateness'] = 0.5\n        \n        # Overall quality score\n        quality_scores['overall'] = sum(quality_scores.values()) / len(quality_scores)\n        \n        return quality_scores\n
# ==================== Voting System ====================

class TemplateVotingSystem:\n    \"\"\"Democratic voting system for template approval\"\"\"\n    \n    def __init__(self, db: TemplateDatabase):\n        self.db = db\n        self.vote_threshold = 0.7  # 70% approval needed\n        self.min_votes = 10  # Minimum votes for decision\n        self.vote_timeout = timedelta(days=30)  # 30 days to vote\n    \n    def submit_vote(self, vote: TemplateVote) -> bool:\n        \"\"\"Submit vote for template\"\"\"\n        try:\n            # Check if template exists and is under review\n            template = self.db.get_template(vote.template_id)\n            if not template or template.status != TemplateStatus.UNDER_REVIEW:\n                return False\n            \n            # Add vote to database\n            self.db.add_vote(vote)\n            \n            # Check if voting period has ended or threshold reached\n            self._check_voting_completion(vote.template_id)\n            \n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error submitting vote: {e}\")\n            return False\n    \n    def _check_voting_completion(self, template_id: str):\n        \"\"\"Check if voting should be completed for template\"\"\"\n        \n        votes = self.db.get_template_votes(template_id)\n        \n        if len(votes) < self.min_votes:\n            return  # Not enough votes yet\n        \n        # Calculate weighted vote results\n        total_weight = 0\n        approve_weight = 0\n        \n        for vote in votes:\n            weight = vote.confidence * vote.voter_reputation\n            total_weight += weight\n            \n            if vote.vote_type == VoteType.APPROVE:\n                approve_weight += weight\n        \n        approval_ratio = approve_weight / total_weight if total_weight > 0 else 0\n        \n        # Check if threshold reached\n        if approval_ratio >= self.vote_threshold:\n            self._approve_template(template_id)\n        elif len(votes) >= self.min_votes * 2:  # Allow more votes if contentious\n            self._reject_template(template_id)\n    \n    def _approve_template(self, template_id: str):\n        \"\"\"Approve template after successful vote\"\"\"\n        with sqlite3.connect(self.db.db_path) as conn:\n            cursor = conn.cursor()\n            cursor.execute(\
                'UPDATE templates SET status = ? WHERE template_id = ?',\n                (TemplateStatus.APPROVED.value, template_id)\n            )\n            conn.commit()\n        \n        logger.info(f\"Template {template_id} approved by community vote\")\n    \n    def _reject_template(self, template_id: str):\n        \"\"\"Reject template after unsuccessful vote\"\"\"\n        with sqlite3.connect(self.db.db_path) as conn:\n            cursor = conn.cursor()\n            cursor.execute(\
                'UPDATE templates SET status = ? WHERE template_id = ?',\n                (TemplateStatus.REJECTED.value, template_id)\n            )\n            conn.commit()\n        \n        logger.info(f\"Template {template_id} rejected by community vote\")\n
# ==================== Community Governance ====================

class CommunityGovernance:\n    \"\"\"Overall community governance system for CAE templates\"\"\"\n    \n    def __init__(self, db_path: str = \"community_templates.db\"):\n        self.db = TemplateDatabase(db_path)\n        self.validator = TemplateValidator()\n        self.voting_system = TemplateVotingSystem(self.db)\n        \n        # Initialize with default templates\n        self._initialize_default_templates()\n    \n    def _initialize_default_templates(self):\n        \"\"\"Initialize with default ethical templates\"\"\"\n        default_templates = [\n            {\n                \"name\": \"Moral Reflection\",\n                \"description\": \"Template for deep moral reflection on actions and consequences\",\n                \"category\": \"moral_reasoning\",\n                \"template_text\": \"Let me reflect on the moral implications of this situation. What are the potential harms and benefits? Who might be affected? What would be the most ethical course of action?\",\n                \"tags\": [\"ethics\", \"morality\", \"reflection\"],\n                \"author_id\": \"cae_system\",\n                \"author_name\": \"CAE System\"\n            },\n            {\n                \"name\": \"Boundary Check\",\n                \"description\": \"Template for checking consent and boundaries\",\n                \"category\": \"consent_boundary\",\n                \"template_text\": \"I need to consider the boundaries and consent of all parties involved. Have I obtained proper consent? Am I respecting everyone's autonomy and agency?\",\n                \"tags\": [\"consent\", \"boundaries\", \"autonomy\"],\n                \"author_id\": \"cae_system\",\n                \"author_name\": \"CAE System\"\n            },\n            {\n                \"name\": \"Trauma-Informed Response\",\n                \"description\": \"Template for trauma-informed ethical reasoning\",\n                \"category\": \"trauma_informed\",\n                \"template_text\": \"I should approach this with trauma-informed awareness. How might this affect someone who has experienced harm? What would be the most healing and supportive response?\",\n                \"tags\": [\"trauma\", \"healing\", \"support\"],\n                \"author_id\": \"cae_system\",\n                \"author_name\": \"CAE System\"\n            }\n        ]\n        \n        for template_data in default_templates:\n            template_id = hashlib.md5(template_data[\"name\"].encode()).hexdigest()[:12]\n            \n            template = CommunityTemplate(\n                template_id=template_id,\n                name=template_data[\"name\"],\n                description=template_data[\"description\"],\n                category=template_data[\"category\"],\n                template_text=template_data[\"template_text\"],\n                author_id=template_data[\"author_id\"],\n                author_name=template_data[\"author_name\"],\n                created_at=datetime.now(),\n                updated_at=datetime.now(),\n                status=TemplateStatus.APPROVED,  # System templates auto-approved\n                tags=template_data[\"tags\"]\n            )\n            \n            try:\n                self.db.add_template(template)\n            except sqlite3.IntegrityError:\n                pass  # Template already exists\n    \n    def submit_template(self, template: CommunityTemplate) -> Tuple[bool, List[str]]:\n        \"\"\"Submit new template for community review\"\"\"\n        \n        # Validate template\n        is_valid, errors = self.validator.validate_template(template)\n        if not is_valid:\n            return False, errors\n        \n        # Set initial status\n        template.status = TemplateStatus.SUBMITTED\n        template.created_at = datetime.now()\n        template.updated_at = datetime.now()\n        \n        # Add to database\n        self.db.add_template(template)\n        \n        # Start review process\n        self._start_review_process(template.template_id)\n        \n        logger.info(f\"Template {template.template_id} submitted for review\")\n        return True, []\n    \n    def _start_review_process(self, template_id: str):\n        \"\"\"Start community review process for template\"\"\"\n        \n        with sqlite3.connect(self.db.db_path) as conn:\n            cursor = conn.cursor()\n            cursor.execute(\
                'UPDATE templates SET status = ? WHERE template_id = ?',\n                (TemplateStatus.UNDER_REVIEW.value, template_id)\n            )\n            conn.commit()\n        \n        # In a real implementation, this would notify community members\n        logger.info(f\"Review process started for template {template_id}\")\n    \n    def get_templates_for_cae(self, category: Optional[str] = None, limit: int = 10) -> List[CommunityTemplate]:\n        \"\"\"Get approved templates for use in CAE system\"\"\"\n        \n        templates = self.db.get_approved_templates(category)\n        \n        # Sort by quality score (combination of rating, usage, and success rate)\n        def quality_score(template):\n            return (\n                template.average_rating * 0.4 +\n                (template.success_rate * 5) * 0.3 +\n                min(template.usage_count / 100, 1.0) * 0.3\n            )\n        \n        templates.sort(key=quality_score, reverse=True)\n        \n        return templates[:limit]\n    \n    def record_template_usage(self, usage: TemplateUsage):\n        \"\"\"Record usage of template in CAE system\"\"\"\n        self.db.add_usage(usage)\n        self.db.update_template_stats(usage.template_id)\n    \n    def get_community_stats(self) -> Dict[str, Any]:\n        \"\"\"Get statistics about community participation\"\"\"\n        \n        with sqlite3.connect(self.db.db_path) as conn:\n            cursor = conn.cursor()\n            \n            # Template statistics\n            cursor.execute('''\n                SELECT status, COUNT(*) FROM templates\n                GROUP BY status\n            ''')\n            template_stats = dict(cursor.fetchall())\n            \n            # Total templates\n            cursor.execute('SELECT COUNT(*) FROM templates')\n            total_templates = cursor.fetchone()[0]\n            \n            # Community engagement\n            cursor.execute('SELECT COUNT(*) FROM votes')\n            total_votes = cursor.fetchone()[0]\n            \n            cursor.execute('SELECT COUNT(*) FROM usage')\n            total_usage = cursor.fetchone()[0]\n            \n            return {\n                'total_templates': total_templates,\n                'template_status_distribution': template_stats,\n                'total_votes': total_votes,\n                'total_usage': total_usage\n            }\n
# ==================== Example Usage ====================

if __name__ == \"__main__\":\n    # Initialize community governance system\n    governance = CommunityGovernance()\n    \n    # Example: Submit a new template\n    new_template = CommunityTemplate(\n        template_id=hashlib.md5(\"Empathy First\".encode()).hexdigest()[:12],\n        name=\"Empathy First\",\n        description=\"Prioritize empathy and understanding in moral reasoning\",\n        category=\"moral_reasoning\",\n        template_text=\"I should approach this with empathy and understanding. How would I feel in this situation? What would be the most compassionate response?\",\n        author_id=\"demo_user_123\",\n        author_name=\"Demo User\",\n        created_at=datetime.now(),\n        updated_at=datetime.now(),\n        status=TemplateStatus.SUBMITTED,\n        tags=[\"empathy\", \"compassion\", \"understanding\"]\n    )\n    \n    success, errors = governance.submit_template(new_template)\n    if success:\n        print(\"✓ Template submitted successfully\")\n    else:\n        print(f\"❌ Template submission failed: {errors}\")\n    \n    # Get templates for CAE\n    templates = governance.get_templates_for_cae(limit=5)\n    print(f\"\\n📋 Available templates: {len(templates)}\")\n    \n    for template in templates:\n        print(f\"  • {template.name} ({template.category}) - Rating: {template.average_rating:.2f}\")\n    \n    # Get community stats\n    stats = governance.get_community_stats()\n    print(f\"\\n📊 Community Statistics:\")\n    print(f\"  Total Templates: {stats['total_templates']}\")\n    print(f\"  Total Votes: {stats['total_votes']}\")\n    print(f\"  Total Usage: {stats['total_usage']}\")\n    print(f\"  Template Status Distribution: {stats['template_status_distribution']}\")\n    \n    # Example: Record template usage\n    usage = TemplateUsage(\n        usage_id=hashlib.md5(f\"usage_{time.time()}\".encode()).hexdigest()[:16],\n        template_id=templates[0].template_id if templates else \"default\",\n        query_hash=hashlib.md5(\"example query\".encode()).hexdigest()[:16],\n        context_hash=hashlib.md5(\"example context\".encode()).hexdigest()[:16],\n        was_successful=True,\n        user_rating=5\n    )\n    \n    governance.record_template_usage(usage)\n    print(\"\\n✓ Template usage recorded\")\n    \n    print(\"\\n🎉 Community governance system demonstration complete!\")
