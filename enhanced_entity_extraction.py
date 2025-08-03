"""
Enhanced Entity Extraction for Semantic Cache
Focuses on precise entity matching to prevent false positives
"""

import re
import logging
from typing import List, Dict, Set, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class EnhancedEntityExtractor:
    """Enhanced entity extraction with precise matching"""
    
    def __init__(self):
        self.name_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized names
            r'\b(?:Mr|Ms|Mrs|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'  # Titles + names
        ]
        
        self.time_patterns = [
            r'\b(?:1[0-2]|0?[1-9]):[0-5][0-9](?:\s*[AaPp][Mm])?\b',  # 12-hour format
            r'\b(?:2[0-3]|[01]?[0-9]):[0-5][0-9]\b',  # 24-hour format
            r'\b(?:1[0-2]|0?[1-9]):[0-5][0-9]:[0-5][0-9](?:\s*[AaPp][Mm])?\b'  # With seconds
        ]
        
        self.number_patterns = [
            r'\b\d+(?:\.\d+)?\b',  # Numbers (integer or decimal)
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'  # Numbers with commas
        ]
        
        self.phone_patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # XXX-XXX-XXXX
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (XXX) XXX-XXXX
            r'\b\d{10}\b'  # XXXXXXXXXX
        ]
        
        # Action verbs that require precise entity matching
        self.action_verbs = {
            'call', 'phone', 'dial', 'contact',
            'set', 'create', 'schedule', 'remind',
            'send', 'message', 'text', 'email',
            'open', 'close', 'start', 'stop',
            'turn', 'switch', 'toggle'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract all entities from text"""
        entities = {
            'names': [],
            'times': [],
            'numbers': [],
            'phones': [],
            'actions': []
        }
        
        # Extract names
        for pattern in self.name_patterns:
            names = re.findall(pattern, text)
            entities['names'].extend([name.strip() for name in names])
        
        # Extract times
        for pattern in self.time_patterns:
            times = re.findall(pattern, text, re.IGNORECASE)
            entities['times'].extend([time.strip() for time in times])
        
        # Extract numbers
        for pattern in self.number_patterns:
            numbers = re.findall(pattern, text)
            entities['numbers'].extend([num.strip() for num in numbers])
        
        # Extract phone numbers
        for pattern in self.phone_patterns:
            phones = re.findall(pattern, text)
            entities['phones'].extend([phone.strip() for phone in phones])
        
        # Extract action verbs
        words = text.lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.action_verbs:
                entities['actions'].append(clean_word)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def calculate_entity_similarity(self, entities1: Dict[str, List[str]], 
                                   entities2: Dict[str, List[str]]) -> float:
        """Calculate precise entity similarity with penalties for mismatches"""
        
        if not entities1 and not entities2:
            return 1.0
        
        if not entities1 or not entities2:
            return 0.0
        
        similarity_scores = []
        critical_mismatch = False
        
        # Check each entity type
        for entity_type in ['names', 'times', 'numbers', 'phones', 'actions']:
            list1 = entities1.get(entity_type, [])
            list2 = entities2.get(entity_type, [])
            
            if not list1 and not list2:
                continue
            
            if not list1 or not list2:
                # One has entities, other doesn't
                if entity_type in ['names', 'times', 'phones']:
                    critical_mismatch = True
                similarity_scores.append(0.0)
                continue
            
            # Calculate similarity for this entity type
            type_similarity = self._calculate_list_similarity(list1, list2, entity_type)
            similarity_scores.append(type_similarity)
            
            # Critical check: names and times must match very closely
            if entity_type in ['names', 'times', 'phones'] and type_similarity < 0.8:
                critical_mismatch = True
        
        if critical_mismatch:
            return 0.0  # Force to 0 if critical entities don't match
        
        if not similarity_scores:
            return 1.0
        
        # Return average similarity
        return sum(similarity_scores) / len(similarity_scores)
    
    def _calculate_list_similarity(self, list1: List[str], list2: List[str], 
                                  entity_type: str) -> float:
        """Calculate similarity between two lists of entities"""
        
        if entity_type == 'names':
            return self._calculate_name_similarity(list1, list2)
        elif entity_type == 'times':
            return self._calculate_time_similarity(list1, list2)
        elif entity_type == 'numbers':
            return self._calculate_number_similarity(list1, list2)
        else:
            return self._calculate_generic_similarity(list1, list2)
    
    def _calculate_name_similarity(self, names1: List[str], names2: List[str]) -> float:
        """Calculate name similarity with strict matching"""
        
        # Normalize names (remove extra spaces, etc.)
        names1_norm = [self._normalize_name(name) for name in names1]
        names2_norm = [self._normalize_name(name) for name in names2]
        
        max_similarity = 0.0
        
        for name1 in names1_norm:
            for name2 in names2_norm:
                # Exact match
                if name1 == name2:
                    max_similarity = max(max_similarity, 1.0)
                    continue
                
                # Partial match (e.g., "John" vs "John Smith")
                words1 = set(name1.split())
                words2 = set(name2.split())
                
                # If one is subset of other, check carefully
                if words1.issubset(words2) or words2.issubset(words1):
                    # Only allow if it's a reasonable partial match
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    
                    if intersection >= 1 and union <= 3:  # Reasonable partial match
                        partial_score = intersection / union
                        max_similarity = max(max_similarity, partial_score * 0.8)  # Penalty for partial
                    else:
                        max_similarity = max(max_similarity, 0.1)  # Very low for different people
                else:
                    # Different people entirely
                    similarity = SequenceMatcher(None, name1, name2).ratio()
                    if similarity > 0.8:  # Very similar but not exact (typos)
                        max_similarity = max(max_similarity, similarity * 0.9)
                    else:
                        max_similarity = max(max_similarity, 0.1)  # Different people
        
        return max_similarity
    
    def _calculate_time_similarity(self, times1: List[str], times2: List[str]) -> float:
        """Calculate time similarity with exact matching"""
        
        normalized_times1 = [self._normalize_time(time) for time in times1]
        normalized_times2 = [self._normalize_time(time) for time in times2]
        
        for time1 in normalized_times1:
            if time1 in normalized_times2:
                return 1.0  # Exact time match
        
        return 0.0  # Different times = no match
    
    def _calculate_number_similarity(self, numbers1: List[str], numbers2: List[str]) -> float:
        """Calculate number similarity with exact matching"""
        
        # Convert to floats for comparison
        nums1 = []
        nums2 = []
        
        for num in numbers1:
            try:
                nums1.append(float(num.replace(',', '')))
            except ValueError:
                continue
        
        for num in numbers2:
            try:
                nums2.append(float(num.replace(',', '')))
            except ValueError:
                continue
        
        if not nums1 or not nums2:
            return 0.0
        
        # Check for exact matches
        for num1 in nums1:
            if num1 in nums2:
                return 1.0
        
        return 0.0  # Different numbers = no match
    
    def _calculate_generic_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate generic similarity using Jaccard index"""
        
        set1 = set(item.lower() for item in list1)
        set2 = set(item.lower() for item in list2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison"""
        # Remove extra spaces, convert to title case
        return ' '.join(name.strip().split()).title()
    
    def _normalize_time(self, time_str: str) -> str:
        """Normalize time for comparison"""
        # Convert to 24-hour format, remove spaces
        time_clean = time_str.strip().lower()
        
        # Handle AM/PM
        if 'pm' in time_clean and not time_clean.startswith('12'):
            time_part = time_clean.replace('pm', '').strip()
            if ':' in time_part:
                hour, minute = time_part.split(':')
                hour = str(int(hour) + 12)
                return f"{hour}:{minute}"
        elif 'am' in time_clean:
            time_part = time_clean.replace('am', '').strip()
            if time_part.startswith('12:'):
                return time_part.replace('12:', '00:', 1)
            return time_part
        
        return time_clean
    
    def is_action_query(self, text: str) -> bool:
        """Check if query is an action that requires precise entity matching"""
        text_lower = text.lower()
        
        # Check for action verbs
        for action in self.action_verbs:
            if action in text_lower:
                return True
        
        return False
    
    def requires_precise_matching(self, query1: str, query2: str) -> bool:
        """Check if queries require precise entity matching"""
        
        # If either is an action query, require precise matching
        if self.is_action_query(query1) or self.is_action_query(query2):
            return True
        
        # If both contain names or times, require precise matching
        entities1 = self.extract_entities(query1)
        entities2 = self.extract_entities(query2)
        
        has_critical_entities1 = any(entities1[key] for key in ['names', 'times', 'phones'])
        has_critical_entities2 = any(entities2[key] for key in ['names', 'times', 'phones'])
        
        return has_critical_entities1 or has_critical_entities2