"""
Language Parser Module for Natural Language Instructions

Parses instructions like:
- "dock at the charger behind the plant"
- "go to the charger near the obstacle"
- "find the charging station occluded by the plant"

Extracts:
- target: The object to reach (e.g., "charger")
- action: What to do (e.g., "dock", "go", "find")
- relation: Spatial relationship (e.g., "behind", "near", "occluded by")
- landmark: Reference object (e.g., "plant", "obstacle")
- constraints: Approach parameters (standoff distance, heading)

This is a rule-based parser suitable for the controlled vocabulary
of the docking task. For more complex language, consider SpaCy or LLM.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ParsedInstruction:
    """Structured representation of a parsed instruction."""
    target: str  # What to find (charger, dock, station)
    action: str  # What to do (dock, go, find, navigate)
    relation: Optional[str]  # Spatial relation (behind, near, occluded)
    landmark: Optional[str]  # Reference object (plant, obstacle)
    success_predicate: str  # Condition for success
    constraints: Dict = field(default_factory=dict)  # Approach parameters
    raw_instruction: str = ""  # Original instruction
    
    def __repr__(self):
        return (f"ParsedInstruction(\n"
                f"  target='{self.target}',\n"
                f"  action='{self.action}',\n"
                f"  relation='{self.relation}',\n"
                f"  landmark='{self.landmark}',\n"
                f"  success_predicate='{self.success_predicate}',\n"
                f"  constraints={self.constraints}\n"
                f")")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'target': self.target,
            'action': self.action,
            'relation': self.relation,
            'landmark': self.landmark,
            'success_predicate': self.success_predicate,
            'constraints': self.constraints,
            'raw_instruction': self.raw_instruction,
        }


class LanguageParser:
    """
    Rule-based parser for docking instructions.
    
    Vocabulary:
    - Targets: charger, charging station, dock, charging dock, goal
    - Actions: dock, go, navigate, find, reach, approach
    - Relations: behind, near, next to, beside, occluded by, hidden by, blocked by
    - Landmarks: plant, potted plant, obstacle, box, object
    """
    
    # Target synonyms → canonical name
    TARGET_SYNONYMS = {
        'charger': 'charger',
        'charging station': 'charger',
        'charging dock': 'charger',
        'dock': 'charger',
        'docking station': 'charger',
        'charge station': 'charger',
        'goal': 'charger',
        'target': 'charger',
        'green box': 'charger',
        'green cube': 'charger',
    }
    
    # Action synonyms → canonical name
    ACTION_SYNONYMS = {
        'dock': 'dock',
        'dock at': 'dock',
        'go to': 'navigate',
        'go': 'navigate',
        'navigate to': 'navigate',
        'navigate': 'navigate',
        'find': 'find',
        'locate': 'find',
        'reach': 'navigate',
        'approach': 'approach',
        'get to': 'navigate',
        'move to': 'navigate',
    }
    
    # Landmark synonyms → canonical name
    LANDMARK_SYNONYMS = {
        'plant': 'plant',
        'potted plant': 'plant',
        'pot': 'plant',
        'red box': 'plant',
        'red cube': 'plant',
        'red object': 'plant',
        'obstacle': 'obstacle',
        'box': 'obstacle',
        'gray box': 'obstacle',
        'grey box': 'obstacle',
        'object': 'obstacle',
        'obstruction': 'obstacle',
        'blocker': 'obstacle',
    }
    
    # Relation synonyms → canonical name + semantic meaning
    RELATION_SYNONYMS = {
        'behind': ('behind', 'occluded'),
        'in back of': ('behind', 'occluded'),
        'past': ('behind', 'occluded'),
        'beyond': ('behind', 'occluded'),
        'on the other side of': ('behind', 'occluded'),
        'occluded by': ('occluded_by', 'occluded'),
        'hidden by': ('occluded_by', 'occluded'),
        'blocked by': ('occluded_by', 'occluded'),
        'obscured by': ('occluded_by', 'occluded'),
        'near': ('near', 'proximity'),
        'next to': ('near', 'proximity'),
        'beside': ('near', 'proximity'),
        'close to': ('near', 'proximity'),
        'by': ('near', 'proximity'),
        'in front of': ('in_front', 'visible'),
        'before': ('in_front', 'visible'),
    }
    
    # Default constraints for different actions
    DEFAULT_CONSTRAINTS = {
        'dock': {
            'standoff_distance': 0.25,  # How close to get before final alignment
            'approach_heading': 'front',  # Approach from front
            'final_alignment': True,  # Require precise alignment
            'max_approach_speed': 0.08,  # Slow approach for safety
        },
        'navigate': {
            'standoff_distance': 0.5,
            'approach_heading': 'any',
            'final_alignment': False,
            'max_approach_speed': 0.15,
        },
        'find': {
            'standoff_distance': 1.0,
            'approach_heading': 'any',
            'final_alignment': False,
            'max_approach_speed': 0.15,
        },
        'approach': {
            'standoff_distance': 0.4,
            'approach_heading': 'front',
            'final_alignment': False,
            'max_approach_speed': 0.10,
        },
    }
    
    def __init__(self):
        """Initialize the parser with compiled regex patterns."""
        # Build regex patterns for efficient matching
        self._build_patterns()
    
    def _build_patterns(self):
        """Compile regex patterns for parsing."""
        # Pattern for extracting relation + landmark
        # Matches: "behind the plant", "occluded by the obstacle", etc.
        relation_words = '|'.join(re.escape(r) for r in sorted(self.RELATION_SYNONYMS.keys(), key=len, reverse=True))
        landmark_words = '|'.join(re.escape(l) for l in sorted(self.LANDMARK_SYNONYMS.keys(), key=len, reverse=True))
        
        self.relation_pattern = re.compile(
            rf'({relation_words})\s+(?:the\s+)?({landmark_words})',
            re.IGNORECASE
        )
        
        # Pattern for extracting target
        target_words = '|'.join(re.escape(t) for t in sorted(self.TARGET_SYNONYMS.keys(), key=len, reverse=True))
        self.target_pattern = re.compile(
            rf'(?:the\s+)?({target_words})',
            re.IGNORECASE
        )
        
        # Pattern for extracting action (sorted by length to match longer phrases first)
        action_words = '|'.join(re.escape(a) for a in sorted(self.ACTION_SYNONYMS.keys(), key=len, reverse=True))
        self.action_pattern = re.compile(
            rf'^({action_words})',
            re.IGNORECASE
        )
    
    def parse(self, instruction: str) -> ParsedInstruction:
        """
        Parse a natural language instruction into structured form.
        
        Args:
            instruction: Natural language instruction string
            
        Returns:
            ParsedInstruction with extracted components
        """
        original = instruction
        instruction = instruction.strip().lower()
        
        # Extract action (verb at start)
        action = self._extract_action(instruction)
        
        # Extract target (what to find)
        target = self._extract_target(instruction)
        
        # Extract relation and landmark
        relation, landmark = self._extract_relation_landmark(instruction)
        
        # Determine success predicate based on action
        success_predicate = self._get_success_predicate(action, target)
        
        # Get constraints based on action and relation
        constraints = self._get_constraints(action, relation)
        
        return ParsedInstruction(
            target=target,
            action=action,
            relation=relation,
            landmark=landmark,
            success_predicate=success_predicate,
            constraints=constraints,
            raw_instruction=original,
        )
    
    def _extract_action(self, instruction: str) -> str:
        """Extract the action verb from instruction."""
        match = self.action_pattern.search(instruction)
        if match:
            action_text = match.group(1).lower()
            # Find canonical action
            for synonym, canonical in self.ACTION_SYNONYMS.items():
                if action_text == synonym.lower():
                    return canonical
        
        # Default action based on common patterns
        if 'dock' in instruction:
            return 'dock'
        elif 'find' in instruction or 'locate' in instruction:
            return 'find'
        
        return 'navigate'
    
    def _extract_target(self, instruction: str) -> str:
        """Extract the target object from instruction."""
        match = self.target_pattern.search(instruction)
        if match:
            target_text = match.group(1).lower()
            # Find canonical target
            for synonym, canonical in self.TARGET_SYNONYMS.items():
                if target_text == synonym.lower():
                    return canonical
        
        # Default target
        return 'charger'
    
    def _extract_relation_landmark(self, instruction: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract spatial relation and landmark from instruction."""
        match = self.relation_pattern.search(instruction)
        if match:
            relation_text = match.group(1).lower()
            landmark_text = match.group(2).lower()
            
            # Find canonical relation
            relation = None
            for synonym, (canonical, _) in self.RELATION_SYNONYMS.items():
                if relation_text == synonym.lower():
                    relation = canonical
                    break
            
            # Find canonical landmark
            landmark = None
            for synonym, canonical in self.LANDMARK_SYNONYMS.items():
                if landmark_text == synonym.lower():
                    landmark = canonical
                    break
            
            return relation, landmark
        
        return None, None
    
    def _get_success_predicate(self, action: str, target: str) -> str:
        """Determine the success condition based on action and target."""
        if action == 'dock':
            return f'robot_docked_at_{target}'
        elif action == 'find':
            return f'{target}_visible_and_localized'
        else:
            return f'robot_near_{target}'
    
    def _get_constraints(self, action: str, relation: Optional[str]) -> Dict:
        """Get approach constraints based on action and relation."""
        # Start with default constraints for action
        constraints = self.DEFAULT_CONSTRAINTS.get(action, self.DEFAULT_CONSTRAINTS['navigate']).copy()
        
        # Modify based on relation (occlusion requires more careful approach)
        if relation in ['behind', 'occluded_by']:
            constraints['expect_occlusion'] = True
            constraints['require_exploration'] = True
            constraints['min_confidence_to_approach'] = 0.6
        else:
            constraints['expect_occlusion'] = False
            constraints['require_exploration'] = False
            constraints['min_confidence_to_approach'] = 0.4
        
        return constraints
    
    def get_detection_prompts(self, parsed: ParsedInstruction) -> Dict[str, str]:
        """
        Generate detection prompts for Grounding DINO based on parsed instruction.
        
        IMPORTANT: Always includes all three classes (charger, plant, obstacle)
        for safety - even if not mentioned in the instruction, we need to detect
        obstacles for collision avoidance.
        
        Args:
            parsed: ParsedInstruction from parse()
            
        Returns:
            Dictionary mapping class names to detection prompts
        """
        prompts = {}
        
        # Target prompt (charger) - always included
        if parsed.target == 'charger':
            prompts['charger'] = 'green box . green cube . green block . charging station'
        else:
            prompts['charger'] = f'{parsed.target} . green box . charging station'
        
        # ALWAYS include plant prompt for collision avoidance
        prompts['plant'] = 'red box . red cube . red block . potted plant'
        
        # ALWAYS include obstacle prompt for safety
        prompts['obstacle'] = 'gray box . gray cube . gray block'
        
        return prompts
    
    def get_planner_config(self, parsed: ParsedInstruction) -> Dict:
        """
        Generate planner configuration based on parsed instruction.
        
        Args:
            parsed: ParsedInstruction from parse()
            
        Returns:
            Dictionary with planner parameters
        """
        config = {
            'target_class': parsed.target,
            'landmark_class': parsed.landmark,
            'relation_type': parsed.relation,
            'success_predicate': parsed.success_predicate,
            'confidence_threshold': parsed.constraints.get('min_confidence_to_approach', 0.4),
            'expect_occlusion': parsed.constraints.get('expect_occlusion', False),
            'standoff_distance': parsed.constraints.get('standoff_distance', 0.3),
        }
        
        return config


# Convenience function for quick parsing
def parse_instruction(instruction: str) -> ParsedInstruction:
    """Parse an instruction string into structured form."""
    parser = LanguageParser()
    return parser.parse(instruction)


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Language Parser Module - Test Suite")
    print("="*60)
    
    parser = LanguageParser()
    
    # Test instructions
    test_instructions = [
        "dock at the charger behind the plant",
        "go to the charging station near the obstacle",
        "find the charger occluded by the potted plant",
        "navigate to the dock hidden by the red box",
        "approach the green box next to the gray box",
        "locate the charging dock blocked by the obstacle",
    ]
    
    for instruction in test_instructions:
        print(f"\n{'='*60}")
        print(f"Input: \"{instruction}\"")
        print("-"*60)
        
        parsed = parser.parse(instruction)
        print(parsed)
        
        print("\nDetection Prompts:")
        prompts = parser.get_detection_prompts(parsed)
        for cls, prompt in prompts.items():
            print(f"  {cls}: '{prompt}'")
        
        print("\nPlanner Config:")
        config = parser.get_planner_config(parsed)
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
