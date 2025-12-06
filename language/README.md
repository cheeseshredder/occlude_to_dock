# Language Module for Occlude-to-Dock

## Overview

This module provides natural language instruction parsing for the language-guided docking task. It converts human-readable instructions into structured configurations for the perception and planning modules.

## Pipeline Flow

```
User Instruction
       ↓
┌──────────────────────────────────────────────────────────────┐
│  "dock at the charger behind the plant"                      │
└──────────────────────────────────────────────────────────────┘
       ↓
   Language Parser
       ↓
┌──────────────────────────────────────────────────────────────┐
│  ParsedInstruction:                                          │
│    target: "charger"                                         │
│    action: "dock"                                            │
│    relation: "behind"                                        │
│    landmark: "plant"                                         │
│    success_predicate: "robot_docked_at_charger"              │
│    constraints:                                              │
│      - standoff_distance: 0.25m                              │
│      - expect_occlusion: True                                │
│      - min_confidence_to_approach: 0.6                       │
└──────────────────────────────────────────────────────────────┘
       ↓
   ┌───────────────────┬───────────────────┐
   ↓                   ↓                   ↓
Detection Prompts   Planner Config    Success Predicate
   ↓                   ↓                   ↓
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Grounding   │   │   POMDP     │   │ Termination │
│   DINO      │   │  Planner    │   │   Check     │
└─────────────┘   └─────────────┘   └─────────────┘
```

## Supported Vocabulary

### Targets (What to find)
- `charger`, `charging station`, `dock`, `docking station`, `goal`
- `green box`, `green cube` (visual synonyms)

### Actions (What to do)
- `dock` → Full docking with precise alignment
- `navigate`, `go to`, `reach` → Navigate near target
- `find`, `locate` → Locate and confirm target
- `approach` → Get close but don't dock

### Relations (Spatial)
- `behind`, `past`, `beyond` → Target is occluded
- `occluded by`, `hidden by`, `blocked by` → Explicit occlusion
- `near`, `next to`, `beside` → Proximity relationship
- `in front of` → Target is visible

### Landmarks (Reference objects)
- `plant`, `potted plant`, `red box` → Plant obstacle
- `obstacle`, `box`, `gray box` → Generic obstacle

## Usage

### Basic Parsing

```python
from language.parser import LanguageParser, parse_instruction

# Quick parse
parsed = parse_instruction("dock at the charger behind the plant")

# Or with explicit parser instance
parser = LanguageParser()
parsed = parser.parse("dock at the charger behind the plant")

print(parsed.target)      # "charger"
print(parsed.action)      # "dock"
print(parsed.relation)    # "behind"
print(parsed.landmark)    # "plant"
```

### Getting Detection Prompts

```python
parser = LanguageParser()
parsed = parser.parse("dock at the charger behind the plant")
prompts = parser.get_detection_prompts(parsed)

# Returns:
# {
#     'charger': 'green box . green cube . green block . charging station',
#     'plant': 'red box . red cube . red block . potted plant'
# }
```

### Getting Planner Configuration

```python
parser = LanguageParser()
parsed = parser.parse("dock at the charger behind the plant")
config = parser.get_planner_config(parsed)

# Returns:
# {
#     'target_class': 'charger',
#     'landmark_class': 'plant',
#     'relation_type': 'behind',
#     'success_predicate': 'robot_docked_at_charger',
#     'confidence_threshold': 0.6,
#     'expect_occlusion': True,
#     'standoff_distance': 0.25
# }
```

## Command Line Usage

```bash
# Run batch tests with custom instruction
isaac-python.bat batch_test_pomdp.py --trials 5 --instruction "dock at the charger behind the plant"

# Different instructions
isaac-python.bat batch_test_pomdp.py --trials 5 --instruction "find the charging station near the obstacle"
```

## Test the Parser

```bash
# Run parser tests (no Isaac Sim required)
python test_language_parser.py
```

## Design Decisions

1. **Rule-based parsing**: Simple, deterministic, fast - no external dependencies
2. **Canonical forms**: All synonyms map to canonical names for consistent downstream processing
3. **Constraint propagation**: Occlusion relations automatically set higher confidence thresholds
4. **Extensible vocabulary**: Easy to add new synonyms by updating the dictionaries

## Files

- `language/parser.py` - Main parser implementation
- `language/__init__.py` - Module exports
- `test_language_parser.py` - Test suite (no Isaac Sim required)

## Integration Points

1. **Perception** (`perception/vision_model.py`): Uses detection prompts
2. **Planner** (`planner/pomdp_planner.py`): Uses confidence threshold and occlusion flags
3. **Batch Test** (`batch_test_pomdp.py`): Parses instruction and configures modules
