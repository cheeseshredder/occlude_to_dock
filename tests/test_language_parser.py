"""
Test script for the Language Parser Module

This script tests the language parser without requiring Isaac Sim.
Run with: python test_language_parser.py
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from language.parser import LanguageParser, parse_instruction


def test_parser():
    """Test the language parser with various instructions."""
    
    print("="*70)
    print("LANGUAGE PARSER TEST SUITE")
    print("="*70)
    
    parser = LanguageParser()
    
    # Test cases: (instruction, expected_target, expected_action, expected_relation, expected_landmark)
    test_cases = [
        # Standard instruction
        ("dock at the charger behind the plant", "charger", "dock", "behind", "plant"),
        
        # Different phrasing
        ("go to the charging station near the obstacle", "charger", "navigate", "near", "obstacle"),
        
        # Occlusion language
        ("find the charger occluded by the potted plant", "charger", "find", "occluded_by", "plant"),
        
        # Hidden/blocked variants
        ("navigate to the dock hidden by the red box", "charger", "navigate", "occluded_by", "plant"),
        
        # Different word order
        ("approach the green box next to the gray box", "charger", "approach", "near", "obstacle"),
        
        # Blocked variant
        ("locate the charging dock blocked by the obstacle", "charger", "find", "occluded_by", "obstacle"),
        
        # Simple instruction without relation
        ("dock at the charger", "charger", "dock", None, None),
        
        # Minimal instruction
        ("find the dock", "charger", "find", None, None),
    ]
    
    passed = 0
    failed = 0
    
    for instruction, exp_target, exp_action, exp_relation, exp_landmark in test_cases:
        print(f"\n{'='*70}")
        print(f"Input: \"{instruction}\"")
        print("-"*70)
        
        parsed = parser.parse(instruction)
        
        # Check results
        checks = [
            ("target", parsed.target, exp_target),
            ("action", parsed.action, exp_action),
            ("relation", parsed.relation, exp_relation),
            ("landmark", parsed.landmark, exp_landmark),
        ]
        
        all_passed = True
        for name, actual, expected in checks:
            status = "✅" if actual == expected else "❌"
            if actual != expected:
                all_passed = False
                print(f"  {status} {name}: got '{actual}', expected '{expected}'")
            else:
                print(f"  {status} {name}: '{actual}'")
        
        # Check that detection prompts ALWAYS include all 3 classes
        prompts = parser.get_detection_prompts(parsed)
        required_classes = ['charger', 'plant', 'obstacle']
        for cls in required_classes:
            if cls not in prompts:
                all_passed = False
                print(f"  ❌ Missing required class '{cls}' in detection prompts!")
            else:
                print(f"  ✅ detection_prompts['{cls}']: present")
        
        if all_passed:
            passed += 1
            print(f"\n  ✅ TEST PASSED")
        else:
            failed += 1
            print(f"\n  ❌ TEST FAILED")
        
        # Show additional outputs
        print(f"\n  Success predicate: {parsed.success_predicate}")
        print(f"  Constraints: {parsed.constraints}")
        
        print(f"\n  Detection prompts:")
        for cls, prompt in prompts.items():
            print(f"    {cls}: '{prompt}'")
        
        config = parser.get_planner_config(parsed)
        print(f"\n  Planner config:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed}/{passed+failed} tests passed")
    print("="*70)
    
    return failed == 0


def demo_integration():
    """Demo how the parser integrates with the system."""
    
    print("\n" + "="*70)
    print("INTEGRATION DEMO")
    print("="*70)
    
    # Simulate what happens in batch_test_pomdp.py
    instruction = "dock at the charger behind the plant"
    
    print(f"\n1. User provides instruction: \"{instruction}\"")
    
    # Parse instruction
    parser = LanguageParser()
    parsed = parser.parse(instruction)
    
    print(f"\n2. Parser extracts:")
    print(f"   - Target: {parsed.target}")
    print(f"   - Action: {parsed.action}")
    print(f"   - Relation: {parsed.relation}")
    print(f"   - Landmark: {parsed.landmark}")
    
    # Get detection prompts
    detection_prompts = parser.get_detection_prompts(parsed)
    
    print(f"\n3. Detection prompts generated for Grounding DINO:")
    for cls, prompt in detection_prompts.items():
        print(f"   - {cls}: '{prompt}'")
    
    # Get planner config
    planner_config = parser.get_planner_config(parsed)
    
    print(f"\n4. Planner configuration:")
    print(f"   - confidence_threshold: {planner_config['confidence_threshold']}")
    print(f"   - expect_occlusion: {planner_config['expect_occlusion']}")
    print(f"   - standoff_distance: {planner_config['standoff_distance']}")
    
    print(f"\n5. Success predicate: {parsed.success_predicate}")
    
    print("\n" + "="*70)
    print("This demonstrates the language-to-docking pipeline!")
    print("="*70)


if __name__ == "__main__":
    # Run tests
    success = test_parser()
    
    # Show integration demo
    demo_integration()
    
    # Exit with status
    sys.exit(0 if success else 1)
