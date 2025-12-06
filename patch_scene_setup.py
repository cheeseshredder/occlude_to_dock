"""
EMERGENCY PATCH for setup_environment.py
Fixes the size parameter type issue that causes crashes in scene 9+

Run this BEFORE collecting data:
    python patch_scene_setup.py
"""

import os

setup_file = "scene/setup_environment.py"

print("=" * 60)
print("PATCHING setup_environment.py")
print("=" * 60)

# Read the file
with open(setup_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Count occurrences of the problem
problem_count = content.count('size=np.array([')

print(f"\nFound {problem_count} instances of 'size=np.array(['")

if problem_count == 0:
    print("✓ File is already patched!")
    exit(0)

print("\n⚠ File needs patching...")

# Fix: Replace size=np.array(...) with scale=np.array(...) and size=1.0

# Fix walls
content = content.replace(
    'size=np.array([room_size[0], wall_thickness, wall_height])',
    'scale=np.array([room_size[0]/2, wall_thickness/2, wall_height/2]),\n                size=1.0,'
)

content = content.replace(
    'size=np.array([wall_thickness, room_size[1], wall_height])',
    'scale=np.array([wall_thickness/2, room_size[1]/2, wall_height/2]),\n                size=1.0,'
)

# Fix charger
content = content.replace(
    'size=np.array(charger_cfg[\'size\'])',
    'scale=np.array([charger_size[0]/2, charger_size[1]/2, charger_size[2]/2]),\n                size=1.0,'
)

# Add charger_size variable if not present
if 'charger_size = charger_cfg' not in content:
    content = content.replace(
        '# 1. Charger (target object)\n        charger_cfg = objects_config[\'charger\']',
        '# 1. Charger (target object)\n        charger_cfg = objects_config[\'charger\']\n        charger_size = charger_cfg[\'size\']'
    )

# Fix distractors
content = content.replace(
    'size=np.array(dist_cfg.get(\'size\', [0.35, 0.25, 0.45]))',
    'scale=np.array([dist_size[0]/2, dist_size[1]/2, dist_size[2]/2]),\n                    size=1.0,'
)

# Add dist_size variable if not present
if 'dist_size = dist_cfg' not in content:
    content = content.replace(
        'for i, distractor_key in enumerate([\'distractor_1\', \'distractor_2\']):\n            dist_cfg = objects_config[distractor_key]',
        'for i, distractor_key in enumerate([\'distractor_1\', \'distractor_2\']):\n            dist_cfg = objects_config[distractor_key]\n            dist_size = dist_cfg.get(\'size\', [0.35, 0.25, 0.45])'
    )

# Fix robot fallback
content = content.replace(
    'size=np.array([0.3, 0.3, 0.2])',
    'scale=np.array([0.105, 0.08, 0.088]),\n                    size=1.0,'
)

# Write back
with open(setup_file, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
with open(setup_file, 'r', encoding='utf-8') as f:
    new_content = f.read()

remaining = new_content.count('size=np.array([')

print(f"\n✓ Patched! Remaining issues: {remaining}")

if remaining == 0:
    print("✅ All size=np.array() instances fixed!")
    print("\nYou can now run:")
    print("  python calibration/collect_dataset.py")
else:
    print(f"⚠ Warning: {remaining} instances still remain")
    print("Manual fix may be needed")

print("=" * 60)
