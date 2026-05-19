"""
Simple test script to demonstrate FrankaSimulator and SemanticPerceptionConverter
without needing ROS running. Use this to test before integrating with ROS node.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import llm_planner modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_franka_semantic_sim import FrankaSimulator, SemanticPerceptionConverter
import yaml


def test_franka_simulator():
    """Test basic simulator functionality"""
    print("=" * 60)
    print("Testing Franka Simulator")
    print("=" * 60)
    
    # Initialize simulator
    sim = FrankaSimulator()
    converter = SemanticPerceptionConverter()
    
    # Load world objects
    world_objects = {
        'mug': {
            'location': 'table',
            # 'type': 'cup',
            # 'subparts': ['body', 'handle']
        },
        'plate': {
            'location': 'table',
            # 'type': 'container',
            # 'subparts': ['body']
        },
        'bottle': {
            'location': 'shelf',
            # 'type': 'container',
            # 'subparts': ['body', 'cap']
        }
    }
    
    sim.set_world_objects(world_objects)
    print(f"\n✓ Loaded objects: {list(world_objects.keys())}")
    
    # Initial state
    print("\n--- Initial State ---")
    state = sim.get_state()
    perception = converter.state_to_perception_dict(state, {
        'objects': sim.world_objects,
        'visible_objects': sim.visible_objects,
    })
    print(f"Location: {perception['location']}")
    print(f"Visible objects: {list(perception['observed_obj'].keys())}")
    print(f"Grasped objects: {perception['grasped_obj']}")
    
    # Test 1: Grasp object
    print("\n--- Test 1: Grasp mug ---")
    success = sim.grasp_object('mug')
    print(f"✓ Grasped: {success}")
    state = sim.get_state()
    perception = converter.state_to_perception_dict(state, {
        'objects': sim.world_objects,
        'visible_objects': sim.visible_objects,
    })
    print(f"Grasped objects: {perception['grasped_obj']}")
    print(f"Visible objects: {list(perception['observed_obj'].keys())}")
    
    # Test 2: Move to location
    print("\n--- Test 2: Move to shelf ---")
    success = sim.move_to_location('shelf')
    print(f"✓ Moved: {success}")
    state = sim.get_state()
    perception = converter.state_to_perception_dict(state, {
        'objects': sim.world_objects,
        'visible_objects': sim.visible_objects,
    })
    print(f"Location: {perception['location']}")
    print(f"Visible objects at shelf: {list(perception['observed_obj'].keys())}")
    
    # Test 3: Place object
    print("\n--- Test 3: Place mug on shelf ---")
    success = sim.place_object('shelf')
    print(f"✓ Placed: {success}")
    state = sim.get_state()
    perception = converter.state_to_perception_dict(state, {
        'objects': sim.world_objects,
        'visible_objects': sim.visible_objects,
    })
    print(f"Grasped objects: {perception['grasped_obj']}")
    print(f"Visible objects: {list(perception['observed_obj'].keys())}")
    print(f"Mug location: {sim.world_objects['mug']['location']}")
    
    # Test 4: Move back and grasp bottle
    print("\n--- Test 4: Move back to table and grasp bottle ---")
    sim.move_to_location('table')
    success = sim.grasp_object('bottle')
    print(f"✓ Grasped bottle by cap: {success}")
    state = sim.get_state()
    perception = converter.state_to_perception_dict(state, {
        'objects': sim.world_objects,
        'visible_objects': sim.visible_objects,
    })
    print(f"Grasped: {perception['grasped_obj']}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    # Show full perception dict
    print("\nFinal Perception Dict (YAML format):")
    print(yaml.dump(perception, default_flow_style=False))


if __name__ == "__main__":
    test_franka_simulator()
