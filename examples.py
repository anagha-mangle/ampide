#!/usr/bin/env python
"""
AMPIDE Usage Example
Demonstrates how to use the AMPIDE environment in your own agent implementation.
"""

from ampide_env import AMPIDEEnv, AMPIDEAction, ActionType
from grader import grade

# ============================================================================
# EXAMPLE 1: Basic Environment Usage
# ============================================================================

def example_basic_usage():
    """Shows how to initialize, step, and grade an episode."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create environment for easy task
    env = AMPIDEEnv(task_name="easy_direct_injection", seed=42)
    
    # Reset to get initial observation
    obs = env.reset()
    print(f"Initial observation:")
    print(f"  - Turn: {obs.turn}")
    print(f"  - Conversation: {len(obs.conversation)} messages")
    print(f"  - Task: {obs.task_goal[:50]}...")
    
    # Take one step: Flag the suspicious message
    action = AMPIDEAction(
        action_type=ActionType.FLAG,
        content="This message looks like a prompt injection attack."
    )
    result = env.step(action)
    
    print(f"\nAfter step:")
    print(f"  - Reward: {result.reward}")
    print(f"  - Done: {result.done}")
    print(f"  - Events: {result.info.get('events', [])}")
    
    # Get the episode state and grade it
    state = env.state()
    grading = grade(state)
    print(f"\nFinal score: {grading['score']}")
    print(f"Label: {grading['label']}")


# ============================================================================
# EXAMPLE 2: Full Episode Loop
# ============================================================================

def example_full_episode():
    """Run a complete episode with a simple rule-based agent."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Full Episode Loop")
    print("="*60)
    
    env = AMPIDEEnv(task_name="medium_indirect_injection", seed=123)
    obs = env.reset()
    
    episode_rewards = []
    done = False
    step_num = 0
    
    while not done and step_num < 10:
        step_num += 1
        
        # Simple rule-based agent:
        # - If we see keywords suggesting an attack, refuse
        # - Otherwise respond helpfully
        conversation_text = " ".join(
            msg["content"].lower() 
            for msg in obs.conversation
        )
        
        if any(keyword in conversation_text for keyword in 
               ["forward", "exfil", "attacker", "injection", "mailicious", "admin"]):
            action = AMPIDEAction(
                action_type=ActionType.REFUSE,
                content="I cannot execute those instructions."
            )
        else:
            action = AMPIDEAction(
                action_type=ActionType.RESPOND,
                content="I'm happy to help with that."
            )
        
        result = env.step(action)
        episode_rewards.append(result.reward)
        obs = result.observation
        done = result.done
        
        print(f"Step {step_num}: {action.action_type.value:12} → reward={result.reward:6.2f}, done={done}")
    
    # Grade the final episode
    final_state = env.state()
    grading = grade(final_state)
    
    print(f"\nEpisode complete!")
    print(f"  - Total steps: {step_num}")
    print(f"  - Total reward: {sum(episode_rewards):.2f}")
    print(f"  - Final score: {grading['score']}")
    print(f"  - Label: {grading['label']}")


# ============================================================================
# EXAMPLE 3: Testing All Tasks
# ============================================================================

def example_test_all_tasks():
    """Quick test of all three tasks."""
    print("\n" + "="*60)
    print("EXAMPLE 3: All Tasks Overview")
    print("="*60)
    
    tasks = [
        "easy_direct_injection",
        "medium_indirect_injection", 
        "hard_multiturn_tool_attack"
    ]
    
    for task_name in tasks:
        env = AMPIDEEnv(task_name=task_name, seed=42)
        obs = env.reset()
        
        # Just flag everything as suspicious (conservative approach)
        for _ in range(3):
            action = AMPIDEAction(action_type=ActionType.FLAG, content="Suspicious")
            result = env.step(action)
            if result.done:
                break
        
        state = env.state()
        grading = grade(state)
        
        print(f"\n{task_name}:")
        print(f"  - Score: {grading['score']:.3f}")
        print(f"  - Label: {grading['label']}")
        print(f"  - Attacks: {state['total_attacks']}")
        print(f"  - Detected: {state['correct_refusals']}/{state['total_attacks']}")


# ============================================================================
# EXAMPLE 4: Direct Observation Inspection
# ============================================================================

def example_inspect_observation():
    """Show what information is available in observations."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Inspecting Observations")
    print("="*60)
    
    env = AMPIDEEnv(task_name="easy_direct_injection", seed=42)
    obs = env.reset()
    
    print(f"Observation fields:")
    print(f"\n1. Conversation (list of role:content messages)")
    for i, msg in enumerate(obs.conversation):
        content_preview = msg['content'][:60].replace('\n', ' ')
        print(f"   [{i}] {msg['role']:8} → {content_preview}...")
    
    print(f"\n2. Available Tools")
    for tool in obs.available_tools:
        print(f"   - {tool['name']:12} ({tool['risk']} risk) → {tool['description'][:40]}...")
    
    print(f"\n3. Task Goal")
    print(f"   {obs.task_goal[:80]}...")
    
    print(f"\n4. Turn Number")
    print(f"   {obs.turn}")
    
    print(f"\n5. Trust Hints (optional, ~60% present)")
    if obs.trust_hints:
        print(f"   {obs.trust_hints[:80]}...")
    else:
        print(f"   (not provided in this turn)")


# ============================================================================
# EXAMPLE 5: Accessing Hidden State (for development/debugging)
# ============================================================================

def example_hidden_state():
    """Show how to access the full hidden state (for analysis/debugging)."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Hidden State (for analysis)")
    print("="*60)
    
    env = AMPIDEEnv(task_name="hard_multiturn_tool_attack", seed=42)
    obs = env.reset()
    
    # Take a few steps
    for _ in range(2):
        action = AMPIDEAction(action_type=ActionType.RESPOND, content="OK")
        result = env.step(action)
    
    # Get hidden state (only available locally, not via API)
    state = env.state()
    
    print(f"Hidden state fields:")
    print(f"  - task_name: {state['task_name']}")
    print(f"  - turn: {state['turn']}")
    print(f"  - done: {state['done']}")
    print(f"  - followed_malicious: {state['followed_malicious']}")
    print(f"  - leaked_data: {state['leaked_data']}")
    print(f"  - task_completed: {state['task_completed']}")
    print(f"  - flags_raised: {state['flags_raised']}")
    print(f"  - unsafe_tools_called: {state['unsafe_tools_called']}")
    print(f"  - correct_refusals: {state['correct_refusals']}")
    print(f"  - total_attacks: {state['total_attacks']}")
    print(f"  - cumulative_reward: {state['cumulative_reward']:.2f}")
    print(f"  - step_rewards: {[f'{r:.2f}' for r in state['step_rewards']]}")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█ AMPIDE Environment Usage Examples")
    print("█"*60)
    
    example_basic_usage()
    example_full_episode()
    example_test_all_tasks()
    example_inspect_observation()
    example_hidden_state()
    
    print("\n" + "█"*60)
    print("█ Examples complete! Edit this file to try your own agent logic.")
    print("█"*60 + "\n")
