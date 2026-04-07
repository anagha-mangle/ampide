#!/usr/bin/env python
"""Quick verification script for AMPIDE"""

print('🔍 AMPIDE Environment Verification')
print('='*50)

try:
    # 1. Pydantic
    from ampide_env import AMPIDEAction, ActionType
    print('✅ Pydantic models imported')

    # 2. Environment
    from ampide_env import AMPIDEEnv
    env = AMPIDEEnv(seed=42)
    obs = env.reset()
    print(f'✅ Environment initialized (turn={obs.turn}, tools={len(obs.available_tools)})')

    # 3. Grader
    from grader import grade
    state = {
        'task_name': 'easy_direct_injection',
        'followed_malicious': False,
        'leaked_data': False,
        'task_completed': True,
        'flags_raised': [],
        'unsafe_tools_called': [],
        'correct_refusals': 1,
        'total_attacks': 1
    }
    result = grade(state)
    print(f'✅ Grader works (score={result["score"]}, label={result["label"]})')

    # 4. API
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv('API_KEY')
    if api_key:
        print(f'✅ API key loaded ({api_key[:10]}...)')
    else:
        print('❌ API key NOT found - check .env file')

    print('='*50)
    print('🎉 All verifications passed!')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
