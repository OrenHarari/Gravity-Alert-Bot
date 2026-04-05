import json
try:
    with open('unified_dashboard_data.json', encoding='utf-8') as f:
        d = json.load(f)
    if isinstance(d['assets'], list):
        for asset in d['assets']:
            strategies = asset.get('strategies', [])
            for strat_data in strategies:
                name = strat_data.get('name', 'Unknown')
                metrics = strat_data.get('metrics', {})
                print(f"Strategy: {name}")
                print(f"Keys: {list(metrics.keys())}")
                break
except Exception as e:
    import traceback
    traceback.print_exc()
