import sys
sys.path.insert(0, '../../source')
sys.path.insert(0, '../../main')

import pickle

with open('output/ecev/bin/single/samchn1_10.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
if isinstance(data, dict):
    print(f"Dict keys: {list(data.keys())}")
    
    # Check if it's a ChainPointer structure
    if 'head' in data:
        print("  Has 'head' key (ChainPointer structure)")
        head = data['head']
        if head is not None:
            print(f"  head type: {type(head)}")
            if hasattr(head, 'ob'):
                print(f"  head.ob type: {type(head.ob)}")
                ob = head.ob
                if hasattr(ob, 'en'):
                    print(f"  head.ob.en: {ob.en}")
                if hasattr(ob, 'jm'):
                    print(f"  head.ob.jm: {ob.jm}")
    
    # Check for array data
    for key in ['particles', 'samples', 'sp']:
        if key in data:
            val = data[key]
            print(f"  {key} type: {type(val)}, length: {len(val) if hasattr(val, '__len__') else 'N/A'}")
            if hasattr(val, '__len__') and len(val) > 0:
                first = val[0]
                print(f"    First element: {type(first)}")
                attrs_to_check = ['en', 'jm', 'm', 'w', 'exit_flag', 'x', 'v']
                for attr in attrs_to_check:
                    if hasattr(first, attr):
                        v = getattr(first, attr)
                        print(f"      {attr}: {v}")
                
                # Check energy range
                import numpy as np
                energies = np.array([p.en for p in val])
                print(f"    Energy range: [{np.min(energies):.3f}, {np.max(energies):.3f}]")
                print(f"    Angular momentum range: [{np.min([p.jm for p in val]):.3f}, {np.max([p.jm for p in val]):.3f}]")
elif isinstance(data, list):
    print(f"Length: {len(data)}")
    if len(data) > 0:
        first = data[0]
        print(f"First element: {type(first)}")

