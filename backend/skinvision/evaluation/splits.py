"""Stable group allocation without using outcome-dependent selection."""
import hashlib


def stable_order(identifier, seed=20261119):
    return hashlib.sha256(f'{seed}:{identifier}'.encode()).hexdigest()


def grouped_split(rows, seed=20261119, fractions=(.7,.15,.15)):
    if len(fractions)!=3 or min(fractions)<0 or abs(sum(fractions)-1)>1e-8:
        raise ValueError('Invalid split fractions')
    parents = {}
    def find(node):
        parents.setdefault(node,node)
        if parents[node] != node:
            parents[node] = find(parents[node])
        return parents[node]
    nodes_by_row = []
    for row in rows:
        nodes = [prefix+str(row[key]) for key,prefix in [('patient_id','patient:'),('lesion_id','lesion:')] if row.get(key)]
        if not nodes:
            raise ValueError('Group ID required; never silently claim image splits prevent patient leakage')
        for node in nodes:
            parents[find(node)] = find(nodes[0])
        nodes_by_row.append(nodes)
    components = {}
    for node in parents:
        components.setdefault(find(node),[]).append(node)
    group_names = {root:min(nodes) for root,nodes in components.items()}
    groups = {}
    for row,nodes in zip(rows,nodes_by_row):
        group = group_names[find(nodes[0])]
        groups.setdefault(group,[]).append(row)
    result = []
    for group in sorted(groups):
        value = int(stable_order(group,seed),16) / 2**256
        split = 'train' if value<fractions[0] else 'val' if value<sum(fractions[:2]) else 'test'
        for row in groups[group]:
            result.append({**row,'split':split,'group_id':group})
    return result
