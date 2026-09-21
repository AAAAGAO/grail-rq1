"""Keep Identification knowledge independently of official API resolution.

Occurrence IDs are scoped to pairs, never asserted to be official API FQNs.
No speculative API relations are created here.
"""
from collections import defaultdict
try:
    from scripts.graph_reference_overrides import apply_reference_overrides
except ModuleNotFoundError:
    from graph_reference_overrides import apply_reference_overrides


def knowledge_entity_id(row):
    if row.get('canonical_api'):
        return row['canonical_api']
    pair = row.get('pair_id', '')
    if not pair:
        raise ValueError('Knowledge retention requires a stable pair_id')
    return f'knowledge-occurrence:{pair}'


def knowledge_entity_kind(row):
    if row.get('canonical_api'):
        return 'resolved_api'
    return {
        'excluded_package_document': 'package_document',
        'excluded_resource_concept': 'resource_concept',
        'excluded_manifest_concept': 'manifest_concept',
        'excluded_user_defined_type': 'user_defined_type_occurrence',
        'owner_resolved': 'overload_unresolved_occurrence',
    }.get(row.get('resolution_status'), 'unresolved_occurrence')


def annotate_knowledge_rows(rows):
    return [dict(r, knowledge_entity_id=knowledge_entity_id(r),
                 knowledge_entity_kind=knowledge_entity_kind(r), corpus_eligible='1')
            for r in apply_reference_overrides(rows)]


def retain_knowledge_nodes(nodes, rows):
    rows = apply_reference_overrides(rows)
    grouped = defaultdict(list)
    pair_ids = set()
    for row in rows:
        pair = row.get('pair_id', '')
        if not pair or pair in pair_ids:
            raise ValueError(f'Missing or duplicate pair_id: {pair}')
        pair_ids.add(pair)
        grouped[knowledge_entity_id(row)].append(row)
    result = {n['node_id']: dict(n) for n in nodes}
    if len(result) != len(nodes):
        raise ValueError('Duplicate node IDs')
    for entity, local in grouped.items():
        first = local[0]
        if entity not in result:
            result[entity] = {
                'node_id': entity,
                'kind': ('resolved_api_without_structure' if first.get('canonical_api')
                         else knowledge_entity_kind(first)),
                'name': first.get('raw_api', ''), 'owner': first.get('canonical_owner', ''),
                'parameters': '', 'return_type': '', 'has_ku': '1', 'pair_count': 0,
                'pair_ids': '', 'raw_apis': '', 'reference': '', 'reference_origins': '',
                'artifact': first.get('symbol_artifact', ''),
            }
    for entity, node in result.items():
        local = grouped.get(entity, [])
        node.update(
            has_ku='1' if local else '0', pair_count=len(local),
            pair_ids=' | '.join(r['pair_id'] for r in local),
            raw_apis=' | '.join(sorted({r.get('raw_api', '') for r in local})),
            identity_status=('resolved' if not local or local[0].get('canonical_api')
                             else local[0].get('resolution_status', 'unresolved')),
            identity_candidates=' | '.join(dict.fromkeys(r.get('resolution_candidates', '') for r in local if r.get('resolution_candidates'))),
        )
        if local:
            node['reference'] = next((r.get('reference', '') for r in local if r.get('reference')), '')
            node['reference_origins'] = ' | '.join(sorted({r.get('reference_origin', '') for r in local if r.get('reference_origin')}))
        node['reference_alignment_status'] = ' | '.join(sorted({
            r.get('reference_alignment_status', 'not_reviewed_in_occurrence_audit') for r in local
        }))
    return list(result.values())
