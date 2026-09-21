"""Optional terminal scoring with stable ties and no candidate replacement."""

from scripts.evidence_candidate_retention import assess_candidates


def candidate_order_ties(pair_ids, assessments):
    """Retain controller order only when all scoring keys are equal."""
    if len(pair_ids) != len(set(pair_ids)) or set(pair_ids) != set(assessments):
        raise ValueError("scores must cover every unique candidate exactly once")
    return sorted(pair_ids, key=lambda p: (
        -int(assessments[p]["ku_support"] > 0),
        -assessments[p]["api_fit"],
        -assessments[p]["ku_support"],
    ))


def score_terminal_pairs(chat, query, pair_ids, lookup):
    if len(pair_ids) > 30:
        raise ValueError("terminal scoring cannot change the 30-pair budget")
    _, diagnostic, calls = assess_candidates(chat, query, pair_ids, lookup)
    ranked = candidate_order_ties(pair_ids, diagnostic["assessments"])
    diagnostic.update(
        policy="independent-score",
        tie_break="controller_candidate_order",
        selected_pair_ids=ranked,
    )
    return ranked, diagnostic, calls
