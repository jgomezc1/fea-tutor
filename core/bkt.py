"""
Bayesian Knowledge Tracing (BKT) implementation for student mastery estimation.

Updates the probability that a student has mastered a skill based on observed
performance (correct/incorrect) using the standard BKT formulas.
"""


def update_mastery(p_mastery: float, observed_correct: bool, params: dict) -> float:
    """
    Update mastery probability given an observation.

    Args:
        p_mastery: Current probability of mastery (0.0 to 1.0)
        observed_correct: Whether the student's response was correct
        params: Dict with keys 'p_slip', 'p_guess', 'p_transit'

    Returns:
        Updated probability of mastery
    """
    p_slip = params["p_slip"]
    p_guess = params["p_guess"]
    p_transit = params["p_transit"]

    if observed_correct:
        # P(mastered | correct) using Bayes' theorem
        p_correct_given_mastered = 1.0 - p_slip
        p_correct_given_not_mastered = p_guess
        p_correct = (p_correct_given_mastered * p_mastery +
                     p_correct_given_not_mastered * (1.0 - p_mastery))

        if p_correct == 0:
            p_mastered_given_obs = 0.0
        else:
            p_mastered_given_obs = (p_correct_given_mastered * p_mastery) / p_correct
    else:
        # P(mastered | incorrect) using Bayes' theorem
        p_incorrect_given_mastered = p_slip
        p_incorrect_given_not_mastered = 1.0 - p_guess
        p_incorrect = (p_incorrect_given_mastered * p_mastery +
                       p_incorrect_given_not_mastered * (1.0 - p_mastery))

        if p_incorrect == 0:
            p_mastered_given_obs = 0.0
        else:
            p_mastered_given_obs = (p_incorrect_given_mastered * p_mastery) / p_incorrect

    # Apply learning transition: even if not mastered, there's a chance the
    # student learned from this interaction
    p_mastery_updated = (p_mastered_given_obs +
                         (1.0 - p_mastered_given_obs) * p_transit)

    return min(max(p_mastery_updated, 0.0), 1.0)


def check_mastery(p_mastery: float, threshold: float, level0_passed: bool,
                  unresolved_misconceptions: list) -> bool:
    """
    Conservative mastery check. All three conditions must be met:
    1. P(mastery) >= threshold
    2. Student passed a level 0 (no scaffolding) assessment
    3. No unresolved misconceptions

    Args:
        p_mastery: Current mastery probability
        threshold: Mastery threshold (default 0.90)
        level0_passed: Whether student passed assessment without scaffolding
        unresolved_misconceptions: List of unresolved misconception IDs

    Returns:
        True if mastery criteria are met
    """
    return (p_mastery >= threshold and
            level0_passed and
            len(unresolved_misconceptions) == 0)
