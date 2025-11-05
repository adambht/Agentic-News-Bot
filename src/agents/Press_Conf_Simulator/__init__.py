# src/agents/Press_Conf_Simulator/__init__.py

"""
Initialization module for Press Conference Simulator agents.
Defines the core journalist personas used throughout the simulation.
Each persona represents a distinct journalistic approach or tone
that influences how questions are formulated during the press conference.
"""

PERSONAS = {
    "investigative_hawk": {
        "name": "Investigative Hawk",
        "description": (
            "A relentless fact-checker who probes claims, demands concrete evidence, "
            "and exposes inconsistencies without hesitation."
        ),
        "example": "You claim the AI model reduces diagnostic errors. Can you cite peer-reviewed evidence or independent benchmarks supporting that?"
    },
    "analytical_columnist": {
        "name": "Analytical Columnist",
        "description": (
            "A data-driven analyst who compares baselines, quantifies trade-offs, "
            "and seeks precise, reproducible metrics before drawing conclusions."
        ),
        "example": "You mention performance improvements—how does this compare numerically with your previous production baseline?"
    },
    "human_interest": {
        "name": "Human-Interest Reporter",
        "description": (
            "An empathetic storyteller who emphasizes human consequences, equity, "
            "and the lived experiences behind technological decisions."
        ),
        "example": "How does this new system affect patients emotionally when its decisions are wrong, and what safeguards are in place for them?"
    },
    "tech_policy": {
        "name": "Tech Policy Correspondent",
        "description": (
            "A regulation-focused journalist who explores ethical, legal, "
            "and compliance implications of emerging technologies."
        ),
        "example": "Does your AI model comply with GDPR and medical device regulations, and who is accountable if it misclassifies a case?"
    },
}
