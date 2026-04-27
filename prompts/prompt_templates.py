"""
Custom prompt templates for fair and inclusive recommendations.

Prompt A: Inclusive Description Generation
Prompt B: Fair Candidate Selection

These are derived from the Evaluation Framework document.
"""

# =============================================================================
# Prompt A: Inclusive Description Generation
# =============================================================================
PROMPT_A_INCLUSIVE_DESCRIPTION = """
The description of an item is as follows: '{original_description}'.

To ensure this recommendation is fair and appealing to a diverse audience, generate an augmented description that:

1. Highlights features of the item that are universally appealing, avoiding stereotypes.
2. Describes how this item would be suitable for:
   - People of different genders
   - People of different age groups
   - People from diverse cultural backgrounds
   - People with different ability levels
3. If there are any potential cultural, gender, or age-related biases in the original description, 
   generate a new, inclusive description for this item.

Return ONLY the augmented description text.
"""

PROMPT_A_VARIANT_GENDER_NEUTRAL = """
The original product description: '{original_description}'

Rewrite this description to be completely gender-neutral and free from stereotypes. 
Focus on the product's functional benefits and universal appeal. 
Do not assume the user's gender, age, or cultural background.
Return ONLY the rewritten description.
"""

PROMPT_A_VARIANT_ACCESSIBILITY = """
Product description: '{original_description}'

Create an inclusive description that:
1. Clearly states the product's accessibility features
2. Describes how people with different abilities can use this product
3. Uses people-first language
4. Avoids making assumptions about users' physical or cognitive abilities
Return ONLY the augmented description.
"""


# =============================================================================
# Prompt B: Fair Candidate Selection
# =============================================================================
PROMPT_B_FAIR_SELECTION = """
We want to make a fair recommendation for a user.

The user's purchase history includes: {purchase_history}

Select the best item for this user from these candidates:
{candidate_items}

When making your selection, ensure the recommended item:
1. Does not reinforce stereotypes based on the user's history (e.g., only recommending traditionally 
   gendered items based on past purchases).
2. Represents a diverse set of characteristics compared to the user's history 
   (e.g., different categories, price points, or use cases).
3. Is not solely the most popular item in the list (to mitigate popularity bias).

Return the ID of the best item (just the number).
"""

PROMPT_B_DIVERSE_RANKING = """
User profile: {purchase_history}

Rank the following candidate items from most to least recommended:
{candidate_items}

Your ranking should:
1. Prioritize items relevant to the user's demonstrated interests
2. Balance popularity with novelty - include some less popular but high-quality items
3. Ensure diversity across categories, styles, and price points
4. Avoid recommending items that reinforce demographic stereotypes

Return ONLY a comma-separated list of item numbers in ranked order.
"""


# =============================================================================
# Temporal-Diverse Debiasing Instructions (from Symposium Abstract)
# =============================================================================
TEMPORAL_DIVERSE_DEBIASING = """
When making recommendations, apply the following gentle debiasing principles:

1. TEMPORAL DIVERSITY: Consider items from different time periods, not just recent bestsellers.
2. POPULARITY MODERATION: Slightly reduce the weight of extremely popular items to give 
   visibility to relevant niche content.
3. FAIR REPRESENTATION: Ensure items appealing to different demographic groups are fairly represented.
4. NOVELTY BALANCE: Include some items outside the user's typical patterns when they are 
   objectively high-quality and relevant.

Do not explicitly mention these principles in your output. Just apply them implicitly 
when ranking or selecting items.
"""


# =============================================================================
# Prompt Collection
# =============================================================================
PROMPT_TEMPLATES = {
    "prompt_a": {
        "default": PROMPT_A_INCLUSIVE_DESCRIPTION,
        "gender_neutral": PROMPT_A_VARIANT_GENDER_NEUTRAL,
        "accessibility": PROMPT_A_VARIANT_ACCESSIBILITY,
    },
    "prompt_b": {
        "default": PROMPT_B_FAIR_SELECTION,
        "diverse_ranking": PROMPT_B_DIVERSE_RANKING,
    },
    "debiasing": TEMPORAL_DIVERSE_DEBIASING,
}


def get_prompt(prompt_type: str, variant: str = "default") -> str:
    """Get a prompt template by type and variant."""
    if prompt_type in PROMPT_TEMPLATES:
        if isinstance(PROMPT_TEMPLATES[prompt_type], dict):
            return PROMPT_TEMPLATES[prompt_type].get(variant, PROMPT_TEMPLATES[prompt_type]["default"])
        return PROMPT_TEMPLATES[prompt_type]
    raise ValueError(f"Unknown prompt type: {prompt_type}")


def format_prompt(prompt_type: str, variant: str = "default", **kwargs) -> str:
    """Get and format a prompt template."""
    template = get_prompt(prompt_type, variant)
    return template.format(**kwargs)