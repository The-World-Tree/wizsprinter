"""Ranking spells *within* a category.

The matcher in `sprinty_combat` answers "does this card satisfy `any<aoe>`?"
with a yes or a no, which leaves every match equally good and forces a strategy
to name spells explicitly just to express a preference::

    storm lord | tempest

That is school-specific DSL written only to say "the big one first", and every
school that carries more than one AoE needs its own copy of it.

This module supplies the missing half: given the cards that matched, which is
the *best* one.  A category with no ranker registered keeps the caller's
original order, so adding a category here is purely additive.

Damage ranking, the only category implemented so far, is two keys:

1. Fixed-damage spells before per-pip ones.  Tempest's 80 is 80 *per pip*, so
   its param is not comparable with Storm Lord's 755 — and per-pip spells are
   held in reserve as a backup hit anyway, since they cast at any pip count.
   They therefore sort last as a class rather than by value.
2. Expected damage, descending.

WHY A TREE AND NOT A SUM
    `effect_param` carries the whole story on a live client: it is the *total*
    for a damage-over-time effect (not the per-round tick — `param_per_round`
    reads 0), and it already includes any enchantment on the card.

    What it does not carry is whether two damage effects both land.  A spell
    that rolls a random amount stores one effect per possible value, so an
    enchanted Humongofrog reads as five sibling damage effects of 575, 585,
    595, 605 and 615 — one of which happens.  Summing those scores it 2975 and
    ranks it above a Drop Bear Fury that really does hit for 765.

    The game distinguishes the two cases by *container*: `EffectListSpellEffect`
    holds effects that all land, `RandomSpellEffect` and `VariableSpellEffect`
    hold alternatives.  `get_inner_card_effects` flattens all of them alike,
    which is right for matching — "does any leaf satisfy this requirement?" —
    and wrong for scoring.  So scoring keeps the shape instead, as `AllOf` and
    `OneOf` nodes, and the caller maps the game's containers onto them.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

from .combat_api import SpellType, TemplateSpell


@dataclass(frozen=True)
class DamageAmount:
    """A leaf: one damage number that counts toward this category's score."""

    param: int


@dataclass(frozen=True)
class AllOf:
    """Every child lands — an initial hit plus its damage-over-time tail.

    Also the empty node: `AllOf(())` scores zero, which is what a card with no
    damage that counts for this category should be worth.
    """

    children: Tuple["DamageNode", ...] = ()


@dataclass(frozen=True)
class OneOf:
    """Exactly one child lands, so the score is the mean over the branches.

    Covers a random damage spread, the pip ladder of an X-pip spell, and the
    branches of a conditional effect. The mean is the expected damage for the
    random case, which is the one that actually distorts ranking.
    """

    children: Tuple["DamageNode", ...] = ()


DamageNode = Union[DamageAmount, AllOf, OneOf]


def expected_damage(node: DamageNode) -> float:
    """Expected damage for an effect tree."""
    if isinstance(node, DamageAmount):
        return float(node.param)
    if isinstance(node, AllOf):
        return sum(expected_damage(child) for child in node.children)
    if isinstance(node, OneOf):
        if not node.children:
            return 0.0
        return sum(expected_damage(child) for child in node.children) / len(node.children)
    raise TypeError(f"not a damage node: {node!r}")


@dataclass(frozen=True)
class CardFacts:
    """What ranking needs to know about one card, already classified.

    Attributes:
        is_per_pip: The card's damage scales with pips spent rather than being
            fixed, so its params are not comparable with a fixed-damage card's.
        damage: The effect tree for this category's damage.
        enchant_bonus: Damage a not-yet-applied enchant would add to this card
            if it were chosen. Zero for a card that cannot receive the pending
            enchant, or when no enchant is in hand.
    """

    is_per_pip: bool
    damage: DamageNode = AllOf()
    enchant_bonus: int = 0

    @property
    def total(self) -> float:
        return expected_damage(self.damage) + self.enchant_bonus


SortKey = Callable[[CardFacts], tuple]


def damage_sort_key(facts: CardFacts) -> tuple:
    """Sort key for damage-dealing categories. Lower sorts first."""
    return (1 if facts.is_per_pip else 0, -facts.total)


@dataclass(frozen=True)
class CategoryRanker:
    """How to rank one spell category.

    Attributes:
        aoe_only: Score only the effects aimed at a whole team, rather than
            every enemy-targeting effect.
        sort_key: Maps the card's facts to a key that sorts best-first.
    """

    aoe_only: bool
    sort_key: SortKey


# One row per rankable category. A SpellType absent from here is left in the
# caller's order, which is what every category did before ranking existed.
CATEGORY_RANKERS = {
    SpellType.type_aoe: CategoryRanker(aoe_only=True, sort_key=damage_sort_key),
    SpellType.type_damage: CategoryRanker(aoe_only=False, sort_key=damage_sort_key),
}

# Checked in order, first match wins, so `any<damage,aoe>` ranks as an AoE —
# the narrower of the two requirements the template actually asked for.
RANKER_PRECEDENCE = (
    SpellType.type_aoe,
    SpellType.type_damage,
)


def ranker_for(template: TemplateSpell) -> Optional[CategoryRanker]:
    """The ranker for a template's category, or None to keep the caller's order.

    `requirements` is heterogeneous — it holds SpellType values alongside
    GambitSpec/ClearSpec/EchoSpec/SwapSpec filters — so this only ever looks
    for the SpellType members it knows about.
    """
    if not isinstance(template, TemplateSpell):
        return None
    for spell_type in RANKER_PRECEDENCE:
        if spell_type in template.requirements:
            return CATEGORY_RANKERS[spell_type]
    return None


def rank_facts(facts: Sequence[CardFacts], ranker: CategoryRanker) -> list:
    """Return the indices of `facts`, best first.

    Sorting indices rather than the cards themselves keeps this free of any
    dependency on what a card is. The sort is stable, so cards that tie hold
    the order they were given — which preserves the enchanted-first ordering
    `get_cards()` establishes for categories where damage cannot break a tie.
    """
    return sorted(range(len(facts)), key=lambda i: ranker.sort_key(facts[i]))
