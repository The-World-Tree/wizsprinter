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

THE TWO POLICIES
    Damage (`<damage>`, `<aoe>`) ranks on expected damage, descending, with
    per-pip spells demoted to a class of their own: Tempest's 80 is 80 *per
    pip*, not comparable with Storm Lord's 755, and X-pip spells are held in
    reserve as a backup hit anyway since they cast at any pip count.

    Buffs and debuffs (`<blade>`, `<trap>`, `<charm>`, `<ward>`, the
    incoming/outgoing pairs, and the `<mod_*>` enchantments) rank on the
    magnitude of the buff, descending.  Magnitude rather than signed value
    because a ward and a charm are *defined* by a negative param: a -70 Tower
    Shield is the stronger of two shields, not the weaker.  Flat modifiers are
    demoted the way per-pip damage is -- a +225 flat blade and a +35% blade
    both read as a bare int, so comparing them by value would hand every
    contest to the flat one.  Flat buffs are rare; percentage always wins.

    Only one blade of a given id is consumed per hit, but blades do stack --
    two Storm Blades both land if one of them has been sharpened -- so there is
    no "save the big one for later" case to model here.  The strongest is
    simply the one to cast.

    What does not add up is the several blades a *single* card can apply, since
    they go on different schools: see `BestOf`.

THE TIEBREAKS, shared by both policies
    1. Lower pip cost.  The same effect for fewer pips is strictly better.
    2. Spend the card that cannot be improved or saved, in this order:
       item card, then the deck copy, then the treasure card.  An item card
       cannot be enchanted and cannot be kept for later, so holding it gains
       nothing; a treasure card is worth keeping back.
    3. Enchanted before unenchanted.  The enchanted card cannot be improved
       further, while the plain one can still take an enchant next round.

WHY A TREE AND NOT A SUM
    `effect_param` carries the whole story on a live client: it is the *total*
    for a damage-over-time effect (not the per-round tick -- `param_per_round`
    reads 0), and it already includes any enchantment on the card.

    What it does not carry is whether two effects both land.  A spell that
    rolls a random amount stores one effect per possible value, so an enchanted
    Humongofrog reads as five sibling damage effects of 575, 585, 595, 605 and
    615 -- one of which happens.  Summing those scores it 2975 and ranks it
    above a Drop Bear Fury that really does hit for 765.

    The game distinguishes the two cases by *container*: `EffectListSpellEffect`
    holds effects that all land, `RandomSpellEffect` and `VariableSpellEffect`
    hold alternatives.  `get_inner_card_effects` flattens all of them alike,
    which is right for matching -- "does any leaf satisfy this requirement?" --
    and wrong for scoring.  So scoring keeps the shape instead, as `AllOf` and
    `OneOf` nodes, and the caller maps the game's containers onto them.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional, Sequence, Tuple, Union

from .combat_api import SpellType, TemplateSpell


class CardSource(IntEnum):
    """Where a card came from, ordered by which to spend first.

    Sorted on directly, so the member values are the policy: an item card is
    dead weight in hand (it cannot be enchanted and cannot be saved for a later
    duel), a deck card is the ordinary case, and a treasure card is worth
    keeping back.
    """

    item = 0
    deck = 1
    treasure = 2


@dataclass(frozen=True)
class Amount:
    """A leaf: one param that counts toward this category's score.

    Always a magnitude. Wards and charms carry a negative `effect_param` by
    definition, and a bigger negative is a better ward, so the caller takes the
    absolute value on the way in and this module never sees a sign.
    """

    param: int


@dataclass(frozen=True)
class AllOf:
    """Every child lands -- an initial hit plus its damage-over-time tail.

    Also the empty node: `AllOf(())` scores zero, which is what a card with
    nothing that counts for this category should be worth.
    """

    children: Tuple["ScoreNode", ...] = ()


@dataclass(frozen=True)
class BestOf:
    """Every child lands, but only the best one counts toward the score.

    A blade card that applies several blades applies them to *different*
    schools: a sharpened Epiphany Blade puts 55% on Myth and 55% on Storm, and
    whichever attack follows is helped by one of them, never both. Summing them
    scores it 110 for a 55% buff, which inverts every close comparison -- a
    plain Epiphany Blade sums to 90 and buries a sharpened Mythblade it in fact
    ties with at 45%, and loses to on pip cost. So buff categories combine with
    max where damage combines with sum.

    Also the empty node: `BestOf(())` scores zero.
    """

    children: Tuple["ScoreNode", ...] = ()


@dataclass(frozen=True)
class OneOf:
    """Exactly one child lands, so the score is the mean over the branches.

    Covers a random damage spread, the pip ladder of an X-pip spell, and the
    branches of a conditional effect. The mean is the expected value for the
    random case, which is the one that actually distorts ranking.
    """

    children: Tuple["ScoreNode", ...] = ()


ScoreNode = Union[Amount, AllOf, BestOf, OneOf]


def expected_value(node: ScoreNode) -> float:
    """Expected magnitude for an effect tree."""
    if isinstance(node, Amount):
        return float(node.param)
    if isinstance(node, AllOf):
        return sum(expected_value(child) for child in node.children)
    if isinstance(node, BestOf):
        return max((expected_value(child) for child in node.children), default=0.0)
    if isinstance(node, OneOf):
        if not node.children:
            return 0.0
        return sum(expected_value(child) for child in node.children) / len(node.children)
    raise TypeError(f"not a score node: {node!r}")


@dataclass(frozen=True)
class CardFacts:
    """What ranking needs to know about one card, already classified.

    Attributes:
        is_per_pip: The card's damage scales with pips spent rather than being
            fixed, so its params are not comparable with a fixed-damage card's.
        score: The effect tree for this category.
        enchant_bonus: Value a not-yet-applied enchant would add to this card
            if it were chosen. Zero for a card that cannot receive the pending
            enchant, or when no enchant is in hand.
        is_flat: The card's buff is a flat amount rather than a percentage, so
            its param is not comparable with a percentage card's.
        pip_cost: Total pips to cast, shadow pips included.
        source: Item card, deck card or treasure card.
        is_enchanted: The card already carries an enchant, so it cannot be
            improved any further.
    """

    is_per_pip: bool = False
    score: ScoreNode = AllOf()
    enchant_bonus: int = 0
    is_flat: bool = False
    pip_cost: int = 0
    source: CardSource = CardSource.deck
    is_enchanted: bool = False

    @property
    def total(self) -> float:
        return expected_value(self.score) + self.enchant_bonus


SortKey = Callable[[CardFacts], tuple]


def _tiebreaks(facts: CardFacts) -> tuple:
    """The tail both policies share, applied once value has failed to decide."""
    return (facts.pip_cost, int(facts.source), 0 if facts.is_enchanted else 1)


def damage_sort_key(facts: CardFacts) -> tuple:
    """Sort key for damage-dealing categories. Lower sorts first."""
    return (1 if facts.is_per_pip else 0, -facts.total) + _tiebreaks(facts)


def buff_sort_key(facts: CardFacts) -> tuple:
    """Sort key for buff and debuff categories. Lower sorts first."""
    return (1 if facts.is_flat else 0, -facts.total) + _tiebreaks(facts)


@dataclass(frozen=True)
class CategoryRanker:
    """How to rank one spell category.

    Attributes:
        score_type: The SpellType whose per-effect predicate decides which
            leaves count toward the score. Usually the category itself; `aoe`
            scores damage, since its own predicate only asks about targeting.
        aoe_only: Score only the effects aimed at a whole team, rather than
            every effect the predicate accepts.
        effects_stack: Whether effects that land together add up. Damage does —
            an initial hit plus its damage-over-time tail both hurt. Buffs do
            not: a card applying several blades applies them to different
            schools, so only the best one helps the attack that follows. The
            caller builds `AllOf` or `BestOf` accordingly.
        sort_key: Maps the card's facts to a key that sorts best-first.
    """

    score_type: SpellType
    aoe_only: bool
    effects_stack: bool
    sort_key: SortKey


def _damage(spell_type: SpellType, aoe_only: bool = False) -> CategoryRanker:
    return CategoryRanker(
        score_type=spell_type, aoe_only=aoe_only, effects_stack=True, sort_key=damage_sort_key
    )


def _buff(spell_type: SpellType) -> CategoryRanker:
    return CategoryRanker(
        score_type=spell_type, aoe_only=False, effects_stack=False, sort_key=buff_sort_key
    )


# One row per rankable category. A SpellType absent from here is left in the
# caller's order, which is what every category did before ranking existed.
CATEGORY_RANKERS = {
    SpellType.type_aoe: _damage(SpellType.type_damage, aoe_only=True),
    SpellType.type_damage: _damage(SpellType.type_damage),
    SpellType.type_mod_damage: _buff(SpellType.type_mod_damage),
    SpellType.type_mod_heal: _buff(SpellType.type_mod_heal),
    SpellType.type_mod_pierce: _buff(SpellType.type_mod_pierce),
    SpellType.type_blade: _buff(SpellType.type_blade),
    SpellType.type_trap: _buff(SpellType.type_trap),
    SpellType.type_charm: _buff(SpellType.type_charm),
    SpellType.type_ward: _buff(SpellType.type_ward),
    SpellType.type_inc_damage: _buff(SpellType.type_inc_damage),
    SpellType.type_out_damage: _buff(SpellType.type_out_damage),
    SpellType.type_inc_heal: _buff(SpellType.type_inc_heal),
    SpellType.type_out_heal: _buff(SpellType.type_out_heal),
    SpellType.type_pierce: _buff(SpellType.type_pierce),
}

# Checked in order, first match wins, so `any<damage,aoe>` ranks as an AoE --
# the narrower of the two requirements the template actually asked for -- and
# `any<blade,out_damage>` ranks as a blade rather than as the broader
# effect-type family a blade happens to belong to.
RANKER_PRECEDENCE = (
    SpellType.type_aoe,
    SpellType.type_damage,
    SpellType.type_mod_damage,
    SpellType.type_mod_heal,
    SpellType.type_mod_pierce,
    SpellType.type_blade,
    SpellType.type_trap,
    SpellType.type_charm,
    SpellType.type_ward,
    SpellType.type_inc_damage,
    SpellType.type_out_damage,
    SpellType.type_inc_heal,
    SpellType.type_out_heal,
    SpellType.type_pierce,
)


def ranker_for(template: TemplateSpell) -> Optional[CategoryRanker]:
    """The ranker for a template's category, or None to keep the caller's order.

    `requirements` is heterogeneous -- it holds SpellType values alongside
    GambitSpec/ClearSpec/EchoSpec/SwapSpec filters -- so this only ever looks
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
    dependency on what a card is. The sort is stable, so cards that tie on
    every key hold the order they were given.
    """
    return sorted(range(len(facts)), key=lambda i: ranker.sort_key(facts[i]))
