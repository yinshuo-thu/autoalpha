"""
factor_idea_generator.py — Multi-Agent Factor Idea Generator

Supports both Evolutionary / Genetic Algorithm techniques and 
LLM API interactions to self-optimize and propose new factor formulas.

Methods:
1. Template Instantiation (Initial population)
2. Mutation (Operator swapping, window tweaking, structure changes)
3. Crossover (Combining AST subtrees from two parent factors)
4. Prompt-to-Template Mapping (For Formula Lab UI)
5. LLM API Generation (Using GPT models via AutoAgent)
"""
import os
import json
import random
import copy
from core.factor_experience import format_experiences_for_prompt, retrieve_relevant_experiences
from core.llm_mining_log import append_llm_mining_record
from formula_validator import validate_formula
from formula_parser import parse_formula, ast_to_string, FuncCallNode, BinaryOpNode, UnaryOpNode, NumberNode, FieldNode
from data_catalog import RAW_FIELDS, DERIVED_FIELD_TEMPLATES
from operator_catalog import OPERATORS

FAST_LLM_SYSTEM_PROMPT = (
    "You generate one quantitative factor in JSON only. "
    "Fields: open_trade_px, high_trade_px, low_trade_px, close_trade_px, trade_count, volume, dvolume, vwap. "
    "Ops: lag, delta, ts_pct_change, ts_mean, ts_ema, ts_std, ts_sum, ts_max, ts_min, ts_median, "
    "ts_quantile, ts_rank, ts_zscore, ts_minmax_norm, ts_decay_linear, ts_corr, ts_cov, "
    "cs_rank, cs_zscore, cs_demean, cs_scale, cs_winsorize, cs_neutralize, "
    "safe_div, signed_power, abs, sign, neg, log, signed_log, sqrt, clip, tanh, sigmoid, "
    "ifelse, gt, ge, lt, le, mean_of, weighted_sum, combine_rank. "
    "Never use future information: no resp, no trading_restriction, no forward/lead semantics, "
    "and every delay/lookback must be a positive integer. "
    "Return exactly one JSON object with keys thought_process, formula, postprocess, lookback_days."
)

# ── 1. Initial Templates (Seeds) ──
TEMPLATES = [
    # Mean Reversion
    "rank(sub(div(close_trade_px, vwap), 1))",
    "ts_mean(close_trade_px, {w1}) / close_trade_px - 1",
    "demean(delay(close_trade_px, {w1}) / close_trade_px)",
    
    # Momentum / Trend
    "ts_zscore(close_trade_px, {w1})",
    "delta(close_trade_px, {w1}) / delay(close_trade_px, {w1})",
    "ts_rank(close_trade_px, {w1})",
    
    # Volume / Liquidity
    "rank(volume) / rank(ts_mean(volume, {w1}))",
    "zscore(dvolume)",
    "ts_corr(close_trade_px, volume, {w1})",
    
    # Volatility
    "ts_std(close_trade_px, {w1}) / ts_mean(close_trade_px, {w1})",
    "rank(sub(high_trade_px, low_trade_px))",
    
    # Complex Combinations
    "rank(ts_corr(delay(close_trade_px, 1), close_trade_px, {w1})) + rank(ts_corr(volume, close_trade_px, {w2}))",
    "ifelse(gt(volume, ts_mean(volume, {w1})), rank(delta(close_trade_px, {w2})), neg(rank(delta(close_trade_px, {w2}))))"
]

def generate_initial_population(num_factors=10):
    """Generate initial formulas from templates with random parameters."""
    formulas = []
    attempts = 0
    max_attempts = max(20, num_factors * 12)
    while len(formulas) < num_factors and attempts < max_attempts:
        template = random.choice(TEMPLATES)
        w1 = random.choice([5, 10, 15, 20, 30, 60])
        w2 = random.choice([5, 10, 15, 20])
        formula_str = template.format(w1=w1, w2=w2)
        if _formula_is_generation_safe(formula_str):
            formulas.append(formula_str)
        attempts += 1
    return formulas


# ── 2. AST Extraction & Selection ──
def extract_nodes(node, types):
    """Recursively extract nodes of given types."""
    res = []
    if isinstance(node, types):
        res.append(node)
    
    if isinstance(node, FuncCallNode):
        for arg in node.args:
            res.extend(extract_nodes(arg, types))
    elif isinstance(node, BinaryOpNode):
        res.extend(extract_nodes(node.left, types))
        res.extend(extract_nodes(node.right, types))
    elif isinstance(node, UnaryOpNode):
        res.extend(extract_nodes(node.operand, types))
        
    return res


# ── 3. Mutation Operations ──
def mutate_parameter(ast):
    """Slightly adjust numeric parameters (e.g., window sizes)."""
    new_ast = copy.deepcopy(ast)
    num_nodes = extract_nodes(new_ast, (NumberNode,))
    if num_nodes:
        target = random.choice(num_nodes)
        if target.value > 1:
            delta = int(target.value * random.uniform(-0.2, 0.2))
            if delta == 0: delta = random.choice([-1, 1])
            target.value = max(1.0, float(int(target.value + delta)))
    return new_ast

def mutate_operator(ast):
    """Swap operators within the same category (e.g., add <-> sub)."""
    new_ast = copy.deepcopy(ast)
    # Binary Ops
    binary_nodes = extract_nodes(new_ast, (BinaryOpNode,))
    if binary_nodes and random.random() < 0.5:
        target = random.choice(binary_nodes)
        if target.op in ['+', '-']: target.op = '-' if target.op == '+' else '+'
        elif target.op in ['*', '/']: target.op = '/' if target.op == '*' else '*'
        return new_ast
        
    # Function calls
    func_nodes = extract_nodes(new_ast, (FuncCallNode,))
    if func_nodes:
        target = random.choice(func_nodes)
        meta = OPERATORS.get(target.name)
        if meta:
            cat = meta['category']
            same_cat_ops = [n for n, m in OPERATORS.items() if m['category'] == cat and len(m['params'].split(',')) == len(target.args)]
            if same_cat_ops:
                target.name = random.choice(same_cat_ops)
    return new_ast

def mutate_field(ast):
    """Swap data fields (e.g., volume <-> dvolume)."""
    new_ast = copy.deepcopy(ast)
    field_nodes = extract_nodes(new_ast, (FieldNode,))
    if field_nodes:
        target = random.choice(field_nodes)
        allowed = list(RAW_FIELDS.keys())
        target.name = random.choice(allowed)
    return new_ast

def mutate_formula(formula_str):
    """Apply a random mutation to a formula."""
    try:
        ast = parse_formula(formula_str)
        mutations = [mutate_parameter, mutate_operator, mutate_field]
        mutation = random.choice(mutations)
        new_ast = mutation(ast)
        return ast_to_string(new_ast)
    except Exception:
        return formula_str # Fallback


# ── 4. Crossover Operations ──
def crossover_formulas(f1_str, f2_str):
    """Combine components of two parent formulas."""
    try:
        ast1 = parse_formula(f1_str)
        ast2 = parse_formula(f2_str)
        
        funcs1 = extract_nodes(ast1, (FuncCallNode, BinaryOpNode))
        funcs2 = extract_nodes(ast2, (FuncCallNode, BinaryOpNode))
        
        if funcs1 and funcs2:
            new_ast = copy.deepcopy(ast1)
            # Find a node in new_ast to replace
            target_parent = [new_ast] # DUmmy root
            
            # Simple wrapper to find parent and replace - for EA, replacing entire subtrees is common
            # To keep it simple, we can combine them at root
            op = random.choice(['+', '-', '*', '/'])
            if op in ['+', '-']:
                return f"({f1_str}) {op} ({f2_str})"
            else:
                 return f"({f1_str}) {op} ({f2_str})"
    except Exception:
        return f1_str

    return f1_str

# ── 5. Main Idea Generation Endpoint ──
def generate_ideas_from_parents(parents, num_ideas=5):
    """
    Given a list of parent dicts {'formula': '...'}, 
    use crossover and mutation to generate self-optimized ideas.
    """
    if not parents:
        formulas = generate_initial_population(num_ideas)
        return [
            {
                'formula': f,
                'rationale': 'Randomly initialized from templates.',
                'validation': validate_formula(f).to_dict(),
            }
            for f in formulas
        ]
        
    ideas = []
    parent_formulas = [p['formula'] for p in parents if 'formula' in p]
    if not parent_formulas:
        formulas = generate_initial_population(num_ideas)
        return [
            {
                'formula': f,
                'rationale': 'Initialized from safe templates because no usable parent formulas were found.',
                'validation': validate_formula(f).to_dict(),
            }
            for f in formulas
        ]

    attempts = 0
    max_attempts = max(20, num_ideas * 12)
    while len(ideas) < num_ideas and attempts < max_attempts:
        method = random.random()
        if method < 0.4 and len(parent_formulas) >= 2:
            # Crossover
            p1, p2 = random.sample(parent_formulas, 2)
            child = crossover_formulas(p1, p2)
            _append_safe_idea(ideas, {'formula': child, 'rationale': 'Crossover of top factors.'}, "ea_crossover")
        elif method < 0.8:
            # Mutation
            p = random.choice(parent_formulas)
            child = mutate_formula(p)
            _append_safe_idea(ideas, {'formula': child, 'rationale': 'Mutated from a top factor.'}, "ea_mutation")
        else:
            # Fresh blood
            seeds = generate_initial_population(1)
            if seeds:
                _append_safe_idea(ideas, {'formula': seeds[0], 'rationale': 'Fresh exploration template.'}, "ea_template")
        attempts += 1
            
    return ideas

# ── 5.b Main LLM Idea Generation Endpoint ──
def generate_ideas_with_llm(parents, num_ideas=1, depth_hint: str = "3-4"):
    """
    Given a list of parent dicts {'formula': '...', 'Score': ...},
    construct a prompt and call the LLM API to generate new formulas.
    """
    try:
        from research.auto_agent import query_llm
    except ImportError:
        print("[LLM Error] Could not import query_llm from research.auto_agent. Falling back to EA.")
        return generate_ideas_from_parents(parents, num_ideas)

    history = [{"role": "system", "content": FAST_LLM_SYSTEM_PROMPT}]

    if not parents:
        prompt = (
            "No baseline factors yet. Generate one new 15-minute DSL alpha. "
            f"Target AST depth around {depth_hint}. Prefer price-volume or VWAP microstructure. JSON only."
        )
    else:
        best = parents[0]
        compact_parents = _format_compact_parent_lines(parents)
        prompt = (
            f"Current best score factor:\n{compact_parents or best.get('formula')}\n"
            f"Generate one mutated or orthogonal 15-minute DSL factor with AST depth around {depth_hint}. "
            "Keep it different from the examples above and return JSON only."
        )

    experience_block = _build_experience_prompt(prompt, parents)
    if experience_block:
        prompt = f"{prompt}\n\n{experience_block}"

    history.append({"role": "user", "content": prompt})

    ideas = []

    try:
        res_json = query_llm(history, mining_source="generate_ideas_with_llm")
        formula = res_json.get("formula")
        if formula and _formula_is_generation_safe(formula, source="llm", prompt_excerpt=prompt):
            ideas.append({
                "formula": formula,
                "rationale": res_json.get("thought_process", "Auto-generated by LLM based on parents."),
                "source": "llm",
                "validation": validate_formula(formula).to_dict(),
            })
    except Exception as e:
        print(f"[LLM Error] API request failed: {e}. Generating fallback idea...")

    if len(ideas) < num_ideas:
        seed_pool = [{"formula": idea["formula"]} for idea in ideas if idea.get("formula")]
        seed_pool.extend(parents)
        ideas.extend(generate_ideas_from_parents(seed_pool, num_ideas - len(ideas)))

    return ideas


def _format_parents_for_prompt(parents, limit: int = 6) -> str:
    """Summarize prior factors for LLM context (experience / exploration)."""
    if not parents:
        return ""
    lines = []
    for i, p in enumerate(parents[:limit], 1):
        fn = p.get("factor_name") or f"factor_{i}"
        f = (p.get("formula") or "").strip()
        if not f:
            continue
        sc = p.get("Score", 0)
        ic = p.get("IC", 0)
        pg = p.get("PassGates", False)
        cls = p.get("classification", "")
        lines.append(
            f"{i}. [{fn}] formula={f} | Score={sc} IC={ic} PassGates={pg} class={cls}"
        )
    return "\n".join(lines) if lines else ""


def _format_compact_parent_lines(parents, limit: int = 2) -> str:
    if not parents:
        return ""
    lines = []
    for idx, p in enumerate(parents[:limit], 1):
        formula = (p.get("formula") or "").strip()
        if not formula:
            continue
        lines.append(f"{idx}) {formula}")
    return "\n".join(lines)


def generate_ideas_with_prompt(prompt_text, parents=None, num_ideas=3, depth_hint: str = "3-4"):
    """
    Generate factor ideas from a natural-language prompt.
    Prefers LLM output when configured; falls back to rule-based prompt mapping.
    When ``parents`` is provided (e.g. top leaderboard rows), injects them so Claude can
    build on or orthogonalize prior exploration.
    """
    parents = parents or []
    ideas = []

    try:
        from research.auto_agent import query_llm

        compact_parents = _format_compact_parent_lines(parents)
        user_a = f"Research: {prompt_text}\n"
        if compact_parents:
            user_a += (
                "Current stronger factors:\n"
                f"{compact_parents}\n"
            )
        experience_block = _build_experience_prompt(prompt_text, parents)
        if experience_block:
            user_a += f"{experience_block}\n"
        user_a += (
            f"Need one 15-minute DSL factor, different from the examples above, "
            f"with AST depth around {depth_hint}. Prefer price-volume or VWAP microstructure. "
            "Return JSON only."
        )

        history = [
            {"role": "system", "content": FAST_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_a},
        ]

        print("[LLM] generate_ideas_with_prompt: requesting completion (may take 30–120s)...", flush=True)
        response = query_llm(history, mining_source="generate_ideas_with_prompt")
        print("[LLM] generate_ideas_with_prompt: got response, validating formula...", flush=True)
        formula = response.get("formula")
        if formula and _formula_is_generation_safe(formula, source="prompt_llm", prompt_excerpt=prompt_text):
            ideas.append({
                "formula": formula,
                "rationale": response.get("thought_process", f"Generated from prompt: {prompt_text}"),
                "postprocess": response.get("postprocess"),
                "lookback_days": response.get("lookback_days"),
                "source": "llm",
                "validation": validate_formula(formula).to_dict(),
            })
    except Exception:
        pass

    if len(ideas) < num_ideas:
        if ideas:
            seed_pool = [{"formula": idea["formula"]} for idea in ideas if idea.get("formula")]
            seed_pool.extend(parents[:2])
            for item in generate_ideas_from_parents(seed_pool, num_ideas - len(ideas)):
                if len(ideas) >= num_ideas:
                    break
                _append_safe_idea(ideas, {
                    "formula": item["formula"],
                    "rationale": item.get("rationale", "Mutated from a prior factor to avoid an extra LLM round."),
                    "source": "ea_mutation",
                }, "prompt_fallback_mutation")

    if len(ideas) < num_ideas:
        fallback = generate_from_prompt(prompt_text)
        for item in fallback:
            if len(ideas) >= num_ideas:
                break
            _append_safe_idea(ideas, {
                "formula": item["formula"],
                "rationale": item.get("desc", f"Mapped from prompt: {prompt_text}"),
                "source": "rule_based",
            }, "rule_based_prompt")

    if len(ideas) < num_ideas:
        ideas.extend(generate_ideas_from_parents(parents, num_ideas - len(ideas)))

    return ideas[:num_ideas]

# ── 6. UI Prompt-to-Formula Mapping ──
def generate_from_prompt(prompt_text):
    """
    Natural Language -> Formula candidates (Rule-based mapping for Formula Lab).
    """
    prompt = prompt_text.lower()
    candidates = []
    
    if 'vwap' in prompt and 'dev' in prompt:
        candidates.append({'formula': 'close_trade_px / vwap - 1', 'desc': 'Simple VWAP deviation'})
        candidates.append({'formula': 'ts_mean(close_trade_px / vwap - 1, 10)', 'desc': 'Smoothed VWAP deviation'})
    
    if 'reversion' in prompt or 'momentum' in prompt:
        candidates.append({'formula': 'rank(delta(close_trade_px, 5))', 'desc': 'Cross-sectional short-term momentum'})
        candidates.append({'formula': 'neg(ts_zscore(close_trade_px, 10))', 'desc': 'Time-series mean reversion'})
        
    if 'volume' in prompt:
        candidates.append({'formula': 'volume / ts_mean(volume, 20)', 'desc': 'Volume ratio'})
        candidates.append({'formula': 'rank(volume) * rank(delta(close_trade_px, 1))', 'desc': 'Price-volume trend'})

    if not candidates:
        # Fallback to random templates
        formulas = generate_initial_population(2)
        candidates = [{'formula': f, 'desc': 'Suggested exploration formula'} for f in formulas]
        
    return candidates


def _build_experience_prompt(query_text, parents) -> str:
    parent_block = _format_parents_for_prompt(parents or [], limit=4)
    query = "\n".join(part for part in [query_text or "", parent_block] if part)
    experiences = retrieve_relevant_experiences(query, limit=4)
    if not experiences:
        return ""
    return (
        "Relevant prior experience for reference only. Use it as soft context, not as a constraint; "
        "you can deliberately go orthogonal if you have a better hypothesis:\n"
        f"{format_experiences_for_prompt(experiences)}"
    )


def _append_safe_idea(ideas, idea, source: str) -> bool:
    formula = (idea.get("formula") or "").strip()
    if not formula:
        return False
    validation = validate_formula(formula)
    if not validation.valid:
        _log_generation_rejection(formula, validation.errors, source)
        return False
    checked = dict(idea)
    checked["formula"] = formula
    checked["validation"] = validation.to_dict()
    ideas.append(checked)
    return True


def _formula_is_generation_safe(formula: str, source: str = "generator", prompt_excerpt: str = "") -> bool:
    validation = validate_formula((formula or "").strip())
    if validation.valid:
        return True
    _log_generation_rejection(formula, validation.errors, source, prompt_excerpt=prompt_excerpt)
    return False


def _log_generation_rejection(formula: str, errors, source: str, prompt_excerpt: str = "") -> None:
    try:
        append_llm_mining_record(
            {
                "event": "generated_formula_rejected",
                "source": source,
                "formula": formula,
                "errors": list(errors or []),
                "user_prompt_excerpt": (prompt_excerpt or "")[:800],
            }
        )
    except Exception:
        pass
