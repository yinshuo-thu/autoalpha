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
import random
import copy
from formula_parser import parse_formula, ast_to_string, FuncCallNode, BinaryOpNode, UnaryOpNode, NumberNode, FieldNode
from data_catalog import RAW_FIELDS, DERIVED_FIELD_TEMPLATES
from operator_catalog import OPERATORS

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
    for _ in range(num_factors):
        template = random.choice(TEMPLATES)
        w1 = random.choice([5, 10, 15, 20, 30, 60])
        w2 = random.choice([5, 10, 15, 20])
        formula_str = template.format(w1=w1, w2=w2)
        formulas.append(formula_str)
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
        return [{'formula': f, 'rationale': 'Randomly initialized from templates.'} for f in formulas]
        
    ideas = []
    parent_formulas = [p['formula'] for p in parents if 'formula' in p]
    
    for _ in range(num_ideas):
        method = random.random()
        if method < 0.4 and len(parent_formulas) >= 2:
            # Crossover
            p1, p2 = random.sample(parent_formulas, 2)
            child = crossover_formulas(p1, p2)
            ideas.append({'formula': child, 'rationale': f'Crossover of top factors.'})
        elif method < 0.8:
            # Mutation
            p = random.choice(parent_formulas)
            child = mutate_formula(p)
            ideas.append({'formula': child, 'rationale': f'Mutated from a top factor.'})
        else:
            # Fresh blood
            child = generate_initial_population(1)[0]
            ideas.append({'formula': child, 'rationale': 'Fresh exploration template.'})
            
    return ideas

# ── 5.b Main LLM Idea Generation Endpoint ──
def generate_ideas_with_llm(parents, num_ideas=1):
    """
    Given a list of parent dicts {'formula': '...', 'Score': ...},
    construct a prompt and call the LLM API to generate new formulas.
    """
    try:
        from research.auto_agent import query_llm, SYSTEM_PROMPT
    except ImportError:
        print("[LLM Error] Could not import query_llm from research.auto_agent. Falling back to EA.")
        return generate_ideas_from_parents(parents, num_ideas)
        
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if not parents:
        prompt = "We have no baseline factors yet. Generate a completely new, innovative alpha formula."
    else:
        # Build prompt using the history
        best = parents[0]
        prompt = f"Our current best factor is:\nFormula: {best.get('formula')}\nScore: {best.get('Score')}\nIC: {best.get('IC')}. "
        prompt += "Please generate a mutated or completely orthogonal new factor that could perform even better."
        
    history.append({"role": "user", "content": prompt})
    
    ideas = []
    for _ in range(num_ideas):
        try:
            res_json = query_llm(history)
            ideas.append({
                'formula': res_json.get('formula', "rank(close_trade_px)"),
                'rationale': res_json.get('thought_process', "Auto-generated by LLM based on parents.")
            })
        except Exception as e:
            print(f"[LLM Error] API request failed: {e}. Generating fallback idea...")
            ideas.extend(generate_ideas_from_parents(parents, 1))
            
    return ideas

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
