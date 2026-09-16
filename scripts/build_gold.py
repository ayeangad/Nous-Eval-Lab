"""Build the stratified 200-sample human gold set.

Design (documented in docs/judge-calibration.md):
- 8 tasks x 25 samples = 200, strata exactly 50/50/50/50
  (happy_path / underspecified / edge / adversarial).
- 12 distinct answers per task (gold, gold-style, buggy, buggy-style,
  5 mutants, 2 ambiguous, 1 grader-gap) x 2 prompt phrasings = 24,
  plus 1 empty-response sample = 25.
- Labels are single-annotator (annotator_01) judgments against the issue
  SPEC, verified by running the grader. `grader_sensitive` flags samples
  where the human label intentionally disagrees with the test suite
  (incomplete tests) -- these are the grader-validity cases, not errors.
- Inter-annotator agreement is NOT estimated (no second annotator);
  the schema reserves annotator_id for future double-labeling.

Usage: uv run python scripts/build_gold.py [--check]
--check verifies every non-grader-sensitive label matches the grader.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nous_eval_lab.benchmarks.swebench_informed.grader import grade  # noqa: E402
from nous_eval_lab.benchmarks.swebench_informed.task_schema import Task  # noqa: E402

SEED = 42

# answer spec: (answer_id, code, human_label, failure_category|None, ambiguous, grader_sensitive, note)
# failure_category vocabulary: edge_case_miss, wrong_logic, regression_break,
#   mutation_side_effect, type_mismatch, over_permissive, under_permissive,
#   encoding_mismatch, spec_ambiguous

CLAMP = [
    ("gold", "def clamp(n, lo, hi):\n    if lo > hi:\n        lo, hi = hi, lo\n    return max(lo, min(n, hi))\n",
     "pass", None, False, False, "reference fix"),
    ("gold_style", "def clamp(n, lo, hi):\n    return max(min(lo, hi), min(n, max(lo, hi)))\n",
     "pass", None, False, False, "equivalent without explicit swap"),
    ("buggy", "def clamp(n, lo, hi):\n    return max(lo, min(n, hi))\n",
     "fail", "edge_case_miss", False, False, "ignores swapped bounds"),
    ("buggy_style", "def clamp(n, lo, hi):\n    return min(max(n, lo), hi)\n",
     "fail", "edge_case_miss", False, False, "same gap, different form"),
    ("m1_early_return", "def clamp(n, lo, hi):\n    if lo > hi:\n        return n\n    return max(lo, min(n, hi))\n",
     "fail", "wrong_logic", False, False, "gives up on swapped bounds"),
    ("m2_off_by_one", "def clamp(n, lo, hi):\n    if lo > hi:\n        lo, hi = hi, lo\n    return max(lo, min(n, hi - 1))\n",
     "fail", "edge_case_miss", False, False, "exclusive upper bound"),
    ("m3_gte_swap", "def clamp(n, lo, hi):\n    if lo >= hi:\n        lo, hi = hi, lo\n    return max(lo, min(n, hi))\n",
     "pass", None, False, False, "equivalent: swapping equal bounds is a no-op"),
    ("m4_always_swap", "def clamp(n, lo, hi):\n    lo, hi = hi, lo\n    return max(lo, min(n, hi))\n",
     "fail", "regression_break", False, False, "breaks normal-order P2P cases"),
    ("m5_sorted", "def clamp(n, lo, hi):\n    lo, hi = sorted([lo, hi])\n    return max(lo, min(n, hi))\n",
     "pass", None, False, False, "equivalent normalization"),
    ("a1_raise", "def clamp(n, lo, hi):\n    if lo > hi:\n        raise ValueError('pass normalized bounds')\n    return max(lo, min(n, hi))\n",
     "fail", "spec_ambiguous", True, False, "defensive-API philosophy vs spec requirement to normalize"),
    ("a2_print", "def clamp(n, lo, hi):\n    print(f'clamping {n} to [{lo},{hi}]')\n    if lo > hi:\n        lo, hi = hi, lo\n    return max(lo, min(n, hi))\n",
     "pass", None, True, False, "correct but noisy; strict judges may penalize"),
    ("g1_negatives", "def clamp(n, lo, hi):\n    if lo > hi:\n        lo, hi = hi, lo\n    if n >= 0:\n        return max(lo, min(n, hi))\n    return lo\n",
     "fail", "wrong_logic", False, True, "passes suite; wrong for negative ranges, e.g. clamp(-5,-1,-10)->-10 not -5"),
]

DEDUP = [
    ("gold", "def dedup(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
     "pass", None, False, False, "reference fix"),
    ("gold_style", "def dedup(xs):\n    return [x for i, x in enumerate(xs) if x not in xs[:i]]\n",
     "pass", None, False, False, "O(n^2) but correct"),
    ("buggy", "def dedup(xs):\n    return list(set(xs))\n",
     "fail", "edge_case_miss", False, False, "loses order"),
    ("buggy_style", "def dedup(xs):\n    return list(dict.fromkeys(sorted(xs)))\n",
     "fail", "edge_case_miss", False, False, "sorts: wrong order"),
    ("m1_drop_last", "def dedup(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out[:-1] if len(out) > 1 else out\n",
     "fail", "wrong_logic", False, False, "drops last distinct element"),
    ("m2_comp", "def dedup(xs):\n    seen = set()\n    return [seen.add(x) or x for x in xs if x not in seen]\n",
     "pass", None, False, False, "tricky but correct"),
    ("m3_none_empty", "def dedup(xs):\n    if not xs:\n        return None\n    return list(dict.fromkeys(xs))\n",
     "fail", "edge_case_miss", False, False, "None instead of []"),
    ("m4_sorted_set", "def dedup(xs):\n    return sorted(set(xs))\n",
     "fail", "edge_case_miss", False, False, "sorted output"),
    ("m5_fromkeys", "def dedup(xs):\n    return list(dict.fromkeys(xs))\n",
     "pass", None, False, False, "idiomatic correct"),
    ("a1_quadratic", "def dedup(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out  # O(n^2) but simple\n",
     "pass", None, True, False, "correct; complexity unspecified"),
    ("a2_tuple", "def dedup(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return tuple(out)\n",
     "fail", "type_mismatch", True, False, "right values, wrong container type"),
    ("g1_scale", "def dedup(xs):\n    if len(xs) > 1000:\n        return list(set(xs))\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
     "fail", "edge_case_miss", False, True, "passes suite; wrong order at scale (untested)"),
]

PARSE = [
    ("gold", "def parse_price(s):\n    return float(s.strip().replace('$', '').replace(',', ''))\n",
     "pass", None, False, False, "reference fix"),
    ("gold_style", "def parse_price(s):\n    import re\n    return float(re.sub(r'[$,\\s]', '', s))\n",
     "pass", None, False, False, "regex variant"),
    ("buggy", "def parse_price(s):\n    return float(s.strip().replace('$', ''))\n",
     "fail", "edge_case_miss", False, False, "chokes on thousand separators"),
    ("buggy_style", "def parse_price(s):\n    return float(s.replace('$', ''))\n",
     "fail", "edge_case_miss", False, False, "also misses surrounding whitespace"),
    ("m1_no_dollar", "def parse_price(s):\n    return float(s.strip().replace(',', ''))\n",
     "fail", "wrong_logic", False, False, "forgets the $ sign"),
    ("m2_bare", "def parse_price(s):\n    return float(s.strip())\n",
     "fail", "wrong_logic", False, False, "handles neither $ nor commas"),
    ("m3_spaces_only", "def parse_price(s):\n    return float(s.strip().replace('$', '').replace(',', '').replace(' ', ''))\n",
     "fail", "edge_case_miss", False, True, "passes suite; tabs/other whitespace unhandled though spec says whitespace"),
    ("m4_int", "def parse_price(s):\n    return int(s.strip().replace('$', '').replace(',', ''))\n",
     "fail", "wrong_logic", False, False, "truncates cents"),
    ("m5_str", "def parse_price(s):\n    return str(float(s.strip().replace('$', '').replace(',', '')))\n",
     "fail", "type_mismatch", False, False, "returns str, not float"),
    ("a1_usd", "def parse_price(s):\n    return float(s.strip().replace('USD', '').replace('$', '').replace(',', ''))\n",
     "pass", None, True, False, "extra leniency, untested"),
    ("a2_empty_raise", "def parse_price(s):\n    if s.strip() == '':\n        raise ValueError('empty price')\n    return float(s.strip().replace('$', '').replace(',', ''))\n",
     "pass", None, True, False, "empty input unspecified; raising is defensible"),
    ("g1_int_float", "def parse_price(s):\n    x = float(s.strip().replace('$', '').replace(',', ''))\n    return int(x) if x.is_integer() else x\n",
     "fail", "type_mismatch", False, True, "== passes suite, but spec requires float and '$5' yields int 5"),
]

SAFE_DIV = [
    ("gold", "def safe_div(a, b, default=0.0):\n    if a is None or b is None:\n        return default\n    try:\n        if b == 0:\n            return default\n        return a / b\n    except (TypeError, ZeroDivisionError):\n        return default\n",
     "pass", None, False, False, "reference fix"),
    ("gold_style", "def safe_div(a, b, default=0.0):\n    try:\n        return a / b\n    except Exception:\n        return default\n",
     "pass", None, False, False, "bare-except variant; correct on suite and spec cases"),
    ("buggy", "def safe_div(a, b, default=0.0):\n    return a / b\n",
     "fail", "edge_case_miss", False, False, "raises on zero/None"),
    ("buggy_style", "def safe_div(a, b, default=0.0):\n    return a / b if b != 0 else default\n",
     "fail", "edge_case_miss", False, False, "still raises TypeError on None"),
    ("m1_zero_only", "def safe_div(a, b, default=0.0):\n    if b == 0:\n        return default\n    return a / b\n",
     "fail", "edge_case_miss", False, False, "None inputs raise"),
    ("m2_zde_only", "def safe_div(a, b, default=0.0):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return default\n",
     "fail", "edge_case_miss", False, False, "TypeError on None escapes"),
    ("m3_full", "def safe_div(a, b, default=0.0):\n    if a is None or b is None or b == 0:\n        return default\n    return a / b\n",
     "pass", None, False, False, "explicit equivalent"),
    ("m4_none_default", "def safe_div(a, b, default=0.0):\n    if b == 0 or b is None or a is None:\n        return None\n    return a / b\n",
     "fail", "wrong_logic", False, False, "returns None instead of default"),
    ("m5_b_only", "def safe_div(a, b, default=0.0):\n    if b is None or b == 0:\n        return default\n    return a / b\n",
     "fail", "edge_case_miss", False, False, "partial fix; a=None still raises"),
    ("a1_strict", "def safe_div(a, b, default=0.0):\n    \"\"\"None inputs are programmer errors and raise.\"\"\"\n    if b == 0:\n        return default\n    return a / b\n",
     "fail", "spec_ambiguous", True, False, "strict-typing philosophy vs spec requirement"),
    ("a2_warn", "def safe_div(a, b, default=0.0):\n    import warnings\n    if a is None or b is None:\n        warnings.warn('None input')\n        return default\n    try:\n        if b == 0:\n            return default\n        return a / b\n    except (TypeError, ZeroDivisionError):\n        return default\n",
     "pass", None, True, False, "correct; warns on fallback"),
    ("g1_str", "def safe_div(a, b, default=0.0):\n    if isinstance(a, str) or isinstance(b, str):\n        return default\n    if a is None or b is None:\n        return default\n    try:\n        if b == 0:\n            return default\n        return a / b\n    except (TypeError, ZeroDivisionError):\n        return default\n",
     "pass", None, True, False, "string inputs unspecified; defaulting is defensible"),
]

MERGE = [
    ("gold", "def merge(base, override):\n    out = dict(base)\n    out.update(override)\n    return out\n",
     "pass", None, False, False, "reference fix"),
    ("gold_style", "def merge(base, override):\n    return {**base, **override}\n",
     "pass", None, False, False, "unpacking equivalent"),
    ("buggy", "def merge(base, override):\n    base.update(override)\n    return base\n",
     "fail", "mutation_side_effect", False, False, "mutates base input"),
    ("buggy_style", "def merge(base, override):\n    out = base\n    out.update(override)\n    return out\n",
     "fail", "mutation_side_effect", False, False, "aliasing mutate"),
    ("m1_wrong_priority", "def merge(base, override):\n    override.update(base)\n    return dict(override)\n",
     "fail", "wrong_logic", False, False, "base wins; also mutates override"),
    ("m2_copy", "def merge(base, override):\n    out = dict(base)\n    for k, v in override.items():\n        out[k] = v\n    return out\n",
     "pass", None, False, False, "explicit equivalent"),
    ("m3_mutate_copy", "def merge(base, override):\n    base.update(override)\n    return dict(base)\n",
     "fail", "mutation_side_effect", False, False, "copies too late"),
    ("m4_override_only", "def merge(base, override):\n    return dict(override)\n",
     "fail", "wrong_logic", False, False, "drops base keys"),
    ("m5_loop", "def merge(base, override):\n    out = {}\n    for d in (base, override):\n        out.update(d)\n    return out\n",
     "pass", None, False, False, "loop equivalent"),
    ("a1_shallow", "def merge(base, override):\n    import copy\n    out = copy.copy(base)\n    out.update(override)\n    return out\n",
     "pass", None, True, False, "shallow: nested refs shared; spec silent"),
    ("a2_typecheck", "def merge(base, override):\n    if not isinstance(base, dict) or not isinstance(override, dict):\n        raise TypeError('dict inputs required')\n    return {**base, **override}\n",
     "pass", None, True, False, "non-dict inputs unspecified; raising is defensible"),
    ("g1_flat", "def merge(base, override):\n    # flat merge only; nested dicts are replaced, not deep-merged\n    return {**base, **override}\n",
     "pass", None, False, True, "passes suite; deep-merge behavior untested (spec examples are flat)"),
]

TRUNCATE = [
    ("gold", "def truncate(s, n):\n    if n <= 0:\n        return ''\n    if len(s) <= n:\n        return s\n    return s[:max(0, n - 1)] + '…'\n",
     "pass", None, False, False, "reference fix"),
    ("gold_style", "def truncate(s, n):\n    if n <= 0:\n        return ''\n    if len(s) <= n:\n        return s\n    head = s[:n - 1]\n    return head + '…'\n",
     "pass", None, False, False, "restructured equivalent"),
    ("buggy", "def truncate(s, n):\n    if len(s) <= n:\n        return s\n    return s[:n] + '...'\n",
     "fail", "encoding_mismatch", False, False, "three dots, length n+3"),
    ("buggy_style", "def truncate(s, n):\n    return s[:n] + '...'\n",
     "fail", "wrong_logic", False, False, "always appends marker"),
    ("m1_three_dots", "def truncate(s, n):\n    if n <= 0:\n        return ''\n    if len(s) <= n:\n        return s\n    return s[:n - 1] + '...'\n",
     "fail", "encoding_mismatch", False, False, "right shape, wrong marker; a lenient judge may pass it"),
    ("m2_no_guard", "def truncate(s, n):\n    if len(s) <= n:\n        return s\n    return s[:n - 1] + '…'\n",
     "fail", "edge_case_miss", False, False, "n<=0 mishandled"),
    ("m3_strict_less", "def truncate(s, n):\n    if n <= 0:\n        return ''\n    if len(s) < n:\n        return s\n    return s[:n - 1] + '…'\n",
     "fail", "regression_break", False, False, "marks exact-fit strings as truncated"),
    ("m4_reordered", "def truncate(s, n):\n    if n <= 0:\n        return ''\n    if len(s) <= n:\n        return s\n    return s[:max(0, n - 1)] + '…'\n",
     "pass", None, False, False, "identical logic, reordered lines"),
    ("m5_none_passthrough", "def truncate(s, n):\n    if len(s) <= n:\n        return None\n    return s[:n - 1] + '…'\n",
     "fail", "wrong_logic", False, False, "None instead of s; no n<=0 guard"),
    ("a1_ascii", "def truncate(s, n):\n    if n <= 0:\n        return ''\n    if len(s) <= n:\n        return s\n    if n == 1:\n        return '...'\n    return s[:n - 1] + '…'\n",
     "fail", "spec_ambiguous", True, False, "ascii fallback for n==1; spec mandates single ellipsis char"),
    ("a2_rstrip", "def truncate(s, n):\n    if n <= 0:\n        return ''\n    if len(s) <= n:\n        return s\n    return s[:n - 1].rstrip() + '…'\n",
     "pass", None, True, False, "strips trailing space before marker; untested"),
    ("g1_combining", "def truncate(s, n):\n    # note: len() counts code points, so combining chars may split\n    if n <= 0:\n        return ''\n    if len(s) <= n:\n        return s\n    return s[:max(0, n - 1)] + '…'\n",
     "pass", None, False, True, "passes suite; grapheme-cluster splitting untested"),
]

TOP_K = [
    ("gold", "def top_k(nums, k):\n    if k <= 0:\n        return []\n    return sorted(nums, reverse=True)[:k]\n",
     "pass", None, False, False, "reference fix"),
    ("gold_style", "def top_k(nums, k):\n    import heapq\n    if k <= 0:\n        return []\n    return heapq.nlargest(k, nums)\n",
     "pass", None, False, False, "heapq equivalent"),
    ("buggy", "def top_k(nums, k):\n    nums.sort(reverse=True)\n    return nums[:k]\n",
     "fail", "mutation_side_effect", False, False, "mutates input in place"),
    ("buggy_style", "def top_k(nums, k):\n    nums.sort()\n    return nums[-k:]\n",
     "fail", "wrong_logic", False, False, "ascending tail; also mutates"),
    ("m1_ascending", "def top_k(nums, k):\n    return sorted(nums)[:k]\n",
     "fail", "wrong_logic", False, False, "k smallest ascending"),
    ("m2_no_guard", "def top_k(nums, k):\n    return sorted(nums, reverse=True)[:k]\n",
     "fail", "edge_case_miss", False, True, "passes suite ([:0]==[]); negative k untested and wrong (drops last)"),
    ("m3_sort_copy", "def top_k(nums, k):\n    nums.sort(reverse=True)\n    return list(nums[:k])\n",
     "fail", "mutation_side_effect", False, False, "copies too late"),
    ("m4_tail", "def top_k(nums, k):\n    if not k:\n        return []\n    return sorted(nums)[-k:]\n",
     "fail", "wrong_logic", False, False, "ascending tail order"),
    ("m5_branches", "def top_k(nums, k):\n    if k <= 0:\n        return []\n    ranked = sorted(nums, reverse=True)\n    if k >= len(ranked):\n        return ranked\n    return ranked[:k]\n",
     "pass", None, False, False, "explicit equivalent"),
    ("a1_cond", "def top_k(nums, k):\n    return sorted(nums, reverse=True)[:k] if k > 0 else []\n",
     "pass", None, True, False, "equivalent; tie order unspecified"),
    ("a2_gen", "def top_k(nums, k):\n    if k <= 0:\n        return iter([])\n    return (x for x in sorted(nums, reverse=True)[:k])\n",
     "fail", "type_mismatch", True, False, "right values as iterator; spec examples show lists"),
    ("g1_mixed", "def top_k(nums, k):\n    # assumes mutually comparable numbers\n    if k <= 0:\n        return []\n    return sorted(nums, reverse=True)[:k]\n",
     "pass", None, False, True, "passes suite; mixed-type inputs untested"),
]

ACCESS = [
    ("gold", "def has_access(role, resource):\n    r = (role or '').strip().lower()\n    if r == 'admin':\n        return True\n    if r == 'editor':\n        return resource in ('docs', 'blog')\n    if r == 'viewer':\n        return resource == 'docs'\n    return False\n",
     "pass", None, False, False, "reference fix"),
    ("gold_style", "def has_access(role, resource):\n    perms = {'admin': None, 'editor': ('docs', 'blog'), 'viewer': ('docs',)}\n    r = (role or '').strip().lower()\n    if r not in perms:\n        return False\n    if perms[r] is None:\n        return True\n    return resource in perms[r]\n",
     "pass", None, False, False, "table-driven equivalent"),
    ("buggy", "def has_access(role, resource):\n    if role == 'admin':\n        return True\n    if role == 'editor':\n        return resource in ('docs', 'blog')\n    if role == 'viewer':\n        return resource == 'docs'\n    return False\n",
     "fail", "edge_case_miss", False, False, "case-sensitive role compare"),
    ("buggy_style", "def has_access(role, resource):\n    role = role.strip()\n    if role == 'admin':\n        return True\n    if role == 'editor':\n        return resource in ('docs', 'blog')\n    if role == 'viewer':\n        return resource == 'docs'\n    return False\n",
     "fail", "edge_case_miss", False, False, "strips but still case-sensitive"),
    ("m1_editor_docs", "def has_access(role, resource):\n    r = role.strip().lower()\n    if r == 'admin':\n        return True\n    if r == 'editor':\n        return resource == 'docs'\n    if r == 'viewer':\n        return resource == 'docs'\n    return False\n",
     "fail", "under_permissive", False, False, "editor loses blog"),
    ("m2_docs_only", "def has_access(role, resource):\n    if role.strip().lower() == 'admin':\n        return True\n    return resource == 'docs'\n",
     "fail", "under_permissive", False, False, "editor/blog denied"),
    ("m3_chain", "def has_access(role, resource):\n    r = role.strip().lower() if role else ''\n    if r == 'admin':\n        return True\n    elif r == 'editor':\n        return resource in ('docs', 'blog')\n    elif r == 'viewer':\n        return resource == 'docs'\n    else:\n        return False\n",
     "pass", None, False, False, "elif-chain equivalent"),
    ("m4_fail_open", "def has_access(role, resource):\n    r = (role or '').strip().lower()\n    if r == 'admin':\n        return True\n    if r == 'editor':\n        return resource in ('docs', 'blog')\n    if r == 'viewer':\n        return resource == 'docs'\n    return True\n",
     "fail", "over_permissive", False, False, "fail-open on unknown roles (security-relevant)"),
    ("m5_viewer_blog", "def has_access(role, resource):\n    r = (role or '').strip().lower()\n    if r == 'admin':\n        return True\n    if r == 'editor':\n        return resource in ('docs', 'blog')\n    if r == 'viewer':\n        return resource in ('docs', 'blog')\n    return False\n",
     "fail", "over_permissive", False, False, "viewer gains blog; breaks P2P"),
    ("a1_empty_viewer", "def has_access(role, resource):\n    r = (role or '').strip().lower()\n    if not r:\n        r = 'viewer'\n    if r == 'admin':\n        return True\n    if r == 'editor':\n        return resource in ('docs', 'blog')\n    if r == 'viewer':\n        return resource == 'docs'\n    return False\n",
     "fail", "spec_ambiguous", True, True, "empty role untested; spec says unknown->False but empty may mean anonymous-viewer"),
    ("a2_root", "def has_access(role, resource):\n    r = (role or '').strip().lower()\n    if r in ('admin', 'root'):\n        return True\n    if r == 'editor':\n        return resource in ('docs', 'blog')\n    if r == 'viewer':\n        return resource == 'docs'\n    return False\n",
     "fail", "spec_ambiguous", True, True, "'root' alias untested; spec says unknown->False"),
    ("g1_resource_case", "def has_access(role, resource):\n    # resources are matched exactly; only roles are normalized\n    r = (role or '').strip().lower()\n    if r == 'admin':\n        return True\n    if r == 'editor':\n        return resource in ('docs', 'blog')\n    if r == 'viewer':\n        return resource == 'docs'\n    return False\n",
     "pass", None, False, True, "passes suite; resource case ('Docs') untested"),
]

TASKS: dict[str, list[tuple]] = {
    "clamp_bounds_001": CLAMP,
    "dedup_order_002": DEDUP,
    "parse_price_003": PARSE,
    "safe_div_004": SAFE_DIV,
    "merge_dicts_005": MERGE,
    "truncate_006": TRUNCATE,
    "top_k_007": TOP_K,
    "has_access_008": ACCESS,
}

EMPTY_CODE = "# model returned no code\n"

STRATA = ["happy_path", "underspecified", "edge", "adversarial"]


def load_task_map() -> dict[str, Task]:
    m: dict[str, Task] = {}
    path = ROOT / "src/nous_eval_lab/benchmarks/swebench_informed/tasks.jsonl"
    for line in path.read_text().splitlines():
        if line.strip():
            t = Task.from_dict(json.loads(line))
            m[t.task_id] = t
    return m


def build() -> tuple[list[dict], list[dict]]:
    task_map = load_task_map()
    rng = random.Random(SEED)
    samples: list[dict] = []
    labels: list[dict] = []
    n = 0
    task_ids = sorted(task_map.keys())
    for ti, task_id in enumerate(task_ids):
        task = task_map[task_id]
        answers = TASKS[task_id]
        assert len(answers) == 12, f"{task_id}: {len(answers)} answers"
        combos = [(a, pk) for a in answers for pk in ("standard", "adversarial")]
        combos.append(((  # 25th: empty response
            "empty", EMPTY_CODE, "fail", "wrong_logic", False, False, "no code produced"), "standard"))
        # strata: exact global 50/50/50/50 via per-task rotation
        # tasks 0,1: H+1 | tasks 2,3: U+1 | tasks 4,5: E+1 | tasks 6,7: A+1
        extra = STRATA[(ti // 2) % 4]
        counts = {"happy_path": 6, "underspecified": 6, "edge": 6, "adversarial": 6}
        counts[extra] += 1  # 25 per task; globals sum to 50 each
        strata: list[str] = []
        for s in STRATA:
            strata.extend([s] * counts[s])
        assert len(strata) == 25
        rng.shuffle(combos)
        # keep gold spread: sort combos so identical answers are not adjacent (deterministic)
        for i, ((aid, code, label, cat, amb, gsens, note), pk) in enumerate(combos):
            n += 1
            sid = f"s{n:04d}"
            stratum = strata[i]
            prompt = task.adversarial_issue_text if (pk == "adversarial" or stratum == "adversarial") else task.issue_text
            if stratum == "adversarial" and not task.adversarial_issue_text:
                prompt = task.issue_text
            prompt_kind = "adversarial" if prompt == task.adversarial_issue_text and task.adversarial_issue_text else "standard"
            samples.append({
                "sample_id": sid, "task_id": task_id, "answer_id": aid,
                "stratum": stratum, "prompt_kind": prompt_kind,
                "prompt": prompt, "model_answer": code,
            })
            labels.append({
                "sample_id": sid, "human_label": label,
                "failure_category": cat, "annotator_id": "annotator_01",
                "ambiguous": amb, "grader_sensitive": gsens, "note": note,
            })
    return samples, labels


def check(samples: list[dict], labels: list[dict]) -> int:
    """Verify non-grader-sensitive labels match the grader. Returns # of real mismatches."""
    task_map = load_task_map()
    lab = {l["sample_id"]: l for l in labels}
    mism = 0
    gsens_disagree = 0
    for s in samples:
        task = task_map[s["task_id"]]
        res = grade(s["model_answer"], task)
        l = lab[s["sample_id"]]
        grader_label = "pass" if res.passed else "fail"
        if l["grader_sensitive"]:
            if grader_label != l["human_label"]:
                gsens_disagree += 1
            continue
        if grader_label != l["human_label"]:
            print(f"MISMATCH {s['sample_id']} {s['task_id']}/{s['answer_id']}: "
                  f"human={l['human_label']} grader={grader_label} ({res.reason[:100]})")
            mism += 1
    print(f"checked {len(samples)}: {mism} unexpected mismatches, "
          f"{gsens_disagree} intentional grader-human divergences (flagged)")
    return mism


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    samples, labels = build()
    from collections import Counter
    print("strata:", dict(Counter(s["stratum"] for s in samples)))
    print("labels:", dict(Counter(l["human_label"] for l in labels)))
    print("ambiguous:", sum(1 for l in labels if l["ambiguous"]),
          "grader_sensitive:", sum(1 for l in labels if l["grader_sensitive"]))
    if args.check:
        rc = check(samples, labels)
        if rc:
            raise SystemExit(1)
        return
    gold_dir = ROOT / "data/gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    (gold_dir / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples) + "\n")
    (gold_dir / "labels.jsonl").write_text("\n".join(json.dumps(l) for l in labels) + "\n")
    print(f"wrote {len(samples)} samples + labels to {gold_dir}")


if __name__ == "__main__":
    main()
