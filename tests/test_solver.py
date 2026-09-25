"""solver 单元测试：固定用例 + 小规模暴力枚举交叉验证。"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solver import (  # noqa: E402
    CapacityExceededError,
    TargetUnreachableError,
    Ware,
    append_to_plan,
    compare_kiln_assignments,
    min_capacity_for_target,
    select_within_kiln_limit,
    solve,
)


def _feasible(items_by_id, group, capacity):
    total = sum(items_by_id[i].volume for i in group)
    if total > capacity:
        return False
    lo = max(items_by_id[i].temp_low for i in group)
    hi = min(items_by_id[i].temp_high for i in group)
    return lo <= hi


def _brute_force(items, capacity):
    """枚举所有集合划分，返回 (最少窑次, 字典序最小方案)。"""
    ids = sorted(i.item_id for i in items)
    by_id = {i.item_id: i for i in items}
    best = None  # (窑次数, 方案元组)

    def partitions(s, groups):
        nonlocal best
        if not s:
            plan = tuple(sorted(tuple(sorted(g)) for g in groups))
            k = len(plan)
            if best is None or (k, plan) < (best[0], best[1]):
                best = (k, plan)
            return
        elem = s[0]
        rest = s[1:]
        # elem 放入已有某一组
        for gi in range(len(groups)):
            ng = groups[gi] | {elem}
            if _feasible(by_id, ng, capacity):
                new_groups = groups[:gi] + [ng] + groups[gi + 1:]
                partitions(rest, new_groups)
        # elem 单开一组
        if _feasible(by_id, {elem}, capacity):
            partitions(rest, groups + [{elem}])

    partitions(ids, [])
    return best


def _run(items, capacity):
    kilns = solve(items, capacity)
    plan = tuple(k.item_ids for k in kilns)
    return len(kilns), plan


def test_overcapacity_single_item():
    items = [Ware("a", 11, 100, 200)]
    try:
        solve(items, 10)
        assert False, "应抛出 CapacityExceededError"
    except CapacityExceededError:
        pass


def test_basic_capacity_split():
    # 每件 6，容量 10 -> 必须 3 窑
    items = [Ware(str(i), 6, 100, 200) for i in range(1, 4)]
    k, plan = _run(items, 10)
    assert k == 3
    assert plan == (("1",), ("2",), ("3",))


def test_temperature_intersection():
    # 1: [100,120] 2: [130,140] 3: [110,130]，容量任意
    # 1 与 2 不可同窑；1+3=[110,120] 可同窑(温度110)，2+3=[130,130](温度130)
    # 两种 2 窑方案：(1,3)(2) 与 (1)(2,3)；("1",) 是 ("1","3") 的前缀，
    # 字典序更小，故整体字典序最小方案为 [1] + [2,3]。
    items = [Ware("1", 1, 100, 120), Ware("2", 1, 130, 140), Ware("3", 1, 110, 130)]
    kilns = solve(items, 100)
    assert len(kilns) == 2
    assert kilns[0].item_ids == ("1",)
    assert kilns[0].firing_temp == 100
    assert kilns[1].item_ids == ("2", "3")
    assert kilns[1].firing_temp == 130


def test_firing_temp_is_intersection_lower_bound():
    items = [Ware("a", 1, 100, 300), Ware("b", 1, 150, 250), Ware("c", 1, 200, 400)]
    kilns = solve(items, 100)
    assert len(kilns) == 1
    assert kilns[0].firing_temp == 200  # max(100,150,200)
    assert kilns[0].total_volume == 3


def test_lex_tiebreak_example():
    # 构造窑次相同时的字典序选择：
    # 四件器物，每件体积 6，容量 10 -> 只能两两？不行，6+6>10 -> 每窑一件
    items = [Ware("b", 6, 1, 2), Ware("a", 6, 1, 2), Ware("c", 6, 1, 2)]
    k, plan = _run(items, 10)
    assert k == 3
    # 输入乱序，输出仍按编号升序、窑次按序列排序
    assert plan == (("a",), ("b",), ("c",))


def test_lex_tiebreak_pairing():
    # 容量 10，每件 5，温度全兼容 => 2 窑，两两组合。
    # 可能最优方案 (1,2)(3,4) / (1,3)(2,4) / (1,4)(2,3)，字典序最小为 (1,2)(3,4)
    items = [Ware(str(i), 5, 100, 200) for i in range(1, 5)]
    k, plan = _run(items, 10)
    assert k == 2
    assert plan == (("1", "2"), ("3", "4"))


def test_random_brute_force():
    rng = random.Random(42)
    for trial in range(300):
        n = rng.randint(1, 7)
        ids = [f"id{i}" for i in range(n)]
        capacity = rng.randint(5, 15)
        items = []
        for i in range(n):
            vol = rng.randint(1, 8)
            lo = rng.randint(900, 1000)
            hi = lo + rng.randint(0, 100)
            items.append(Ware(ids[i], vol, lo, hi))
        if any(it.volume > capacity for it in items):
            try:
                solve(items, capacity)
                assert False, f"trial {trial}: 应抛出 CapacityExceededError"
            except CapacityExceededError:
                continue
        got_k, got_plan = _run(items, capacity)
        exp_k, exp_plan = _brute_force(items, capacity)
        assert got_k == exp_k, f"trial {trial}: {got_k} != {exp_k}, {items}"
        assert got_plan == exp_plan, (
            f"trial {trial}: {got_plan} != {exp_plan}, items={items}, cap={capacity}"
        )


def test_n14_performance_and_correctness():
    # n=14 全兼容、容量只够单件 -> 14 窑，确保性能可接受
    items = [Ware(f"i{i:02d}", 5, 1000, 1000) for i in range(14)]
    import time

    t0 = time.time()
    k, plan = _run(items, 9)
    elapsed = time.time() - t0
    assert k == 14
    assert elapsed < 5.0, f"n=14 耗时 {elapsed:.2f}s"

    # 全部可同窑 -> 1 窑，烧成温度取交集下界
    items2 = [Ware(f"i{i:02d}", 1, 1000 + i, 2000) for i in range(14)]
    kilns = solve(items2, 1000)
    assert len(kilns) == 1
    assert kilns[0].firing_temp == 1013
    assert kilns[0].total_volume == 14


def test_min_capacity_basic():
    # 三件各 6、温度全兼容：3 窑需容量 6，2 窑需 12，1 窑需 18
    items = [Ware(str(i), 6, 100, 200) for i in range(1, 4)]
    cap, kilns = min_capacity_for_target(items, 3)
    assert cap == 6 and len(kilns) == 3
    cap, kilns = min_capacity_for_target(items, 2)
    assert cap == 12 and len(kilns) == 2
    # ("1",) 是 ("1","2") 的前缀，字典序更小，故首窑只装 "1"
    assert [k.item_ids for k in kilns] == [("1",), ("2", "3")]
    cap, kilns = min_capacity_for_target(items, 1)
    assert cap == 18 and len(kilns) == 1
    assert kilns[0].item_ids == ("1", "2", "3")
    assert kilns[0].firing_temp == 100
    assert kilns[0].total_volume == 18


def test_min_capacity_target_equals_item_count():
    # 目标 = 器物数时，最大单件体积即可（每件一窑）
    items = [Ware("a", 5, 100, 200), Ware("b", 9, 100, 200)]
    cap, kilns = min_capacity_for_target(items, 2)
    assert cap == 9 and len(kilns) == 2


def test_min_capacity_unreachable():
    # 温度区间互不兼容：任何容量下 1 与 2 都不可同窑
    items = [Ware("1", 3, 100, 120), Ware("2", 4, 130, 140)]
    try:
        min_capacity_for_target(items, 1)
        assert False, "应抛出 TargetUnreachableError"
    except TargetUnreachableError:
        pass
    # 目标 2 可达：最小容量为最大单件体积
    cap, kilns = min_capacity_for_target(items, 2)
    assert cap == 4 and len(kilns) == 2


def test_min_capacity_invalid_target():
    items = [Ware("a", 5, 100, 200)]
    try:
        min_capacity_for_target(items, 0)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_min_capacity_matches_linear_scan():
    # 随机小规模用例：与 [最大单件, 总体积] 全范围线性扫描交叉验证，
    # 同时校验返回方案与 solve 在该容量下的字典序最小方案一致
    rng = random.Random(7)
    for trial in range(200):
        n = rng.randint(1, 7)
        items = []
        for i in range(n):
            vol = rng.randint(1, 8)
            lo = rng.randint(900, 1000)
            hi = lo + rng.randint(0, 100)
            items.append(Ware(f"id{i}", vol, lo, hi))
        max_vol = max(it.volume for it in items)
        total = sum(it.volume for it in items)
        counts = {cap: len(solve(items, cap)) for cap in range(max_vol, total + 1)}
        for target in range(1, n + 1):
            feasible = [c for c, k in counts.items() if k <= target]
            if not feasible:
                try:
                    min_capacity_for_target(items, target)
                    assert False, f"trial {trial}: 应抛出 TargetUnreachableError"
                except TargetUnreachableError:
                    continue
            exp_cap = min(feasible)
            got_cap, kilns = min_capacity_for_target(items, target)
            assert got_cap == exp_cap, (
                f"trial {trial} target {target}: {got_cap} != {exp_cap}"
            )
            assert len(kilns) == counts[got_cap] <= target
            exp_kilns = solve(items, exp_cap)
            assert [k.item_ids for k in kilns] == [k.item_ids for k in exp_kilns]


# ---------------------------------------------------------------------------
# 限窑次接单 select_within_kiln_limit
# ---------------------------------------------------------------------------


def _brute_force_production_limit(items, capacity, max_kilns):
    """枚举全部 2^n 个子集，返回择优保留的编号元组（与接口相同的三级排序）。"""
    ordered = sorted(items, key=lambda it: it.item_id)
    by_id = {it.item_id: it for it in items}
    best = None  # (件数, 总体积)，并列时取编号升序列表字典序最小者
    for mask in range(1, 1 << len(ordered)):
        members = tuple(it.item_id for j, it in enumerate(ordered) if (mask >> j) & 1)
        if len(solve([by_id[i] for i in members], capacity)) > max_kilns:
            continue
        total = sum(by_id[i].volume for i in members)
        if (
            best is None
            or (len(members), total) > (best[0], best[1])
            or ((len(members), total) == (best[0], best[1]) and members < best[2])
        ):
            best = (len(members), total, members)
    return best[2]


def test_production_limit_basic_count():
    # 每件 6、容量 10 -> 每窑至多 1 件；上限 2 窑 -> 保留 2 件，
    # 件数与总体积均并列，取编号升序列表字典序最小者 ("1","2")。
    items = [Ware(str(i), 6, 100, 200) for i in range(1, 4)]
    result = select_within_kiln_limit(items, 10, 2)
    assert result.kept_item_ids == ("1", "2")
    assert result.delayed_item_ids == ("3",)
    assert len(result.kilns) == 2
    assert [k.item_ids for k in result.kilns] == [("1",), ("2",)]


def test_production_limit_one_kiln():
    # 温度互不兼容：1 与 2 不可同窑，1 窑内只能保留温度区间最大者。
    items = [
        Ware("1", 3, 100, 120),
        Ware("2", 4, 130, 140),
        Ware("3", 2, 110, 125),
    ]
    result = select_within_kiln_limit(items, 100, 1)
    assert result.kept_item_ids == ("1", "3")
    assert result.delayed_item_ids == ("2",)
    assert len(result.kilns) == 1
    assert result.kilns[0].item_ids == ("1", "3")
    assert result.kilns[0].firing_temp == 110
    assert result.kilns[0].total_volume == 5


def test_production_limit_volume_tiebreak():
    # 容量 10、上限 1 窑：体积 7/6/5 两两之和均超容，每件只能单独成窑，
    # 件数同为 1 时保留总体积最大者 -> "1"(7)。
    items = [
        Ware("1", 7, 100, 200),
        Ware("2", 6, 100, 200),
        Ware("3", 5, 100, 200),
    ]
    result = select_within_kiln_limit(items, 10, 1)
    assert result.kept_item_ids == ("1",)
    assert result.delayed_item_ids == ("2", "3")


def test_production_limit_id_tiebreak_standard_lex():
    # 标准字符串字典序：("10",) < ("2",)，故上限 1 窑、件数与体积并列时
    # 保留 "10" 而非 "2"。
    items = [
        Ware("2", 6, 100, 200),
        Ware("10", 6, 100, 200),
        Ware("3", 6, 100, 200),
    ]
    result = select_within_kiln_limit(items, 10, 1)
    assert result.kept_item_ids == ("10",)
    assert result.delayed_item_ids == ("2", "3")


def test_production_limit_actual_kilns_may_be_fewer():
    # 上限 3 窑但保留器物可全部装入 2 窑时，实际装窑取最少窑次。
    items = [Ware(str(i), 6, 100, 200) for i in range(1, 4)]
    result = select_within_kiln_limit(items, 10, 3)
    assert result.kept_item_ids == ("1", "2", "3")
    assert result.delayed_item_ids == ()
    assert len(result.kilns) == 3

    # 同样 3 件、容量 12（两两可同窑）、上限 3 窑 -> 全部保留且实际 2 窑
    items = [Ware(str(i), 6, 100, 200) for i in range(1, 4)]
    result = select_within_kiln_limit(items, 12, 3)
    assert result.kept_item_ids == ("1", "2", "3")
    assert len(result.kilns) == 2


def test_production_limit_max_equals_item_count_keeps_all():
    items = [
        Ware("a", 3, 100, 120),
        Ware("b", 5, 130, 140),
    ]
    result = select_within_kiln_limit(items, 100, 2)
    assert result.kept_item_ids == ("a", "b")
    assert result.delayed_item_ids == ()
    assert len(result.kilns) == 2


def test_production_limit_invalid_max_kilns():
    items = [Ware("a", 5, 100, 200)]
    try:
        select_within_kiln_limit(items, 10, 0)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_production_limit_oversized_item_still_errors():
    # 原方案保存时单件不会超容，此处保持 solver 语义一致：超容即报错。
    items = [Ware("a", 11, 100, 200)]
    try:
        select_within_kiln_limit(items, 10, 1)
        assert False, "应抛出 CapacityExceededError"
    except CapacityExceededError:
        pass


def test_production_limit_random_brute_force():
    # 随机小规模用例：保留集合与装窑明细同时交叉验证暴力枚举 / solve。
    rng = random.Random(99)
    for trial in range(300):
        n = rng.randint(1, 7)
        items = []
        for i in range(n):
            vol = rng.randint(1, 8)
            lo = rng.randint(900, 1000)
            hi = lo + rng.randint(0, 100)
            items.append(Ware(f"id{i}", vol, lo, hi))
        capacity = rng.randint(max(it.volume for it in items), 20)
        max_kilns = rng.randint(1, n)

        result = select_within_kiln_limit(items, capacity, max_kilns)
        exp_kept = _brute_force_production_limit(items, capacity, max_kilns)
        assert result.kept_item_ids == exp_kept, (
            f"trial {trial}: kept {result.kept_item_ids} != {exp_kept}, "
            f"items={items}, cap={capacity}, max_kilns={max_kilns}"
        )
        # 延期列表 = 全量编号 - 保留编号，均升序
        all_ids = tuple(sorted(it.item_id for it in items))
        kept_set = set(exp_kept)
        assert result.delayed_item_ids == tuple(i for i in all_ids if i not in kept_set)
        # 装窑明细与 solve 在保留子集上的结果一致（最少窑次 + 字典序）
        by_id = {it.item_id: it for it in items}
        exp_kilns = solve([by_id[i] for i in exp_kept], capacity)
        assert len(exp_kilns) <= max_kilns
        assert [k.item_ids for k in result.kilns] == [k.item_ids for k in exp_kilns]
        assert [k.total_volume for k in result.kilns] == [
            k.total_volume for k in exp_kilns
        ]
        assert [k.firing_temp for k in result.kilns] == [
            k.firing_temp for k in exp_kilns
        ]


# ---------------------------------------------------------------------------
# 方案对比 compare_kiln_assignments
# ---------------------------------------------------------------------------


def _brute_force_compare(source_kilns, target_kilns):
    """枚举全部一对一配对，返回 (配对序列, 留窑的共同器物集合)。

    配对序列按来源序号升序；最优解并列时取序列字典序最小者（与接口一致）。
    """
    from itertools import combinations, permutations

    m, k = len(source_kilns), len(target_kilns)
    pair_count = min(m, k)
    src_of = {x: i + 1 for i, kiln in enumerate(source_kilns) for x in kiln}
    tgt_of = {x: j + 1 for j, kiln in enumerate(target_kilns) for x in kiln}
    common = set(src_of) & set(tgt_of)

    best = None  # (留窑数取负, 配对序列, 留窑集合)
    for srcs in combinations(range(1, m + 1), pair_count):
        for tgts in permutations(range(1, k + 1), pair_count):
            pairs = tuple(zip(srcs, tgts))
            pair_map = dict(pairs)
            stayed = {
                x for x in common if pair_map.get(src_of[x]) == tgt_of[x]
            }
            key = (-len(stayed), pairs)
            if best is None or key < best[:2]:
                best = (key[0], pairs, stayed)
    return best[1], best[2]


def _check_compare_result(result, source_kilns, target_kilns):
    """校验对比结果与暴力枚举一致，且各编号列表升序、划分完备。"""
    src_ids = {x for kiln in source_kilns for x in kiln}
    tgt_ids = {x for kiln in target_kilns for x in kiln}
    common = src_ids & tgt_ids

    exp_pairs, exp_stayed = _brute_force_compare(source_kilns, target_kilns)
    assert result.pairs == exp_pairs, (
        f"pairs {result.pairs} != {exp_pairs}, "
        f"source={source_kilns}, target={target_kilns}"
    )
    # 配对数取两边窑次数较小值，来源序号升序且互不重复、目标序号互不重复
    assert len(result.pairs) == min(len(source_kilns), len(target_kilns))
    assert [s for s, _ in result.pairs] == sorted(s for s, _ in result.pairs)
    assert len({t for _, t in result.pairs}) == len(result.pairs)

    assert set(result.stayed_item_ids) == exp_stayed
    assert list(result.stayed_item_ids) == sorted(result.stayed_item_ids)
    # 换窑 = 共同器物 - 留窑；未落在配对窑内也算换窑
    assert set(result.moved_item_ids) == common - exp_stayed
    assert list(result.moved_item_ids) == sorted(result.moved_item_ids)
    # 单方独有编号
    assert set(result.only_in_source_item_ids) == src_ids - tgt_ids
    assert list(result.only_in_source_item_ids) == sorted(
        result.only_in_source_item_ids
    )
    assert set(result.only_in_target_item_ids) == tgt_ids - src_ids
    assert list(result.only_in_target_item_ids) == sorted(
        result.only_in_target_item_ids
    )


def test_compare_basic():
    # 来源 [a b] [c]，目标 [a] [b c]：唯一最优配对 (1,1)(2,2)，b 换窑
    result = compare_kiln_assignments([["a", "b"], ["c"]], [["a"], ["b", "c"]])
    assert result.pairs == ((1, 1), (2, 2))
    assert result.stayed_item_ids == ("a", "c")
    assert result.moved_item_ids == ("b",)
    assert result.only_in_source_item_ids == ()
    assert result.only_in_target_item_ids == ()


def test_compare_tiebreak_lex_by_source():
    # 来源 3 窑、目标 2 窑：(1,1)(3,2) 与 (2,1)(3,2) 留窑数同为 2，
    # 配对序列字典序最小者为 (1,1)(3,2)。
    result = compare_kiln_assignments(
        [["a"], ["b"], ["c"]], [["a", "b"], ["c"]]
    )
    assert result.pairs == ((1, 1), (3, 2))
    assert result.stayed_item_ids == ("a", "c")
    assert result.moved_item_ids == ("b",)


def test_compare_unpaired_kiln_counts_as_moved():
    # 目标仅 1 窑：配对 (1,1) 后 b 所在来源窑 2 未配对，未落在配对窑内算换窑
    result = compare_kiln_assignments([["a"], ["b"]], [["a", "b"]])
    assert result.pairs == ((1, 1),)
    assert result.stayed_item_ids == ("a",)
    assert result.moved_item_ids == ("b",)


def test_compare_no_common_items():
    # 无共同器物：留窑数恒为 0，字典序最小配对为 (1,1)
    result = compare_kiln_assignments([["x", "y"]], [["p", "q"], ["r"]])
    assert result.pairs == ((1, 1),)
    assert result.stayed_item_ids == ()
    assert result.moved_item_ids == ()
    assert result.only_in_source_item_ids == ("x", "y")
    assert result.only_in_target_item_ids == ("p", "q", "r")


def test_compare_identical_plans():
    # 同一方案对比：恒等配对，全部留窑
    kilns = [["a", "b"], ["c"]]
    result = compare_kiln_assignments(kilns, kilns)
    assert result.pairs == ((1, 1), (2, 2))
    assert result.stayed_item_ids == ("a", "b", "c")
    assert result.moved_item_ids == ()


def test_compare_standard_lex_order_of_ids():
    # 编号列表按标准字符串字典序升序（"10" < "2"）
    result = compare_kiln_assignments(
        [["2"], ["10"], ["3"]], [["2"], ["10"], ["3"]]
    )
    assert result.stayed_item_ids == ("10", "2", "3")


def test_compare_empty_kilns_rejected():
    try:
        compare_kiln_assignments([], [["a"]])
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_compare_random_brute_force():
    # 随机小规模用例：配对序列与全部编号列表同暴力枚举交叉验证
    rng = random.Random(2024)
    for trial in range(150):
        pool = [f"id{i}" for i in range(rng.randint(1, 6))]

        def random_plan():
            items = [x for x in pool if rng.random() < 0.7] or [
                pool[rng.randrange(len(pool))]
            ]
            rng.shuffle(items)
            kilns = [[] for _ in range(rng.randint(1, min(4, len(items))))]
            for x in items:
                kilns[rng.randrange(len(kilns))].append(x)
            return [sorted(kk) for kk in kilns if kk]

        source_kilns = random_plan()
        target_kilns = random_plan()
        result = compare_kiln_assignments(source_kilns, target_kilns)
        _check_compare_result(result, source_kilns, target_kilns)


# ---------------------------------------------------------------------------
# 已保存方案追加器物 append_to_plan
# ---------------------------------------------------------------------------


def _brute_force_append(existing_kilns, all_items, new_ids, capacity):
    """枚举新器物向原窑/追加窑的全部分配，返回 (追加窑次数, 整体方案元组)。

    原器物与原窑次序固定；每件新器物分配给某个原窑（可行时）或某个
    追加窑（组间不可行时新开一窑，组按编号序列排序）。
    先按追加窑次数最少、再按整体编号序列字典序最小择优，与接口规则一致。
    """
    by_id = {it.item_id: it for it in all_items}

    def feasible(members):
        return _feasible(by_id, members, capacity)

    # ---- 每个原窑可并入的新器物子集（位掩码） ----
    addable = []
    for group in existing_kilns:
        row = set()
        for mask in range(1, 1 << len(new_ids)):
            members = list(group) + [new_ids[i] for i in range(len(new_ids))
                                     if (mask >> i) & 1]
            if feasible(members):
                row.add(mask)
        addable.append(row)

    # ---- 枚举新器物到追加窑的无序划分（与 test 中集合划分同构） ----
    def appended_partitions(s):
        if not s:
            yield ()
            return
        elem = s[0]
        rest = s[1:]
        for parts in appended_partitions(rest):
            for gi in range(len(parts)):
                group = parts[gi] + (elem,)
                if feasible(group):
                    yield parts[:gi] + (tuple(sorted(group)),) + parts[gi + 1:]
            if feasible((elem,)):
                yield parts + ((elem,),)

    best = None  # (追加窑次数, 整体方案元组)
    m = len(existing_kilns)
    full = (1 << len(new_ids)) - 1

    # 枚举各原窑并入的子集（无序、不相交），剩余器物进追加窑
    def enum_original(j, used_mask, inserted):
        nonlocal best
        if j == m:
            left = full ^ used_mask
            left_ids = tuple(new_ids[i] for i in range(len(new_ids))
                             if (left >> i) & 1)
            orig_kilns = tuple(
                tuple(sorted(list(existing_kilns[jj]) + [
                    new_ids[i] for i in range(len(new_ids))
                    if (inserted[jj] >> i) & 1
                ]))
                for jj in range(m)
            )
            for groups in appended_partitions(left_ids):
                appended = tuple(sorted(groups))
                plan = orig_kilns + appended
                key = (len(appended), plan)
                if best is None or key < best:
                    best = key
            return
        # 原窑 j 不并入
        enum_original(j + 1, used_mask, inserted + (0,))
        avail = full ^ used_mask
        sub = avail
        while sub:
            if sub in addable[j]:
                enum_original(j + 1, used_mask | sub, inserted + (sub,))
            sub = (sub - 1) & avail

    enum_original(0, 0, ())
    assert best is not None
    return best


def _check_append_kiln_attrs(kilns, existing_kilns, all_items, capacity):
    """校验窑属性（体积/温区）与约束，并确认原器物与原窑次序不变。"""
    by_id = {it.item_id: it for it in all_items}
    for kiln in kilns:
        members = kiln.item_ids
        assert list(members) == sorted(members)
        assert kiln.total_volume == sum(by_id[x].volume for x in members)
        assert kiln.total_volume <= capacity
        assert kiln.firing_temp == max(by_id[x].temp_low for x in members)
        assert kiln.firing_temp <= min(by_id[x].temp_high for x in members)

    m = len(existing_kilns)
    for j, group in enumerate(existing_kilns):
        merged = kilns[j].item_ids
        # 原器物必须仍在第 j 窑，且相对次序不变
        pos = 0
        for x in group:
            while merged[pos] != x:
                pos += 1
        assert set(merged) >= set(group)


def test_append_fits_existing_no_new_kiln():
    # 原窑 [o1,o2](体积10满) [o3] 容量10：n1 可并入窑2，无需追加窑。
    existing = [["o1", "o2"], ["o3"]]
    items = [
        Ware("o1", 6, 100, 200),
        Ware("o2", 4, 100, 200),
        Ware("o3", 5, 100, 200),
        Ware("n1", 5, 150, 250),
    ]
    kilns = append_to_plan(existing, items, ["n1"], 10)
    assert len(kilns) == 2
    assert kilns[0].item_ids == ("o1", "o2")
    assert kilns[1].item_ids == ("n1", "o3")
    assert kilns[1].total_volume == 10
    assert kilns[1].firing_temp == 150  # max(100, 150)


def test_append_needs_appended_kiln():
    # 原窑 [o1,o2] 已满，n1 无处可并入 -> 追加 1 窑。
    existing = [["o1", "o2"]]
    items = [
        Ware("o1", 5, 100, 200),
        Ware("o2", 5, 100, 200),
        Ware("n1", 5, 100, 200),
    ]
    kilns = append_to_plan(existing, items, ["n1"], 10)
    assert [k.item_ids for k in kilns] == [("o1", "o2"), ("n1",)]
    assert kilns[1].firing_temp == 100


def test_append_appended_kiln_packs_new_items():
    # n1/n2 互不兼容（无温区交集），原窑均满 -> 追加 2 窑，按 solve 字典序排列。
    existing = [["o1"]]
    items = [
        Ware("o1", 10, 1000, 1100),
        Ware("n1", 1, 100, 120),
        Ware("n2", 1, 130, 140),
    ]
    kilns = append_to_plan(existing, items, ["n1", "n2"], 10)
    assert [k.item_ids for k in kilns] == [("o1",), ("n1",), ("n2",)]


def test_append_temperature_intersection_shrinks():
    # 原窑 [o1](100~300)：n1(200~400) 可并入（交集 200~300，烧成 200），
    # n2(310~400) 温区不交 -> n2 进追加窑；即便并入 n1 后 n2 也无法同窑。
    existing = [["o1"]]
    items = [
        Ware("o1", 1, 100, 300),
        Ware("n1", 1, 200, 400),
        Ware("n2", 1, 310, 400),
    ]
    kilns = append_to_plan(existing, items, ["n1", "n2"], 10)
    assert [k.item_ids for k in kilns] == [("n1", "o1"), ("n2",)]
    assert kilns[0].firing_temp == 200
    assert kilns[0].total_volume == 2
    assert kilns[1].firing_temp == 310


def test_append_tiebreak_split_lex():
    # n1 可并入窑1或窑2（均不增窑次）：
    #   (n1,o1)(o2) 整体序列 vs (o1)(n1,o2) -> 首窑 ("n1","o1") < ("o1",)，
    #   取前者。
    existing = [["o1"], ["o2"]]
    items = [
        Ware("o1", 5, 100, 200),
        Ware("o2", 5, 100, 200),
        Ware("n1", 5, 100, 200),
    ]
    kilns = append_to_plan(existing, items, ["n1"], 10)
    assert [k.item_ids for k in kilns] == [("n1", "o1"), ("o2",)]
    # 若新编号字典序大于原窑成员，并入窑1会使 ("o1",) 前缀变长反而更大，
    # 故最优为并入窑2：(o1)(o2,z9)。
    items2 = [
        Ware("o1", 5, 100, 200),
        Ware("o2", 5, 100, 200),
        Ware("z9", 5, 100, 200),
    ]
    kilns2 = append_to_plan(existing, items2, ["z9"], 10)
    assert [k.item_ids for k in kilns2] == [("o1",), ("o2", "z9")]


def test_append_split_across_kilns():
    # 容量 10、原窑各一件体积 6：n1/n2 各体积 4，可分别并入两窑。
    # 编号 n1<n2<o1<o2（"n1"<"n2"<"o1"），枚举所有 0/1 追加窑方案：
    # 全不并入 -> 追加窑 [n1,n2]，追加 1 窑；
    # 各并入一窑 -> (n1,o1)(n2,o2)，追加 0 窑，最优。
    existing = [["o1"], ["o2"]]
    items = [
        Ware("o1", 6, 100, 200),
        Ware("o2", 6, 100, 200),
        Ware("n1", 4, 100, 200),
        Ware("n2", 4, 100, 200),
    ]
    kilns = append_to_plan(existing, items, ["n1", "n2"], 10)
    assert [k.item_ids for k in kilns] == [("n1", "o1"), ("n2", "o2")]
    assert kilns[0].total_volume == 10 and kilns[1].total_volume == 10


def test_append_preserves_original_order():
    # 三件原窑各容量充足、温度全兼容：n 可并入任一窑且均为 0 追加窑。
    # ("a",) 是 ("a","n") 的前缀更小，故尽量后置并入：
    # 前两窑保持原样，n 并入末窑 c。
    existing = [["a"], ["b"], ["c"]]
    items = [
        Ware("a", 1, 100, 200),
        Ware("b", 1, 100, 200),
        Ware("c", 1, 100, 200),
        Ware("n", 1, 100, 200),
    ]
    kilns = append_to_plan(existing, items, ["n"], 10)
    assert [k.item_ids for k in kilns] == [("a",), ("b",), ("c", "n")]
    # 新编号字典序小于全部原成员时，则并入首窑使整体序列最小
    items2 = items[:3] + [Ware("0a", 1, 100, 200)]
    kilns2 = append_to_plan(existing, items2, ["0a"], 10)
    assert [k.item_ids for k in kilns2] == [("0a", "a"), ("b",), ("c",)]


def test_append_original_totals_recomputed():
    # 原窑记录的体积/温度可能只存了结果，属性必须从器物重新汇总且新窑也合规。
    existing = [["p1", "p3"]]
    items = [
        Ware("p1", 3, 100, 300),
        Ware("p3", 4, 150, 250),
        Ware("x1", 2, 200, 400),
    ]
    kilns = append_to_plan(existing, items, ["x1"], 100)
    assert kilns[0].item_ids == ("p1", "p3", "x1")
    assert kilns[0].total_volume == 9
    assert kilns[0].firing_temp == 200  # max(100, 150, 200)


def test_append_oversized_new_item_errors():
    existing = [["o1"]]
    items = [Ware("o1", 1, 100, 200), Ware("big", 11, 100, 200)]
    try:
        append_to_plan(existing, items, ["big"], 10)
        assert False, "应抛出 CapacityExceededError"
    except CapacityExceededError:
        pass


def test_append_invalid_counts_and_ids():
    try:
        append_to_plan([["o1"]], [Ware("o1", 1, 100, 200)], [], 10)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    # 合并后超过 14 件
    existing = [[f"o{i}"] for i in range(13)]
    items = [Ware(f"o{i}", 1, 100, 200) for i in range(13)] + [
        Ware("n1", 1, 100, 200),
        Ware("n2", 1, 100, 200),
    ]
    try:
        append_to_plan(existing, items, ["n1", "n2"], 10)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    # 新编号与原器物重复
    existing = [["o1"]]
    items = [Ware("o1", 1, 100, 200)]
    try:
        append_to_plan(existing, items, ["o1"], 10)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_append_all_new_fit_one_appended_kiln():
    # 原窑均满，但新器物彼此兼容且一窑可容 -> 恰好追加 1 窑。
    existing = [["o1", "o2"]]
    items = [
        Ware("o1", 5, 100, 200),
        Ware("o2", 5, 100, 200),
        Ware("n1", 3, 100, 200),
        Ware("n2", 3, 100, 200),
    ]
    kilns = append_to_plan(existing, items, ["n1", "n2"], 10)
    assert [k.item_ids for k in kilns] == [("o1", "o2"), ("n1", "n2")]


def test_append_n14_performance():
    # 13 个原窑各一件（容量 10、体积 9）+ 1 件新器物：可并入任一窑。
    existing = [[f"o{i:02d}"] for i in range(13)]
    items = [Ware(f"o{i:02d}", 9, 1000, 1200) for i in range(13)]
    items.append(Ware("n0", 1, 1000, 1100))
    import time

    t0 = time.time()
    kilns = append_to_plan(existing, items, ["n0"], 10)
    elapsed = time.time() - t0
    assert len(kilns) == 13  # 0 追加窑
    # "n0" < "o00"，并入窑0 使首窑变为 ("n0","o00")，全局字典序最小。
    assert kilns[0].item_ids == ("n0", "o00")
    assert kilns[1].item_ids == ("o01",)
    assert elapsed < 5.0, f"n=14 追加耗时 {elapsed:.2f}s"

    # 7 原窑各满（体积 10），7 件新器物体积各 1 不可并入原窑、
    # 但彼此可同窑 -> 恰好追加 1 窑
    existing = [[f"o{i}"] for i in range(7)]
    items = [Ware(f"o{i}", 10, 100, 200) for i in range(7)]
    items += [Ware(f"n{i}", 1, 100, 200) for i in range(7)]
    t0 = time.time()
    kilns = append_to_plan(existing, items, [f"n{i}" for i in range(7)], 10)
    elapsed = time.time() - t0
    assert len(kilns) == 8
    assert [k.item_ids for k in kilns[:7]] == [(f"o{i}",) for i in range(7)]
    assert kilns[7].item_ids == tuple(f"n{i}" for i in range(7))
    assert elapsed < 5.0, f"n=14 追加耗时 {elapsed:.2f}s"


def test_append_random_brute_force():
    # 随机小规模用例：追加窑次数与整体方案同暴力枚举交叉验证。
    rng = random.Random(2026)
    for trial in range(250):
        n_old = rng.randint(1, 5)
        n_new = rng.randint(1, min(4, 7 - n_old))
        capacity = rng.randint(5, 14)

        old_items = []
        for i in range(n_old):
            vol = rng.randint(1, 8)
            lo = rng.randint(900, 1000)
            hi = lo + rng.randint(0, 100)
            old_items.append(Ware(f"o{i}", vol, lo, hi))
        # 单件超容时 solve 已会报错；追加语义同样报错，跳过
        if any(it.volume > capacity for it in old_items):
            continue
        # 先用 solve 得到一份原方案（保证窑次可行）
        old_kilns_obj = solve(old_items, capacity)
        existing = [list(k.item_ids) for k in old_kilns_obj]

        new_items = []
        for i in range(n_new):
            vol = rng.randint(1, 8)
            lo = rng.randint(900, 1000)
            hi = lo + rng.randint(0, 100)
            new_items.append(Ware(f"n{i}", vol, lo, hi))

        all_items = old_items + new_items
        new_ids = [it.item_id for it in new_items]

        if any(it.volume > capacity for it in new_items):
            try:
                append_to_plan(existing, all_items, new_ids, capacity)
                assert False, f"trial {trial}: 应抛出 CapacityExceededError"
            except CapacityExceededError:
                continue

        kilns = append_to_plan(existing, all_items, new_ids, capacity)
        exp_appended, exp_plan = _brute_force_append(
            existing, all_items, new_ids, capacity
        )

        got_plan = tuple(k.item_ids for k in kilns)
        # 追加窑次数最少
        got_appended = len(kilns) - len(existing)
        assert got_appended == exp_appended, (
            f"trial {trial}: 追加窑次 {got_appended} != {exp_appended}, "
            f"items={all_items}, cap={capacity}"
        )
        # 并列时整体字典序最小
        assert got_plan == exp_plan, (
            f"trial {trial}: {got_plan} != {exp_plan}, "
            f"items={all_items}, cap={capacity}, existing={existing}"
        )
        _check_append_kiln_attrs(kilns, existing, all_items, capacity)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n全部 {len(fns)} 个测试通过")