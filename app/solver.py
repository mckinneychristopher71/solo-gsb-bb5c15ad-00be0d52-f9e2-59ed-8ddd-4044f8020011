"""共窑排程核心算法。

输入 1~14 件器物（唯一编号、正整数体积、允许烧成温度闭区间）与单窑容量，
输出最少窑次的装窑方案；窑次相同时选择整体字典序最小的方案。

同窑规则：
  * 总体积不超过单窑容量；
  * 各器物温度区间存在公共交集；
  * 烧成温度取交集下界（= 各区间下界的最大值）。

算法：
  1. 枚举全部 2^n 个器物子集，递推每个子集的总体积与温度交集，判定能否成窑。
  2. 位掩码 DP 求每个掩码所需最少窑次 dp[mask]。
  3. 贪心重建字典序最小方案：每一步在剩余器物中，找到包含剩余器物
     最小编号、且取出后剩余部分仍可在剩余窑次内装完的、
     器物编号序列字典序最小的子集作为下一窑。

限窑次接单（select_within_kiln_limit）复用上述子集表与 dp：
枚举全部子集，在 dp[mask] 不超过窑次上限的子集内，按「保留件数最多 →
保留总体积最大 → 保留编号升序列表字典序最小」择优；选中器物的装窑
仍调用 solve，沿用最少窑次与字典序排序。

已保存方案追加器物（append_to_plan）：原有器物与原窑次序固定不变，
每件新器物可并入某个仍有余量且温区相交的原窑，或进入追加窑次。
先枚举新器物的可并入子集（逐原窑：体积不超余量、温区与原窑区间相交），
再用「原窑序 × 新器物位掩码」DP 求最少追加窑次（f[j][mask] =
并入 j..m-1 号原窑后剩余新器物的最少窑次 = 新器物子集的 dp），
并按「原窑合并后编号列表 + 追加窑次列表」的整体字典序贪心重建；
追加窑次本身的装窑直接复用 solve（最少窑次、字典序最小）。

方案对比（compare_kiln_assignments）：以共同器物编号为范围，把来源与
目标窑次一对一配对（配对数取两边窑次数较小值），使留在对应窑中的共同
器物数最大。权重 w[i][j] = 来源窑 i 与目标窑 j 的共同器物数，问题即
二分图最大权匹配；用「按来源窑序 × 目标窑子集」的位掩码 DP 求最优值，
再按「（来源序号、目标序号）配对序列字典序最小」贪心重建配对。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class CapacityExceededError(ValueError):
    """单件器物体积超过单窑容量，无法烧成。"""


class TargetUnreachableError(ValueError):
    """容量取器物总体积时最少窑次仍超过目标窑次，目标不可达。"""


@dataclass(frozen=True)
class Ware:
    item_id: str
    volume: int
    temp_low: int
    temp_high: int


@dataclass(frozen=True)
class Kiln:
    """一窑：装入的器物编号（升序）、总体积、烧成温度。"""

    item_ids: tuple[str, ...]
    total_volume: int
    firing_temp: int


@dataclass(frozen=True)
class ProductionLimit:
    """限窑次接单结果：保留/延期器物编号（升序）与保留器物的装窑明细。"""

    kept_item_ids: tuple[str, ...]
    delayed_item_ids: tuple[str, ...]
    kilns: tuple[Kiln, ...]


@dataclass(frozen=True)
class PlanComparison:
    """两份方案的比较结果。

    pairs 为（来源窑次序号, 目标窑次序号）配对，按来源序号升序；
    各编号列表均按标准字符串字典序升序。
    """

    pairs: tuple[tuple[int, int], ...]
    stayed_item_ids: tuple[str, ...]        # 留在对应窑的共同器物
    moved_item_ids: tuple[str, ...]         # 换窑的共同器物（含未落在配对窑内者）
    only_in_source_item_ids: tuple[str, ...]
    only_in_target_item_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Tables:
    """子集枚举与位掩码 DP 的中间结果（位 i 对应编号升序后的第 i 件）。"""

    n: int
    ids: list[str]
    mask_volume: list[int]
    mask_temp_low: list[int]   # 交集下界（烧成温度）
    mask_temp_high: list[int]  # 交集上界
    valid: bytearray
    dp: list[int]


def _check_item_count(n: int) -> None:
    if n == 0:
        raise ValueError("至少需要一件器物")
    if n > 14:
        raise ValueError("器物数量不得超过 14 件")


def _build_tables(items: list[Ware], capacity: int) -> _Tables:
    """枚举 2^n 个子集，递推可成窑标记、子集总体积/温度交集与最少窑次 dp。

    :raises CapacityExceededError: 存在单件体积超过容量的器物。
    """

    n = len(items)
    _check_item_count(n)

    # 器物按编号字典序升序排列；位 i 对应排序后的第 i 件。
    ordered = sorted(items, key=lambda it: it.item_id)
    ids = [it.item_id for it in ordered]
    volumes = [it.volume for it in ordered]
    low = [it.temp_low for it in ordered]
    high = [it.temp_high for it in ordered]

    oversized = [it.item_id for it in ordered if it.volume > capacity]
    if oversized:
        raise CapacityExceededError(
            f"以下器物单件体积超过单窑容量 {capacity}，无法烧成: "
            + ", ".join(oversized)
        )

    size = 1 << n

    # ---- 1. 每个子集的总体积、温度交集与可行性（低位加一件递推） ----
    valid = bytearray(size)
    mask_volume = [0] * size
    mask_temp_low = [0] * size   # 交集下界（烧成温度）
    mask_temp_high = [0] * size  # 交集上界
    for mask in range(1, size):
        bit = mask & -mask  # 最低位
        i = bit.bit_length() - 1
        prev = mask ^ bit
        total = mask_volume[prev] + volumes[i]
        if prev == 0:
            inter_low, inter_high = low[i], high[i]
            feasible = volumes[i] <= capacity
        else:
            inter_low = max(mask_temp_low[prev], low[i])
            inter_high = min(mask_temp_high[prev], high[i])
            feasible = (
                total <= capacity
                and inter_low <= inter_high
                and bool(valid[prev])
            )
        mask_volume[mask] = total
        mask_temp_low[mask] = inter_low
        mask_temp_high[mask] = inter_high
        if feasible:
            valid[mask] = 1

    # ---- 2. 位掩码 DP：最少窑次 ----
    # dp[0] = 0；dp[mask] = 1 + min(dp[mask ^ g])，g 为 mask 的非空可行子集
    dp = [n + 1] * size
    dp[0] = 0
    for mask in range(1, size):
        best = dp[mask]
        sub = mask
        while sub:
            if valid[sub]:
                candidate = dp[mask ^ sub] + 1
                if candidate < best:
                    best = candidate
            sub = (sub - 1) & mask
        dp[mask] = best

    return _Tables(n, ids, mask_volume, mask_temp_low, mask_temp_high, valid, dp)


def solve(items: list[Ware], capacity: int) -> list[Kiln]:
    """求最少窑次、字典序最小的装窑方案。

    器物编号按标准字符串字典序排列（如 "1" < "10" < "2"），
    窑内编号升序、窑次间的整体字典序比较均以此为准。

    :raises CapacityExceededError: 存在单件体积超过容量的器物。
    :raises ValueError: 器物列表为空或超过 14 件。
    """

    tables = _build_tables(items, capacity)
    n = tables.n
    ids = tables.ids
    valid = tables.valid
    dp = tables.dp
    mask_volume = tables.mask_volume
    mask_temp_low = tables.mask_temp_low
    size = 1 << n

    # ---- 3. 贪心重建字典序最小方案 ----
    full = size - 1
    allowed_kilns = dp[full]
    left = full
    kilns: list[Kiln] = []

    while left:
        first = (left & -left).bit_length() - 1  # 剩余器物中编号最小者
        rest_bits = left ^ (1 << first)

        # 枚举包含 first 的所有子集，在“取出后剩余器物能在
        # allowed_kilns-1 窑内装完”的可行组中，取器物编号序列字典序最小者。
        best_group = 0
        best_tuple: tuple[str, ...] | None = None
        sub_rest = rest_bits
        while True:
            group = sub_rest | (1 << first)
            if valid[group] and dp[left ^ group] <= allowed_kilns - 1:
                members = tuple(
                    ids[i] for i in range(n) if (group >> i) & 1
                )
                if best_tuple is None or members < best_tuple:
                    best_tuple = members
                    best_group = group
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest_bits

        kilns.append(
            Kiln(
                item_ids=best_tuple,
                total_volume=mask_volume[best_group],
                firing_temp=mask_temp_low[best_group],
            )
        )
        left ^= best_group
        allowed_kilns -= 1

    return kilns


def append_to_plan(
    existing_kilns: Sequence[Sequence[str]],
    all_items: Sequence[Ware],
    new_item_ids: Sequence[str],
    capacity: int,
) -> list[Kiln]:
    """在保持原方案器物与原窑次序不变的前提下，把新器物并入方案。

    每件新器物可：
      * 并入某个原窑——并入后该窑总体积不超过 ``capacity``，且新器物
        与窑内全部器物的温度区间存在公共交集；
      * 或进入追加窑次（原窑之后新建的窑），同窑规则与 solve 一致。

    全局最优：先令**追加窑次数最少**；并列时按「原窑次序的各窑升序编号
    列表 + 追加窑次序的各窑升序编号列表」取整体字典序最小者。
    烧成温度取窑内温区交集下界。原器物在原窑内的相对次序、原窑之间的
    次序均保持不变（窑内新编号按字符串字典序插入升序位置）。

    :param existing_kilns: 原方案各窑的器物编号（列表顺序即原窑次序）
    :param all_items: 原方案器物与新器物的完整列表
    :param new_item_ids: 本次追加的新器物编号（与原方案编号互不重复）
    :param capacity: 沿用原方案的单窑容量
    :returns: 合并后的装窑明细（原窑在前、追加窑在后）。
    :raises CapacityExceededError: 存在单件体积超过容量的新器物。
    :raises ValueError: 新器物为空、合并后超过 14 件，或编号非法。
    """

    k = len(new_item_ids)
    if k == 0:
        raise ValueError("至少需要一件新器物")
    if len(existing_kilns) + k > 14 or len(all_items) > 14:
        raise ValueError("合并后器物数量不得超过 14 件")

    by_id = {it.item_id: it for it in all_items}
    if any(nid not in by_id for nid in new_item_ids) or len(by_id) != len(all_items):
        raise ValueError("器物编号必须唯一")
    existing_ids = {x for group in existing_kilns for x in group}
    if any(nid in existing_ids for nid in new_item_ids):
        raise ValueError("新器物编号不得与原方案器物重复")
    for group in existing_kilns:
        if not group:
            raise ValueError("原方案的窑次不得为空")

    # ---- 新器物子集表与最少窑次 dp（位 i 对应编号升序后的第 i 件新器物） ----
    new_wares = [by_id[nid] for nid in new_item_ids]
    tables = _build_tables(list(new_wares), capacity)
    size = 1 << k
    dp = tables.dp

    # ---- 每个原窑可并入的新器物子集 ----
    # addable[j][mask] = 1 表示 mask 可并入第 j 个原窑：
    # 并入后总体积不超容量，且温区与原窑区间（lo_j, hi_j）仍有交集。
    m = len(existing_kilns)
    addable: list[bytearray] = []
    for group in existing_kilns:
        members = [by_id[x] for x in group]
        used_volume = sum(w.volume for w in members)
        kiln_low = max(w.temp_low for w in members)
        kiln_high = min(w.temp_high for w in members)
        row = bytearray(size)
        for mask in range(1, size):
            if (
                used_volume + tables.mask_volume[mask] <= capacity
                and max(tables.mask_temp_low[mask], kiln_low)
                <= min(tables.mask_temp_high[mask], kiln_high)
            ):
                row[mask] = 1
        addable.append(row)

    # ---- f[j][mask]：第 j..m-1 号原窑处理完、仍未并入（即 mask）的
    #      新器物，最少需要多少追加窑次；并入方案不影响该值，仅取决于 mask。 ----
    INF = k + 1
    f: list[bytearray] = [bytearray() for _ in range(m + 1)]
    f[m] = bytearray(dp)
    for j in range(m - 1, -1, -1):
        cur = bytearray([INF]) * size
        nxt = f[j + 1]
        row = addable[j]
        for mask in range(size):
            best = nxt[mask]  # 新器物不并入原窑 j
            sub = mask
            while sub:
                if row[sub]:
                    cand = nxt[mask ^ sub]
                    if cand < best:
                        best = cand
                sub = (sub - 1) & mask
            cur[mask] = best
        f[j] = cur

    # ---- 贪心重建：逐原窑取「最优可达且合并后编号序列字典序最小」的并入
    #      子集。追加窑次排在所有原窑之后，原窑序列前缀较小者整体必较小，
    #      故在最少追加窑次约束下只需按原窑前缀字典序贪心；注意比较的是
    #      「原窑成员 ∪ 并入新器物」合并后的完整元组（不并入时原窑保持
    #      原样，可能比并入任何新编号都小）。 ----
    remaining = size - 1
    insert_masks: list[int] = [0] * m
    for j in range(m):
        nxt = f[j + 1]
        row = addable[j]
        original_members = tuple(sorted(existing_kilns[j]))
        best_tuple: tuple[str, ...] | None = None
        best_sub = -1
        sub = remaining
        while True:
            if (sub == 0 or row[sub]) and nxt[remaining ^ sub] == f[j][remaining]:
                members = tuple(sorted(
                    original_members
                    + tuple(tables.ids[i] for i in range(k) if (sub >> i) & 1)
                ))
                if best_tuple is None or members < best_tuple:
                    best_tuple = members
                    best_sub = sub
            if sub == 0:
                break
            sub = (sub - 1) & remaining
        assert best_sub >= 0
        insert_masks[j] = best_sub
        remaining ^= best_sub

    # ---- 合并后的原窑（原器物不动，新编号按升序插入）；best_tuple 已排序 ----
    kilns: list[Kiln] = []
    for j, group in enumerate(existing_kilns):
        ins = insert_masks[j]
        inserted = [tables.ids[i] for i in range(k) if (ins >> i) & 1]
        members = tuple(sorted((*group, *inserted)))
        member_wares = [by_id[x] for x in members]
        kilns.append(
            Kiln(
                item_ids=members,
                total_volume=sum(w.volume for w in member_wares),
                firing_temp=max(w.temp_low for w in member_wares),
            )
        )

    # ---- 剩余新器物进入追加窑次：装窑完全复用 solve（最少窑次、字典序最小） ----
    if remaining:
        leftover_ids = tuple(
            tables.ids[i] for i in range(k) if (remaining >> i) & 1
        )
        kilns.extend(solve([by_id[x] for x in leftover_ids], capacity))

    return kilns


def select_within_kiln_limit(
    items: list[Ware], capacity: int, max_kilns: int
) -> ProductionLimit:
    """限窑次接单：在最多 max_kilns 窑内选择可烧器物。

    使用给定单窑容量与各器物的体积、温度区间，枚举全部 2^n 个器物子集，
    仅考虑最少窑次 ``dp[mask] ≤ max_kilns`` 的子集，按以下优先级择优：
      1. 保留件数最多；
      2. 保留器物总体积最大；
      3. 仍并列时，保留编号升序列表（标准字符串字典序）最小。
    选中后装窑完全复用 solve：实际窑次取保留器物的最少窑次
    （可能少于 max_kilns），方案取整体字典序最小者。

    :raises CapacityExceededError: 存在单件体积超过容量的器物。
    :raises ValueError: 器物列表为空/超过 14 件，或 max_kilns 小于 1。
    """

    n = len(items)
    _check_item_count(n)
    if max_kilns < 1:
        raise ValueError("限窑次至少为 1")

    tables = _build_tables(items, capacity)
    size = 1 << n

    # ---- 在可于 max_kilns 窑内装完的子集中择优 ----
    best_mask = 0
    best_count = 0
    best_volume = 0
    best_ids: tuple[str, ...] | None = None

    for mask in range(1, size):
        if tables.dp[mask] > max_kilns:
            continue
        count = mask.bit_count()
        total = tables.mask_volume[mask]
        if best_ids is not None and (
            count < best_count
            or (count == best_count and total < best_volume)
        ):
            continue
        members = tuple(
            tables.ids[i] for i in range(n) if (mask >> i) & 1
        )
        if (
            best_ids is not None
            and count == best_count
            and total == best_volume
            and members >= best_ids
        ):
            continue
        best_mask = mask
        best_count = count
        best_volume = total
        best_ids = members

    # max_kilns >= 1，且 _build_tables 已排除单件超容器物，
    # 每件器物均可单独成窑，至少能保留一件。
    assert best_ids is not None and best_mask != 0

    # ---- 保留器物的装窑沿用现有最少窑次与方案排序 ----
    by_id = {it.item_id: it for it in items}
    kept_wares = [by_id[i] for i in best_ids]
    kilns = tuple(solve(kept_wares, capacity))

    kept_set = set(best_ids)
    delayed_ids = tuple(i for i in tables.ids if i not in kept_set)

    return ProductionLimit(
        kept_item_ids=best_ids,
        delayed_item_ids=delayed_ids,
        kilns=kilns,
    )


def min_capacity_for_target(
    items: list[Ware], target_kiln_count: int
) -> tuple[int, list[Kiln]]:
    """求使最少窑次不超过 target_kiln_count 的最小单窑容量。

    在 [最大单件体积, 器物总体积] 的整数容量范围内搜索：
    容量不小于最大单件体积时 solve 不会抛出 CapacityExceededError，
    容量达到总体积时体积约束名存实亡（仅剩温区规则）。

    窑次数随容量单调不增——容量放宽只会扩大可成窑子集集合，
    dp 值不会变大——故二分查找最小可行容量；
    每个容量点的判定与方案排序完全复用 solve。

    :returns: (最小容量, 该容量下最少窑次且字典序最小的装窑方案)。
    :raises TargetUnreachableError: 容量取总体积时最少窑次仍超过目标。
    :raises ValueError: 器物列表为空或超过 14 件，或目标窑次小于 1。
    """

    n = len(items)
    if n == 0:
        raise ValueError("至少需要一件器物")
    if n > 14:
        raise ValueError("器物数量不得超过 14 件")
    if target_kiln_count < 1:
        raise ValueError("目标窑次至少为 1")

    lo = max(it.volume for it in items)
    hi = sum(it.volume for it in items)

    # best_kilns 始终保持为 solve(items, hi) 的结果。
    best_kilns = solve(items, hi)
    if len(best_kilns) > target_kiln_count:
        raise TargetUnreachableError(
            f"容量取器物总体积 {hi} 时最少仍需 {len(best_kilns)} 窑，"
            f"超过目标 {target_kiln_count} 窑，目标不可达"
        )

    while lo < hi:
        mid = (lo + hi) // 2
        kilns = solve(items, mid)
        if len(kilns) <= target_kiln_count:
            hi = mid
            best_kilns = kilns
        else:
            lo = mid + 1

    return lo, best_kilns


def compare_kiln_assignments(
    source_kilns: Sequence[Sequence[str]],
    target_kilns: Sequence[Sequence[str]],
) -> PlanComparison:
    """比较两份排程方案，求来源与目标窑次的一对一配对。

    以共同器物编号为范围，配对数取两边窑次数的较小值，使留在对应窑中的
    共同器物数最大；最优解并列时，按来源序号排序的（来源序号、目标序号）
    配对序列取字典序最小。共同器物未落在配对窑内（所在来源窑或目标窑
    未被配对）也算换窑。

    记 w[i][j] 为来源窑 i 与目标窑 j 的共同器物数，则「留窑数最大」即
    二分图最大权匹配。窑次数不超过器物数（≤ 14），用位掩码 DP 求解：
    g[i][tmask] 表示来源窑 i..m 在目标窑已用集合 tmask 下、恰好配满
    pair_count 对时剩余配对最多还能保留的共同器物数（不可行为 -1）；
    再按配对序列字典序最小贪心重建。

    :param source_kilns: 来源方案各窑的器物编号（窑次序号从 1 开始，与列表顺序一致）
    :param target_kilns: 目标方案各窑的器物编号
    :raises ValueError: 任一方案的窑次列表为空。
    """

    m = len(source_kilns)
    k = len(target_kilns)
    if m == 0 or k == 0:
        raise ValueError("每份方案至少包含一个窑次")

    # 器物编号 -> 所在窑次序号（1 起）
    src_of: dict[str, int] = {}
    for idx, kiln in enumerate(source_kilns, start=1):
        for item_id in kiln:
            src_of[item_id] = idx
    tgt_of: dict[str, int] = {}
    for idx, kiln in enumerate(target_kilns, start=1):
        for item_id in kiln:
            tgt_of[item_id] = idx

    common = sorted(set(src_of) & set(tgt_of))

    # w[i][j]：来源窑 i 与目标窑 j（序号均从 1 开始）的共同器物数
    w = [[0] * (k + 1) for _ in range(m + 1)]
    for item_id in common:
        w[src_of[item_id]][tgt_of[item_id]] += 1

    pair_count = min(m, k)
    size = 1 << k
    full = size - 1

    # ---- 位掩码 DP：最大留窑数（恰好配满 pair_count 对才可行） ----
    g = [[-1] * size for _ in range(m + 2)]
    row_end = g[m + 1]
    for tmask in range(size):
        if tmask.bit_count() == pair_count:
            row_end[tmask] = 0
    for i in range(m, 0, -1):
        row = g[i]
        nxt = g[i + 1]
        wi = w[i]
        for tmask in range(size):
            best = nxt[tmask]  # 来源窑 i 不配对
            avail = full ^ tmask
            while avail:
                bit = avail & -avail
                j = bit.bit_length()  # bit 位 0 起 -> 目标窑序号 j（1 起）
                prev = nxt[tmask | bit]
                if prev >= 0:
                    cand = wi[j] + prev
                    if cand > best:
                        best = cand
                avail ^= bit
            row[tmask] = best

    # ---- 贪心重建字典序最小的配对序列 ----
    # 序列按来源序号升序；若某个最优完成配对了来源窑 i，则下一对必为
    # (i, j)，字典序恒小于任何以更大来源序号开头的序列，故逐窑取满足
    # 最优值的最小目标序号即可。
    pairs: list[tuple[int, int]] = []
    tmask = 0
    for i in range(1, m + 1):
        row = g[i]
        nxt = g[i + 1]
        wi = w[i]
        chosen_bit = 0
        chosen_j = 0
        avail = full ^ tmask
        while avail:
            bit = avail & -avail  # 目标序号升序枚举，首个可行者即最小
            j = bit.bit_length()
            prev = nxt[tmask | bit]
            if prev >= 0 and wi[j] + prev == row[tmask]:
                chosen_bit = bit
                chosen_j = j
                break
            avail ^= bit
        if chosen_bit:
            pairs.append((i, chosen_j))
            tmask |= chosen_bit
        # 否则来源窑 i 不配对（此时必有 g[i+1][tmask] == g[i][tmask]）

    # ---- 汇总留窑 / 换窑 / 单方独有编号（均按编号升序） ----
    pair_map = dict(pairs)
    stayed: list[str] = []
    moved: list[str] = []
    for item_id in common:
        if pair_map.get(src_of[item_id]) == tgt_of[item_id]:
            stayed.append(item_id)
        else:
            moved.append(item_id)

    only_in_source = sorted(set(src_of) - set(tgt_of))
    only_in_target = sorted(set(tgt_of) - set(src_of))

    return PlanComparison(
        pairs=tuple(pairs),
        stayed_item_ids=tuple(stayed),
        moved_item_ids=tuple(moved),
        only_in_source_item_ids=tuple(only_in_source),
        only_in_target_item_ids=tuple(only_in_target),
    )
