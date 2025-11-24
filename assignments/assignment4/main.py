import sys
sys.setrecursionlimit(1000000)

# --------------------
# 位打包：每行用 12bits 存区间 [l,r]
# 优势：状态只有一个整数，极快的 hash 与 copy
ROW_FIELD_BITS = 12
LR_BITS = 6
LR_MASK = (1 << LR_BITS) - 1
ROW_MASK = (1 << ROW_FIELD_BITS) - 1

def pack_lr(l, r):
    return (l << LR_BITS) | r

def unpack_lr(field):
    l = (field >> LR_BITS) & LR_MASK
    r = field & LR_MASK
    return l, r

def get_row_field(state_int, row_idx):
    off = row_idx * ROW_FIELD_BITS
    return (state_int >> off) & ROW_MASK

def set_row_field(state_int, row_idx, new_field):
    off = row_idx * ROW_FIELD_BITS
    mask = ROW_MASK << off
    return (state_int & ~mask) | ((new_field & ROW_MASK) << off)

# --------------------
# 输入
def parse_input(path="input.txt"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return rows
    n = int(lines[0])  # 实际不需要，用 rows 自身长度即可
    for ln in lines[1:]:
        parts = ln.split(",")
        rows.append([int(x) for x in parts])
    return rows

# --------------------
# 前缀和：用于快速区间求和（少用，但便捷）
def make_prefix(rows):
    prefs = []
    for row in rows:
        s = 0
        ps = [0]
        for v in row:
            s += v
            ps.append(s)
        prefs.append(ps)
    return prefs

def interval_sum(pref, l, r):
    return pref[r+1] - pref[l]

# --------------------
# 初始状态：所有行均为 [0, len(row)-1]
def build_initial_state(rows):
    s = 0
    for i, row in enumerate(rows):
        s = set_row_field(s, i, pack_lr(0, len(row)-1))
    return s

# --------------------
# negamax(alpha-beta):
# 返回：当前玩家看到的 (红分 - 蓝分)
# 关键优化：
#   - 使用整数状态 + 整形 key，缓存极快
#   - 历史启发减少深层搜索
def negamax(state_int, is_red, alpha, beta, rows, prefs, nrows, cache, history_table):
    key = (state_int << 1) | (1 if is_red else 0)
    if key in cache:
        return cache[key]

    # --- 快速终止：剩牌数为 0 或 1 时无需继续搜索 ---
    rem = 0
    last_val = 0
    get_field = get_row_field
    unpack = unpack_lr
    for i in range(nrows):
        field = get_field(state_int, i)
        l, r = unpack(field)
        if l <= r:
            cnt = r - l + 1
            rem += cnt
            if rem == 1:
                last_val = rows[i][l]
                break
    if rem == 0:
        cache[key] = 0
        return 0
    if rem == 1:
        res = last_val if is_red else -last_val
        cache[key] = res
        return res

    sign = 1 if is_red else -1

    # --- 生成所有可选动作，并根据即时收益 + 历史启发排序 ---
    moves = []
    append = moves.append
    rows_local = rows

    for i in range(nrows):
        field = get_field(state_int, i)
        l, r = unpack(field)
        if l > r:
            continue

        # left
        v_l = rows_local[i][l]
        new_state_l = set_row_field(state_int, i, pack_lr(l+1, r))
        id_l = (i << 1) | 0
        h_l = history_table.get(id_l, 0)
        append((sign * v_l + (h_l >> 1), new_state_l, v_l, id_l))

        # right
        if r > l:
            v_r = rows_local[i][r]
            new_state_r = set_row_field(state_int, i, pack_lr(l, r-1))
            id_r = (i << 1) | 1
            h_r = history_table.get(id_r, 0)
            append((sign * v_r + (h_r >> 1), new_state_r, v_r, id_r))

    moves.sort(reverse=True, key=lambda x: x[0])

    # --- alpha-beta 主循环 ---
    best = -10**15 if is_red else 10**15
    a, b = alpha, beta

    for _, new_state, card, move_id in moves:
        if is_red:
            val = card + negamax(new_state, False, a, b, rows, prefs, nrows, cache, history_table)
            if val > best:
                best = val
            if best > a:
                a = best
            if a >= b:
                history_table[move_id] = history_table.get(move_id, 0) + 1
                break
        else:
            val = negamax(new_state, True, a, b, rows, prefs, nrows, cache, history_table) - card
            if val < best:
                best = val
            if val < b:
                b = val
            if a >= b:
                history_table[move_id] = history_table.get(move_id, 0) + 1
                break

    cache[key] = best
    return best

# --------------------
# 主程序：只做初始层枚举 + 调 negamax
def main():
    rows = parse_input("input.txt")
    if not rows:
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write("第1行 左端 牌点数0\n")
            f.write("小红: 0 小蓝: 0\n")
        return

    prefs = make_prefix(rows)
    nrows = len(rows)
    state0 = build_initial_state(rows)
    total = sum(sum(r) for r in rows)

    cache = {}
    history_table = {}

    # --- 枚举红方第一步 ---
    get_field = get_row_field
    unpack = unpack_lr

    initial = []
    for i in range(nrows):
        field = get_field(state0, i)
        l, r = unpack(field)
        if l > r:
            continue
        # left
        v_l = rows[i][l]
        initial.append((v_l,
                        set_row_field(state0, i, pack_lr(l+1, r)),
                        i, 0))
        # right
        if r > l:
            v_r = rows[i][r]
            initial.append((v_r,
                            set_row_field(state0, i, pack_lr(l, r-1)),
                            i, 1))

    initial.sort(reverse=True, key=lambda x: x[0])

    best_move = None
    best_val = -10**15
    alpha = -10**15
    beta = 10**15

    for v, new_state, row, side in initial:
        val = v + negamax(new_state, False, alpha, beta,
                          rows, prefs, nrows, cache, history_table)
        if val > best_val:
            best_val = val
            best_move = (row, "左端" if side == 0 else "右端", v)
        if best_val > alpha:
            alpha = best_val

    diff = best_val
    red = (total + diff) // 2
    blue = total - red

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(f"第{best_move[0]+1}行 {best_move[1]} 牌点数{best_move[2]}\n")
        f.write(f"小红: {red} 小蓝: {blue}\n")

if __name__ == "__main__":
    main()
