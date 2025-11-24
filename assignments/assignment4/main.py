#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from functools import lru_cache

# -------------------- 配置 --------------------
sys.setrecursionlimit(20000)
INPUT_PATH = Path("input.txt")
OUTPUT_PATH = Path("output.txt")

# -------------------- 全局数据 --------------------
# 使用全局变量以供求解函数快速访问，避免参数传递
R0, R1, R2, R3 = [], [], [], []

# -------------------- 针对不同行数的极速求解器 --------------------
# 利用 lru_cache (C后端) 代替手写字典
# 移除循环，直接展开逻辑，减少解释器开销

@lru_cache(maxsize=None)
def solve_1(l0, r0):
    # 只有一行，其实就是贪心拿大的？不，还是得递归，因为是轮流拿
    # 但为了统一逻辑，使用同样的 minimax 结构
    if l0 > r0: return 0
    
    # 左边
    v1 = R0[l0] - solve_1(l0 + 1, r0)
    
    # 右边 (如果剩余超过1张)
    if l0 < r0:
        v2 = R0[r0] - solve_1(l0, r0 - 1)
        if v2 > v1:
            return v2
    return v1

@lru_cache(maxsize=None)
def solve_2(l0, r0, l1, r1):
    best = -99999999
    can_move = False

    # Row 0
    if l0 <= r0:
        can_move = True
        # Left
        v = R0[l0] - solve_2(l0 + 1, r0, l1, r1)
        if v > best: best = v
        # Right
        if l0 < r0:
            v = R0[r0] - solve_2(l0, r0 - 1, l1, r1)
            if v > best: best = v

    # Row 1
    if l1 <= r1:
        can_move = True
        # Left
        v = R1[l1] - solve_2(l0, r0, l1 + 1, r1)
        if v > best: best = v
        # Right
        if l1 < r1:
            v = R1[r1] - solve_2(l0, r0, l1, r1 - 1)
            if v > best: best = v

    return best if can_move else 0

@lru_cache(maxsize=None)
def solve_3(l0, r0, l1, r1, l2, r2):
    best = -99999999
    can_move = False

    # Row 0
    if l0 <= r0:
        can_move = True
        v = R0[l0] - solve_3(l0 + 1, r0, l1, r1, l2, r2)
        if v > best: best = v
        if l0 < r0:
            v = R0[r0] - solve_3(l0, r0 - 1, l1, r1, l2, r2)
            if v > best: best = v

    # Row 1
    if l1 <= r1:
        can_move = True
        v = R1[l1] - solve_3(l0, r0, l1 + 1, r1, l2, r2)
        if v > best: best = v
        if l1 < r1:
            v = R1[r1] - solve_3(l0, r0, l1, r1 - 1, l2, r2)
            if v > best: best = v

    # Row 2
    if l2 <= r2:
        can_move = True
        v = R2[l2] - solve_3(l0, r0, l1, r1, l2 + 1, r2)
        if v > best: best = v
        if l2 < r2:
            v = R2[r2] - solve_3(l0, r0, l1, r1, l2, r2 - 1)
            if v > best: best = v
            
    return best if can_move else 0

@lru_cache(maxsize=None)
def solve_4(l0, r0, l1, r1, l2, r2, l3, r3):
    best = -99999999
    can_move = False

    # Row 0
    if l0 <= r0:
        can_move = True
        v = R0[l0] - solve_4(l0 + 1, r0, l1, r1, l2, r2, l3, r3)
        if v > best: best = v
        if l0 < r0:
            v = R0[r0] - solve_4(l0, r0 - 1, l1, r1, l2, r2, l3, r3)
            if v > best: best = v

    # Row 1
    if l1 <= r1:
        can_move = True
        v = R1[l1] - solve_4(l0, r0, l1 + 1, r1, l2, r2, l3, r3)
        if v > best: best = v
        if l1 < r1:
            v = R1[r1] - solve_4(l0, r0, l1, r1 - 1, l2, r2, l3, r3)
            if v > best: best = v

    # Row 2
    if l2 <= r2:
        can_move = True
        v = R2[l2] - solve_4(l0, r0, l1, r1, l2 + 1, r2, l3, r3)
        if v > best: best = v
        if l2 < r2:
            v = R2[r2] - solve_4(l0, r0, l1, r1, l2, r2 - 1, l3, r3)
            if v > best: best = v
            
    # Row 3
    if l3 <= r3:
        can_move = True
        v = R3[l3] - solve_4(l0, r0, l1, r1, l2, r2, l3 + 1, r3)
        if v > best: best = v
        if l3 < r3:
            v = R3[r3] - solve_4(l0, r0, l1, r1, l2, r2, l3, r3 - 1)
            if v > best: best = v

    return best if can_move else 0

# -------------------- 主逻辑 --------------------

def main():
    global R0, R1, R2, R3
    
    # 1. 读取输入
    # 优化：快速读取，不使用复杂的 split 逻辑，假设输入格式规范
    try:
        with INPUT_PATH.open("r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return

    if not lines: return
    
    rows_data = []
    total_sum = 0
    
    # 第一行是 n，跳过
    for line in lines[1:]:
        if not line.strip(): continue
        # 快速解析整数
        parts = tuple(map(int, line.split(',')))
        if parts:
            rows_data.append(parts)
            total_sum += sum(parts)

    n_rows = len(rows_data)
    
    # 填充全局变量
    if n_rows >= 1: R0 = rows_data[0]
    if n_rows >= 2: R1 = rows_data[1]
    if n_rows >= 3: R2 = rows_data[2]
    if n_rows >= 4: R3 = rows_data[3]

    # 2. 根节点决策
    # 我们需要找出第一步最优解，同时复用 solver 的缓存
    
    best_val = -99999999
    best_move_str = ""
    
    # 辅助：根据行数调用对应的 solver
    # args: (l0, r0, l1, r1...)
    
    # 构造初始索引
    inits = []
    for r in rows_data:
        inits.append(0)           # l
        inits.append(len(r) - 1)  # r
        
    # 将初始索引转为 Tuple 方便切片传参，虽然下面是手动展开
    # 但为了第一步逻辑清晰，我们手动展开第一步
    
    def call_solver(current_idxs):
        if n_rows == 1: return solve_1(*current_idxs)
        if n_rows == 2: return solve_2(*current_idxs)
        if n_rows == 3: return solve_3(*current_idxs)
        if n_rows == 4: return solve_4(*current_idxs)
        return 0

    # 遍历每一个可能的起手
    for i in range(n_rows):
        l, r = inits[i*2], inits[i*2+1]
        
        # 如果行空 (l > r)，跳过
        if l > r: continue
        
        # --- 尝试左端 ---
        card_l = rows_data[i][l]
        
        # 构造拿走左边后的索引列表
        next_idxs = list(inits)
        next_idxs[i*2] += 1 # l + 1
        
        # 递归获得对手的最优分差 (opponent_diff)
        # 当前分差 = card - opponent_diff
        diff_l = card_l - call_solver(next_idxs)
        
        if diff_l > best_val:
            best_val = diff_l
            best_move_str = f"第{i+1}行 左端 牌点数{card_l}"
            
        # --- 尝试右端 (如果 l != r) ---
        if l < r:
            card_r = rows_data[i][r]
            
            next_idxs_r = list(inits)
            next_idxs_r[i*2+1] -= 1 # r - 1
            
            diff_r = card_r - call_solver(next_idxs_r)
            
            if diff_r > best_val:
                best_val = diff_r
                best_move_str = f"第{i+1}行 右端 牌点数{card_r}"

    # 3. 计算最终分
    # diff = Red - Blue
    # sum  = Red + Blue
    # Red = (sum + diff) / 2
    if best_move_str == "":
        # 没有任何牌的情况
        OUTPUT_PATH.write_text("第1行 左端 牌点数0\n小红: 0 小蓝: 0\n", encoding="utf-8")
    else:
        red_score = (total_sum + best_val) // 2
        blue_score = (total_sum - best_val) // 2
        OUTPUT_PATH.write_text(f"{best_move_str}\n小红: {red_score} 小蓝: {blue_score}\n", encoding="utf-8")

if __name__ == "__main__":
    main()