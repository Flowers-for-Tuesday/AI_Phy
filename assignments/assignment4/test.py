# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
批量运行测试案例，验证正确性并检测性能
"""

import os
import sys
import time
import subprocess


def run_test(test_id, solution_script='main.py'):
    """运行单个测试案例并验证输出"""
    test_dir = f'test_cases/test{test_id:02d}'
    
    if not os.path.exists(f'{test_dir}/input.txt'):
        print(f"测试案例 {test_id} 不存在")
        return False, 0
    
    # 把 input.txt 复制到当前目录
    os.system(f'cp {test_dir}/input.txt input.txt')
    
    # 运行 main.py
    start_time = time.time()
    try:
        result = subprocess.run(
            ['python3', solution_script],
            timeout=10,
            capture_output=True,
            text=True
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"测试 {test_id:02d}: 运行失败")
            print(result.stderr)
            return False, elapsed_time
        
        if not os.path.exists('output.txt'):
            print(f"测试 {test_id:02d}: 未生成 output.txt")
            return False, elapsed_time
        
        # ★ 读取 output.txt
        with open('output.txt', 'r', encoding='utf-8') as f:
            user_output = f.read().strip()
        
        # ★ 读取 answer.txt
        answer_path = f'{test_dir}/answer.txt'
        if not os.path.exists(answer_path):
            print(f"测试 {test_id:02d}: answer.txt 不存在，无法比对")
            return False, elapsed_time
        
        with open(answer_path, 'r', encoding='utf-8') as f:
            correct_output = f.read().strip()
        
        # ★ 比较
        if user_output == correct_output:
            print(f"测试 {test_id:02d}: ✓ 正确 (耗时: {elapsed_time:.2f}s)")
            return True, elapsed_time
        else:
            print(f"测试 {test_id:02d}: ✗ 错误 (耗时: {elapsed_time:.2f}s)")
            print("---- 输出差异 ----")
            print("你的 output.txt:")
            print(user_output)
            print("\n正确 answer.txt:")
            print(correct_output)
            print("------------------")
            return False, elapsed_time
            
    except subprocess.TimeoutExpired:
        print(f"测试 {test_id:02d}: ✗ 超时 (>10s)")
        return False, 10.0
    
    except Exception as e:
        print(f"测试 {test_id:02d}: 异常 - {e}")
        return False, 0.0


def main():
    print("=" * 60)
    print("开始批量测试")
    print("=" * 60)
    
    total_tests = 20
    passed = 0
    stats = []
    
    for test_id in range(1, total_tests + 1):
        success, elapsed_time = run_test(test_id)
        if success:
            passed += 1
            stats.append((test_id, elapsed_time))
        print()
    
    # 清理
    if os.path.exists('input.txt'):
        os.remove('input.txt')
    if os.path.exists('output.txt'):
        os.remove('output.txt')
    
    print("=" * 60)
    print(f"总计: {total_tests}")
    print(f"通过: {passed}")
    print(f"失败: {total_tests - passed}")
    print("=" * 60)
    
    if stats:
        print("\n性能统计:")
        print("-" * 60)
        print(f"{'测试ID':<10} {'时间(s)':<10}")
        print("-" * 60)
        for tid, t in stats:
            print(f"{tid:<10} {t:<10.2f}")
        print("-" * 60)
        print(f"总耗时: {sum(t for _, t in stats):.2f}s")
        print("=" * 60)


if __name__ == '__main__':
    main()
