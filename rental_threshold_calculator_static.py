#!/usr/bin/env python3
# rental_threshold_calculator.py
# Dynamic threshold calculator for single-resource rental with terminal penalty constraint
# Parameters and empirical price distribution are embedded.
# Usage:
#   python rental_threshold_calculator.py
# Follow the prompts to input R (remaining days) and x (remaining inventory).
# Aug. 29 2025, Kong

#Horizon T, 决策周期长度,意味着你从今天到 T 天之后，会不断收到客户的报价。,在计算时，用它来估算总的潜在报价数（= 每天到达率 × T）。
T = 30

#Arrival rate per day 指每天平均会有多少个客户报价
#结合周期 T，可以得到预期总报价数：N=到达率×T
arrival_rate = 2 

#Cost c（成本基准价）,每个产品（例如一套房子）的“保本租金
c = 70  #cost 

#Salvage s（残值 / 期末处理价值）,指在整个周期结束时，如果还有没出租掉的产品，你能得到的残余价值。
#在DKS服务器租赁场景里通常是 0（没租出去就是白白浪费）
s = 0   #Salvage s 

#Empirical prices (16)（经验报价集合）
prices = [72, 72, 70, 68, 65, 62, 60, 60, 58, 58, 57, 56, 56, 55, 50, 50] #Uses your empirical price distribution， RMB K

'''
What the calculator does
Uses your empirical price distribution
[72, 72, 70, 68, 65, 62, 60, 60, 58, 58, 57, 56, 56, 55, 50, 50] (equal weight).
Target: leftover ≤ 3; failure if leftover > 5.
'''

def tail_prob(th, prices):
    n = len(prices)
    return sum(1 for p in prices if p >= th) / n

def recommend_threshold(R, x):
    unique_thresholds = sorted(set(prices), reverse=True)  # e.g., [72,70,68,65,62,60,58,57,56,55,50]
    threshold_tail = {th: tail_prob(th, prices) for th in unique_thresholds}
    if R <= 0:
        return None, 0.0, 0.0, x, "R<=0 (no time left)."
    if x <= 0:
        return None, 0.0, 0.0, 0.0, "No inventory."
    N_R = arrival_rate * R
    # Profit-first when x<=3
    if x <= 3:
        th = 70
        tp = threshold_tail.get(th, 0.0)
        exp_accepts = min(x, N_R * tp)
        L = x - exp_accepts
        return th, tp, exp_accepts, L, "x<=3: Profit-first (accept >=70)."
    q_req = max(0, x-3) / (2*R)
    # Find highest price threshold whose tail prob >= q_req
    candidate = None
    for th in unique_thresholds:  # 72 downwards
        tp = threshold_tail[th]
        if tp >= q_req:
            candidate = th
            break
    if candidate is None:
        candidate = min(unique_thresholds)
    th = candidate
    tp = threshold_tail[th]
    exp_accepts = min(x, N_R * tp)
    L = x - exp_accepts
    note = f"Base threshold from q_req: {th} (q_req={q_req:.4f})."
    # Safety check: ensure leftover <=5 by relaxing (lowering threshold) if needed
    sorted_low = sorted(unique_thresholds)  # ascending
    if L > 5:
        idx = sorted_low.index(th)
        j = idx - 1
        adjusted = False
        while j >= 0:
            th2 = sorted_low[j]
            tp2 = threshold_tail[th2]
            exp2 = min(x, N_R * tp2)
            L2 = x - exp2
            if L2 <= 5:
                th, tp, exp_accepts, L = th2, tp2, exp2, L2
                note += f" Safety adjusted to {th} to keep leftover <=5."
                adjusted = True
                break
            j -= 1
        if not adjusted:
            th = sorted_low[0]
            tp = threshold_tail[th]
            exp_accepts = min(x, N_R * tp)
            L = x - exp_accepts
            note += f" Even lowest threshold {th} could not achieve leftover<=5."
    return th, tp, exp_accepts, L, note

def main():
    try:
        R = float(input("Enter remaining days R: ").strip())
        x = float(input("Enter remaining inventory x: ").strip())
    except Exception as e:
        print("Invalid input.", e)
        return
    th, tp, exp_acc, L, note = recommend_threshold(R, x)
    print("--- Result ---")
    print(f"Suggested threshold: {th}")
    print(f"Tail probability (P>=threshold): {tp:.4f}")
    print(f"Expected accepts (capped by inventory): {exp_acc:.2f}")
    print(f"Expected leftover: {L:.2f}")
    print(f"Note: {note}")

if __name__ == "__main__":
    main()
