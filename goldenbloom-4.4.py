#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
goldenbloom —— 炸金花 动态贝叶斯博弈模拟器
====================================================================
每位参与者在自己的决策点:
  1) 用贝叶斯公式,根据"发牌概率 + 已观察到的行动 + 弃牌亮牌 + 开牌胜负"更新对每个对手手牌的信念 μ;
  2) 对每个可行行动(弃牌 / 跟注 / 加注到某个整数下注额 / 开牌并指定对手)×(看牌 / 闷牌),
     从信念中抽样对手手牌,按行为策略模型 σ 推演到终局,估计期望净收益
        V(a|I) = E[获得的底池 - 本人总投入];
  3) 选择 V 最大的行动:a* ∈ argmax V(a|I)。
行为策略模型 σ(a | 状态, 手牌强度) 由多局自博弈的经验频率迭代学习,逼近 (σ*, μ*) 均衡条件。

规则(按最新约定):
  - 52 张牌,2~17 人(默认 5 人),每人 3 张;A=14;
    豹子>同花顺>同花>顺子>对子>散牌;QKA 算顺子,A23 不算;花色不分大小;
    同牌型按三张点数从大到小依次比较;散牌 235 胜任何豹子;点数完全相同,提出开牌者负。
  - 每人先交相同底注(默认 1),随机确定先手,按座位顺序轮流行动。
  - 底注、跟注、下注、加注、开牌,日志和参数里的数字一律指"实际支付金额",不再另算倍数。
      闷牌者(未看牌)按"闷牌价"支付;看牌者必须支付闷牌价的 2 倍(--mult 可调),
      因此看牌者的支付额恒为 mult 的整数倍(保证闷牌价为整数)。
      首轮先手自定下注;之后每个人的闷牌价不得低于上一家(看牌者支付额随之不低于 2×)。
      例:底注 5 人底池 5;庄家闷牌支付 1 → 底池 6;下一位若已看牌须支付 2 → 底池 8;
          再下一位若闷牌只付 1 → 底池 9;若某看牌者"加注至 10",则他支付 10,
          之后看牌者每次付 10,闷牌者每次付 5。
      内部用 base 表示当前闷牌价,看牌价=mult*base;--max-bet 是单次支付上限(任何玩家)。
  - 动作:弃牌 / 跟注(x 不变) / 加注(x 提高) / 开牌。
      开牌:支付当前应付金额,指定一名在局玩家比牌,对方不得拒绝且无需付费;
            亮牌仅限开牌双方,其他人只知道败者出局;败者退出,胜者继续。
      弃牌:退出本局,已投入不退,弃牌会向全场亮牌。
      看牌免费,看过后不能恢复闷牌;开牌双方互相亮牌,因此闷牌者开牌后也等于看过自己的牌。
  - 每次下注独立支付,不追补差额。最多行动 20 次(弃牌/开牌也算),
    达到上限后按原座位顺序轮流免费指定对手开牌,直至只剩一人。
  - 不抽水:净收益 = 获得的底池 - 本人总投入。

用法:
  python goldenbloom.py --games 20 --train 30
  python goldenbloom.py --games 3 --verbose
  python goldenbloom.py --baseline          # 0 号座位=期望收益最大化,其他=朴素阈值策略
  python goldenbloom.py --stats             # 牌型概率
  python goldenbloom.py --selftest          # 校验下注/底池规则
依赖: numpy
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 多进程时避免 BLAS 线程互相抢 CPU
import argparse
import itertools
import multiprocessing as mp
import queue
import random
import sys
import time

import numpy as np

# =============================== 牌与牌型 ===============================
RANK_NAME = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
SUITS = '♠♥♣♦'
TYPE_NAME = {6: '豹子', 5: '同花顺', 4: '同花', 3: '顺子', 2: '对子', 1: '散牌'}


def card_str(c):
    r = c // 4 + 2
    return f"{SUITS[c % 4]}{RANK_NAME.get(r, str(r))}"


def evaluate(cards):
    """返回 (牌型T, x1, x2, x3),x1>=x2>=x3;牌编号 c:点数=c//4+2,花色=c%4"""
    r = sorted((c // 4 + 2 for c in cards), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    straight = (r[0] - 1 == r[1] and r[1] - 1 == r[2])  # QKA 成立;A23(14,3,2) 不成立
    if r[0] == r[1] == r[2]:
        t = 6
    elif flush and straight:
        t = 5
    elif flush:
        t = 4
    elif straight:
        t = 3
    elif r[0] == r[1] or r[1] == r[2]:
        t = 2
    else:
        t = 1
    return (t, r[0], r[1], r[2])


def cmp_ev(a, b):
    """1=a大,-1=b大,0=完全相同;含散牌 235 胜豹子的特例"""
    if a[0] == 1 and a[1:] == (5, 3, 2) and b[0] == 6:
        return 1
    if b[0] == 1 and b[1:] == (5, 3, 2) and a[0] == 6:
        return -1
    return (a > b) - (a < b)


# ---------- 预计算:22100 种组合、牌型类别、两两胜负矩阵、强度分位数 ----------
COMBOS = list(itertools.combinations(range(52), 3))
N = len(COMBOS)
COMBO_INDEX = {c: i for i, c in enumerate(COMBOS)}
EV_LIST = [evaluate(c) for c in COMBOS]
CLASSES = sorted(set(EV_LIST))
CID = {e: i for i, e in enumerate(CLASSES)}
C = len(CLASSES)
CLS = np.array([CID[e] for e in EV_LIST])
MULT = np.bincount(CLS, minlength=C)
KEY = np.array([e[0] * 3375 + e[1] * 225 + e[2] * 15 + e[3] for e in CLASSES], dtype=np.int64)
# B[i,j]:类别 i 对类别 j 的得分(胜1,平0.5,负0)
B = (KEY[:, None] > KEY[None, :]).astype(np.float64) + 0.5 * (KEY[:, None] == KEY[None, :])
_s235 = CID[(1, 5, 3, 2)]
_leop = np.array([i for i, e in enumerate(CLASSES) if e[0] == 6])
B[_s235, _leop] = 1.0
B[_leop, _s235] = 0.0
B_GT = B - 0.5 * np.eye(C)   # 严格获胜(平局算输):提出开牌者获胜的条件
B_GE = B + 0.5 * np.eye(C)   # 获胜或平局:被开牌者获胜的条件
Q_CLASS = B @ MULT / N        # 击败随机手牌的概率(分位数)
Q_COMBO = Q_CLASS[CLS]
Q_LIST = Q_COMBO.tolist()
BUCKET_COMBO = np.minimum(9, (Q_COMBO * 10).astype(int))
BUCKET_LIST = BUCKET_COMBO.tolist()
MASK = np.array([sum(1 << c for c in cs) for cs in COMBOS], dtype=np.uint64)
MASK_INT = [int(x) for x in MASK]


def print_stats():
    cnt = {t: 0 for t in TYPE_NAME}
    for e in EV_LIST:
        cnt[e[0]] += 1
    print(f"组合总数 {N}")
    for t in (6, 5, 4, 3, 2, 1):
        print(f"  {TYPE_NAME[t]:<4} {cnt[t]:>6}  {cnt[t] / N * 100:7.3f}%")
    print("(A23 不算顺子,故顺子/同花顺为 11 种起点,与规则一致)")
    print("近似 n 人局获胜概率 W_n ≈ q^(n-1),q=该手牌击败随机手牌的概率:")
    for name, cs in [("AAA", (48, 49, 50)), ("A高散牌 AK9", (48, 45, 30)),
                     ("对子 22A", (0, 1, 50)), ("散牌 235", (0, 5, 14))]:
        q = Q_LIST[COMBO_INDEX[tuple(sorted(cs))]]
        print(f"  {name:<12} q={q:.3f}  W_3={q ** 2:.3f}  W_5={q ** 4:.3f}")


# =============================== 规则与状态 ===============================
class Rules:
    def __init__(self, n=5, ante=1, min_bet=1, max_pay=20, max_actions=20, mult=2):
        assert 2 <= n <= 17
        # min_bet: 先手最低支付(闷牌价);max_pay: 单次支付上限(看牌者的支付额)
        self.n, self.ante, self.min_bet = n, ante, min_bet
        self.max_pay = max_pay
        self.max_bet = max_pay // mult      # 内部闷牌价上限,使看牌者支付不超过 max_pay
        self.max_actions, self.mult = max_actions, mult


class State:
    # base = 当前闷牌价 x:闷牌者支付 x,看牌者支付 mult*x
    __slots__ = ('active', 'invest', 'seen', 'base', 'hands', 'turn', 'nact', 'na')

    def clone(self):
        s = State.__new__(State)
        s.active, s.invest, s.seen = self.active[:], self.invest[:], self.seen[:]
        s.base, s.hands, s.turn, s.nact, s.na = self.base, self.hands, self.turn, self.nact, self.na
        return s


def price(st, p, R):
    """玩家 p 此刻按当前下注额应付的金额"""
    return st.base * R.mult if st.seen[p] else st.base


def stake_level(st, R):
    return min(3, st.base - R.min_bet)


def advance(st, p):
    n = len(st.active)
    j = p
    for _ in range(n):
        j = (j + 1) % n
        if st.active[j]:
            st.turn = j
            return


def resolve(st, opener, target):
    """开牌:败者退出;完全相同则提出开牌者负。开牌双方互相亮牌,所以胜者此后视为已看牌"""
    if cmp_ev(EV_LIST[st.hands[opener]], EV_LIST[st.hands[target]]) > 0:
        w, l = opener, target
    else:
        w, l = target, opener
    st.active[l] = False
    st.na -= 1
    st.seen[w] = True
    return w, l


def apply_action(st, p, kind, arg, R):
    """kind:0弃牌 1跟注 2加注(arg=新下注额x,须大于当前x) 3开牌(arg=目标)。返回开牌结果 (胜,负) 或 None"""
    res = None
    if kind == 0:
        st.active[p] = False
        st.na -= 1
    elif kind == 1:
        st.invest[p] += price(st, p, R)
    elif kind == 2:
        st.base = arg
        st.invest[p] += price(st, p, R)
    else:
        st.invest[p] += price(st, p, R)
        res = resolve(st, p, arg)
    st.nact += 1
    if st.na > 1:
        advance(st, p)
    return res


# =============================== 行为策略模型 σ ===============================
# 可观察动作类型: 0弃牌 1闷跟 2看跟 3闷加 4看加 5闷开 6看开
class PolicyTable:
    """σ(a | 已被观察到的看牌状态, 下注额水平, 手牌强度十分位) 的经验频率表(带先验)"""

    def __init__(self, strength=6.0):
        self.counts = np.zeros((2, 4, 10, 7))
        for f in range(2):
            for s in range(4):
                for b in range(10):
                    q = (b + 0.5) / 10
                    fold = max(0.03, 0.85 - 1.1 * q) * (1 + 0.2 * s)
                    rais = 0.03 + max(0.0, q - 0.7) * 0.8
                    opn = 0.03 + max(0.0, q - 0.4) * 0.25 + 0.04 * s
                    call = max(0.05, 1 - fold - rais - opn)
                    seen_v = np.array([fold, 0, call, 0, rais, 0, opn], dtype=float)
                    seen_v /= seen_v.sum()
                    blind_v = np.array([0.03, 0.70, 0, 0.15, 0, 0.12, 0])
                    v = seen_v if f == 1 else 0.5 * seen_v + 0.5 * blind_v
                    self.counts[f, s, b] = strength * v + 0.02
        self.refresh()

    def update(self, f, s, b, a):
        self.counts[f, s, b, a] += 1.0

    def refresh(self):
        self.probs = self.counts / self.counts.sum(-1, keepdims=True)
        self.cum = np.cumsum(self.probs, -1).tolist()


# =============================== 推演(rollout) ===============================
def showdown_known(st):
    """行动次数用尽后的免费轮流开牌(推演中手牌已知:挑自己能赢的最强对手)"""
    n = len(st.active)
    p = st.turn
    while st.na > 1:
        if st.active[p]:
            mine = EV_LIST[st.hands[p]]
            tgt, tq, wt, wq = None, -1.0, None, 2.0
            for t in range(n):
                if t != p and st.active[t]:
                    qt = Q_LIST[st.hands[t]]
                    if cmp_ev(mine, EV_LIST[st.hands[t]]) > 0 and qt > tq:
                        tgt, tq = t, qt
                    if qt < wq:
                        wt, wq = t, qt
            resolve(st, p, tgt if tgt is not None else wt)
        p = (p + 1) % n


def simulate(st, me, R, table, rnd):
    """从 st 起,所有玩家按 σ 的行为分布行动到终局,返回 me 的净收益"""
    n = R.n
    cum = table.cum
    hands = st.hands
    while st.na > 1:
        if st.nact >= R.max_actions:
            showdown_known(st)
            break
        p = st.turn
        row = cum[1 if st.seen[p] else 0][stake_level(st, R)][BUCKET_LIST[hands[p]]]
        u = rnd.random()
        a = 6
        for k in range(7):
            if u < row[k]:
                a = k
                break
        if a in (2, 4, 6):
            st.seen[p] = True
        if a == 0:
            apply_action(st, p, 0, 0, R)
        elif a <= 2:
            apply_action(st, p, 1, 0, R)
        elif a <= 4:
            nb = st.base + 1 + (1 if rnd.random() < 0.3 else 0)
            nb = min(nb, R.max_bet)
            if nb <= st.base:
                apply_action(st, p, 1, 0, R)
            else:
                apply_action(st, p, 2, nb, R)
        else:
            opp = [t for t in range(n) if t != p and st.active[t]]
            apply_action(st, p, 3, opp[int(rnd.random() * len(opp))], R)
    return (sum(st.invest) if st.active[me] else 0) - st.invest[me]


# =============================== 信念与胜率 ===============================
def p_beat(g, p, h, opp):
    """在公共信念下(排除自己手牌和已知死牌),手牌 h 击败各对手 j 的概率"""
    ok = (MASK & np.uint64(MASK_INT[h] | g.dead_for(p))) == 0
    row = B[CLS[h]]
    out = {}
    for j in opp:
        w = g.w[j] * ok
        s = w.sum()
        if s <= 0:
            out[j] = 0.5
            continue
        agg = np.bincount(CLS, weights=w, minlength=C)
        out[j] = float(row @ agg / s)
    return out


# =============================== 智能体 ===============================
class Agent:
    def choose_target(self, g, p):
        """免费开牌阶段:看牌(免费),指定"自己最可能赢"的对手"""
        st = g.st
        st.seen[p] = True
        opp = [j for j in range(g.R.n) if st.active[j] and j != p]
        pb = p_beat(g, p, st.hands[p], opp)
        return max(opp, key=lambda j: pb[j])


class ThresholdAgent(Agent):
    """朴素基线:先看牌;弱牌弃,强牌加注,两人时较强即开牌,其余跟注"""

    def decide(self, g, p):
        st, R = g.st, g.R
        if not st.seen[p]:
            return (True, None, None)
        q = Q_LIST[st.hands[p]]
        if q < 0.45:
            return (False, 0, 0)
        if q > 0.93 and st.base + 1 <= R.max_bet:
            return (False, 2, st.base + 1)
        if st.na == 2 and q > 0.6:
            return (False, 3, self.choose_target(g, p))
        return (False, 1, 0)


class EVAgent(Agent):
    """期望净收益最大化 + 贝叶斯信念 + 蒙特卡洛推演"""

    def __init__(self, seed=0, nroll=24, nlook=6, nroll_look=4):
        self.rnd = random.Random(seed)
        self.nroll, self.nlook, self.nroll_look = nroll, nlook, nroll_look

    # ---------- 从信念中抽样"世界"(所有在局者的手牌) ----------
    def _draw(self, w, used):
        if used:
            w = w * ((MASK & np.uint64(used)) == 0)
        tot = w.sum()
        if tot <= 0:
            w = ((MASK & np.uint64(used)) == 0).astype(float)
            tot = w.sum()
        i = int(np.searchsorted(np.cumsum(w), self.rnd.random() * tot))
        return min(i, N - 1)

    def _draw_uniform(self, used):
        idx = np.flatnonzero((MASK & np.uint64(used)) == 0)
        return int(idx[int(self.rnd.random() * len(idx))])

    def sample_my_hand(self, g, p):
        return self._draw_uniform(g.dead_for(p))

    def sample_worlds(self, g, p, hand, opp, nw):
        n = g.R.n
        worlds = []
        base_used = g.dead_for(p)
        for _ in range(nw):
            hands = [0] * n
            used = base_used | (MASK_INT[hand] if hand is not None else 0)
            order = opp[:]
            self.rnd.shuffle(order)
            for j in order:
                i = self._draw(g.w[j], used)
                hands[j] = i
                used |= MASK_INT[i]
            hands[p] = hand if hand is not None else self._draw_uniform(used)
            worlds.append(hands)
        return worlds

    # ---------- 候选行动 ----------
    def _raises(self, g):
        """加注到的整数下注额:当前+1、+2、翻倍、上限,最多取其中 3 个有代表性的"""
        st, R = g.st, g.R
        opts = sorted({x for x in (st.base + 1, st.base + 2, st.base * 2, R.max_bet) if st.base < x <= R.max_bet})
        if len(opts) > 3:
            opts = [opts[0], opts[len(opts) // 2], opts[-1]]
        return [(2, x) for x in opts]

    def seen_cands(self, g, p, h, opp):
        cands = [(1, 0)] + self._raises(g)
        pb = p_beat(g, p, h, opp)
        order = sorted(opp, key=lambda j: pb[j])
        targets = [order[-1]]
        if len(order) >= 3:
            targets.append(order[0])
        return cands + [(3, t) for t in targets]

    def blind_cands(self, g, p, opp):
        cands = [(1, 0)] + self._raises(g)
        eq = {j: float((g.w[j] * Q_COMBO).sum() / g.w[j].sum()) for j in opp}
        order = sorted(opp, key=lambda j: eq[j])
        targets = [order[0]]
        if len(order) >= 3:
            targets.append(order[-1])
        return cands + [(3, t) for t in targets]

    # ---------- 期望净收益估计 ----------
    def eval_cands(self, g, p, cands, seen_flag, worlds):
        st, R = g.st, g.R
        evs = [0.0] * len(cands)
        for hands in worlds:
            for i, (kind, arg) in enumerate(cands):
                s = st.clone()
                s.hands = hands
                s.seen[p] = seen_flag
                apply_action(s, p, kind, arg, R)
                evs[i] += simulate(s, p, R, g.table, self.rnd)
        return [e / len(worlds) for e in evs]

    # ---------- 决策:返回 (是否先看牌, kind, arg) ----------
    def decide(self, g, p):
        st = g.st
        opp = [j for j in range(g.R.n) if st.active[j] and j != p]
        inv = st.invest[p]
        if st.seen[p]:
            h = st.hands[p]
            cands = self.seen_cands(g, p, h, opp)
            worlds = self.sample_worlds(g, p, h, opp, self.nroll)
            evs = self.eval_cands(g, p, cands, True, worlds)
            b = max(range(len(cands)), key=evs.__getitem__)
            return (False,) + cands[b] if evs[b] >= -inv else (False, 0, 0)
        # 闷牌状态:比较"继续闷牌行动"与"先看牌再行动"的期望收益
        cands = self.blind_cands(g, p, opp)
        worlds = self.sample_worlds(g, p, None, opp, self.nroll)
        evs = self.eval_cands(g, p, cands, False, worlds)
        best_blind = max(max(evs), -inv)
        tot = 0.0
        for _ in range(self.nlook):
            h = self.sample_my_hand(g, p)
            cs = self.seen_cands(g, p, h, opp)
            ws = self.sample_worlds(g, p, h, opp, self.nroll_look)
            e = self.eval_cands(g, p, cs, True, ws)
            tot += max(max(e), -inv)
        if tot / self.nlook > best_blind:
            return (True, None, None)
        b = max(range(len(cands)), key=evs.__getitem__)
        return (False,) + cands[b] if evs[b] >= -inv else (False, 0, 0)


# =============================== 一局游戏 ===============================
class Game:
    def __init__(self, R, agents, table, rng, verbose=False):
        self.R, self.agents, self.table, self.rng, self.verbose = R, agents, table, rng, verbose

    def log(self, msg):
        if self.verbose:
            print(msg)

    def dead_for(self, p):
        """玩家 p 已知不在他人手里的牌:全场已亮牌的弃牌者 + p 开牌时私下看到的败者手牌"""
        return self.dead | self.pdead[p]

    def hand_str(self, j):
        h = COMBOS[self.st.hands[j]]
        return ''.join(card_str(c) for c in h) + f"({TYPE_NAME[EV_LIST[self.st.hands[j]][0]]})"

    def _after_open(self, opener, w_, l_, free=False):
        """开牌后的信息更新:亮牌仅限双方;其他人只知道"胜者的牌击败了败者的牌" """
        st = self.st
        self.log(f"   亮牌(仅座位{opener}与座位{w_ if w_ != opener else l_}可见) "
                 f"座位{w_}:{self.hand_str(w_)} 胜  座位{l_}:{self.hand_str(l_)} 负")
        aggl = np.bincount(CLS, weights=self.w[l_], minlength=C)
        aggl = aggl / aggl.sum()
        M = B_GT if w_ == opener else B_GE        # 开牌者须严格获胜;被开牌者平局也算胜
        fac = (M @ aggl)[CLS]
        nw = self.w[w_] * fac
        if nw.sum() > 0:
            self.w[w_] = nw / nw.sum()
        self.pdead[w_] |= MASK_INT[st.hands[l_]]  # 胜者私下看到了败者的牌

    def play(self):
        R = self.R
        n = R.n
        perm = self.rng.permutation(52).tolist()
        st = State()
        st.hands = [COMBO_INDEX[tuple(sorted(perm[3 * i:3 * i + 3]))] for i in range(n)]
        st.active, st.invest, st.seen = [True] * n, [R.ante] * n, [False] * n
        st.base, st.nact, st.na = R.min_bet, 0, n
        st.turn = int(self.rng.integers(n))
        self.st = st
        self.w = [np.ones(N) for _ in range(n)]   # 公共后验信念(未归一权重)
        self.dead = 0
        self.pdead = [0] * n
        self.first = [None] * n                    # 每位玩家的第一次行动 (kind, 行动时是否已看牌)
        self.stats = {'fold': 0, 'call': 0, 'raise': 0, 'open': 0, 'blind_act': 0, 'seen_act': 0}
        self.log(f"—— 新局:{n} 人,先手座位 {st.turn},底注 {R.ante},底池 {sum(st.invest)} ——")
        while st.na > 1 and st.nact < R.max_actions:
            p = st.turn
            flag = int(st.seen[p])                 # 行动前"可被观察到"的看牌状态
            sl = stake_level(st, R)
            ag = self.agents[p]
            look, kind, arg = ag.decide(self, p)
            if look:
                st.seen[p] = True
                self.log(f"座位{p} 看牌")
                _, kind, arg = ag.decide(self, p)
            was_seen = st.seen[p]
            if self.first[p] is None:
                self.first[p] = (kind, was_seen)
            a = 0 if kind == 0 else (1, 3, 5)[kind - 1] + int(was_seen)
            self.table.update(flag, sl, BUCKET_LIST[st.hands[p]], a)
            # 贝叶斯更新:按 σ 的似然重新加权此玩家的手牌分布
            L = self.table.probs[flag, sl, :, a]
            self.w[p] = self.w[p] * L[BUCKET_COMBO]
            self.w[p] /= self.w[p].sum()
            before = st.invest[p]
            if kind == 0:                           # 弃牌亮牌:全场可见,这三张牌成为公共死牌
                self.dead |= MASK_INT[st.hands[p]]
            res = apply_action(st, p, kind, arg, R)
            paid = st.invest[p] - before
            self.stats[('fold', 'call', 'raise', 'open')[kind]] += 1
            self.stats['seen_act' if was_seen else 'blind_act'] += 1
            # 所有金额均指实际支付额:加注"至 N"即本次支付 N
            name = ('弃牌', '跟注', '加注至', '开牌→座位')[kind]
            tail = {0: '', 1: f' {paid}', 2: f' {paid}', 3: f'{arg} 支付{paid}'}[kind]
            if kind in (1, 2):
                name = name if kind == 2 else '跟注'
            self.log(f"座位{p} {'看牌' if was_seen else '闷牌'}{name}{tail}"
                     f" (底池{sum(st.invest)};此后闷牌付{st.base}/看牌付{st.base * R.mult})")
            if kind == 0:
                self.log(f"   弃牌亮牌(全场可见) 座位{p}:{self.hand_str(p)}")
            if res:
                self._after_open(p, res[0], res[1])
        if st.na > 1:                               # 行动次数用尽 → 免费轮流开牌
            self.log("达到行动次数上限,进入免费轮流开牌")
            p = st.turn
            while st.na > 1:
                if st.active[p]:
                    t = self.agents[p].choose_target(self, p)
                    w_, l_ = resolve(st, p, t)
                    self.log(f"座位{p} 免费开牌→座位{t}")
                    self._after_open(p, w_, l_)
                p = (p + 1) % n
        winner = st.active.index(True)
        pot = sum(st.invest)
        net = [-st.invest[i] + (pot if i == winner else 0) for i in range(n)]
        self.winner, self.pot = winner, pot
        self.log(f"胜者:座位{winner},底池 {pot},净收益 {net}")
        return net


# =============================== 进度显示 ===============================
def fmt_time(sec):
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s_ = divmod(r, 60)
    return f"{h}:{m:02d}:{s_:02d}" if h else f"{m:02d}:{s_:02d}"


class Progress:
    """单行刷新的进度条:已完成/总数、百分比、已用时间、预计剩余时间(每秒最多刷新一次)"""

    def __init__(self, total, label, enabled=True):
        self.total, self.label, self.enabled = total, label, enabled
        self.t0 = self.last = time.time()

    def update(self, done, force=False):
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last < 1.0 and done < self.total:
            return
        self.last = now
        el = now - self.t0
        eta = el / done * (self.total - done) if done else 0
        sys.stdout.write(f"\r{self.label}: {done}/{self.total} ({done / self.total:.1%})  "
                         f"已用 {fmt_time(el)}  预计剩余 {fmt_time(eta)}   ")
        sys.stdout.flush()
        if done >= self.total:
            sys.stdout.write("\n")


# =============================== 规则自检 / 主程序 ===============================
def selftest():
    """复现约定的例子:5 人底池5;0号闷牌下注1→6;1号看牌付2→8;2号闷牌付1→9"""
    R = Rules(5)
    st = State()
    st.hands = [0] * 5
    st.active, st.invest, st.seen = [True] * 5, [1] * 5, [False] * 5
    st.base, st.nact, st.na, st.turn = R.min_bet, 0, 5, 0
    assert sum(st.invest) == 5
    apply_action(st, 0, 1, 0, R)
    assert sum(st.invest) == 6
    st.seen[1] = True
    apply_action(st, 1, 1, 0, R)
    assert sum(st.invest) == 8
    apply_action(st, 2, 1, 0, R)
    assert sum(st.invest) == 9
    st.seen[3] = True
    apply_action(st, 3, 2, 3, R)       # 看牌者加注到 x=3,支付 6
    assert st.invest[3] == 1 + 6 and st.base == 3
    apply_action(st, 4, 1, 0, R)       # 闷牌者只付 x=3
    assert st.invest[4] == 1 + 3
    print("下注规则自检通过:", st.invest)


def new_hs():
    return {t: {'deals': 0, 'acted': 0, 'looked': 0, 'kinds': [0, 0, 0, 0], 'win': 0, 'net': 0.0, 'inv': 0.0}
            for t in TYPE_NAME}


def run_job(args, games, seed, counter=None, show=True, every=0):
    """单个进程内:预热学习 σ → 模拟 games 局 → 返回统计汇总"""
    R = Rules(args.players, args.ante, args.min_bet, args.max_bet, args.max_actions, args.mult)
    rng = np.random.default_rng(seed)
    table = PolicyTable()
    nr = args.rollouts
    agents = [EVAgent(seed * 100 + i, nr) for i in range(R.n)]
    if args.baseline:
        agents = [agents[0]] + [ThresholdAgent() for _ in range(R.n - 1)]
    if args.train and not args.baseline:
        for a in agents:                       # 预热阶段用较少推演次数加速
            a.nroll, a.nlook, a.nroll_look = max(8, nr // 3), 3, 2
        prog = Progress(args.train, "预热", show and not args.verbose)
        for k in range(args.train):
            Game(R, agents, table, rng).play()
            table.refresh()
            prog.update(k + 1)
        for a in agents:
            a.nroll, a.nlook, a.nroll_look = nr, 6, 4
    res = {'total': np.zeros(R.n), 'wins': np.zeros(R.n), 'pots': [], 'HS': new_hs(), 'ckpt': [],
           'acts': {'fold': 0, 'call': 0, 'raise': 0, 'open': 0, 'blind_act': 0, 'seen_act': 0}}
    prog = Progress(games, "对局", show and not args.verbose)
    for k in range(games):
        g = Game(R, agents, table, rng, verbose=args.verbose)
        net = g.play()
        assert sum(net) == 0, "零和检验失败(无抽水,净收益之和应为0)"
        res['total'] += net
        res['wins'][g.winner] += 1
        res['pots'].append(g.pot)
        for key in res['acts']:
            res['acts'][key] += g.stats[key]
        for i in range(R.n):                      # 按开局手牌类型统计策略与结果
            h = res['HS'][EV_LIST[g.st.hands[i]][0]]
            h['deals'] += 1
            h['win'] += int(i == g.winner)
            h['net'] += net[i]
            h['inv'] += g.st.invest[i]
            if g.first[i] is not None:
                h['acted'] += 1
                h['kinds'][g.first[i][0]] += 1
                h['looked'] += int(g.first[i][1])
        if not args.baseline:
            table.refresh()
        prog.update(k + 1)
        if every and (k + 1) % every == 0:        # 记录"累计到第 k+1 局"各座位的总净收益
            res['ckpt'].append(res['total'].copy())
        if counter is not None:
            with counter.get_lock():
                counter.value += 1
    return res


def _worker(args, games, seed, counter, q, every):
    q.put(run_job(args, games, seed, counter, show=False, every=every))


def merge(results):
    out = results[0]
    for r in results[1:]:
        out['total'] = out['total'] + r['total']
        out['wins'] = out['wins'] + r['wins']
        out['pots'] = out['pots'] + r['pots']
        m = min(len(out['ckpt']), len(r['ckpt']))
        out['ckpt'] = [out['ckpt'][j] + r['ckpt'][j] for j in range(m)]
        for k in out['acts']:
            out['acts'][k] += r['acts'][k]
        for t in out['HS']:
            a, b = out['HS'][t], r['HS'][t]
            for f in ('deals', 'acted', 'looked', 'win', 'net', 'inv'):
                a[f] += b[f]
            a['kinds'] = [x + y for x, y in zip(a['kinds'], b['kinds'])]
    return out


def report_rounds(args, res, step):
    """每隔 step 局输出一次各座位的累计净收益(并可写入 CSV)"""
    n = args.players
    rows = [((j + 1) * step, c) for j, c in enumerate(res['ckpt'])]
    if not rows or rows[-1][0] != args.games:
        rows.append((args.games, res['total']))
    print("\n各座位累计净收益(每行 = 打完该局数后的累计值):")
    print(f"{'局数':>8}" + ''.join(f"{'座位' + str(i):>10}" for i in range(n)) + f"{'合计':>8}")
    for g_, c in rows:
        print(f"{g_:>8}" + ''.join(f"{c[i]:>+10.0f}" for i in range(n)) + f"{c.sum():>+8.0f}")
    if args.csv:
        with open(args.csv, 'w', encoding='utf-8-sig') as f:
            f.write("局数," + ','.join(f"座位{i}" for i in range(n)) + "\n")
            for g_, c in rows:
                f.write(f"{g_}," + ','.join(f"{c[i]:.0f}" for i in range(n)) + "\n")
        print(f"已写入 {args.csv}")


def report(args, res, elapsed):
    n, games = args.players, args.games
    total, wins, acts = res['total'], res['wins'], res['acts']
    print(f"\n共 {games} 局,{n} 人,平均底池 {np.mean(res['pots']):.1f},用时 {fmt_time(elapsed)}")
    print("座位  胜局  平均净收益")
    for i in range(n):
        tag = ' (EV智能体)' if args.baseline and i == 0 else ''
        print(f"  {i}   {int(wins[i]):>3}   {total[i] / games:+.2f}{tag}")
    print(f"净收益总和 = {total.sum():+.0f} (应为 0)")
    tot_act = sum(acts[k] for k in ('fold', 'call', 'raise', 'open'))
    print(f"行动统计(全部 {tot_act} 次行动,不含看牌与免费开牌):",
          ' '.join(f"{nm}{acts[k]}({acts[k] / tot_act:.0%})"
                   for k, nm in (('fold', '弃牌'), ('call', '跟注'), ('raise', '加注'), ('open', '开牌'))),
          f"| 闷牌时行动 {acts['blind_act']} 次,看牌后行动 {acts['seen_act']} 次")
    if args.hand_stats:
        print("\n按【开局手牌类型】统计(首次行动的选择 + 全局结果;样本少时仅供参考):")
        print(f"{'牌型':<5}{'发牌数':>7}{'首动看牌':>9}{'首动弃牌':>9}{'跟注':>7}{'加注':>7}{'开牌':>7}"
              f"{'胜率':>8}{'均投入':>8}{'均净收益':>9}")
        for t in (6, 5, 4, 3, 2, 1):
            h = res['HS'][t]
            if h['deals'] == 0:
                print(f"{TYPE_NAME[t]:<5}{0:>7}   (本次模拟未出现该牌型,请增大 --games)")
                continue
            a = max(h['acted'], 1)
            k = h['kinds']
            print(f"{TYPE_NAME[t]:<5}{h['deals']:>7}{h['looked'] / a:>9.0%}{k[0] / a:>9.0%}{k[1] / a:>7.0%}"
                  f"{k[2] / a:>7.0%}{k[3] / a:>7.0%}{h['win'] / h['deals']:>8.0%}"
                  f"{h['inv'] / h['deals']:>8.1f}{h['net'] / h['deals']:>+9.2f}")


HELP_TEXT = """
goldenbloom —— 炸金花 动态贝叶斯博弈模拟器

基本用法:  python goldenbloom.py [参数]        查看本帮助: python goldenbloom.py help

【一、模拟规模与速度】
  --players N        玩家人数 2~17                              (默认 5)
  --games N          正式统计的总局数                           (默认 10)
  --train N          预热自博弈局数,学习行为策略 σ;多进程时每个进程各预热 N 局   (默认 20)
  --rollouts N       每次决策的蒙特卡洛推演次数,越大越准越慢     (默认 24)
  --workers N        并行进程数,0 = 使用全部 CPU 核心            (默认 1)
  --seed N           随机种子                                   (默认 0)

【二、规则参数】(金额均指实际支付额)
  --ante N           底注                                       (默认 1)
  --min-bet N        先手最低支付金额(闷牌价)                   (默认 1)
  --max-bet N        单次支付金额上限(看牌者的支付额)           (默认 20)
  --mult N           看牌者支付为闷牌者的几倍                   (默认 2)
  --max-actions N    最多行动次数,用完后免费轮流开牌            (默认 20)

【三、输出与统计】
  --verbose          逐局打印过程(出牌、亮牌、底池),此时关闭进度条并强制单进程
  --hand-stats       按开局手牌类型(豹子/同花顺/同花/顺子/对子/散牌)统计首次行动
                     (看牌/弃牌/跟注/加注/开牌比例)、胜率、平均投入、平均净收益
  --report-every X   每隔 X 局输出一次各座位累计净收益      (默认 0 = 自动,约 10 行)
  --csv 文件名       把累计净收益表写入 CSV(可用 Excel 画曲线)
  (运行时自带进度条:已完成局数、已用时间、预计剩余时间)

【四、其他模式】
  --baseline         0 号座位为期望收益最大化智能体,其余为朴素阈值策略,用于检验策略优势
  --stats            打印 6 种牌型的组合数与概率,以及部分手牌的近似获胜概率 后退出
  --selftest         校验下注/底池规则(闷牌付 x、看牌付 mult*x)后退出
  help / -h / --help 显示本帮助

【五、常用示例】
  python goldenbloom.py --games 3 --verbose
        看 3 局完整过程
  python goldenbloom.py --players 6 --games 10000 --train 60 --rollouts 8 --hand-stats --workers 0
        6 人 1 万局、全核并行,并统计各牌型策略
  python goldenbloom.py --players 5 --games 1000 --report-every 100 --csv profit.csv
        每 100 局输出各座位累计净收益并存 CSV
  python goldenbloom.py --players 4 --games 300 --baseline
        检验 EV 智能体对朴素策略的优势
  python goldenbloom.py --stats
        查看牌型概率

【六、规则速记】
  牌型:豹子>同花顺>同花>顺子>对子>散牌;QKA 算顺子,A23 不算;散牌 235 胜任何豹子;
        点数完全相同时提出开牌者负。
  下注:闷牌者付闷牌价 x,看牌者付 mult*x;后一家的闷牌价不得低于上一家;
        "加注至 N" 即本次支付 N。
  亮牌:弃牌向全场亮牌;开牌亮牌仅限开牌双方,败者出局。
  净收益 = 获得的底池 - 本人总投入(无抽水,各座位净收益之和为 0)。
"""


def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() in ('help', '-h', '--help', '-help', '/?'):
        print(HELP_TEXT)
        return
    ap = argparse.ArgumentParser(description="goldenbloom 炸金花贝叶斯博弈模拟", add_help=False)
    ap.add_argument('--players', type=int, default=5)
    ap.add_argument('--games', type=int, default=10, help="正式统计的总局数")
    ap.add_argument('--train', type=int, default=20, help="预热自博弈局数(学习行为策略 σ;多进程时每个进程各预热这么多局)")
    ap.add_argument('--ante', type=int, default=1, help="底注")
    ap.add_argument('--min-bet', type=int, default=1, help="先手最低支付金额(闷牌价)")
    ap.add_argument('--max-bet', type=int, default=20, help="单次支付金额上限(看牌者的支付额)")
    ap.add_argument('--max-actions', type=int, default=20, help="最多行动次数")
    ap.add_argument('--mult', type=int, default=2, help="看牌者支付为闷牌者的几倍")
    ap.add_argument('--rollouts', type=int, default=24, help="每次决策的蒙特卡洛推演次数(越大越准越慢)")
    ap.add_argument('--workers', type=int, default=1, help="并行进程数(0=使用全部 CPU 核心)")
    ap.add_argument('--report-every', type=int, default=0, help="每隔多少局输出一次各座位累计净收益(0=自动,约 10 行)")
    ap.add_argument('--csv', type=str, default='', help="把累计净收益表写入该 CSV 文件")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verbose', action='store_true', help="打印每局过程")
    ap.add_argument('--baseline', action='store_true', help="0号座位 EV 智能体,其余为朴素策略")
    ap.add_argument('--hand-stats', action='store_true', help="按开局手牌类型统计策略(首动选择/胜率/净收益)")
    ap.add_argument('--stats', action='store_true', help="打印牌型概率后退出")
    ap.add_argument('--selftest', action='store_true', help="校验下注规则后退出")
    args = ap.parse_args()
    if args.stats:
        print_stats()
        return
    if args.selftest:
        selftest()
        return

    W = (os.cpu_count() or 1) if args.workers == 0 else args.workers
    W = max(1, min(W, args.games))
    if args.verbose:
        W = 1
    every_total = args.report_every or max(1, args.games // 10)
    every_w = every_total if W == 1 else max(1, every_total // W)   # 多进程:每进程按 1/W 间隔记录
    step = every_w * W
    t0 = time.time()
    if W == 1:
        res = run_job(args, args.games, args.seed, None, True, every_w)
    else:
        print(f"使用 {W} 个进程并行;每个进程先各自预热 {args.train} 局,预热期间进度显示为 0")
        counter = mp.Value('i', 0)
        q = mp.Queue()
        shares = [args.games // W + (1 if i < args.games % W else 0) for i in range(W)]
        procs = [mp.Process(target=_worker, args=(args, shares[i], args.seed + 1000 * i, counter, q, every_w), daemon=True)
                 for i in range(W)]
        for p in procs:
            p.start()
        prog = Progress(args.games, f"对局({W}进程)")
        results = []
        while len(results) < W:
            try:
                results.append(q.get(timeout=1.0))
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    raise SystemExit("工作进程异常退出")
            prog.update(counter.value, force=True)
        for p in procs:
            p.join()
        prog.update(args.games, force=True)
        res = merge(results)
    report(args, res, time.time() - t0)
    report_rounds(args, res, step)


if __name__ == '__main__':
    main()
