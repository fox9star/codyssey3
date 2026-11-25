#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini NPU Simulator (MAC 연산 기반 패턴 판별 시뮬레이터)
- 외부 라이브러리(NumPy 등) 없이 순수 Python 표준 라이브러리만 사용
- 기능:
  1. 모드 1: 사용자 입력 (3x3 필터 A, B 및 3x3 패턴 입력 검증, MAC 연산, 판정, 성능 분석)
  2. 모드 2: data.json 분석 (라벨 정규화, 스키마/크기 검증, Epsilon 기반 비교, 크기별 성능 분석, 결과 리포트)
  3. 보너스 1: 2차원 배열 vs 1차원 배열(메모리 접근 단순화) MAC 연산 성능 비교
  4. 보너스 2: N x N 십자가(Cross) 및 X 패턴 자동 생성기
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# 부동소수점 허용오차 (Epsilon) 기본값
EPSILON: float = 1e-9
# 성능 측정 기본 반복 횟수 (최소 기준: 10회 이상)
DEFAULT_REPEAT: int = 10


# =====================================================================
# 1. 데이터 구조 및 패턴 생성기 (보너스 과제 포함)
# =====================================================================
class Grid2D:
    """
    n x n 크기의 2차원 패턴 및 필터를 저장하고
    특정 위치(row, col)의 값을 읽고 쓸 수 있는 데이터 구조 클래스.
    """

    def __init__(self, size: int, data: Optional[List[List[float]]] = None) -> None:
        if size <= 0:
            raise ValueError(f"크기(size)는 1 이상이어야 합니다: {size}")
        self.size: int = size
        self._grid: List[List[float]] = [
            [0.0 for _ in range(size)] for _ in range(size)
        ]
        if data is not None:
            if len(data) != size or any(len(row) != size for row in data):
                raise ValueError(
                    f"데이터 크기가 {size}x{size}와 일치하지 않습니다."
                )
            for r in range(size):
                for c in range(size):
                    self.set(r, c, float(data[r][c]))

    def set(self, row: int, col: int, value: float) -> None:
        """특정 위치 (row, col)에 값을 저장한다."""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError(
                f"인덱스 범위 초과: ({row}, {col}), 크기: {self.size}x{self.size}"
            )
        self._grid[row][col] = float(value)

    def get(self, row: int, col: int) -> float:
        """특정 위치 (row, col)의 값을 읽어온다."""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError(
                f"인덱스 범위 초과: ({row}, {col}), 크기: {self.size}x{self.size}"
            )
        return self._grid[row][col]

    def to_2d_list(self) -> List[List[float]]:
        """내부 2차원 배열 복사본을 반환한다."""
        return [row[:] for row in self._grid]

    def to_1d_list(self) -> List[float]:
        """
        [보너스 과제 1] 메모리 접근 최적화를 위해
        2차원 배열(N x N)을 길이 N^2의 1차원 배열로 변환한다.
        """
        flat: List[float] = []
        for row in self._grid:
            flat.extend(row)
        return flat


def generate_cross_pattern(n: int, fg: float = 1.0, bg: float = 0.0) -> List[List[float]]:
    """
    [보너스 과제 2] N x N 크기의 십자가(Cross, '+') 패턴을 자동 생성한다.
    - 홀수 N: 중앙 행(n // 2)과 중앙 열(n // 2)에 fg 값을 배치
    - 짝수 N: 중앙 2개 행/열에 fg 값을 배치
    """
    if n <= 0:
        raise ValueError("패턴 크기 N은 1 이상이어야 합니다.")
    mid_indices = {n // 2} if n % 2 == 1 else {n // 2 - 1, n // 2}
    grid: List[List[float]] = []
    for r in range(n):
        row: List[float] = []
        for c in range(n):
            if r in mid_indices or c in mid_indices:
                row.append(float(fg))
            else:
                row.append(float(bg))
        grid.append(row)
    return grid


def generate_x_pattern(n: int, fg: float = 1.0, bg: float = 0.0) -> List[List[float]]:
    """
    [보너스 과제 2] N x N 크기의 X자('X') 패턴을 자동 생성한다.
    - 주대각선(r == c)과 부대각선(r + c == n - 1)에 fg 값을 배치
    """
    if n <= 0:
        raise ValueError("패턴 크기 N은 1 이상이어야 합니다.")
    grid: List[List[float]] = []
    for r in range(n):
        row: List[float] = []
        for c in range(n):
            if r == c or (r + c == n - 1):
                row.append(float(fg))
            else:
                row.append(float(bg))
        grid.append(row)
    return grid


# =====================================================================
# 2. 라벨 정규화 (표준 라벨: Cross, X)
# =====================================================================
def normalize_label(raw_label: Any) -> Optional[str]:
    """
    입력된 필터 키 또는 expected 라벨을 프로그램 내부 표준 라벨('Cross', 'X')로 정규화한다.
    - '+' / 'cross' / 'Cross' -> 'Cross'
    - 'x' / 'X'               -> 'X'
    - 'UNDECIDED'             -> 'UNDECIDED'
    - 인식할 수 없는 라벨은 None 반환
    """
    if not isinstance(raw_label, str):
        return None
    cleaned = raw_label.strip().lower()
    if cleaned in ("+", "cross", "plus"):
        return "Cross"
    if cleaned in ("x", "times", "*"):
        return "X"
    if cleaned in ("undecided", "tie", "판정 불가", "판정불가"):
        return "UNDECIDED"
    return None


# =====================================================================
# 3. MAC(Multiply-Accumulate) 연산 및 점수 비교 정책
# =====================================================================
def mac_calculate_2d(pattern: List[List[float]], filter_grid: List[List[float]]) -> float:
    """
    2차원 패턴과 필터의 같은 위치 원소를 곱하고 누적합을 구하는 MAC 연산 (외부 라이브러리 미사용).
    시간 복잡도: O(N^2) (곱셈 N^2회, 덧셈 N^2회)
    """
    n = len(pattern)
    accumulator: float = 0.0
    for r in range(n):
        p_row = pattern[r]
        f_row = filter_grid[r]
        for c in range(n):
            accumulator += p_row[c] * f_row[c]
    return accumulator


def flatten_2d(grid: List[List[float]]) -> List[float]:
    """2차원 배열(N x N)을 1차원 배열(길이 N^2)로 변환한다."""
    flat: List[float] = []
    for row in grid:
        flat.extend(row)
    return flat


def mac_calculate_1d(pattern_1d: List[float], filter_1d: List[float]) -> float:
    """
    [보너스 과제 1] 1차원 배열(길이 N^2)로 변환된 패턴과 필터의 MAC 연산.
    중첩 루프 및 2단계 리스트 인덱싱(grid[r][c])을 제거하여 메모리 접근을 단순화한다.
    """
    length = len(pattern_1d)
    accumulator: float = 0.0
    for i in range(length):
        accumulator += pattern_1d[i] * filter_1d[i]
    return accumulator


def judge_scores(
    score_a: float,
    score_b: float,
    label_a: str = "Cross",
    label_b: str = "X",
    tie_label: str = "UNDECIDED",
    epsilon: float = EPSILON,
) -> str:
    """
    허용오차(epsilon) 기반 점수 비교 정책:
    - |score_a - score_b| < epsilon 이면 동점(tie_label) 처리
    - score_a > score_b 이면 label_a 반환
    - score_b > score_a 이면 label_b 반환
    """
    if abs(score_a - score_b) < epsilon:
        return tie_label
    return label_a if score_a > score_b else label_b


def format_score(score: float, other_score: Optional[float] = None) -> str:
    """
    점수 출력 포맷터:
    - 두 점수 차이가 매우 미세하거나(부동소수점 오차 구간) 소수점 정밀 표시가 필요한 경우 소수점 16자리까지 표시
    - 일반적인 정수/단순 실수 점수는 깔끔하게 표시 (예: 1.0, 5.0)
    """
    if other_score is not None and 0 < abs(score - other_score) < 1e-4:
        return f"{score:.16f}"
    if abs(score - round(score, 4)) > 1e-12:
        return f"{score:.16f}"
    return f"{score:.1f}" if abs(score - round(score, 1)) < 1e-9 else f"{score:.4f}"


# =====================================================================
# 4. 성능 측정 함수 (I/O 제외, 순수 연산 구간 반복 측정)
# =====================================================================
def measure_mac_time_ms(
    pattern_2d: List[List[float]],
    filter_2d: List[List[float]],
    repeat: int = DEFAULT_REPEAT,
) -> float:
    """
    2차원 MAC 연산을 repeat회 반복 측정하여 1회당 평균 연산 시간(ms)을 반환한다.
    I/O 시간을 제외하고 순수 MAC 연산 함수 호출 구간만 time.perf_counter()로 측정한다.
    """
    start = time.perf_counter()
    for _ in range(repeat):
        mac_calculate_2d(pattern_2d, filter_2d)
    elapsed_sec = time.perf_counter() - start
    return (elapsed_sec * 1000.0) / repeat


def measure_mac_1d_time_ms(
    pattern_1d: List[float],
    filter_1d: List[float],
    repeat: int = DEFAULT_REPEAT,
) -> float:
    """
    [보너스 과제 1] 1차원 배열 MAC 연산을 repeat회 반복 측정하여 평균 연산 시간(ms)을 반환한다.
    """
    start = time.perf_counter()
    for _ in range(repeat):
        mac_calculate_1d(pattern_1d, filter_1d)
    elapsed_sec = time.perf_counter() - start
    return (elapsed_sec * 1000.0) / repeat


def print_performance_table(
    sizes: List[int],
    custom_filters: Optional[Dict[int, Dict[str, List[List[float]]]]] = None,
    repeat: int = DEFAULT_REPEAT,
    show_bonus_comparison: bool = True,
) -> None:
    """
    크기별 MAC 연산 평균 시간(ms)과 연산 횟수(N^2) 표를 출력한다.
    보너스 과제로 2D 배열 vs 1D 배열(메모리 접근 최적화) 성능 비교도 함께 출력한다.
    """
    print(f"\n#---------------------------------------")
    print(f"# 성능 분석 (평균/{repeat}회)")
    print(f"#---------------------------------------")
    print(f"{'크기':<10} {'평균 시간(ms)':<16} {'연산 횟수'}")
    print("-" * 40)

    comparison_rows: List[Tuple[int, float, float, int]] = []

    for n in sizes:
        if custom_filters and n in custom_filters and "Cross" in custom_filters[n]:
            f_2d = custom_filters[n]["Cross"]
        else:
            f_2d = generate_cross_pattern(n)
        p_2d = generate_cross_pattern(n)

        avg_2d_ms = measure_mac_time_ms(p_2d, f_2d, repeat=repeat)
        ops_count = n * n
        size_label = f"{n}×{n}"
        print(f"{size_label:<10} {avg_2d_ms:<16.4f} {ops_count}")

        if show_bonus_comparison:
            p_1d = flatten_2d(p_2d)
            f_1d = flatten_2d(f_2d)
            avg_1d_ms = measure_mac_1d_time_ms(p_1d, f_1d, repeat=repeat)
            comparison_rows.append((n, avg_2d_ms, avg_1d_ms, ops_count))

    if show_bonus_comparison and comparison_rows:
        print(f"\n[보너스] 2D vs 1D(메모리 접근 단순화) MAC 연산 성능 비교 (평균/{repeat}회)")
        print(f"{'크기':<10} {'2D 평균(ms)':<14} {'1D 평균(ms)':<14} {'속도 향상':<12} {'연산 횟수(N²)'}")
        print("-" * 66)
        for n, t_2d, t_1d, ops in comparison_rows:
            size_label = f"{n}×{n}"
            if t_2d > 0:
                diff_pct = ((t_2d - t_1d) / t_2d) * 100.0
                speed_str = f"{diff_pct:+.1f}%"
            else:
                speed_str = "0.0%"
            print(f"{size_label:<10} {t_2d:<14.4f} {t_1d:<14.4f} {speed_str:<12} {ops}")


# =====================================================================
# 5. 모드 1: 사용자 입력 (3x3) 처리 및 검증
# =====================================================================
def prompt_matrix_From_console(title: str, size: int = 3) -> Grid2D:
    """
    콘솔에서 size x size 크기의 행렬을 한 줄씩(공백 구분) 입력받아 Grid2D 객체로 반환한다.
    - 행/열 개수 불일치 또는 숫자 파싱 실패 시 안내 문구를 출력하고 재입력을 유도한다.
    """
    print(f"{title} ({size}줄 입력, 공백 구분)")
    grid = Grid2D(size)
    row_idx = 0

    while row_idx < size:
        try:
            line = input()
        except EOFError:
            raise RuntimeError("입력이 중단되었습니다 (EOF).")

        tokens = line.strip().split()
        if len(tokens) != size:
            print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요. (현재 {len(tokens)}개 입력됨, {row_idx + 1}번째 줄 재입력)")
            continue

        parsed_values: List[float] = []
        parse_error = False
        for token in tokens:
            try:
                parsed_values.append(float(token))
            except ValueError:
                parse_error = True
                break

        if parse_error:
            print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요. (숫자가 아닌 값 포함, {row_idx + 1}번째 줄 재입력)")
            continue

        for col_idx, val in enumerate(parsed_values):
            grid.set(row_idx, col_idx, val)
        row_idx += 1

    return grid


def run_mode_1() -> None:
    """
    모드 1 실행 흐름:
    필터 A, B 입력 -> 저장 확인 -> 패턴 입력 -> MAC 연산 -> 결과 판정 -> 성능 분석(3x3) 출력
    """
    size = 3
    print("\n#----------------------------------------")
    print("# [1] 필터 입력")
    print("#----------------------------------------")
    filter_a = prompt_matrix_From_console("필터 A", size=size)
    print()
    filter_b = prompt_matrix_From_console("필터 B", size=size)

    # 저장 확인 (Grid2D.get 호출 검증)
    _ = filter_a.get(0, 0)
    _ = filter_b.get(size - 1, size - 1)
    print(f"✓ {size}×{size} 필터 A, B 저장 완료")

    print("\n#----------------------------------------")
    print("# [2] 패턴 입력")
    print("#----------------------------------------")
    pattern = prompt_matrix_From_console("패턴", size=size)
    _ = pattern.get(0, 0)

    # MAC 연산 및 시간 측정 (10회 반복 평균)
    pat_2d = pattern.to_2d_list()
    fa_2d = filter_a.to_2d_list()
    fb_2d = filter_b.to_2d_list()

    score_a = mac_calculate_2d(pat_2d, fa_2d)
    score_b = mac_calculate_2d(pat_2d, fb_2d)

    # 두 필터에 대한 MAC 연산 평균 시간(ms) 측정
    time_a_ms = measure_mac_time_ms(pat_2d, fa_2d, repeat=DEFAULT_REPEAT)
    time_b_ms = measure_mac_time_ms(pat_2d, fb_2d, repeat=DEFAULT_REPEAT)
    avg_time_ms = (time_a_ms + time_b_ms) / 2.0

    judgment = judge_scores(
        score_a,
        score_b,
        label_a="A",
        label_b="B",
        tie_label=f"판정 불가 (|A-B| < {EPSILON:g})",
        epsilon=EPSILON,
    )

    print("\n#----------------------------------------")
    print("# [3] MAC 결과")
    print("#----------------------------------------")
    print(f"A 점수: {format_score(score_a, score_b)}")
    print(f"B 점수: {format_score(score_b, score_a)}")
    print(f"연산 시간(평균/{DEFAULT_REPEAT}회): {avg_time_ms:.4f} ms")
    print(f"판정: {judgment}")

    # 3x3 성능 분석 표 출력
    print_performance_table(
        sizes=[3],
        custom_filters={3: {"Cross": fa_2d, "X": fb_2d}},
        repeat=DEFAULT_REPEAT,
        show_bonus_comparison=True,
    )


# =====================================================================
# 6. 모드 2: data.json 로드, 스키마/크기 검증 및 일괄 분석
# =====================================================================
def validate_2d_numeric_matrix(matrix: Any, expected_size: int) -> Tuple[bool, str]:
    """
    2차원 숫자 배열이 expected_size x expected_size 크기인지, 모든 원소가 숫자인지 검증한다.
    반환값: (검증 통과 여부, 오류 사유 메시지)
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        return False, "2차원 리스트 형식이 아닙니다 (비어 있거나 리스트가 아님)"

    actual_rows = len(matrix)
    if not all(isinstance(row, list) for row in matrix):
        return False, "내부 행(row)이 리스트 형식이 아닙니다"

    col_lengths = {len(row) for row in matrix}
    if len(col_lengths) != 1:
        return False, f"행마다 열 개수가 불규칙합니다 (행 수: {actual_rows}, 열 길이 집합: {sorted(col_lengths)})"

    actual_cols = next(iter(col_lengths))
    if actual_rows != expected_size or actual_cols != expected_size:
        return (
            False,
            f"크기 불일치 (필터 {expected_size}×{expected_size} vs 입력 {actual_rows}×{actual_cols})",
        )

    for r_idx, row in enumerate(matrix):
        for c_idx, val in enumerate(row):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return False, f"숫자가 아닌 데이터 포함: 위치 ({r_idx}, {c_idx}) 값 '{val}'"

    return True, "OK"


def extract_size_from_key(key: str) -> Optional[int]:
    """
    'size_{N}' 또는 'size_{N}_{idx}' 형태의 키에서 크기 N(정수)을 추출한다.
    추출 실패 시 None 반환.
    """
    match = re.match(r"^size_(\d+)(?:_.+)?$", key.strip())
    if not match:
        return None
    n = int(match.group(1))
    return n if n > 0 else None


def run_mode_2(json_path: str = "data.json") -> Dict[str, Any]:
    """
    모드 2 실행 흐름:
    필터 로드 -> 패턴 로드/검증 -> MAC 연산/판정/PASS-FAIL 출력
    -> 성능 분석(3x3 포함, 5x5/13x13/25x25) 출력 -> 결과 요약 출력
    """
    if not os.path.exists(json_path):
        print(f"\n[오류] '{json_path}' 파일을 찾을 수 없습니다.")
        return {"total": 0, "passed": 0, "failed": 0, "failures": []}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"\n[오류] '{json_path}' JSON 파싱 실패: {e}")
        return {"total": 0, "passed": 0, "failed": 0, "failures": []}

    if not isinstance(raw_data, dict):
        print(f"\n[오류] '{json_path}'의 최상위 구조가 JSON 객체(dict)가 아닙니다.")
        return {"total": 0, "passed": 0, "failed": 0, "failures": []}

    raw_filters = raw_data.get("filters")
    raw_patterns = raw_data.get("patterns")

    if not isinstance(raw_filters, dict) or not isinstance(raw_patterns, dict):
        print("\n[오류] 'data.json'에 'filters' 또는 'patterns' 객체가 존재하지 않습니다.")
        return {"total": 0, "passed": 0, "failed": 0, "failures": []}

    # ----------------------------------------
    # [1] 필터 로드 및 정규화
    # ----------------------------------------
    print("\n#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")

    loaded_filters: Dict[int, Dict[str, Grid2D]] = {}
    filter_errors: Dict[int, str] = {}

    for size_key, filter_pair in raw_filters.items():
        n = extract_size_from_key(size_key)
        if n is None:
            print(f"✗ '{size_key}' 필터 키 형식 오류 (size_{{N}} 형식이 아님)")
            continue

        if not isinstance(filter_pair, dict):
            filter_errors[n] = f"'{size_key}' 필터 데이터가 객체(dict)가 아님"
            print(f"✗ {size_key:<7} 필터 로드 실패: {filter_errors[n]}")
            continue

        normalized_pair: Dict[str, Grid2D] = {}
        err_msg: Optional[str] = None

        for f_key, f_matrix in filter_pair.items():
            std_label = normalize_label(f_key)
            if std_label not in ("Cross", "X"):
                err_msg = f"알 수 없는 필터 라벨 키 '{f_key}' (정규화 불가)"
                break
            is_valid, reason = validate_2d_numeric_matrix(f_matrix, expected_size=n)
            if not is_valid:
                err_msg = f"{std_label} 필터 검증 실패 - {reason}"
                break
            normalized_pair[std_label] = Grid2D(n, f_matrix)

        if err_msg:
            filter_errors[n] = err_msg
            print(f"✗ {size_key:<7} 필터 로드 실패: {err_msg}")
        elif "Cross" not in normalized_pair or "X" not in normalized_pair:
            missing = [lbl for lbl in ("Cross", "X") if lbl not in normalized_pair]
            filter_errors[n] = f"필수 필터 누락 ({', '.join(missing)})"
            print(f"✗ {size_key:<7} 필터 로드 실패: {filter_errors[n]}")
        else:
            loaded_filters[n] = normalized_pair
            print(f"✓ {size_key:<7} 필터 로드 완료 (Cross, X)")

    # ----------------------------------------
    # [2] 패턴 분석 (라벨 정규화 및 스키마/크기 검증 적용)
    # ----------------------------------------
    print("\n#---------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")

    total_count = 0
    pass_count = 0
    fail_count = 0
    failures: List[Tuple[str, str]] = []

    for pat_key, pat_obj in raw_patterns.items():
        total_count += 1
        print(f"--- {pat_key} ---")

        # 1) 패턴 키에서 N 추출
        n = extract_size_from_key(pat_key)
        if n is None:
            reason = f"패턴 키 형식 오류 ('{pat_key}'에서 크기 N을 추출할 수 없음)"
            print(f"판정: ERROR | FAIL ({reason})")
            fail_count += 1
            failures.append((pat_key, reason))
            continue

        # 2) 해당 size_N 필터 존재 확인
        if n not in loaded_filters:
            err_detail = filter_errors.get(n, f"size_{n} 필터가 존재하지 않음")
            reason = f"필터 매칭 실패 ({err_detail})"
            print(f"판정: ERROR | FAIL ({reason})")
            fail_count += 1
            failures.append((pat_key, reason))
            continue

        # 3) 패턴 객체 스키마 검증 (input, expected 키 확인)
        if not isinstance(pat_obj, dict):
            reason = "스키마 오류 (패턴 항목이 dict 형식이 아님)"
            print(f"판정: ERROR | FAIL ({reason})")
            fail_count += 1
            failures.append((pat_key, reason))
            continue

        if "input" not in pat_obj or "expected" not in pat_obj:
            missing_keys = [k for k in ("input", "expected") if k not in pat_obj]
            reason = f"스키마 오류 (필수 키 누락: {', '.join(missing_keys)})"
            print(f"판정: ERROR | FAIL ({reason})")
            fail_count += 1
            failures.append((pat_key, reason))
            continue

        raw_input = pat_obj["input"]
        raw_expected = pat_obj["expected"]

        # 4) expected 라벨 정규화 검증
        expected_label = normalize_label(raw_expected)
        if expected_label is None:
            reason = f"라벨 정규화 오류 (지원하지 않는 expected 값: '{raw_expected}')"
            print(f"판정: ERROR | expected: {raw_expected} | FAIL ({reason})")
            fail_count += 1
            failures.append((pat_key, reason))
            continue

        # 5) 필터와 패턴의 크기 일치 및 숫자 배열 검증
        is_valid_matrix, matrix_reason = validate_2d_numeric_matrix(raw_input, expected_size=n)
        if not is_valid_matrix:
            print(f"판정: ERROR | expected: {expected_label} | FAIL ({matrix_reason})")
            fail_count += 1
            failures.append((pat_key, matrix_reason))
            continue

        # 6) Grid2D 저장 및 MAC 연산 수행
        pattern_grid = Grid2D(n, raw_input)
        cross_filter = loaded_filters[n]["Cross"]
        x_filter = loaded_filters[n]["X"]

        pat_2d = pattern_grid.to_2d_list()
        score_cross = mac_calculate_2d(pat_2d, cross_filter.to_2d_list())
        score_x = mac_calculate_2d(pat_2d, x_filter.to_2d_list())

        judgment = judge_scores(
            score_cross,
            score_x,
            label_a="Cross",
            label_b="X",
            tie_label="UNDECIDED",
            epsilon=EPSILON,
        )

        print(f"Cross 점수: {format_score(score_cross, score_x)}")
        print(f"X 점수: {format_score(score_x, score_cross)}")

        if judgment == "UNDECIDED":
            reason = f"동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| = {abs(score_cross - score_x):.2e} < {EPSILON:g})"
            print(f"판정: {judgment} | expected: {expected_label} | FAIL (동점 규칙)")
            fail_count += 1
            failures.append((pat_key, "동점(UNDECIDED) 처리 규칙에 따라 FAIL"))
        elif judgment == expected_label:
            print(f"판정: {judgment} | expected: {expected_label} | PASS")
            pass_count += 1
        else:
            reason = f"판정 불일치 (판정: {judgment}, expected: {expected_label})"
            print(f"판정: {judgment} | expected: {expected_label} | FAIL (판정 불일치)")
            fail_count += 1
            failures.append((pat_key, reason))

    # ----------------------------------------
    # [3] 성능 분석 (3x3 포함, 5x5, 13x13, 25x25)
    # ----------------------------------------
    perf_filters: Dict[int, Dict[str, List[List[float]]]] = {
        n: {
            "Cross": pair["Cross"].to_2d_list(),
            "X": pair["X"].to_2d_list(),
        }
        for n, pair in loaded_filters.items()
    }
    perf_sizes = sorted(set([3, 5, 13, 25] + list(loaded_filters.keys())))
    print_performance_table(
        sizes=perf_sizes,
        custom_filters=perf_filters,
        repeat=DEFAULT_REPEAT,
        show_bonus_comparison=True,
    )

    # ----------------------------------------
    # [4] 결과 요약
    # ----------------------------------------
    print("\n#---------------------------------------")
    print("# [4] 결과 요약")
    print("#---------------------------------------")
    print(f"총 테스트: {total_count}개")
    print(f"통과: {pass_count}개")
    print(f"실패: {fail_count}개")

    if failures:
        print("\n실패 케이스:")
        for case_id, reason in failures:
            print(f"- {case_id}: {reason}")
    else:
        print("\n실패 케이스: 없음 (모든 케이스가 라벨 정규화 및 허용오차 정책을 통과함)")

    print('\n(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션에 작성)')

    return {
        "total": total_count,
        "passed": pass_count,
        "failed": fail_count,
        "failures": failures,
    }


# =====================================================================
# 7. 보너스 모드 3: N x N 패턴 생성기 및 시뮬레이션
# =====================================================================
def run_mode_3_generator() -> None:
    """
    [보너스 과제 2] 사용자가 입력한 크기 N에 대해 N x N Cross/X 필터와 패턴을 자동 생성하고,
    MAC 연산 및 2D vs 1D 성능 분석을 즉시 수행한다.
    """
    print("\n#----------------------------------------")
    print("# [보너스] N×N 패턴 생성기 및 MAC 시뮬레이션")
    print("#----------------------------------------")
    while True:
        raw_n = input("생성할 패턴 크기 N을 입력하세요 (3 이상의 정수, 예: 3, 5, 13, 25): ").strip()
        try:
            n = int(raw_n)
            if n < 3:
                print("입력 오류: N은 3 이상의 정수여야 합니다.")
                continue
            break
        except ValueError:
            print("입력 오류: 정수를 입력하세요.")

    cross_mat = generate_cross_pattern(n)
    x_mat = generate_x_pattern(n)

    print(f"\n✓ {n}×{n} Cross 패턴 및 X 패턴 자동 생성 완료")
    if n <= 15:
        print(f"\n[{n}×{n} Cross 패턴 미리보기]")
        for row in cross_mat:
            print(" ".join(str(int(v)) for v in row))
        print(f"\n[{n}×{n} X 패턴 미리보기]")
        for row in x_mat:
            print(" ".join(str(int(v)) for v in row))

    score_cc = mac_calculate_2d(cross_mat, cross_mat)
    score_cx = mac_calculate_2d(cross_mat, x_mat)
    judgment_c = judge_scores(score_cc, score_cx, "Cross", "X", "UNDECIDED")

    print(f"\n[생성된 {n}×{n} Cross 패턴 입력 시 MAC 결과]")
    print(f"- Cross 필터 점수: {format_score(score_cc, score_cx)}")
    print(f"- X 필터 점수    : {format_score(score_cx, score_cc)}")
    print(f"- 판정 결과      : {judgment_c}")

    print_performance_table(
        sizes=[n],
        custom_filters={n: {"Cross": cross_mat, "X": x_mat}},
        repeat=DEFAULT_REPEAT,
        show_bonus_comparison=True,
    )


# =====================================================================
# 8. 메인 콘솔 엔트리포인트
# =====================================================================
def main() -> None:
    print("=== Mini NPU Simulator ===\n")
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 패턴 생성기 (보너스: N×N Cross/X 자동 생성 및 시뮬레이션)")

    while True:
        try:
            choice = input("선택: ").strip()
        except EOFError:
            return

        if choice == "1":
            run_mode_1()
            break
        elif choice == "2":
            default_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
            target_json = "data.json" if os.path.exists("data.json") else default_json
            run_mode_2(target_json)
            break
        elif choice == "3":
            run_mode_3_generator()
            break
        else:
            print("잘못된 선택입니다. 1, 2 또는 3을 입력하세요.")


if __name__ == "__main__":
    main()
