# 프랙탈 평면 (BOJ 1030) 문제 정리

## 문제 정의
- 크기 N, 시간 s, 중심부 크기 K, 출력 범위 R1, R2, C1, C2가 주어집니다.
- 전체 격자의 크기는 N^s × N^s이며, 재귀적 패턴으로 채워집니다.
- 각 레벨 t에서 중앙의 K × K 블록(크기 K·N^(t-1))은 검정(1), 나머지는 한 단계 낮은 차원의 패턴으로 둘러싸이는 구조입니다.

## 알고리즘 아이디어
1. 출력해야 할 각 셀 (r, c)에 대해 최상위 레벨 t = s부터 1까지 순차 검사합니다.
2. 현재 레벨 t에서 부분 단위 크기 `block = N^(t-1)`, 전체 크기 `size = N^t`를 계산합니다.
3. 중앙 블록의 시작 인덱스 `start = (size - K*block)//2`, 종료 인덱스 `end = start + K*block`를 구합니다.
4. `r % size`, `c % size`가 [start, end) 범위 안에 있으면 검정(1)으로 결정 후 중단합니다.
5. 그렇지 않으면 다음 레벨로 넘어가기 위해 `r %= block`, `c %= block` 연산으로 좌표를 변환합니다.
6. 모든 레벨을 통과하면 흰색(0)이라 판단합니다.

## 시간 복잡도
- 한 셀당 O(s) 연산, 전체 출력 영역 크기 M = (R2−R1+1)×(C2−C1+1)
- 총 O(M·s), s ≤ 50, M ≤ 10^4 정도까지 충분히 빠르게 처리 가능합니다.

## 최적화 포인트
- 재귀 호출 대신 반복문 사용으로 호출 오버헤드 제거
- `N^t` 연산을 매번 계산하지 않도록 미리 리스트에 저장

```python
import sys
input = sys.stdin.readline

# 입력
s, N, K, R1, R2, C1, C2 = map(int, input().split())

# N의 거듭제곱 미리 계산
N_pows = [1] * (s + 1)
for i in range(1, s + 1):
    N_pows[i] = N_pows[i - 1] * N

# 결과 행렬 초기화
rows = R2 - R1 + 1
cols = C2 - C1 + 1
matrix = [['0'] * cols for _ in range(rows)]

# 각 셀에 대해 패턴 결정
for i in range(rows):
    for j in range(cols):
        r = R1 + i
        c = C1 + j
        value = '0'
        for t in range(s, 0, -1):
            size = N_pows[t]
            block = N_pows[t - 1]
            start = (size - K * block) // 2
            end = start + K * block
            if start <= r % size < end and start <= c % size < end:
                value = '1'
                break
        matrix[i][j] = value

# 결과 출력
for line in matrix:
    print(''.join(line))
```
