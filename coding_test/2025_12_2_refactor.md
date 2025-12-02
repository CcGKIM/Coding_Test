---

# ✨ 리팩토링 버전 (깔끔한 구조 + 중복 제거)

### ✔ 핵심 리팩토링 포인트

* **가로/세로 탐색을 일반화**해서 하나의 함수로 처리
* 벽 두 칸의 양 옆(혹은 위/아래)을 확인하는 로직도 **공통화**
* `occupied`는 `(x, y, 방향)` 대신 **그림 전체 단위로 저장**
* 중복되는 범위 체크 → `in_board()`로 통합
* 훨씬 구조적이고 깔끔한 코드

---

# ✅ 리팩토링된 코드

```python
N, M = map(int, input().split())
board = [input().strip() for _ in range(N)]

occupied = set()  # (x, y) 두 칸 모두 저장
cnt = 0

def in_board(x, y):
    return 0 <= x < N and 0 <= y < M

# dx, dy: 벽 2칸의 방향 (가로 ↔ 세로)
# adj: 그림을 걸 공간이 필요한 방향(좌우/상하)
def try_place(x, y, dx, dy, adj):
    global cnt

    x2, y2 = x + dx, y + dy
    if not (in_board(x, y) and in_board(x2, y2)):
        return

    # 벽 두 칸이어야 설치 가능
    if board[x][y] != 'X' or board[x2][y2] != 'X':
        return

    # 두 벽칸의 옆 공간이 모두 빈 공간인지 확인
    ax1, ay1 = x + adj[0], y + adj[1]
    ax2, ay2 = x2 + adj[0], y2 + adj[1]

    if not (in_board(ax1, ay1) and in_board(ax2, ay2)):
        return
    if board[ax1][ay1] != '.' or board[ax2][ay2] != '.':
        return

    # 이미 사용된 벽이라면 설치 불가
    if (x, y) in occupied or (x2, y2) in occupied:
        return

    # 설치!
    occupied.add((x, y))
    occupied.add((x2, y2))
    cnt += 1


for i in range(N):
    for j in range(M):
        # 가로 방향: 두 칸 (i, j), (i, j+1)
        # 위쪽이 빈 공간일 때 adj = (-1, 0)
        # 아래쪽이 빈 공간일 때 adj = (+1, 0)
        try_place(i, j, 0, 1, (-1, 0))  # 위쪽
        try_place(i, j, 0, 1, (1, 0))   # 아래쪽

        # 세로 방향: 두 칸 (i, j), (i+1, j)
        # 왼쪽 빈 공간 adj = (0, -1)
        # 오른쪽 빈 공간 adj = (0, +1)
        try_place(i, j, 1, 0, (0, -1))  # 왼쪽
        try_place(i, j, 1, 0, (0, 1))   # 오른쪽

print(cnt)
```

---

# 🧠 리팩토링 후 구조적 개선

### ✔ 중복 감소

기존에는 가로/세로 함수가 따로 있어서
조건문이 반복되고 코드가 매우 길어졌음.

→ 이제 `try_place()` 하나로 모든 경우 처리.

---

### ✔ 방향성 단순화

가로 그림

- 벽 방향: `(0, 1)`
- 빈 공간 방향: `(-1,0)`(위), `(1,0)`(아래)

세로 그림

- 벽 방향: `(1,0)`
- 빈 공간 방향: `(0,-1)`(왼), `(0,1)`(오)

→ 방향을 파라미터로 전달만 하면 됨.

---

### ✔ occupied 처리 간단

예전: `(x, y, direction)`
지금: 그냥 `(x, y)`
→ “이 벽은 이미 어떤 그림에 쓰였다”만 체크하면 충분함.

---

# 📌 시간 복잡도

- 모든 칸에서 최대 4개의 설치 시도
  → **O(N\*M)**
  원래 코드와 동일하지만 훨씬 깔끔함.

---
