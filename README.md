# aiden.kim-til
aiden.kim - 'Today I Learned'

## 2025.02.03(Mon), Aiden's TIL
### 1. 재귀 함수(Recursive Function)과 꼬리 호출(Tail Call)의 차이점

1. 재귀 함수 (Recursive Function)
재귀 함수는 자기 자신을 호출하는 함수.
재귀 함수는 보통 기저 조건이 필요하고, 이 조건에 도달할 때까지 자기 자신을 반복해서 호출.

```
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
```

위 예시에서 factorial(n) 함수는 n == 0이 될 때까지 자기 자신을 호출.
각 호출에서 곱셈을 수행하므로, 호출된 함수가 끝난 후에 곱셈이 이루어짐. 이때 호출이 끝날 때까지 스택에 여러 함수가 쌓이게 되어, 스택이 꽉 차면 **스택 오버플로우** 가 발생할 수 있음.

2. 꼬리 호출 (Tail Call)
꼬리 호출은 재귀 함수의 한 형태로, 함수의 마지막 동작이 다른 함수의 재귀 호출인 경우를 말함.
꼬리 호출에서는 추가적인 계산이 없이 바로 재귀 호출만 이루어짐. 이 방식은 함수 호출 후 추가적인 작업 없이 바로 반환되기 때문에,
스택을 쌓지 않고 이전 함수 호출을 재사용할 수 있어 메모리 효율적임.

```
def factorial_tail_recursive(n, accumulator=1):
    if n == 0:
        return accumulator
    else:
        return factorial_tail_recursive(n - 1, accumulator * n)  # 마지막 동작이 재귀 호출

```

이 함수에서는 재귀 호출 후 추가 계산이 없음. 바로 accumulator 값을 계산하여 결과를 반환. 그래서 재귀 호출 후에 다른 계산이 필요 없고, 이전 호출 스택을 재사용할 수 있어 메모리 효율이 좋음.

3. 차이점 정리:
   - 일반적인 재귀 함수에서는 각 함수 호출이 끝나고 나서 추가적인 계산이 이루어지기 때문에, 호출 스택에 쌓이는 정보가 계속해서 남음.
   - 꼬리 호출에서는 재귀 호출이 함수의 마지막 작업이므로, 추가적인 계산이 없고 스택을 재사용할 수 있기 때문에 메모리 효율적.
  
4. Python에서 꼬리 호출
   - Python에서는 꼬리 호출 최적화를 지원하지 않기 때문에, **꼬리 호출을 하더라도 스택 오버플로우가 발생할 수 있음.**

## 2025.02.04(Tue), Aiden's TIL
### Numpy

1. Numpy: 대규모 다차원 배열 및 행렬 연산을 위한 고성능 수학 함수와 도구를 제공하는 라이브러리.
    - 아래와 같이 라이브러리 임포트.
```
import numpy as np
```

2. 차원: 배열을 구성하는 축의 개수.
   - 배열의 속성을 아래와 같이 확인하여 차원 조회 가능
```
array.ndim
```

3. 형태: 차원과 각 차원의 크기를 나타내는 것.
   - 배열의 속성을 아래와 같이 확인하여 차원 조회 가능
```
array.shape
```

4. 인덱스: 배열 내 특정 요소의 위치를 나타내는 정수.
5. 인덱싱: 정수, 슬라이싱, 불리언 인덱싱, 팬시 인덱싱의 방법으로 데이터를 조회하는 것.
   - 정수를 통한 인덱싱
```
array = np.array([10, 20, 30, 40])
print(array[0])
print(array[-1])
```
    - 슬라이싱
```
print(array[1:4])
print(array[:3])
print(array[::2])
```
    - 불리언 인덱싱
```
filtered = data[data > 10]
print(filtered)
```
    - 팬시 인덱싱
```
matrix = np.array([[1, 2], [3, 4], [5, 6]])
rows = [0, 2]
print(matrix[rows])
```
6. 유니버셜 함수: 배열의 모든 요소에 동일한 연산을 적용할 수 있는 함수
```
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.add(a, b))
print(np.subtract(a, b))
print(np.multiply(a, b))
print(np.divide(a, b))
print(np.power(a, 2))
```

### Pandas

1. Pandas: 구조화된 데이터의 조작과 분석을 위한 데이터프레임 및 시리즈 객체를 제공하는 라이브러리
```
import pandas as pd
```
2. 시리즈: 인덱스를 기반으로 하는 1차원 구조
```
# 리스트를 이용한 Series 생성
series = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
```
3. 데이터프레임: 행과 열로 구성된 2차원 테이블 형태의 데이터 구조
```
data = {'이름': ['홍길동', '김철수', '박영희'],
        '나이': [25, 30, 28],
        '성별': ['남', '남', '여']}

df = pd.DataFrame(data)
print(df)
```
4. Grouping: 데이터를 특정 기준에 따라 그룹화하여 다양한 연산을 수행하는 기능
```
# 예제 데이터 생성
data = {
    '이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬', '신사임당'],
    '부서': ['영업', '영업', '인사', '인사', 'IT', 'IT'],
    '급여': [5000, 5500, 4800, 5100, 6000, 6200]
}

df = pd.DataFrame(data)

# 부서별 급여 평균 계산
grouped = df.groupby('부서')['급여'].mean()
print(grouped)
```
5. Merging: 여러 데이터프레임을 공통 열 또는 인덱스를 기준으로 결합하는 과정.
```
import pandas as pd

df1 = pd.DataFrame({'고객ID': [1, 2, 3],
                    '이름': ['홍길동', '김철수', '이영희']})

df2 = pd.DataFrame({'고객ID': [2, 3, 4],
                    '구매액': [10000, 20000, 30000]})

result = pd.merge(df1, df2, on='고객ID', how='inner')
print(result)
```
6. Pivot: 데이터를 특정 기준에 따라 재구성하여 요약 통계를 계산하고, 행과 열을 재배치하여 데이터를 쉽게 분석할 수 있도록 하는 과정.
```
# 샘플 데이터 생성
data = {
    '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    '제품': ['A', 'B', 'A', 'B'],
    '판매량': [100, 200, 150, 250]
}

df = pd.DataFrame(data)

# 피벗 적용
df_pivot = df.pivot(index='날짜', columns='제품', values='판매량')
print(df_pivot)
```

