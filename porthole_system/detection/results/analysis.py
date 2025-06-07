import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'AppleGothic'  # 맥 기본 한글 폰트

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# CSV 파일 읽기
df = pd.read_csv('detection_results.csv')

# 기본 통계 정보
print("=== 기본 통계 ===")
print(df.groupby('label')['depth_mm'].describe())

# 각 라벨별 개수
print("\n=== 라벨별 분포 ===")
print(df['label'].value_counts())

# 박스플롯으로 라벨별 깊이 분포 비교
plt.figure(figsize=(12, 8))

# 1. 박스플롯
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='label', y='depth_mm')
plt.title('라벨별 깊이 분포 (박스플롯)')
plt.xticks(rotation=45)

# 2. 바이올린 플롯
plt.subplot(2, 2, 2)
sns.violinplot(data=df, x='label', y='depth_mm')
plt.title('라벨별 깊이 분포 (바이올린 플롯)')
plt.xticks(rotation=45)

# 3. 히스토그램
plt.subplot(2, 2, 3)
for label in df['label'].unique():
    if pd.notna(label):  # NaN 값 제외
        subset = df[df['label'] == label]['depth_mm']
        plt.hist(subset, alpha=0.6, label=label, bins=20)
plt.xlabel('Depth (mm)')
plt.ylabel('Frequency')
plt.title('깊이별 히스토그램')
plt.legend()

# 4. 산점도
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='depth_mm', y='confidence', hue='label')
plt.title('깊이 vs 신뢰도 (라벨별)')

plt.tight_layout()
plt.show()

# depth_class와 label 비교 (자동 분류 vs 실제 라벨)
print("=== 자동 분류 vs 실제 라벨 비교 ===")
comparison = pd.crosstab(df['depth_class'], df['label'], margins=True)
print(comparison)

# 일치도 계산
df_clean = df.dropna(subset=['label'])  # NaN 제외
matches = (df_clean['depth_class'] == df_clean['label']).sum()
total = len(df_clean)
accuracy = matches / total * 100
print(f"\n자동 분류 정확도: {accuracy:.1f}% ({matches}/{total})")

# 현재 임계값 확인 및 최적화
def analyze_thresholds(df):
    # 라벨별 깊이 범위 분석
    label_stats = df.groupby('label')['depth_mm'].agg(['min', 'max', 'mean', 'median'])
    print("=== 라벨별 깊이 통계 ===")
    print(label_stats)
    
    # 최적 임계값 제안
    shallow_max = df[df['label'] == 'shallow']['depth_mm'].quantile(0.75)
    medium_min = df[df['label'] == 'medium']['depth_mm'].quantile(0.25)
    medium_max = df[df['label'] == 'medium']['depth_mm'].quantile(0.75)
    
    print(f"\n=== 제안 임계값 ===")
    print(f"Shallow/Medium 경계: {(shallow_max + medium_min) / 2:.2f}mm")
    print(f"Medium/Deep 경계: {medium_max:.2f}mm 이상")

analyze_thresholds(df[df['label'] != 'none'])

# 라벨과 깊이가 일치하지 않는 이상치 찾기
def find_outliers(df):
    outliers = []
    
    for idx, row in df.iterrows():
        depth = row['depth_mm']
        label = row['label']
        
        # 각 라벨별 예상 범위와 비교
        if label == 'shallow' and depth > 10:
            outliers.append((idx, row['filename'], depth, label, 'shallow인데 깊이가 너무 큼'))
        elif label == 'medium' and (depth < 3 or depth > 15):
            outliers.append((idx, row['filename'], depth, label, 'medium 범위를 벗어남'))
        elif label == 'deep' and depth < 10:
            outliers.append((idx, row['filename'], depth, label, 'deep인데 깊이가 얕음'))
    
    return outliers

outliers = find_outliers(df)
print("=== 이상치 데이터 ===")
for outlier in outliers[:10]:  # 상위 10개만 출력
    print(f"파일: {outlier[1]}, 깊이: {outlier[2]}mm, 라벨: {outlier[3]}, 이유: {outlier[4]}")

def comprehensive_analysis(csv_path):
    df = pd.read_csv(csv_path)
    
    print("📊 포트홀 깊이-라벨 관계 분석 리포트")
    print("=" * 50)
    
    # 1. 데이터 개요
    print(f"총 레코드 수: {len(df)}")
    print(f"유니크 파일 수: {df['filename'].nunique()}")
    print(f"라벨 종류: {df['label'].unique()}")
    
    # 2. 라벨별 깊이 통계
    print("\n📏 라벨별 깊이 통계:")
    stats = df.groupby('label')['depth_mm'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print(stats.round(2))
    
    # 3. 분류 정확도
    df_clean = df[(df['label'] != 'none') & (df['depth_class'] != 'none')]
    if len(df_clean) > 0:
        accuracy = (df_clean['depth_class'] == df_clean['label']).mean() * 100
        print(f"\n🎯 자동 분류 정확도: {accuracy:.1f}%")
    
    # 4. 문제점 식별
    print("\n⚠️  발견된 문제점:")
    
    # 깊이가 0이 아닌데 none 라벨
    zero_depth_labeled = df[(df['depth_mm'] > 0) & (df['label'] == 'none')]
    if len(zero_depth_labeled) > 0:
        print(f"- 깊이가 있는데 none 라벨: {len(zero_depth_labeled)}개")
    
    # 라벨 불일치
    mismatch = df[(df['depth_class'] != df['label']) & (df['label'] != 'none') & (df['depth_class'] != 'none')]
    if len(mismatch) > 0:
        print(f"- 자동분류와 라벨 불일치: {len(mismatch)}개")

# 실행
comprehensive_analysis('detection_results.csv')