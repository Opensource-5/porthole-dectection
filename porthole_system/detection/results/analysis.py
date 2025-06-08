import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
import matplotlib.font_manager as fm

font_path = '/Users/localgroup/Library/Fonts/NanumSquare_acB.ttf'

# 폰트 파일이 존재하는지 확인하고 설정
import os
if os.path.exists(font_path):
    # 폰트 파일을 matplotlib에 등록
    fm.fontManager.addfont(font_path)
    
    # 등록된 폰트 이름 찾기
    fonts = [f for f in fm.fontManager.ttflist if 'NanumSquare' in f.name]
    
    if fonts:
        font_name = fonts[0].name
        plt.rcParams['font.family'] = font_name
        print(f"폰트 설정 완료: {font_name}")
    else:
        plt.rcParams['font.family'] = 'AppleGothic'
        print("나눔스퀘어 폰트를 찾을 수 없어 AppleGothic 사용")
else:
    plt.rcParams['font.family'] = 'AppleGothic'
    print("폰트 파일이 존재하지 않아 AppleGothic 사용")

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.rcParams['font.size'] = 18  # 기본 폰트 크기

# CSV 파일 읽기
df = pd.read_csv('detection_results.csv')

# 기본 통계 정보
print("=== 기본 통계 ===")
print(df.groupby('label')['depth_delta'].describe())

# 각 라벨별 개수
print("\n=== 라벨별 분포 ===")
print(df['label'].value_counts())

# 박스플롯으로 라벨별 깊이 분포 비교
plt.figure(figsize=(12, 8))

# 1. 박스플롯
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='label', y='depth_delta')
plt.title('Depth Distribution by Label (Box Plot)')
plt.xticks(rotation=0)
plt.setp(plt.gca().get_yticklabels(), rotation=0)

# 2. 바이올린 플롯
plt.subplot(2, 2, 2)
sns.violinplot(data=df, x='label', y='depth_delta')
plt.title('Depth Distribution by Label (Violin Plot)')
plt.xticks(rotation=0)
plt.setp(plt.gca().get_yticklabels(), rotation=0)

# 3. 히스토그램
plt.subplot(2, 2, 3)
for label in df['label'].unique():
    if pd.notna(label):  # NaN 값 제외
        subset = df[df['label'] == label]['depth_delta']
        plt.hist(subset, alpha=0.6, label=label, bins=20)
plt.xlabel('depth_delta')
plt.ylabel('Frequency')
plt.title('Histogram by Depth')
plt.legend()
plt.setp(plt.gca().get_yticklabels(), rotation=0)

# 4. 산점도
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='depth_delta', y='confidence', hue='label')
plt.title('Depth vs Confidence (by Label)')
plt.ylabel('confidence')
plt.legend(title='')
plt.setp(plt.gca().get_yticklabels(), rotation=0)

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
    label_stats = df.groupby('label')['depth_delta'].agg(['min', 'max', 'mean', 'median'])
    print("=== 라벨별 깊이 통계 ===")
    print(label_stats)
    
    # 최적 임계값 제안
    shallow_max = df[df['label'] == 'shallow']['depth_delta'].quantile(0.75)
    medium_min = df[df['label'] == 'medium']['depth_delta'].quantile(0.25)
    medium_max = df[df['label'] == 'medium']['depth_delta'].quantile(0.75)
    
    print(f"\n=== 제안 임계값 ===")
    print(f"Shallow/Medium 경계: {(shallow_max + medium_min) / 2:.2f}")
    print(f"Medium/Deep 경계: {medium_max:.2f} 이상")

analyze_thresholds(df[df['label'] != 'none'])

# 라벨과 깊이가 일치하지 않는 이상치 찾기
def find_outliers(df):
    outliers = []
    
    for idx, row in df.iterrows():
        depth = row['depth_delta']
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
    print(f"파일: {outlier[1]}, 깊이: {outlier[2]}, 라벨: {outlier[3]}, 이유: {outlier[4]}")

def comprehensive_analysis(csv_path):
    df = pd.read_csv(csv_path)
    
    print("📊 Pothole Depth-Label Relationship Analysis Report")
    print("=" * 50)
    
    # 1. 데이터 개요
    print(f"Total Records: {len(df)}")
    print(f"Unique Files: {df['filename'].nunique()}")
    print(f"Label Types: {df['label'].unique()}")
    
    # 2. 라벨별 깊이 통계
    print("\n📏 Depth Statistics by Label:")
    stats = df.groupby('label')['depth_delta'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print(stats.round(2))
    
    # 3. 분류 정확도
    df_clean = df[(df['label'] != 'none') & (df['depth_class'] != 'none')]
    if len(df_clean) > 0:
        accuracy = (df_clean['depth_class'] == df_clean['label']).mean() * 100
        print(f"\n🎯 Automatic Classification Accuracy: {accuracy:.1f}%")
    
    # 4. 문제점 식별
    print("\n⚠️  Identified Issues:")
    
    # 깊이가 0이 아닌데 none 라벨
    zero_depth_labeled = df[(df['depth_delta'] > 0) & (df['label'] == 'none')]
    if len(zero_depth_labeled) > 0:
        print(f"- Has depth but labeled as 'none': {len(zero_depth_labeled)} cases")
    
    # 라벨 불일치
    mismatch = df[(df['depth_class'] != df['label']) & (df['label'] != 'none') & (df['depth_class'] != 'none')]
    if len(mismatch) > 0:
        print(f"- Mismatch between auto-classification and label: {len(mismatch)} cases")

# 실행
comprehensive_analysis('detection_results.csv')