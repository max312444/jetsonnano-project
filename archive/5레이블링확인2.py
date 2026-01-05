# 각도와 속도의 분포를 확인
import seaborn as sns

# 각도와 속도 분포 확인
plt.figure(figsize=(12, 6))

# 각도 분포
plt.subplot(1, 2, 1)
sns.histplot(df['angle'], kde=True, bins=30)
plt.title("각도 분포")

# 속도 분포
plt.subplot(1, 2, 2)
sns.histplot(df['speed'], kde=True, bins=30)
plt.title("속도 분포")

plt.tight_layout()
plt.show()
