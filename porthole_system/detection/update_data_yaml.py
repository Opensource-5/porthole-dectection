import yaml
from glob import glob
import os

def update_data_yaml():
    # 학습 이미지 경로 수집
    train_img_list = glob('train/images/*.jpg')
    print("Train:", len(train_img_list))

    # 검증 이미지 경로 수집
    valid_img_list = glob('valid/images/*.jpg')
    print("Valid:", len(valid_img_list))

    # 테스트 이미지 경로 수집
    test_img_list = glob('test/images/*.jpg')
    print("Test:", len(test_img_list))

    # 디렉토리 존재 여부 확인 및 생성
    os.makedirs('train', exist_ok=True)
    os.makedirs('valid', exist_ok=True)

    # 학습/검증 이미지 경로 텍스트 파일 저장
    with open('train/train.txt', 'w') as f:
        f.write('\n'.join(train_img_list) + '\n')

    with open('valid/val.txt', 'w') as f:
        f.write('\n'.join(valid_img_list) + '\n')

    # data.yaml 업데이트
    try:
        with open('data.yaml', 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        # data.yaml이 없는 경우 기본 데이터 생성
        data = {
            'nc': 1,  # 클래스 수 (기본값 설정, 필요에 따라 수정)
            'names': ['porthole']  # 클래스 이름 (기본값 설정, 필요에 따라 수정)
        }

    # 경로 업데이트
    data['train'] = 'train/train.txt'
    data['val'] = 'valid/val.txt'

    # 수정된 데이터 저장
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print("Updated data.yaml:")
    print(data)

if __name__ == "__main__":
    update_data_yaml()
