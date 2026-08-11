# cashier_policy_mujoco

`cashier_policy`의 MuJoCo 시뮬레이션 파이프라인(듀얼암 Piper + 목, Meta Quest 텔레옵)만
추출한 저장소입니다. 실물 로봇(CAN/피어-SDK) 관련 코드는 없습니다.

## 구성

- **텔레옵**: `bi_piper_xr` (XRoboToolkit PC Service + `xrobotoolkit_sdk`, placo IK).
  Meta Quest / Pico 공용. `bi_piper_xr_mujoco`는 IK 시드가 심 홈포즈로 맞춰진 변형.
- **로봇(심)**: `bi_piper_mujoco` → `PiperMujocoEnv` (MuJoCo capstone 씬, 듀얼암+목 16차원).
- **키보드 폴백**: `bi_piper_keyboard` (VR 없이 테스트용).
- **정책**: `diffusion`, `vita` (+ 각각의 `observers`/`networks`).

## 셋업 (venv + pip만 — conda 금지)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,sim,teleop]"
```

### XRoboToolkit (Meta Quest 텔레옵에 필요)

1. **PC 서비스** 설치: https://github.com/XR-Robotics/XRoboToolkit-PC-Service 릴리즈에서
   OS에 맞는 `.deb`/`.exe` 받아 설치 후 실행 (`/opt/apps/roboticsservice/runService.sh`).
2. **파이썬 바인딩**(`xrobotoolkit_sdk`) 빌드: https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind
   README의 conda 대신 venv+pip로 (`pip install pybind11` 후 `python setup.py install`).
   빌드된 `libPXREARobotSDK.so`가 venv 밖(임시 디렉토리)에 있으면 나중에 import가 깨지니,
   `site-packages`로 복사하고 `patchelf --set-rpath '$ORIGIN'`로 경로를 고정해둘 것.
3. **Quest 앱**: https://github.com/XR-Robotics/XRoboToolkit-Unity-Client-Quest 릴리즈의
   사전빌드 APK를 `adb install`.
4. **연결**: 이 PC에 Wi-Fi가 없다면 USB로 `adb reverse tcp:63901 tcp:63901` 걸고, Quest
   앱에서 PC service IP로 `127.0.0.1` 입력 (Wi-Fi 있으면 PC의 실제 LAN IP 입력).

## 사용법

```bash
# 텔레옵 (뷰어: MuJoCo 3D + --display_data=true 주면 Rerun 4캠도)
python -m flare.scripts.teleop_xr \
  --robot.type=bi_piper_mujoco --teleop.type=bi_piper_xr_mujoco --display_data=true
# 텔레옵 중 오른쪽 컨트롤러 B버튼 = 씬(상품 배치) 리셋

# 데이터 수집 (HuggingFace 업로드 전 `hf auth login` 필요)
python -m flare.scripts.record \
  --robot.type=bi_piper_mujoco --teleop.type=bi_piper_xr_mujoco \
  --dataset.repo_id=<username>/<dataset-name> --dataset.fps=30 \
  --dataset.num_episodes=<N> --dataset.single_task="<설명>" --dataset.push_to_hub=true
# 에피소드 제어: 오른쪽 스틱 우/좌 = 저장+다음/재녹화, 왼쪽 Y버튼 = 전체 종료
# (키보드도 병행 지원: → 저장, ← 재녹화, ESC 중단)

# 데이터 수집 중 실시간 YOLO 바코드 검출 (--yolo=true 한 줄로 켜고 끔)
# 최초 1회: pip install -e ".[sim,teleop,yolo]"
python -m flare.scripts.record \
  --robot.type=bi_piper_mujoco --teleop.type=bi_piper_xr_mujoco \
  --dataset.repo_id=<username>/<dataset-name> --dataset.fps=30 \
  --dataset.num_episodes=<N> --dataset.single_task="<설명>" --dataset.push_to_hub=true \
  --yolo=true
# 카메라별 바코드 검출 시작/종료를 콘솔에 로그로 찍음 (녹화되는 데이터셋 자체는 안 바뀜)
# 가중치: src/flare/assets/yolo/barcode_v6.pt (BarcodeDetection 리포 v6, 기본값)
# --yolo_conf=0.4(기본) / --yolo_model_path=<경로> / --yolo_camera_keys='["head","left_wrist","right_wrist"]'로 조정 가능

# 변환 (LeRobot → Zarr)
python -m flare.scripts.convert --local-dir <경로> --target-fps 10 -o <출력경로>

# 학습
flare-train policy=diffusion task=sim_barcode_scan

# 심 추론
python -m flare.scripts.eval_sim --checkpoint <경로>/training_state.pt --policy diffusion
```

## 참고

- 상품은 spam 1개만 활성화되어 있음(MJCF에 나머지 4개는 주석 처리, 완전 삭제 아님).
