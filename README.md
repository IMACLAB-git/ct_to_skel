# ct2skel — 수술 전 CT ↔ SKEL 3D 신체 모형 정합 · STL · 분해 뷰어

수술 전 CT(DICOM)를 입력하면

1. HU 임계값으로 **피부(체표)** / **뼈** 를 분할하고 (선택: TotalSegmentator로 뼈별 라벨링)
2. [SKEL](https://github.com/MarilynKeller/SKEL) (Keller et al., SIGGRAPH Asia 2023) 의 자세 `q`(46) · 체형 `β`(10) · 위치를 CT 표면과 관절 위치에 **최적화(정합)** 하고
3. CT 피부 / CT 뼈(부위별) / SKEL 피부 / SKEL 뼈(24개 부위별) 를 **동일 좌표계의 STL** 로 내보낸 뒤
4. 브라우저에서 **오버레이 · 나란히 비교 · 오차맵 · 분해(exploded) 뷰** 로 확인합니다.

```
DICOM ─▶ HU 볼륨 ─▶ 피부/뼈 마스크 (+TotalSegmentator) ─▶ 표면 메시
      ─▶ SKEL 정합 (rigid → 체형+척추 → 전신 자세) ─▶ STL + manifest.json ─▶ 웹 뷰어
```

## 1. 설치

```powershell
git clone <this repo> ct_to_skel ; cd ct_to_skel
git clone --depth 1 https://github.com/MarilynKeller/SKEL external/SKEL     # 이미 포함되어 있으면 생략
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128       # GPU. CPU만 쓰면 pip install torch
pip install -e .
$env:PYTHONUTF8 = "1"                                                      # Windows(cp949) 에서 SKEL setup.py 인코딩 오류 방지
pip install --no-deps --no-build-isolation -e external/SKEL
pip install --no-build-isolation "git+https://github.com/MarilynKeller/chumpy"
```

### SKEL 모델 파일 (필수, 라이선스 등록 필요)

SKEL 모델 가중치는 비상업 연구용 라이선스로 배포되므로 저장소에 포함되어 있지 않습니다.

1. https://skel.is.tue.mpg.de/ 에서 계정 등록 → Download 에서 `skel_models_v1.x.zip` 다운로드
2. `skel_male.pkl`, `skel_female.pkl` 을 `external/SKEL/data/skel/` 에 복사
   (또는 `--skel-dir <경로>` / 환경변수 `SKEL_MODELS=<경로>`)

모델 파일이 없으면 `run` 은 CT 분할·STL 내보내기까지만 수행하고 정합은 건너뜁니다.

### (선택) TotalSegmentator — 뼈별 라벨 + 관절 랜드마크

```powershell
pip install TotalSegmentator
```

있으면 `--totalseg` 로 자동 실행됩니다. 대퇴골두·슬관절·상완골두·견봉·L1·T1·C1·ASIS 등에서(SKEL 척추 관절은 각 분절의 상단에 있음) **SKEL 관절 목표점**을 만들고
(`ct2skel/landmarks.py`), CT 뼈를 SKEL 24개 부위와 같은 단위로 나눠 분해 뷰에 씁니다.
없으면 뼈 부위 분할은 정합된 SKEL 뼈 조각과의 최근접 대응으로 대체됩니다.

## 2. 실행

```powershell
# 합성 팬텀으로 파이프라인 점검 (환자 데이터 불필요)
python -m ct2skel synth --out data/phantom
python -m ct2skel run --input data/phantom/dicom --out out/phantom --no-fit
python -m ct2skel serve out/phantom          # http://127.0.0.1:8000

# 실제 CT
python -m ct2skel run --input D:/CT/case01_dicom --out out/case01 --gender male --totalseg
python -m ct2skel run --input case01.nii.gz --out out/case01 --labels D:/totalseg_out   # 미리 만든 라벨 사용
python -m ct2skel serve out/case01
```

주요 옵션

| 옵션 | 의미 |
|---|---|
| `--gender male/female` | 기본값은 DICOM PatientSex |
| `--init-pose arms_down / arms_up / tpose` | 촬영 자세 초기값 (팔 내림 / 팔 올림) |
| `--fit-arms` | 팔을 CT에 정합 (상완골 ICP + 라벨 없는 전완 조각). 기본: 팔 전체를 체형 모델로 추정해 어깨에 매닮 |
| `--limb-prior` / `--stature-cm N` | (선택) 추정 사지 길이의 인체측정 사전: 신장(기본은 CT 상완골 길이에서 Trotter-Gleser 회귀로 추정)에서 대퇴·경골·상완·전완의 관절 간 길이를 정해 정합에 넣음. SKEL 체형 공간이 대퇴·경골 길이를 함께 움직여 다리 전체만 짧아지므로 기본은 끔 |
| `--iters-scale 0.3` | 최적화 반복 수 배율 (빠른 확인용) |
| `--skin-hu -400`, `--bone-hu 250` | 분할 임계값 |
| `--skin-faces / --bone-faces` | STL 면 수 상한 (decimation) |
| `--err-max 20` | 오차맵 색 범위(mm) |
| `--no-fit` | 분할 + CT STL 만 |
| `--no-icp` / `--no-skin-refine` / `--no-poses` | CT 기반 보정 단계·자세 프리셋 생략 |
| `--bone-close-mm 2.5`, `--bone-smooth 30` | 환자 뼈 정리 강도 |
| `--claim-mm 8` | 라벨이 놓친 뼈 복셀(HU>임계값)을 8 mm 안의 가장 가까운 뼈 부위에 편입(늑골–척추 틈 메움); 0이면 끔 |
| (자동) | 라벨이 전혀 없는 큰 뼈 조각(5 cm³ 이상, 예: appendicular 라벨이 없을 때의 전완·하퇴)은 `ct_bone_unlab_*` 로 따로 내보내고, 가까운 SKEL 부위(척골+요골 등)를 강체 ICP로 거기에 맞춤 |

SKEL 모델로 만든 **정답이 있는 팬텀** 으로 정합 정확도를 닫힌 루프로 검증할 수 있습니다:

```powershell
python -m ct2skel synth --out data/skel_phantom --from-skel --gender male   # ground_truth.json 생성
python -m ct2skel run --input data/skel_phantom/dicom --out out/skel_phantom --gender male
python scripts/compare_ground_truth.py data/skel_phantom/ground_truth.json out/skel_phantom
```

검증 결과 (SKEL v1.1.1 male, 2.5 mm 팬텀, RTX 4060, 정합 466 s):

| 지표 | 값 |
|---|---|
| CT 피부 → SKEL 표면 거리 | 평균 1.7 mm, p95 3.4 mm |
| CT 뼈 → SKEL 골격 거리 | 평균 1.5 mm, p95 3.5 mm |
| 관절 위치 오차 (24개) | 평균 3.2 mm, 최대 6.3 mm |
| 자세 오차 (46 DOF) | 평균 1.7°, 최대 12° (발가락 mtp) |
| 체형 β 오차 | 평균 0.23, 최대 0.74 |

### 공개 실제 CT 결과 (TotalSegmentator dataset v2.0.1 small, 1.5 mm, 라벨 제공)

```powershell
python -m ct2skel run --input data/public/ts_small/s1397/ct.nii.gz --labels data/public/ts_small/s1397/segmentations --out out/s1397 --gender male --init-pose arms_up
python scripts/diag_projection.py out/s1397        # 정면·측면 투영 + 관절 쌍 그림
```

| 케이스 | 범위 | 피부 CT→SKEL | 뼈 CT→SKEL | 관절 평균 | 비고 |
|---|---|---|---|---|---|
| example_ct (3 mm, 복부·골반) | 부분 | 10.5 mm | 6.3 mm | 4.0 mm | TotalSegmentator 라벨, 팔 FOV 밖 |
| s1397 (m, 73세, 전신 외상) | 머리–대퇴 | 17.5 mm | 13.3 mm | 12.7 mm | 거구(몸통 폭 39 cm), 팔 머리 옆 |
| s1369 (f, 73세, 전신 외상) | 머리–대퇴 | 15.7 mm | 10.5 mm | 12.4 mm | 팔 머리 위 |

위 표는 파라미터 정합만의 수치이고, ICP 재정합·피부 보정 후(s1397, `scripts/check_continuity.py`)에는 몸통 피부 1.5–3 mm, 대퇴 2 mm, 어깨 3.5 mm 수준입니다.
머리 옆에 놓인 팔은 상완골만 CT에 온전히 있고 전완은 1/3(라벨 없음), 손은 FOV 밖이므로, 기본 설정은 팔 전체를 CT에 맞추지 않고 체형 모델의 팔을 어깨에 매답니다(팔 자세는 뷰어에서 바꾸는 용도). 위 표의 피부 수치는 팔 피부 점을 뺀 값입니다.
SKEL은 통계적 체형 모델이라 개인 골격 형태(골반 폭, 늑골 곡률 등)는 5–15 mm 수준으로만 근사됩니다.

## 3. 출력 구조

```
out/case01/
├── stl/
│   ├── ct_skin.stl                 CT 체표
│   ├── ct_bone_all.stl             CT 뼈 전체
│   ├── ct_bone_<part>.stl          CT 뼈 부위별 (pelvis, femur_r, thorax, …)
│   ├── skel_skin.stl               정합된 SKEL 피부
│   ├── skel_bone_all.stl           정합된 SKEL 골격 전체
│   └── skel_bone_<part>.stl        SKEL 골격 24 부위별
├── ply/*_err.ply                   정점 색 = 상대 표면까지의 거리(mm) (오차맵 모드)
├── manifest.json                   부위 목록·색·중심·bbox, 관절 위치, 정합 지표, β/q/trans, 좌표계 변환
├── skel_params.npz                 betas, poses, trans
├── ct_volume.bin                   8-bit CT 볼륨 (Y,Z,X 순, SKEL 프레임) — 뷰어 슬라이스 대조용
├── poses.json, bone_weights.bin, ct_skin_weights.bin, model_verts.npz   자세 변경용(프리셋·스키닝 가중치)
├── stl/ct_cartilage.stl            늑연골(뼈 아님, 기본 숨김)
├── index.html, app.js              뷰어 (three.js, CDN)
└── totalseg/                       (--totalseg 사용 시) ct.nii.gz + labels/
```

**좌표계**: 모든 STL 은 SKEL 프레임(x = 환자 왼쪽, y = 머리 방향, z = 앞쪽) 의 **mm** 단위입니다.
DICOM LPS(mm) 로 되돌리는 4×4 행렬이 `manifest.json → frame.matrix_skelmm_to_lps` 에 있습니다.

## 4. 뷰어

| 기능 | 설명 |
|---|---|
| Overlay / Side by side / Error map | 같은 자리 겹침 · SKEL 을 옆으로 이동 · 표면 거리 색맵 |
| Part colours / CT vs SKEL | 부위별 색 · 출처별(CT 파랑, SKEL 주황) 색 |
| Exploded view · Bones | 뼈 조각을 몸 중심에서 방사형으로 분리 (`E` 키 토글) |
| Layers | 피부 층을 앞으로 분리해 골격 노출 |
| CT ↔ SKEL gap | 두 모델 사이 간격 |
| Axial clip | 상하 방향 절단면 |
| Parts 트리 | 부위별 표시/숨김, 평균 오차(mm) |
| Fit metrics | 피부/뼈 양방향 표면 거리(평균, p95), 관절 오차, β |
| CT slice · Height | 축상 CT 슬라이스 높이 선택(`↑`/`↓`, Shift = 10장). 해당 높이에 CT 영상 평면이 3D 장면에 놓임 |
| CT slice · Window | 연조직 / 뼈 / 폐 / 전체 윈도우 |
| CT slice · Clip meshes above slice | 슬라이스 위쪽 메시를 잘라 단면과 영상을 같이 봄 |
| Fill outside CT with body model | CT 촬영 범위 밖(다리·머리)과 추정 부위(`estimated_parts`, 기본은 양팔 전체)는 이 환자 체형에 맞춰 정합된 SKEL 모델로 채워 회색 반투명 "추정" 표시. 자세 변경도 같이 따라감. 끌 수 있음 |
| Patient bones | `CT surface`(기본, 환자 CT에서 뽑은 뼈) / `template-fitted`(SKEL 템플릿을 CT 뼈에 변형, 실험적) |
| 2D slice panel | 우상단 패널: CT 영상 위에 CT 메시(파랑)·SKEL(주황, 피부는 점선) 단면 윤곽선과 근처 SKEL 관절(빨강) 표시. A/P/R/L 방사선 관례 |

슬라이스 볼륨은 `run` 시 자동으로 `ct_volume.bin`(8-bit, 최대 320 px, ≤512 슬라이스)으로 내보내며, 이미 만든 결과 폴더에는
`python -m ct2skel volume --input <DICOM> --out out/case01` 로 추가할 수 있습니다. URL 예: `index.html?slice=0.4&window=bone&sliceclip=1`.

## 5. 정합 방법 요약 (`ct2skel/fit.py`, `ct2skel/refine.py`)

CT가 기준이고 SKEL은 참조 모델입니다. 라벨(`--labels`/`--totalseg`)이 있으면 파라미터 정합 뒤에 CT 기반 보정을 수행합니다.

1. **파라미터 정합**(아래) — SKEL의 β·q·전역 스케일을 CT 체표·뼈 표면·관절 목표에 맞춤
2. **뼈 단위 ICP**(`refine.align_bones`) — SKEL 뼈 조각을 대응 CT 뼈(라벨 합집합)에 강체 trimmed-ICP로 정렬(잔차 2–4 mm). **팔은 기본적으로 추정 부위**입니다(다리와 같은 규칙): 견갑골까지만 CT에 맞추고 상완골·척골·요골·손은 SKEL 체형 모델 그대로 초기 자세(`--init-pose`)로 어깨에 매달립니다. 팔 DOF는 고정되고, SKEL 팔은 어떤 CT 항에도 들어가지 않으며, CT 쪽의 상완골·라벨 없는 전완 조각은 숨김으로 내보내고 **CT 피부에서도 팔 부분을 잘라냅니다**(가장 가까운 CT 뼈가 팔 뼈이거나 가장 가까운 SKEL 부위가 팔인 정점; `ct_skin`에는 몸통·머리만 남고 추정 팔이 그 자리를 채움). `--fit-arms` 를 주면 상완골 ICP + 라벨 없는 조각에 대한 척골+요골 ICP + 팔 격자 초기화로 팔을 CT에 맞춥니다
3. ICP 결과를 **목표로 바꿔 재정합**(2회) — 관절 중심(가중치 50)과, CT가 뼈 길이의 60 % 이상 덮은 뼈에 한해 **뼈 방향**(`joint_rot`, 가중치 20; 라벨 없는 전완은 0.5, 주축만이면 0.3)을 목표로 삼음. 촬영 밖 뼈(대퇴 근위부만 찍힌 경우 등)는 누운 자세 사전(고관절 1000배)을 유지. 매 스텝 SKEL `pose_limits`로 각도를 잘라내고(`clamp_limits`) β는 ±3 로 제한. `--fit-arms` 일 때 팔은 재정합 전에 (팔꿈치, 어깨 축회전, 회내외) 격자 탐색(`fit.init_arm_from_targets`)으로 초기화해 국소 최소를 피함
   CT 밖 사지의 길이는 체형 공간(β)이 정합니다(비만·고령 환자에서는 β가 ±3에 붙어 비율이 불안정). 선택 옵션 `--limb-prior` 는 CT에 온전히 찍힌 상완골 길이로 신장을 추정(Trotter–Gleser 1958)하고 같은 회귀로 대퇴·경골·상완·전완의 기대 길이(관절 간 거리)를 재정합 목표(`FitTargets.limb_lengths`)로 넣지만, SKEL 체형 공간은 대퇴와 경골 길이를 따로 바꾸지 못해 s1397에서는 다리 전체가 11 cm 짧아지고 신장이 139 cm 가 됐습니다. 기본은 끄고 `metrics.refine.stature_cm` 로 추정 신장만 참고합니다.
4. **표시되는 SKEL 피부·뼈 = 파라미터 모델 자체**(`metrics.refine.display = "parametric"`). ICP 변환은 잔차로만 기록(`metrics.refine.bone_transforms`, `scripts/check_residuals.py`)하므로 어떤 자세로 바꿔도 뼈가 관절에서 떨어지지 않음
5. **피부 비강체 보정**(`refine.refine_skin`) — SKEL 피부 정점을 CT 체표 최근접점으로 이동시키되 메시 그래프에서 변위장을 평활화(FOV 밖·팔 등 CT가 없는 정점은 조화 확장). 피부 오차 1–3 mm. `--no-icp`, `--no-skin-refine` 로 끌 수 있음
6. **환자 뼈**: 라벨 마스크에 닫힘(2.5 mm)·구멍 채우기·섬 제거·Taubin 30회를 적용한 매끈한 CT 뼈(`ct_bone_<part>`); 늑연골은 `ct_cartilage`로 분리(기본 숨김). CT 범위 밖은 이 환자 체형에 맞춘 SKEL 뼈가 "추정" 그룹으로 이어짐. `ct2skel refresh --out DIR` 로 가중치(뼈·CT 피부)·뷰어만 재생성
7. **자세 변경**(`ct2skel/pose.py`) — 뼈도 SKEL 골격 스키닝 가중치(`skel_weights`, 같은 부위의 SKEL 골격 정점 중 최근접; 견갑골 아래 늑골이나 견봉이 이웃 뼈에 묶이지 않도록 부위를 제한)로 움직이므로 사지는 강체, 요추·흉곽은 SKEL처럼 연속적으로 휘고 관절면이 벌어지지 않음 (`bone_weights.bin`): SKEL 운동학으로 분절별 강체 변환 M_j = G'_j·G_j⁻¹ 를 계산해 CT 뼈·템플릿 뼈는 강체로, 피부는 SKEL 스키닝 가중치(CT 피부는 최근접 SKEL 정점 가중치)로 움직임. `poses.json` 에 프리셋(팔 내림/올림, 무릎 90°, 고관절 45°, 앉기, 몸통 굴곡, 머리 회전)과 11단계 보간 프레임 저장. 뷰어 "Pose" 섹션에서 선택·보간
   ```powershell
   python -m ct2skel pose --out out/case01 --set "hip_flexion_r=60,knee_angle_r=100" --name "right leg bent"
   ```
   **팔 내림 프리셋**: `arms_down` 은 팔이 환자 몸통에 파묻히지 않을 만큼만 자동으로 벌립니다(`pose.arms_down_clearance`: 견갑골 리듬을 적용한 상태에서, 어깨 아래 12 cm부터의 팔 뼈 정점 중 CT 피부 안쪽 8 mm 이상에 있는 것이 3 % 이하이고 상완골 정점 중 CT 흉곽 뼈 10 mm 이내가 3 % 이하가 될 때까지 5° 단위로 외전; 좌우 중 큰 각을 양쪽에 적용; 비만 환자 s1397 은 35°. 상완 피부는 몸통 지방에 묻히는 것이 정상이라 검사에서 뺌). CT 피부의 스키닝 가중치는 추정 부위(팔)를 제외한 SKEL 정점에서 찾은 뒤 CT 메시 그래프 위에서 30회 확산시켜(`pose.ct_skin_vertex_weights`) 부위 경계(머리|견갑골 등)가 약 1 cm 에 걸쳐 섞이게 하므로, 올린 팔 옆의 머리·몸통 피부가 팔을 따라 끌려가거나 견갑골 경계에서 찢어지지 않습니다. 서버 STL 내보내기도 같은 가중치를 씁니다.
   **견갑골**: SKEL에는 자세별 관절각 라이브러리가 없고(가동범위 `pose_limits`만 있음) 견갑골 3 DOF는 자유 변수라, 팔을 올리고 찍은 CT의 견갑골(거상·상방회전)이 팔을 내린 자세에서도 그대로 남습니다. 그래서 `pose.apply_scapula_rhythm` 이 팔 거상각에 따라 견갑골을 움직입니다(견갑상완 리듬: 30° 이하 휴지, 180°에서 SKEL 한계까지). CT 자세 근처(±60°)에서는 CT로 정합한 견갑골을 유지하고 멀어질수록 규칙을 따릅니다. 프리셋·`ct2skel pose`·뷰어("Scapula follows arm" 체크박스, 기본 켬)에 모두 적용됩니다.
8. **웹에서 직접 자세 조작**: `ct2skel serve` 는 SKEL 순운동학 API(`/api/info`, `/api/pose`, `/api/save`, `/api/export`)를 겸합니다. 뷰어 "Pose" 섹션에 부위별(고관절·무릎·발목·척추·머리·어깨·팔꿈치/손목) 관절 각도 슬라이더가 생기고, 드래그하면 서버가 분절 변환을 계산해 즉시 반영됩니다(GPU 약 0.15 s). 가동범위는 SKEL `pose_limits`, 좌우 미러, 더블클릭으로 CT 각도 복귀, CT 범위 밖 관절은 흐리게 표시. "Save pose"는 `poses.json`에, "Export STL"은 `out/<case>/poses/<name>/` 에 현재 자세의 STL을 저장합니다. "Ghost of CT pose"로 원래 자세를 반투명으로 겹쳐 봅니다.

파라미터 정합 세부:

- 목표: CT 체표 표면 샘플, CT 뼈 표면 샘플, (있으면) 관절 목표점 24개 + 가중치
- 손실: 양방향 chamfer (CT→SKEL, SKEL→CT; SKEL 점은 CT 촬영 범위 내부만 사용해 **부분 촬영**에 대응),
  관절 L2, 자세 사전(초기 자세 근처), 관절 가동범위 페널티(`skel.kin_skel.pose_limits`), 견갑골 정규화, β L2
- 단계: ① 전역 회전·이동 → ② 관절 목표만으로 자세 정렬(랜드마크 있을 때) → ③ β + 전역 스케일 + 자세(관절 가중치 5, 절단 15 cm) → ④ 미세 조정(절단 8 cm), Adam + cosine LR
- 촬영 범위 밖 뼈의 DOF는 고정, 누운 자세 사전(고관절·무릎 강함, 어깨 약함), 전역 스케일은 골반 기준 0.8–1.25
- SKEL 메시는 매 반복 면 위에서 무작위 샘플링(미분 가능)해 6890 정점의 거친 해상도 편향을 줄임
- 평가: SKEL 표면을 30만 점으로 조밀 샘플링한 뒤 점-표면 거리(mm) 통계

## 6. 외부 웹 공개 (GitHub Pages)

뷰어는 정적 파일이므로 FK 서버 없이도 호스팅할 수 있습니다(저장된 프리셋 자세 + 보간 슬라이더; 실시간 관절 슬라이더는 `ct2skel serve` 전용).

```powershell
python -m ct2skel publish --out out/s1397 --dest site          # site/s1397/ 에 뷰어 파일만 복사 (약 140 MB, --no-volume --no-err 로 축소)
python scripts/deploy_pages.py out/s1397                       # publish + gh-pages 브랜치 커밋/푸시 -> https://<user>.github.io/<repo>/
```

`deploy_pages.py` 는 `./site` 를 orphan `gh-pages` 브랜치(워크트리 `.gh-pages/`)에 미러링해 푸시합니다. 처음 한 번은 GitHub 저장소 Settings → Pages 에서 브랜치 `gh-pages` 를 선택하거나 스크립트가 출력하는 `gh api` 명령을 실행하세요.
**병원 데이터는 올리지 마세요** — 공개 허가된 데이터(예: TotalSegmentator 데이터셋, CC BY 4.0)만 배포합니다. SKEL 모델 파일(`*.pkl`)과 `external/`, `data/`, `out/` 은 `.gitignore` 로 제외됩니다.

## 7. 테스트

```powershell
python -m pytest -q
```

SKEL 가중치 없이도 돌아가도록 인터페이스가 같은 `DummySKEL` 로 정합·내보내기를 검증합니다.

## 한계 / 주의

- SKEL 관절 정의는 OpenSim 기반이라 CT 랜드마크 규칙(`landmarks.py`)은 근사이며 가중치 0.3–1.0 으로 반영됩니다. 필요하면 규칙을 수정하세요.
- 촬영 범위 밖 관절의 자세 DOF는 자동으로 고정됩니다(`FitConfig.freeze_unsupported`). 병원 CT에 팔·손이 들어 있으면 TotalSegmentator `appendicular_bones`(학술 라이선스)로 척골·요골·손 라벨을 만들어 주는 것이 전완·손 정합에 가장 효과적입니다. 상완골두는 같은 쪽 견갑골·쇄골에 가까운 끝으로 판정하므로 팔을 올린 촬영에서도 동작합니다.
- 실제 CT에서는 HU 임계값 대신 라벨(`--labels` 또는 `--totalseg`)로 뼈 마스크를 만드는 것이 훨씬 안정적입니다(조영제·저해상도 부분체적·금속). `appendicular_bones` 작업은 TotalSegmentator 무료 학술 라이선스(`totalseg_set_license`)가 필요하며 없으면 건너뜁니다.
- 공개 데이터: TotalSegmentator small dataset v2.0.1 (Zenodo 10047263, CC BY 4.0, 102례, 1.5 mm, 117 구조 라벨 포함) — `data/public/ts_small/<id>/ct.nii.gz` + `segmentations/` 를 그대로 `--input`/`--labels` 로 쓸 수 있습니다.
- 팔 자세 초기값(`--init-pose`)이 실제와 반대이면 어깨 정합이 국소 최소에 빠질 수 있습니다 (`arms_down`: 손이 골반 높이, `arms_up`: 손이 머리 위 — SKEL v1.1.1 로 확인).
- CT 촬영 범위 밖(예: 다리·머리)의 SKEL 은 체형 사전과 자세 정규화로만 결정됩니다.
- SKEL 모델·코드는 비상업 연구용 라이선스입니다.
