# L4 24GB 한국어 취업교육 추천 LLM 벤치마크

**조사 기준일: 2026-10-31.** 1장짜리 NVIDIA L4 24GB에서 로컬 LLM을 서빙하며, **한국어 이력서↔채용공고(JD) 체크리스트** 분석에 필요한 품질과 실측 성능을 비교하는 테스트 프로그램입니다.

> **실제 L4에서 생성한 실측 결과는 포함되어 있지 않습니다.** `summary.json`과 `comparison.md`는 여러분의 GPU에서 명령을 실행한 후 생성됩니다. 프로그램의 테스트가 통과했다는 것은 평가기의 동작을 검증했다는 뜻이지 LLM 모델의 정확도를 검증했다는 뜻은 아닙니다.

## 어떤 모델을 비교하나?

| 프로필 | 실제 배포 모델 | 엔진 | 모델 다운로드 | 비고 |
|---|---|---|---|---|
| `qwen35_fp8_vllm` | `RedHatAI/Qwen3.5-9B-FP8-dynamic` | vLLM | `hf download` | L4 공식 실측 속도 미확보 |
| `gemma4_12b_qat_gguf` | `google/gemma-4-12B-it-qat-q4_0-gguf` | llama.cpp | GGUF 단일 파일 6.98GB | 약 28tok/s 공개 L4 기록은 **이 파일이 아니라 vLLM QAT** |
| `gemma4_12b_qat_vllm` | `google/gemma-4-12B-it-qat-w4a16-ct` | vLLM | `hf download` | Gemma의 별도 선택적 비교 프로필 |
| `hyperclovax_14b_q4_gguf` | `naver-ellm/HyperCLOVAX-SEED-Think-14B-GGUF` | NAVER fork llama.cpp 권장 | GGUF 단일 파일 8.92GB | 라이선스 및 포크 호환성 확인 |
| `qwen38_27b_q4_gguf` | `ggml-org/Qwen3.8-27B-GGUF` | llama.cpp | GGUF 단일 파일 약 19GB | 모델이 24GB에 들어가도 문맥·동시성 제한 필요 |

비교 수치는 `MODEL_COMPARISON.md` 참조.

## 사전 요구사항

- Linux, NVIDIA 드라이버와 `nvidia-smi`에서 **L4 24GB**가 확인되어야 합니다. 장치 인덱스 0을 기본으로 사용합니다.
- Python **3.10 이상**. `benchmark.py`는 **Python 표준 라이브러리만** 사용하므로 추가 패키지 설치가 불필요합니다.
- GGUF 사용 시 `llama-server` CUDA 빌드가 필요합니다. 모델별 아키텍처 지원이 포함된 **최신 버전**이 필요할 수 있습니다.
- vLLM 사용 시 해당 모델을 지원하는 vLLM, PyTorch, CUDA 조합이 필요합니다. 서로 다른 버전의 설치가 충돌한다면 llama.cpp용, vLLM용 **별도 가상환경**을 권장합니다.
- GGUF 또는 Hugging Face 모델의 다운로드는 `--download`를 넣었을 때만 이루어집니다. 모델당 수~수십 GB의 저장공간이 필요합니다.
- HyperCLOVAX는 원본 모델의 **별도 이용 약관**을 확인하고, 가능하면 [NAVER llama.cpp 포크](https://github.com/NAVER-Cloud-HyperCLOVA-X/llama.cpp)의 빌드를 사용하세요.

### 0) 프로그램 자체 점검 — GPU 없어도 가능

```bash
cd l4_llm_benchmark
python -m unittest discover -s tests -v
```

6개의 테스트가 성공하면 스트리밍 응답 처리, 체크리스트 평가, CSV/JSON 결과 기록 기능이 동작한다는 뜻입니다. **이 결과는 실제 모델 벤치마크가 아닙니다.**

### 1-A) 가장 빠른 시작: 이미 구동 중인 로컬 서버에 연결

vLLM이나 llama.cpp가 `/v1/chat/completions` 형태의 OpenAI 호환 API를 제공한다면 다음 명령만 사용하세요.

```bash
python benchmark.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_SERVED_MODEL_ID \
  --output-dir results/my-model \
  --repeats 1 \
  --perf-repeats 3
```

서버에서 모델 ID를 자동 탐색하려면 `--model`을 생략할 수 있습니다. llama.cpp 기본 포트는 보통 8080이므로 `--base-url http://127.0.0.1:8080/v1`을 사용하세요.

### 1-B) 자동 시작→벤치마크→종료 (추천)

Hugging Face CLI를 설치해 둡니다.

```bash
pip install -U huggingface_hub
```

GGUF 모델을 실행하려면 `llama-server`를 CUDA 지원으로 빌드해 PATH에 넣거나, `--llama-server-bin /path/to/llama-server`를 지정하세요.

```bash
# ① Gemma 4 공식 QAT GGUF; 파일이 없으면 명시적으로 내려받습니다.
python run_model.py --profile gemma4_12b_qat_gguf --download

# ② Qwen3.8 27B Q4_K_M; 약 19GB 다운로드, 24GB GPU 메모리 주의
python run_model.py --profile qwen38_27b_q4_gguf --download

# ③ HyperCLOVAX 14B Q4_K_M; 전용 호환 llama.cpp 빌드 필요할 수 있음
python run_model.py \
  --profile hyperclovax_14b_q4_gguf \
  --llama-server-bin /path/to/NAVER-llama.cpp/build/bin/llama-server \
  --download

# ④ Qwen3.5 9B FP8; vLLM 설치 후 실행
python run_model.py --profile qwen35_fp8_vllm

# ⑤ Gemma 4 12B QAT vLLM (GGUF와 다른 양자화)
python run_model.py --profile gemma4_12b_qat_vllm
```

**중요:** `--download`를 생략한 vLLM은 모델이 HF 캐시에 없으면 **vLLM이 내부적으로 다운로드를 진행할 수 있습니다.** 외부 다운로드를 절대 피해야 한다면 미리 모델을 로컬에 받고 `profiles.json`의 `model_repo`를 로컬 경로로 바꾸세요.

모델은 한 번에 **하나씩만** 로딩합니다. `run_model.py`는 모델 시작 전, 준비 완료 시, 테스트 중 피크 VRAM을 기록하며 프로세스를 종료합니다. 기존에 GPU를 점유하는 다른 프로세스가 있으면 측정된 VRAM은 모델 순수 사용량이 아닙니다.

### 2) 결과 비교

```bash
python compare_results.py \
  results/qwen35_fp8_vllm \
  results/gemma4_12b_qat_gguf \
  results/hyperclovax_14b_q4_gguf \
  results/qwen38_27b_q4_gguf \
  --out comparison.md
```

결과 파일은 다음과 같습니다.

- `results/<profile>/summary.json`: 정확도·속도 요약과 GPU 피크.
- `results/<profile>/results.jsonl`: 각 요청의 생성 결과 원문, 평가 세부정보, 토큰 수·지연 시간.
- `results/<profile>/results.csv`: 엑셀로 열 수 있는 상세 수치.
- `results/<profile>/server.log`: 자동 실행 시 서버 실행 로그.
- `comparison.md`, `comparison.csv`: 모델별 최종 비교표.

`run_all.sh`는 기본 4개 프로필을 순차 실행하지만 필요한 바이너리가 모두 있어야 하며, **모델 다운로드가 완료된 환경에서만** 쓰는 것을 권장합니다.

### 3) 어떤 테스트를 하는가?

`cases.json`에는 **IT 3개, 디자인 3개, 반도체·제조 3개, 건설 기능직 3개, 외식 2개** 총 **14개**의 가상 JD/이력서 사례와 정답(31개 요구사항)이 포함되어 있습니다.

모델은 다음 항목의 JSON을 생성합니다.

```json
{
  "requirements": [
    {
      "id": "R1",
      "level": "required",
      "user_status": "have",
      "resume_quote": "이력서 원문의 정확한 일부",
      "jd_quote": "JD 원문의 정확한 일부"
    }
  ]
}
```

모델의 점수로 나타내는 것은 **사용자 역량이 아니라 테스트 정답과의 일치율**입니다. `needs_learning` 상태는 사용자가 직접 선택한 경우만 허용한다는 서비스 원칙에 따라, 이력서만 분석할 때는 `have`/`unknown`만 출력합니다.

평가 항목은 다음과 같습니다.

| 지표 | 뜻 | 주의 |
|---|---|---|
| `status_accuracy` | 사용자 보유 근거 상태와 정답의 일치율 | 14개 사례의 `have`/`unknown` |
| `level_accuracy` | `required`/`preferred` 정확도 | 실제 조건 문장을 정확히 구분하는지 |
| `evidence_valid_rate` | 실제 JD·이력서 원문을 그대로 인용했는지 | 인용이 **사실이라는 것 이상으로 숙련도를 입증하지는 않음** |
| `false_have_count` | 모르는 역량을 보유로 판단한 오류 | 서비스에서 가장 중요하게 볼 지표 |
| `perf_median_ttft_s` | 서버에 요청한 뒤 처음 출력이 전송될 때까지의 시간 중위값 | 서버의 reasoning 출력 정책에 영향 |
| `perf_median_decode_tps` | 출력 시작 뒤 스트리밍 토큰/초의 중위값 | 토큰 수는 `server_usage`를 우선함 |
| `gpu_peak_gib` | 서버 실행에서 측정한 GPU 전체 사용 피크 | 모델 순수 가중치 크기가 아님 |

모델이 스트리밍 응답에서 정확한 `completion_tokens`를 제공하지 않으면 `/tokenize`를 사용해 대략적으로 계산합니다. 둘 다 불가능하면 `tokens=null`로 기록하며 **토큰/초를 조작해 생성하지 않습니다**. `token_source`가 `server_usage`인지 확인하세요.

테스트가 편향되지 않게 하려면 **동일한 JD/이력서, 동일한 최대 출력 토큰, 동일한 동시 요청 수, 동일한 문맥 길이, 가능하면 동일한 추론 모드**를 설정하세요. 서로 다른 모델의 토크나이저를 사용하므로 tok/s만으로 사용자 관점의 속도를 비교하면 안 됩니다. **요청 전체 시간과 추출 품질**을 함께 평가해야 합니다.

### 제한과 권장하는 추가 실험

- `--max-tokens 700`은 기본값입니다. 추론 모델이 내부 생각에 너무 많은 토큰을 소비해 출력이 잘리는 경우 `--max-tokens 1500`으로 높이거나, 모델이 공식 지원하는 방법으로 thinking을 끄고 별도 테스트하세요. 이것은 모델별로 인터페이스가 달라 기본적으로 강제하지 않습니다.
- 기본 `temperature=0`입니다. 모델에서 권장한 생성 설정과 다를 수 있으므로 정밀 비교 시 변경할 수 있게 코드를 수정하세요.
- 단일 출력 속도만 측정합니다. 동시성 4·8·16 사용자 서버 처리량은 별도의 부하테스트가 필요합니다.
- PDF 인식/OCR 테스트는 포함하지 않습니다. 실제 이력서 파싱 이후 생성된 **텍스트**의 추출/비교를 검증합니다.
- 이력서의 실제 수행 능력은 자동 확인할 수 없습니다. 테스트의 `have`는 **문서에 명시된 근거가 확인되었다**는 뜻이지 검증된 숙련도 인증이 아닙니다.
- 소수 합성 사례로 전 직종 일반화 성능이 증명되지는 않습니다. MVP 모델 선택 뒤 실제 전문 용어와 모호한 조건을 포함하는 별도의 검수셋을 추가하세요.
- 개인 이력서를 바로 테스트셋으로 만들지 마세요. 실명, 전화번호, 이메일은 제거하고 권한이 있는 자료만 사용하세요.

## 출처

- [NVIDIA L4 24GB, 300GB/s](https://www.nvidia.com/en-us/data-center/l4/)
- [Qwen FP8 배포본](https://huggingface.co/RedHatAI/Qwen3.5-9B-FP8-dynamic)
- [Qwen3.5-9B FP8·INT4 GPU memory sizing (Blackwell; L4 미실측)](https://huggingface.co/InfiniCloud/Qwen3.5-9B-GPTQ-INT4-G128)
- [Gemma 4 QAT 선택과 메모리](https://gemma.ai/get-started/qat/)
- [Gemma 4 QAT GGUF 6.98GB](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
- [Gemma 4 QAT L4 vLLM 배포 사례 (단일 속도 사례)](https://builder.aws.com/content/3FB3TkfssfXGTiGRzBSq4gHSYwK/deploying-gemma-12b-to-aws-ec2-with-nvidia-l4)
- [HyperCLOVAX Q4_K_M 8.92GB](https://huggingface.co/naver-ellm/HyperCLOVAX-SEED-Think-14B-GGUF)
- [HyperCLOVA VRAM 추정표](https://canrunllm.com/model/hyperclovax-seed-think-14b)
- [Qwen3.8 Q4_K_M 파일](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
- [Qwen3.8 27B 단일 L4 성능, 기본 14.6tok/s](https://llamaperf.com/?engine=1&gpu=160&reports=all)
- [Qwen3.8 단일 L4 고급 튜닝 37.4tok/s](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)
- [Qwen3.8 Q4_K_M 실측 8K VRAM 18.42GiB (다른 GPU)](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/discussions/6)
