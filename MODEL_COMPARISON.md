# L4 24GB 모델 비교표

**조사일: 2026-10-31**. `8K`는 최대 문맥 길이를 8192토큰으로 제한하는 실험 조건. **아래 수치는 서로 다른 엔진, 설정, 양자화를 섞은 공개 자료**이므로 통일된 실측 결과가 아님. 최종 비교는 제공한 벤치마크 프로그램으로 측정할 것.

| 모델 | 서빙 배포 형태 | 파일/가중치 | 8K VRAM 참고 | 단일 L4 출력 속도 | 신뢰성 및 주의사항 |
|---|---|---|---|---|---|
| Qwen3.5-9B | `RedHatAI/...FP8-dynamic`, vLLM | FP8 모델 적재 12.76GiB(Blackwell의 다른 배포본) | L4 실측 미확보. **12.76GiB는 총 GPU 사용량이 아닌 모델 적재만 측정**; vLLM 메모리 예약으로 더 커질 수 있음 | **FP8 동일 구성 L4 실측 미확보**. BF16 다른 구성 L4 약 11tok/s는 FP8과 비교 불가 | FP8은 Ada L4에서 지원되지만 호환 vLLM 버전 필요 |
| Gemma 4 12B | Google 공식 QAT `Q4_0` GGUF, llama.cpp | GGUF **6.98GB** | Q4 8K 일반 참고 **약 7.8~9GB 추정** (동일 QAT Q4_0 파일 L4 실측 아님) | **동일 GGUF L4 실측 미확보**. vLLM W4A16 별도 배포본은 **28.03tok/s 단일 사례** | QAT GGUF와 vLLM W4A16 정확도 및 속도를 동일시하지 말 것 |
| HyperCLOVAX-SEED-Think-14B | Q4_K_M GGUF, 호환 llama.cpp | **8.92GB** | 약 **10.4GB 추정** (8K, F16 KV 등 가정) | **단일 L4 실측 미확보** | NAVER llama.cpp 포크 권장, 라이선스 확인 |
| Qwen3.8-27B | `ggml-org/...` Q4_K_M, llama.cpp | 약 **19GB** | **18.42GiB 실측** (다른 GPU, 동일 파일/8K/flash-attn). L4에는 메모리 여유가 제한됨 | 기본 L4 **14.6tok/s 보고**, 고급 튜닝 L4 **37.4tok/s 보고** (다른 모델 양자화/추론기법) | 단일 L4에서 109.3tok/s라고 표시된 공개 결과는 멀티 GPU이므로 제외 |

## 판단

- **안정적인 첫 구현:** Qwen3.5-9B FP8 또는 Gemma 4 12B QAT.
- **한국어 특화 비교 대상:** HyperCLOVAX 14B. L4 성능 수치와 서빙 포크 호환성은 실측 필요.
- **크고 느린 추론 품질 비교 대상:** Qwen3.8-27B Q4. 기본 측정의 14.6tok/s는 **출력 500토큰에 단순 계산 약 34초**이고 프리필, 첫 토큰 지연, 추론 토큰 등을 포함하면 더 걸릴 수 있음.
- **모든 모델 공통:** 모델의 자신감이나 응답 유창성은 실제 역량 보유 여부의 증거가 아니다. 근거 없는 `have` 판정과 가짜 자격 추론이 없는 모델을 우선 선택할 것.

## 출처 상세

- NVIDIA: https://www.nvidia.com/en-us/data-center/l4/
- FP8 참고: https://huggingface.co/InfiniCloud/Qwen3.5-9B-GPTQ-INT4-G128
- Qwen BF16 L4 사례: https://supercomputer.sh/docs/examples/qwen3.5-9b
- Gemma 4 QAT 공식: https://gemma.ai/get-started/qat/
- Gemma 공식 GGUF: https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf
- Gemma L4 단일 사례: https://builder.aws.com/content/3FB3TkfssfXGTiGRzBSq4gHSYwK/deploying-gemma-12b-to-aws-ec2-with-nvidia-l4
- Gemma 메모리 참고: https://canrunllm.com/ko/guides/gemma-4-models
- Hyper 공식 GGUF: https://huggingface.co/naver-ellm/HyperCLOVAX-SEED-Think-14B-GGUF
- Hyper 메모리 추정: https://canrunllm.com/model/hyperclovax-seed-think-14b
- Qwen3.8 GGUF: https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF
- Qwen3.8 8K 메모리 측정: https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/discussions/6
- Qwen3.8 단일 L4 기본 속도: https://llamaperf.com/?engine=1&gpu=160&reports=all
- Qwen3.8 단일 L4 고급 튜닝: https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4
- 멀티 GPU 오기 주의: https://llm-bench.io/benchmarks/cmt87wd40009k01lcjvkpox1n
