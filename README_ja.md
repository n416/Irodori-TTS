# Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
[![VoiceDesign](https://img.shields.io/badge/VoiceDesign-HuggingFace-orange)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v3-Demo)
[![VoiceDesign Demo](https://img.shields.io/badge/VoiceDesign%20Demo-HuggingFace%20Space-red)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

[English README](./README.md)

**Irodori-TTS**の学習および推論用コードです。Flow MatchingベースのText-to-Speech（音声合成）モデルとなります。アーキテクチャや学習の設計は主に[Echo-TTS](https://jordandarefsky.com/blog/2025/echo/)を踏襲しており、生成ターゲットとして[DACVAE](https://github.com/facebookresearch/dacvae)の連続潜在表現（continuous latents）を使用しています。

OpenAI互換の推論APIサーバーについては、[Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server)をご覧ください。

> [!IMPORTANT]
> `main` ブランチは**v3**のコードベースを反映しており、**Irodori-TTS-500M-v3**ベースモデルリリースでの使用を想定しています。
> 現在のコードは、**Irodori-TTS-500M-v2-VoiceDesign**を含む**Irodori-TTS-500M-v2**チェックポイントとの後方互換性を維持しています。
> もし以前のv2コードベースの状態が必要な場合は、`v2`タグを使用してください。以前のv1コードが必要な場合は、`v1`タグを使用してください。
> v1のチェックポイントおよび前処理は、v2/v3とは互換性がありません。
> 以前の公開済みv1モデルは、[Aratako/Irodori-TTS-500M](https://huggingface.co/Aratako/Irodori-TTS-500M)で利用可能です。

モデルの重みと音声サンプルについては、[ベースモデルカード](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)および[VoiceDesignモデルカード](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)をご参照ください。

## 特徴

- **Flow Matching TTS**: 連続DACVAE潜在表現に対するRectified Flow Diffusion Transformer (RF-DiT)
- **Voice Cloning (音声クローン)**: 参照音声からのゼロショット音声クローニング
- **Voice Design (音声デザイン)**: キャプション条件付けによるスタイル制御
- **自動発話時間予測**: v3ベースモデルは、手動での `--seconds` 指定なしに出力長を自動で推定します
- **自動電子透かし**: 生成された音声には、利用可能な場合[SilentCipher](https://github.com/sony/silentcipher)による透かしが自動的に埋め込まれます
- **マルチGPU学習**: `uv run torchrun`による分散学習（勾配蓄積、混合精度 bf16、W&Bロギング対応）
- **PEFT LoRA ファインチューニング**: リリース済みチェックポイントに対する、PEFT/LoRAを用いたパラメータ効率の良い適応
- **柔軟な推論**: CLI、Gradio Web UI、HuggingFace Hubのチェックポイント読み込みに対応

## アーキテクチャ

現在のコードベースは、密接に関連する2つのチェックポイントファミリーをサポートしています。

1. **ベースモデル (`Aratako/Irodori-TTS-500M-v3`)**:
   テキストエンコーダ ＋ 参照潜在表現エンコーダ ＋ Diffusion Transformer ＋ 長さ予測器（Duration Predictor）。参照潜在表現エンコーダは、話者やスタイルの条件付けのために、参照音声からパッチ化されたDACVAE潜在表現を消費します。v2ベースチェックポイントも推論用に引き続きサポートされています。
2. **VoiceDesignモデル (`Aratako/Irodori-TTS-500M-v2-VoiceDesign`)**:
   テキストエンコーダ ＋ キャプションエンコーダ ＋ Diffusion Transformer。キャプションエンコーダはスタイル制御のテキストを消費し、話者/参照分岐は無効化されます。v3のVoiceDesignリリースはまだないため、このパスは引き続きv2チェックポイントを使用します。

共通の構成要素：

1. **テキストエンコーダ**: 事前学習済みLLMから初期化されたトークン埋め込みに、自己注意機構(Self-attention)＋SwiGLU Transformerレイヤー（RoPE付き）を続けたもの
2. **条件エンコーダ**: ベースモデル用の参照潜在表現エンコーダ、またはVoiceDesignモデル用のキャプションエンコーダ
3. **Diffusion Transformer**: Joint-attention DiTブロック、Low-Rank AdaLN（タイムステップ条件付き適応的レイヤー正規化）、ハーフRoPE、およびSwiGLU MLP
4. **長さ予測器 (Duration Predictor)**: v3ベースモデルには、出力の長さを自動的に推定するための統合予測器が含まれています

音声は、チェックポイントで構成されたコーデックを介して連続潜在表現のシーケンスとして表現されます。リリース済みのv2/v3チェックポイントは、48kHzの波形再構成のために32次元の[Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)コーデックを使用しています。

## インストール

```bash
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync
```

**注意**: Linux/Windows（CUDAあり）の場合、PyTorchはcu128インデックスから自動的にインストールされます。macOS（MPS）やCPUのみでの使用の場合、`uv sync` はデフォルトのPyTorchビルドをインストールします。

## クイックスタート

### 簡単な推論

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### 参照音声なしの推論

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### VoiceDesign推論

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

その後、`http://localhost:7860` でUIにアクセスできます。
ホスティングされたv3デモは [Aratako/Irodori-TTS-500M-v3-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v3-Demo) で利用可能です。

VoiceDesignチェックポイントの場合は、専用のUIを使用してください：

```bash
uv run python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

ホスティングされたVoiceDesignデモは [Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo) で利用可能です。

`gradio_app.py` は `Aratako/Irodori-TTS-500M-v3` 用です。`gradio_app_voicedesign.py` は `Aratako/Irodori-TTS-500M-v2-VoiceDesign` 用です。

## 推論

### CLI

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

ローカルのチェックポイント（`.pt` または `.safetensors`）もサポートされています：

```bash
uv run python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

VoiceDesignチェックポイントはキャプションの条件付けもサポートしています：

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた、近い距離感の女性話者" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

マージされたチェックポイントをエクスポートしなくても、推論時にLoRAアダプターディレクトリを動的にロードできます：

```bash
uv run python infer.py \
  --checkpoint path/to/base_model.safetensors \
  --lora-adapter outputs/irodori_tts_lora/checkpoint_final \
  --text "こんにちは、私はAIです。これはLoRA推論のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_lora.wav
```

### 出力長 (Output Duration)

v3ベースモデルは、長さの予測を推論時に統合しています。
`--seconds` が省略された場合、入力テキストと（話者条件付けモデルの場合は）参照音声から出力長を推定し、その推定された長さで音声を生成します。予測された長さを乗算（倍率設定）するには `--duration-scale` を使用します（`>1` で長く、`<1` で短く）。正確な長さを指定したい場合は、手動で `--seconds` を渡してください。

古いv2チェックポイントは、固定長の30秒ターゲットで学習されていました。これらはv3コードベースでも引き続きサポートされ、手動の `--seconds` も受け付けますが、デフォルト以外の長さを強制すると音声品質が低下する可能性があります。自動またはスケールでの長さ制御にはv3ベースモデルを推奨します。

### Sway Sampling

より高速な試験的推論のために、Sway Samplingを少ないEulerステップ数と組み合わせることができます：

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --num-steps 6 \
  --t-schedule-mode sway \
  --sway-coeff -1.0 \
  --output-wav outputs/sample_sway.wav
```

### 推論に関する追加の注意点

チューニングのガイダンスや推論オプションの詳細な説明については、[パラメータガイド](docs/parameters.md)をご参照ください。

生成された音声は、依存関係およびモデルファイルが利用可能な場合、[SilentCipher](https://github.com/sony/silentcipher) による透かし処理が自動的に適用されます。

## 学習 (Training)

### 1. マニフェストの準備（DACVAE潜在表現の事前計算）

Hugging Faceデータセットの音声をDACVAE潜在表現にエンコードし、学習用のJSONLマニフェストを生成します。

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

マニフェストに `speaker_id` を含める場合（話者条件付け学習用）：

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

マニフェストに `caption` を含める場合（キャプション条件付きVoiceDesign学習用）：

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --caption-column caption \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

キャプション条件付きVoiceDesignモデルを学習する際、`speaker_id` はオプションです。VoiceDesignパスでは話者/参照の条件付けが無効化され、`text + caption` から学習します。

これにより、以下のようなJSONLマニフェストが生成されます：

```json
{"text": "こんにちは", "caption": "落ち着いた、近い距離感の女性話者", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### 2. 学習の実行

シングルGPUでの学習：

```bash
uv run python train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

v3リリースの学習は2つのフェーズを使用します。本体（body）を学習した後、フェーズ1のチェックポイントから統合予測器を初期化します：

```bash
uv run python train.py \
  --config configs/train_500m_v3_phase2_duration.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_duration \
  --init-checkpoint outputs/irodori_tts/checkpoint_final.pt
```

VoiceDesignの学習には専用のconfigを使用します：

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design
```

`configs/train_500m_v2_voice_design.yaml` では `use_caption_condition: true` が設定されており、話者/参照分岐が無効化されています。キャプションなしのconfigでは、`speaker_id` / 参照入力が利用可能な場合に引き続き話者条件付けが使用されます。

VoiceDesignのconfigでは、オプションでキャプション分岐のウォームアップを行うための `caption_warmup: true` も有効化されています。`warmup_steps` はLRスケジューラーを制御し、`caption_warmup_steps` は通常の共同学習が再開されるまでキャプションなしの勾配が破棄される期間を制御します。

### v3 長さ予測器 (Duration Predictor) の学習

v3の学習では2つのフェーズを使用します：`configs/train_500m_v3_phase1_body.yaml` で可変長のDiT本体を学習し、次に `configs/train_500m_v3_phase2_duration.yaml` で本体を凍結して長さ予測器を学習します。

長さ予測器は、Huber lossを使用して `log1p(num_frames)` を回帰します。現在のv3フェーズ2 configでは、アブレーションから選択されたトークン合計長さ予測器を使用しています。アーキテクチャの詳細はパラメータガイドを参照してください。

マルチGPU DDP学習：

```bash
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --device cuda
```

学習では、`model` と `train` セクションを持つYAML設定ファイルをサポートしています。CLI引数はYAML値よりも優先されます。利用可能なすべてのオプションについては `uv run python train.py --help` を参照してください。
モデルおよび学習設定フィールドのより詳細な説明については、[パラメータガイド](docs/parameters.md)を参照してください。

#### リリース済み重みからのファインチューニング

リリース済みの推論用重み（`.safetensors`）から新しい学習を開始します。これはモデルの重みのみを初期化し、オプティマイザやスケジューラーの状態は新規から開始されます。v3ベースリリースの場合、LoRA configはデフォルトで長さ予測器を保存されたアダプターの一部として維持します。

```bash
uv run python train.py \
  --config configs/train_500m_v3_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v3.safetensors
```

キャプション条件付きVoiceDesign LoRAファインチューニング：

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v2-VoiceDesign.safetensors
```

LoRAターゲットのプリセット、アダプターの保存動作、再開の詳細については、[パラメータガイド](docs/parameters.md)で説明されています。

#### 中断された学習の再開

既存の学習をチェックポイントから再開します。フルモデルの学習では `.pt` を使用し、LoRAの学習ではチェックポイントディレクトリを使用します。どちらもオプティマイザ、スケジューラー、ステップの状態を復元します。

```bash
uv run python train.py \
  --config configs/train_500m_v3_phase1_body.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --resume outputs/irodori_tts/checkpoint_0010000.pt
```

LoRAの再開例：

```bash
uv run python train.py \
  --config configs/train_500m_v3_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --resume outputs/irodori_tts_lora/checkpoint_0010000
```

もしLoRAチェックポイントを別の環境に移動し、元のベースチェックポイントのパスが有効でなくなった場合は、`--resume` とともに `--init-checkpoint path/to/base_model.safetensors` を渡すことで、保存されているベースモデルのパスを上書きできます。

### 3. チェックポイントの変換

学習チェックポイントを推論専用のsafetensorsフォーマットに変換します：

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

LoRAアダプターのチェックポイントも直接変換できます：

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/irodori_tts_lora/checkpoint_final
```

LoRAアダプターチェックポイントは変換時に自動的にベースモデルにマージされるため、エクスポートされた `.safetensors` ファイルはそのまま推論に利用可能です。アダプターをマージしたくない場合は、アダプターディレクトリを直接 `infer.py --lora-adapter` または対応するGradioフィールドに渡してください。

## プロジェクト構成

```text
Irodori-TTS/
├── train.py                    # 学習のエントリーポイント（DDP対応）
├── infer.py                    # CLI推論
├── gradio_app.py               # Gradio web UI
├── gradio_app_voicedesign.py   # VoiceDesignチェックポイント用 Gradio web UI
├── prepare_manifest.py         # Dataset -> DACVAE潜在表現 の前処理
├── convert_checkpoint_to_safetensors.py  # チェックポイント変換ツール
│
├── docs/
│   └── parameters.md         # 詳細なパラメータガイド
│
├── irodori_tts/                # コアライブラリ
│   ├── model.py                # TextToLatentRFDiT アーキテクチャ
│   ├── rf.py                   # Rectified Flowユーティリティ ＆ Euler CFGサンプリング
│   ├── codec.py                # DACVAE コーデックラッパー
│   ├── dataset.py              # データセットとコレーター
│   ├── tokenizer.py            # 事前学習済みLLMトークナイザーラッパー
│   ├── config.py               # モデル/学習/サンプリング設定のデータクラス
│   ├── inference_runtime.py    # キャッシュ対応・スレッドセーフな推論ランタイム
│   ├── lora.py                 # PEFT LoRA統合ヘルパー
│   ├── text_normalization.py   # 日本語テキスト正規化
│   ├── optim.py                # Muon + AdamW オプティマイザ
│   └── progress.py             # 学習進捗トラッカー
│
└── configs/
    ├── train_500m_v3_phase1_body.yaml        # 500M v3 本体学習用config
    ├── train_500m_v3_phase2_duration.yaml    # 500M v3 長さ予測器学習用config
    ├── train_500m_v3_lora.yaml               # 500M v3 LoRAファインチューニング用config
    ├── train_500m_v2.yaml                    # 500M v2 後方互換モデルconfig
    ├── train_500m_v2_lora.yaml               # 500M v2 LoRAファインチューニング用config
    ├── train_500m_v2_voice_design.yaml       # 500M v2 VoiceDesign フルファインチューニング用config
    ├── train_500m_v2_voice_design_lora.yaml  # 500M v2 VoiceDesign LoRAファインチューニング用config
    ├── train_500m.yaml                       # 500M v1 モデルconfig
    └── train_2.5b.yaml                       # 2.5B パラメータモデルconfig
```

## ライセンス

- **コード**: [MIT License](LICENSE)
- **モデルの重み**: ライセンスの詳細については、[ベースモデルカード](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)および[VoiceDesignモデルカード](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)をご参照ください。

## 謝辞

このプロジェクトは以下の研究や成果物に基づいています：

- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — アーキテクチャと学習設計の参考
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE
- [SilentCipher](https://github.com/sony/silentcipher) — 音声の電子透かし

## 引用 (Citation)

```bibtex
@misc{irodori-tts,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model with Emoji-driven Style Control},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Aratako/Irodori-TTS}}
}
```
